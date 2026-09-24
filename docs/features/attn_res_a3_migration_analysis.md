# Attention Residual A5 -> A3 迁移分析与实现记录

## 结论

- 目标对象是 `attn_res_fwd` 算子族，位于 `csrc/attention/`：
  `attn_res_fwd`、`attn_res_fwd_with_add`、`attn_res_fwd_fused`、
  `attn_res_fwd_prefill`。
- A3 原先只构建了基础 `attn_res_fwd`；本次为 `with_add/fused/prefill`
  增加独立的 arch22 kernel entry、fused/resident/reload 实现，并保留 A5 的
  arch35 路径。
- 精度合同已明确：`raw_prefix`、bank 写回、`materialized` 必须与“小算子”参考
  逐 bit 一致；输出 RMSNorm 允许 `rtol=1e-2, atol=1e-2`。迁移到 A3 后必须保持
  同样的判定，不能把 bitwise 降级为松阈值。

## 现状与证据

### 构建范围

`csrc/build_aclnn.sh`：

- A3 (`ascend910_93`) 列表现在包含四个 attention residual 入口。
- A5 (`ascend950`) 列表包含 `attn_res_fwd`、`attn_res_fwd_with_add`、
  `attn_res_fwd_fused`、`attn_res_fwd_prefill`。

A3 运行时会从 ACLNN 包加载这三个 fused 入口；当前 Windows 环境没有 CANN/A3
设备，无法在本地生成二进制或采集设备日志。

### 内核架构分流

- `csrc/attention/attn_res_fwd/op_kernel/attn_res_fwd.cpp`：arch22 入口。
- `csrc/attention/attn_res_fwd/op_kernel/attn_res_fwd_apt.cpp`：arch35 入口。
- `with_add/fused/prefill` 三个目录新增 `op_kernel/*_a3.cpp`，只 include
  `arch22/attn_res_fwd_reload.h` 和 `arch22/attn_res_fwd_resident.h`；原有
  `*_apt.cpp` 继续只服务 arch35。
- arch22/arch35 均由
  `template <typename D_IN, bool FUSED_PREFIX = false, bool PREFILL_CACHE = false>`
  承载，ABI、tiling key 与 host tiling 结构保持不变。

### Tiling

`csrc/attention/attn_res_fwd/op_host/attn_res_fwd_tiling.cpp` 已经识别
`AttnResFwdFused`、`AttnResFwdPrefill`、`AttnResFwdWithAdd` 的 node type，并写入
`fusedChain/fusedAdd/outputNormEps/blockWriteIdx/blockTokenStride`。这部分是
平台无关的 host tiling，A3 可复用；`UB_AVAIL_BYTES` 目前按 192 KiB 估算。

## 算子语义与精度合同

### 公式

对 token `t`，候选向量由有效历史 block `v[t, 0..N-1]` 和当前 prefix
`v[t, N]` 组成：

```text
inv_rms[t, n] = rsqrt(mean_h(v[t, n, h]^2) + norm_eps)
score[t, n]   = sum_h(v[t, n, h] * inv_rms[t, n]
                      * norm_weight[h] * proj_weight[0, h])
probs[t, :]   = softmax(score[t, :])
hidden[t, h]  = sum_n(probs[t, n] * v[t, n, h])
```

中间以 FP32 计算，输出回 BF16。

### Fused 的额外合同

`tests/ut/models/test_kimi_attn_res_add_cpu.py` 的小算子参考定义了黄金语义：

- `raw_prefix = prefix` 或 `prefix + addend`，加法在 BF16 下完成并回写 BF16。
- `materialized/value = attn_res_fwd(raw_prefix, blocks[:, :valid], ...)`
  或 `raw_prefix`（`valid == 0`）。
- `block_write_idx >= 0` 时，将 `raw_prefix` 写回 bank 的对应 slot。
- `output_norm_weight is not None` 时：
  `output = (rms_norm(value, output_norm_weight, output_norm_eps))`。

`tests/e2e/nightly/single_node/ops/singlecard_ops/test_attn_res_fwd.py` 的 NPU
判定要求：

- `raw_prefix`、`materialized`、bank 写回、输入不可变：`rtol=0, atol=0`。
- 最终 RMSNorm 输出：`rtol=1e-2, atol=1e-2`（CANN RMSNorm 允许不同归约树）。

### 精度风险与方向

- A5 -> A3 是反向迁移。cannbot-skills 的 `ascendc-cross-gen-port-light` 主要覆盖
  arch22 -> arch35，但其精度规则可逆用：API 差异、rounding point、归约顺序必须
  逐项核对，不能照搬结论。
- arch35 对 subnormal 采用 FTZ；arch22 默认支持 subnormal。反向迁移时，A3
  kernel 与 A3 小算子 golden 都运行在 arch22 上，因此应以 A3 小算子为真值，
  而不是试图复刻 A5 的 subnormal 行为。
- `InvRmsInPlace` 已经显式采用 `Sqrt + Div(1/x)`，避免直接 `Rsqrt` 的近似差异；
  这是对齐 ops-nn RMSNorm 的关键，A3 移植必须保留该实现，不要改成 `Rsqrt`。
- `SoftmaxSmallVec` 先减 max 再 `Exp`，满足稳定性要求；A3 移植需保持相同的
  计算顺序，避免 softmax 归约树变化破坏 bitwise 合同。
- Fused 中 `prefix + addend` 的 BF16 舍入点不能移动。若 A3 实现先转 FP32 相加
  再回 BF16，结果可能与 `prefix + addend` 的 BF16 舍入不一致；应直接在 BF16 上
  做 `Add`，与当前 A5 `fused_reference` 一致。

## 性能分析

- A5 fused 的价值是每个残差点只启动一个 kernel，覆盖 add + bank write +
  AttnRes + 可选 output RMSNorm + materialized 输出；A3 若退回“小算子”组合，
  至少增加 Add、AttnRes、RMSNorm 多次 launch 和中间 tensor 往返。
- A3 arch22 的基础 `attn_res_fwd` 已有 RELOAD/RESIDENT 两条路径，且 `reduce_common.h`
  提供了 `InvRmsInPlace`、`BroadcastScalarMulTensor`、`MulAddRowByBrcBlock` 等
  vector 级原语。arch22 fused 可直接复用这些原语，不必照搬 arch35 的 RegBase
  MicroAPI。
- A5 的性能结论（例如 prefill weight cache 带来 4.77% 单算子和）不能直接搬到
  A3：A3 的 UB、AIV 数量、vector/scalar pipe 比例不同，必须在 A3 上重测。
- 优先级建议：
  1. `with_add`：逻辑增量最小，先打通 A3 fused 的最小闭环。
  2. `fused`：K3 主链必需，包含 add + bank write + output norm + materialized。
  3. `prefill`：仅在 A3 实测证明 weight cache 有收益时再上；否则 Python 侧继续
     `_use_attn_res_prefill_kernel()` 返回 False，A3 走 `fused`。

## 迁移方案

### 阶段 0：环境与基准

- 准备 A3 设备、CANN、asc-devkit 全量仓。
- 在 A3 上固化“小算子参考”：
  `attn_res_fwd` + `prefix + addend` + `npu_rms_norm`。
- 固化现有 `tests/e2e/nightly/single_node/ops/singlecard_ops/test_attn_res_fwd.py`
  作为回归基线。

### 阶段 1：合同冻结

- ABI 不变：torch binding、tiling key `40010/40020/40011/40021`、workspace、
  输出张量别名规则不变。
- 精度判定不变：bitwise 与 `rtol=1e-2, atol=1e-2` 分层保留。
- 覆盖义务：`valid=0..8`、`add/none`、`norm/none`、bank slice/stride、
  `mix=True/False`、`save_materialized`、图模式 replay。

### 阶段 2：arch22 内核移植

- 扩展 `csrc/attention/attn_res_fwd/op_kernel/arch22/attn_res_fwd_reload.h` 和
  `attn_res_fwd_resident.h`：
  - `AttnResFwdInitParams` 增加 `addend/prefixOut/outputNorm/materialized`。
  - 模板增加 `FUSED_PREFIX` 与可选 `PREFILL_CACHE`。
  - 复用 arch22 `reduce_common.h` 的 vector helper 实现 `PrepareFusedPrefix`、
    bank write、output RMSNorm、materialized 分支。
- 为三个 A5-only 目录增加 arch22 入口文件，或与 `attn_res_fwd.cpp` 相同的
  include 分流，确保 A3 构建走 arch22、A5 仍走 arch35。
- 保持 host tiling 的 `fusedChain/fusedAdd` 逻辑不变；A3 只新增内核实现，不重写
  tiling 公式。

### 阶段 3：构建与单算子验证

- 将三个算子加入 `csrc/build_aclnn.sh` 的 A3 列表。
- 在 A3 上跑 `test_attn_res_fwd.py` 全量用例。
- 用 `tests/ut/models/test_kimi_attn_res_add_cpu.py` 验证 Python 侧路由和小算子
  黄金语义。

### 阶段 4：性能验收

- 对比 A3 上“小算子组合”与 fused 的逐 case 中位数；只有实测 fused 不劣化时才
  启用 fused 路径。
- `prefill` 单独 A/B，收益为负则不注册 A3 prefill。

## 风险与阻断点

- 当前环境是 Windows、无 CANN/NPU，无法在本机编译或跑 NPU 精度/性能用例；arch22
  fused 内核仍需要 A3 设备闭环验证。
- `UB_AVAIL_BYTES=192KiB` 是 host tiling 常量，A3 fused 额外分配
  `fusedPrefix/prefixOut/outputNorm` 后需要重新核对 UB 预算，避免大 H 误选
  RESIDENT。
- A3 的 bank stride 与 A5 一致依赖 torch adapter 的 view 合同；不要为了 A3 改为
  `Contiguous` 打包，否则会破坏 stride/PP 边界语义。

## 当前交付与设备侧门禁

已完成：

1. `arch22` reload/resident 增加 fused prefix、BF16 add、bank write、materialized
   与 output RMSNorm，保留原有 `InvRmsInPlace` 和 `SoftmaxSmallVec` 计算顺序。
2. 三个 A3 kernel entry 与 `ascend910_93` OpAICore 配置已接入；tiling ABI 和
   tiling key 未改动。
3. `csrc/build_aclnn.sh` 的 A3 operator inventory 已覆盖四个 attention residual
   入口。

设备侧必须完成：

1. 使用 A3/CANN 环境编译安装，执行 `test_attn_res_fwd.py` 全量用例。
2. 对 `with_add`、`fused`、`prefill` 与小算子组合分别采集中位延迟；只有 fused
   不低于组合基线时才在产品配置中启用。
3. 精度保持 raw prefix、materialized、bank write bitwise 对齐，最终 RMSNorm
   使用现有 `rtol=1e-2, atol=1e-2` 合同。

## 2026-09-24 批量 helper 回归结果

`SubLastDimRow1NoBrc` 和 `MulLastDimRow1NoBrc` 已从每 8 个 FP32 元素一次的循环
改为完整 64-FP32 repeat + tail 路径。A3 clean build 安装到
`opp_batch_helper` 后，T=1、H=128 的 B=62/63/64/65 全部通过，maxerr=0，未出现
507035 或 UB 对齐错误；with_add 和 fused 直接 reference 用例也全部通过。

性能比较必须使用 A3 小算子组合，而不是修复前 `opp_final`。同输入、20 次 warmup、
100 次 NPU synchronize 重复下，融合算子与小算子组合的 P50（微秒）如下：

| B | H | A3 小算子组合 | A3 融合算子 |
|---:|---:|---:|---:|
| 1 | 128 | 263.49 | 54.10 |
| 3 | 256 | 266.01 | 53.54 |
| 64 | 128 | 247.30 | 74.25 |
| 1 | 4096 | 244.58 | 54.23 |
| 3 | 4096 | 247.27 | 57.80 |
| 64 | 4096 | 243.81 | 119.91 |

这些 case 的融合算子均低于小算子组合；T=129、H=7168、valid=4 的 prefill 直接
回归 maxerr=0.0078125、无 NaN，满足 RMSNorm 误差门限。正式结论仍需补齐官方
pytest 环境和 graph replay。当前 pytest 还受到已安装 vLLM 与源码 patch API 不匹配影响：
`vllm.v1.attention.backends.utils` 缺少 `resolve_kv_cache_layout`。

后续审查还修复了 fused no-add 分支：当 `fusedChain != 0` 且 addend 为空时，
reload/resident 都会回写 `prefixOut`，保证 `raw_prefix=prefix` 的合同不依赖未初始化
内存。该修改已重新编译并用 fused no-add 直接用例验证。
