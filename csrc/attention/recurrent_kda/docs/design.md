# RecurrentKda 设计方案

## 1. 背景

KDA recurrent 用于小步长 decode/MTP 场景，按 token 顺序更新 KDA 状态并生成输出。当前接口支持
raw gate、`A_log/dt_bias/lower_bound/safe_gate`、`beta sigmoid` 和 `state_v_first=True` 状态布局。该实现的目标
是在一个 KDA 独立算子中完成这些语义，避免把 gate 预处理拆成 `KdaGateCumsum + recurrent` 两段公共实现，也不复用
GDN recurrent 的函数接口。

## 2. 目标与非目标

### 2.1 目标

- 提供 Ascend C 实现类型的 `recurrent_kda`，覆盖 `torch.ops._C_ascend`、aclnn 和 `<<<>>>` 入口。
- 支持 `raw gate`、`A_log`、`dt_bias`、`lower_bound`、`safe_gate`、`beta sigmoid`、`allow_neg_eigval`。
- 支持 `BSND` 和 `TND`，支持 device `cu_seqlens`、容量化 state pool 和二维 speculative slot 索引。
- 保持 KDA recurrent 与 GDN recurrent 接口解耦。

### 2.2 非目标

- 不支持 `state_v_first=False`。
- 不支持把 `KdaGateCumsum` 的输出流水作为公开前置步骤。
- 不在本 PR 扩展长序列 chunk recurrent；每条 recurrent 序列长度当前限制为 `<=8`。

## 3. 能力边界

实现类型：`ascendc`

支持范围由本文、API 文档和 Kimi K3 单算子测试共同约束。

| 维度 | 支持范围 |
| --- | --- |
| SOC | `ascend910b`、`ascend910_93`、`ascend950` |
| Layout | `BSND`、`TND` |
| Q/K/V dtype | BF16 |
| K/V | `K=128,V=128` 或 `K=128,V=256` |
| Gate/Beta dtype | FP32/BF16/FP16，aclnn 预处理为 FP32 |
| State dtype | FP32/BF16 |
| Gate 模式 | 预计算 step log gate；kernel 内 raw gate |
| State layout | `state_v_first=True`，`[state_capacity,H_v,V,K]` |

## 4. 数学与接口语义

若 `use_qk_l2norm_in_kernel=True`，先对每个 token 的 `q/k` 在最后一维做 L2 normalize。随后 `q` 乘以 `scale`。

`use_gate_in_kernel=false` 时，`g` 已经是 step log gate。`use_gate_in_kernel=true` 时，kernel 内转换 raw gate：

```text
gate = -exp(A_log) * softplus(g + dt_bias)                 # safe_gate=False
gate = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))   # safe_gate=True
```

`use_beta_sigmoid_in_kernel=true` 时：

```text
beta = sigmoid(beta)
if allow_neg_eigval:
    beta = 2 * beta
```

每个 token 的 recurrent 更新为：

```text
S = exp(gate_t) * S
delta = beta_t * (v_t - S @ k_t)
S = S + outer(delta, k_t)
o_t = S @ q_t
```

`H_v % H_k == 0`，每个 value head 通过 `floor(h_v / (H_v / H_k))` 映射到一个 query/key head。

## 5. 整体架构

- `op_host/op_api/aclnn_recurrent_kda.cpp`：校验 dtype、layout、shape、可选输入组合；不读取 device metadata 的值；
  对输入做连续化与必要 cast；非连续 state 通过临时连续 tensor 和 `ViewCopy` 回写原 view；创建 L0 executor。
- `op_host/recurrent_kda_tiling.cpp`：读取输入 shape、属性和可选输入存在性，填充 `RecurrentKdaTilingData`，计算 block dim 与 UB 切分。
- `op_kernel/recurrent_kda.cpp`：单个 AIV kernel 完成 q/k normalize、raw gate 转换、state decay、delta 更新、输出和最终状态写回。
- `recurrent_kda_torch_adpt.h`：提供 `_C_ascend` 适配，按输入 rank 选择 TND/BSND 并保持 state alias。

## 6. Tiling 设计

### 6.1 任务划分

Tiling 按逻辑序列和 value head 拆分任务。`cu_seqlens` 必传，`seq_num=len(cu_seqlens)-1`，使用与
fla-org 一致的累计边界语义；第 `i` 条序列范围为 `[cu_seqlens[i], cu_seqlens[i+1])`。当前单任务面向
recurrent 小步长，每个相邻 offset 的差值限制为 `<=8`。末项表示有效 token 数，可小于图捕获的
token capacity。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `vectorCoreNum` | `uint32_t` | 使用的 AIV core 数 |
| `ubCalSize` / `ubRestBytes` | `uint32_t` | UB 计算区大小与余量 |
| `t` / `seqLen` | `uint32_t` | token 总数与输入物理序列轴长度 |
| `nk` / `dk` | `uint32_t` | query/key head 数与 K 维 |
| `nv` / `dv` | `uint32_t` | value head 数与 V 维 |
| `sBlockNum` / `b` | `uint32_t` | state pool 容量与逻辑序列数 |
| `ssmStateStride` | `uint32_t` | 二维 state 索引的行 stride；packed 一维索引时为 0 |
| `vStep` | `uint32_t` | 单次处理的 V 方向步长 |
| `stateOutBufferNum` / `attnOutBufferNum` | `uint32_t` | UB ring buffer 数 |
| `scale` / `lowerBound` | `float` | query scale 与 safe gate 下界 |
| `layout` | `uint32_t` | BSND/TND 编码 |
| `has*` | `uint32_t` | 可选输入存在性 |
| `use*` / `safeGate` / `stateVFirst` | `uint32_t` | 属性分支 |

### 6.3 模板化方案与 tiling key

当前 kernel 通过 `REGISTER_TILING_DEFAULT(RecurrentKdaTilingData)` 使用默认 tiling data，不引入额外 tiling key。
dtype 固定为 BF16 QKV，state dtype 由编译宏 `DTYPE_STATE` 实例化；layout、gate/beta 分支和可选输入存在性由
tiling data 驱动，避免把 runtime shape 组合扩张到 tiling key。

## 7. Kernel 设计

### 7.1 计算流程

1. 解析当前任务对应的逻辑序列、value head、query/key head 和状态槽。
2. 加载初始状态到 UB 工作区；`initial_state=None` 的场景由 Python/legacy wrapper 预先创建全零状态。
3. 对每个 token 加载 `q/k/v/g/beta`，按属性执行 q/k normalize、raw gate 转换和 beta sigmoid。
4. 对状态做 gate decay，计算 `S @ k`、delta 和 outer 更新。
5. 计算 `S @ q` 写回 `out`，并把状态原位写回命中的 state pool 槽。

### 7.2 内存规划

| 内存层级 | Buffer | 生命周期 | 用途 |
| --- | --- | --- | --- |
| GM | 输入/输出 tensor | kernel 全程 | q/k/v/g/beta/state/out 与 device metadata |
| GM | workspace | 当前不依赖用户可见 workspace 中转 | 保留 aclnn 标准接口 |
| UB | state 工作区 | 单任务内复用 | 保存 `[V,K]` 局部状态片段 |
| UB | q/k/v/g/beta 临时区 | 单 token 内复用 | normalize、gate、beta 与输出计算 |

### 7.3 流水与同步

算子为 AIV-only recurrent 小步长 kernel，单任务内部按 token 串行维护状态依赖。调用侧保证不同活跃序列不共享写槽；
任务分配保证每个 `(seq/value head)` 的输出区域独立。`--cce-auto-sync=off` 保持不变，UB 复用由本 kernel 内顺序和
必要的 vector pipe barrier 维护，不依赖 GDN recurrent 的同步协议。

### 7.4 边界处理

- `cu_seqlens` 首项必须为 0，offset 必须单调不减，末项为有效 token 数且不得超过输入 token
  capacity。host 只检查 rank/dtype，device kernel 校验值。
- 相邻 offset 相等时表示空序列；该序列不产生 token 输出，也不读取 `ssm_state_indices/num_accepted_tokens` 或 state。
- 末项小于 capacity 时，仅 `[0,cu_seqlens[-1])` 的有效输出和 state side effect 有定义；padding tail 输出不作保证。
- 一维索引按 packed token 偏移读取，二维索引按 `[seq_idx,step]` 读取。
- `num_accepted_tokens` 用于定位 MTP decode 的初始状态槽。

## 8. 平台设计

| 平台 | SOC | 实现策略 | 分支差异 |
| --- | --- | --- | --- |
| A2 | `ascend910b` | 公共 AIV kernel | 无 |
| A3 | `ascend910_93` | 公共 AIV kernel | 无 |
| A5 | `ascend950` | 公共 AIV kernel，编译选项保留 A5 特化 hook | 当前无算法差异 |

## 9. 精度设计

Q/K/V 公开输入为 BF16，gate/beta 在 aclnn 预处理后以 FP32 进入 kernel。state 可为 FP32 或 BF16；推荐 FP32 状态以降低
小步长多 token 更新中的累计误差。CPU reference 与 Kimi K3 设备用例共同位于
`tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_kda_recurrent_ascendc_npu.py`，BF16 输出使用
`rtol=0.02, atol=0.02`。

## 10. 性能设计

目标场景是 decode/MTP recurrent 小步长，主要瓶颈是状态矩阵读写和每 token 的 `S @ k`、`outer(delta,k)`、`S @ q`。
当前实现优先完成 API 与功能语义闭环，后续可针对 `V/K` 固定值扩展更细的向量化、双缓冲和多 token 并行策略。

## 11. 测试设计

测试矩阵覆盖：

- BSND raw gate + unsafe gate + beta sigmoid。
- BSND safe gate + `allow_neg_eigval`。
- TND 预计算 log gate + 空 initial_state。
- Kimi/KDA 关键泛化 shape：GVA head 映射、`K=128,V=128` dense raw gate、`K=128,V=256` TND safe gate。
- Kimi K3 TP16：本地 head 数 6、`K=V=128`、BF16 Q/K/V、raw+safe gate、BSND decode/MTP 1-8；
  覆盖普通单 token decode、`[0,1,4,4,5]` 非等长序列、二维 slot 索引、容量化 state pool 和未命中槽保持不变。
- Kimi H96/D128 smoke：`H=H_v=96,K=V=128,safe_gate=True`，运行时生成 `cu_seqlens`，
  覆盖其长度和值泛化；每段 recurrent 长度仍遵循当前 `<=8` 限制。
- Kimi 完整长上下文 stress target：`T_total=12288,H=H_v=96,K=V=128,safe_gate=True` 记录在 JSON，
  因当前单 kernel 计算量较大，不纳入默认通过矩阵。
- 负向参数组合：长序列、`safe_gate` 与 raw gate 组合、`state_v_first=false`。

设备侧主入口为 `tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_kda_recurrent_ascendc_npu.py`，
底层 PTA 入口为 `csrc/attention/recurrent_kda/tests/pta/test_accuracy.py`。

## 12. 验证修复记录

A2/A5 精度验证过程中修复以下问题：

- `scale/lower_bound` 的 aclnn 公开参数为 `double`，AICore attr 为 `float`；L0 加入显式 cast，避免 tiling 读取到错误标量。
- A5 tiling 对 `GetOriginShape()` 返回对象生命周期更敏感；tiling context 改为持有 shape 副本，避免 optional/required shape 指针悬挂。
- q/k L2 normalize 后从 UB 标量读取 norm 前补充 V->S 同步，避免 scalar 读到旧值导致 raw/safe gate case 精度偏差。

## 13. 已知限制与演进计划

- 当前仅支持 BF16 QKV。
- 当前 `K/V` 仅支持 `K=128,V=128` 或 `K=128,V=256`。
- 当前每段 recurrent 序列长度限制为 `<=8`。
- 当前仅支持 `state_v_first=True`。
- 显式 slot 模式要求所有活跃写槽互不冲突，且 slot 值位于 state pool 容量范围内。
- 后续若扩展长序列或更多 dtype，需要同步更新 op_host 校验、tiling、kernel、JSON case 和 API 文档。
