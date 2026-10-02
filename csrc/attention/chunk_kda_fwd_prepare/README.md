# ChunkKdaFwdPrepare

FLA NPU 新增代码保留 [BSD 3-Clause](../LICENSE.fla-npu) 及文件头指定的 [CANN 许可证](../LICENSE.CANN)。

## vLLM Ascend 集成范围

本仓模板仅编译 BF16 gate、FP32 已激活 beta、safe sigmoid gate、exp2、FORWARD 输出，Q/K norm 支持 identity 和 L2。
通过现有 `chunk_kda_fwd` torch binding 的 V2 分支调用，随 `csrc/build_aclnn.sh` 构建。
不需要安装 FLA Python 包。下文保留上游完整接口说明；未编译的模式不属于本仓支持范围。

来源：`flashserve/flash-linear-attention-npu@81a7346ddb13340e03cdd2e14cbbbe777bafbcdf`。

[设计文档](docs/design.md) | [API 文档](docs/api.md)

## 功能

`ChunkKdaFwdPrepare` 完成 KDA 前向中的 Q/K 归一化、gate 前缀和、chunk 内三角系统准备和
Post-WU 计算。它只负责 Prepare 边界，不计算 chunk 间状态递推和最终 attention 输出。

稳定 Python 入口为：

```python
from fla_npu.ops.ascendc import chunk_kda_fwd_prepare

outputs = chunk_kda_fwd_prepare(
    q,
    k,
    v,
    g,
    beta,
    scale=1.0,
    layout="BNSD",
    chunk_size=64,
    epsilon=1e-6,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    lower_bound=-5.0,
    use_exp2=False,
    a_log=None,
    dt_bias=None,
    cu_seqlens=None,
    chunk_indices=None,
    backward_mode="save",
)
```

该入口通过 ctypes 直调 `aclnnChunkKdaFwdPrepare`，不注册 legacy `torch.ops.npu` 接口。

## 输入输出

`q/k/v` 固定为 BF16，K/V 维固定为 128，`chunk_size` 固定为 64。`g/beta` 支持
BF16 或 FP32，`a_log/dt_bias` 固定为 FP32。

| 输入 | dense shape | 无 batch shape | 说明 |
| --- | --- | --- | --- |
| `q/k` | BNSD `[B,HK,T,128]`；BSND `[B,T,HK,128]` | NTD `[HK,T,128]`；TND `[T,HK,128]` | BF16 |
| `v/g` | BNSD `[B,HV,T,128]`；BSND `[B,T,HV,128]` | NTD `[HV,T,128]`；TND `[T,HV,128]` | `v` 为 BF16；`g` 为 BF16/FP32 |
| `beta` | BNSD `[B,HV,T]`；BSND `[B,T,HV]` | NTD `[HV,T]`；TND `[T,HV]` | BF16/FP32 |
| `a_log` | `[HV]` | `[HV]` | 仅 kernel 内计算 gate 时必传 |
| `dt_bias` | `[HV*128]` | `[HV*128]` | 可选，逻辑 shape 为 `[HV,128]` |

`cu_seqlens` 和 `chunk_indices` 是可选 host int array，不是 tensor。前者 shape 为
`[sequence_count+1]`；后者若提供，扁平长度必须为 `2*total_chunks`。

`B/T/HK/HV` 必须为正数且可由 `uint32_t` 表示；GVA 要求 `0 < HK <= HV` 且
`HV % HK == 0`。所有输出固定为 head-major：

`BSND/TND` 输入按 token 读取同一行内的各个 head，跨 head DMA stride 必须可由
`uint32_t` 表示。具体要求为 `(HK-1)*128*2`、`(HV-1)*128*2`、
`(HV-1)*128*sizeof(g)` 和 `(HV-1)*sizeof(beta)` 均不超过 `UINT32_MAX`。

| 输出 | dense shape | 无 batch shape | dtype |
| --- | --- | --- | --- |
| `gk` | `[B,HV,T,128]` | `[HV,T,128]` | FP32 |
| `Aqk/Akk` | `[B,HV,T,64]` | `[HV,T,64]` | BF16 |
| `w/u/qg/kg/qg_scaled` | `[B,HV,T,128]` | `[HV,T,128]` | BF16 |
| `q_hat/k_hat` | `[B,HK,T,128]` | `[HK,T,128]` | BF16 |
| `q_rstd/k_rstd` | `[B,HK,T]` | `[HK,T]` | FP32 |
| `beta_eff` | `[B,HV,T]` | `[HV,T]` | FP32 |

返回顺序固定为 13 个槽位；未按当前策略保留的槽位返回 `None`：

```text
gk, Aqk, Akk, w, u, qg, kg, qg_scaled,
q_hat, k_hat, q_rstd, k_rstd, beta_eff
```

算子 IR 和 kernel ABI 中这 13 个输出全部声明为 `REQUIRED`，因此输出编号以及 workspace、
tiling 参数位置不会随策略变化。L2 接口允许未保留的七项传 `nullptr`；L0 在交给 launcher
前使用不会被当前编译实例写入的合法 descriptor 占位，避免部分 CANN 版本压缩空输出。

按实际消费者分类时，同一个公开输出可以同时属于正向与反向保存量：

| 类别 | 数据 | 消费方 |
| --- | --- | --- |
| 后续正向使用 | `gk/w/u/kg`；`Aqk/qg_scaled` | FwdH；Finalize |
| 反向使用或保存 | `q_hat/k_hat/q_rstd/k_rstd/beta_eff/Aqk/Akk/gk/w/qg/kg` | KDA backward 或重计算策略 |
| 后续算子的可选状态结果 | `h/final_state` | 由 FwdH 产生，不属于 Prepare 的 13 个输出 |

其中 `qg_scaled` 只服务正向 Finalize；`Akk/qg` 在 Prepare 完成 Post-WU 后不再被
正向消费。`u` 服务 FwdH。`gk/Aqk/w/u/kg/qg_scaled` 是跨算子正向流水必需数据，
不受反向策略影响。

### 输出保留策略

`backward_mode` 只控制反向检查点的公开分配和 GM 写回，支持以下四档：

| `backward_mode` | 场景 | 13 个返回槽位中非空的数据 |
| --- | --- | --- |
| `"none"` | 完全不需要反向 | `gk/Aqk/w/u/kg/qg_scaled` |
| `"forward"` | 只要公开必选的 `Akk`，不需要反向重计算中间量 | 上述六项，加 `Akk` |
| `"recompute"` | 有反向，允许重计算 chunk-local 中间量 | 上述六项，加 `Akk/q_hat/k_hat/q_rstd/k_rstd/beta_eff` |
| `"save"` | 有反向，不重计算 | 全部 13 项；相对 `recompute` 额外保存 `qg` |

默认值为 `"save"`，保持原来 13 项全部返回的兼容行为。各档都执行相同的前向数学计算、
静态 UB/L1 布局和跨核同步，仅关闭未请求结果的公开 GM 写回；`Akk/qg` 等数据在本算子
内部仍按 C4/C5/C7 或 BF16 舍入语义使用。四档由 TilingKey 的编译期 `OUTPUT_MODE`
选择，Stage 内不读取运行时输出 mask。

完整 forward 的 `output_final_state` 与 `return_intermediate_states` 是和
`backward_mode` 正交的用户输出开关，只控制 `final_state/h`。这两个数据由 FwdH/完整
forward 产生，不属于 Prepare 的 13 个输出，也不会进入 Prepare 的 tiling 或 kernel 分支。

## 模式

- `use_qk_l2norm_in_kernel=false`：`q_hat=q`、`k_hat=k`，有效 token 的 rstd 为 1。
- `use_gate_in_kernel=false`：`g` 是自然对数域的单 token gate step；算子仍执行 cumsum。
- `use_gate_in_kernel=true`：由 `g + dt_bias` 和 `a_log` 计算 Softplus 或 SafeSigmoid gate。
- `use_beta_sigmoid_in_kernel=false`：`beta_eff=fp32(beta)`。
- `use_beta_sigmoid_in_kernel=true`：`beta_eff=sigmoid(beta)`；启用
  `allow_neg_eigval` 时为 `2*sigmoid(beta)`。
- `use_exp2=false/true`：分别在自然对数域或 log2 域保存累计 gate，数学语义等价。

`safe_gate=true` 只允许和 `use_gate_in_kernel=true` 一起使用，此时 `lower_bound` 必须位于
`[-5,0)`。`allow_neg_eigval=true` 必须同时启用 `use_beta_sigmoid_in_kernel`。

## 变长序列

`cu_seqlens` 是 host int array，必须从 0 开始、以总 token 数结束并单调不减。
`chunk_indices` 若提供，必须严格等于按 `chunk_size=64` 生成的
sequence-major `(sequence_id, local_chunk_id)` 序列。rank-4 变长容器要求 `B=1`。

## 实现

内核支持 A2、A3 和 A5，采用 MIX AIC/AIV 八阶段流水：

```text
V0 -> V1 -> C2 -> V3 -> C4 -> C5 -> V6 -> C7
```

Prepare 首先按 chunk 分核，只有 chunk 数不足以覆盖已用核时才按完整 GVA Q/K 头组切分
head。S 固定为 4，C2 以四个 16 行 query band 生成因果 score。UB、L1 和 workspace 均为
静态地址规划；A5 使用 Mutex 表达核内 pipe 生命周期，A2/A3 使用 HardEvent。跨核采用
ready/free 双向握手，生产者不会覆盖仍被消费者使用的 slot。

生产实现位于 `op_kernel/`。
