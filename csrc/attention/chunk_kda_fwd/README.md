# ChunkKdaFwd

## 功能

`ChunkKdaFwd` 对齐不涉及 CP 切分的 FLA `chunk_kda_fwd` 顶层语义。公共接口接收 raw gate 或已激活的
自然对数 gate；Gate、Prepare、PostWu、FwdH 和 Finalize 均在一个物理 `ChunkKdaFwd` L0 内完成，
L2 不再拼接或依次发射多个阶段 L0。

Shape 符号与布局约定见 [KDA 模型符号表](../README.md#model-shape-symbols)。

## Gate 公式

令 `x = g + dt_bias`。逐 token、逐 K 维的自然对数衰减为：

```text
use_gate_in_kernel = false:
    gate = g

use_gate_in_kernel = true, safe_gate = false:
    gate = -exp(A_log) * softplus(x)

use_gate_in_kernel = true, safe_gate = true:
    gate = lower_bound * sigmoid(exp(A_log) * x)
```

随后在每个 chunk 内计算：

```text
gk_i = cumsum(gate)_i / ln(2)
```

因此后续 `exp2(gk)` 与自然指数 gate 严格绑定，不暴露额外 gate scale。

## 输入

| 名称 | 必选性 | Shape/Dtype | 说明 |
| --- | --- | --- | --- |
| `q/k` | 必选 | 输入 layout 对应 Shape；FP16/BF16 | Query/Key |
| `v` | 必选 | 输入 layout 对应 Shape；与 q 同 dtype | Value |
| `g` | 必选 | 输入 layout 对应 K 维 Shape；FP32/BF16 | raw gate 或已激活自然对数 gate |
| `beta` | 必选 | 去掉 g 的 K 维；FP32/BF16 | Delta 系数 |
| `A_log` | 条件必选 | `[H_v]`，FP32 | `use_gate_in_kernel=true` 时必选 |
| `dt_bias` | 可选 | `[H_v*K]`，FP32 | gate bias |
| `initial_state` | 可选 | `[N,H_v,K,V]` 或 `[N,H_v,V,K]`，FP32 | 由 `state_v_first` 解释 |
| `cu_seqlens` | 可选 | `[N+1]`，INT64 | 变长序列 |
| `chunk_indices` | 可选 | `[2*N_c]`，INT64 | canonical chunk 顺序 |

`layout` 只描述上述输入。BSND/TND 由 L2 使用 `l0op::Transpose` 转为内部 BNSD/NTD。

## 输出

Python 返回顺序为：

```text
(attn_out, final_state, gk, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state)
```

- `attn_out` 固定为 BSND/TND。
- `final_state` 固定按序列排列，末两维服从 `state_v_first`。
- `Aqk/Akk` 始终返回，固定为 head-major。
- `gk/w/u/qg/kg/v_new` 是供反向使用的 head-major 中间量。
- 公开 `h` 固定为 sequence-major；内部 `hCompute` 保持 head-major 供 Finalize 使用。
- 第 12 个返回值是 Python 层对 `initial_state` 的原对象透传，不是 aclnn 输出。

输出保留策略对齐 fla-org
[`chunk_kda_fwd`](https://github.com/fla-org/flash-linear-attention/blob/0f0f0c97af39343855b43bbbaddcedfda5cb9d77/fla/ops/kda/chunk_fwd.py)
提交 `0f0f0c97af39343855b43bbbaddcedfda5cb9d77`：

| 条件 | 返回 |
| --- | --- |
| `output_final_state=true` | 返回 `final_state`，否则为 `None` |
| `use_gate_in_kernel=false` 或 `disable_recompute=true` | 返回 `gk` |
| 始终 | 返回 `Aqk/Akk` |
| `disable_recompute=true` | 返回 `w/u/qg/kg/v_new` |
| `disable_recompute=true` 或 `return_intermediate_states=true` | 返回 `h` |

这是 `fla_npu.ops.ascendc.chunk_kda_fwd` 的低层 12 返回值语义；不涉及 CP。aclnn L2 不接收
`output_final_state/disable_recompute/return_intermediate_states`，每个可选输出是否写出仅由对应
输出指针是否为空决定。`w/u/qg/kg/v_new/h` 的 L0 阶段固定写内部 compute 张量，L2 仅在
对应指针非空时通过 `ViewCopy` 导出；`gkOut` 非空时直接复用为 `gkCompute`，避免目标场景
额外复制整张 FP32 gate。内部 `hCompute` 是 FwdH 到 Finalize 的必需 head-major 阶段结果；
公开 `hOut` 非空时，L2 转为 sequence-major 后导出。`hOut` 为空时仍创建 `hCompute`，但不
作为第 11 个 Python 返回值公开。

## 属性

| 名称 | 默认值 | 支持范围 |
| --- | --- | --- |
| `layout` | `BSND` | `BSND/BNSD/TND/NTD` |
| `scale` | 必传 | 通常为 `K**-0.5` |
| `chunk_size` | `64` | `64/128` |
| `output_final_state` | `false` | bool |
| `safe_gate` | `false` | bool |
| `lower_bound` | `-5.0` | safe raw gate 时 `[-5,0)` |
| `use_gate_in_kernel` | `false` | bool |
| `disable_recompute` | `false` | bool |
| `return_intermediate_states` | `false` | bool |
| `state_v_first` | `false` | bool |

## 支持范围

- A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`)。
- `K/V` 为 `[16,256]` 内 16 的倍数；交付重点覆盖 K=128、V=128/256。
- `chunk_size` 为 64/128。
- TND/NTD 均支持多 head。
- 变长调用最多 1024 条逻辑序列，rank-4 变长输入要求 B=1。

## 验证

唯一用例规格是 `tests/op_cases/chunk_kda_fwd.json`。数值测试位于
`tests/operators/chunk_kda_fwd/accuracy/`，性能使用 `tests/operators/chunk_kda_fwd/performance/profile.py`
和 `msopprof`。

完整 API 见 [API 文档](docs/api.md)，阶段和内存设计见 [设计文档](docs/design.md)。
