# ChunkKdaFwd API

## Python 主入口

```python
from fla_npu.ops.ascendc import chunk_kda_fwd

outputs = chunk_kda_fwd(
    q, k, v, g, beta, scale, chunk_size,
    layout="BSND",
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_indices=None,
    safe_gate=False,
    lower_bound=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    disable_recompute=False,
    return_intermediate_states=False,
    state_v_first=False,
)
```

返回：

```text
(attn_out, final_state, gk, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state)
```

可选输出在 Python 层返回 `None`。`Aqk/Akk` 始终存在；其余保留策略见算子 README。

## aclnn

```cpp
aclnnStatus aclnnChunkKdaFwdGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    bool safeGate,
    double lowerBound,
    bool useGateInKernel,
    bool stateVFirst,
    const aclTensor *attnOut,
    const aclTensor *finalStateOut,
    const aclTensor *gkOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *vNewOut,
    const aclTensor *hOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnChunkKdaFwd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

aclnn L2 只描述张量与算法契约，不接收或解释 autograd 重计算策略：

- `attnOut/aqkOut/akkOut` 是必选输出。
- `finalStateOut/gkOut/wOut/uOut/qgOut/kgOut/vNewOut/hOut` 均为相互独立的可选输出。
- `w/u/qg/kg/vNew/h` 的 L0 阶段固定写内部 compute 张量；对应可选输出非空时，L2 通过
  `ViewCopy` 导出，为空时只保留前向内部生命周期。`gkOut` 非空时直接复用为 `gkCompute`，
  避免目标场景额外复制整张 FP32 gate。
- `finalStateOut != nullptr` 同时表示本次需要计算并写出最终状态。
- `hCompute` 是 FwdH 到 Finalize 的内部必需 head-major 张量；`hOut` 是独立的公开可选输出。
  `hOut == nullptr` 不会跳过内部 `hCompute`，只是不向调用方公开该中间状态；非空时由
  L2 转为固定 sequence-major 后导出。

`output_final_state/disable_recompute/return_intermediate_states` 只存在于 Python 和 legacy torch
包装层，由上层按 FLA 的保留策略决定向 L2 传入哪些输出指针。

## 输入与输出布局

`layout` 只解释 q/k/v/g/beta 输入。输出固定为：

- `attnOut`: BSND 或 TND。
- `finalStateOut`: `[N,H_v,K,V]` 或 `stateVFirst=true` 时 `[N,H_v,V,K]`。
- `gkOut/AqkOut/AkkOut/wOut/uOut/qgOut/kgOut/vNewOut`: BNSD/NTD。
- `hOut`: dense 为 `[B,N_c,H_v,K,V]`，varlen 为 `[N_c,H_v,K,V]`；
  `stateVFirst=true` 时交换末两维。

完整 Shape 表见 [KDA 模型符号表](../../README.md#model-shape-symbols)。

## Gate 语义

```text
useGateInKernel=false:
    gate = g
useGateInKernel=true, safeGate=false:
    gate = -exp(A_log) * softplus(g + dt_bias)
useGateInKernel=true, safeGate=true:
    gate = lowerBound * sigmoid(exp(A_log) * (g + dt_bias))
gk = chunk_local_cumsum(gate) / ln(2)
```

`safeGate` 的 true/false 都支持；`useGateInKernel=false` 时仍支持 `safeGate=true` 的后续稳定计算路径。

## 示例

```python
import torch
from fla_npu.ops.ascendc import chunk_kda_fwd

B, T, H, K, V = 1, 128, 4, 128, 128
q = torch.randn(B, T, H, K, device="npu", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn(B, T, H, V, device="npu", dtype=torch.float16)
g = -torch.rand(B, T, H, K, device="npu", dtype=torch.float32) * 0.01
beta = torch.rand(B, T, H, device="npu", dtype=torch.float32)

attn_out, final_state, *_ = chunk_kda_fwd(
    q, k, v, g, beta, K ** -0.5, 64,
    layout="BSND",
    output_final_state=True,
    safe_gate=True,
)
assert attn_out.shape == (B, T, H, V)
assert final_state.shape == (B, H, K, V)
```

## 调用途径

| 路径 | 入口 |
| --- | --- |
| 稳定 Python | `fla_npu.ops.ascendc.chunk_kda_fwd` |
| aclnn | `aclnnChunkKdaFwdGetWorkspaceSize/aclnnChunkKdaFwd` |
| legacy | 显式加载后的 `torch.ops.npu.npu_chunk_kda_fwd` |
| 受限直调样例 | `torch.ops.ascend_ops.chunk_kda_fwd_direct` |

直调样例仅覆盖 dense BNSD、K=128、V=128/256，并保留“调用方传入已累计 gk”的低层测试接口；
公开顶层语义以稳定 Python/aclnn 接口为准。
