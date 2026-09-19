# split_qkv_tp_rmsnorm_rope

## Description

- **Function**: Implements the two-stage Tensor Parallel QKV split, global Q/K RMSNorm, and NeoX-style RoPE pipeline used by MiniMax-M2 attention. The public operator contains `_split_qkv_and_compute_local_qk_var_kernel`, a TP all-reduce, and `_apply_global_rmsnorm_kernel`; the two kernels therefore form one semantic operator.
- **Formula**: On TP rank `r`, let local Q and K widths be `Q` and `K`, and let `W = tp_world`. The first kernel computes local means

  $$
  \mu_{q,r}=\frac{1}{Q}\sum_{j=0}^{Q-1}q_{r,j}^2,
  \qquad
  \mu_{k,r}=\frac{1}{K}\sum_{j=0}^{K-1}k_{r,j}^2.
  $$

  The wrapper all-reduces these values and the second kernel computes

  $$
  \mu_q=\frac{1}{W}\sum_{r=0}^{W-1}\mu_{q,r},
  \qquad
  z_{q,r,j}=q_{r,j}\operatorname{rsqrt}(\mu_q+\epsilon)w_{q,r,j},
  $$

  with the same equations for K. For equal-width TP shards, this is the mean over the global sharded Q or K vector. Q and K are then reshaped into local heads and rotated over their first `R` dimensions:

  $$
  y_{0:R/2}=z_{0:R/2}c-z_{R/2:R}s,
  \qquad
  y_{R/2:R}=z_{R/2:R}c+z_{0:R/2}s.
  $$

  V is copied unchanged.
- **Algorithm flow** (processed token by token):
  1. Flatten the local `[Q | K | V]` input and launch `min(T, num_vector_cores)` programs.
  2. `_split_qkv_and_compute_local_qk_var_kernel` grid-strides over token tiles, loads power-of-two-padded feature blocks with masks, copies Q/K/V, and writes fp32 local Q/K means to `[T, 2]`.
  3. If `tp_world > 1`, all-reduce the local means across the initialized tensor-parallel group.
  4. `_apply_global_rmsnorm_kernel` assigns each program a contiguous token range, applies the global scale and per-feature local weights, performs partial or full NeoX RoPE, and stores Q/K in place.
- **Supported modes**: Atlas A2 and Atlas A3 for MiniMax-M2/M2.5 Tensor Parallel inference, with a registered fake implementation for graph tracing. The current single-card accuracy test has been verified on Atlas A2; multi-rank and Ascend 950 accuracy are not claimed by the current test evidence.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `input` | Input | Local TP shard of the fused projection, laid out as Q followed by K and V | bf16 | Contiguous 2-D, `[T, Q + 2 * K]` |
| `q_weight` | Input | Per-feature RMSNorm weight for the local Q shard | fp32 | Contiguous 1-D, `[Q]` |
| `k_weight` | Input | Per-feature RMSNorm weight for the local K shard | fp32 | Contiguous 1-D, `[K]` |
| `q_hidden_size` | Input (attribute) | Local flattened Q width `Q` | int32 | Scalar |
| `kv_hidden_size` | Input (attribute) | Local flattened K and V width `K` | int32 | Scalar |
| `head_dim` | Input (attribute) | Per-head feature size `D` | int32 | Scalar |
| `rotary_dim` | Input (attribute) | Number of leading dimensions per head transformed by RoPE | int32 | Scalar |
| `eps` | Input (attribute) | Positive global RMSNorm epsilon | fp32 | Scalar |
| `tp_world` | Input (attribute) | Number of equal-width tensor-parallel shards participating in the mean | int32 | Scalar |
| `cos` | Input | Cosine cache; the kernel reads the first `rotary_dim / 2` values of each token row | bf16 | Contiguous and viewable as `[T, C]`, where `C >= rotary_dim / 2` |
| `sin` | Input | Sine cache; the kernel reads the first `rotary_dim / 2` values of each token row | bf16 | Contiguous and viewable as `[T, C]`, where `C >= rotary_dim / 2` |
| `q_output` | Output | Globally normalized and rotated local Q shard | bf16 | Contiguous 2-D, `[T, Q]` |
| `k_output` | Output | Globally normalized and rotated local K shard | bf16 | Contiguous 2-D, `[T, K]` |
| `v_output` | Output | Local V shard copied from the input | bf16 | Contiguous 2-D, `[T, K]` |

## Constraints

- `input.shape[1]` must equal `q_hidden_size + 2 * kv_hidden_size`; both hidden sizes must be divisible by `head_dim`.
- `rotary_dim` must be positive, even, and no greater than `head_dim`. `cos` and `sin` must be viewable as token-major rows with at least `rotary_dim / 2` values and compatible row strides. The direct test uses compact `[T, rotary_dim / 2]` rows; the MiniMax production cache may use repeated full-width `[1, T, 1, rotary_dim]` rows, of which the kernel reads the first half.
- `q_weight.numel()` must equal the local `q_hidden_size`; `k_weight.numel()` must equal the local `kv_hidden_size`. This operator uses per-feature shard weights, not a single `[head_dim]` weight shared across heads.
- `tp_world` must be positive. When `tp_world > 1`, it must match the initialized tensor-model-parallel process group, and every rank must use equal Q/K shard widths and identical token ordering and count. `tp_world=1` does not enter a collective.
- `input` must be dense contiguous with row stride exactly `q_hidden_size + 2 * kv_hidden_size`; the first kernel hard-codes this stride. The first kernel pads feature loads to powers of two and masks the padded columns.
- `T` must be positive. Although the wrapper contains an empty-output return, it first evaluates `input.view(T, -1)`, so a zero-element input is not accepted by the current implementation.
- The distributed all-reduce is part of the operator semantics. The current test uses only `tp_world=1`, so multi-rank numerical behavior remains a documented test gap.

## Origin and Differences

- **Origin**: Developed in vLLM-Ascend PR #7376 as a native fused TP operator for the MiniMax-M2/M2.5 attention path. Later changes optimized the apply stage and added grid-stride batched loading for the local-statistics stage.
- **Differences**:
    - NPU adaptation for performance: fuses local QKV splitting and fp32 statistics, reduces only two scalars per token across TP ranks, and applies global RMSNorm and RoPE in a second vector-core kernel;
    - Modified for a specific vllm-ascend logic or different input parameters: computes RMSNorm across the globally TP-sharded flattened Q/K vectors, consumes per-feature local weights, and is inserted directly into the patched MiniMax-M2 attention forward path.

## Test Cases

The direct custom-op test covers `T={1,8,32}`, `(Hq,Hkv)={(6,1),(8,2)}`, `D=128`, full and partial RoPE with `R={128,64}`, bf16 QKV/cos/sin, fp32 per-feature weights, and `tp_world=1`. Q, K, and V are compared with a PyTorch reference using `rtol=5e-3, atol=5e-2`. The test executes both Triton kernels but deliberately does not enter the multi-rank all-reduce path.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_tp_rmsnorm_rope.py
```
