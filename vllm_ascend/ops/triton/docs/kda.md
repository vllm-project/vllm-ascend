# KDA (Kimi Delta Attention)

## Description

- **Function**: Provides the Ascend Triton forward implementation of KDA. `chunk_kda` decomposes prefill into 64-token chunks, while `fused_recurrent_kda` evaluates the recurrent update used by decode. The module also contains the KDA decay-gate and gated normalization helpers.
- **Formula**: Let the mathematical recurrent state be $S_t \in \mathbb{R}^{K \times V}$ and the key-dimension decay gate be $g_t \in \mathbb{R}^{K}$. After optional Q/K normalization, KDA computes

  $$
  \widetilde{S}_t = \operatorname{Diag}(\exp(g_t)) S_{t-1},
  $$

  $$
  \delta_t = \beta_t \odot \left(v_t - \widetilde{S}_t^{\mathsf T} k_t\right),
  $$

  $$
  S_t = \widetilde{S}_t + k_t \delta_t^{\mathsf T}, \qquad
  o_t = \text{scale} \cdot S_t^{\mathsf T} q_t.
  $$

  Kernels physically store the transposed state as $S_t^{\mathsf T} \in \mathbb{R}^{V \times K}$, with the key dimension K contiguous. In the chunk path, $G_i = \sum_{r=0}^{i} g_r$ and token interactions use $E_{ij}=\exp(G_i-G_j)$. The KKT kernels construct a strictly lower-triangular correction $L$ and causal query-key matrix $P$, after which the solve kernels compute $M=(I+L)^{-1}$.
- **Algorithm flow**:
  1. When requested, normalize chunk Q/K with the standalone tiled L2Norm kernel. Recurrent mode performs Q/K normalization inside its fused kernel. The persistent L2Norm kernel is also available as a shared helper.
  2. Compute the chunk cumulative vector gate with `chunk_local_cumsum_vector_kernel`, then convert it to base-2 exponent space. `chunk_local_cumsum_scalar_kernel` is the scalar-gate variant of the shared cumulative-sum utility and is not called by `chunk_kda`.
  3. Build the inter-subchunk and intra-subchunk KKT blocks with `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter` and `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra`.
  4. Invert each lower-triangular block with `solve_tril_16x16_kernel_kda`, then assemble 32x32 and 64x64 inverses with `merge_16x16_to_32x32_inverse_kernel_kda` and `merge_16x16_to_64x64_inverse_kernel_kda`.
  5. Recompute W/U with `recompute_w_u_fwd_kernel`. `chunk_gated_delta_rule_fwd_kernel_h_blockdim64_kda` propagates the state using FP32 accumulation and FP32 final states; stored per-chunk state snapshots use the key dtype.
  6. Produce the chunk output with `chunk_gla_fwd_kernel_o`.
  7. For recurrent execution, `fused_recurrent_gated_delta_rule_fwd_kernel` combines vector decay, optional Q/K normalization, beta application, output generation, and state update. It supports flattened variable-length inputs and slot-indexed in-place state updates.
  8. `kda_gate_fwd_kernel` implements the standalone negative-softplus decay gate. `layer_norm_gated_fwd_kernel` and `layer_norm_gated_fwd_kernel1` implement the gated LayerNorm/RMSNorm helper for feature dimensions up to and above 512, respectively.
- **Supported modes**: Fixed-length and flattened variable-length chunk inputs, recurrent execution, and slot-indexed in-place recurrent states. The raw kernels in this PR were validated with FP32, FP16, and BF16 inputs on Atlas A3. Other Ascend platforms are not claimed by this test result.

## Parameters

### `chunk_kda`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `q` | Input | Query tensor `[B, T, H, K]` | fp32 / fp16 / bf16 | ND |
| `k` | Input | Key tensor `[B, T, H, K]` | same as `q` | ND |
| `v` | Input/Output | Value work buffer `[B, T, H, V]`; the chunk output is written to this buffer | same as `q` | ND |
| `g` | Input | Key-dimension decay gate `[B, T, H, K]`, consumed through `exp(g)` | fp32 / fp16 / bf16 | ND |
| `beta` | Input | Scalar update factor per token and head, `[B, T, H]` | fp32 / fp16 / bf16 | ND |
| `initial_state` | Input | Required initial state `[N, H, V, K]` in physical `[V, K]` layout | fp32 | ND |
| `scale` | Attribute | Query scale; defaults to `K ** -0.5` | fp32 | scalar |
| `cu_seqlens` | Input | Cumulative offsets `[N + 1]` for a flattened variable-length batch | int32 / int64 | ND |
| `output_final_state` | Attribute | Controls whether the FP32 state after the last chunk is returned | bool | scalar |
| `use_qk_l2norm_in_kernel` | Attribute | Applies the standalone tiled L2Norm kernel to Q/K before chunk execution | bool | scalar |
| `prebuilt_meta` | Attribute | Optional cached metadata containing 64-token `chunk_indices_chunk64` and `chunk_offsets_chunk64` | Python object | N/A |
| `o` | Output | Attention output `[B, T, H, V]`, backed by the contiguous value work buffer | fp32 / fp16 / bf16 | ND |
| `final_state` | Output | Optional final state `[N, H, V, K]` in physical `[V, K]` layout | fp32 | ND |

### `fused_recurrent_kda`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `q` | Input | Query tensor `[B, T, H, K]` | fp32 / fp16 / bf16 | ND |
| `k` | Input | Key tensor `[B, T, H, K]` | same as `q` | ND |
| `v` | Input | Value tensor `[B, T, HV, V]` | fp32 / fp16 / bf16 | ND |
| `g` | Input | Vector decay gate `[B, T, HV, K]` | fp32 / fp16 / bf16 | ND |
| `beta` | Input | Required scalar beta `[B, T, HV]` or value-vector beta `[B, T, HV, V]` | fp32 / fp16 / bf16 | ND |
| `initial_state` | Input/Input-Output | Required slot- or token-indexed state whose per-state layout is `[HV, V, K]`; aliases `final_state` in in-place mode | fp32 | ND |
| `scale` | Attribute | Query scale; defaults to `K ** -0.5` | fp32 | scalar |
| `cu_seqlens` | Input | Cumulative offsets `[N + 1]` for flattened variable-length input | int32 / int64 | ND |
| `inplace_final_state` | Attribute | Updates slot-indexed `initial_state` in place when enabled | bool | scalar |
| `use_qk_l2norm_in_kernel` | Attribute | Fuses Q/K L2 normalization into the recurrent kernel | bool | scalar |
| `ssm_state_indices` | Input | State-slot mapping; one-dimensional for one-token sequences or two-dimensional for multi-token sequences | int32 / int64 | ND |
| `o` | Output | Output storage is allocated like `k`; the current public KDA integration therefore requires `H = HV` and `K = V` | fp32 / fp16 / bf16 | ND |
| `final_state` | Output | In-place state alias, or token-indexed states `[T, HV, V, K]` in non-in-place flattened mode | fp32 | ND |

### Standalone KDA helpers

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `g` | Input | Raw gate `[..., H * K]` for `fused_kda_gate`, or activation gate `[..., D]` for `rms_norm_gated` | fp32 / fp16 / bf16 | ND |
| `A` | Input | Per-head log-decay coefficient for `fused_kda_gate`, shaped `[H]` or `[1, 1, H, 1]` | fp32 / fp16 / bf16 | ND |
| `head_k_dim` | Attribute | Key dimension K used to reshape the fused gate output to `[..., H, K]` | int32 | scalar |
| `g_bias` | Input | Optional contiguous gate bias `[H, K]` or equivalent flattened `[H * K]` | fp32 / fp16 / bf16 | ND |
| `beta` | Attribute | Softplus beta used by `fused_kda_gate`; defaults to `1.0` | fp32 | scalar |
| `threshold` | Attribute | Softplus linear-approximation threshold; defaults to `20.0` | fp32 | scalar |
| `x` | Input/Input-Output | Input `[..., D]` to `rms_norm_gated`; its contiguous work buffer may be reused for the output | fp32 / fp16 / bf16 | ND |
| `weight` | Input | RMSNorm weight `[D]`; the value may be `None` | fp32 / fp16 / bf16 | ND |
| `bias` | Input | RMSNorm bias `[D]`; the value may be `None` | fp32 / fp16 / bf16 | ND |
| `residual` | Input | Optional residual with the same shape as `x` | fp32 / fp16 / bf16 | ND |
| `activation` | Attribute | Gate activation: `swish`/`silu` or `sigmoid` | string | scalar |
| `prenorm` | Attribute | Returns both the gated output and residual output when enabled | bool | scalar |
| `residual_in_fp32` | Attribute | Uses FP32 for a newly allocated residual path when no residual tensor is supplied | bool | scalar |
| `eps` | Attribute | RMSNorm epsilon; defaults to `1e-6` | fp32 | scalar |
| `gate_output` | Output | Negative-softplus gate `[..., H, K]` from `fused_kda_gate` | fp32 | ND |
| `norm_output` | Output | Gated RMSNorm output with the same shape and dtype as `x` | fp32 / fp16 / bf16 | ND |

## Constraints

- Tensor inputs to one operation must reside on the same NPU and agree on their batch, token, head, and feature dimensions as described above.
- `initial_state` and `beta` are required by the current chunk and recurrent wrappers even though some Python annotations provide `None` defaults.
- Chunk mode uses a fixed chunk size of 64 and 16-token inverse sub-blocks, and supports `K <= 256`. Its state layout is `[N, H, V, K]`; the state is not updated in place.
- With `cu_seqlens`, inputs must be flattened to `B = 1`; offsets must start at 0, end at T, and be nondecreasing.
- Without chunk Q/K normalization, Q and K must already have the dense layout expected by the raw kernels. The wrappers make V, G, beta, and state contiguous; normalization helpers make their own inputs contiguous.
- The current recurrent output allocation requires the public integration shape `H = HV` and `K = V`. The more general H/HV and K/V raw-kernel contract must not be used through this wrapper without matching output storage.
- In recurrent in-place mode, `ssm_state_indices` is required. A one-dimensional mapping is valid only when every sequence contains one token; multi-token sequences require a row-major two-dimensional mapping with a contiguous token dimension.
- Recurrent state slots use `[HV, V, K]` with contiguous K. Slot indices less than or equal to zero are null entries: the kernel returns without producing a valid output for that sequence, so callers must ignore those positions.
- Recurrent non-in-place final-state storage is supported for flattened `B = 1` input. The public wrapper does not expose the low-level speculative-decoding `num_accepted_tokens` path.
- L2Norm and gated LayerNorm/RMSNorm require `D <= 65536 / element_size` for each row.
- For `fused_kda_gate`, `A.numel() * head_k_dim` must equal `g.shape[-1]`; `g_bias`, when present, must contain one K-element bias vector per head.
- Only inference forward paths are implemented; backward/autograd kernels are not provided.

## Origin and Differences

- **Origin**: The KDA algorithm and GPU Triton decomposition are based on the `flash-linear-attention` KDA implementation used by vLLM. The copied source retains its MIT license notice; vLLM Ascend ports the implementation to its Ascend Triton backend.
- **Differences**:
    - Uses Ascend-oriented grids, tiling, block pointers, vector-core L2Norm, and specialized triangular-solve and state-update kernels.
    - Converts cumulative gates with `log2(e)` for `exp2` evaluation and keeps selected KKT values, accumulators, and final recurrent states in FP32.
    - Accepts prebuilt chunk metadata and supports slot-indexed in-place recurrent states for inference integration.
    - Provides forward-only inference kernels and does not include the upstream backward/autograd or newer training-oriented options.

## Test Cases

The tests contain one test function per production kernel and one direct raw `kernel[grid](...)` launch in each function. Parameterization produces 25 collected cases for 16 kernels; every collected case therefore executes one target-kernel launch and compares it with an independent PyTorch CPU FP32 reference. The suite covers FP32, FP16, BF16, fixed and variable-length layouts, partial chunks, non-power-of-two dimensions, optional inputs, and in-place/non-in-place state paths. On Atlas A3, the complete suite result is `25 passed, 0 failed`.

| Test file | Kernels | Cases | Precision tolerance |
| --- | ---: | ---: | --- |
| `test_kda_aux_kernels.py` | 7 | 7 | cumsum `1e-5/1e-5`; L2Norm `3e-4/1e-3`; solve/merge `5e-4/5e-4` (`rtol/atol`) |
| `test_kda_core_kernels.py` | 7 | 14 | low-precision primary outputs `2e-3` to `6e-2`; normalization mean/rstd `8e-4` |
| `test_kda_state_kernels.py` | 2 | 4 | FP16 `1e-2/1e-2`; BF16 `3e-2/3e-2` (`rtol/atol`) |

```bash
pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kda_aux_kernels.py \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kda_core_kernels.py \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kda_state_kernels.py
```
