# split_qkv_rmsnorm_rope_vnorm

## Description

- **Function**: Fused pre-attention stage for models that normalize V. In a single Triton kernel it splits the fused QKV projection, applies per-head RMSNorm to Q, K and V, applies NeoX-style RoPE to Q and K, and stores the three tensors separately. It replaces the host-side chain of `split` × 1, `npu_rms_norm` × 3 and `npu_rotary_embedding` × 1, which wrote every intermediate back to global memory. `split_qkv_rmsnorm_rope` cannot serve these models because it copies V out of the fused tensor without normalizing it.
- **Formula**:
    - Input `input`: `[num_tokens, q_hidden_size + 2 * kv_hidden_size]`, laid out as `[ q \| k \| v ]`.
    - Split by column offset (`q_hidden_size`, `kv_hidden_size`, `kv_hidden_size`):
        - `q`: `[num_tokens, q_head_num, head_dim]`
        - `k`: `[num_tokens, kv_head_num, head_dim]`
        - `v`: `[num_tokens, kv_head_num, head_dim]` (normed, no RoPE)
    - RMSNorm over the head dimension (`D = head_dim`), reduction in float32:
        - `rstd = rsqrt(sum(x^2) / D + eps)`
        - `q, k`: `y = x * rstd * w [+ b]` with the learnable scale `w` (`q_weight` / `k_weight`)
        - `v`: `y = x * rstd`, no scale, no bias, matching `RMSNorm(head_dim, has_weight=False)`
    - NeoX RoPE over the first `R = rope_dim` elements of each head (`half = R // 2`), applied to the **normed** Q and K only (`y` is the output of the RMSNorm step above); V is stored without rotation:
        - `y1 = y[..., :half]`, `y2 = y[..., half:R]`
        - `rotated = [y1 * cos - y2 * sin, y2 * cos + y1 * sin]`
        - `out = [rotated, y[..., R:]]`; the tail beyond `R` passes through un-rotated, which only occurs when `rope_dim < head_dim`
        - `cos`/`sin` are the per-position rows gathered from `cos_sin_cache` (`[max_position, rope_dim] = concat(cos, sin)`); `cos` is a row of `[0, rope_dim/2)` and `sin` is a row of `[rope_dim/2, rope_dim)`.
- **Algorithm flow** (processed token by token, independently):
    1. Launch with `grid = (num_vectorcore,)`; each vector core owns a contiguous range of `cdiv(num_tokens, num_vectorcore)` tokens and handles the full hidden dimension for them.
    2. Size a UB-aware token tile (`batch_size_per_iter_per_vec`) from the UB budget (`UB_SIZE = 85 KB`) and a per-token element factor, so every buffer of the tile stays resident.
    3. One loop over token tiles, sharing the two norm weights (loaded once into registers):
        - Load the whole fused row `[ q \| k \| v ]` in one masked load and view it head-major as `[tile * (q_head_num + 2 * kv_head_num), head_dim]`.
        - Gather `cos_sin_cache[positions]` once per tile into `[tile, rope_dim]` (`get_element` + `insert_slice`), then split into `cos` / `sin` broadcast over every head.
        - Run **one** RMSNorm reduction (`tl.sum` over `head_dim`, accumulated in float32) covering every Q, K and V head of the tile.
        - Slice Q (head offset `0`), K (offset `q_head_num`) and V (offset `q_head_num + kv_head_num`) with `extract_slice`; apply `q_weight` / `k_weight` (and optional biases) and NeoX RoPE to Q and K; store V directly.
    4. All intermediates stay in UB/registers; global memory is touched only on input load and output store.
- **Supported modes**: Atlas A2 (Triton kernel). Atlas A3 takes the same `BaseDeviceAdaptor` dispatch path but has not been validated. Used by the Gemma4 pre-attention path in `vllm_ascend/patch/worker/patch_gemma4.py` when `HAS_TRITON`, the norm weights and cos/sin cache are bf16, the layer does not share KV, no q/k/v norm carries a bias, and one token's tiles fit a vector core. Works in both eager and graph-capture modes. **Not supported on Ascend 950**: that platform routes `split_qkv_rmsnorm_rope` to a SIMT kernel that replaces the in-kernel scalar cos/sin gather with a pre-gathered buffer, and no SIMT variant of this operator exists yet; callers fall back to the unfused chain there.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `input` | Input | Fused QKV projection laid out as `[ q \| k \| v ]` (`[num_tokens, q_hidden_size + 2 * kv_hidden_size]`) | bf16 | ND |
| `cos_sin_cache` | Input | RoPE cache `[ max_position_embeddings, rope_dim ] = concat(cos, sin)` | bf16 | ND |
| `positions` | Input | Per-token positions used to index `cos_sin_cache` | int64 | ND (1-D) |
| `q_weight` | Input | RMSNorm scale for Q (`[head_dim]`) | bf16 | ND (1-D) |
| `k_weight` | Input | RMSNorm scale for K (`[head_dim]`) | bf16 | ND (1-D) |
| `q_bias` | Input (optional) | Post-norm bias for Q (`[head_dim]`); unused when `None` | bf16 | ND (1-D) |
| `k_bias` | Input (optional) | Post-norm bias for K (`[head_dim]`); unused when `None` | bf16 | ND (1-D) |
| `q_hidden_size` | Input (attribute) | `q_head_num * head_dim` | int32 | scalar |
| `kv_hidden_size` | Input (attribute) | `kv_head_num * head_dim` | int32 | scalar |
| `head_dim` | Input (attribute) | Head dimension shared by Q, K and V; RMSNorm reduces and RoPE views by this | int32 | scalar |
| `eps` | Input (attribute) | RMSNorm epsilon, applied identically to Q, K and V | fp32 | scalar |
| `q_output` | Output | Normed + RoPE'd Q `[ num_tokens, q_hidden_size ]` | bf16 | ND |
| `k_output` | Output | Normed + RoPE'd K `[ num_tokens, kv_hidden_size ]` | bf16 | ND |
| `v_output` | Output | Normed V `[ num_tokens, kv_hidden_size ]` | bf16 | ND |

## Constraints

- `input.dtype` must be `bfloat16`, as must `q_weight`, `k_weight` and `cos_sin_cache`; the kernel holds its cos/sin, norm and RoPE buffers as `tl.bfloat16`, so the fused path is only taken for bf16.
- `input` must be 2-D `[num_tokens, hidden]`; `num_tokens` is flattened from `batch * seq_len` by the caller.
- `positions` must be 1-D of `int64`, with every element `< max_position_embeddings` (the row count of `cos_sin_cache`).
- `head_dim > 0`, and both `q_hidden_size` and `kv_hidden_size` must be exact multiples of `head_dim`. The kernel reinterprets the loaded row head-major, so a remainder would misalign the per-head reduction and the Q/K/V slice offsets.
- `rope_dim` is the last dim of `cos_sin_cache` and must be even and `<= head_dim`. RoPE rotates only the first `rope_dim` elements of each head; the rest is copied unchanged.
- The RoPE cache must be NeoX style: `cos_sin_cache[p]` holds `cos` then `sin`, each `rope_dim / 2` wide.
- RMSNorm is applied to `q`, `k` and `v`; only `q` and `k` take a learnable scale and an optional bias, and only `q` and `k` are rotated. A single `eps` is shared by all three.
- One token's buffers must fit a single vector core's UB. `qkv_rmsnorm_rope_vnorm_fits_ub(q_hidden_size, kv_hidden_size, head_dim, rope_dim)` reports this and callers must fall back to the unfused chain when it returns `False`. The budget is per TP rank, so a larger tensor parallel size makes more layers eligible.
- `q_hidden_size`, `kv_hidden_size`, `head_dim`, `eps` and all tile sizes are compile-time `constexpr`; `num_tokens` is dynamic (the token loop and masks handle arbitrary token counts).
- Only for NPU inference (prefill/decode).

## Origin and Differences

- **Origin**: Derived from `vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.py`. The Q/K stage (grid, UB tiling, per-tile cos/sin gather, `extract_slice` / `insert_slice` RoPE and partial-RoPE handling) is inherited from it, with the head-major view widened from `q_head_num + kv_head_num` to `q_head_num + 2 * kv_head_num` so the V heads share it.
- **Differences**:
    - Modified for a specific vllm-ascend logic or different input parameters: V is normalized per head with a weight-less RMSNorm instead of being copied out of the fused tensor, which is what Gemma4 requires (`RMSNorm(head_dim, has_weight=False)`). The base operator's unfused replacement for these models would leave the V norm as a separate op.
    - NPU adaptation for performance: because V now needs the same float32 reduction scratch as Q and K, the separate wide-tile V loop that the base operator uses for a plain copy no longer pays for itself. V is folded into the Q/K loop, so a single RMSNorm reduction covers every Q, K and V head of the tile and each token's input arrives as one contiguous load instead of two disjoint column ranges. The UB tiling factor widens accordingly: the loaded slice and its float32 copy grow from `q + kv` to `q + 2 * kv` elements, plus one bfloat16 copy of the V output. `qkv_rmsnorm_rope_vnorm_fits_ub` exposes the resulting per-token budget so callers can resolve eligibility before the kernel is reached.

## Test Cases

- Single-operator accuracy test: `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_rope_vnorm.py`, parameterized over real `gemma-4-31B-it` per-rank shapes: sliding layers (`head_dim=256`, 32 Q / 16 KV heads) and full attention layers (`head_dim=512`, 32 Q / 4 KV heads), each divided by the tensor parallel size. Both the no-bias and the with-bias paths are covered. `rope_dim == head_dim` throughout, because `Gemma4RotaryEmbedding` passes `rotary_dim = head_size` to the base class and therefore builds a full-width cos/sin cache even on proportional-RoPE layers.
- Precision tolerance follows the operator-type/data-type convention: bf16 normalization plus rotation uses `atol=5e-2, rtol=5e-3`, matching `test_split_qkv_rmsnorm_rope.py`.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_rope_vnorm.py
```
