# split_qkv_index_rmsnorm_rope

## Description

- **Function**: Fused sparse-attention prepare stage for MiniMax-M3. In a single Triton kernel it splits the concatenated projection output, applies Gemma RMSNorm, applies NeoX-style RoPE to the main Q/K and the indexer Q/K, and optionally clamps (±448) and casts to FP8 E4M3. It replaces the original host-side chain of `narrow`/`split` × 5, `npu_rms_norm` × 4, `npu_rotary_embedding` × 2, and `clamp` + `cast` × 4, which wrote every intermediate back to global memory.
- **Formula**:
    - Input `input`: `[num_tokens, q_hidden_size + 2 * kv_hidden_size + index_q_size + idx_head_dim]`, laid out as `[q | k | v | index_q | index_k]`.
    - Split by column offset (`q_hidden_size`, `kv_hidden_size`, `index_q_size`, `idx_head_dim`):
        - `q`: `[num_tokens, q_head_num, head_dim]`
        - `k`: `[num_tokens, kv_head_num, head_dim]`
        - `v`: `[num_tokens, kv_head_num, head_dim]` (copied only, no norm / no RoPE)
        - `index_q`: `[num_tokens, index_q_head_num, idx_head_dim]`
        - `index_k`: `[num_tokens, 1, idx_head_dim]` (shared single head)
    - Gemma RMSNorm over the head dimension (`D = head_dim` for main Q/K, `D = idx_head_dim` for the indexer):
        - `rstd = rsqrt(sum(x^2) / D + eps)`
        - `y = x * rstd * (1 + w)`
    - NeoX RoPE over the rotary head (`R = min(rope_dim, D)`, `half = R // 2`):
        - `x1 = x[..., :half]`, `x2 = x[..., half:R]`
        - `rotated = [x1 * cos - x2 * sin, x2 * cos + x1 * sin]`
        - `y = [rotated, x[..., R:]]`
        - `cos`/`sin` are the per-position rows gathered from `cos_sin_cache` (`[max_position, rope_dim] = concat(cos, sin)`); `cos` is a row of `[0, rope_dim/2)` and `sin` is a row of `[rope_dim/2, rope_dim)`.
    - FP8 (optional): `y = clamp(y, -448, 448).to(float8_e4m3fn)` for `attn_out_fp8` (Q/K/V) and `indexer_out_fp8` (indexer Q/K) respectively.
- **Algorithm flow** (processed token by token, independently):
    1. Pre-gather cos/sin in the Python wrapper: `cos_sin_gathered = cos_sin_cache[positions]` (one `aclnnIndex`), producing `[num_tokens, rope_dim]`. The kernel loads cos from offset 0 and sin from `rope_dim / 2` via contiguous offset — this avoids the non-deterministic in-kernel scalar gather (`get_element` + `insert_slice`), which schedules unstably under concurrent memory traffic on Ascend Triton.
    2. Launch with `grid = (num_vectorcore,)`; each vector core owns a contiguous token range.
    3. Three loops share the four weights (loaded once into registers) and a UB-aware token tile (`_tokens_per_iter`, sized from the NPU UB: A2 = 192 KB, A5 = 248 KB):
        - Section 1 (main QK): load `[ q | k ]` by offset, Gemma RMSNorm (`tl.sum` reduction over `head_dim`), `×(1 + w)`, NeoX RoPE, clamp/cast if `attn_out_fp8`, store `q_out`/`k_out`.
        - Section 2 (V): load `[ v ]` by offset, clamp/cast if `attn_out_fp8`, store `v_out` (no norm, no RoPE).
        - Section 3 (indexer QK): load `[ index_q | index_k ]` by offset, Gemma RMSNorm (`tl.sum` reduction over `idx_head_dim`), `×(1 + w)`, NeoX RoPE (half width), clamp/cast if `indexer_out_fp8`, store `index_q_out`/`index_k_out`.
    4. All intermediates stay in UB/registers; global memory is touched only on input load and output store.
- **Supported modes**: Atlas A2 and Ascend 950 (Triton kernel), used by the MiniMax-M3 sparse-attention prepare path in `vllm_ascend/models/minimax_m3/minimax_m3.py` when `HAS_TRITON`, the input is bf16 on NPU, `positions` is 1-D, and the rotary embedding is NeoX style; works in both eager and graph-capture modes.

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `input` | Input | Fused QKV projection of MiniMax-M3, laid out as `[ q \| k \| v \| index_q \| index_k ]` | bf16 | ND |
| `cos_sin_cache` | Input | RoPE cache `[ max_position_embeddings, rope_dim ] = concat(cos, sin)` | bf16 | ND |
| `positions` | Input | Per-token positions used to index `cos_sin_cache` | int64 | ND (1-D) |
| `q_weight` | Input | Gemma RMSNorm weight `1 + w` for Q (`[head_dim]`) | fp32 | ND (1-D) |
| `k_weight` | Input | Gemma RMSNorm weight `1 + w` for K (`[head_dim]`) | fp32 | ND (1-D) |
| `index_q_weight` | Input | Gemma RMSNorm weight `1 + w` for index Q (`[idx_head_dim]`) | fp32 | ND (1-D) |
| `index_k_weight` | Input | Gemma RMSNorm weight `1 + w` for index K (`[idx_head_dim]`) | fp32 | ND (1-D) |
| `q_bias` | Input (optional) | Post-norm bias for Q (`[head_dim]`); unused when `None` | fp32 | ND (1-D) |
| `k_bias` | Input (optional) | Post-norm bias for K (`[head_dim]`); unused when `None` | fp32 | ND (1-D) |
| `q_hidden_size` | Input (attribute) | `q_head_num * head_dim` | int32 | scalar |
| `kv_hidden_size` | Input (attribute) | `kv_head_num * head_dim` | int32 | scalar |
| `index_q_size` | Input (attribute) | `index_q_head_num * idx_head_dim` | int32 | scalar |
| `head_dim` | Input (attribute) | Main head dimension; RMSNorm reduces and RoPE views by this | int32 | scalar |
| `idx_head_dim` | Input (attribute) | Indexer head dimension; RMSNorm reduces by this | int32 | scalar |
| `eps` | Input (attribute) | RMSNorm epsilon | fp32 | scalar |
| `attn_out_fp8` | Input (attribute) | Whether to clamp/cast main Q/K/V to E4M3 | bool | scalar |
| `indexer_out_fp8` | Input (attribute) | Whether to clamp/cast indexer Q/K to E4M3 | bool | scalar |
| `q_out` | Output | Normed + RoPE'd Q `[ num_tokens, q_hidden_size ]` | bf16 or fp8 | ND |
| `k_out` | Output | Normed + RoPE'd K `[ num_tokens, kv_hidden_size ]` | bf16 or fp8 | ND |
| `v_out` | Output | V copy `[ num_tokens, kv_hidden_size ]` | bf16 or fp8 | ND |
| `index_q_out` | Output | Normed + RoPE'd index Q `[ num_tokens, index_q_size ]` | bf16 or fp8 | ND |
| `index_k_out` | Output | Normed + RoPE'd index K `[ num_tokens, idx_head_dim ]` | bf16 or fp8 | ND |

## Constraints

- `input.dtype` must be `bfloat16`; the fused path is only taken for bf16 inputs.
- `positions` must be 1-D of `int64`, with every element `< max_position_embeddings` (the row count of `cos_sin_cache`).
- Output dtype is `bfloat16` unless the corresponding FP8 switch is on, in which case it is `float8_e4m3fn` (values pre-clamped to ±448).
- `q_hidden_size`, `kv_hidden_size`, `index_q_size`, `head_dim`, `idx_head_dim`, and `eps` are compile-time `constexpr`; `num_tokens` is dynamic (the token loop and masks handle arbitrary token counts).
- `rope_dim` is the last dim of `cos_sin_cache`; RoPE rotates only `min(rope_dim, head_dim)` (main) and `min(rope_dim, idx_head_dim)` (indexer) head elements, the rest of the head is copied unchanged.
- Gemma RMSNorm is applied to `q`, `k`, `index_q`, `index_k` only; `v` is copied without norm/RoPE.
- Only for NPU inference (prefill/decode); `num_tokens` is flattened from `batch * seq_len`.

## Origin and Differences

- **Origin**: Derived from the MiniMax-M3 `_sparse_prepare` fallback path in `vllm_ascend/models/minimax_m3/minimax_m3.py`, which ran `narrow`/`split` × 5, `npu_rms_norm` × 4, `npu_rotary_embedding` × 2, and `clamp`(+`cast`) × 4 as independent operators.
- **Differences**:
    - NPU adaptation for performance: fuses all steps into one Triton kernel so intermediates stay in UB/registers instead of round-tripping through global memory; UB-aware token tiling sized per NPU (A2 = 192 KB, A5 = 248 KB) and parallelized over vector cores.
    - Modified for a specific vllm-ascend logic or different input parameters: consumes the fused concat `[ q \| k \| v \| index_q \| index_k ]` produced by MiniMax-M3's single `qkv_proj`; cos/sin are pre-gathered in Python (`aclnnIndex`) instead of a scalar in-kernel gather, to keep the RoPE deterministic on Ascend Triton.

## Test Cases

- Single-operator accuracy test: `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_index_rmsnorm_rope.py`, parameterized over real MiniMax-M3 shapes (`num_q_heads=64`, `num_kv_heads=4`, `head_dim=128`; indexer `16 × idx_head_dim`) plus a broader generic space, on both FP8 and non-FP8 outputs.
- Precision tolerance follows the operator-type/data-type convention: non-FP8 outputs use `atol=5e-2, rtol=5e-3`; FP8 outputs use `atol=0.5, rtol=0.125` (FP8 E4M3 has 3 mantissa bits, one quantization step is 0.0625~0.25 for `|x| ∈ [1, 8)`).

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_index_rmsnorm_rope.py
```