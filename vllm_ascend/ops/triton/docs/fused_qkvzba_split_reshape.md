# fused_qkvzba_split_reshape_cat

## Description

- **Function**: Splits and re-arranges the fused QKVZ and BA projections of the GDN (Gated Delta Network) linear attention into type-contiguous layouts. It replaces the host-side `split` + `reshape`/`cat` chain in the `gqa_interleaved_layout=True` path with a single fused Triton kernel.
- **Formula**: Pure data movement (load/store only, no arithmetic):
    - Input `mixed_qkvz`: `[num_tokens, num_heads_qk * (Q + K + V + Z)]`, where each head block is `[Q(head_qk), K(head_qk), V(v_heads_per_qk * head_v), Z(v_heads_per_qk * head_v)]` and `v_heads_per_qk = num_heads_v // num_heads_qk`
    - Input `mixed_ba`: `[num_tokens, num_heads_qk * (B + A)]`, where each head block is `[B(v_heads_per_qk), A(v_heads_per_qk)]`
    - Output `mixed_qkv`: `[num_tokens, Q_all | K_all | V_all]` (concatenated by type, `Q_all = K_all = num_heads_qk * head_qk`, `V_all = num_heads_v * head_v`)
    - Output `z`: `[num_tokens, num_heads_v, head_v]`
    - Output `b`, `a`: `[num_tokens, num_heads_v]`
- **Algorithm flow** (processed row by row, independently):
  1. Compute grid: `grid_size = min(num_vectorcore, total_rows)` vector cores, each processing `rows_per_vec = ceil(total_rows / grid_size)` rows.
  2. Tile rows: `rows_per_iter` rows per tile, derived from the UB (unified buffer) budget (`ub_size`, `elements_per_row`) and capped by `MAX_ROWS_PER_ITER`, to keep each load/store block within the UB.
  3. For each row tile, iterate over `NUM_HEADS_QK` head groups with `tl.static_range`:
     - Load Q/K/V/Z from the interleaved `mixed_qkvz` head block.
     - Load B/A from the interleaved `mixed_ba` head block.
     - Store Q/K/V into the type-contiguous `mixed_qkv` layout, Z into `z`, and B/A into `b`/`a`, each using its own row stride; out-of-range rows are masked.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950 (Triton kernel), used by the GDN linear attention `gqa_interleaved_layout=True` path (e.g. Qwen3-Next) in `vllm_ascend/ops/gdn.py`; works in both eager and graph-capture modes.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `mixed_qkvz` | Input | Fused QKVZ projection in interleaved GQA layout `[num_tokens, num_heads_qk * (Q + K + V + Z)]` | fp32 / fp16 / bf16 | ND |
| `mixed_ba` | Input | Fused BA projection in interleaved GQA layout `[num_tokens, num_heads_qk * (B + A)]` | fp32 / fp16 / bf16 | ND |
| `num_heads_qk` | Input (attribute) | Number of Q/K head groups per TP rank (`cdiv(num_k_heads, tp_size)`) | int32 | scalar |
| `num_heads_v` | Input (attribute) | Number of V/Z heads per TP rank (`cdiv(num_v_heads, tp_size)`) | int32 | scalar |
| `head_qk` | Input (attribute) | Head dimension of Q/K (`head_k_dim`) | int32 | scalar |
| `head_v` | Input (attribute) | Head dimension of V/Z (`head_v_dim`) | int32 | scalar |
| `mixed_qkv` | Output | QKV concatenated by type `[num_tokens, num_heads_qk * head_qk * 2 + num_heads_v * head_v]` | same as `mixed_qkvz` | ND |
| `z` | Output | Z state `[num_tokens, num_heads_v, head_v]` | same as `mixed_qkvz` | ND |
| `b` | Output | B decay `[num_tokens, num_heads_v]` | same as `mixed_ba` | ND |
| `a` | Output | A decay `[num_tokens, num_heads_v]` | same as `mixed_ba` | ND |

## Constraints

- `num_heads_v` must be divisible by `num_heads_qk` (`v_heads_per_qk = num_heads_v // num_heads_qk >= 1`).
- `mixed_qkvz.shape[1]` must equal `num_heads_qk * (head_qk * 2 + v_heads_per_qk * head_v * 2)`.
- `mixed_ba.shape[1]` must equal `num_heads_qk * v_heads_per_qk * 2`.
- Output dtypes are inherited from the inputs (`mixed_qkv`/`z` from `mixed_qkvz`; `b`/`a` from `mixed_ba`).
- `NUM_HEADS_QK`, `NUM_HEADS_V`, `HEAD_QK`, `HEAD_V`, and all row strides are compile-time `constexpr`; the row loop and row masks handle arbitrary `total_rows`, so dynamic token counts are supported.
- Only for inference (decoding/prefill) on NPU; `num_tokens` is flattened from `batch * seq_len`.

## Origin and Differences

- **Origin**: Based on the fused split/reshape logic of the flash-linear-attention project (MIT license, see header of the source file), rewritten as an Ascend NPU Triton kernel. It replaces the torch `split` + `reshape`/`cat` chain in the GDN attention `gqa_interleaved_layout=True` path of `vllm_ascend/ops/gdn.py`.
- **Differences**:
    - NPU adaptation for performance: fuses the split + reshape/cat into a single Triton kernel to avoid multiple device-side memory copies; parallelized over vector cores with UB-aware row tiling (`rows_per_iter`) to improve AICore utilization;
    - Modified for a specific vllm-ascend logic or different input parameters: consumes the interleaved GQA layout `[seq, num_heads_qk, (Q, K, V, Z)]` / `[seq, num_heads_qk, (B, A)]` produced by the `gqa_interleaved_layout` linear projections, and outputs the type-contiguous `mixed_qkv` expected by the `qwen_gdn_attention_core` custom op.

## Test Cases

The test covers both real inference shapes (Qwen3-GDN-10B, `gqa_interleaved_layout=True`, TP=1: `num_heads_qk=16`, `num_heads_v=128`, `head_qk=512`, `head_v=512`) and a broader generic parameter space. As this is a pure data-movement operator (no arithmetic), the unified precision tolerance is bit-exact (`rtol=0, atol=0`).

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_qkvzba_split_reshape_cat.py
```
