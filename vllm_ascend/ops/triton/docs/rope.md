# _triton_rope

Source: `vllm_ascend/ops/triton/rope.py` (host wrapper: `rope_forward_triton`).

## Description

- **Function**: Applies rotary position embedding (RoPE) to the query and key tensors **in place**, in a single kernel. It supports partial rotary (`rope_dim != head_dim`), both NeoX-style and GPT-J (interleaved) style, and two ways of supplying the rotation table: a pre-selected `cos`/`sin` pair, or the raw `cos_sin_cache` plus `positions`. It backs `rope_forward_oot` in `vllm_ascend/ops/rotary_embedding.py` (the `torch_npu.npu_mrope` replacement used when Triton is available).
- **Formula**: For each token row `m`, each head, and the first `rope_dim` elements of the head (the remaining `head_dim - rope_dim` elements are left untouched):

    ```text
    if IS_NEOX_STYLE:  x1 = x[..., : rope_dim // 2],  x2 = x[..., rope_dim // 2 : rope_dim]
    else:              x1 = x[..., 0 : rope_dim : 2], x2 = x[..., 1 : rope_dim : 2]

    o1 = x1 * cos(m) - x2 * sin(m)
    o2 = x2 * cos(m) + x1 * sin(m)

    if IS_NEOX_STYLE:  out = cat((o1, o2), dim=-1)
    else:              out = stack((o1, o2), dim=-1).flatten(-2)
    ```

    where `cos(m)`, `sin(m)` are the `rope_dim / 2` rotation coefficients of position `m`, loaded in fp32.
- **Algorithm flow** (processed row by row, independently):
  1. Host side (`rope_forward_triton`): use `q`/`k` directly when they are contiguous, otherwise create contiguous copies, then read `num_tokens, n_q_head, head_dim` from `q` and `n_kv_head` from `k`, pick the head tile `BLOCK_SIZE_HEAD` (`64` for NeoX style, `32` otherwise, further clamped to `16` when `head_dim >= 256` to avoid UB overflow on A2/A3), pad the rotary dimension with `pad_rope_dim = next_power_of_2(rope_dim)`, and launch a persistent grid of `n_row = min(num_tokens, get_vectorcore_num())` programs.
  2. Host side: select the rotation-table mode — `cos_sin_cache` + `positions` (`USE_COS_SIN=True`) or an already position-selected `cos`/`sin` pair (`USE_COS_SIN=False`, with `rope_dim` inferred as `cos.shape[-1] * 2` when `rope_dim == -1`). Passing neither raises `ValueError`.
  3. Kernel side: each program strides over the token rows with `for row_idx in tl.range(pid, num_tokens, num_programs)`, so any `num_tokens` is covered by a fixed grid.
  4. Kernel side: load the rotation coefficients for the row and cast them to fp32. With `USE_COS_SIN=True`, `pos_idx = positions[row_idx]` indexes `cos_sin_cache`, whose row holds `[cos(0 : rope_dim // 2), sin(rope_dim // 2 : rope_dim)]`; otherwise `cos`/`sin` are indexed by `row_idx` directly. `cos_mask = arange(0, pad_rope_dim // 2) < rope_dim // 2` masks the padding.
  5. Kernel side: tile the Q heads in chunks of `BLOCK_SIZE_HEAD`. For NeoX style, load the two halves `[BLOCK_SIZE_HEAD, pad_rope_dim // 2]` at offsets `0` and `rope_dim // 2`, rotate, and store both halves back. For GPT-J style, load the `[BLOCK_SIZE_HEAD, pad_rope_dim // 2, 2]` pairs with `tl.split`, rotate, re-interleave with `tl.join`, and store. Head and rotary masks keep the tail heads and the padded rotary lanes inactive.
  6. Kernel side: repeat the same tiled loop for the K heads (`n_kh`), then advance to the next row.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950.

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `q_ptr` | Input / Output (in place) | Query tensor `[num_tokens, n_qh, hd]`, contiguous; rotated in place | fp16 / bf16 / fp32 | ND |
| `q_row_stride` | Input (attribute) | Row stride of `q` (`q.stride(0)`, i.e. `n_qh * hd`) | int32 | scalar |
| `k_ptr` | Input / Output (in place) | Key tensor `[num_tokens, n_kh, hd]`, contiguous; rotated in place | same as `q_ptr` | ND |
| `k_row_stride` | Input (attribute) | Row stride of `k` (`k.stride(0)`, i.e. `n_kh * hd`) | int32 | scalar |
| `cos_ptr` | Input | Position-selected cosine table `[num_tokens, rope_dim // 2]`; `None` when `USE_COS_SIN=True` | fp16 / bf16 / fp32 | ND |
| `cos_row_stride` | Input (attribute) | Row stride of `cos`; `None` when `USE_COS_SIN=True` | int32 | scalar |
| `sin_ptr` | Input | Position-selected sine table `[num_tokens, rope_dim // 2]`; `None` when `USE_COS_SIN=True` | fp16 / bf16 / fp32 | ND |
| `sin_row_stride` | Input (attribute) | Row stride of `sin`; `None` when `USE_COS_SIN=True` | int32 | scalar |
| `cos_sin_ptr` | Input | Raw rotation cache `[max_position_embeddings, rope_dim]`, each row being `[cos, sin]`; `None` when `USE_COS_SIN=False` | fp16 / bf16 / fp32 | ND |
| `cos_sin_row_stride` | Input (attribute) | Row stride of `cos_sin_cache` (`rope_dim`); `None` when `USE_COS_SIN=False` | int32 | scalar |
| `pos_ptr` | Input | Token positions `[num_tokens]` used to index `cos_sin_cache`; `None` when `USE_COS_SIN=False` | int64 | ND |
| `num_tokens` | Input (attribute) | Number of token rows to process | int32 | scalar |
| `n_qh` | Input (attribute) | Number of query heads, compile-time `constexpr` | int32 | scalar |
| `n_kh` | Input (attribute) | Number of key/value heads, compile-time `constexpr` | int32 | scalar |
| `hd` | Input (attribute) | Head dimension `head_size`, compile-time `constexpr` | int32 | scalar |
| `rope_dim` | Input (attribute) | Rotary dimension (`rotary_dim`), compile-time `constexpr` | int32 | scalar |
| `pad_rope_dim` | Input (attribute) | `next_power_of_2(rope_dim)`, compile-time `constexpr` | int32 | scalar |
| `BLOCK_SIZE_HEAD` | Input (attribute) | Number of heads per tile, compile-time `constexpr` (`64` NeoX / `32` GPT-J, clamped to `16` when `hd >= 256`) | int32 | scalar |
| `IS_NEOX_STYLE` | Input (attribute) | `True` for half-split (NeoX) rotation, `False` for interleaved (GPT-J) rotation, compile-time `constexpr` | bool | scalar |
| `USE_COS_SIN` | Input (attribute) | `True` to read `cos_sin_ptr` + `pos_ptr`, `False` to read `cos_ptr`/`sin_ptr`, compile-time `constexpr` | bool | scalar |

## Constraints

- `q` and `k` must be 3-D `[num_tokens, num_heads, head_dim]`. Already-contiguous inputs are modified in place. For a non-contiguous input, the wrapper creates and rotates a contiguous copy instead, so the caller's original view is unchanged and the returned tensor must be used.
- `rope_dim <= head_dim` (asserted by the wrapper) and `rope_dim` must be even; when `rope_dim < head_dim` the trailing `head_dim - rope_dim` elements of every head are passed through unchanged.
- Exactly one rotation-table mode must be supplied: `cos_sin_cache` together with `positions` (then `positions.shape[0] == num_tokens`), or `cos` and `sin` together (then `cos.shape[0] == sin.shape[0] == num_tokens`); otherwise the wrapper raises `ValueError`.
- With `cos`/`sin`, each row holds `rope_dim // 2` coefficients (they must not be duplicated to `rope_dim`); passing `rope_dim = -1` makes the wrapper infer `rope_dim = cos.shape[-1] * 2`. With `cos_sin_cache`, each row holds `rope_dim` values laid out as `[cos | sin]`, and `rope_dim` must be a power of two: the kernel starts the sine load at `pad_rope_dim // 2`, where `pad_rope_dim = next_power_of_2(rope_dim)`, rather than at `rope_dim // 2`.
- `positions` values must be within `[0, cos_sin_cache.shape[0])`; they are loaded as `int64`.
- The `[BLOCK_SIZE_HEAD, pad_rope_dim]` tile (twice that for the GPT-J pair layout) must fit into UB. This is what the `head_dim >= 256` clamp to `BLOCK_SIZE_HEAD = 16` guards against on Atlas A2/A3; very large `head_dim` combined with a large `BLOCK_SIZE_HEAD` would overflow UB.
- Rotation is computed in fp32 and cast back to the input dtype on store; `q` and `k` must share the same dtype and head dimension.
- `num_tokens` is dynamic (the grid is capped by the vector-core count and the kernel strides over rows), so the kernel is graph-mode friendly: the launch grid does not depend on runtime tensor values, and no host synchronization occurs inside.

## Origin and Differences

- **Origin**: Developed for vllm-ascend as the Triton implementation of the vLLM `RotaryEmbedding.forward` / `rotary_embedding` custom op; it replaces `torch_npu.npu_mrope` in `rope_forward_oot` when Triton is available.
- **Differences**:
    - NPU adaptation for performance: a persistent grid of `min(num_tokens, get_vectorcore_num())` programs with an inner row-stride loop instead of one program per token, plus head tiling via `BLOCK_SIZE_HEAD` (reduced to `16` for `head_dim >= 256`) to keep every load/store block inside UB; Q and K are rotated in the same kernel and written in place, avoiding the extra allocations of the native path;
    - Modified for a specific vllm-ascend logic or different input parameters: accepts both the `cos_sin_cache` + `positions` form used by `rope_forward_oot` and the pre-selected `cos`/`sin` form used by fused attention paths (`USE_COS_SIN`), supports `rope_dim != head_dim` partial rotary with the pass-through tail, and handles NeoX and GPT-J layouts in one kernel through `IS_NEOX_STYLE`.

## Test Cases

> [!NOTE]
> Single-operator accuracy test cases are placed under `tests/e2e/nightly/single_node/ops/singlecard_ops/triton`.

`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rope.py` covers both wrapper paths — `test_rotary_embedding_triton_kernel` (pre-selected `cos`/`sin`) and `test_rotary_embedding_triton_kernel_with_cos_sin_cache` (`cos_sin_cache` + `positions`, `max_position_embeddings=262144`) — against a PyTorch-native reference. Shapes follow the models served on NPU: `(head_size, rotary_dim)` of `(128, 128)` (full rotary, e.g. Qwen/Llama-class models) and `(64, 32)` (partial rotary), `(num_q_heads, num_k_heads)` of `(64, 1)` and `(96, 8)` (GQA/MQA), `num_tokens` of `1, 4, 8, 16, 1024` (decode through prefill), both `is_neox_style` values, and bf16/fp16. Accuracy comparison uses the unified tolerance for this element-wise fp32-accumulating operator: `atol=rtol=1e-3`.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rope.py -k "not siso"
```
