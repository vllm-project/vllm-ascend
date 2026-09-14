# _triton_rope_siso

Source: `vllm_ascend/ops/triton/rope.py` (host wrapper: `rope_forward_triton_siso`).

## Description

- **Function**: Single-input single-output (SISO) variant of [`_triton_rope`](./rope.md): applies rotary position embedding (RoPE) **in place** to one `[num_tokens, num_heads, head_dim]` tensor instead of a Q/K pair. It supports partial rotary (`rope_dim != head_dim`), NeoX and GPT-J (interleaved) styles, and either a pre-selected `cos`/`sin` pair or `cos_sin_cache` + `positions`. It is used by the SFA (sparse flash attention) path in `vllm_ascend/attention/sfa_v1.py`, where the lightweight Q and K projections are rotated one tensor at a time.
- **Formula**: For each token row `m`, each head, and the first `rope_dim` elements of the head (the remaining `head_dim - rope_dim` elements are left untouched):

    ```text
    if IS_NEOX_STYLE:  x1 = x[..., : rope_dim // 2],  x2 = x[..., rope_dim // 2 : rope_dim]
    else:              x1 = x[..., 0 : rope_dim : 2], x2 = x[..., 1 : rope_dim : 2]

    o1 = x1 * cos(m) - x2 * sin(m)
    o2 = x2 * cos(m) + x1 * sin(m)
    ```

    with `o1` written back to the position of `x1` and `o2` to the position of `x2`; `cos(m)`, `sin(m)` are the `rope_dim / 2` rotation coefficients of position `m`, loaded in fp32.
- **Algorithm flow** (processed row by row, independently):
  1. Host side (`rope_forward_triton_siso`): use `qk` directly when it is contiguous, otherwise create a contiguous copy, then read `num_tokens, n_head, head_dim`, assert `rope_dim <= head_dim`, and pad both the head count and the rotary dimension to powers of two (`pad_n_head = next_power_of_2(n_head)`, `pad_rope_dim = next_power_of_2(rope_dim)`). The grid is a persistent `n_row = min(num_tokens, get_vectorcore_num())` programs.
  2. Host side: select the rotation-table mode — `cos_sin_cache` + `positions` (`USE_COS_SIN=True`) or a pre-selected `cos`/`sin` pair (`USE_COS_SIN=False`, with `rope_dim` inferred as `cos.shape[-1] * 2` when `rope_dim == -1`). Passing neither raises `ValueError`.
  3. Kernel side: each program strides over the token rows with `for row_idx in tl.range(pid, num_tokens, num_programs)`, so a fixed grid covers any `num_tokens`.
  4. Kernel side: load the rotation coefficients of the row in fp32. With `USE_COS_SIN=True`, `pos_idx = positions[row_idx]` indexes `cos_sin_cache`, whose row holds `[cos(0 : rope_dim // 2), sin(rope_dim // 2 : rope_dim)]`; otherwise `cos`/`sin` are indexed by `row_idx`. `cos_mask = arange(0, pad_rope_dim // 2) < rope_dim // 2` masks the padding lanes.
  5. Kernel side: load the two rotation halves for **all** heads of the row at once as a `[pad_n_h, pad_rope_dim // 2]` tile. The offsets differ per style: NeoX uses `head * hd + i` and `head * hd + i + rope_dim // 2`; GPT-J uses `head * hd + 2 * i` and `head * hd + 2 * i + 1`. The mask `(head < n_h) & (i < rope_dim // 2)` disables the padded heads and lanes.
  6. Kernel side: compute `x1 * cos - x2 * sin` and `x2 * cos + x1 * sin`, store both tiles back to their own offsets, and advance to the next row.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950.

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `qk_ptr` | Input / Output (in place) | Tensor to rotate `[num_tokens, n_h, hd]`, contiguous; rotated in place and returned | fp16 / bf16 / fp32 | ND |
| `qk_row_stride` | Input (attribute) | Row stride of `qk` (`qk.stride(0)`, i.e. `n_h * hd`) | int32 | scalar |
| `cos_ptr` | Input | Position-selected cosine table `[num_tokens, rope_dim // 2]`; `None` when `USE_COS_SIN=True` | fp16 / bf16 / fp32 | ND |
| `cos_row_stride` | Input (attribute) | Row stride of `cos`; `None` when `USE_COS_SIN=True` | int32 | scalar |
| `sin_ptr` | Input | Position-selected sine table `[num_tokens, rope_dim // 2]`; `None` when `USE_COS_SIN=True` | fp16 / bf16 / fp32 | ND |
| `sin_row_stride` | Input (attribute) | Row stride of `sin`; `None` when `USE_COS_SIN=True` | int32 | scalar |
| `cos_sin_ptr` | Input | Raw rotation cache `[max_position_embeddings, rope_dim]`, each row being `[cos, sin]`; `None` when `USE_COS_SIN=False` | fp16 / bf16 / fp32 | ND |
| `cos_sin_row_stride` | Input (attribute) | Row stride of `cos_sin_cache` (`rope_dim`); `None` when `USE_COS_SIN=False` | int32 | scalar |
| `pos_ptr` | Input | Token positions `[num_tokens]` used to index `cos_sin_cache`; `None` when `USE_COS_SIN=False` | int64 | ND |
| `num_tokens` | Input (attribute) | Number of token rows to process | int32 | scalar |
| `n_h` | Input (attribute) | Number of heads of `qk`, compile-time `constexpr` | int32 | scalar |
| `hd` | Input (attribute) | Head dimension `head_size`, compile-time `constexpr` | int32 | scalar |
| `rope_dim` | Input (attribute) | Rotary dimension, compile-time `constexpr` | int32 | scalar |
| `pad_n_h` | Input (attribute) | `next_power_of_2(n_h)`, the head-tile size, compile-time `constexpr` | int32 | scalar |
| `pad_rope_dim` | Input (attribute) | `next_power_of_2(rope_dim)`, compile-time `constexpr` | int32 | scalar |
| `BLOCK_SIZE` | Input (attribute) | Reserved tile-size attribute, set to `pad_n_head` by the wrapper; the kernel currently tiles by `pad_n_h` and does not read it | int32 | scalar |
| `IS_NEOX_STYLE` | Input (attribute) | `True` for half-split (NeoX) rotation, `False` for interleaved (GPT-J) rotation, compile-time `constexpr` | bool | scalar |
| `USE_COS_SIN` | Input (attribute) | `True` to read `cos_sin_ptr` + `pos_ptr`, `False` to read `cos_ptr`/`sin_ptr`, compile-time `constexpr` | bool | scalar |

## Constraints

- `qk` must be 3-D `[num_tokens, n_head, head_dim]`. An already-contiguous input is modified in place. For a non-contiguous input, the wrapper creates and rotates a contiguous copy instead, so the caller's original view is unchanged and the returned tensor must be used.
- `rope_dim <= head_dim` (asserted by the wrapper) and `rope_dim` must be even; when `rope_dim < head_dim` the trailing `head_dim - rope_dim` elements of every head are passed through unchanged.
- Exactly one rotation-table mode must be supplied: `cos_sin_cache` together with `positions` (then `positions.shape[0] == num_tokens`), or `cos` and `sin` together (then `cos.shape[0] == sin.shape[0] == num_tokens`); otherwise the wrapper raises `ValueError`.
- With `cos`/`sin`, each row holds `rope_dim // 2` coefficients (they must not be duplicated to `rope_dim`). Unlike `rope_forward_triton`, `rope_forward_triton_siso` must be called with an explicit `rope_dim`: it computes `pad_rope_dim` before the `rope_dim == -1` inference, so the `-1` default would propagate an invalid padded dimension to the kernel. With `cos_sin_cache`, `rope_dim` must additionally be a power of two because the kernel starts the sine load at `pad_rope_dim // 2`, not at `rope_dim // 2`.
- `positions` values must be within `[0, cos_sin_cache.shape[0])`; they are loaded as `int64`.
- All heads of a row are processed in a single `[pad_n_h, pad_rope_dim // 2]` tile — unlike `_triton_rope`, there is no head tiling — so `n_head * rope_dim` (rounded up to powers of two) must fit into UB. The operator therefore targets small head counts, such as the single-head lightweight K/Q of the SFA path (`n_head = 1`, `head_dim = 128`); a large `n_head` combined with a large `head_dim` can overflow UB.
- Rotation is computed in fp32 and cast back to the input dtype on store.
- `num_tokens` is dynamic (the grid is capped by the vector-core count and the kernel strides over rows), so the kernel is graph-mode friendly: the launch grid does not depend on runtime tensor values, and no host synchronization occurs inside.

## Origin and Differences

- **Origin**: Developed for vllm-ascend, derived from `_triton_rope` in the same file (which implements the vLLM `rotary_embedding` custom op); it replaces the `torch.split` + `torch_npu.npu_rotary_mul` + `torch.cat` chain in the SFA lightweight-index path of `vllm_ascend/attention/sfa_v1.py`.
- **Differences**:
    - NPU adaptation for performance: rotates a single tensor in place with a persistent grid of `min(num_tokens, get_vectorcore_num())` programs and an inner row-stride loop; the whole row (all heads) is handled in one tile, which removes the head-loop overhead for the small-head-count SFA case, and the in-place update removes the split/concat copies of the `npu_rotary_mul` path;
    - Modified for a specific vllm-ascend logic or different input parameters: single input / single output instead of the Q+K pair, so callers that rotate Q and K in separate steps (SFA `q_li` / `k_li`) do not have to build a dummy tensor; keeps the same `cos_sin_cache` + `positions` and pre-selected `cos`/`sin` dual interface, the partial-rotary pass-through, and the NeoX / GPT-J switch as `_triton_rope`.

## Test Cases

> [!NOTE]
> Single-operator accuracy test cases are placed under `tests/e2e/nightly/single_node/ops/singlecard_ops/triton`.

`test_rotary_embedding_triton_kernel_siso` in `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rope.py` compares the kernel against a PyTorch-native reference over `head_size` of `64` and `128` (the SFA lightweight head dimension), `rotary_dim` of `32` and `64` (partial rotary, matching `qk_rope_head_dim`), `num_heads = 64`, `num_tokens` of `1, 4, 8, 16, 1024` (decode through prefill), both `is_neox_style` values, and bf16/fp16. Accuracy comparison uses the unified tolerance for this element-wise fp32-accumulating operator: `atol=rtol=1e-3`.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rope.py -k "siso"
```
