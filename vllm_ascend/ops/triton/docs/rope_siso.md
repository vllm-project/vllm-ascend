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
  2. Host side: select the rotation-table mode 鈥?`cos_sin_cache` + `positions` (`USE_COS_SIN=True`) or a pre-selected `cos`/`sin` pair (`USE_COS_SIN=False`, with `rope_dim` inferred as `cos.shape[-1] * 2` when `rope_dim == -1`). Passing neither raises `ValueError`.
  3. Host side: the kernel takes two pointers, `qk_ptr` (input) and `output_ptr` (output). The wrapper passes the **same tensor address** to both, making the operation in-place; the split allows callers to reuse one buffer for read and write explicitly, letting the compiler reason about pointer aliasing and avoid extra buffer copies.
  4. Kernel side: offsets and masks are computed **once outside the token loop** and reused across all rows 鈥?`qk_offsets`, `qk_mask` for the full `[pad_n_h, pad_rope_dim]` tile and `cos_sin_offsets` / `cos_sin_mask` for the rotation coefficients.
  5. Kernel side: each program strides over the token rows with `for row_idx in tl.range(pid, num_tokens, num_programs)`, so a fixed grid covers any `num_tokens`.
  6. Kernel side: load the rotation coefficients of the row in fp32. With `USE_COS_SIN=True`, `pos_idx = positions[row_idx]` indexes `cos_sin_cache`, whose row holds `[cos(0 : rope_dim // 2), sin(rope_dim // 2 : rope_dim)]`; the whole padded row is loaded once and split into `cos_row` / `sin_row` with `extension.extract_slice`. Otherwise `cos`/`sin` are indexed by `row_idx`. The masks disable the padding lanes.
  7. Kernel side: load **all** heads of the row at once as a `[pad_n_h, pad_rope_dim]` tile with a single `tl.load`, then split it with `extension.extract_slice` into the two rotation halves. The slice strides differ per style: NeoX uses contiguous halves (stride `[1, 1]`, offsets `[0, 0]` and `[0, rope_dim // 2]`); GPT-J uses even/odd lanes (stride `[1, 2]`, offsets `[0, 0]` and `[0, 1]`). The mask `(head < n_h) & (dim < rope_dim)` disables the padded heads and lanes.
  8. Kernel side: compute `x1 * cos - x2 * sin` and `x2 * cos + x1 * sin`, recombine the two halves into the full tile with `extension.insert_slice`, write it back to `output_ptr` with a single `tl.store`, and advance to the next row. This reduces global memory traffic from 4 loads + 2 stores per row (old implementation) to 2 loads + 1 store.
- **Supported modes**: Atlas A2, Atlas A3, and 950PR&950DT Products.

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `qk_ptr` | Input | Tensor to rotate `[num_tokens, n_h, hd]`, contiguous; read from here | fp16 / bf16 / fp32 | ND |
| `qk_row_stride` | Input (attribute) | Row stride of `qk` (`qk.stride(0)`, i.e. `n_h * hd`) | int32 | scalar |
| `cos_ptr` | Input | Position-selected cosine table `[num_tokens, rope_dim // 2]`; `None` when `USE_COS_SIN=True` | fp16 / bf16 / fp32 | ND |
| `cos_row_stride` | Input (attribute) | Row stride of `cos`; `None` when `USE_COS_SIN=True` | int32 | scalar |
| `sin_ptr` | Input | Position-selected sine table `[num_tokens, rope_dim // 2]`; `None` when `USE_COS_SIN=True` | fp16 / bf16 / fp32 | ND |
| `sin_row_stride` | Input (attribute) | Row stride of `sin`; `None` when `USE_COS_SIN=True` | int32 | scalar |
| `cos_sin_ptr` | Input | Raw rotation cache `[max_position_embeddings, rope_dim]`, each row being `[cos, sin]`; `None` when `USE_COS_SIN=False` | fp16 / bf16 / fp32 | ND |
| `cos_sin_row_stride` | Input (attribute) | Row stride of `cos_sin_cache` (`rope_dim`); `None` when `USE_COS_SIN=False` | int32 | scalar |
| `pos_ptr` | Input | Token positions `[num_tokens]` used to index `cos_sin_cache`; `None` when `USE_COS_SIN=False` | int64 | ND |
| `output_ptr` | Output | Rotated tensor `[num_tokens, n_h, hd]`; the wrapper passes the **same tensor as `qk_ptr`** for in-place operation, but callers may pass a distinct buffer for out-of-place output | fp16 / bf16 / fp32 | ND |
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

- `qk` must be 3-D `[num_tokens, n_head, head_dim]`. The wrapper passes the same tensor as both `qk_ptr` and `output_ptr`, so a contiguous input is modified in place and returned. For a non-contiguous input, the wrapper creates and rotates a contiguous copy instead, so the caller's original view is unchanged and the returned tensor must be used.
- `rope_dim <= head_dim` (asserted by the wrapper) and `rope_dim` must be even; when `rope_dim < head_dim` the trailing `head_dim - rope_dim` elements of every head are passed through unchanged.
- Exactly one rotation-table mode must be supplied: `cos_sin_cache` together with `positions` (then `positions.shape[0] == num_tokens`), or `cos` and `sin` together (then `cos.shape[0] == sin.shape[0] == num_tokens`); otherwise the wrapper raises `ValueError`.
- With `cos`/`sin`, each row holds `rope_dim // 2` coefficients (they must not be duplicated to `rope_dim`). Unlike `rope_forward_triton`, `rope_forward_triton_siso` must be called with an explicit `rope_dim`: it computes `pad_rope_dim` before the `rope_dim == -1` inference, so the `-1` default would propagate an invalid padded dimension to the kernel. With `cos_sin_cache`, `rope_dim` must additionally be a power of two because the kernel loads a padded row of `pad_rope_dim` and slices the sine half from `pad_rope_dim // 2`, not from `rope_dim // 2`.
- `positions` values must be within `[0, cos_sin_cache.shape[0])`; they are loaded as `int64`.
- All heads of a row are processed in a single `[pad_n_h, pad_rope_dim]` tile 鈥?unlike `_triton_rope`, there is no head tiling 鈥?so `n_head * rope_dim` (rounded up to powers of two) must fit into UB. Note that the optimized single-tile implementation loads the full `pad_rope_dim` (not `pad_rope_dim // 2`) width, which roughly doubles the per-row UB footprint versus the previous two-half implementation; the operator therefore targets small head counts, such as the single-head lightweight K/Q of the SFA path (`n_head = 1`, `head_dim = 128`); a large `n_head` combined with a large `head_dim` can overflow UB.
- Rotation is computed in fp32 and cast back to the input dtype on store.
- `num_tokens` is dynamic (the grid is capped by the vector-core count and the kernel strides over rows), so the kernel is graph-mode friendly: the launch grid does not depend on runtime tensor values, and no host synchronization occurs inside.

## Origin and Differences

- **Origin**: Developed for vllm-ascend, derived from `_triton_rope` in the same file (which implements the vLLM `rotary_embedding` custom op); it replaces the `torch.split` + `torch_npu.npu_rotary_mul` + `torch.cat` chain in the SFA lightweight-index path of `vllm_ascend/attention/sfa_v1.py`.
- **Differences**:
    - NPU adaptation for performance: rotates a single tensor with a persistent grid of `min(num_tokens, get_vectorcore_num())` programs and an inner row-stride loop; the whole row (all heads) is handled in one tile, which removes the head-loop overhead for the small-head-count SFA case, and the in-place update removes the split/concat copies of the `npu_rotary_mul` path;
    - Memory-traffic optimization: offsets/masks are hoisted out of the token loop, and each row is now handled with a **single load + single store** using `extension.extract_slice` / `extension.insert_slice` (previously two loads + two stores), reducing global memory traffic per row from 6 ops to 3 ops; input/output pointers are split so the wrapper can pass the same tensor address to both and enable explicit in-place reuse;
    - Modified for a specific vllm-ascend logic or different input parameters: single input / single output instead of the Q+K pair, so callers that rotate Q and K in separate steps (SFA `q_li` / `k_li`) do not have to build a dummy tensor; keeps the same `cos_sin_cache` + `positions` and pre-selected `cos`/`sin` dual interface, the partial-rotary pass-through, and the NeoX / GPT-J switch as `_triton_rope`.

## Test Cases

> [!NOTE]
> Single-operator accuracy test cases are placed under `tests/e2e/nightly/single_node/ops/singlecard_ops/triton`.

`test_rotary_embedding_triton_kernel_siso` in `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rope.py` compares the kernel against a PyTorch-native reference over `head_size` of `64` and `128` (the SFA lightweight head dimension), `rotary_dim` of `32` and `64` (partial rotary, matching `qk_rope_head_dim`), `num_heads = 64`, `num_tokens` of `1, 4, 8, 16, 1024` (decode through prefill), both `is_neox_style` values, and bf16/fp16. Accuracy comparison uses the unified tolerance for this element-wise fp32-accumulating operator: `atol=rtol=1e-3`.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rope.py -k "siso"
```
