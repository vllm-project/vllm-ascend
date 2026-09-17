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

### FP8 E4M3 output variant

- **Function**: When `rope_forward_triton` is called with `out_dtype=torch.float8_e4m3fn`, it dispatches to `_triton_rope_fp8`. This variant applies NeoX-style RoPE to bf16 Q/K and writes separate fixed-scale E4M3 outputs, avoiding an intermediate rotated bf16 tensor in the MiniMax-M3 sparse index path. For partial rotary, the non-rotary tail is copied through the same FP8 conversion.
- **Formula**: For `half = rope_dim // 2`:

    ```text
    x1 = x[..., :half]
    x2 = x[..., half:rope_dim]
    r1 = x1 * cos(m) - x2 * sin(m)
    r2 = x2 * cos(m) + x1 * sin(m)

    quant_fp8(v) = cast_float8_e4m3fn(clamp(v, -448, 448))
    out[..., :rope_dim] = quant_fp8(cat((r1, r2), dim=-1))
    out[..., rope_dim:] = quant_fp8(x[..., rope_dim:])
    ```

    The rotation is evaluated in fp32 before the fixed-scale E4M3 conversion.
- **Algorithm flow** (processed row by row, independently):
  1. The Ascend `rope_forward_oot` dispatcher selects the Triton path for `out_dtype=torch.float8_e4m3fn`; `rope_forward_triton` then requires `cos_sin_cache`, `positions`, and an explicit `rope_dim`, and `_rope_forward_triton_fp8` makes Q/K contiguous and validates or allocates contiguous E4M3 output buffers.
  2. The host computes `pass_dim = head_dim - rope_dim`, pads the rotary half and pass-through tail independently to powers of two, and selects Q/K head tiles as `min(next_power_of_2(num_heads), 16)`.
  3. It launches `min(num_tokens, max(get_vectorcore_num() * 8, 256))` persistent programs; each program strides over token rows by the number of launched programs.
  4. For each row, the kernel indexes `cos_sin_cache` with `positions[row_idx]`, loads the cosine and sine halves, and converts them to fp32.
  5. It processes Q heads in `BLOCK_QH` tiles, rotates in fp32, clamps each result to `[-448, 448]`, and stores directly as E4M3. A non-rotary tail is clipped and converted in the same way.
  6. It repeats the operation for K using `BLOCK_KH`, then advances to the next token row.
- **Supported modes**: Ascend 950 (A5). Atlas A2 and Atlas A3 do not support this FP8 execution path.

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

### FP8 E4M3 output variant

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `q_ptr` | Input | Query tensor `[num_tokens, n_qh, hd]`, contiguous | bf16 | ND |
| `q_row_stride` | Input (attribute) | Row stride of `q` (`q.stride(0)`, normally `n_qh * hd`) | int32 | scalar |
| `k_ptr` | Input | Key tensor `[num_tokens, n_kh, hd]`, contiguous | bf16 | ND |
| `k_row_stride` | Input (attribute) | Row stride of `k` (`k.stride(0)`, normally `n_kh * hd`) | int32 | scalar |
| `q_out_ptr` | Output | Rotated and quantized query tensor `[num_tokens, n_qh, hd]`, contiguous | float8_e4m3fn | ND |
| `q_out_row_stride` | Input (attribute) | Row stride of `q_out` (`q_out.stride(0)`) | int32 | scalar |
| `k_out_ptr` | Output | Rotated and quantized key tensor `[num_tokens, n_kh, hd]`, contiguous | float8_e4m3fn | ND |
| `k_out_row_stride` | Input (attribute) | Row stride of `k_out` (`k_out.stride(0)`) | int32 | scalar |
| `cos_sin_ptr` | Input | Raw rotation cache `[max_position_embeddings, rope_dim]`; each row is `[cos, sin]` | bf16 | ND |
| `cos_sin_row_stride` | Input (attribute) | Row stride of `cos_sin_cache` | int32 | scalar |
| `pos_ptr` | Input | Token positions `[num_tokens]` used to index `cos_sin_cache` | int64 | ND |
| `num_tokens` | Input (attribute) | Number of token rows to process | int32 | scalar |
| `n_qh` | Input (attribute) | Number of query heads, compile-time `constexpr` | int32 | scalar |
| `n_kh` | Input (attribute) | Number of key heads, compile-time `constexpr` | int32 | scalar |
| `hd` | Input (attribute) | Head dimension, compile-time `constexpr` | int32 | scalar |
| `rope_dim` | Input (attribute) | Even rotary dimension no larger than `hd`, compile-time `constexpr` | int32 | scalar |
| `pad_half` | Input (attribute) | `next_power_of_2(rope_dim // 2)`, compile-time `constexpr` | int32 | scalar |
| `pad_pass` | Input (attribute) | `next_power_of_2(pass_dim)` for a non-empty tail, otherwise `1`; compile-time `constexpr` | int32 | scalar |
| `pass_dim` | Input (attribute) | Non-rotary tail width `hd - rope_dim`, compile-time `constexpr` | int32 | scalar |
| `BLOCK_QH` | Input (attribute) | Query-head tile `min(next_power_of_2(n_qh), 16)`, compile-time `constexpr` | int32 | scalar |
| `BLOCK_KH` | Input (attribute) | Key-head tile `min(next_power_of_2(n_kh), 16)`, compile-time `constexpr` | int32 | scalar |
| `FP8_MAX` | Input (attribute) | E4M3 finite-range clipping bound, fixed to `448.0`, compile-time `constexpr` | fp32 | scalar |

## Constraints

- `q` and `k` must be 3-D `[num_tokens, num_heads, head_dim]`. Already-contiguous inputs are modified in place. For a non-contiguous input, the wrapper creates and rotates a contiguous copy instead, so the caller's original view is unchanged and the returned tensor must be used.
- `rope_dim <= head_dim` (asserted by the wrapper) and `rope_dim` must be even; when `rope_dim < head_dim` the trailing `head_dim - rope_dim` elements of every head are passed through unchanged.
- Exactly one rotation-table mode must be supplied: `cos_sin_cache` together with `positions` (then `positions.shape[0] == num_tokens`), or `cos` and `sin` together (then `cos.shape[0] == sin.shape[0] == num_tokens`); otherwise the wrapper raises `ValueError`.
- With `cos`/`sin`, each row holds `rope_dim // 2` coefficients (they must not be duplicated to `rope_dim`); passing `rope_dim = -1` makes the wrapper infer `rope_dim = cos.shape[-1] * 2`. With `cos_sin_cache`, each row holds `rope_dim` values laid out as `[cos | sin]`, and `rope_dim` must be a power of two: the kernel starts the sine load at `pad_rope_dim // 2`, where `pad_rope_dim = next_power_of_2(rope_dim)`, rather than at `rope_dim // 2`.
- `positions` values must be within `[0, cos_sin_cache.shape[0])`; they are loaded as `int64`.
- The `[BLOCK_SIZE_HEAD, pad_rope_dim]` tile (twice that for the GPT-J pair layout) must fit into UB. This is what the `head_dim >= 256` clamp to `BLOCK_SIZE_HEAD = 16` guards against on Atlas A2/A3; very large `head_dim` combined with a large `BLOCK_SIZE_HEAD` would overflow UB.
- Rotation is computed in fp32 and cast back to the input dtype on store; `q` and `k` must share the same dtype and head dimension.
- `num_tokens` is dynamic (the grid is capped by the vector-core count and the kernel strides over rows), so the kernel is graph-mode friendly: the launch grid does not depend on runtime tensor values, and no host synchronization occurs inside.

### FP8 E4M3 output variant

- The FP8 path is selected only by calling `rope_forward_triton(..., out_dtype=torch.float8_e4m3fn)`. Other non-`None` output dtypes are rejected.
- Q and K must be 3-D tensors with the same `num_tokens` and `head_dim`. The validated model path and test coverage use bf16 inputs.
- Outputs are out of place. Caller-provided `q_out` and `k_out` must be contiguous `torch.float8_e4m3fn` tensors with shapes matching Q and K; otherwise the wrapper allocates them.
- Only NeoX-style half-split rotation is supported. Passing `is_neox_style=False` raises `NotImplementedError`.
- The FP8 path requires `cos_sin_cache` together with `positions`; the pre-selected `cos`/`sin` mode is not supported.
- `rope_dim` must be explicit, even, greater than zero, and no larger than `head_dim`. A non-rotary tail is still clipped and converted to E4M3.
- Each position must be in `[0, cos_sin_cache.shape[0])`. Each cache row contains `rope_dim // 2` cosine values followed by the same number of sine values.
- Conversion uses a fixed scale of `1.0`: values are clipped to `[-448, 448]` and cast directly to `torch.float8_e4m3fn`; no dynamic scale is calculated or returned.
- Q/K head tiles are capped at `16` to bound UB usage. Masks suppress padded heads and padded rotary or pass-through lanes.
- The launch grid depends only on tensor shapes and the vector-core count, and the kernel performs no data-dependent host synchronization, so it is compatible with the graph-capture path used by MiniMax-M3.

## Origin and Differences

- **Origin**: Developed for vllm-ascend as the Triton implementation of the vLLM `RotaryEmbedding.forward` / `rotary_embedding` custom op; it replaces `torch_npu.npu_mrope` in `rope_forward_oot` when Triton is available.
- **Differences**:
    - NPU adaptation for performance: a persistent grid of `min(num_tokens, get_vectorcore_num())` programs with an inner row-stride loop instead of one program per token, plus head tiling via `BLOCK_SIZE_HEAD` (reduced to `16` for `head_dim >= 256`) to keep every load/store block inside UB; Q and K are rotated in the same kernel and written in place, avoiding the extra allocations of the native path;
    - Modified for a specific vllm-ascend logic or different input parameters: accepts both the `cos_sin_cache` + `positions` form used by `rope_forward_oot` and the pre-selected `cos`/`sin` form used by fused attention paths (`USE_COS_SIN`), supports `rope_dim != head_dim` partial rotary with the pass-through tail, and handles NeoX and GPT-J layouts in one kernel through `IS_NEOX_STYLE`.

### FP8 E4M3 output variant

- **Origin**: Developed from `_triton_rope` to support the MiniMax-M3 sparse index path with a native E4M3 index cache.
- **Differences**:
    - NPU adaptation for performance: fuses RoPE and fixed-scale E4M3 conversion into the output stores, uses separate Q/K head tiles capped at `16`, and launches up to `max(get_vectorcore_num() * 8, 256)` programs to expose enough row parallelism on A5;
    - Modified for a specific vllm-ascend logic or different input parameters: writes separate FP8 outputs instead of updating bf16 Q/K in place, accepts only `cos_sin_cache` + `positions`, supports only NeoX layout, clips the rotated values and partial-RoPE tail to the E4M3 finite range, and uses a fixed quantization scale of `1.0`.

## Test Cases

> [!NOTE]
> Single-operator accuracy test cases are placed under `tests/e2e/nightly/single_node/ops/singlecard_ops/triton`.

`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rope.py` covers both wrapper paths — `test_rotary_embedding_triton_kernel` (pre-selected `cos`/`sin`) and `test_rotary_embedding_triton_kernel_with_cos_sin_cache` (`cos_sin_cache` + `positions`, `max_position_embeddings=262144`) — against a PyTorch-native reference. Shapes follow the models served on NPU: `(head_size, rotary_dim)` of `(128, 128)` (full rotary, e.g. Qwen/Llama-class models) and `(64, 32)` (partial rotary), `(num_q_heads, num_k_heads)` of `(64, 1)` and `(96, 8)` (GQA/MQA), `num_tokens` of `1, 4, 8, 16, 1024` (decode through prefill), both `is_neox_style` values, and bf16/fp16. Accuracy comparison uses the unified tolerance for this element-wise fp32-accumulating operator: `atol=rtol=1e-3`.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rope.py -k "not siso"
```

### FP8 E4M3 output variant

`test_rotary_embedding_triton_kernel_fp8` in the same test file compares `rope_forward_triton(..., out_dtype=torch.float8_e4m3fn)` against a PyTorch-native fp32-accumulating reference followed by clipping and E4M3 conversion. It covers `(num_tokens, num_q_heads, num_k_heads, head_size, rotary_dim)` values of `(1, 2, 1, 128, 128)`, `(17, 8, 1, 128, 64)`, and `(1024, 8, 1, 128, 128)`, spanning single-token decode, partial rotary, a non-power-of-two token count, and multi-row prefill. The position-zero case injects values outside the E4M3 finite range to verify saturation to `+448` and `-448`. Accuracy comparison uses `atol=rtol=0.125` after converting actual and reference E4M3 tensors to fp32.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rope.py -k "fp8"
```
