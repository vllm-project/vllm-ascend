# RecoverSSM Convolution State Compaction

## Description

[`compact_conv_state`](../mamba/compact_conv_state.py#L141) copies the
accepted convolution history into the final and optional boundary pages
selected by the [commit planner](prepare_commit_plan.md). For accepted length
`c > 0` and retained history `W`, it copies source slots
`[c-1:c-1+W]` to destination slots `[0:W]`. The optional boundary copy
uses the boundary recovery length in place of `c`.

The grid is `(ceil(C / BLOCK_D), B, L)` for `C` channels, `B`
speculative rows and `L` layers. The wrapper selects a 4096-channel tile on
Ascend 950 or with a shared unit channel stride, and 256 otherwise.
Non-950 launches use the one-dimensional gather path for a known non-unit
channel stride. With `dim_stride=None`, the kernel loads each layer's
channel stride from the stride table.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `conv_state_ref` | Input/Output | Reference view used for channel count and element type. The kernel updates the caches addressed by the table | BF16 in GDN tests | Logical `[pages, C, slots]` |
| `conv_state_base_addrs` | Input | Device base address of each layer's cache | INT64 | Contiguous `[L]` |
| `conv_state_block_strides`, `conv_state_dim_strides`, `conv_state_token_strides` | Input | Per-layer page, channel and token strides in elements | INT64 | Contiguous `[L]` each |
| `state_indices` | Input | Source page per speculative row | INT32 in standalone tests | Strided `[B]` |
| `commit_lens`, `final_state_indices` | Input | Accepted length and final page per row | INT32 in GDN | Contiguous vectors |
| `boundary_state_indices`, `boundary_recovery_lens` | Input | Boundary page and its accepted prefix length | INT32 in GDN | Contiguous vectors |
| `conv_history_len` | Attribute | Number of history slots copied to each destination | Python int | Positive scalar |
| `align_mode` | Attribute | Enables the boundary copy for a valid boundary page | Python bool | Scalar |
| `dim_stride` | Attribute | Shared channel stride, or `None` for per-layer strides | Python int or `None` | Scalar |
| `null_block_id` | Attribute | Page IDs at or below this value are inactive | Python int | Scalar |

The wrapper returns `None`. The final and optional boundary histories are
written in place to the layer caches.

## Constraints

The caller supplies same-device metadata, live layer backing addresses with a
common element type and channel count, and sufficient capacity for all source
slices and destinations. Positive page IDs must be in range. Cross-request
read/write conflicts must be avoided. A row with a null source, null final
destination, or zero accepted length leaves the caches untouched. All plan
vectors remain required in non-align mode.

The GDN integration exercises this operator in eager and commit graph paths.
The standalone cases below run eagerly.

## Origin and Differences

The implementation was extracted from the GDN RecoverSSM Mamba helper. It
compacts history already prepared by the caller's convolution path and
uses explicit per-layer page and inner strides.

## Test Cases

[`test_compact_conv_state_npu.py`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compact_conv_state_npu.py) compares full
backing storage against a PyTorch slice-copy reference with zero tolerance.
The four cases cover 127 channels and Qwen-derived channel counts
2048/2560/5120, both physical orientations, align on/off, out-of-place and
in-place copies, null and zero-acceptance rows, and untouched page padding.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compact_conv_state_npu.py
```

## Example

For `W=3` and `c=4`, copy source slots `[3:6]` into the final page's
`[0:3]`. If the plan also supplies a valid boundary page at accepted length
two, copy `[1:4]` into that page's `[0:3]`.
