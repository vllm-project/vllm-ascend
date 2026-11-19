# Fused GDN projection preparation

Adapted from [vllm-ascend PR #21820](https://github.com/vllm-project/vllm-ascend/pull/21820), source head `407a3cd368fda50bc0ccab70ced62a6f5e5afdea`.

`fused_gdn_prepare_kernel` combines three data-movement tasks in one Triton launch:

- Copy the QKV prefix from packed, non-interleaved QKVZ into dense rows.
- Split packed BA into dense B and A tensors, preserving their dtype.
- Zero every row of the GDN core output, including graph padding.

Z is a view of the QKVZ suffix. There is no normalization, activation, convolution, or recurrent-state update in this kernel. Its outputs must match the corresponding tensor slices and zero initialization exactly.

## Qwen3.6-35B integration

The inspected Qwen3.6-35B-A3B-w8a8 checkpoint uses `Qwen3_5MoeForConditionalGeneration`, with 16 linear K heads, 32 linear V heads and 128-dimensional K/V heads. The existing Qwen patch dispatches its forward to `AscendGatedDeltaNetAttention.forward`.

For tensor parallel size P, the local packed layout is:

| Tensor | Shape |
| --- | --- |
| QKVZ | `[T, 12288 / P]` |
| Dense QKV | `[T, 8192 / P]` |
| Z view | `[T, 32 / P, 128]` |
| Packed BA | `[T, 64 / P]` |
| Dense B and A | each `[T, 32 / P]` |
| Core output | `[T, 32 / P, 128]` |

The fused path runs for the existing packed non-interleaved projection branch when BA is local to the TP rank, both projections have unit column stride, and QKVZ has the hidden-state dtype. Separate QKV/Z projections, interleaved layouts, replicated BA and unsupported column strides retain their previous behavior. Row strides and token slices are supported by the kernel.

The kernel uses a `(token, width-block)` launch grid and 1024-element blocks. Only block zero writes B/A. The wrapper checks that local V heads fit the block and validates shapes, devices and output-buffer layout before launch. An empty token batch returns correctly shaped tensors without launching a kernel.

## Core, graph and prefill compatibility

The existing GDN custom core and norm remain in place. The fused path reuses the initialized core buffer directly; fallback paths retain `torch.zeros`. The core only overwrites actual tokens, so full-buffer initialization remains necessary for profile calls and graph/MTP padding.

PR #22010 operates later, on convolution output entering GDN prefill. Its head-first QKV storage optimization is preserved and is independent of this projection preparation.

The source PR reports A5 captured-device improvements; these are not an A3 or whole-model performance guarantee. Test kernel equality and graph replay on the target hardware before drawing performance conclusions. An eager-only timing may include additional Python/Triton overhead.

## Tests

- CPU dispatch and validation: `tests/ut/ops/test_gdn_prepare_dispatch.py`, `tests/ut/ops/test_fused_gdn_prepare_validation.py`.
- Existing prefill regression: `tests/ut/ops/test_gdn_layout.py`.
- Hardware equality, row strides, TP shapes, aliasing, compile and ACLGraph padding: `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_gdn_prepare.py`.
