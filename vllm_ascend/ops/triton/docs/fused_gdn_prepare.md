# Generic fused GDN projection preparation

`fused_gdn_prepare_kernel` copies packed QKV and TP-local B/A into contiguous
outputs and zeros every row of the core output in one launch. Z remains a
view; normalization, gating and recurrent state are unchanged.

The grid is `(tokens, ceil(max(qkv_width, core_width) / 1024))`. Each program
handles one row and 1024 columns; only column block zero copies B/A. Empty
batches allocate empty outputs without launching a kernel. Row and column
block addresses are promoted to int64 before multiplying strides or widths.

The model uses this path only for non-interleaved packed projections with
matching dimensions, local BA, unit column strides and supported floating
dtypes. Replicated BA, separate projections and other unsupported inputs
retain the existing path. All padding rows are initialized to zero.

A functional custom operator keeps allocations and writes together to avoid
AOT clones of fresh output mutation arguments. Its fake implementation returns
metadata for all four fresh outputs.

This PR extracts the generic kernel from #22489. It does not include the
four-row Prefill kernel, FLA wheel changes, GDN core changes, MRoPE reuse,
or other attention fusions. It is an independent alternative to the
Prefill-only PR #23248; their model-integration hunks overlap and need to be
combined explicitly if both are merged. This PR alone uses the generic path
for supported large batches too, with no prefill-specific speedup claim.

CPU tests cover dispatch, output metadata and GDN forward integration with a
simulated launcher. NPU tests cover exact copies/zeroing, TP-local dimensions,
empty/small batches, row padding, column tails, wide projections and ACLGraph
replay. Hardware and end-to-end performance validation remain required.
