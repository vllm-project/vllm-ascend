# Fused GDN preparation for large prefill batches

This path copies packed QKV and local B/A into contiguous outputs and zeros
all rows of the core output buffer in one Triton launch. Z stays a view.
It does not change normalization, gating, convolution or recurrent state.

The non-interleaved packed-projection branch selects it for at least 1024
tokens, FP16/BF16 QKVZ, unit column strides, TP-local BA, and QKV/core widths
at most 8192. Other inputs retain the existing split/contiguous/zeros path.
The name describes the intended large-batch workload; dispatch depends on
shape, not a host-side prefill/decode flag.

Each program handles four complete token rows. For 16384 tokens and width
8192 this launches 4096 programs. The four-row tile bounds UB use; row address
arithmetic is promoted to int64 before multiplying strides. Tail rows are
masked, and output zeroing includes graph-padding tokens.

A functional custom-op boundary keeps output allocation and writes together,
so AOT functionalization does not treat fresh outputs as mutated inputs that
need cloning. The fake implementation describes all four fresh outputs.

This is the prefill-only extraction from PR #22489. No small-batch prepare
kernel, FLA wheel dependency, ChunkGDN/RecurrentGDN adaptation, MRoPE reuse,
or other attention fusion is included.

CPU tests cover dispatch and the actual GDN forward with a simulated kernel.
NPU tests check exact copies, zero initialization, TP-local shapes, row padding,
mixed dtypes and ACLGraph replay. Hardware tests and end-to-end performance
must be run on the target Ascend device before claiming a speedup.
