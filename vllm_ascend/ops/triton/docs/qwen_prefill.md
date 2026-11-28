# Qwen prefill layout kernels

The batch-one BF16 vision path reads Q and K directly from a contiguous
token-major QKV projection with head dimension 72. Each program rotates both
halves of one token plane using the original half-width cosine and sine tables.
The vision block calls the torch-npu binding for CANN aclnnAddLayerNorm with
additional_output=True. Its first output feeds the MLP and its fourth output
carries the residual for the following addition.

For contiguous, non-interleaved GDN projections, split_packed_qkvz copies the
QKV prefix and Z suffix into separate dense outputs in a single launch. This
still transfers both outputs through device memory. Prefixes of the QKV output
can be supplied to Conv1d and Z can be flattened for gated normalization.

The split QKV RMSNorm MRoPE kernel optionally reads three rows of token
positions with explicit strides, including a broadcast plane stride of zero.
Positions are valid model position IDs indexing the cache. The unit-offset
option rounds the adjusted weights to the original weight dtype before use.
