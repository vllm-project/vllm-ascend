# GLM5 Next KPool tail compression

The entry point is `glm5_next_kpool_tail_compress_and_write_cache_triton` in
`vllm_ascend/ops/triton/glm5_next_kpool_tail_compress.py`. It updates the raw
request-owned tail and the BF16 cache of completed pools. Sparse TopK selection
continues to use the existing Ascend Lightning Indexer path.

## Tensor and address contract

- Raw keys and gates are FP32. Tail storage is `[num_blocks, 2, C, head_dim]`,
  with key plane 0 and gate plane 1. All four tensor strides are passed to the
  kernels; padded blocks and noncompact planes are supported.
- `R = index_kpool` controls pool completion, softmax and APE indexing.
  `C >= R` is the ring capacity. Non-MTP uses `C=R`; speculative decoding
  uses `C=R+num_speculative_tokens`, retaining one complete pool plus all
  not-yet-accepted lookahead rows.
- Each request owns one tail block. Historical position p is addressed with
  `tail_block_table[request, 0]` and offset `p % C`. Absolute positions never
  select additional block-table columns.
- The runner supplies circular `tail_slot_mapping = block_id * C + p % C`.
  Normal, fused and draft mapping use the same address contract. Negative
  slots are invalid. The metadata builder consumes this stable buffer.
- Completed pool storage remains BF16 `[num_blocks, entries_per_block, 1, D]`.
  Its mapping is independent of tail mapping; incomplete pools have no output
  slot. APE remains FP32 `[R, D]` and uses pool offset, not ring offset.
- Requests occupy contiguous query rows. `cum_query_lens` contains request end
  offsets (without a leading zero); `seq_lens` are absolute end positions.
  Within each request positions are contiguous and agree with these lengths.

## Ordered launches

The compression launch gathers each completed pool. Rows inside the current
query come directly from FP32 input; earlier rows come from the old tail ring.
The kernel applies softmax over the R gate-plus-APE rows for each dimension,
then writes the weighted key into its BF16 compressed slot. Only real,
nonnegative, pool-completing positions with a valid compressed slot write.

The second launch seeds only the final min(query length, C) rows of each
request. For contiguous positions these rows occupy distinct modulo-C slots,
so each slot has a unique writer even for prefills spanning many ring wraps.
Short requests update only the slots they visit and preserve other history.
This launch occurs after every historical read in the compression launch;
there is no dependence on execution order between concurrent Triton programs.

Tail writes apply to every real token, whether or not it completes a pool.
`compute_topk=False` skips selection but still advances both caches. An empty
batch returns without a launch. Padding rows beyond the final query end and
negative positions/slots do not write, even if a reused graph buffer contains
positive slot values in its padded region.

## Precision and lifecycle

The structural migration preserves FP32 raw K/gate and BF16 compressed K.
BF16 tail is a separate numerical change because rounding only historical
rows can make pool output depend on chunk boundaries.

The scheduler owns allocation, release and prefix-cache boundaries. Tail
blocks are request-local and not prefix shared; a prefix hit must leave a
pool-aligned boundary or provide reconstruction of the missing raw history.
This operator cannot reconstruct an uninitialized historical tail itself.

C=R guarantees retention for monotonic non-MTP prefill/decode: each incomplete
pool needs at most R-1 earlier rows, and all complete pools in a long query read
the current input before seeding. MTP target verification writes its committed
first row plus up to `num_speculative_tokens` candidate rows before acceptance
is known. `C=R+num_speculative_tokens` prevents those candidates from wrapping
onto the committed open pool. Rejection then needs only the normal logical
length rollback: replacement tokens overwrite the rejected positions while the
historical rows required to recompute a boundary pool remain intact.

## Verification

Independent tests should compare pooled outputs with an uncompressed FP32
reference for every chunk-boundary residue modulo R, long multi-request
prefills, decode and C>R. Use nonadjacent block IDs and padded block/plane/token
strides with guard values to detect accidental compact-layout addressing.
Verify final tail slot contents, incomplete-pool writes, empty batches,
negative sentinels, positive stale graph padding and repeated graph replay.
Kernel and model results must be recorded on the target NPU; CPU reference or
syntax checks alone do not establish device correctness or performance.
