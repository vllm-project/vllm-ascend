# Local fresh KV and asynchronous writeback

This experimental sparse-offload option lets every TP rank fill current-forward
KV directly from its local activations. TP0 writes the shared Host history pool
on a separate NPU stream. It is disabled by default.

Add `"local_kv_writeback_overlap": true` to the existing
`additional_config.sparse_kv_offload_config`. All TP ranks must use the same
configuration. Eager and full decode graphs are supported; piecewise graph
capture is rejected. Combining this option with `use_fused_overlap` is rejected. Existing sparse-offload restrictions, including no DCP,
PCP or PP, still apply. Prefill and mixed batches retain their original path.

## Ordering and ownership

Each layer has persistent K/V and slot-mapping staging on every rank. The main
stream fills it before recording a ready event. TP0's writeback stream waits for
that event, constructs D2H descriptors, and submits the copy. Shared D2H
buffers are safe because their construction and consumption are serialized on
the writeback stream.

The CPU descriptor builder preserves LRU slot assignment but zeros Host-load
lengths for fresh rows. The local fill kernel uses packed query boundaries and
request identity to populate those slots, including multi-request and
speculative query layouts. Stable historical misses still load from Host.

Attention does not wait for its layer's writeback. After the last target/MTP
attention, the main stream joins the writeback stream **inside the graph**.
This prevents staging reuse from racing with a previous replay's writeback.
The next forward has one TP visibility broadcast before it reads Host history;
MTP has its own forward boundary. This is not a promise of zero synchronization.

For 78 layers, 64 staging rows, and BF16 K/V widths 512/64, persistent K/V staging
uses about 5.48 MiB per rank, plus 39 KiB of slot metadata. This is fresh KV only,
not a second full history cache.

## Validation

The original implementation includes standalone CPU descriptor, NPU local-fill,
and MemFabric writeback tests. Those three test files are not included in this
port. Their results on the original branch do not validate this upstream port.

Full-model, multi-rank validation is still required. Before enabling
the option for a workload, compare generated token IDs/logits with the flag
disabled, including request completion/slot reuse and speculative rollback.
Then compare TPOT and NPU traces under otherwise identical settings. Inspect
writeback stream placement, compute overlap and the final join wait.

MemFabric copy consumes vector-core resources. Staging, local fill, CPU LRU and
history loads still cost time, so a separate stream does not guarantee a
throughput or latency improvement.

## Port provenance

Ported from commit `fceb6dc994c54d8e5b256da0201cc11b88ce0763`, excluding its three test files.
The CPU mask helper is registered in `csrc/torch_binding.cpp` rather than the
original runtime-compiled extension. The current main-branch fused-overlap
path is preserved when this option is disabled.
