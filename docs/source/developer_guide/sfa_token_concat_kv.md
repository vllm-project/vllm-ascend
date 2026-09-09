# SFA token-concatenated KV storage

This refactor prepares the Python KV lifecycle for DeepSeek-V3.2 SFA.
Each unquantized main layer owns a dense parent with shape
`(blocks, tokens, 1, nope_dim + rope_dim)`. Attention still receives two
NoPE/RoPE views. Their token stride is the full parent width, and the RoPE
storage offset follows NoPE inside each token. Indexer tensors remain separate.

This is a draft integration requiring compatible external kernels and NPU
validation. It does not implement kernels. In particular, passing a strided
view to an existing kernel does not establish that the kernel supports it.

## Allocation and ownership

The v1 and v2 workers select this layout from the actual layer backend and
specification. Main descriptors allocate `num_blocks * page_size_bytes` for
each layer; legacy `shared_by` descriptors retain physical aliasing. Storage
blocks can be split into integral kernel blocks without copying. Padded pages
and unsupported head geometry fail explicitly.

C8 main caches, indexers, non-SFA draft attention, hybrid Mamba layouts and
DeepSeek-V4 keep their existing paths. The native `SfaRemoteD2HConnector` is the
adapted transfer route. Other connectors, including `MultiConnector` and
external implementations reusing the native connector name, retain separate
contiguous main caches. This transport selection does not detect kernel support.

## PD and sparse offload

The producer reconstructs a typed parent from validated views and retains it
through registration. Wire metadata identifies `token_concat`, dtype, head
geometry, the parent page size and the manager/kernel block conversion.
The consumer rejects missing/legacy layout tags or incompatible geometry
before building read descriptors. Mixed old/new PD deployments require both
sides to be upgraded; no wire conversion is provided.

The offload manager allocates one pinned CPU parent per layer and broadcasts
its address, shape and manager page size to TP peers. A peer needing CPU views
restores the contiguous parent through the existing pointer-view interface,
then splits it locally. Main transfer descriptors copy full pages; indexer
and optional scale transfers retain their separate block lists.

Current-KV writeback uses physical block and token strides. The existing CPU
resident-address planner ABI is preserved: Python remaps its source
addresses to the parent token stride, retaining destination addresses and
component copy lengths. Persistent CPU KV views are passed through the fused
wrapper without a `.contiguous()` copy.

## Local validation

The following suites run with CPU PyTorch and pytest, without vLLM or NPU
initialization:

```bash
pytest -q --confcutdir=tests/ut/worker tests/ut/worker/test_sfa_kv_layout.py
pytest -q --confcutdir=tests/ut/kv_offload tests/ut/kv_offload/test_sfa_parent_transfer_cpu.py
pytest -q --confcutdir=tests/ut/distributed/kv_transfer/sparse_kv_offload \
  tests/ut/distributed/kv_transfer/sparse_kv_offload/test_cache_layout.py
```

The transfer suite loads complete production protocol/read modules with only
logging, network discovery and the metadata base stubbed. It copies real CPU
memory and verifies both component views, unordered pages, TP ownership and
metadata rejection. It does not exercise memfabric.

Project UT additionally covers v1/v2 allocation and reshape, legacy aliases,
connector selection, producer registration, consumer state, offload allocation,
wire metadata, request completion and the fused wrapper. These use the existing
mock NPU test environment, not a device runtime.

## Device validation still required

| Area | Required evidence |
| --- | --- |
| External kernel inputs | Correct token/block stride and alias handling for cache writers, indexer/gather and attention consumers |
| KV lifecycle | Prefill, decode, prefix reuse, chunking, block reuse and eviction across nonconsecutive pages |
| Graph | Capture/replay and in-place cache updates retain addresses and visible data |
| MTP and CP | Target/draft ownership, speculative rollback, PCP/DCP block geometry and request reordering |
| PD/offload | Real registration, RDMA reads, TP peer views, current-token writeback, resident misses and fused overlap |
| Numerical/performance | Accuracy versus baseline, memory accounting, throughput and latency |

Keep the integration in draft until the required external capabilities and
runtime evidence are available. Kernel development is a separate work item.
