# HiSparse performance inclusion decisions

This record separates measured CPU improvements from optional device kernels and
from final-head model qualification. The PR now includes page-descriptor CPU
vectorization in addition to the platform-backend feature. It does not import the
entire experimental integration branch.

## Included: CPU page-descriptor construction

The common worker computes source/destination byte addresses for every transferred
page across every layer. Profiling identified this preparation in the gap after
draft, before the next Target submission. Vectorization fetches each current
tensor's geometry once, validates storage identity/layout each call, and computes
the descriptor matrix with NumPy instead of repeated per-layer arithmetic.

It preserves transfer-major/layer-major order, duplicate transfers, IDs, extents,
the existing descriptor allocator and the single submission callback. It adds no
kernel, stream, D2H/H2D payload or completion event. Page retirement, Host durability,
Resident eviction and count-copy/MTP ordering are unchanged.

### Evidence

Recorded A5 Host CPU experiment: 79 layers / 21 Resident groups, real CPU Torch
tensor metadata, median wall time in microseconds:

| Page transfers | Original | Vector v1 | Included vector v2 |
| --- | ---: | ---: | ---: |
| 1 | 1556.043 | 317.487 | 183.798 |
| 8 | 1572.593 | 333.477 | 198.658 |
| 32 | 1612.333 | 368.376 | 234.973 |

These measure CPU descriptor preparation, not DMA latency or per-token savings.
The v2 experiment passed 57 byte/geometry checks. Source SHA256 before formatting:
`34d390b114bf54339b24d90d38b88e81ba9f3355c94ed7e23e6e0635077f66cb`.
The port retains AST-identical `_PageGeometry`, `_geometry` and `enqueue_transfers`
definitions. Its cold installer is separately adapted for the PR's paired source.

The v2 builder was also present in completed full GLM-5.2, TP8/DP1, MTP3/V2,
single-D, C8-input-128/output-8192 model runs. The first CPU-submission package
recorded ON/OFF mean TPOT regression 4.5912%; the later unused-staging package
recorded 3.8293%. Both used two rounds per arm with settled reported zero misses.
This verifies that the integration executed; it is **not** an isolated model
speedup attributable to descriptor vectorization. Other changes, output tokens
and MTP acceptance differ. The 3% goal and final P/D/accuracy acceptance remain open.

For traceability, the corresponding integration source-tree SHA256 values are:

- CPU-submission package: `46e0d50c1a43ef23f4b4108ee11c4e62c48db6b97547e6b015f5b3a7a1c01429`.
- Later package: `b7bf04864e45cbc7255ba5b307455a91944e5a430759058f3f0e70beb6339063`.

Neither integration tree is this rebased PR head. Its final native/model result
must be obtained independently; no final-head model pass is claimed.

### Activation and compatibility

The common interface lacks a descriptor-builder hook. A temporary Ascend worker
wrapper is installed by `AscendHiSparseBackend.check_available()` before serving.
It replaces only `_enqueue_transfers` for NPU handles and reviewed common files.
Unrecognized sources, another owner's wrapper and unsupported geometry retain
the original. Storage rebinding after geometry creation is rejected before DMA.
No global installation happens merely by importing the module.

Supported common worker source SHA256 values:

- PR62592 paired prefetch revision `1fd827f6df466fc9b591428fcc977be76ffc6861`:
  `81067d36662399fa5fc517bb3f334b78d48dd35247d5f405fcbd3e5982875197`.
- Measured frozen integration worker:
  `1953b3e582502d16df38bba4a5966e0b98492feb67b65cdac7f2b0397ca4b229`.

The two `_enqueue_transfers` definitions are AST-identical. Their submission
implementations differ, so the wrapper deliberately calls the active worker's
original submission method instead of copying one implementation over the other.
An external CPU comparison using both real source definitions passed 144 changing
descriptor comparisons with identical counts, bytes and transfer IDs. Submission
was a recording callback, not physical DMA.

Portable regression command:

```bash
python -B -m pytest --noconftest -q \
  tests/ut/distributed/kv_transfer/test_hisparse_page_descriptors.py
```

Observed: 32 passed. Production modules are file-loaded to avoid plugin/device
initialization. Cases include changed current values, byte gaps, storage offsets,
BF16-sized/opaque rows, 79-layer layouts, duplicates, sentinels, invalid ranges,
unsupported layout fallback, CPU/nonwriter paths and foreign/source-version guards.
These do not execute NPU kernels or prove physical DMA/event lifetime correctness.
The existing 12 isolated backend cases also passed after adding the cold installer.
All 26 changed Python files passed Ruff, format and syntax checks. The required
full `bash format.sh ci` attempt timed out after 55 seconds while installing the
actionlint environment; standard CI remains unpassed. The timeout cleanup stopped
only that local check's process tree, not any model service.

## Existing primitives: individual necessity review

The original `ops/hisparse.py` and `ops/triton/hisparse.py` are unchanged by this
performance update. They are NPU implementations of backend operations, not newly
proven performance optimizations. Keep the distinction when reviewing them:

| Primitive | Purpose and decision |
| --- | --- |
| `_translate`, `_lookup` | Logical/physical and lookup helpers; keep for current resolver, no standalone speedup claim. |
| `prepare_residency_kernel` | Current-row Resident/Hot classification and miss preparation; required by the current implementation, not proof of minimal launch count. |
| `classify_residency_kernel` | Miss/key membership for replacement; keep, optional fused alternatives not imported. |
| `protect_residency_kernel` | Protect currently selected Hot entries; correctness requirement when misses require replacement. |
| `partition_residency_kernel` | Partition replacement candidates; keep current semantics rather than unqualified policy changes. |
| `update_lru_kernel` | Maintain replacement order; a simpler all-Resident fast path needs separate evidence. |
| `assign_residency_kernel` | Assign Hot slots and mapping for misses; do not remove based only on a zero-miss benchmark. |
| `finish_residency_kernel` | Produce final attention indices/counts and miss lists; keep current contract. |
| `residency_stats_kernel` | Export existing hit/miss statistics; not a new fusion or claimed speedup. |
| `gather_rows_kernel` | Copy missing opaque Host rows into Hot; preserve bytes, offsets and graph behavior. |
| `invalidate_kernel` | Prevent stale Hot hits after writes; separate grouped variants lack isolated benefit evidence here. |
| `pack_attention_rows_kernel` | Bridge selected packed rows to the existing SFA consumer layout; no CANN Attention compute modification. |
| `scatter_packed_kv_kernel` | Write new K/RoPE into the packed cache; no projection or quantization algorithm modification. |

## Excluded optimization candidates

| Candidate | Decision and reason |
| --- | --- |
| Custom AscendC Attention | Removed from the PR snapshot with its native registration and dedicated tests; the existing SFA operator covers the active path. |
| AscendC union resolver | Separate residency path; not needed for the current feature contract, final-head benefit not established. |
| C8 map-pack/fused resolver and 32-row fusion | Current feature rejects attention C8; bundle measurements do not establish each kernel's marginal benefit. |
| C8 Indexer fused write | NPU primitive evidence exists, but isolated model benefit and this feature's C8 integration are missing. |
| Grouped/recycled invalidation | Some variants passed correctness; some were slower. No blanket benefit claim or automatic inclusion. |
| Fused Resident update | Changes the device update kernel; CPU packing and combined model runs do not isolate that kernel's benefit. |
| Mapped AIV Host writeback | Changes physical write execution and descriptor lifetimes; earlier variants regressed, current combined results are insufficient for this port. |
| Lazy MTP / row-map / unused staging | Scope-certified D-only policy with further C8/grouped-invalidation dependencies; standalone port and marginal benefit not established here. |
| Persistent gather / follower query-prepack | No qualified standalone end-to-end benefit for this head. |
| Block-table prefix view reuse | CPU-only qualification; no completed NPU/model result. |

"Excluded" is a decision about this PR's evidence and scope, not proof that the
candidate is useless. Do not add kernels solely because they appeared in a faster
combined package. Revisit them with independent correctness and benefit evidence.
