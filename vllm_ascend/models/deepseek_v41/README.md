# DeepSeek V4.1 eager bring-up

The V4.1 backbone is registered through `model.py`, matching the package
structure used by DeepSeek V4. There is no separate `modeling.py` and no
model-side KV-cache owner.

## Current execution path

- `model.py` owns the V4.1 backbone, delayed mHC handoff, attention projections,
  source references, compressor invocation and the correctness-first eager
  attention path.
- `attention/dsa_v41.py` owns paged-cache registration and metadata plus the
  unfused SWA/long-KV gather, attention and cache-scatter operations.
- `core/deepseek_v41.py` owns cache specs, hybrid grouping, sizing, allocation
  and reshape.
- `compressor.py` owns ratio1/ratio2 compressor parameters and the ratio2
  FP32 state cache. `indexer.py` owns Index-K cache updates, Indexer scoring,
  candidate-block filtering and chronological TopK selection.

The model reuses DeepSeek V4's quantization-aware projection, MoE and output
projection implementations. Small operators replace the fused DSA kernel for
the initial eager milestone.

## Hybrid cache layout

The allocator follows the DeepSeek V4 layer-tuple pattern with one global
block-ID lifecycle and four separate layer-outermost buffers. Each buffer
contains all blocks for one shared slot; each block contains the owning
group's resource tuple. Different groups own distinct simultaneous live IDs.
Within the merged group, KV and index share an ID at disjoint byte offsets.

| Group | Resources | Logical block size | Slots |
| --- | --- | --- | --- |
| G0 | C2 KV/index at layers 2, 8, 14; C1 KV/index at layer 20 | 128 | 0-3 |
| G1 | FP32 circular compressor state at layers 2, 8, 14 | 32 | 0-2; slot 3 unused |
| G2-G11 | SWA layers 0-39, four consecutive layers per group, window 128 | 128 | 0-3 |

Without DSpark there are 12 groups and 51 cache specs. The base attention block size is
128 at production dimensions. C2 stores 64 compressed rows per logical block;
C1 and SWA store 128 rows. State stores 32 uncompressed FP32 rows with width
1024 in one private ring page per request.
C2 and C1 share the original-token block table, with compression applied by
per-layer metadata builders. No paired-block mapping is needed.

For MLA width 512, index width 128 and one KV head:

| Slot | Full-context tuple | Page bytes | Component offsets (bytes) |
| --- | --- | --- | --- |
| 0-2 | C2 KV + INT8 index K + FP16 scales | 131072 | KV 0; K 65536; scale 73728 |
| 3 | C1 KV + INT8 index K + FP16 scales | 147712 | KV 0; K 131072; scale 147456 |

C2 KV has 65536 payload bytes. Its index spec is padded from 8320 to 65536
bytes, placing 57216 unused bytes after the scales. SWA/state occupy offset
zero and are padded to their assigned slot capacity. Thirty SWA layers have
no padding; the ten layers in slot 3 have 16640 padding bytes per page. Each
state spec uses all 131072 bytes as 32 ring rows; G1 leaves slot 3
reserved but unused. Padding is applied to cloned specs and is idempotent.

With `N` global IDs (including the reserved null ID), the raw uint8 buffers
have sizes `N * [131072, 131072, 131072, 147712]`. Allocation, startup admission,
maximum-length sizing and concurrency use their sum: **540928 bytes =
528.25 KiB per global ID**. Merged C1/C2 block demand is counted once.

Typed zero-copy views retain the slot's physical page stride, not the
component's page size. C2 KV is `[N,64,1,512]` BF16; index K/scales are
`[N,64,1,128]` INT8 and `[N,64,1,1]` FP16. C1 uses the same widths with 128
rows. SWA is `[N,128,1,512]` BF16; state is `[N,32,1,1024]` FP32. Components
other than state need not be contiguous. State must fill its slot contiguously;
no whole-context gather is introduced by allocation.

### DSpark in the same four slots

Aurora DSpark uses `DeepseekV41CacheBackend` and the V4.1 execution path for
both DSA_CP settings. Noncausal draft queries pass explicit physical SWA
indices to SparseFlashMla with mask mode 0. CP slices these global indices
and preserves the full visible KV length for each local request. Context KV
writes use the same stride-aware cache scatter as the target model. The V1
proposer remains eager; this routing does not enable draft graph capture.

The optional Aurora DSpark model adds one group, G12, containing exactly three
`DeepseekV41DraftSWASpec` resources: `mtp.0.self_attn.swa_cache`,
`mtp.1.self_attn.swa_cache`, and `mtp.2.self_attn.swa_cache`. They occupy offset
zero in slots 0, 1, and 2 respectively; G12 leaves slot 3 unused. The target
groups and their padding remain unchanged. This is **13 groups, 54 specs and
four physical buffers**, still **540928 bytes per global ID**.

Each draft view is `[N,128,1,512]` BF16 with 131072-byte block stride and no
padding. The planner validates draft geometry against target SWA and rejects
foreign resources, compressed drafts, extra draft layers, and any geometry
that would enlarge the existing slots. An explicit draft spec keeps G12
separate from target SWA while reusing its `SlidingWindowManager` semantics.

G12 owns its own block table and live global IDs. All three draft layers use
that table, accessing different physical slots at the same ID. Target groups
use other live IDs, so sharing the backing does not share live target KV data.
Release/preemption returns IDs to the common pool. G12 adds one group's SWA
page demand, not three groups' demand; available-memory sizing still divides
by 540928, and rank shrinking changes only N.

DSpark context KV is projected independently for each draft layer using the
inherited DSV4 SWA backend and the group's own slot mappings. The target exports
the incoming residual streams from the checkpoint-selected auxiliary layers.
Composite-config selection reads Aurora's text config. Target and draft MoE
dispatchers are selected by expert/execution shape to avoid sharing mutable
dispatch state across incompatible expert counts. With DSpark, `auto` cache
dtype is resolved to BF16 before constructing the inherited DSV4 draft backend.

The 32-row FP32 target ring requires **1..31 speculative tokens**. A verifier
writes the anchor plus S speculative input rows. After accepting A drafts, the
next forward starts at `P+A+1`; if that position is odd it needs row `P+A`.
At most S newer rows follow it, so S below 32 preserves it through the tail
write. S=32 can overwrite the anchor after complete rejection and is rejected
at initialization. Per-batch limits cannot exceed the configured maximum.
Rejected compressed KV/index rows remain outside the accepted sequence length
and are overwritten when those positions are recomputed.

Target eager and `FULL_DECODE_ONLY` modes retain their existing dispatch;
the V1 DSpark proposer runs eagerly. Draft graph capture is not enabled.

### Earlier design comparisons

The original block-outermost implementation reserved 393216 bytes per ID
across 17 groups with separate C1/C2 groups, both using logical block 128.
The unimplemented comparison design kept C2 logical block 256 (128 stored
rows), C1 block 128 and 17 groups sharing three 147712-byte slots: 432.75 KiB
per ID. The new design reserves 528.25 KiB per ID but reduces the number of
full-context and SWA-group IDs. Compare memory for the same request workload,
including state retention and free/null IDs, rather than comparing the
per-ID divisor alone. Operator and serving performance remain unmeasured.

Index source layers 24, 28, 32 and 36 compute new selections in the reference
architecture but do not own another copy of the long KV or Index K. Candidate
blocks originate at layer 20. Consumers retain the source prefix and retrieve
the source cache from `static_forward_context`; shared modules are never
re-registered under consumer layers.

## Supported milestone and remaining accuracy work

The runtime contract is model runner V1, eager or `FULL_DECODE_ONLY` mode,
BF16 cache, hybrid KV management, PP/DCP/PCP equal to one, and
tensor/data/expert parallel serving. `FULL_DECODE_ONLY` retains Aurora main's
eager prefill and full-graph decode dispatch. DSpark is the only supported
speculative method, subject to the retention bound above. Prefix caching is
supported for cacheable attention groups; the circular compressor state remains
request-local and is excluded from prefix-cache hits. KV transfer and other
graph modes fail closed.

The fallback attends over local SWA plus the compressed rows selected by the
Indexer/Candidate path. Engram execution is intentionally disabled: its two
roughly 196 GB embedding tables require a distributed HBM layout, while the
temporary CPU/NFS mmap implementation was both prohibitively slow and
numerically unverified. Full accuracy still requires HBM-sharded Engram at
layers 1 and 14 and reference FP8/FP4 rounding. These omissions must not be
interpreted as full model accuracy.

## Circular compressor integration

The v0.27.1 compatibility layer registers a circular spec and manager inside
vLLM-Ascend. G1 owns one global ID per request, retained until finish or
preemption. Its block table has one column; ordinary position-to-page slot
mapping is disabled. C2 uses the original global ID and `position % 32`.
The ring's 128-KiB contiguous page matches the shared-slot stride without
block-ID expansion or copying the cache. Full C1/C2 and SWA groups keep their
existing address calculations. Thirty SWA layers remain unpadded and ten
retain 16.25 KiB padding per page.

C2 retains the existing FP32 projection weights and computation. The projected
Triton entry point pools current-chunk rows plus prior ring residuals before
updating the last 32 rows of the ring. State remains FP32, including values
that BF16 would round away. The pooled output is BF16 and passes through the
existing model RMSNorm, preserving its epsilon and rounding order. C1 and the
standalone Triton compressor API retain their paths.

Completed pairs occupy their completion-token output rows, matching existing
long-KV/index slot mappings and source RoPE metadata. Per-source output buffers
are allocated before memory profiling. Device metadata uses persistent builder
buffers for graph replay; inactive requests have zero lengths and IDs. Dummy
capture requests receive distinct non-null ring IDs; idle DP synchronization
runs skip ring state updates. Model runner V1 supports eager prefill and
`FULL_DECODE_ONLY` dispatch, with runtime correctness still unverified.

G1 admission now reserves one ID regardless of sequence length. Slot capacity
and proportional rank shrinking are unchanged: total backing bytes remain
`N * 540928`. Prefix caching reuses cacheable attention groups while keeping the
circular scratch state request-local; it does not enable KV transfer, V2, or
unsupported parallel modes.

## Validation

For the earlier block-outermost implementation, on the A3 remote container
the W8A8 checkpoint loads under TP4/DP4/EP, the
service becomes healthy, and greedy eager smoke requests return coherent
answers (`2+2 -> 4`, Chinese capital question -> Beijing). The focused cache and
mHC suite passes 30 tests. Formal task-level and long-context accuracy remain
follow-up gates.

The merged four-slot implementation has local static validation only;
neither eager nor full-graph decode correctness has been verified for it.
Allocator/worker/metadata tests and slot-backed QLI/SparseFlashMla tests are
provided, but their torch/NPU execution is deferred. The earlier serving
results above do not validate this layout. Remote synchronization, builds,
correctness verification and performance measurements require a later run.

Circular-ring changes have **local static checks only**. Run the dependency-free
`python3 tests/check_aurora_ring_static.py` to check placement and ring arithmetic.
The circular manager, metadata, and `test_deepseek_v41_ring_compressor.py` tests
are authored but have not been executed with torch/NPU. Numerical correctness,
full-decode graph replay, model serving, and performance remain **not verified**.
No remote synchronization, builds, or device execution were performed for this
change. Remote Run Manifest evidence belongs to a later requested phase.

DSpark coverage adds dependency-free grouping/allocation checks and 5797
acceptance/rejection schedules. Torch tests cover shared storage, group-ID
isolation, rank shrinking, configuration/auxiliary-state wiring and per-group
draft metadata. NPU tests cover draft context stores in the actual shared
backings and ring residuals after rejection. These torch/NPU tests are authored
but **not executed**; DSpark correctness, target graph replay and performance
remain **not verified**.
