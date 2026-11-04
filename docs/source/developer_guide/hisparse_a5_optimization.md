# A5 HiSparse optimization implementation

This PR installs the working A5 optimization modules in `vllm_ascend/`.
They are selected through the backend and attention factories, not by copying
the experimental directory over an installation.

## Scope and selection

`VLLM_ASCEND_HISPARSE_PROFILE` is a non-sensitive, startup-only setting:

- `baseline` (default): the existing platform backend, including current-Core ABI support.
- `a5_c8_main`: the merged-main A5 optimization implementation. Exact normalized Core
  source hashes are checked before adapters are installed. Unknown versions fail
  explicitly. Restart workers after changing the profile or source.

The former `a5_c8_62592` pre-merge profile is rejected. Its historical results
remain recorded; new reproduction uses merged main only.

The A5 profile includes mapped Host writeback and lifetime ownership, CPU page
descriptors, Resident updates, grouped/recycled invalidation, lazy MTP and row-map
preparation, unused staging removal, bounded block-table view reuse, C8
mapping/gather and Indexer-write primitives, and MTP group prefetch. Private
Host pools, existing payload fences and count-copy/MTP ordering are preserved.
Existing CANN Attention compute kernels are used; the unused custom Attention
core remains removed. Kernels with no independent benefit claim are not described
as individually validated improvements.

The HiSparse P/D transport and its source tests are included as installed code.
Its final integration/accuracy/performance qualification remains pending. Lazy
Host mirrors are restricted to the recorded consumer policy; producers require
eager mirrors. CP/PP, multi-DP, A3 and other payload formats are not qualified by
the historical A5 result.

## Check out merged Core and build

Use an A5 environment with CANN 9.2, Torch 2.10, torch-npu 2.10.post7 and a
matching triton-ascend installation. The preparer checks out the upstream-main
commit `d3fb7166e0ac9db82158129face12ae28209757a` and verifies tree
`d7eb5822d87cc449f1377cf59bf499116832bb9f`. It fetches no PR head and creates no
local merge. PR62592, PR60103, PR63801 and PR64989 are merged at this revision; unmerged
vLLM proposals, including the private Host allocation hook, are not dependencies.

This is the source pair under integration review after the November 4 cleanup.
CPU checks do not qualify NPU execution. Historical measurements below retain
their original Core versions and do not qualify this newer pair.

## Common ownership after upstream merges

The plugin uses Core's speculative group prefetch from PR63801. Core decides
bounded decode eligibility, queues follower layers on the existing copy stream
and records one completion event per layer. The copied Ascend staging method
has been removed. The NPU gather implementation remains a device operation.

Core also owns Host mirror completion and the grouped invalidator's pointer
table. The duplicated Ascend mirror method and invalidator manager are removed;
the optional lazy-MTP policy calls Core's existing invalidator. The NPU kernel
still compares rewritten row IDs and invalidates Hot entries.

PR64989 detaches temporary handle/runtime bindings during profiling cleanup and
cleans partial initialization and shutdown through the same ownership boundary.
The plugin adds no second unbind/reset path. Native DMA tickets, pending payload
owners, registration and completion fences remain Ascend responsibilities;
detaching a view alone does not establish completion of an in-flight copy.

Unused backend stream/event/LRU factory methods and the obsolete prefetch flag
are removed. Core already creates ordinary streams/events and initializes LRU
through int32 before casting. The actual backend IPC-event contract is retained.
The baseline resolver workspace is owned by leader cache operations and shared
by followers, matching the optimized backend without Core `backend_state`.

The resolver uses Core's common control flow after aligning the Python cache-op
signature. NPU kernel bodies are unchanged; optional diagnostic outputs are
keyword-only. The remaining stream capture adapter and packed prefill mapping
address device differences. Attention arithmetic, HMA and Coordinator policy
are unchanged.

```bash
python tools/hisparse/prepare_paired_core.py --output /path/to/vllm-main-pinned
VLLM_TARGET_DEVICE=empty uv pip install --no-deps --no-build-isolation -e /path/to/vllm-main-pinned
SOC_VERSION=ascend950dt_9582 uv pip install --no-deps --no-build-isolation -e .
python csrc/hisparse_runtime/build.py --cann-root "$ASCEND_HOME_PATH" \
  --npu-arch "$NPU_ARCH" --resolver-ub-bytes "$RESOLVER_UB_BYTES" \
  --output /path/to/new-native-output --install
```

Set the two build parameters from the matching SDK's SIMT target and usable UB
capacity. Both native libraries must be rebuilt for the Tensor ABI. Binaries,
weights and machine-specific startup scripts are not committed.

## Repeat the single-D configuration

Use full GLM-5.2 W4A8C8 weights, TP8/DP1 and MTP3. The saved configuration keeps
78 target layers plus the MTP layer, FP8 Attention/Indexer with FP32 scales,
memory 0.8, Hot10240, Host128GiB/rank, maximum length139264, batched8192,
maximum sequences32, no prefix caching, and the original graph shapes/warmup.

```bash
python tools/hisparse/serve_reproduction.py --model "$MODEL" --arm off --port 8029
# After that owned instance has exited and released its resources:
python tools/hisparse/serve_reproduction.py --model "$MODEL" --arm on --port 8029
```

`--print-command` prints either configuration without starting a model.
`--host-ip` accepts the actual data-network address when one is required.
`reference_configuration.json` contains both exact argument lists. The fixed
`C8-L128-O8192.jsonl` preserves prompt IDs, temperature0, seed42 and generation
limits; only the served model alias is changed to `hisparse-review`.

Use EvalScope1.13.0 in a separate environment, one unscored primer and two
measured rounds per arm. Keep formal timing free of profilers/observers. Record
actual output IDs, success count, MTP position acceptance, settled HiSparse
counters, TPOT/TTFT distributions and a running8/waiting0 decode window.
Missing counters do not establish zero misses. These fixed forced-length
outputs are not dataset accuracy results. P/D requires its own end-to-end test.

## Merged-main qualification checkpoint (2026-11-03)

The Tensor resolver ABI preserves the parallel all-Resident path and releases
callback-local Tensor owners after submission; allocator stream records retain
the queued device storage. Dynamic shape, padding, nonzero-offset, lifetime and
real graph-owner checks passed in the private A5 runtime. This is independent of
Attention arithmetic and does not change count-copy/MTP submission order.

Current-batch RoPE and request-row prefixes reuse only their certified current
layout, with per-round values and group-local addresses kept current. Large
written-slot invalidation sorts exact `(persistent state, Host slot)` keys,
then probes each group's table. Small Decode writes keep their previous kernel.
At 8192 writes, 32 states, 22 groups and Hot10240, a paired NPU primitive run
measured 37.07 ms before and 3.08 ms after, including table reset. This was beside
an idle owned model, not an exclusive-host model benchmark. Nine changing replay
cases matched an independent integer oracle. Follower query prepacking retained
KV readiness and passed three interleaved graph owners across six shapes;
its separate end-to-end contribution has not been isolated, so that scheduling
change is excluded from the submitted implementation.

A full GLM-5.2 W4A8C8 A5 single-D run used TP8/DP1, MTP3, V2, C4,
32768 input/8192 output, EvalScope 1.13.0, temperature0, seed42 and no prefix
cache. Both arms completed two formal rounds without profiler or observers:

| Round | OFF TPOT (ms) | ON TPOT (ms) | OFF/ON MTP acceptance |
| --- | ---: | ---: | --- |
| 1 | 12.1283 | 12.5453 | 59.89% / 59.89% |
| 2 | 12.1332 | 12.0475 | 59.89% / 64.92% |

All 16 requests completed with 8192 actual output IDs each. ON settled reported
miss/H2D deltas and eight-rank idle miss tails were zero; these are not physical
DMA byte measurements. ON round1's output IDs and per-position MTP counts matched
both OFF rounds: TPOT regression remains about **3.42%**. Round2 changed output
mode and must not be pooled to claim the 3% goal. A small pre-existing external
NPU context was preserved, so these are shared-host results. Three bounded
natural controls passed; they are not a dataset accuracy result.

These private runtime results qualify the exercised paths and recorded source
revisions, not all rebased package combinations, A3, miss workloads or final P/D.
A subsequent group-preparation/clean-padding candidate completed output- and MTP-matched
N2 at 12.5430/12.5452 ms, still about 3.41% above OFF. Its local CPU reduction
did not produce a material model gain, so those changes are excluded from this PR.
Local resolver CPU contracts passed 11 tests and page-descriptor contracts passed
32. Focused Ruff/format and whitespace checks passed. Full `bash format.sh ci`
was attempted but stalled installing actionlint; it was interrupted and is not
reported as a full CI pass. New checked-in NPU test bodies still require CI.

## Existing evidence and outstanding checks

The saved source tree `9e4da1dd92448aa8883bc900099b12840cd985cd2baa30b34f694a286b6a6734`
completed C8/input128/output8192, two rounds per arm, single-D TP8/MTP3.
OFF/ON mean TPOT was 16.511773/16.946701ms, or +2.6341%; individual changes
were -0.6374% and +5.9794%. Decode throughput was 480.0999/464.8775tok/s.
Each arm returned 131072 token IDs with16/16 successes. Reported ON miss/H2D
deltas settled at zero; physical DMA bytes were not measured. MTP acceptance
and long outputs differed. Stable3%, final P/D and final accuracy remain open.
Sanitized metrics are in `hisparse_a5_reference_result.json`.

Before this submission,132 focused CPU checks passed. Earlier current-Core ABI
tests passed12 actual NPU cases with zero skips. Neither result qualifies the
entire relocated implementation. Historical P/D CPU tests exposed stale fixture,
native-boundary and wire-contract failures; they remain recorded for follow-up.
The author requested publishing now and continuing validation on the new machines.
No additional model evaluation or full CI pass is asserted for this submission.
AI assistance was used; human review and merge acceptance remain required.

## Review scope

The historical full-source archive and superseded investigation documents have
been removed from the review diff. Commit `a623a45b2392c6dd61c53405e08700307046deb8`
retains them for source provenance. The archive was not imported by the installed
package. Its removal does not change a kernel, model result or execution policy.
The archive-only lint exclusion and a workstation-specific profiling destination
were also removed. Reproduction launches do not enable profiling.

The remaining implementation has these dependencies:

| Area | Reason to retain |
| --- | --- |
| Backend, binding, stream and dtype integration | Connect common HiSparse storage and completion ownership to NPU resources. |
| Resolver, gather, C8 layout and Indexer writes | Produce the existing Attention operator's input layout and move selected cache rows. They are not replacement Attention compute kernels. |
| Host DMA and immutable descriptors | Preserve allocation lifetimes and transfer completion while reducing writeback preparation. |
| Resident updates and invalidation | Keep reused/written slots consistent with the device residency tables. |
| Lazy MTP, row preparation, staging and view reuse | Retain the explicit, source-scoped optimization profile; individual model benefit is not implied. |
| P/D connector and composition | Preserve the requested P/D feature and safe cache-release ordering; final qualification is still pending. |
| Focused tests and native build sources | Check retained behavior and build required libraries from source. |
| Baseline versus paired optimized modules | Support different reviewed Core contracts. They cannot be collapsed by deleting one implementation before a current-main port is validated. |

Latest-main model validation and pruning of superseded Core adapters are in
progress. Historical paired-profile measurements do not qualify latest main.

## Current-main allocation and native operand compatibility

The native A5 C8 operators require E4M3FN operands even when common Core
represents packed FP8 storage as UInt8. The SFA and Indexer wrappers use
`native_c8_cache_dtype` for their native operands and typed cache views. This
does not convert the backing bytes or modify Attention compute kernels.

Prolog V3 weight preparation casts the source to dense ND before transposing
or splitting, then retains the original final NZ operands. It does not modify
the original source Parameter in place. BF16, INT8 and FP8 source tensors in
both ND and NZ storage passed the complete production preparation method on
A5, including exact values and source-Parameter preservation.

The optional Core byte-allocation callback is forwarded through every raw
cache allocation, including aligned transfer buffers, shared Indexer K/scale
backing and KVPP bundles. The callback receives the physical byte count and
retains allocation ownership. Invalid dtype, size, contiguity or device is
rejected rather than repaired with a hidden replacement allocation. Default
allocation retains the existing zeroing and alignment policy.

QSA declares its draft architecture and live CPU query-boundary requirement
on its Ascend backend. Common worker setup no longer imports an optional
model module to classify unrelated attention backends. This replaces the
temporary empty-type import fallback; custom allocation callbacks are now
implemented rather than rejected.

For this follow-up, 39 isolated CPU checks passed. Twelve actual A5 NPU checks
passed against runtime commit `069de6b02b089b583203c00c3a5b293607a840a0`:
one real Indexer constructor/weight-lifecycle test with three changing-input
graph replays, six full Prolog weight-preparation cases, three aligned callback
storage/lifetime cases, and two shared Indexer K/scale allocation cases.
The Indexer quantization relative error against a CPU Hadamard/dequantization
reference was about 2.5%. Earlier fixture conversion failures are retained;
they are not counted as production-method passes.

These are component checks. The runtime port retains newer-main changes that
are absent from this PR's older base, so its complete source is not identical
to the PR head. Full-model serving, generation, performance, final P/D and A3
qualification remain separate. Full lint again stalled installing the
actionlint environment; finite Ruff and diff checks passed.

Current-main generation also changed private rejection-sampling interfaces:
draft logits can carry a logical-request-to-local-row mapping, and logits
statistics take the total logit extent. The Ascend adapter now checks actual
kernel parameter declarations rather than inferring this ABI from the package
version string. It forwards the mapping to common statistics and Ascend
verification/resampling without changing sampling probabilities. Tensor-only
and one-hot drafts retain their original paths. Sixteen CPU dispatch checks
passed, and eleven actual A5 checks passed on runtime commit
`50c9c8d46d86ddb6315156f28d6f4e6e1bcfafeb`: six greedy CPU-oracle cases,
two mapped-versus-logical random-cache cases and three changing graph replays.

The coordinator also forwards an explicitly supplied hash-lookahead window
through all factory routes to the owning block pool. Omitted arguments retain
the older paired Core API. Twelve focused CPU forwarding checks passed.
These compatibility fixes do not qualify full-model outputs: the preceding
model returned an empty first response and then failed in the old sampling
ABI. Its clean exit status is not counted as a successful model test.

The current Core routers also consume a per-token MoE padding mask. Ascend
now refreshes real and padded rows after dummy/profile runs, matching Core's
per-batch contract. Three CPU cases and four NPU mask/post-routing cases passed;
the latter mocked the raw TopK operation and checked the production mask path.
This correction did **not** remove the premature EOS in the new-main OFF model.
It is not established as that output problem's root cause or a performance gain.
Device input/logit-row diagnosis remains in progress. No new-main performance
or maximum-concurrency claim is made while natural outputs are invalid.

Actual eight-rank input observation subsequently found all 20 Prefill IDs were
zero, while positions, request row31, sequence length20 and logit index19 were
consistent. Core's new `UvaBufferPool.copy_chunks_to_uva` writes through
`np.concatenate(..., out=buf.np[:n])`. That ndarray view bypasses Ascend's
monitored setter, so the emulated UVA buffer was not marked dirty and its NPU
payload stayed zero. This happens with HiSparse OFF too.

The formal fix supplies an Ascend pool implementation for bulk writes: retain
the same pinned destination and pool lifecycle, mark the written prefix in
constant-size bookkeeping, then perform the existing asynchronous fallback
copy before device consumption. It preserves real UVA and Core's explicit-copy
fallback. It does not disable UVA, alter generation parameters, or change an
Attention compute kernel. Five portable CPU checks passed. Twelve actual A5
cases passed for each of the PR-base and current-main candidate variants,
including reproducing the old zero-token write, changed/noncontiguous request
rows, real Core Prefill, changing graph replays, uint64 bit patterns, pool
growth/reuse and mixed bulk/sparse writes. These component runs shared only
our own diagnostic service and were not timed model benchmarks.

The current-main candidate retains that main revision's existing prefix-write
callbacks; the older PR base uses the equivalent explicit prefix tracking.
After activation on current-main runtime commit
`587d0a18641899a0cb8b0919531ec16541d42c81`, all eight ranks received the exact
request Prefill IDs. Three bounded natural controls returned the reviewed
answers (`2`, `323`, and counting from 1 to 10). The task-only input/logit
observers were restored to the original methods on all eight ranks before
performance timing. An unscored 16,384-input/8,192-output primer completed,
with all 8,192 output token IDs retained. These checks establish this bounded
OFF-model result, not dataset accuracy or ON-model qualification.

The current-main OFF baseline subsequently completed two formal rounds per
shape (C4, output 8,192):

| Input tokens | Mean TPOT, two rounds | Pooled steady decode throughput | Successes / output IDs |
| --- | ---: | ---: | ---: |
| 16,384 | 11.420669 ms | 365.7324 tok/s | 8/8 / 65,536 |
| 32,768 | 12.118077 ms | 354.7869 tok/s | 8/8 / 65,536 |

The steady windows require sampled running4/waiting0 and a flat prompt-token
counter (162.95 and 160.93 seconds pooled across the two rounds). Each shape
had identical generated token IDs and MTP counts between its OFF rounds, with
zero reported preemptions. These are single-D, current-main baseline results;
the source-pinned historical optimization profile was not enabled. ON and
capacity sweeps are still pending. No current-main ON/OFF regression or
maximum-concurrency result is claimed yet. Both variants' failed preparation
attempts and raw evidence remain recorded; no failed attempt is counted as a
pass.

### Reviewed current-main C8 port

The initial `a5_c8_main` port used the five reviewed Core files at
`5c8ae7c74aaa99d1f7fbeea942fd7282e3413810`. The current profile pins the newer
merged revision in the checkout instructions above. It checks the entire file set;
mixing old/new files or moving to another main revision is rejected. Use
`tools/hisparse/serve_reproduction.py --profile a5_c8_main` with either arm.
That launcher removes the obsolete Indexer C8 layer flag and retains the
reviewed `weight_nz_mode=1` MXFP/Prolog policy.

The port forwards `row_width` instead of the removed `head_size` argument and
adapts Core's `[group,state,page]` Resident allocation through a stride-only
`[state,group,page]` view for the existing update kernel. Its fallback receives
the original Core layout. The eager Resident path constructs a current batch
view when Core no longer supplies its old helper. No cache contents, selection
policy, import reservations, Attention compute kernel or common source changes.

Python 3.12 also adds an empty `type_params` field to function ASTs. This made
the unchanged MTP certificate fail the older Python 3.11 fingerprint. The
guard now normalizes only that empty field, preserving its original reviewed
digest and rejecting executable/generic changes. The actual new-image raw
digest and identical source bytes were checked before activation.

The Indexer spec builder reads the selected backend's explicit C8 capability;
it retains rejection for C4 and unsupported backends. The selected C8 backend
still checks the complete native K/scale geometry before allocating storage.
This fixes an earlier unconditional worker guard encountered after weight
loading, rather than disabling Indexer format validation.

Fourteen CPU profile/guard cases, four additional launcher cases and eight
Indexer capability cases passed. Actual A5 runtime
`f4eed3d77ce602ab0004f4a4c5e4a5a783014046` passed eight
component cases: the real current-main cold adapter installation, three
group-first Resident updates against independent CPU values, three changing
Resident graph replays, and the existing mapped resolver/gather/scatter oracle
with graph replay. Its native sources were rebuilt in the matching new image.
The earlier certificate-guard failure is retained. Full ON model, matched
ON/OFF performance and maximum active concurrency are still pending.

Current Core's generic FP8 MLA source spec also describes a 576-byte row in
UInt8 storage. The Ascend spec builder now declares the native C8 Host source
as 656 one-byte elements in E4M3FN before Core plans or allocates pages: 512
NoPE bytes, 16 FP32 scale bytes and 128 BF16 RoPE bytes. The strict cache-bind
geometry check remains unchanged. This corrects the storage declaration; it
does not resize an allocation after planning or convert cached KV contents.
Four production-branch CPU cases passed. Runtime
`f422eac15ee3eb6231670779215f9494b9685dac` then passed eleven actual component
cases, adding two real Core Host-view layouts and reproduction of the rejected
576-byte source spec to the preceding eight cases. Same allocation addresses
and all byte patterns were preserved. Full ON-model qualification remains
pending.

### Current-main Indexer writes and capture-only mirrors

The Indexer write entry now dispatches tagged native HiSparse C8 caches to
the existing page-stride-aware writer. Flattening these shared slabs with
`.view()` fails because neighboring pages have gaps. Runtime
`01420d7d37cd8f3a96f9e0bd2205555ea9b0f4c9` passed seventeen actual A5 component
cases, including six changed-input writes across eager/graph modes. The
independent oracle checks K bytes, FP32 scale bits and untouched page gaps.
The INT8-only fused store and Attention compute kernels are unchanged.

V2's FULL-graph warmup and capture both call forward with runtime mode NONE.
Submitting a per-layer Host mirror in these forwards incorrectly creates
connector work without a real step boundary; the second forward then hits
the existing duplicate-mirror guard. Both Ascend HiSparse metadata builders
now mark capture metadata explicitly. Device KV writes remain in the graph;
only the CPU mirror callback is omitted for that metadata. Real eager
forwards still call the original finish method, and real graph replay keeps
the connector's post-forward mirror owner.

Four focused CPU checks passed. Runtime
`a65d03e4c27b4c4e8fa03883cae09c1c28733752` passed six actual A5 component cases:
the real cold adapters, three device-write graph replays with no capture Host
callbacks, a real Core finish submission and the preserved duplicate guard.
These component checks do not establish full-model correctness, performance
or the active-concurrency limit. Full ON-model validation is continuing.

### Page-locked Host storage for both writeback paths

The first new-main natural request exposed a native D2H failure at
`aclrtMemcpyAsync`, reported asynchronously at a later operation. A focused
same-payload matrix on the new A5 image passed public Torch copies, queued
native copies to pinned storage and queued copies to registered pinned
storage. Both legacy-registered mmap and V2 MAPPED-only mmap reproduced the
native DMA failure. Device-accessible mapping alone was insufficient for
the asynchronous DMA path on this runtime.

The optimized Host allocator now requests PTA pinned storage and retains a
page-aligned view of that same allocation for its existing mapping owner.
Pool size, payload format, descriptor order, streams and completion fences
are unchanged; there is no second Host pool or Attention kernel change.
Three CPU ownership/alignment cases passed. Runtime
`c7f5536467c329294aab58baf3339160fd5f793c` passed five actual A5 cases: cold
adapters, byte-exact native and AIV queued writes, resolver/gather graph
replay and an actual 128 GiB allocation with first/last native copies.
Full-model ON measurements remain pending; the earlier failed request is
excluded from performance results.

The first allocator-only model attempt still failed: common HiSparse creates
its own private backing before calling `pin_tensor`, so the standalone
allocator change had not reached that path. The optimized backend now installs
an exact-source adapter at `allocate_pinned_host_pool`, before any Host cache
views exist. It returns one pinned backing and its exact registered alias to
Core, and retains that registered owner in the existing DMA lifetime manager.
Non-owning backends retain their original allocator; shared-region allocation
is untouched. Remove this adapter when Core exposes a backend allocation hook.

Three forwarding/fallback/source-rejection CPU cases passed. Runtime
`2a95cc754bada8c8a9ca74ef2cbd92dff16c6ca6` passed five actual A5 cases through
the real `HiSparseHostPool.allocate` entry, including the actual 128 GiB pool,
registered-owner retention, byte-exact native/AIV writes and mapped graph
replay. This closes the component integration gap; full-model requests and
performance still require their own results.

## Matched zero-miss result on November 4

The native Prolog V3 wrapper can write the current opaque C8 payload directly
to the strided Resident cache. Prefill and unsupported shapes retain staging
and masked scatter. Padding uses the current Resident slot table, whose producer
writes -1 for invalid rows; ordinary Attention slots are a different owner.
The Host mirror completion point and native Attention arithmetic are unchanged.

Within a fresh Core ForwardContext, exact Resident address/layout/row/stream
aliases share one int64 conversion. Each later forward performs a fresh copy;
captured replay executes that copy with current values. Strong Tensor and storage
owners prevent address recycling during the scope. Actual eight-rank capture
showed 78 Target consumers and 20 conversions, including distinct Core views
over the same group slots.

The recorded private runtime `95d1e7fb543865ab8178bd50895ddbb176b78025`
used merged Core `5c8ae7c74aaa99d1f7fbeea942fd7282e3413810`, the configuration above,
C4, 32768 input and 8192 actual output tokens/request. EvalScope 1.13.0 ran one
unscored primer and N2, with no profiler or observation hooks during timing.

| Round | OFF TPOT ms | ON TPOT ms | MTP acceptance, both arms |
| --- | ---: | ---: | ---: |
| 1 | 12.128254 | 12.472067 | 59.893611% |
| 2 | 12.133250 | 12.451795 | 59.893611% |

Mean TPOT increased 2.730074%, within the 3% goal for this shape. All 16 requests
succeeded, with 8192 actual output IDs each. Input/output hashes, draft/accepted
counts and per-position MTP counts matched exactly across both arms. Settled
reported miss/H2D deltas and eight-rank idle tails were zero; these are not
physical DMA measurements. A small external NPU context was preserved, so this
is a shared-host result and N2 is not a claim of statistical significance.

The actual wrapper has 21 CPU branch/oracle cases and 21 forward/alias/stream
scope cases; 40 changing real-NPU graph cases covered five graph owners,
padding/holes, three alias-view consumers and fresh same-source graph ownership.
The 3% goal is not established for other lengths, concurrency, A3 or P/D.
Miss workloads and final integration/accuracy qualification remain separate.

The earlier alias-identity-only preparation was never model-dispatched. A direct
write without shared conversion produced changed MTP modes and no demonstrated
isolated gain; those results are retained as investigation evidence, not a win.
The private runtime also retained an earlier CPU page-preparation experiment;
that unproven experiment is not added by this commit. The recorded private
runtime result does not replace a final clean-build qualification of the PR head.
