# Experimental Qwen3.8 Flash Next decode optimizations

These optimizations target the Qwen3.8 Flash Next adaptation with Model Runner V2.
They do not require changes to the vLLM source tree. The three environment variables
below default to `1`, accept `0` or `1`, and contain no sensitive information. No
exports are needed to enable them. Set a variable to `0` before starting the worker
processes to disable its behavior; restart the service after changing it.

| Environment variable | Effect when set to `1` |
| --- | --- |
| `VLLM_ASCEND_QWEN4_EXP_FP32_STABLE_PARTITION` | Use FP32 keys for the stable valid/invalid sparse-index partition. |
| `VLLM_ASCEND_QWEN_MRV2_EARLY_QUEUE` | Permit earlier CPU submission of eligible pure decode batches. |
| `VLLM_ASCEND_QWEN_MRV2_MM_DECODE_EARLY_QUEUE` | Extend early submission to multimodal requests whose complete prompt has been certified as processed. Requires `VLLM_ASCEND_QWEN_MRV2_EARLY_QUEUE=1`. |

For example, to disable early submission and retain the synchronized path:

```bash
export VLLM_ASCEND_QWEN_MRV2_EARLY_QUEUE=0
```

Setting only `VLLM_ASCEND_QWEN_MRV2_MM_DECODE_EARLY_QUEUE=0` disables the multimodal
extension while leaving eligible text decode submission enabled. The stable partition
is independent of early submission and changes only its boolean keys: `0` and `1` are
represented exactly in FP32, the sort remains stable, and the sparse indices retain
their integer dtype.

## Dependency and cache-layout compatibility

This implementation is based on Qwen support PR #16981 at
`eb43043aa94e20aa74cf9883d3a3554a12b8b350`, with its pinned vLLM revision
`544f182d9f0c9bfe8bd7645525951d5e3692a542`. That dependency already includes the
Model Runner V2, MTP, graph-replay, and padded-metadata adaptations from PR #17249;
the older MRV2 commits are not applied again.

The dependency uses packed QSA/GDN cache pages and 128-token GBSA kernel pages.
The metadata integration retains the existing owner's persistent buffers and
block tables; it does not restore the older contiguous-cache allocation. FP32
partition keys also retain the newer GBSA residual-block-mode-zero path, including
explicit incomplete tails and the 512-entry sparse-selection limit. Neither the
cache allocation nor the native GBSA operator is replaced by this optimization.
The GBSA dependency now uses the CANN transformer operator API rather than bundled
operator sources, so the runtime must supply its matching native operators.

## Original QSA metadata inside the target FULL graph

The original Torch QSA metadata producer is captured at the entry of the existing
target model FULL graph. Python metadata construction stays in `prepare_attn`; its
device work is deferred until the captured model forward. The builder returns views
of the cache owner's existing output buffers, so the original producer and attention
layers execute in the same graph with the existing buffer ABI. There is no separate
metadata graph, extra graph replay, fused metadata kernel, or staging copy.

This integration is automatic for the supported Qwen model state. Its configuration
gate checks `FULL_DECODE_ONLY`, DP1, PP1, DCP1, PCP1, a fixed positive MTP K,
and a non-eager target. Dynamic or adaptive drafting, LoRA capture cases, and DBO
retain the original immediate producer path.
The target query length follows the configured K+1; zero-length padding is
permitted. There is no separate 32-token limit: input and output views must fit
their existing persistent buffers, which are checked before capture.
The historical requested capture sizes were
`[1, 2, 4, 8, 12, 16, 20, 24, 28, 32]`; with K3 target query length four, the
applicable target plans are `4, 8, 12, 16, 20, 24, 28, 32`. The effective target
capture grid must contain every K+1-token bucket up to its configured maximum.
Missing, invalid, or sparse grids disable this metadata
migration for the entire configuration, avoiding partial capture or mismatched
rounded replay plans. Unsupported configurations and non-FULL calls
retain the original immediate producer path. MTP retains its existing calls; this
change does not migrate MTP metadata or change MTP graph capture. No embedding or
TP AllReduce preparation graph is introduced.

Each target model state owns its pending and captured plans, including the tensors
whose lifetimes must span capture and replay. A context variable routes builder calls
only within the current target preparation context. The existing Ascend import-time
assignment selects the metadata wrapper once per worker; there is no per-request
function patching or builder-class replacement, and no upstream source is edited.
A weak registry and counters support diagnostic
snapshots and do not choose or own execution plans.

Before native replay, the code checks the plan against the captured input/output
addresses, device, dtype, shape, stride, and Python-specialized geometry. Changed
aliases, missing plans, duplicate shared-builder patterns, or unsupported geometry in
the selected FULL path raise an error before submission rather than replaying an
incompatible graph. These replay failures are distinct from the synchronized queue
fallback described below.

## Earlier CPU submission

The usual path synchronizes the CPU with the corrected device-token-count snapshot
before preparing the next batch. The early-submission path keeps that exact D2H snapshot on
every step, but replaces the CPU wait with a wait on the device stream for an eligible
batch. CPU bookkeeping can then use a conservative upper bound while exact device
positions, sequence lengths, KV slots, rejection state, and subsequent writes keep
their dependencies. It does not change MTP graph capture or sampling.

Admission is deliberately restricted to the audited Qwen model state, MTP speculator,
and target/draft attention backend and builder pairs. It requires DP1, PP1, DCP1,
PCP1, fixed positive MTP K, and a uniform FULL decode batch. Both asynchronous and
synchronous scheduling are admitted; synchronous scheduling may offer less overlap.
Prefix caching, Mamba cache modes, and the global KV-sharing fast-prefill setting
do not disable stable decode submission. Request arrival, resume, preemption,
completion, active encoder work, prompt embeddings, sequence-parallel MoE, DBO,
LoRA, weight offload, connectors, structured output, and unsupported configurations
fall back to the synchronized path. Bounds must fit both target and draft model limits.

Prefix-cache restoration and partial-hit block copies keep their original ordering.
New or resumed requests, prefill steps, and batches with KV block copies still take
the exact synchronized path. Only a subsequent stable decode batch can use the
upper bound. Mamba block-boundary decisions, state copies, accepted counts, and
postprocessing retain the exact device inputs. This does not add fast-prefill or
cache-mode support beyond what the underlying model and runtime already support.

For multimodal requests, scheduler flags alone cannot prove that the last image or
video prompt token has been processed. The worker first consumes a synchronized,
exact snapshot and certifies prompt completion. The certificate binds the request
incarnation, slot, prompt length, media objects and positions, model state, and encoder
cache. Lifecycle changes invalidate the proof. Until a matching certificate exists,
the worker uses the original synchronized path. Realtime inputs, embedding masks,
prompt embeddings, and other unaudited media layouts are excluded.

## Tensor parallelism and remaining limits

Text early submission, certified multimodal early submission, and target metadata
capture do not require a particular TP size. Model sharding must still be supported
by the underlying model, attention operators, and runtime. These changes do not add
support for a TP configuration that the underlying implementation cannot run.
The queue dependency is a per-rank device-count snapshot; the metadata producer
uses per-rank token and cache tensors rather than TP head geometry. Their device
dependencies and replay-address checks are retained for every TP size.

DP1, PP1, DCP1, and PCP1 remain required by both paths. Increasing one of these
dimensions keeps the original synchronized queue and immediate metadata producer.
DP can pad a local batch to another rank's graph shape, which needs a separate
proof for the producer's CPU-specialized mapped-token total and idle batches.
PP and context parallelism change token ownership or CPU consumers and require
separate adaptations.
Other remaining limits are:

| Path | Conditions retained |
| --- | --- |
| Early submission | Exact Qwen model-state and MTP-speculator types; approved target/draft attention builder pairs; fixed positive K without dynamic drafting; a uniform FULL verification batch with K+1 tokens per request and K drafts. |
| Early submission | Pure decode with no new, resumed, preempted, or finished requests; conservative lengths within both target and draft limits; no active decoder replay. |
| Early submission | No prompt embeddings, MM prefix-LM, scheduled encoder work, sequence-parallel MoE, SP bypass, DBO, LoRA, weight offload, KV/EC transfer metadata, auxiliary connectors, adaptive verification, sparse-KV offload, KV block copies, or structured output. The exact snapshot must have been produced on the same stream. |
| Multimodal extension | Both queue switches enabled, a valid exact prompt-completion certificate, stable request/slot/media identities, and non-realtime media layouts without embedding masks, MM pruning, encoder-decoder inputs, or active deepstack embeddings. |
| Target metadata capture | `FULL_DECODE_ONLY`, non-eager target, fixed positive K, no dynamic/adaptive drafting, LoRA, or DBO; positive integer requested capture sizes whose rounded target grid contains every K+1-token bucket. Actual input/output views must fit the existing owner buffers. |
| Target metadata replay | Query lengths zero or K+1, mapped-token count within the padded token count, uncached external token mapping, and one QSA producer per Common metadata object. Captured pointers, devices, dtypes, shapes, strides, static arguments, and query length must match before replay. |

An ineligible queue batch takes the synchronized fallback. Unsupported metadata
configurations and non-FULL calls use the immediate original producer. A violated
metadata alias or geometry invariant after selecting the captured path raises
before native replay; it is not silently bypassed.
The CPU admission and plan-lifecycle tests exercise TP sizes 1, 2, 4, 8, 16, and 32.
These CPU checks do not establish model execution, numerical correctness, or
performance on those TP configurations. The TP4 hardware run below exercises one
text-decode configuration; the other expanded configurations still need hardware
validation.
CPU tests also cover prefix-cache settings, Mamba modes `none`/`align`/`all`, both
scheduling settings, static K values 1/2/3/4/5/8, and metadata batches larger than
32 tokens. These checks establish control flow and buffer handling; the expanded
configurations still require native NPU output/cache and performance validation.

## Validation before production use

These behaviors are experimental, and the three switches are enabled by default.
CPU tests cover the certificate lifecycle, admission guards, stream-wait decisions,
and metadata plan lifecycle; they do not establish NPU execution correctness.
Validate token outputs, cache correctness, actual graph replay, synchronization,
and performance on the intended hardware and workload. Compare latency or throughput
using the same model, environment, and request set. A shorter gap in one profiling
trace is insufficient to establish an end-to-end speedup or generation correctness.

The historical single-variant collection on server 101 used multimodal inputs of
10,000 total tokens, 100 output tokens, concurrency two, TP8, and prefix caching off.
It recorded 99 successful target replays per rank and 97 multimodal early-queue steps.
A separate 24-case check exercised 72 NPU metadata parity updates. These observations
are not a same-environment baseline comparison. Three of seven generation-output
comparisons differed, so the collection does not establish generation correctness or
an isolated performance gain. No new hardware collection was performed for the
source cleanup and default-on update.

With the refreshed dependency, a separate Ascend A3 run used pure-text input of
900 tokens, output of 100 tokens, concurrency one, TP4/DP1/EP, prefix caching on,
Mamba cache mode `align`, asynchronous scheduling, and MTP K3. All three switches
were unset and resolved to their enabled defaults. The service used a 1024-token
model limit, one sequence, a 128-token batching budget, a 512 MiB KV-cache budget,
and FULL decode capture sizes `[1, 2, 4]`. The pinned vLLM revision remained
`544f182d9f0c9bfe8bd7645525951d5e3692a542`, with matching CANN transformer operators.

Two warmup requests, one unprofiled request, and one profiled request completed.
The formal request reused 896 cached prompt tokens; all four outputs had identical
hashes. Every rank recorded successful target and MTP native graph replays during
warmup. Observation wrappers were removed before profiling, and target metadata
replay and text early-submission counters continued increasing afterward, without
capture or replay-guard failures. Profiling began after the first nonempty streamed
text, and all four raw and independently parsed rank directories passed SHA256 and
SQLite integrity checks. The formal request accepted 59 of 123 drafted tokens over
41 drafts, giving an acceptance fraction of 47.97% and mean acceptance length of
2.439. This is a workload result, not a model-accuracy evaluation.

The TP0 trace contained 32 interior complete graph cycles after excluding three
cycles at each end. The MTP-last-graph to next-target-graph interval had a median
of 0.507330 ms and range of 0.488650–0.817716 ms. Graph roles were inferred from the
repeated native graph structure. The interval includes intervening kernels and
communication; it is not whole-device idle time or a matched-baseline speedup.
No multimodal request or other TP size was exercised by this collection. Broader
cache/output correctness and same-environment performance regression validation
remain required. Historical profiling from the previous cache layout does not
validate this revision.
