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

## Original QSA metadata inside the target FULL graph

The original Torch QSA metadata producer is captured at the entry of the existing
target model FULL graph. Python metadata construction stays in `prepare_attn`; its
device work is deferred until the captured model forward. The builder returns views
of the cache owner's existing output buffers, so the original producer and attention
layers execute in the same graph with the existing buffer ABI. There is no separate
metadata graph, extra graph replay, fused metadata kernel, or staging copy.

This integration is automatic for the supported Qwen model state. Its configuration
gate checks `FULL_DECODE_ONLY`, TP8, DP1, PP1, DCP1, PCP1, three speculative tokens,
and a non-eager target. The audited configuration uses fixed MTP K3; the metadata
gate does not independently check the speculative method or dynamic drafting.
The audited target geometry has at most 32 padded tokens and query length four
(zero-length padding is permitted). The tested requested capture sizes are
`[1, 2, 4, 8, 12, 16, 20, 24, 28, 32]`; with K3 target query length four, the
applicable target plans are `4, 8, 12, 16, 20, 24, 28, 32`. Larger or different
capture lists are not covered by this bounded implementation. Unsupported configurations and non-FULL calls
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
and target/draft attention backend and builder pairs. It requires TP8, DP1, PP1, DCP1,
PCP1, asynchronous scheduling, fixed MTP K3, and a uniform FULL decode batch. Prefix
caching and Mamba caching must be disabled. Request arrival, resume, preemption,
completion, active encoder work, prompt embeddings, sequence-parallel MoE, DBO,
LoRA, weight offload, connectors, structured output, and unsupported configurations
fall back to the synchronized path. Bounds must fit both target and draft model limits.

For multimodal requests, scheduler flags alone cannot prove that the last image or
video prompt token has been processed. The worker first consumes a synchronized,
exact snapshot and certifies prompt completion. The certificate binds the request
incarnation, slot, prompt length, media objects and positions, model state, and encoder
cache. Lifecycle changes invalidate the proof. Until a matching certificate exists,
the worker uses the original synchronized path. Realtime inputs, embedding masks,
prompt embeddings, and other unaudited media layouts are excluded.

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
