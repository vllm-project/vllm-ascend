# Experimental Qwen3.8 Flash Next decode optimizations

These optional optimizations target the Qwen3.8 Flash Next adaptation with Model Runner V2.
They do not require changes to the vLLM source tree. All four environment variables
default to `0`, accept `0` or `1`, and contain no sensitive information. Set them before
starting the worker processes.

| Environment variable | Effect when set to `1` |
| --- | --- |
| `VLLM_ASCEND_QWEN4_EXP_FUSED_METADATA` | Use the Ascend Triton QSA metadata producer with the cache owner's existing output buffers. |
| `VLLM_ASCEND_QWEN4_EXP_FP32_STABLE_PARTITION` | Use FP32 keys for the stable valid/invalid sparse-index partition. |
| `VLLM_ASCEND_QWEN_MRV2_EARLY_QUEUE` | Permit earlier CPU submission of eligible pure decode batches. |
| `VLLM_ASCEND_QWEN_MRV2_MM_DECODE_EARLY_QUEUE` | Extend early submission to multimodal requests whose complete prompt has been certified as processed. Requires `VLLM_ASCEND_QWEN_MRV2_EARLY_QUEUE=1`. |

For example, to enable early submission for both text and completed multimodal prompts:

```bash
export VLLM_ASCEND_QWEN_MRV2_EARLY_QUEUE=1
export VLLM_ASCEND_QWEN_MRV2_MM_DECODE_EARLY_QUEUE=1
```

The two QSA switches are independent of early submission. The fused producer replaces
the Torch metadata builder and retains its buffer ABI; it does not create another
preparation graph or move metadata generation into the target model graph. The stable
partition changes only its boolean keys: `0` and `1` are represented exactly in FP32,
the sort remains stable, and the sparse indices retain their integer dtype.

The fused producer is selected at worker import time and also applies to prefill and
MTP metadata. Its coverage is separate from the admission guards for early submission.
Validate all relevant phases before enabling it; the queue guards do not provide a
per-batch fallback for the fused kernel.

## Earlier CPU submission

The usual path synchronizes the CPU with the corrected device-token-count snapshot
before preparing the next batch. The optional path keeps that exact D2H snapshot on
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

## Validation before enabling in production

These switches are experimental and remain disabled by default. CPU tests cover the
certificate lifecycle, admission guards, and stream-wait decisions; they do not prove
NPU execution correctness. Validate token outputs, cache correctness, actual graph
replay, synchronization, and performance on the intended hardware and workload.
Compare latency or throughput using the same model, environment, and request set.
A shorter gap in a single profiling trace is insufficient to establish an end-to-end
speedup or generation correctness.
