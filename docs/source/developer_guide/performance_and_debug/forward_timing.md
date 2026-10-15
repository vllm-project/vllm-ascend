# Asynchronous forward timing

Forward timing records NPU device-stream intervals for real inference in the
standard Model Runner V1. It is disabled by default and does not require an
extra package or a kernel rebuild. Model Runner V2 and specialized V1 subclasses
are not supported by this initial implementation; enabling it there is rejected.

## Enable

Merge the following into the existing additional configuration:

```json
{
  "forward_time_metrics_config": {
    "enabled": true,
    "window_size": 1000,
    "target_batch_sizes": [],
    "collect_draft": true
  }
}
```

Select Model Runner V1 using the existing runner selection settings and restart
the inference workers. Do not replace other additional configuration fields.
No event is allocated when the feature is disabled.

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Enable device-event collection. |
| `window_size` | `1000` | Accepted samples per window and maximum active + pending samples per worker. Target and draft share the window. |
| `target_batch_sizes` | `[]` | Actual request BS filter for both roles; empty means all sizes. |
| `collect_draft` | `true` | Collect MTP/EAGLE/EAGLE3 through the common proposer execution path. Set false for other speculative methods. |

Invalid fields, non-positive sizes and non-boolean switches fail configuration
validation. No new environment variable is introduced.

## Measurement boundaries

- **Target**: one business `_model_forward` call, including its graph replay.
- **Draft**: one common proposer's `_runnable` invocation, covering the entire
  merged draft chain, including sampling inside that callable. It is not a
  per-draft-token duration. The graph object and its cache are unchanged.
- Dummy/profile forwards and graph capture use separate call sites and are
  excluded. Draft calls without a scheduler output are not collected.
- Phase is derived from attention state: prefill, decode, mixed or verify.
  `ChunkedPrefill` maps to mixed. Speculative target steps with scheduled draft
  tokens map to verify; speculative draft steps map to decode.
- BS is `input_batch.num_reqs`, not the padded token count or graph bucket size.

Start/end events are recorded on the same stream. Completed samples are read
only after a non-blocking `query()`. Collection never calls `synchronize()` or
`wait_event()`, creates a collective, or transfers model tensors to the CPU.
Unrelated auxiliary-stream work is not guaranteed to be included unless it is
ordered into the measured stream. These are stream intervals, not a sum of
kernel execution times or the client's request latency.

Event failures preserve the inference result and never retry a forward. Pool
and pending sizes are bounded; overflow increments dropped counts. Normal
worker shutdown emits the tail and counts unfinished active/pending samples as
dropped without waiting for them. SIGKILL does not guarantee a tail log.
With no new forward, there is no background polling or periodic tail flush.

## Read and summarize

The worker's normal logger emits records such as:

```text
Forward timing: window=1 rank=0 dp_rank=0 pp_rank=0 tp_rank=0 role=target phase=verify bs=1 count=5 avg_ms=62.000 min_ms=59.000 max_ms=65.000 dropped=0 errors=0
```

Log timestamps describe publication, not the precise start of every sampled
forward. Inspect every worker's enabled message and timing output; a successful
HTTP request alone does not prove coverage.

```bash
python tools/forward_metrics_summary.py prefill.log decode.log --tp-rank 0
```

The helper weights window averages by sample count and keeps files, worker
PIDs, DP/PP/TP ranks, role, phase and BS separate. Default TP rank 0 avoids
counting every member of a TP group as a separate inference step. Do not sum
parallel rank durations as if they were sequential request latency.

## Compare with benchmarks

Record model, code revisions, topology, MTP length, input/output lengths, cache
state and request distribution. Separate prefix prewarming from measured
requests. Use an isolated load and record sampling boundaries; partial windows
and background requests prevent exact per-request attribution. No request ID
label or unbounded per-request history is retained by this collector.

A target + draft round can generate multiple tokens, so compare its duration
with TPOT only after accounting for the actual generated tokens per round.
TTFT additionally contains scheduling, potentially several prefill chunks,
KV transfer and first-token processing. Whole-run throughput includes time
outside the two measured forward boundaries.

Measure disabled versus enabled runs under the same conditions to assess
instrumentation overhead. Agreement with client TPOT is a consistency check,
not an overhead measurement, an accuracy bound or proof of identical model
outputs. Use the existing profiler when operator traces or independent timing
reference are needed; this feature complements service metrics with bounded
asynchronous target/draft device-time summaries.
