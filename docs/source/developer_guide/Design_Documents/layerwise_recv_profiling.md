# Layerwise receive queue profiling

The optional receive timeline records FIFO tasks and their dependencies without
changing dispatch order. Configure it in `kv_connector_extra_config`:

```json
{
  "use_layerwise": true,
  "layerwise_trace_path": "/data/traces/recv",
  "layerwise_trace_capacity": 100000,
  "layerwise_trace_interval_ms": 1
}
```

The parent directory is created when profiling starts. Each receive worker writes
`<path>.tp<rank>.pid<pid>.json` every five seconds and during connector shutdown.
Snapshots replace the previous file atomically. Use a distinct prefix for each
independent run. After the measured interval, allow a snapshot to complete before
stopping the server; some engine exit paths do not call connector shutdown.
Omitting `layerwise_trace_path` disables collection and the observer thread.

| Option | Default | Meaning |
| --- | --- | --- |
| `layerwise_trace_path` | unset | Output prefix; unset disables profiling. |
| `layerwise_trace_capacity` | 100000 | Positive integer bounding stored events and retained task probes. |
| `layerwise_trace_interval_ms` | 1 | Dependency query interval, from 1 to 1000 ms. |

## Fields and stages

Each task has a monotonically increasing ID within its worker trace. Enqueue
records include request IDs, layer, groups, save owner, and data path. Timestamps
use `monotonic_ns`; `clock_anchor` relates them to wall-clock time for correlation
with separately collected NPU traces.

- `enqueue` / `dequeue`: host queue residence.
- `save_cpu_wait_begin/end`: save-finished host event wait.
- `save_npu_wait_begin/end`: existing save-event device synchronization.
- `gate_record_wait_begin/end/timeout`: attention gate condition wait.
- `gate_npu_wait_begin/end`: existing attention-event device synchronization.
- `external_slot_wait_begin/end`: external buffer release callback.
- `metadata_begin/end`: address and metadata construction.
- `backend_submit/complete`: synchronous backend call boundaries. This interval
  includes native submission and completion; it is not device-only transfer time.
- `complete/error`: receive task outcome.
- `resources`: 100 ms samples of process peak RSS, NPU allocator usage and peak,
  reserved bytes, and active step lease-key count. RSS is the process lifetime
  peak. Native Store allocations are outside the PyTorch allocator; collect
  device-level memory statistics alongside these fields. Unavailable native
  backend queue depth is `null`.

Collect an NPU profiler trace alongside the host trace to separate native API
submission from device transfer, and to correlate transfer with attention.
Use the server's `--profiler-config.profiler=torch` and
`--profiler-config.torch_profiler_dir=/data/traces/npu` options, then bracket a
short capture with `/start_profile` and `/stop_profile`. Collect benchmark
latencies separately with the NPU profiler inactive.

## Readiness evidence

A diagnostic thread queries already-recorded events; it does not synchronize
NPU work or consume CPU events. Samples contain each observed task's dependency
values, a recheck of the blocked head, and IDs of ready followers. A dispatch
change invalidates an in-progress sample. Tasks not yet published to the FIFO
are excluded. An external dependency without a nonblocking query contract is
`null` (unknown), and cannot establish follower readiness.

These observations identify possible HOL opportunities. They do not establish
that reordering preserves buffer ownership, lease lifetime, or request ordering.
Those constraints require separate analysis before changing scheduling.

The trace is bounded. `dropped > 0` means the capture is incomplete: shorten the
capture or increase capacity before drawing a negative conclusion. Readiness is
sampled; no positive samples does not exclude an opportunity shorter than the
sampling interval. Sampling overhead must be measured with profiling on/off.

## Analysis

```bash
python tools/kv_pool/analyze_recv_trace.py \
  /data/traces/recv.tp0.pid123.json --output /data/traces/summary.json
```

The report includes per-task stage p50/p95/p99, positive HOL observations, affected head
tasks, unknown dependencies, incomplete spans, and the original HOL evidence.
Repeated waits and batched backend calls are summed per task before calculating
stage quantiles; individual span distributions are also retained.
Consecutive positive samples define an observed span, not proof of continuous
readiness between samples. Keep raw traces alongside the summary.

For each backend, vary shared-buffer count and prefetch depth independently,
include dedicated/independent layers, and use single and concurrent requests.
Run each configuration at least three times. Report TTFT, throughput, host/NPU
peak memory, in-flight metadata and backend/lease metrics separately from the
receive-stage distributions. A final scheduling recommendation requires these
end-to-end measurements; the timeline alone is not a performance benchmark.

`tools/kv_pool/bench_recv_serving.py` primes prompts, clears the local prefix
cache outside the measured interval, and records streaming TTFT and throughput.
Use `--prime-base-url` to prime through a separate writer sharing the Store.
The reader must expose the local development `/reset_prefix_cache` endpoint.
The script requires a positive external-prefix-cache hit delta before accepting
the run as a receive benchmark.

## Native backend queue depth

Connector in-flight calls and native queued work are separate quantities.
For backend diagnostics, instrument the native thread pool immediately after
push/pop while its existing queue mutex is held. Record current depth, lifetime
peak and balanced enqueue/dequeue counts, with bounded storage and an explicit
overflow indicator. Keep the diagnostic backend build separate from the normal
installation and measure its enabled/disabled overhead.

For Memcache, distinguish the read/write data pools from client/metadata pools.
Small GVA batches can bypass the read pool entirely. Use a separate chunked-copy
positive control to verify that the data-queue probe is active. For Mooncake's
synchronous Ascend transport, measure the dispatch thread pool; asynchronous
query queues and device command queues are different scopes.

An externally sampled depth can miss short occupancy. A peak updated at every
native queue mutation preserves the process-lifetime maximum, including warmup;
do not label that value as a per-trial sampled maximum. Missing native probes
remain unknown, rather than becoming zero in the connector's resource events.
