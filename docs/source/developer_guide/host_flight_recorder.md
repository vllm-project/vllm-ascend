# Experimental KV-centered DFX

The experimental MRV1/MRV2 recorder keeps bounded, recent CPU scheduling and batch
evidence for accuracy investigations. Host-only mode does not read device tensors.
Separate opt-in device probes add bounded snapshots and reference comparison.
Neither mode adds model invocations, dummy forwards, device synchronization or
collectives. Device mode DOES enqueue extra kernels and pinned D2H copies; it
consumes device memory/bandwidth and is not a zero-overhead production feature.

## Enable explicitly

`dfx_config.enabled` is the master startup switch for all DFX features. Leave
the configuration unset, or pass `--additional-config '{"dfx_config":{"enabled":false}}'`
to disable them, even when individual probe options are configured. Changing this
switch requires restarting the engine; live runtime toggling is not supported.

With `enabled: true`, `device_capture_interval: 0` (the default) keeps device
probes off. Set it to `100`, for example, to sample every 100 local executions,
or `1` to sample each eligible execution. Sampling reduces capture frequency,
not the overhead of each capture, and can miss errors between samples.

```json
{
  "dfx_config": {
    "enabled": true,
    "output_dir": "/secure/dfx",
    "run_id": "deployment-2026-09-20",
    "max_records": 512,
    "max_buffer_bytes": 33554432,
    "max_record_bytes": 1048576,
    "max_dumps": 4,
    "dump_on_violation": true
  }
}
```

Pass this object using `--additional-config`. `enabled` defaults to `false`.
When disabled, the runner does not construct a recorder, allocate capture
buffers, start a writer, copy inputs, or perform recorder I/O. Conditional
checks remain in the execution/input preparation/output paths. No zero-overhead
claim is made; enabled-mode performance must be measured on the deployment's NPU.

MRV1 subclasses remain unsupported. MRV2 uses the same configuration, with the
adapter-specific coverage below. No new environment variable is introduced.

### Model Runner V2

The same `dfx_config` works with `VLLM_USE_V2_MODEL_RUNNER=1`. The adapter records
worker-received scheduler outputs, prepared request order, request-state row
indices, query offsets, host computed-token upper bounds and scheduler-derived
block history. Resume replaces block history; normal updates append; preemption
and completion remove it. Budget overflow resets this mirror and records a gap.
This mirror is not authoritative allocator state.

MRV2 assembles decode/draft tokens on-device. Its host records deliberately set
`input_token_ids` to null; enable device sampling to retain actual prepared
device tokens and positions for sampled executions. Device evidence includes
the `prepare_attn` slot mappings and block tables, and sampled KV pages before
preprocessing/forward and after forward. It is not a snapshot of backend-private
or graph-internal metadata, nor necessarily the final model-specific inputs.
Ordinary eager FA/MLA layouts enable slot/position and valid KV checks; recurrent
and specialized layouts retain evidence without claiming those checks.

Async output checks run only after the original output resolution; no extra wait
or D2H is added for output observation. Source execution IDs are captured before
sampling. Only last PP ranks check sampled output; every worker has independent
evidence. Existing worker-requested idle DP executions are observed, but ordinary
warmup/profiling/capture calls are not recorded. No dummy invocation is added.

MRV2 limitations: raw pre-sampling logits and backend-private GDN metadata are not
captured; CPU KV plan checks are not applicable to device-assembled inputs.
The adapter does not reconstruct complete token history for golden-trace
alignment, so reference comparison is inconclusive. Local transfer registration
audit is not yet adapted. These gaps are recorded explicitly. CP/ubatch device
sampling is skipped because producer ordering is not established; KVPP and KV
transfer omit KV copies for the same reason. Host history still works. Live NPU,
multi-DP, eager/graph and throughput validation remains required.

## Evidence and limitations

- `scheduler_received`: an owned snapshot of the scheduler output received by
  the worker, before runner-local transformations. This is not a scheduler-side
  event stream or an exact snapshot of every subsequent scheduler revision.
- `host_batch_prepared`: request IDs in the runner's actual row order, host token
  preparation values, logical positions, per-request counts, computed-token
  counts, and per-request block IDs. Each record is self-contained for the
  captured host batch, even if the request started before the retained window.
- Numeric NumPy arrays are copied into immutable bytes. Background JSON exports
  encode them as `numpy-base64`; the inspector decodes them for humans.
- Unsupported objects, including device tensors and custom non-dataclass
  connector objects, are represented by a type marker and an `omissions` path.
  An over-budget record is replaced with an explicit incomplete gap record.
- CPU token preparation values can be stale/placeholders under asynchronous or
  speculative execution and can be modified by subsequent CP/device preparation.
  `actual_device_inputs_verified` on this host record is always false. Separate
  device records do not retroactively turn a host preparation into a device input.
- `resolved_output`: emitted token IDs in original request order, host NaN counts
  when already available, bounded local request-generation identities and detector
  findings. Async outputs are observed only after the executor's existing result
  resolution; no extra device wait or copy is introduced.

The file describes these restrictions in its `coverage` field. It supports host
timeline inspection, not exact device replay. Request IDs are not globally
unique generations; use worker epoch plus execution ID to distinguish records.

## Detection and export

CPU checks cover total scheduled count, negative counts, duplicate batch request
IDs, and the mapping between scheduled counts and actual batch order. A violation
queues an export of the records available at that point; a pre-input violation
does not yet contain that step's prepared batch.

Existing diagnostic tools can trigger from the worker execution thread:

```python
runner.dfx_recorder.trigger("token_repeat", source_execution_id=418)
```

Check for `None` first. The worker also exposes `dump_dfx_trace(reason,
source_execution_id=None)` and `get_dfx_stats()` for the executor's existing
worker RPC mechanism. No service endpoint, additional polling thread, signal handler, new
device collective, or automatic wiring into unrelated detectors is added.
The CPU receipt time and source execution ID are separate; never substitute the
current batch for a delayed detector's originating batch.

### Active host detectors

`detect_outputs` defaults to true when DFX is enabled. It checks output row/ID
mapping, token vocabulary bounds, existing NaN counts and existing logprob arrays
for NaN/+Inf. It never enables logprob production or existing NaN computation:
missing observations are counted, not interpreted as clean results. Masked `-Inf`
alternatives are legal. This is not a raw-logits finite detector.

Repeated token periods (1–4 tokens repeated at least 16 times by default) and
configured `token_patterns` raise **symptom** findings, not proof of accuracy
failure. Histories use request generations, survive batch row changes, and are
bounded by `max_tracked_requests` (4096) and `token_history_size` (128). Out-of-order
results are counted and skipped. Old/new requests with reused IDs are separated.
Each resolved batch is limited to `max_checked_tokens` (4096); exceeding it records
a coverage gap and resets continuity. These output checks do not detect arbitrary
finite-but-wrong results without independent reference evidence.

Speculative acceptance uses the immutable request/proposal mapping and resolved
output lengths (accepted drafts plus one target/bonus token). Discarded empty rows
are excluded. A bounded window reports `spec_acceptance_low` below
`spec_acceptance_floor` (0.1) after `spec_min_proposals` (128). This is a workload-
dependent symptom, not evidence that an inherently difficult draft task is wrong.

`detect_host_kv` is a separate opt-in (default false). It independently checks host
position continuity, allocation bounds, request-to-block-table correspondence and
duplicate planned write slots. It skips async/speculative/CP/MRoPE/compressed paths,
hybrid block conversion, circular tables and recurrent-state groups. Skips are
counted. These are **host-plan** checks, not evidence that a device kernel wrote
the expected slot, nor a check of existing dummy execution's derived metadata.

Detections automatically export the ring when `dump_on_violation` is true. A
nonblocking lock protects result/execution thread access; contention drops
diagnostics and increments `contention_drops` instead of blocking inference.
Dropped output observations invalidate token-history continuity. Bounded histories,
drops, skips and missing reference data mean detection is not guaranteed.

### Device mode and ownership observations

Add the following fields to the enabled `dfx_config` above:

```json
{
  "track_block_ownership": true,
  "audit_transfer_registration": true,
  "device_capture_interval": 64,
  "device_capture_layers": [],
  "device_capture_blocks": 2,
  "device_capture_bytes": 262144,
  "device_pending_limit": 2
}
```

All three capabilities default off. An empty layer list rotates through allocated
local layers, one per sampled execution. A nonempty list must contain exact layer
names allocated on this worker. Invalid initialization is counted/logged and does
not abort serving. An enabled configuration is not proof that evidence exists:
export `coverage` is derived from completed records retained in that export.

- `block_ownership`: worker-observed attach/detach/owner-change events, sharing,
  finishes/resumes/preemptions and synchronous non-spec computed-token rollback.
  Associations are bounded by `max_tracked_blocks` (65536). An association epoch
  is NOT an authoritative allocator generation or proof of free/zero/COW.
- `kv_layout`: per-layer/component shape, stride, dtype, storage offset/pointer,
  group ID and explicit NHD/HND layout. It supports tuple caches and separately
  strided recurrent-state components without flattening the shared allocation.
- `device_snapshot`: selected pages before/after the actual target forward,
  input IDs/positions, allowlisted runner-boundary backend metadata and sampled
  raw logits. Bytes encode contiguous logical values; recorded original strides
  do not imply the dump contains physical allocation padding. BF16 bytes are
  preserved, not silently converted to FP32 for storage.
- Real/sync execution type and local probe ID are recorded. Existing idle-DP
  `execute_dummy_batch` is observed, but no new dummy call is issued. Compilation,
  profiling and warmup are skipped. Sync probes sample page zero plus previously
  observed pages, checking changes and writable slots; unobserved pages may still
  be corrupted without detection.
- Copies are ordered on the existing producer stream. Private clones and pinned
  destinations remain alive until `Event.query()` reports completion. The
  existing writer thread polls and checks CPU copies, including final idle steps;
  it never calls `synchronize()`. Unknown event completion retains buffers and
  disables further capture rather than recycling memory prematurely.
- Event/copy failures, pending-queue pressure and byte limits are explicit gaps.
  Device probes do not cover externally ordered writers: when KV transfer or
  sparse offload is enabled, cache snapshots are omitted, not claimed consistent.
- Plain eager AscendMetadata/AscendMLAMetadata, without CP/spec/compression or
  hybrid kernel-block conversion, checks device slot bounds, duplicate/padding
  writes, position/block/slot agreement and changes outside planned slots. NaN/Inf
  checks use verified write positions and, when available, sequence lengths and
  block tables. Invalid tail NaNs are retained as observations, not false alarms.
- Eager, non-CP/non-spec GDN has a limited conv/SSM adapter using actual query
  offsets and state/cache indices. It checks sampled state changes with no active
  query and nonfinite active state. Other recurrent layouts remain evidence-only.
- Graph mode can retain snapshots but does not claim final internal graph metadata
  validation. Microbatch metadata lists and backend-private arguments are not
  automatically treated as equivalent to the runner's metadata dictionary.
- An in-flight separate device-metadata producer (DSA paths) is explicitly skipped:
  its original waits occur inside the model, so pre-forward copying would race it.
  The probe neither forces a new wait nor presents those bytes as a stable snapshot.
- `audit_transfer_registration` currently adapts Mooncake hybrid's local worker
  registration only: logical/physical block scale, byte stride/length, allocation
  bounds, missing views and alias groups. It does NOT verify remote descriptors,
  DMA completion, staging-buffer reuse or transmitted values.

Raw-logits checks rotate a bounded row/vocabulary interval, with exact coordinates
stored in `scalars`. They occur before grammar/sampling processing. They are not
full-vocabulary/full-batch checks. KV pages take priority over the remaining raw
logits export budget. No valid-data region is inferred for unsupported layouts.

Device records may arrive after an output-triggered export. A matching late
snapshot can enqueue a follow-up export, subject to the same queue/quota. If a
historical step was never sampled or has been evicted, its former KV cannot be
recovered from the current cache. This is not a full checkpoint or exact replay.

### Independent reference comparison

`reference_trace` optionally names a trusted local golden trace (`{rank}` is
expanded to the worker rank). It requires device mode. Loading is initialization-
only and file size is bounded by `max_buffer_bytes`. Completed snapshots are
compared in the background; a numerical mismatch automatically exports evidence.
`reference_atol` and `reference_rtol` default to 1e-3 and must be calibrated per
model/dtype. A supplied trace is not automatically an authoritative oracle.

Comparison requires matching model/revision/topology/rank, execution/layer,
request order, selected pages, observed inputs and bounded prompt/output-history
fingerprints. Different requests, missing history, unknown valid regions or
ambiguous reference steps return `inconclusive`, not `pass`. This supports known,
aligned regression workloads, not arbitrary live requests against a static file.
Independent online reference execution is not implemented.

The same comparison is available offline, without loading a model:

```bash
python -m vllm_ascend.dfx.inspect_trace suspect.json --reference golden.json \
  --atol 0.001 --rtol 0.001
```

It compares verified KV regions/recurrent states and aligned logits samples,
reporting mismatched elements and maximum/mean absolute errors. It does not
restore caches into a serving worker, rerun a model or prove which side is correct.

### Historical-case coverage boundary

The 61-case research corpus is not an acceptance result for this patch. Host
mapping/position/write-plan fault injections test relevant failure mechanisms, not
full reproductions of the historical PRs. NaN and repeat symptoms can expose some
consequences but cannot establish root cause. The following remain unimplemented:

- Exhaustive device/kernel/graph metadata validation, allocator-authoritative
  lifetimes, complete draft/CP/compressed adapters and global DP snapshots.
- In-flight KV transfer, offload and copy-on-write race checking; local registration
  arithmetic and before/after sampled pages cannot establish those lifetimes.
- Norm, reduction, RoPE, quantization, MoE scaling and state-merge mathematics:
  require independently vetted references and intermediate-operator probes.
  A provided aligned trace can expose a downstream mismatch but is not a generic
  mathematical oracle or automatic first-divergence localization.
- Weight-load coverage, parsed model semantics and front-end stream order:
  require hooks outside the model runner.
- Rare-token distribution bias: requires statistical tests, not single-token
  agreement or a generic logprob threshold.
- Full checkpoint restoration and exact scheduling/RNG/graph/connector replay.

No claim of 61/61 detection or zero enabled overhead is made. Covering all these
layers is a separate, materially wider change than the bounded runner patch.

`queued` means accepted, **not** written. Inspect completion/error counters and
wait for an `incident-N.json` file. Writes start as `.partial` and are renamed
only after successful close. No fsync/crash-durability guarantee is provided.
Shutdown performs a bounded best-effort drain; abrupt termination may lose data.

```bash
python -m vllm_ascend.dfx.inspect_trace /secure/dfx/worker-ID/incident-1.json \
  --request-id request-A
```

Filtering a request preserves its full co-batch and the matching scheduler
record. An optional `--execution-id` restricts the output. The reader rejects
unknown schemas and never unpickles or runs model code. Inspect trusted local
files only; it is not a hardened untrusted-file ingestion service.

## Bounded resources and failure behavior

`max_records` bounds records, not steps: a sampling step can produce three
records, plus optional ownership/device events. Detector histories have separate
bounded storage, outside the ring budget.
`max_buffer_bytes` uses conservative retained-object accounting, not a
hard process RSS limit. `max_record_bytes` limits each capture; counts, strings
and arrays consume this budget. Minimum byte limit is 4096; the per-record limit
must not exceed the buffer limit. Count range is 2–65536; buffer limit is at most
1 GiB; per-record limit is at most 64 MiB.

The writer can hold one active export and one queued export, each retaining a
snapshot of at most one ring. Budget for up to three ring-sized retained sets
plus temporary capture/JSON/base64 allocations. `max_dumps` (1–100, default 4)
limits exports per worker lifetime, including failed writes. It does not rotate
or delete prior worker directories. Operate an external filesystem quota and
retention policy across deployments/restarts. This is a recent-history recorder,
not unlimited persistence of every scheduling step.

Capture, initialization, and export failures do not abort serving. Counters expose
capture errors, over-budget gaps, eviction, busy/quota-rejected exports, and disk
failures. A busy queue drops the export request rather than waiting for disk.
The Python writer still contends for CPU/GIL: asynchronous I/O is not free.

Device mode additionally retains private device clones and pinned host buffers.
Allow at least `(device_pending_limit + 2) * device_capture_bytes` for each side,
plus allocator/alignment overhead, check temporaries, reference decoding and ring
copies. No automatic KV-cache memory reservation is applied; provide headroom and
measure actual peak usage. The per-record budget must be at least
`2 * device_capture_bytes + 16384`; this is a validation floor, not a guarantee
that arbitrarily rich metadata fits. Shutdown does not wait for unfinished NPU
copies; pending evidence can be lost. Do not enable capture near an OOM limit.

Use the same `run_id` on all DP ranks. Worker UUIDs, hostname, PID, DP rank and
model-parallel rank isolate output directories and restarts. Step numbers are
local; this version does not implement cross-rank causal ordering, globally
consistent snapshots, or identical-step aggregation.

Tokens and metadata may contain sensitive user data. Enable only with approved
storage access/retention policies. New worker directories/files request POSIX
0700/0600 permissions; Windows uses inherited ACLs, which operators must secure.

## Validation before review

Host and CPU-tensor tests can run without vLLM/NPU or repository-wide UT mocks
(the device-probe test file requires CPU PyTorch):

```bash
python -m pytest --confcutdir=tests/ut/dfx tests/ut/dfx -q
```

When vLLM is absent, this isolated suite bypasses only the package-level vLLM
logging bootstrap; recorder, configuration and reader code are not mocked.

Before requesting review, run the existing AscendConfig and MRV1/MRV2 suites, real NPU
serving/evaluation, and OFF/ON A/B throughput, TTFT and TPOT p99 benchmarks. Include
DP, graph replay, asynchronous scheduling, spec decode and CP configurations.
Verify host-only mode causes no new device copies and all modes introduce no new
device synchronization/model invocation. Device mode intentionally adds copies.
This implementation is not a substitute for those tests.

An eager/graph greedy-output parity and export smoke test is provided:

```bash
pytest tests/e2e/pull_request/one_card/test_host_flight_recorder.py -v
python benchmarks/benchmark_host_dfx.py --iterations 1000
```

The latter is a synthetic host-only microbenchmark; it excludes NPU execution,
new-request payloads, incident export contention and end-to-end serving latency.
Do not use it as proof of the throughput budget.
