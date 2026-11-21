# KV Trace

KV Trace is an opt-in diagnostic for investigating KV-cache address-plan errors
in Ascend Model Runner V2 (MRV2). It compares native scheduler declarations with
the target model's final slot mappings, block tables and sequence lengths.
Shared metadata is recorded once; distinct views in a cache group are checked
separately. It is disabled by default.

Use it when you need to check whether a request's scheduled token range and KV
block bindings agree with the runner's final address metadata. An in-range
address can still belong to the wrong token or request; KV Trace checks the
exact destinations as well as capacity.

## Requirements and Limitations

- Use a vLLM/vLLM Ascend combination that includes KV Trace, with MRV2 enabled.
- The collector follows scheduler input, final attention inputs and
  result delivery. Enabling it does not reject an otherwise valid inference
  configuration merely because a particular address check is unavailable.
- Ordinary paged addressing supports allocator/kernel block conversion and DCP
  rank/interleave coordinates. PCP checks the native global plan and its local
  request/token, gather and write-mask projection. Dynamic target batches check
  observed contiguous positions against the scheduler's token budget. PCP uses
  global request IDs and native local mappings; repeated local request rows do
  not disable the global scheduler comparison.
- With speculative decoding enabled, KV Trace checks the target model's
  verification batch in eager and graph execution. During trace configuration,
  a `kv_trace.check` record with `status=unavailable`, `role=draft` and
  `reason=draft_address_not_in_scope` marks draft addresses as outside coverage.
- DCP local-read checks compare the native length fields consumed by the selected
  full-cache or split history/current attention path. Prefill cache-load and
  cumulative-history lengths are checked where those consumers are adapted.
  The global PCP view is checked for global addresses and does not require
  local-length fields.
- Bounded-history, compressed, circular and recurrent-state caches (including
  GDN/Mamba state storage) retain observations with `not_applicable` address
  checks. Encoder/decoder address
  semantics likewise remain observation-only. Collected metadata without a
  native request-row association retains plan-only observations without
  borrowing the target's request order. Unadapted microbatch boundaries
  report `unavailable`. Inspect the check status for each observation.
- Target batches with native request associations retain public request-binding,
  token-budget and query-attribution checks for observation-only storage. Its
  storage-specific address comparisons remain `not_applicable`.
- The current Ascend platform resets `--enable-dbo` to false and a requested
  microbatch size to zero, with a startup warning. Successful serving with those
  requested options does not establish that dual batch overlap (DBO) ran.
- Historical validation results below apply to their recorded revisions.
  PCP/DCP, speculation, additional layouts and multi-rank combinations need their own NPU acceptance
  and performance measurements before production qualification.
- Trace evidence covers worker-local request bindings and final address plans.
  Block lifetime, initialized prefixes, KV values and transfer integrity require
  separate observations. Prefix sharing and reuse of the same physical block ID
  can produce the same worker messages.
- Historical fixed-workload measurements showed at most 0.66% throughput
  decrease with that revision's INFO logging. See the performance section for the tested
  revision, workload and first-token latency.

## Usage

### Configuration

| `kv_trace` option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Enable the MRV2 observer for this engine. |
| `slot` | bool | `true` | Check write slot mappings, including bounds, padding and PCP write projections. |
| `action` | `"log"` or `"raise"` | `"log"` | Log violations or stop returning model results. |

Omitting `kv_trace` or setting `enabled=false` disables the observer. Accepted
keys are `enabled`, `slot` and `action`; their types and values follow the table above.
Configure the feature through `additional_config.kv_trace`, and select MRV2 with
`VLLM_USE_V2_MODEL_RUNNER`.

Set `slot=false` to skip write slot comparisons while retaining write/read logs,
request attribution and block-table/sequence-length checks. Skipped write checks
report `not_applicable` with `reason=slot_disabled`; read checks remain active.
The default `slot=true` preserves the existing checks and does not enable KV Trace
when `enabled=false`.

### Online Serving

Replace `/path/to/model` with a supported dense model and retain your deployment's
other required options. The following example enables detailed logging in FULL
mode for the plan-analysis examples below. Set the environment before starting
the service:

```bash
export MODEL=/path/to/model
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_LOGGING_LEVEL=DEBUG
vllm serve "$MODEL" --served-model-name test-model \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --max-model-len 1024 --max-num-seqs 4 --max-num-batched-tokens 256 \
  --block-size 128 --enable-prefix-caching --enable-chunked-prefill \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4]}' \
  --additional-config '{"kv_trace":{"enabled":true,"slot":true,"action":"log"}}' \
  2>&1 | tee kv-trace-server.log
```

`tee` saves standard output and standard error to `kv-trace-server.log` in the
current directory. KV Trace writes to the vLLM process logger; `tee` creates
and saves the log file.

Use `VLLM_LOGGING_LEVEL=INFO` for normal check summaries and violations. Detailed
binding, write, read and boundary records require DEBUG.

For eager execution, replace `--compilation-config ...` with `--enforce-eager`.
To stop normal output after a violation, change `action` to `raise` and restart
the service. A `raise` failure can terminate EngineCore and require a complete
engine restart.

### Offline Inference

Select MRV2 in the environment before starting the Python process:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

```python
from vllm import LLM

llm = LLM(
    model="/path/to/model",
    additional_config={"kv_trace": {"enabled": True, "slot": True, "action": "raise"}},
)
```

## Log Output and Analysis

### Log Records

Records go through the vLLM logger to the process log:

- `kv_trace.enabled`: the configured failure action and `slot` switch.
- `kv_trace.bind`: reconstructed request bindings, token ranges and native finish/preempt notices.
- `kv_trace.write` / `kv_trace.read`: per-group final execution plans.
- `kv_trace.boundary`: native update or connector public input/result fields.
- `kv_trace.violation`: an address-plan constraint violation and its expected/actual fields.
- `kv_trace.check`: `checked`, `not_applicable`, or `unavailable`, with the check
  or reason and its step/group/view association.

`checked` means that the named comparison ran; inspect `kv_trace.violation`
for mismatches. It does not mean that every cache behavior was validated.

Records within a scheduler step carry `step_id` and available rank coordinates.
Plan records identify `role`, `request_ids`, `group_id` and `view_id`; separate
attention groups can therefore be distinguished without per-layer hooks. A
`target.global` role is the global PCP plan, while `target` is its local view.
During speculative decoding, verification batches appear in `role=target` records.
A configuration record with `reason=draft_address_not_in_scope` applies to
draft addresses. Inspect the `role=target` check records for verification
results.

`kv_trace.check check=dcp_local_reads` lists the native length fields compared.
A `dcp_read_length_mismatch` violation identifies the affected `field` and
includes `expected_lengths`, `actual_lengths` and their `request_ids` row order.
Length fields follow the consumer's request order and local DCP rank. With PCP,
prefill-history records use the original global prefill request order, which can
differ from the local target order. A step without cached history can report
`not_applicable` with reason `no_cached_history` for this comparison.

`log` records violations and continues. `raise` latches the first violation
and stops normal result delivery; the exception can terminate EngineCore.
An audit can complete after native execution has submitted device writes; those
writes remain in place. Observation gaps produce a `kv_trace.check` record with
status `unavailable`. A recorded plan with an unavailable check is not a
successful address audit. Only
`kv_trace.violation` latches a failure and uses ERROR.
`kv_trace.enabled` and `kv_trace.check` use INFO; detailed binding,
write, read and boundary records use DEBUG. Set `VLLM_LOGGING_LEVEL=DEBUG` to
include their complete fields. The existing vLLM logging level controls visibility.

### Address Checks

The expected addresses come from each step's native request bindings and
computed/scheduled token ranges. KV Trace compares them with the final metadata
in the runner's actual batch order.

| Constraint | Example |
| --- | --- |
| Exact token destination | Cross-request slot swaps, shifted writes or writes into the padded tail. |
| Logical block order | The input read table differs from the declared request binding. |
| Declared length | The final input length differs from the scheduled token range. |
| Capacity and reserved storage | A write exceeds cache capacity or violates the configured null-block reservation. |
| Dummy writes | A dummy write target violates the PAD/native null-block rules. |
| Query attribution | Query boundaries or contiguous positions disagree with the declared request budget. |
| Parallel projection | A PCP local request/token, gather or write-mask mapping differs from its native global plan. |
| DCP local reads | Native full-cache, history/current or prefill consumer lengths disagree with the declared range. |

Warmup/capture scratch belongs to synthetic execution. Ordinary dummy writes
use PAD `-1`; native block tables can explicitly enable null-block redirection.

### Observation Boundaries

Bindings record native new/resumed replacements and cached block appends.
Finished/preempted notices remove worker-local request references. Verify
physical block release separately through allocator-side evidence.

Target read/write records describe the final metadata after native execution.
With speculative decoding, target checks compare the verification batch with
its scheduled range. Sampling decisions, accepted-token counts and computed-token
rollback require separate verification.

In asynchronous generation without speculation, the audit can finish during
result retrieval or before the next metadata observation when TP1/PP1,
PCP1/DCP1, no connector, no DBO and a generative model are in use. Other result
paths, including speculative generation, use synchronous checking.
Each audit freezes the request bindings and observed values for its own step.

Native multi-step autoregressive drafting now saves the accepted target end
before building CPU attention metadata or graph task parameters. This fixes
host read lengths that previously retained rejected tokens. The correction also
applies with tracing disabled when DCP or GQA/MLA metadata needs host lengths,
and copies one length per active request from the device to the host once per
multi-step proposal; its performance cost needs separate qualification.
DSA/SFA/QSA without DCP skip this copy. Single-step proposals and synthetic
capture retain their existing length path.

Native PCP restrictions still apply independently of Trace: the pinned vLLM
revision accepts `mtp`, `eagle3`, `dspark` and `dflash`, permits
`FULL_DECODE_ONLY`, and rejects complete `FULL` graphs. Inspect each target
record's status and role for address-check coverage.
The same baseline rejects PCP for the `GDN_ATTN` backend used by Qwen3.5 hybrid
models, including with Trace disabled. Single-rank MTP results on those models
do not establish PCP support.

Zero/copy fields record the native `update_requests` return boundary. Confirm
content readiness separately. Connector records retain public input fields and
the native finished/failed/invalid result sets. A receive can appear in both
the finished and failed sets, so inspect those fields together. Submission
identity and remote-content validity require separate connector evidence.

## Troubleshooting

### Find the Violation

Use the saved process log from the online example. The following example came
from controlled fault injection with `action=raise`.

In a controlled read-table fault run on Ascend 910B4 at revision
`11ba7b73079bda403be5c795f1d6d9c1a43045d4`, Qwen3-0.6B BF16 with MRV2 FULL,
TP1/PP1 returned HTTP 500 with this client error:

```json
{"error":{"message":"The server had an error while processing your request.","type":"InternalServerError","param":null,"code":500}}
```

The service then exited, and a subsequent health request reported
`ConnectionError`. Use the server's violation and exception records to identify
the cause of an HTTP 500. Search the saved log:

```bash
grep -n -B 6 -A 3 -E 'kv_trace\.violation|KVTraceError|KV Trace stopped model output' kv-trace-server.log
```

The same run produced the following records. Timestamps, log levels and source
locations are omitted; process IDs and event fields are retained. In the current
version, the detailed `kv_trace.read` record requires DEBUG logging:

```text
(EngineCore pid=10248) [KV_TRACE] kv_trace.violation operation=read reason=read_target_mismatch group_id=0 phase=forward evidence=execution_plan expected_tables=[[1, 2, 3, 4]] actual_tables=[[2, 1, 3, 4]]
(EngineCore pid=10248) [KV_TRACE] kv_trace.read group_id=0 request_ids=['cmpl-FULL-read-raise-observed-0-bceb1d2c'] evidence=execution_plan full_graph=True block_table=[[2, 1, 3, 4]] seq_lens=[466]
vllm_ascend.observability.kv_trace.kv_trace_runtime.KVTraceError: KV Trace stopped model output: read_target_mismatch; restart the worker before reuse
```

Here group 0 expected the logical block order `[1, 2, 3, 4]` but observed
`[2, 1, 3, 4]`. The first two block IDs were swapped. All IDs can be within
capacity while the read plan targets the wrong blocks. This record establishes
an address-plan mismatch. KV values and physical allocation behavior require
separate verification.

### Identify the Affected Request

Request and step correlation uses the surrounding trace records. In this
single-request example, the following `kv_trace.read` record from the same
EngineCore identifies the request. Use the process and request IDs from your
own log:

```bash
grep -nF 'EngineCore pid=10248' kv-trace-server.log \
  | grep -E 'kv_trace\.(bind|write|read|violation)'
grep -nF 'cmpl-FULL-read-raise-observed-0-bceb1d2c' kv-trace-server.log
```

For concurrent requests, use the actual `request_ids` row order with the
expected/actual table rows and retain the surrounding records. An async audit
can follow a later `kv_trace.bind`, so the nearest bind may describe a later
step. For ambiguous request or step attribution, report the candidate records
and the remaining uncertainty.

### Restart and Verify Recovery

In `raise` mode the failure remains latched and can stop the complete service.
Preserve the log before restarting:

```bash
cp kv-trace-server.log "kv-trace-failure-$(date +%Y%m%d-%H%M%S).log"
```

Remove any test fault-injection hook before retrying. For an unexpected failure,
retain the model, software revisions, launch options, triggering workload and
full log for diagnosis. Restart the complete engine/service through your normal
process manager after the old process exits, preserving the original model,
execution settings and failure action. When retrying a `raise` failure, keep
`action=raise` in the launch command above. Start a fresh engine with a fresh
KV cache. The `action` setting controls the response to a violation; recovery
requires investigating and correcting the address mismatch.

In another terminal, check readiness and retry a small request:

```bash
curl -fsS --retry 30 --retry-connrefused --retry-delay 2 \
  --max-time 5 --retry-max-time 120 \
  -w 'health HTTP %{http_code}\n' http://127.0.0.1:8000/health
curl -sS --fail-with-body --max-time 60 http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test-model","prompt":"Continue: 1, 2, 3,","temperature":0,"max_tokens":8}' \
  -w '\ncompletion HTTP %{http_code}\n'
grep -nE 'kv_trace\.(enabled|check|violation)|KVTraceError' kv-trace-server.log
```

Confirm recovery using three results: health HTTP 200, completion HTTP 200 with
generated text, and zero new violations for the retry. In the fresh server log,
check `kv_trace.enabled` and the `kv_trace.check` summaries for that retry.
DEBUG logging also retains its `kv_trace.write` / `kv_trace.read` details.
The final `grep` reports trace and exception records; check readiness
and request results through the two HTTP responses. The archived evidence
covers detection and service exit. Validate the restart procedure in your
deployment.

## Validation and Performance

### Functional Validation

The repository retains the existing KV Trace regression suite. The service
results below apply to their recorded revisions.

Earlier [follow-up validation](https://github.com/vllm-project/vllm-ascend/pull/22071)
used product sources `473d86c30c78` (delivery `d56026624108`) for native
group/DCP cases, and `79d27b914b4f` (delivery `9ca33f72df80`) for image and Whisper
cases. Real Qwen3 serving compared a controlled native two-group/four-view KV
layout with the original layout and tracing OFF/ON. PCP2+DCP2 runs checked native
local-read lengths in eager and FULL decode-only modes; injected decode/prefill
consumer-length faults exercised both `log` and `raise`. DBO-flag compatibility
tests confirmed native DBO disabling, without overlapping microbatch execution.

These historical cases predate the current simplification and do not establish
its NPU correctness or performance. The manual MTP runs used Qwen3.5-2B.
Native extensions and custom OPP were reused from a v0.30.0 image. Complete the
fully rebuilt pinned-stack and performance qualification for your deployment.

At revision
[`11ba7b73079b`](https://github.com/vllm-project/vllm-ascend/commit/11ba7b73079bda403be5c795f1d6d9c1a43045d4),
[service validation](https://github.com/vllm-project/vllm-ascend/actions/runs/43542402730/attempts/2)
passed both eager and FULL cases, comparing tracing OFF/ON for sequential and
concurrent generation with prefix caching and chunked prefill.

A separate 2026-11-14 Ascend 910B4 run used the same revision's Python feature
sources, vLLM `24420b7a341af37a9bc3186e6fc6062512e0aa58`, and native extensions
and custom OPP from a v0.30.0 image. With Qwen3-0.6B BF16, MRV2, TP1/PP1:

- 32 complete healthy requests passed across eager/FULL and tracing OFF/ON,
  covering sequential, four-way concurrent and streaming generation, plus
  retries after client stream cancellation. Four intentionally cancelled
  streams are counted separately. Paired text/stream outputs matched and each
  healthy engine recorded 2,688 prefix-cache hit tokens.
- Eight controlled write-slot/read-table fault requests produced the expected
  violations: four `log` requests returned HTTP 200; four `raise` requests
  returned HTTP 500 and the service exited. The final two FULL read-table cases
  also recorded the native graph-replay call and return.

These results cover controlled address-plan faults. Production incidents,
performance, MLA, multi-rank TP/PP, Mooncake network transfer, preemption and
long-term stability each require separate validation.

### Historical Performance Measurements

Performance measurements belong to revision
[`2883638529ac`](https://github.com/Tjh-UKN/vllm-ascend/commit/2883638529ac02c9ee1ef29a41a8158da9d0782a),
before the package rename and CI follow-ups. Repeat the benchmark when
evaluating later revisions, including `11ba7b73079b`.

The run used Ascend 910B4, Llama-3.1-8B BF16, MRV2 FULL, TP1/PP1, input/output
lengths 2048/256, prefix caching disabled and normal INFO trace logging enabled.
That revision emitted detailed plan records at INFO. The current INFO/DEBUG
split requires separate measurements at the selected logging level.
Eight fresh engines, four OFF and four ON, completed all 520 requests across
16 formal windows, each longer than 60 seconds.

| Concurrency | Throughput decrease | TTFT increase |
| --- | --- | --- |
| 1 | -0.11% | 4.74% |
| 8 | 0.66% | 5.10% |

TTFT is time to first token. Treat the -0.11% throughput change as measurement
variation. Results apply to this revision and workload; repeat the benchmark
for later revisions. CPU/RSS cost requires separate measurement.
