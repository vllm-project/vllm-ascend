# KV Trace monitoring

KV Trace compares native scheduler declarations with the final address metadata
used by Ascend Model Runner V2, once per KV cache group. It records bindings,
write/read plans and public connector information. It is disabled by default.
Model Runner V1 and attention operators are unchanged.

## Enable KV Trace

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
vllm serve MODEL --additional-config '{"kv_trace":{"enabled":true,"action":"log"}}'
```

| `kv_trace` option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Enable the MRV2 observer for this engine. |
| `action` | `"log"` or `"raise"` | `"log"` | Log violations or stop returning model results. |

Omitting `kv_trace` or setting `enabled=false` disables the observer. Invalid
keys, non-boolean `enabled` and unknown actions are rejected. Environment
variables and `dump_config` do not enable this feature.

```python
from vllm import LLM

llm = LLM(model="MODEL", additional_config={"kv_trace": {"enabled": True, "action": "raise"}})
```

The observer installs in MRV2. It does not patch Scheduler, KVCacheManager or
BlockPool, and it does not add private attributes to SchedulerOutput.

## What is checked

The MRV2 execution entry consumes the original SchedulerOutput before the
runner updates its own request tables. New and resumed requests replace their
worker-local binding; cached requests append native block deltas. Finished and
preempted IDs remove local request references. This is a request map, not an
allocator state machine or proof of physical release.

Each step retains its own immutable binding snapshot and computed/scheduled
token ranges. The checker derives expected addresses from this declaration,
then compares the final metadata in the runner's actual batch order.

| Constraint | Example |
| --- | --- |
| Exact token destination | Cross-request slot swaps, shifted writes or non-PAD write tails. |
| Logical block order | The input read table differs from the declared request binding. |
| Declared length | The final input length disagrees with the scheduled token range. |
| Capacity and reserved storage | A write is outside cache capacity or targets a forbidden null block. |
| Dummy writes | An idle dummy has a non-PAD write target outside the explicitly allowed native null-block range. |

Warmup/capture scratch is not treated as a real scheduled request. Ordinary
dummy writes must use PAD `-1`, except when the native block table explicitly
enables null-block redirection. No per-layer access hooks are installed.

There is no allocation epoch, owners index or `valid_end`. A worker cannot
reliably distinguish prefix-cache reuse from physical reallocation of the same
block ID. Therefore this feature does not prove initialized prefixes, detect
every stale reference, establish actual allocation/free history, or diagnose
KV numerical corruption when the visible declaration and input plan agree.

## Execution paths and supported configurations

Adapted dense PA/MLA MRV2 paths use one final metadata view per cache group in
eager and FULL modes. Allocator and kernel block coordinates must match.
CP, DBO, speculative decoding, encoder/decoder and non-append cache layouts
remain unadapted in this implementation; the checker does not silently apply
the ordinary address formula to those modes.

The final metadata is frozen after the native execution returns. Ordinary
async TP1/PP1 generation retains one pending D2H snapshot and completes it at
the original result getter or before reusing the copy buffers. Other result
paths retain synchronous checking. Each plan carries its own binding snapshot,
so a later message cannot change an earlier pending audit.

Zero/copy fields are recorded when the original `update_requests` returns;
they do not establish content validity. Connector public input fields and
native finished/failed/invalid result sets are recorded without wrapping
transport methods or polling completion queues. A finished receive can also
be failed; the records preserve both fields. There is no submission-identity
or remote-content validity check.

## Read the logs

Records go through the vLLM logger to the process log:

- `kv_trace.enabled`: the configured failure action.
- `kv_trace.bind`: reconstructed request bindings, token ranges and native finish/preempt notices.
- `kv_trace.write` / `kv_trace.read`: per-group final execution plans, not kernel memory accesses.
- `kv_trace.boundary`: native update or connector public input/result fields.
- `kv_trace.violation`: an address-plan constraint violation and its expected/actual fields.

`log` records violations and continues. `raise` latches the first violation
and prevents normal result delivery; the exception can terminate EngineCore.
This does not guarantee detection before a kernel runs or undo device writes.
Missing observations are debug-level skipped checks, not violations, and do
not by themselves latch failure. No `coverage_gap` error is emitted.

## Validation and performance

Run the focused host checks:

```bash
pytest -q --confcutdir=tests/ut/kv_trace tests/ut/kv_trace
```

The suite covers native binding deltas without private scheduler attributes,
prefix sharing, actual batch order, slot/table faults, current FULL metadata,
step-local async snapshots, log/raise, dummy arguments and connector results.
These checks do not establish NPU correctness.

The MRV2 eager/FULL service regression is available for an NPU environment:

```bash
pytest -q tests/e2e/pull_request/one_card/test_kv_trace.py
```

The worker-only change has not been revalidated on NPU or benchmarked. Earlier
Scheduler-instrumented results do not validate its reduced contract. Hardware
fault injection and performance measurements must use the changed source,
fixed models/settings and normal trace logging enabled.

## Implementation guide

| File | Responsibility |
| --- | --- |
| `kv_trace_config.py` | Per-engine `additional_config.kv_trace` options. |
| `kv_trace_state.py` | Worker-local immutable request block tables from native messages. |
| `kv_trace_plan.py` | Step-local bindings, metadata snapshots and address comparisons. |
| `kv_trace_runtime.py` | Local message consumption, pending audits and output failure handling. |
| `kv_trace_runner.py` | MRV2-only installation, batch order, metadata and capture boundaries. |
| `kv_trace_connector.py` | Public connector plan and native result projection, with no lifecycle hooks. |

The feature integration is limited to MRV2 initialization and KV-cache
configuration. No scheduler patch or MRV1 integration is required.
