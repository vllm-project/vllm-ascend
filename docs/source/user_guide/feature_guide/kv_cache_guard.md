# KV cache address guard

KV Guard diagnoses address mappings that can cause cross-request writes,
prefix overwrites, or reads from invalid KV ranges. It compares scheduled
destinations with execution metadata and observed operator arguments.
The feature is disabled by default and supports adapted dense PA and MLA
attention paths in Model Runner V1 and V2.

## Enable the guard

Use the same per-engine configuration for Model Runner V1 and V2:

```bash
vllm serve MODEL --additional-config '{"kv_guard_config":{"enabled":true}}'
```

| `kv_guard_config` option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Enable KV address diagnostics for this engine. |
| `action` | `"log"` or `"raise"` | `"log"` | Report failures or stop returning worker results. |

`action="log"` reports failures while preserving diagnostic execution. To stop
returning worker results after a violation or incomplete audit, select `raise`:

```bash
vllm serve MODEL --additional-config '{"kv_guard_config":{"enabled":true,"action":"raise"}}'
```

For the Python API, pass the same dictionary through `additional_config`:

```python
from vllm import LLM

llm = LLM(model="MODEL", additional_config={"kv_guard_config": {"enabled": True, "action": "raise"}})
```

The scheduler and workers read this configuration independently for each engine.
Omitting `kv_guard_config` or setting `enabled=false` disables instrumentation.
Invalid option names, non-boolean `enabled` values, and unknown actions are rejected.
KV guard environment variables are no longer used.

The first failure is latched. Catching the exception does not reset the worker;
restart it before reuse. This is a worker-level stop, not cancellation of one
request, and it adds no collective failure coordination across ranks.

## What is checked

An allocated block is not necessarily the correct destination for a token.
For example, if requests A and B own blocks 1 and 2, swapping their write slots
still points to allocated memory. Ordinary allocation logs cannot establish
that each token used its intended destination.

The scheduler supplies changed request-to-block bindings, allocation generations,
released requests, and each request's computed and scheduled token counts through
the existing scheduler output. The worker independently derives the expected
slot of every scheduled token from those bindings and the actual batch order.

| Constraint | Failure example |
| --- | --- |
| Exact token destination | Slots are swapped between requests or between tokens in one request. |
| Allocation and ownership | A real request accesses another request's block or a recycled allocation. |
| Logical block order | A request reads its own blocks in the wrong logical order. |
| Initialized prefix | A read extends into unwritten data, or a write overwrites a valid prefix. |
| Reserved storage | A write targets the null block instead of negative-slot padding. |
| Collection completeness | A scheduled write, binding, layer, or current graph plan is missing. |

Each tracked block stores an allocation generation, a valid prefix length, and
request owners. A zero prefix is unwritten; a positive prefix N makes `[0, N)`
readable; an invalid prefix prevents subsequent reads from being accepted.
Completed loads and contiguous appends establish readable prefixes. Completion
and preemption release ownership while preserving reusable cached prefixes.
The scheduler's computed-token count describes a consumer's read requirement;
it does not establish validity. A same-step prefix producer must first supply
legal write evidence, while an already valid prefix still rejects overwrites.
Physical clearing, block copies, and completed receives update validity at their
existing lifecycle boundaries. The scheduler records external-load destinations,
allocation generations, and logical token ranges when slots are allocated,
including requests that are still waiting and have no scheduled tokens.
Connector transport offsets are not interpreted as initialized-prefix lengths.

The common connector methods provide completion evidence. Async receive results
authorize later forwards. Sync loads become readable for a cache group only
after its required layer waits; a missing wait reports incomplete coverage.
FULL replay without observable sync-load completion also reports incomplete
coverage. No extra connector polling or device waits are introduced.
Failed loads discard only their destination blocks, preserving untouched local
prefix blocks. A failed transfer can have overwritten the beginning of a partial
destination block, so that entire block must be recomputed.
Block failures remain attached to their submission until its completion is
drained, even if the connector reports failures and completion in different steps.
Completions must match the submission snapshot and current allocation. Reusing
a request ID while its old receive is outstanding reports incomplete coverage,
because the connector's request-ID-only result cannot distinguish both loads.
Known old submissions may drain after cancellation without changing a new
allocation or treating cancellation itself as a guard failure.
Framework-authorized prefix replay truncates the specified validity range before
write auditing; an ordinary prefix overwrite still fails. Transport drain and
completion results are returned unchanged.

For example, with block size 128 and a loaded 19-token prefix in block 1, the
next token should write slot 147. A dummy write to slot 142 overwrites offset 14;
the guard diagnoses the write and marks the block invalid for subsequent reads.
This example illustrates the metadata checks, not reproduction of a particular
full-model incident.

## Execution paths and supported configurations

| Path | Audit evidence and boundary |
| --- | --- |
| Eager | Ordered per-layer operator arguments, with adapted metadata plans where applicable. |
| FULL replay | The current execution's final attention metadata, inspected after forward and graph parameter update calls. Captured or previous-step plans are not reused as evidence. |
| Runner V1 | Bind before `execute_model`; collect through the existing forward context and attention hooks. |
| Runner V2 | Bind before execute; obtain actual batch IDs and metadata after `model_state.prepare_attn`; audit at the end of execute. Persistent request-state indices are not batch positions. |
| Dummy/profile execution | Check writes against existing KV state; ignore discarded scratch reads. |
| Runner V2 startup capture | Require empty request/cache state; do not audit CPU metadata inside device graph capture. |

Canonical FULL plans can prepare a CPU validity transaction while the final
metadata copies complete. The transaction is committed only when every actual
plan exactly matches the independent scheduler-derived plan and the tracked
state is unchanged. Other plans and suspected violations use the complete audit.
This optimization preserves the audit and failure boundary, including its
device-completion wait.

Live or cache-preserving Runner V2 recapture is rejected, including pending
receives. Normal FULL replay remains audited. Allocator and kernel block
coordinates must match.

The guard explicitly rejects CP, DBO, speculation, encoder/decoder execution,
sliding or circular caches, other compressed or state-cache semantics, unknown
attention implementations, and connectors other than Mooncake. These paths
require their own validity rules before they can be supported.
The Ascend block-transfer implementation is registered as `MooncakeConnectorV1`;
this name and the legacy `MooncakeConnector` name pass the guard's connector check.
The shared load adapter is not an expansion of this connector support list;
other backends still require end-to-end validation before enabling them.

The guard validates metadata and observed arguments. It does not inspect kernel
memory instructions, verify KV numerical values, or detect a kernel writing to
another address despite correct arguments. It cannot prevent or undo submitted
writes, prove transport byte integrity, or guarantee model output correctness.

## Read the logs

Logs use the `[KV_GUARD]` prefix:

- `guard.enabled`: the configured failure action.
- `guard.violation`: an observed constraint violation, with an operation and
  reason such as `write_target_mismatch`, `overwrite_valid_prefix`, or
  `read_unwritten_range`.
- `guard.coverage_gap`: required evidence was missing, so the audit was
  incomplete. This is not a successful check or an exemption from validation.

Runner V2 may submit async loads during `sample_tokens` post-forward cleanup.
Load submission and completion are checked at the connector cleanup boundary,
after the separately audited `execute_model` forward.

Fields identify the layer, request, pool, cache group, block, allocation
generation, and relevant slot or range when available. Repeated failures are
deduplicated with bounded state. Routine validity transitions are not logged.

## Validation and performance

The focused CPU suite exercises allocation generations, ownership, shared
prefixes, exact destinations, read/write order, graph metadata freshness,
failure latching, and the V1/V2 adapters:

```bash
pytest -q tests/ut/debug/test_kv_guard.py \
  tests/ut/debug/test_kv_guard_runtime.py tests/ut/debug/test_kv_guard_v2.py \
  tests/ut/debug/test_kv_guard_loads.py
```

The single-card service regression starts separate disabled and enabled servers,
uses `raise`, repeats a shared prompt, submits concurrent requests, and compares
greedy outputs for both runners in eager and FULL modes:

```bash
pytest -q tests/e2e/pull_request/one_card/test_kv_cache_guard.py
```

CPU mocks establish checker behavior, not NPU execution correctness. Before
merge, record current-revision hardware runs for V1/V2 eager and FULL, prefix
reuse, observed preemption, Mooncake where used, and controlled fault injection.
The service regression does not itself prove that a preemption or prefix-cache
hit occurred; collect those metrics separately. Historical V1 results do not
validate a rebased revision or Runner V2.

Benchmark disabled and enabled configurations on the same hardware, model,
request trace, sequence lengths, concurrency, seed, and graph mode. Record
successful requests, throughput, TTFT, ITL, CPU usage, and run-to-run variation.
Fix the source commits, actual module load paths, software versions, exact
commands, and logs in the validation report. Do not infer device synchronization
cost from CPU checker timings.

The scheduler sends changed block/generation tables and compact request ranges.
Expected token slots are temporary and shared across layers in a cache group;
the guard stores no persistent token history, KV snapshots, or content hashes.
Metadata transfers and access-pattern checks are deduplicated, but enabled mode
can still synchronize device metadata to CPU. Measure deployment overhead.

## Implementation guide and patch lifecycle

| Component | Responsibility |
| --- | --- |
| `vllm_ascend/debug/kv_guard.py` | Allocation, ownership, prefix validity, and read/write rules. |
| `vllm_ascend/debug/kv_guard_config.py` | Validated per-engine options shared by the scheduler and workers. |
| `vllm_ascend/debug/kv_guard_runtime.py` | Shared V1/V2 installation, scheduler binding, evidence collection, audit, and conservative state commit. Includes the V2 batch and capture adapter. |
| `vllm_ascend/patch/platform/patch_kv_guard.py` | Conditional scheduler allocation and request-binding instrumentation. |
| Runner and attention implementations | Existing execution boundaries and ordered eager access arguments. |
| Mooncake receiver | Success/failure validity callbacks after receive completion and reformatting. |

Both runners use one `GuardRuntime` and one address-checking state machine.
The internal V2 adapter only handles sorted batch IDs, metadata timing, and
capture boundaries; it does not implement another checker.

The scheduler constructor is wrapped once; allocation, release, and schedule
methods are instrumented only for engines with `kv_guard_config.enabled=true`.
Current public hooks do not provide both allocation generations and the
independent scheduler-to-worker request contract. Its removal path is an upstream allocation-lifecycle and scheduler
output extension that supplies the same evidence without wrapping instance
methods. Changes to those upstream contracts require adapter and lifecycle
regressions, alongside NPU tests for affected execution paths.

This feature's current scope is address-legality diagnostics. Content tracing,
KV dumps, automatic recovery, and additional cache semantics are separate work.
