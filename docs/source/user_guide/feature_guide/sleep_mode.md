# Sleep Mode Guide

## Overview

Sleep Mode is an API designed to offload model weights and discard KV cache from NPU memory. This functionality is essential for reinforcement learning (RL) post-training workloads, particularly in online algorithms such as PPO, GRPO, or DPO. During training, the policy model typically performs autoregressive generation using inference engines like vLLM, followed by forward and backward passes for optimization.

Since the generation and training phases may employ different model parallelism strategies, it becomes crucial to free KV cache and even offload model parameters stored within vLLM during training. This ensures efficient memory utilization and avoids resource contention on the NPU.

## Getting started

With `enable_sleep_mode=True`, the way we manage memory (malloc, free) in vLLM is under a specific memory pool. During model loading and KV cache initialization, we tag the memory as a map: `{"weight": data, "kv_cache": data}`.

The engine (v0/v1) supports two sleep levels to manage memory during idle periods:

- Level 1 Sleep
    - Action: Offloads model weights and discards the KV cache.
    - Memory: Model weights are moved to CPU memory; KV cache is forgotten.
    - Use Case: Suitable when reusing the same model later.
    - Note: Ensure sufficient CPU memory is available to hold the model weights.

- Level 2 Sleep
    - Action: Discards both model weights and KV cache.
    - Memory: The content of both the model weights and KV cache is forgotten.
    - Use Case: Ideal when switching to a different model or updating the current one.

Since this feature uses the low-level API [AscendCL](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/API/appdevgapi/appdevgapi_07_0000.html), in order to use sleep mode, you should follow the [installation guide](https://docs.vllm.ai/projects/ascend/en/latest/installation.html) and build from source. If you are using < v0.12.0rc1, remember to set `export COMPILE_CUSTOM_KERNELS=1`.

## Optional extra cleanup

By default, sleep mode only releases memory managed by the sleep-mode allocator. For RL workloads that need to return more NPU memory to the trainer, vLLM Ascend also provides an optional extra cleanup path:

```python
llm = LLM(
    "Qwen/Qwen2.5-0.5B-Instruct",
    enable_sleep_mode=True,
    additional_config={
        "rl_config": {
            "enabled": True,
            "sleep_mode_extra_cleanup": True,
        }
    },
)
```

For online serving, pass the same option through `--additional-config`:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --enable-sleep-mode \
    --additional-config '{"rl_config": {"enabled": true, "sleep_mode_extra_cleanup": true}}'
```

When `rl_config.sleep_mode_extra_cleanup` is enabled, `sleep()` additionally:

- clears ACL graph attention workspaces and invalidates captured ACL graph caches when ACL graph is enabled;
- resets the model runner graph manager so ACL graphs can be captured again after wakeup;
- waits for pending pipeline-parallel send work, synchronizes the NPU, and destroys HCCL process groups.

During `wake_up()`, vLLM Ascend restores the HCCL process groups, refreshes MoE dispatcher HCCL metadata, restores sleep-mode allocator memory, and recaptures ACL graphs when needed.

!!! note

    Extra cleanup trades lower sleep-time NPU memory usage for longer wakeup latency. By default, if ACL graph is enabled, `wake_up()` must call `capture_model()` again after the model state has been restored. The experimental graph restoration option below can avoid this recapture for eligible graphs. The former top-level `enable_sleep_mode_extra_cleanup` option has been removed.

For level 2 sleep, wakeup can be split into two phases:

```python
llm.wake_up(tags=["weights"])
# Reload or update model weights here.
llm.wake_up(tags=["kv_cache"])
```

With extra cleanup enabled, ACL graphs are recaptured only when `tags` is `None` or contains `"kv_cache"`. This avoids recapturing graphs before externally reloaded weights and KV-cache state are ready.

## Experimental A5 MRV2 graph restoration

On shutdown, the reuse worker retires its graphs and explicitly releases managed
physical allocations before Python finalization. Late tensor owners retain only
virtual allocation records until their normal free callbacks run. Shutdown is
terminal: inference and wake-up require a new worker afterward.

Worker initialization and compilation preserve the caller's SIGTERM/SIGINT
handlers across lazy backend imports. Some torch-npu backend versions replace
these handlers at import time, preventing the default single-card EngineCore
from entering its shutdown protocol. Backend atexit cleanup remains registered.

Resource teardown and request draining have separate deadlines. On the tested
A5 environment, device finalization took longer than the default best-effort
process exit window, with graph reuse both enabled and disabled. Offline
validation sets the existing vLLM `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=35`
before starting the engine and explicitly waits up to 45 seconds with
`llm.llm_engine.engine_core.shutdown(timeout=45)` on the pinned vLLM revision.
This internal engine-client call is a validation cleanup step, not a new public
API. Record actual exit times and process exit codes; extending a deadline
alone does not establish successful cleanup. Use the serving application's
supported shutdown lifecycle in production.

Set `additional_config.rl_config.sleep_mode_graph_reuse=True` to enable this
experimental path. It defaults to `False` and requires `enable_sleep_mode=True`,
`rl_config.enabled=True`, and `rl_config.sleep_mode_extra_cleanup=True`.
Select Model Runner V2 with `VLLM_USE_V2_MODEL_RUNNER=1` before starting the engine.

The initial support boundary is A5, Model Runner V2, unquantized BF16 Qwen2,
Qwen3, or Qwen3 MoE with ordinary Ascend attention, `FULL_DECODE_ONLY`, and fixed
capture sizes drawn from `[1, 2, 4, 8]`. Tensor parallelism is limited to one or
two ranks, with expert parallelism disabled. MLA, multimodal models, speculative
decoding, LoRA, PP/CP/DP, weight offload, transfer connectors, dynamic EPLB, and
super kernels use recapture. Other attention backends and model families also
use recapture; the decision log includes the reason.

These model-family names describe eligibility rules. The full-model experiments
use Qwen2.5-0.5B-Instruct and Qwen3-30B-A3B; they do not validate every checkpoint
in those families. Dense Qwen3 is eligible but was not separately exercised in
this campaign.

```python
llm = LLM(
    "Qwen/Qwen2.5-0.5B-Instruct",
    dtype="bfloat16",
    enable_sleep_mode=True,
    compilation_config={
        "mode": 3,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 2, 4, 8],
    },
    additional_config={
        "rl_config": {
            "enabled": True,
            "sleep_mode_extra_cleanup": True,
            "sleep_mode_graph_reuse": True,
        },
    },
)
```

Eligible captures use a dedicated CaMem pool. Sleep copies the complete managed
graph pool to pinned CPU memory and unmaps its physical memory, preserving the
virtual addresses. The graph objects, bound events, and business HCCL groups
remain alive. Consequently, sleep retains communication and other unmanaged
memory, and uses additional CPU memory for the graph backup. This is a latency
and memory tradeoff; it does not release all graph or communication resources.

Every rank checks the capture configuration, tensor layouts and addresses,
graph/event ownership, pool records, and communicator identities before sleep
and again after restoration. Same-address weight updates are allowed. Changed
storage, events, or configuration invalidate the certificate, and all ranks
recapture together before inference resumes. Known graph-only restoration
failures also recapture. An unknown native mapping state fails the worker and
requires restart; the old graph is never replayed in that state.

The transaction moves from `READY` to `ASLEEP`, optionally `PARTIAL`, then
`RESTORING`, and finally `READY`. Inference is rejected while the transaction
is incomplete. Repeating sleep at the same level or wake after complete restore
is idempotent. Level 2 supports the `weights` / external weight reload /
`kv_cache` sequence above: restoring weights alone never enables replay.
Control transitions and worker inference entry points share a lock; device
synchronization drains work queued before sleep.

To return to the existing cleanup and recapture behavior, set
`sleep_mode_graph_reuse=False` and start a new engine. No additional HTTP
endpoint is introduced.

### Resource dependencies and reuse certificate

The certificate is implemented by `graph_resource_signature()` in
`vllm_ascend/device_allocator/graph_reuse.py`. It compares object ownership,
addresses and layouts, not weight values. This allows an in-place weight update
while rejecting a replacement storage. All ranks vote before sleep cleanup and
again before graph replay; one incompatible rank makes every rank recapture.

| Dependency | Eligible sleep | Restore and compatibility check |
|---|---|---|
| Weights and model buffers | Keep virtual reservations; level 1 backs up weights, level 2 discards them and saves required model buffers | Restore at the original address; compare parameter/buffer identity, address, shape, stride, dtype and storage offset. Reload level-2 weights before the final wake. |
| KV cache, input buffers, block tables and runner state | Retain owners and virtual reservations; KV contents are discarded | Restore mappings and runner state; compare tensor addresses and layouts. |
| Final decode graph pool | Back up the complete CaMem pool to CPU and release physical mappings; retain the dedicated allocator and pool | Restore contents and addresses; compare pool ID and managed allocation records. Profiling captures do not establish final-capture eligibility. |
| Graphs, captured tensors and attention workspaces | Retain graph objects and necessary storage; managed storage sleeps with its pool | Compare graph descriptors/identity, captured-resource tensors, input/output tensors and attention graph parameters. A replacement workspace requires recapture. |
| Bound task handles and events | Keep strong references for the sleep transaction | Compare task handle/event identities; replacing an event invalidates reuse. The code does not patch opaque native graph dependencies. |
| Business HCCL groups | Retain required GroupCoordinators and device groups | Compare coordinator/device-group identities. Strong references prevent destruction followed by Python-ID reuse from passing the certificate. |
| Capture configuration | Record runner/model/manager identity, graph mode, compiler backend, capture sizes, dtype, attention backend and parallel settings | Recheck eligibility and compare the complete certificate after restoration. |

CaMem reports managed mapped/backup bytes. HCCL, native events and backend
workspace allocations cannot be separated reliably through the public APIs used
here. Report their visible owner counts, logical workspace capacity and aggregate
sleep HBM; do not attribute aggregate HBM differences to individual resources.
CPU backup bytes mean live backups, not process RSS or the pinned-memory cache
peak.

### Native free-callback failures

The pluggable allocator's void free callback can run through a tensor context's
destructor. An exception escaping that callback can terminate the process; C
linkage by itself is not the reason. The native callback therefore catches
exceptions and records a sticky terminal failure. Merely logging and continuing,
or unconditionally deleting a record after a failed driver operation, would
conceal outstanding allocation ownership.

CaMem uses `init_module(malloc, lookup, complete)`: after synchronization, lookup
moves the allocation from the active `pointer_to_data` inventory to a separate
pending-free table, keeping its handle and CPU backup alive. A retiring allocation
is therefore excluded from sleep offload and graph certificates while native
cleanup proceeds. Completion removes the pending record only after unmap,
physical free and virtual-address release succeed. A synchronization failure
leaves the record active because native cleanup has not started. The optional third callback keeps
`init_module(malloc, free)` accepted for legacy callers, whose callback still
owns its original retirement semantics. The legacy path cannot recover a Python
record that its own callback already removed. The CaMem singleton keeps borrowed
callback objects alive.

This retains the existing serialized worker sleep/inference lifecycle; it does
not establish arbitrary external allocation/GC thread safety.

After a native free failure, new mappings, sleep/wake and worker inference
boundaries reject further use. Reuse-enabled execution enters `ERROR`; the flag
cannot be cleared by reinitializing callbacks. When reuse is disabled, execution
still checks an already-existing CaMem singleton before and after the call, but
does not create one solely for this check. These checks add no per-step rank
collective. Final-source A/B measurements include these checks in both arms;
they do not isolate the query guard's own cost.

The failure can occur at different points: unmap may leave a live mapping,
physical free may leave an unmapped physical allocation, virtual-address release
may fail after physical memory was freed, and completion notification may fail
after the virtual address was released. A retained ownership record therefore
does not imply a retained valid address. Such a worker must restart: this is
neither successful graph recapture nor a claim that failed cleanup releases all
memory. Null opaque handles are rejected before dereference, but an arbitrary
forged nonzero pointer is outside the internal handle ownership contract.

### Configuration behavior

Every eligible configuration still requires a valid per-cycle certificate. The
following table describes the implemented decision, rather than validation of
every model size. Unsupported combinations retain the original cleanup/recovery
path; eager mode does not capture a graph. A model that cannot start on the
original path is not made runnable by fallback.

| Configuration | Decision/reason |
|---|---|
| Capture sizes are a nonempty subset of `1, 2, 4, 8`; final MRV2 graphs are captured in the dedicated pool | Reuse candidate. Test sizes independently, including after the first real post-wake request. |
| Any other requested capture size | `capture_sizes_not_validated`; recapture |
| Other graph modes, including eager | `requires_full_decode_only`; original path |
| `attention_v1` with ordinary attention | Reuse candidate |
| Other attention backends | `attention_backend_not_validated`; recapture |
| MLA or multimodal layout | `attention_layout_not_validated`; original path |
| A5, MRV2, BF16, unquantized Qwen2/Qwen3/Qwen3-MoE, TP1/2, EP off | Reuse candidate |
| Other hardware, runner, dtype, quantization or model family | Respectively `hardware_not_validated`, `requires_mrv2`, `requires_bfloat16`, `quantization_not_validated`, or `model_not_validated` |
| EP on; TP outside 1/2; PP/CP/DP outside 1 | `expert_parallel_not_validated` or `parallel_configuration_not_validated` |
| Speculative decoding/MTP or LoRA | `speculation_or_lora_not_validated` |
| KV/weight transfer, weight offload, other KV dtypes, dynamic EPLB or super kernels | Corresponding `*_not_validated` reason; original path |
| Final capture is missing or did not use the sleep pool | `graphs_not_captured_in_sleep_pool`; recapture |
| Inventory throws, or another rank rejects reuse | `resource_inventory_failed` or `peer_not_reusable`; collective recapture |
| Changed certificate after mapping restoration | `resource_signature_changed`; collective recapture |

### Restore and fallback state machine

Control RPCs and worker execution share `transition_lock`. Device synchronization
drains previously enqueued work. For an in-flight request that must finish, use
the existing engine `sleep(mode="wait")` request-draining behavior; the default
abort mode keeps its upstream semantics. A request queued while sleeping cannot
replay until restoration is complete.

| State/transition | Allowed work and next action |
|---|---|
| `READY` -> `ASLEEP` | Synchronize, inventory/vote, then back up and unmap eligible pools. Only `READY` accepts model execution. |
| `ASLEEP` -> `PARTIAL` | Restore only requested tags. A weights-only wake does not enable replay; reload level-2 weights before restoring KV. |
| `ASLEEP`/`PARTIAL` -> `RESTORING` | Restore remaining model and graph pools, vote on restoration health, then certify all dependencies. |
| `RESTORING` -> `READY`, reuse | All ranks agree: retain original graphs, increment reuse count and skip `capture_model()`. |
| `RESTORING` -> `READY`, recapture | Incompatible dependencies or a graph-only restore failure with known mapping state: retire old graphs, release their pool, recapture collectively and resume only after capture succeeds. |
| Any transition -> `ERROR` | Model-memory restoration failed, native mapping state is unsafe, or recovery itself failed: reject inference and require worker restart. Recapture cannot repair missing model data or an unsafe device context. |
| Shutdown -> `CLOSED` | Retire graphs before physical allocation cleanup; terminal until worker restart. |

Same-level repeated sleep and a completed repeated wake are idempotent. A new
sleep from `PARTIAL` is rejected. Recoverable graph incompatibility must not crash
the engine; an unrecoverable restoration failure is deliberately an error, not a
successful fallback or permission to replay an old graph.

### Regression and performance evidence

#### Current R9 local evidence

The [PR evidence section](https://github.com/vllm-project/vllm-ascend/pull/20427)
links the R9 acceptance report, complete data package and exact final local/public
source mapping. This supplements the immutable R7 archive below. The runtime
includes the native allocator corrections: no throwing tensor deleter, active to
pending-free ownership and terminal failure guards. Later test, CI and documentation
changes have separate source indices. R7 timings do not validate these corrections.

| Requirement | Current R9 local evidence |
|---|---|
| Dependencies, certificate and state machine | Resource, reason and transition tables above; supported sizes 1/2/4/8, attention_v1, MRV2 and explicit unsupported-configuration behavior. |
| CPU/NPU regressions and lifecycle | Native host shim: 15 passed; focused CPU: 171 passed, 5 hardware skips and 1 known-baseline ATB deselection; NPU: 176 passed with the same 1 deselection. Final single-card repository E2E: 3 passed with 20 level-1/2 cycles, four faults, real in-flight/queued requests and disabled-flag recapture. |
| Real-model replay and safe fallback | Dense/MoE TP2 each 10 cycles per level and four faults, fixed-input tokens unchanged, unchanged capture counts before and after real requests, stable graph/pool addresses; Dense request boundary, restart-required model restore, TP1 asleep exit and 1024-token request; actual 8-rank MC2 EP8 recaptures. Six strict cases record outer exit 0 and nonempty EngineCore `[0]`. |
| Paired performance | Dense and MoE each 3 fresh-engine A/B pairs, 12 engines total; every engine records nonempty EngineCore `[0]`, valid isolation and a stopped container. All raw values, ranges, pre-sleep controls and memory/restore-phase data are included. |

Family names are eligibility rules, not evidence for every checkpoint. These
full-model results use only Qwen2.5-0.5B-Instruct and Qwen3-30B-A3B; dense Qwen3
was not separately measured. Fixed repeated-text greedy tokens test sleep/replay
consistency, not model quality or exhaustive MoE expert-routing coverage. The same
token hash across these two checkpoints is not a model-identification result.

Final A/B runs use the same rebuilt native object and migrated image, pinned vLLM
`544f182d9f0c9bfe8bd7645525951d5e3692a542`, Torch 2.10.0+cpu and
torch-npu 2.10.0.post7. A and B both include the allocator/guard corrections; the
comparison does not isolate guard cost or compare against an untouched package.
Each family uses its own fixed host in AB/BA/AB order. Private host/board identities
remain local; anonymous summaries certify same-host, fixed-image/source/native,
driver and affinity checks before/after each case. Image directory/env label 9.2.0
does not describe every CANN component: most version files say 9.2.1, and the
original image's ops_transformer 9.3.0 differs from migrated 9.2.1. The archive gives
the targeted component/library inventory; it does not claim identical full images.

Each engine has isolated runtime-config/report paths with hot reload disabled.
The closed workload is 8 requests, 128 input/512 output tokens, one discarded warmup
and 3 measured samples per phase. Only the case literal in those two config paths
is normalized when checking A/B configurations. Thread affinity changes are local
to the experiment's own processes/threads, identical within a family; no global
IRQ/governor/NUMA setting is changed.

| Model / sleep level | Wake reduction, median of 3 pairs (range) | First request change, pair range | Post-wake TPS change, pair range | Extra sleep HBM, MiB/rank, pair range |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B TP2 / L1 | 57.11% (56.23% to 57.36%) | -5.88% to -3.85% | -0.86% to +1.07% | +370.12 to +377.40 |
| Qwen2.5-0.5B TP2 / L2 | 38.73% (37.42% to 39.74%) | -23.07% to -4.33% | -0.74% to +3.17% | +364.22 to +371.75 |
| Qwen3-30B-A3B TP2, EP off / L1 | 28.79% (28.70% to 50.50%) | -22.43% to -1.98% | -1.71% to +1.10% | +350.19 to +372.79 |
| Qwen3-30B-A3B TP2, EP off / L2 | 2.95% (2.58% to 9.35%) | -4.22% to +18.61% | -0.50% to +3.13% | +311.56 to +360.73 |

Wake is host wall time. Level 2 includes weights wake, checkpoint reload,
intermediate validation RPCs and KV wake. First request measures the complete
32-token batch-one request, not HTTP TTFT. TPS includes prefill/decode/host scheduling;
TPOT is host engine-step delivery. A post-wake state RPC separates the wake and
first-request timers; neither is end-to-end client recovery latency.
Host restore/capture timers can nest and must
not be added into a substitute wake time. The complete analysis includes every
paired TPS/TPOT and first-request change, including regressions and outliers;
neither throughput improvement nor absence of regressions is universal.

Used HBM is observed CANN-visible total minus CANN free bytes per rank, not nominal
capacity or Torch-reserved bytes. Retained HBM cannot identify individual HCCL,
event or workspace physical bytes. Graph-pool CPU backups add
54 MiB/rank for Dense and 216 MiB/rank for MoE;
reported CaMem backup bytes exclude separately saved level-2 model buffers and
process RSS/pinned-cache peaks. Full raw samples and numeric analysis are under
`analysis/r9-performance/` in the package, including `raw.csv`, `pairs.csv`,
`summary.csv`, `request-metrics.csv`, `memory-by-rank.csv` and
`host-restore-phases.csv`.

The Dense campaign's per-case outer exit 0 is derived from the reviewed immutable serial
`set-e` launcher, its actually observed exit 0 and all six complete case gates; it
does not have six separate original exit-code files. MoE has explicit case exit
records. Exploratory/interrupted engines and the earlier overwritten MoE shell
with outer exit 1 are excluded rather than pooled into the final repetitions.

The repository E2E must be selected for A5 hardware: its one-card directory's
default A2 lane would skip it. Precise A5 mapping and curated registration are
included; selector CLI checks are not hosted device execution. A maintainer adds
exactly one run-enabling label: `ready-precise`, `ready-all` or `main2main`;
the curated A5 job also requires `ready-a5`. This PR changes CI-routing YAML, so
the existing workflow's CI-pipeline policy can force an all-tests selection even
with `ready-precise`; that label is not a promise to run only this new test.
Pending/skipped jobs, local validation, human review and community-task
acceptance are separate outcomes.

#### Historical R7 archive

The [PR evidence package](https://github.com/user-attachments/files/34211188/task129-a5-sleep-graph-r7-evidence.zip)
contains immutable R7 results, frozen harnesses and all three paired repetitions.
It has SHA256
`cf37e92a0e4d2464f0d95a0ee0e2e32a57b1182e741cc5ed3bfccb1ea1ed69ec`.
The NPU-tested public source is commit `257139421e8ae94052cfc921685e829f2a077eda`
(tree `ddfd46808645886ebaab3db55660bd37f3bb7469`), based on Ascend
`acbab0ef0722f0a1aab7e6a4ac90c6c93f888efe` and vLLM
`544f182d9f0c9bfe8bd7645525951d5e3692a542`. Later changes need their own
validation record; this package does not establish testing of a later HEAD.

Historical shutdown evidence has a narrower scope than functional pass status:
the twelve R7 performance engines each record a nonempty
`normal_shutdown.engine_exitcodes=[0]` list. Six functional/diagnostic cases
(Dense correctness/asleep/fatal-restore, MoE correctness, EP8 fallback and
long-sequence guard) do not record those EngineCore exit codes. Their early
`passed=true`, outer-script exit status or managed-memory cleanup cannot establish
normal EngineCore exit. Do not describe all eighteen cases as explicit zero-exit
evidence. The recorded EngineCore list also does not establish every grandchild
worker's individual exit code. New validation should record process identities
and exit codes, assert that the expected process list is nonempty, and distinguish
functional correctness, allocator cleanup and process shutdown.

| Requirement | Test/source entry | R7 evidence inside the package |
|---|---|---|
| Multi-cycle reuse and staged level-2 wake | `tests/e2e/pull_request/one_card/rlhf/state_transitions/test_sleep_graph_reuse.py`; `tests/ut/device_allocator/test_graph_reuse.py` | `evidence/final-r7-dense-asleep-exit`, `final-r7-dense-correctness`, `final-r7-moe-correctness`: each has ten cycles per level and all four sizes. |
| Stable addresses and physical release | `tests/ut/device_allocator/test_camem.py`; dedicated-pool PoC | Two final-environment PoCs, each ten awake-destruction and ten asleep-destruction cycles; graph contents and addresses match after mapping restoration. |
| Changed parameter address/workspace/event; graph-only restore failure | Certificate unit tests plus frozen model harness | Dense/MoE `result.json` fault records show collective recapture and unchanged generated tokens. Graph restore and event faults originate on rank 0. |
| In-flight/queued requests and partial-wake rejection | Execution-lock unit tests and frozen request-boundary harness | Dense correctness `inflight_boundary`: finish a live request with wait mode, queue another asleep, then verify tokens and no premature replay. |
| Unsafe model restore | Model-pool fault harness | `evidence/final-r7-dense-fatal-restore`: both ranks enter `ERROR`, with no old-graph replay. This is restart-required safety evidence. |
| Unsupported EP path | Configuration tests and actual MoE EP8 run | `evidence/final-r7-ep8-fallback`: MC2 dispatch/combine observed, two cycles per level, all ranks recapture. This does not establish EP reuse. |

To prove reuse, compare graph identities, the runner's
`sleep_capture_model_calls`, lower-level `compilation_counter.num_cudagraph_captured`
and manager replay counters both immediately after wake and after the first real
request. Logs contain `Sleep graph decision=... reason=...` and
`Sleep graph wake action=... reuses=... recaptures=...`. Unchanged capture counts
without a real post-wake request are insufficient. Compare greedy token IDs
before/after sleep and against the recapture arm; inspect logits for any
unexplained difference.

R7 measurements used A5, driver 25.1.rc3.1, CANN directory/env label 9.2.0,
Torch 2.10.0+cpu, torch-npu 2.10.0.post7 and Python 3.12. The actual mixed CANN
component versions are recorded in the targeted audit linked by the R9 report;
the R7 ZIP remains unchanged. Each model has three fresh-engine pairs in
AB/BA/AB order on the same host. A disables the reuse flag and uses extra cleanup
with recapture; B enables reuse. Both arms include common allocator/lifecycle
fixes, so A is not an untouched baseline installation. Per phase, discard one
warmup and average three measured closed batches of eight requests, with 128
input and 512 output tokens. The timing harness only changes its own process
thread affinity; framework CPU binding is disabled.

| Model / sleep level | Wake reduction, median of 3 pairs | First request change, pair range | Post-wake TPS change, pair range | Extra sleep HBM, MiB/rank, pair range |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B TP2 / L1 | 59.12% | -10.00% to -2.57% | -3.85% to +8.35% | +377.00 to +377.40 |
| Qwen2.5-0.5B TP2 / L2 | 37.97% | -6.01% to +12.02% | -0.79% to +9.03% | +371.58 to +372.10 |
| Qwen3-30B-A3B TP2, EP off / L1 | 41.61% | -4.67% to -1.42% | +0.53% to +2.02% | +308.69 to +380.14 |
| Qwen3-30B-A3B TP2, EP off / L2 | 6.40% | -3.17% to -1.84% | +0.79% to +2.20% | +306.80 to +378.48 |

Wake is host wall time. Level 2 includes weights wake, checkpoint reload,
intermediate validation RPCs and KV wake. First request measures the complete
32-token batch-one request, not HTTP TTFT. Throughput includes prefill, decode
and host scheduling; host-observed TPOT uses engine-step token delivery, not
kernel timestamps. HBM deltas use aggregate free-memory observations per rank,
not an isolated measurement of HCCL memory. Managed graph CPU backups are about
54 MiB/rank for Dense and 216 MiB/rank for MoE. The retained HBM and CPU backup are
costs of reuse.

All raw samples, pre-sleep controls, paired changes and host phase timings are
under `analysis/final-r7/` in the package, including `raw.csv`, `pairs.csv`,
`request-metrics.csv`, `memory-by-rank.csv` and `host-restore-phases.csv`. Phase
timers can nest and must not be summed into a substitute wake time. The results
do not establish universal throughput improvement or absence of regressions.

MLA/MTP/dynamic EPLB/A3 lack full-model validation. LoRA failed before sleep in
the tested environment (missing `sgmv_shrink`, or an initial Torch graph-capture
failure); neither attempt counts as a passed fallback. V1/FP16/size16/eager and
some parallel/speculation fallbacks have separate historical R5 records, not R7
re-runs. Required hosted CI, maintainer scope review and community-task acceptance
remain independent of these local experiments.

## Prepare Model Weights

Use the `Qwen2.5-0.5B-Instruct` model weights. With `VLLM_USE_MODELSCOPE=True`, the model will be downloaded automatically from ModelScope.

```{list-table}
:header-rows: 1

* - Model
  - ModelScope Link
* - Qwen2.5-0.5B-Instruct
  - [Qwen/Qwen2.5-0.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct)
```

## Usage

The following is a simple example of how to use sleep mode.

- Offline inference:

    ```python
    import os

    import torch
    from vllm import LLM, SamplingParams
    from vllm.utils.mem_constants import GiB_bytes

    os.environ["VLLM_USE_MODELSCOPE"] = "True"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    if __name__ == "__main__":
        prompt = "How are you?"

        free, total = torch.npu.mem_get_info()
        print(f"Free memory before sleep: {free / 1024 ** 3:.2f} GiB")
        # record npu memory use baseline in case other process is running
        used_bytes_baseline = total - free
        llm = LLM(
            "Qwen/Qwen2.5-0.5B-Instruct",
            enable_sleep_mode=True,
            additional_config={"weight_nz_mode": 0},
        )
        sampling_params = SamplingParams(temperature=0, max_tokens=10)
        output = llm.generate(prompt, sampling_params)

        llm.sleep(level=1)

        free_npu_bytes_after_sleep, total = torch.npu.mem_get_info()
        print(f"Free memory after sleep: {free_npu_bytes_after_sleep / 1024 ** 3:.2f} GiB")
        used_bytes = total - free_npu_bytes_after_sleep - used_bytes_baseline
        # now the memory usage should be less than the model weights
        # (0.5B model, 1GiB weights)
        assert used_bytes < 1 * GiB_bytes

        llm.wake_up()
        output2 = llm.generate(prompt, sampling_params)
        # cmp output
        assert output[0].outputs[0].text == output2[0].outputs[0].text
    ```

- Online serving:
    !!! note

            Considering there may be a risk of malicious access, please make sure you are under a dev-mode, and explicitly specify the dev environment `VLLM_SERVER_DEV_MODE` to expose these endpoints (sleep/wake up).

    ```bash
    export VLLM_SERVER_DEV_MODE="1"
    export VLLM_WORKER_MULTIPROC_METHOD="spawn"
    export VLLM_USE_MODELSCOPE="True"

    vllm serve Qwen/Qwen2.5-0.5B-Instruct \
        --enable-sleep-mode \
        --additional-config='{"weight_nz_mode": 0}'

    # after serving is up, post to these endpoints.
    # /sleep reads level from the query string (JSON body is ignored).

    # --- Level 1: offload weights, discard KV cache ---
    curl -X POST "http://127.0.0.1:8000/sleep?level=1"
    curl -X GET http://127.0.0.1:8000/is_sleeping

    # wake all tags (weights + kv_cache)
    curl -X POST http://127.0.0.1:8000/wake_up
    curl -X GET http://127.0.0.1:8000/is_sleeping

    # serving is available again after Level-1 wake_up
    curl http://127.0.0.1:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "prompt": "The future of AI is",
            "max_tokens": 7,
            "temperature": 0
        }'

    # --- Level 2: discard weights and KV cache ---
    # tags must be in ["weights", "kv_cache"]. After waking weights,
    # reload the checkpoint on every worker before waking kv_cache.
    curl -X POST "http://127.0.0.1:8000/sleep?level=2"
    curl -X POST "http://127.0.0.1:8000/wake_up?tags=weights"
    curl -X POST http://127.0.0.1:8000/collective_rpc \
        -H "Content-Type: application/json" \
        -d '{
            "method": "reload_weights",
            "kwargs": {"weights_path": "Qwen/Qwen2.5-0.5B-Instruct"}
        }'
    curl -X POST "http://127.0.0.1:8000/wake_up?tags=kv_cache"
    ```
