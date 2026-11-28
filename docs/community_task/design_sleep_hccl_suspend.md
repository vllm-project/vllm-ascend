# Sleep Mode: HCCL destroy → suspend/resume lightweight lifecycle

Task: 【vllm-ascend】[Sleep Mode] HCCL 通信域 destroy → suspend/resume 轻量化改造

## 1. Scope

The extra-cleanup sleep path on Ascend previously destroyed every HCCL
process group and recreated it on wake-up. This change adds a capability-gated
lightweight path based on `HcclCommSuspend` / `HcclCommResume`:

- `sleep_hccl_policy=auto` (default): suspend communicators when the runtime
  supports it, otherwise fall back to destroy/restore.
- `sleep_hccl_policy=lightweight`: require suspend/resume; fall back only when
  `sleep_allow_fallback=true`.
- `sleep_hccl_policy=destroy`: keep the historical destroy/restore path.
- `sleep_hccl_policy=skip`: no HCCL cleanup (debugging; not for production).

The allocator (`CaMemAllocator`) still owns weights/KV memory; this change only
replaces the HCCL part of `SleepWakeupManager`.

## 2. Lifecycle state machine

`HcclSleepWakeupManager` is the single owner of the HCCL lifecycle.

| State | Meaning | Allowed next |
|---|---|---|
| `idle` | communicators active | `sleeping` |
| `sleeping` | draining PP sends and suspending/destroying | `asleep`, `idle` (rollback), `failed` |
| `asleep` | allocator sleep is in effect | `waking`, `failed` |
| `waking` | resuming or restoring groups | `idle`, `failed` |
| `failed` | half-recovered; requires operator/engine restart | `idle` (explicit reset) |

Invariants:

1. `sleep()` while `asleep` is an idempotent no-op; `wakeup()` while `idle` is
   an idempotent no-op. Any other illegal transition raises.
2. Before touching HCCL, pending PP sends are drained and the device is
   synchronized. A suspended communicator must not issue any collective —
   including `barrier()` — until it is resumed. This is why the control plane
   for cross-rank coordination has to be non-HCCL (CPU/gloo, external launcher
   barrier, or the engine's own ordering).
3. If suspend fails after some groups were suspended, all of them are resumed
   before the manager either falls back to destroy/restore or raises.
4. Wake resumes only what sleep suspended. A failed resume falls back to
   destroy + restore when `sleep_allow_fallback=true`, otherwise the state
   becomes `failed` and the error is raised.
5. Duplicate `device_group`/communicator handles are suspended once; the
   suspended handle set is cleared after a successful resume.
6. Multi-rank suspend is committed only if every rank reports success in a
   CPU-process-group `MIN` vote. A rejected vote rolls back local suspends and
   all ranks fall back to destroy/restore together. A failed vote
   (timeout/exception, e.g. a dead peer) fails the sleep cycle instead of
   mixing suspend and destroy policies, which would hang on wake-up.

## 3. Compatibility and capability detection

`HcclSuspendBackend` loads `libhccl.so` (via `find_hccl_library()`) and checks
`HcommIsSupportHcclCommResume()` when the symbol exists. On the verified A3
host the capability flag is 0 before `torch_npu` initializes the driver and 1
afterwards, so the check is performed lazily inside the worker process.

Verified environment:

| Component | Version |
|---|---|
| Hardware | Atlas A3 (910_93), 2/4-card TP |
| CANN | 9.2.0 |
| torch_npu | 2.10.0.post7 |
| vLLM | 0.31.0 |
| vllm-ascend | main + this change |

Observed `HcclCommSuspend` semantics on this stack:

- A healthy communicator returns `22 = HCCL_E_SUSPENDING`; the suspension is
  effective and is treated as success.
- `ProcessGroupHCCL.resume_hccl_comm(device_id)` restores the group (~1.3 ms on
  a 2-rank group); raw `HcclCommResume` is kept as a fallback.
- `HcclCommGetStatus` is not supported on this stack (returns 5), so the state
  machine is internal and does not depend on the HCCL status API.
- Suspending does **not** release device memory. The benefit is avoiding
  communicator teardown/rebuild and preserving rank/communicator identity;
  memory is reclaimed by the allocator as before.

Older CANN releases without the suspend/resume symbols, or hosts where the
capability flag is 0, automatically use the destroy/restore path.

## 4. Fallback matrix

| Condition | `auto` | `lightweight` | `destroy` |
|---|---|---|---|
| suspend symbols + capability OK | suspend | suspend | destroy |
| symbols missing / capability 0 | destroy | destroy if fallback enabled, else raise | destroy |
| suspend fails at runtime | resume rolled back, then destroy | same if fallback enabled, else raise | destroy |
| resume fails at wake | destroy+restore if fallback enabled, else `failed` | same | destroy+restore |

`skip` performs no HCCL action in either direction.

## 5. Tests

Unit tests (`tests/ut/device_allocator/test_hccl_suspend.py`,
`test_sleep_mem_optimized.py`, `tests/ut/test_ascend_config.py`):

- capability gating, `HCCL_E_SUSPENDING` handling, symbol failure paths;
- handle extraction and dedupe; partial-suspend rollback;
- state machine transitions, idempotent sleep/wake, repeated cycles;
- suspend failure → destroy fallback, no-fallback → `failed`;
- resume failure → destroy fallback / `failed`;
- policy validation (`auto`/`lightweight`/`destroy`/`skip`).

End-to-end (`tests/e2e/pull_request/two_card/rlhf/state_transitions/test_sleep_hccl_suspend.py`):
TP=2 `AsyncLLM`, cold-start token baseline, repeated sleep/wake, lightweight and
destroy policies, a request issued while sleeping that must be queued until
wake and then produce the baseline tokens, a partial-rank suspend failure that
must roll back to the destroy path on every rank, and shutdown while asleep
followed by a successful replacement engine.

## 6. Raw measurements (2026-11-04, A3 910_93, CANN 9.2, Qwen3.5-4B)

Command: `vllm.entrypoints.openai.api_server --enable-sleep-mode --enforce-eager`,
`rl_config={enabled:true, sleep_mode_extra_cleanup:true}`.

| Policy | TP | sleep (ms) | wake (ms) | first generate after wake (ms) | tokens vs cold start |
|---|---|---|---|---|---|
| lightweight | 2 | 5726 / 4007 / 4289 | 5150 / 5198 / 4675 | 935 / 974 / 895 | identical ×3 |
| destroy | 2 | 4656 / 4238 / 4451 | 5470 / 5254 / 5019 | 1632 / 1645 / 1572 | identical ×3 |
| lightweight (fallback: capability disabled) | 2 | 4641 | 5575 | 1614 | identical |
| lightweight | 4 | 6144 | 5454 | 941 | identical (logprob max diff 0.0) |

HCCL manager stats for the TP=2 lightweight run (per rank): 7 groups suspended
(`world`, `tp`, `dcp`, `pcp`, `pp`, `dp`, `mc2`), `suspend_calls=7`,
`suspend_ok=7`, `suspend_suspending=1`, `suspend_failed=0`, `fallbacks=0`.
The fallback run logged `HCCL suspend/resume unavailable` and destroyed 7 groups;
wake restored them and generation matched.

Queued-request run: a completion issued after sleep completed (latency 8017 ms,
including 5099 ms wake) and matched the cold-start tokens exactly.

Interpretation: on this model/TP size the wall-clock wake is dominated by the
allocator restoring weights/KV (~5 s); the HCCL part of wake is ~1.3 ms versus
communicator recreation on the destroy path. The measurable benefit here is the
first request after wake (~0.9 s vs ~1.6 s) and the preserved communicator
identity; larger TP/EP jobs and slower HCCL rebuilds should widen the gap.
Suspend itself does not free HBM, so peak memory after sleep is slightly higher
than the destroy path on multi-group jobs.

## 7. Known limitations

- `HCCL_E_SUSPENDING` is treated as success based on observed behavior; the
  return code is not documented as a success code.
- A client that disconnects while its request is queued during sleep can still
  race with wake-up. The engine-level queue behavior (request completes after
  wake) is covered; the disconnected-client edge is an upstream vLLM concern.
- Suspend/resume must never be mixed with collectives on the same communicator
  while suspended. Any new code path that performs HCCL work during sleep will
  deadlock.
- The destroy fallback path keeps the historical lifecycle anchor; the
  lightweight path does not create one.
