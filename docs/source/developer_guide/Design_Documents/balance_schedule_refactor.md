# Balance Schedule Refactor

**TL;DR** The existing `patch_balance_schedule.py` copied three large upstream
units verbatim to inject roughly 5 lines of real logic: `Scheduler.schedule()`
(~520 lines), `DPEngineCoreProc.run_busy_loop()` (~40 lines), and
`EngineCoreProc.run_engine_core()` (~55 lines). This refactor, while **strictly
preserving the `balance_flag` semantics**, first deletes the two already-stale
copies `run_busy_loop()` / `run_engine_core()` (replacing them with an engine
core hook on `_has_global_unfinished_reqs` plus a module-level name swap for
conditional activation). The `schedule()` copy is **kept for now** — upstream
exposes no finer-grained hook to borrow, and deleting it depends on contributing
an override seam upstream, tracked as later Phase 2B. The file therefore does
not shrink to a few dozen lines: the `schedule()` body is still a verbatim
upstream copy (**aligned verbatim to release tag `v0.24.0`**, with only 3
balance deltas), and the file is ~830 lines. Aligning to a stable release tag
(rather than a moving main-verified commit hash) makes "verbatim comparison
against upstream" a reproducible drift check — a fixed tag points at the same
source on every CI run. What this round actually removes is the stale-drift
risk of the `run_busy_loop` / `run_engine_core` copies, and fixes a
balance-enabled deadlock (originally "gather inside `schedule()`", then in a
later iteration "gather inside `_process_engine_step`"); gather now sits
immediately after the `_has_global_unfinished_reqs` cross-rank all-reduce.

## Background

### What Balance Scheduling does

With a large `data-parallel-size` and concurrency ≈ `DP × max-num-seqs`,
requests tend to pile up on a subset of DP ranks: the saturated ranks carry
prefill and decode at once and slow down, while the other ranks keep admitting
new requests, widening the gap. Balance scheduling does **not** actively
rebalance the per-rank running counts. Instead it provides a **global admission
gate**: as soon as **any one rank's** running count reaches the cap, **all
ranks** stop admitting new requests from the WAITING queue, giving the
saturated rank a chance to drain its in-flight requests so the gap stops
growing. It is **not** "make the lagging ranks catch up to the leader" (that
semantic is **explicitly rejected** here — see
[Behavior-preservation contract](#behavior-preservation-contract), item 1).

The feature is enabled via `additional_config.enable_balance_scheduling = true`
(the environment variable `VLLM_ASCEND_BALANCE_SCHEDULING` is deprecated). It
supports PD-mixed mode only; validation lives in `vllm_ascend/platform.py` and
`vllm_ascend/ascend_config.py`.

### The real logic (only two spots)

The whole feature reduces to two operations:

1. **Cross-rank sync of the running count** — one `all_gather` per engine step,
   collecting each rank's `len(self.running)`:

   ```python
   def balance_gather(self):  # dp_group is injected into self.dp_group by the engine core
       running_tensor = torch.tensor([len(self.running)], dtype=torch.int, device="cpu")
       dist.all_gather(self.balance_queue, running_tensor, group=self.dp_group)
   ```

2. **The admission gate inside the WAITING scheduling loop** — because every
   rank holds the same gathered vector, this check yields the same result on
   every rank. Whenever **any rank's** running count reached the cap at the end
   of the previous step, **all ranks** stop admitting new WAITING requests this
   step:

   ```python
   balance_flag = max(t.item() for t in self.balance_queue) == self.max_num_running_reqs
   if balance_flag:
       break
   ```

   **Semantic (must be preserved bit-for-bit):** "leader-at-cap ⇒ global freeze
   of admission". It is **not** "make lagging ranks catch up to the leader". See
   [Behavior-preservation contract](#behavior-preservation-contract).

## Problem statement

To inject the two pieces above, the existing `patch_balance_schedule.py` copied
three large upstream units verbatim:

| Copied unit                        | Lines | Reason for copying                                |
|------------------------------------|-------|---------------------------------------------------|
| `Scheduler.schedule()`             | ~520  | Insert the 3-line `balance_flag` gate mid-loop    |
| `DPEngineCoreProc.run_busy_loop()` | ~40   | Call `balance_gather` after every step            |
| `EngineCoreProc.run_engine_core()` | ~55   | Replace with `BalanceDPEngineCoreProc` when DP>1  |

This "copy whole units" approach has three concrete harms:

1. **The `schedule()` copy is now aligned verbatim to a release tag (the
   production pin, currently `v0.24.0`).** The **single source of truth** for
   the release tag is `.github/vllm-release-tag.commit` (CI reads the same file
   via `tr -d '[:space:]' < .github/vllm-release-tag.commit`), currently
   `v0.24.0`; dev/CI actually installs the main-verified commit pointed at by
   `.github/vllm-main-verified.commit` (which carries later scheduler
   evolution). The old patch copied a `schedule()` from an
   older vLLM than v0.24.0, so it was stale as a whole. This round aligns the
   `schedule()` copy **verbatim to the release tag's `Scheduler.schedule()`**,
   keeping only the 3 balance deltas (disabled-path early return,
   `balance_flag` gate, `if request_queue is None: break`); the
   `run_busy_loop()` / `run_engine_core()` copies were deleted in Phase 1.
   **Note: any concrete `v0.24.0` in this document is just a snapshot of the pin
   file's current value — it goes stale as the pin advances and must NOT be
   used as a version authority; any code/test that needs this tag must read the
   file at runtime.**

   **Why align to the v0.24.0 tag rather than the installed main-verified
   commit?** Two reasons: (a) production actually runs the v0.24.0 release, so
   aligning the copy to it keeps production behavior consistent with runtime;
   (b) a fixed git tag points at the **same** source on every CI run, so
   "verbatim comparison of the copy against upstream (allowing only the 3
   deltas)" becomes a **reproducible** drift check — whereas a moving
   main-verified hash makes the comparison drift forward with every commit and
   cannot serve as a stable guardrail.

   **Cost and boundary:** the copy (v0.24.0 logic) and the main-verified
   runtime differ slightly in behavior, but balance's real scheduling path is
   only reached under NPU + DP + MoE, never by CPU UT (see Test plan); and
   those differences do not affect the gate's own semantics. Both supported
   revisions — release tag v0.24.0 and main-verified commit e5588e49 — expose
   `schedule(self, throttle_prefills=False)`, so the override matches that
   shared signature and the disabled path forwards `throttle_prefills`
   directly to `super()`. I.e.: **body aligned to the release tag; signature
   matches both supported revisions; disabled path delegates directly.**

2. **It violates the `AGENTS.md` patch policy.** The policy requires patches to
   be "minimal and focused" with "a long-term plan to contribute upstream". A
   500-line verbatim copy is unreviewable (you cannot see the real change
   without diffing against upstream) and must be re-synced by hand on every vLLM
   upgrade.

3. **Silent, undocumented deviations accumulate.** For example the override
   quietly turned upstream's `assert request_queue is not None` into
   `if request_queue is None: break`. Such deviations make future diffs
   untrustworthy.

> Lesson learned this round: after v0.23 support is dropped, both supported
> revisions expose the same `Scheduler.schedule(self, throttle_prefills=False)`
> contract. Compatibility introspection and version-string branching therefore
> add no value and should be removed. The unit test now requires the override's
> signature to equal the installed upstream signature in each CI lane.

## Design

### Principle

Remove copies, not the feature. Pull gather into the scheduler itself (so the
two EngineCore copies can be deleted), and in the future inject the gate via a
minimal upstream seam (so the `schedule()` copy can be deleted). The
`balance_flag` semantics stay strictly unchanged.

### Implementation status and corrections (this round landed Phase 1 + 2A + 3)

While implementing, we verified upstream's real structure and found two
drafting assumptions did not hold; both are corrected here:

- **Upstream `Scheduler` has no `new_step_starts()` lifecycle hook** (that is a
  `kv_cache_manager` method) and no "per-step scheduling start" overridable
  seam. So gather cannot live inside `schedule()` (see the deadlock lesson in
  [Step 1 — Hook gather onto `_has_global_unfinished_reqs` and delete the EngineCore copies](#step-1--hook-gather-onto-_has_global_unfinished_reqs-and-delete-the-enginecore-copies));
  it lives on the engine core's `_has_global_unfinished_reqs` instead.
- **The scheduler cannot lazily fetch the DP group:** `dp_group` is produced in
  `_init_data_parallel` (earlier than scheduler creation) and there is no global
  registry. So `BalanceDPEngineCoreProc` is **not deleted** but slimmed to a
  subclass that overrides `_has_global_unfinished_reqs` (injects `dp_group` +
  calls `balance_gather` once); the `run_engine_core` copy is replaced by
  patching the module-level `DPEngineCoreProc` name (upstream's `run_engine_core`
  resolves this class by module-global name at call time), and the swap happens
  **only when balance is enabled** (conditional activation).

Accordingly the Phase 1 description and the "post-refactor file shape" below are
both rewritten to match the actual implementation. Phase 3 collapses config
probing to two fallbacks (AscendConfig → additional_config) and **removes the
direct environment-variable read** — `VLLM_ASCEND_BALANCE_SCHEDULING` is still
parsed centrally by `AscendConfig` (as a deprecated fallback for
`additional_config`), but `_balance_scheduling_enabled` no longer reads it
itself, bypassing `AscendConfig`. Details in [Phased rollout](#phased-rollout).

### Step 1 — Hook gather onto `_has_global_unfinished_reqs` and delete the EngineCore copies

The old `BalanceDPEngineCoreProc.run_busy_loop()` and `run_engine_core()` were
verbatim copies of upstream methods and **already stale-drifted**: upstream
`run_busy_loop` switched to `while self._handle_shutdown()`, added
`eep_scaling_state` / `is_sleeping` guards and a trailing `raise SystemExit`,
while the patch still had `while True` + a hand-written signal handler;
`run_engine_core` similarly gained `SignalCallback`, numa, tracer logic. The
goal is to delete both copies.

Three constraints surfaced during implementation:

1. **The scheduler cannot obtain the DP group itself.** `dp_group` is created by
   `parallel_config.stateless_init_dp_group()` inside
   `DPEngineCoreProc._init_data_parallel` and stored on the engine core; that
   method runs in `EngineCoreProc.__init__` **before** `super().__init__()`
   (which creates the scheduler), and there is no registry for lazy lookup. So
   the engine core must hand `dp_group` to the scheduler.
2. **`balance_gather` must run on every rank on every iteration** that is part
   of an active (non-idle) wave (see the deadlock lessons below). `schedule()`
   does not satisfy this — a rank that has drained locally takes a dummy batch
   and never enters `schedule()`.
3. **Balance must not invade configs that do not enable it** (e.g. PD-disaggregated
   recompute / `AsyncRecomputeScheduler`), so the `BalanceDPEngineCoreProc`
   swap must be conditional.

The final landing is therefore:

- **`balance_gather` is pulled into `BalanceScheduler`** (no-arg signature, uses
  `self.dp_group`), but **not called from `schedule()`** — the engine core
  triggers it per step (see below).
- **`BalanceDPEngineCoreProc` is not deleted but slimmed to one override**: it
  hooks `_has_global_unfinished_reqs`, calling `super()._has_global_unfinished_reqs()`
  first (which runs the cross-rank all-reduce every 32 steps) and then, still
  inside that same call, injecting `dp_group` and calling `balance_gather()`
  once. Upstream's `run_busy_loop` calls `_has_global_unfinished_reqs` exactly
  once per iteration on every non-idle path (including dummy-batch iterations
  where a drained rank never enters `schedule()`), so every rank participates in
  the gather every active step. The `run_busy_loop` body is no longer copied.
- **The `run_engine_core` copy is deleted entirely.** Upstream's `run_engine_core`
  (a staticmethod) resolves `DPEngineCoreProc` by module-global name inside its
  body (`engine_core = DPEngineCoreProc(*args, **kwargs)`, see
  [vllm/v1/engine/core.py](https://github.com/vllm-project/vllm)). So a thin
  wrapper wraps `run_engine_core`: at its entry (where `vllm_config` is
  available) it decides, via `_balance_scheduling_enabled`, whether to swap the
  module-level `DPEngineCoreProc` to `BalanceDPEngineCoreProc` or restore the
  upstream original, then calls the original `run_engine_core`. This is
  **conditional activation** — with balance off, upstream's implementation is
  used verbatim; signal handling, `SignalCallback`, numa, and tracer all stay
  upstream-correct.

> **Lesson A — deadlock (gather must not live in `schedule()`).** An earlier
> version put `balance_gather` at the top of `BalanceScheduler.schedule()` and
> argued "between two steps `self.running` only changes inside `schedule()` /
> `update_from_output()`, so the snapshot is equivalent and there is still one
> `all_gather` per step — safe". That argument was **only half right**: the
> value the gate sees is indeed equivalent, but it **ignored that `all_gather`
> is a collective and every rank must participate synchronously**. Under DP MoE,
> a rank that has drained its local requests (`has_requests()` is False) runs
> `execute_dummy_batch()` and **never enters `schedule()`**, so it skips that
> `all_gather` while a still-busy rank calls it and waits forever — **collective
> mismatch, deadlock**. The fact that `_has_global_unfinished_reqs` only truly
> all-reduces every 32 steps and that `engines_running` is sticky in between
> widens this window.
>
> **Lesson B — deadlock (gather must not live in `_process_engine_step` either;
> it must sit immediately after the `_has_global_unfinished_reqs` all-reduce).**
> A later iteration "fixed" Lesson A by moving gather into
> `_process_engine_step` (called every iteration, before the sync and before
> the idle `continue` gate). That **re-introduced a different deadlock**:
> `_has_global_unfinished_reqs` is the **only** point in the busy loop that
> re-synchronizes ranks on wave/idle state. Placing the every-step `all_gather`
> **before** that sync (and before the idle `continue`) decouples the gather
> from wave coordination. At wave boundaries — requests finishing at different
> times per rank, `_process_input_queue` blocking on the next wave / new
> requests, `engines_running` sticky for up to 32 steps — one rank can reach the
> gather while another is still blocked in `_process_input_queue` or in
> `future.result()`. The `all_gather` then deadlocks; the stuck EngineCore can
> no longer drain its worker shared-memory broadcast channel, the worker's
> `sample_tokens` response has nowhere to land, and after 60s the engine dies
> with `RPC call to sample_tokens timed out` (observed intermittently — "5 GPQA
> runs OK, 6th hangs" — because the trigger depends on per-run completion
> timing). This is independent of whether expert-parallel spans DP ranks: the
> failure mechanism is EngineCore↔worker shm exhaustion, not a worker forward
> pass. Conclusion: **`balance_gather` must sit immediately after
> `super()._has_global_unfinished_reqs()`, the only per-iteration cross-rank
> sync, so ranks enter the all-gather having just agreed on `engines_running`.**
> Because `_has_global_unfinished_reqs` is only called on iterations that did
> **not** take the idle `continue`, gather is skipped consistently by every
> rank when all are idle (no rank does an extra gather) — identical to the
> pre-refactor copied `run_busy_loop`, where gather sat right after the
> all-reduce. The lesson is locked by the `_has_global_unfinished_reqs` seam
> guard in the unit tests.

**Timing — bit-for-bit identical to pre-refactor.** Gather now runs at the tail
of `_has_global_unfinished_reqs`, i.e. after `schedule()` + execute +
`update_from_output()` and immediately after the cross-rank all-reduce — exactly
where the pre-refactor engine-core-driven gather sat. The gate consumes that
value in the next step's `schedule()`; behavior is unchanged.

### Step 2 — Replace the `schedule()` copy with a minimal seam

The `balance_flag` gate is inlined mid-`schedule()` by upstream; there is
currently no overridable seam. This is solved in two phases.

**Phase 2A — transitional (no upstream dependency):**

Keep the `schedule()` override, but:

- **The override matches the shared supported signature:**
  `def schedule(self, throttle_prefills: bool = False)`. Both v0.24.0 and
  e5588e49 expose this signature, and the disabled path delegates directly via
  `super().schedule(throttle_prefills)`. The old v0.23 compatibility branch and
  signature introspection are no longer needed.
- Collapse the balance changes into 3 clearly-commented deltas: (1) the
  disabled-path early return delegating to `super()`; (2) the `balance_flag`
  gate inside the WAITING loop; (3) `if request_queue is None: break` (upstream
  has `assert`). Because upstream has no finer-grained hook, the body still has
  to be copied.
- **Verbatim comparison is now reproducible:** the `schedule()` copy is aligned
  to the release tag (only the 3 balance deltas differ), so the fixed tag makes
  "verbatim comparison against upstream" yield the same baseline on every CI
  run. The "intent lock" tests (signature equality, the 3 delta
  lines present, upstream seams still exist) remain as CPU-reachable guardrails,
  and a new "verbatim comparison against the release tag (allowing only the 3
  deltas)" drift test is added (see Test plan). The drift test **reads the tag
  at runtime from `.github/vllm-release-tag.commit`** (same source as CI) — it
  does not hardcode a version or read a design doc; when the pin advances, the
  test automatically compares against the new tag and goes red to signal "the
  copy needs re-syncing".
- "Re-aligning the copy on a pin advance" is now routine maintenance: each time
  the release tag advances, re-apply the 3 deltas onto the new tag's
  `schedule()` (continues until Phase 2B deletes the copy).

**Phase 2B — target (lands with an upstream contribution):**

Contribute a minimal upstream refactor that extracts the WAITING loop's stop
condition into an overridable method:

```python
# upstream vllm/v1/core/sched/scheduler.py
def _should_stop_admitting_waiting(self) -> bool:
    return len(self.running) >= self.max_num_running_reqs
```

Once upstream exposes that seam, the Ascend patch collapses to:

```python
class BalanceScheduler(Scheduler):
    def _should_stop_admitting_waiting(self) -> bool:
        if super()._should_stop_admitting_waiting():
            return True
        return self._balance_enabled and (
            max(t.item() for t in self.balance_queue) >= self.max_num_running_reqs
        )
```

(`>=` and `==` are equivalent here because no rank's `len(running)` can exceed
`max_num_running_reqs`; the contract below pins the semantic to `==` to make
"leader-at-cap ⇒ freeze" explicit and to reject the "catch up to leader"
reinterpretation.)

**Result:** the ~520-line `schedule()` copy is deleted for good; the file no
longer drifts with upstream edits to `schedule()`. This is the "long-term plan
to contribute upstream" that `AGENTS.md` requires.

> **On `>=` vs `==`, and the rejected "temporarily lower the cap" idea.** An
> earlier idea was to reuse upstream's existing break condition for zero-copy by
> temporarily setting
> `self.max_num_running_reqs = min(cap, max(balance_queue))`. That would produce
> a **different** semantic ("make lagging ranks catch up to the leader") and is
> **explicitly rejected** — see the contract below.

### Step 3 — Normalize config probing

`_balance_scheduling_enabled()` collapses to **two fallbacks (AscendConfig →
additional_config)**. After deleting the `run_engine_core` copy, the only caller
is `BalanceScheduler.__init__`, but whether AscendConfig is initialized at that
moment still cannot be guaranteed (the origin of the old top-of-file TODO), so
`additional_config` is kept as a startup-window fallback and the function
returns `False` otherwise. This round tightens one thing relative to the old
implementation:

- **The direct environment-variable read is removed.** The old implementation
  fell back to a bare `os.getenv("VLLM_ASCEND_BALANCE_SCHEDULING")`, violating
  AGENTS.md's "no scattered `os.getenv`". This function no longer reads the
  environment itself — `VLLM_ASCEND_BALANCE_SCHEDULING` is parsed centrally by
  `AscendConfig` (as a deprecated fallback for `additional_config`) and takes
  effect via the main `get_ascend_config().enable_balance_scheduling` path,
  avoiding multiple entry points.
- The top-of-file TODO is updated to "once AscendConfig initialization is moved
  earlier, this can collapse to a single
  `get_ascend_config().enable_balance_scheduling` read".

> Later (once AscendConfig timing is settled): collapse the two fallbacks into a
> single read.

## Behavior-preservation contract

This refactor **must** strictly preserve the following invariants; any
deviation is a bug.

1. **Leader-at-cap ⇒ global freeze.** `balance_flag` is
   `max(balance_queue) == max_num_running_reqs`, computed on every rank from the
   previous step's gathered `len(running)`. When true, no rank admits a new
   WAITING request. The comparison is `==` against the configured
   `max_num_running_reqs` — **not** `>=`, **not** "catch up to leader".
2. **Same inputs ⇒ same outputs.** Given the same `self.running`, `self.waiting`,
   `self.skipped_waiting`, `balance_queue`, and token budget, the refactored
   `schedule()` produces a `SchedulerOutput` identical to the current
   implementation (same scheduled / preempted / resumed sets, same
   `num_scheduled_tokens`, same connector metadata).
3. **Gather cadence unchanged.** Exactly one `all_gather` per active engine step,
   on the same DP group, payload still `len(self.running)`, skipped consistently
   by all ranks when all are idle. Only the call site moved.
4. **Disabled path unchanged.** When `enable_balance_scheduling` is false,
   `_balance_run_engine_core` restores the module-level `DPEngineCoreProc` to
   the upstream original and the engine core runs upstream's implementation
   verbatim; `BalanceScheduler` with `_balance_enabled=False` delegates
   `schedule(throttle_prefills)` to `super().schedule(throttle_prefills)`, does
   not allocate `balance_queue`, and
   performs no collective communication. I.e. balance does not touch any config
   when off (including PD-disaggregated recompute / `AsyncRecomputeScheduler`,
   which is already mutually exclusive with balance via `platform.py`; this is a
   second layer of defense).
5. **Existing constraints still apply.** The `profiling_chunk_config` mutex (see
   `vllm_ascend/ascend_config.py`) and the PD-mixed-mode restriction (see
   `vllm_ascend/platform.py`) are still enforced where they were.

## Post-refactor file shape

After this round (Phase 1 + 2A + 3), the key structure is as follows. The
`schedule()` body is a verbatim copy of release tag `v0.24.0` (cannot be
deleted before Phase 2B), with three documented deltas (disabled-path early return + the `balance_flag` gate in the WAITING loop + `if request_queue is None: break`):

```python
# vllm_ascend/patch/platform/patch_balance_schedule.py
import torch
import torch.distributed as dist
import vllm.v1.core.sched.scheduler as _sched_mod
import vllm.v1.engine.core as _engine_core_mod
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc
# ... other vllm imports ...


def _balance_scheduling_enabled(vllm_config) -> bool:
    try:
        from vllm_ascend.ascend_config import get_ascend_config
        return bool(get_ascend_config().enable_balance_scheduling)
    except Exception:
        pass
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if "enable_balance_scheduling" in additional_config:
        return bool(additional_config["enable_balance_scheduling"])
    return False  # no longer reads the env var itself; VLLM_ASCEND_BALANCE_SCHEDULING is parsed by AscendConfig


class BalanceScheduler(Scheduler):
    def __init__(self, ...):
        super().__init__(...)
        self._balance_enabled = _balance_scheduling_enabled(vllm_config)
        self.dp_group = None  # injected by BalanceDPEngineCoreProc before the first gather
        if self._balance_enabled:
            self.balance_queue = [torch.tensor([0], ...) for _ in range(dp_size)]

    def balance_gather(self):  # uses self.dp_group; no-op when disabled / not injected
        if not self._balance_enabled or self.dp_group is None:
            return
        running_tensor = torch.tensor([len(self.running)], dtype=torch.int, device="cpu")
        dist.all_gather(self.balance_queue, running_tensor, group=self.dp_group)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:  # shared by v0.24.0 and e5588e49
        if not self._balance_enabled:  # delta 1: disabled-path early return
            return super().schedule(throttle_prefills)
        # NOTE: balance_gather is NOT called here -- see BalanceDPEngineCoreProc.
        # ... upstream schedule() body (verbatim-aligned to the v0.24.0 tag) ...
        #   # inside the WAITING loop (deltas 2, 3):
        #   if max(t.item() for t in self.balance_queue) == self.max_num_running_reqs:  # delta 2: leader-at-cap => global freeze
        #       break
        #   request_queue = self._select_waiting_queue_for_scheduling()
        #   if request_queue is None:  # delta 3: keep if-break (upstream has assert)
        #       break
        # ...


class BalanceDPEngineCoreProc(DPEngineCoreProc):
    """Hook _has_global_unfinished_reqs: inject dp_group + one balance_gather per
    active step. Gather MUST sit immediately after super()._has_global_unfinished_reqs()
    (the only per-iteration cross-rank sync) -- NOT inside _process_engine_step
    (which runs before that sync and before the idle continue gate, and would
    deadlock at wave boundaries -> sample_tokens timeout), and NOT inside
    schedule() (drained ranks skip schedule() and would miss the all_gather)."""

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        result = super()._has_global_unfinished_reqs(local_unfinished)
        self.scheduler.dp_group = self.dp_group
        self.scheduler.balance_gather()
        return result


_OriginalDPEngineCoreProc = _engine_core_mod.DPEngineCoreProc
_OriginalRunEngineCore = EngineCoreProc.run_engine_core


def _balance_run_engine_core(*args, dp_rank=0, local_dp_rank=0, **kwargs):
    # Conditional activation: swap the module-level DPEngineCoreProc only when balance is on.
    if _balance_scheduling_enabled(kwargs.get("vllm_config")):
        _engine_core_mod.DPEngineCoreProc = BalanceDPEngineCoreProc
    else:
        _engine_core_mod.DPEngineCoreProc = _OriginalDPEngineCoreProc
    return _OriginalRunEngineCore(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)


# Scheduler is constructed by module-global name when scheduler_cls is unset
# (the PD-mixed balance path); recompute / dynamic-batch / profiling schedulers
# set scheduler_cls and bypass this name, which is correct.
_sched_mod.Scheduler = BalanceScheduler
EngineCoreProc.run_engine_core = staticmethod(_balance_run_engine_core)
```

This round deleted the ~95-line `run_engine_core` + `run_busy_loop` copies and
their dead imports, and added the module docstring, comments, and the
`_balance_run_engine_core` conditional-activation wrapper; the net line count
barely dropped, but **what it removes is stale-drift risk** (the old copies had
fallen behind upstream's `_handle_shutdown` / `eep_scaling_state` /
`SignalCallback` evolution) and it fixes the balance-enabled deadlock (gather
first inside `schedule()`, then inside `_process_engine_step`; now after
`_has_global_unfinished_reqs`).

> Phase 2B (upstream provides `_should_stop_admitting_waiting`) is what deletes
> the ~520-line `schedule()` copy and shrinks the file to roughly **60–80 lines
> with zero verbatim upstream copy**.

## Test plan

1. **Signature + intent lock + verbatim drift test (Phase 2A).** Assert: (a)
   `BalanceScheduler.schedule`'s signature **equals the installed
   `Scheduler.schedule` signature**; both supported revisions share this
   contract, so each CI lane checks the same invariant; (b) the 3
   balance delta lines must exist in the body (disabled-path
   `super().schedule(throttle_prefills)`
   delegation, the `balance_flag` gate in the WAITING loop,
   `if request_queue is None: break`); (c) the `_balance_run_engine_core`
   wrapper is installed and `DPEngineCoreProc` is **not** swapped at import
   (deferred to the wrapper, swapped conditionally on call); (d) upstream
   `DPEngineCoreProc._has_global_unfinished_reqs` still exists (the gather
   injection point — it MUST be called every non-idle iteration or the
   all_gather deadlocks); (e) upstream `Scheduler` seam methods (including
   `_build_kv_connector_meta`, `_inflight_prefill_reserved_blocks`) still exist;
   (f) **verbatim drift detection** — first read the release tag from
   `.github/vllm-release-tag.commit` (same source as CI, **not hardcoded, not
   read from a design doc**), then `git show <tag>:vllm/v1/core/sched/scheduler.py`
   to fetch that tag's `schedule()`, strip the same 3 deltas, and AST-compare it
   verbatim against `BalanceScheduler.schedule`'s source; the two must be
   identical. Reading the pin file means a pin advance **automatically** flips
   the test to compare against the new tag and go red, signaling "the copy needs
   re-syncing" — exactly the maintenance signal we want; if the pin file or tag
   is unreachable (e.g. vLLM not a source checkout, running outside the
   vllm-ascend tree), the test is skipped, not failed.
2. **Behavior-equivalence test.** Build a `BalanceScheduler` with a fake DP
   group and a hand-set `balance_queue`, drive `schedule()` under several
   representative states (leader-at-cap freeze, lagging rank not full, disabled,
   empty waiting), and assert the `SchedulerOutput` is identical (contract item
   2). Reuse the balance test scaffolding in `tests/ut/test_platform.py`.
3. **Gather cadence test.** Mock `torch.distributed.all_gather` (note: inside
   `balance_gather`, `dist = torch.distributed`, so the mock target must be
   `torch.distributed.all_gather`, not `vllm.distributed.all_gather`); assert
   that each `balance_gather()` does exactly one `all_gather`, with payload
   `len(self.running)` and the injected dp_group (contract item 3).
4. **Disabled-path test.** With the flag off, assert `balance_queue` is not
   allocated, `all_gather` is not called, and `schedule(throttle_prefills)`
   delegates to `super().schedule(throttle_prefills)` (contract item 4).
5. **NPU performance check.** Per AGENTS.md's NPU guidance,
   `max(t.item() for t in self.balance_queue)` triggers one host sync per step
   (unavoidable, since this value drives host-side control flow). Profile to
   confirm the refactor introduces **no** extra sync beyond the current one.

## Phased rollout {: #phased-rollout }

| Phase | Scope                                                                                                                  | Risk | Depends on     | Status        |
|-------|------------------------------------------------------------------------------------------------------------------------|------|----------------|---------------|
| 1     | Hook gather onto `_has_global_unfinished_reqs` (after the cross-rank all-reduce — avoids both the schedule()-skip deadlock and the _process_engine_step wave-boundary deadlock); slim `BalanceDPEngineCoreProc` to that hook; delete the `run_engine_core`/`run_busy_loop` copies; `run_engine_core` wrapper conditionally activates `DPEngineCoreProc`; module-level `Scheduler` swap | Low  | none           | ✅ Done       |
| 2A    | Override matches the shared supported signature (`schedule(self, throttle_prefills=False)` on v0.24.0 + e5588e49); **body aligned verbatim to the release tag** (only the 3 balance deltas); disabled path delegates directly to `super()`; signature equality + intent-lock + release-tag verbatim drift tests | Low  | none           | ✅ Done       |
| 3     | Collapse config probing to two fallbacks (AscendConfig → additional_config); remove the direct env-var read (still parsed centrally by AscendConfig) | Low  | Phase 1        | ✅ Done       |
| 2B    | Upstream `_should_stop_admitting_waiting` PR; delete the `schedule()` copy                                            | Med  | upstream review | ⏳ TODO      |
| Tests | Drift regression / behavior equivalence / gather cadence / disabled path / NPU performance check                      | Low  | Phase 1 + 2A   | ⏳ TODO (needs NPU) |

Each phase can be released and rolled back independently. Phases 1, 2A, and 3
can land in the same release; 2B lands when the upstream PR merges.
