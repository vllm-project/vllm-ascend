#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Process-wide runtime-guard orchestration (config sync, detectors, actions).

Owns construction of detectors / ``ReportWriter`` / ``ActionExecutor``. Model
runners ``bind`` the process singleton once; other call sites use
:meth:`RuntimeGuardProcessor.get`.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.outputs import AsyncModelRunnerOutput

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.observability.runtime_guard.action.actions import ActionExecutor
from vllm_ascend.observability.runtime_guard.detector.manager import DetectorManager
from vllm_ascend.observability.runtime_guard.io import RequestIoSnapshotManager
from vllm_ascend.observability.runtime_guard.processor_bus import RuntimeGuardBusMixin
from vllm_ascend.observability.runtime_guard.processor_dump import RuntimeGuardDumpMixin
from vllm_ascend.observability.runtime_guard.processor_report import RuntimeGuardReportMixin
from vllm_ascend.observability.runtime_guard.rank_gate import (
    runner_tp_rank,
    should_run_wave_sync_on_rank,
)
from vllm_ascend.observability.runtime_guard.report import ReportWriter
from vllm_ascend.observability.runtime_guard.state import DumpQuota, RequestGuardStore, WaveTracker

if TYPE_CHECKING:
    from vllm_ascend.observability.runtime_config.config import RuntimeConfig

logger = init_logger_ascend(__name__)

# Shutdown must not hang forever on a stuck due-broadcast (dead peer): bound
# the final merged-bus drain. End-of-wave drains stay unbounded (lockstep).
_SHUTDOWN_BUS_DRAIN_TIMEOUT_S = 10.0


@dataclass
class SamplePhaseResult:
    """Runner-side sample-phase outputs needed by post-sample runtime_guard hooks.

    Returned by the ``sample_fn`` callback passed to
    :meth:`RuntimeGuardProcessor.run_sample_phase`. Bundles the values the runner
    already computes (``ModelRunnerOutput`` + ``sampler_output`` + the
    bookkeeping sync outputs) so hooks 3-8 don't need to re-fetch them.
    """

    scheduler_output: Any
    input_batch: Any
    model_runner_output: Any
    sampler_output: Any
    valid_sampled_token_ids: Any
    req_ids_output_copy: Any
    invalid_req_indices: Any
    finished_req_ids: Any


class RuntimeGuardProcessor(RuntimeGuardBusMixin, RuntimeGuardDumpMixin, RuntimeGuardReportMixin):
    """Process-wide singleton: config sync → detect → report / dump_kv.

    Create / attach a runner with :meth:`bind` (or ``RuntimeGuardProcessor(runner)``).
    Retrieve with :meth:`get`. One worker process should bind at most one model runner.
    """

    _instance: ClassVar[RuntimeGuardProcessor | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls, runner: Any | None = None):
        # ``RuntimeGuardProcessor(runner)`` is an alias of :meth:`bind`.
        if runner is None:
            raise TypeError("RuntimeGuardProcessor() requires a model runner; use bind(runner)")
        return cls.bind(runner)

    def __init__(self, runner: Any | None = None) -> None:
        # Initialization is done in :meth:`bind` / ``_init_from_runner``.
        return

    @classmethod
    def get(cls) -> RuntimeGuardProcessor:
        """Return the process singleton. Raises if :meth:`bind` has not run."""
        inst = cls._instance
        if inst is None:
            raise RuntimeError("RuntimeGuardProcessor is not bound; call RuntimeGuardProcessor.bind(runner) first")
        return inst

    @classmethod
    def try_get(cls) -> RuntimeGuardProcessor | None:
        """Return the process singleton, or None if not bound yet."""
        return cls._instance

    @classmethod
    def bind(cls, runner: Any) -> RuntimeGuardProcessor:
        """Create or rebind the process singleton to ``runner``."""
        if runner is None:
            raise ValueError("RuntimeGuardProcessor.bind requires a model runner")
        with cls._lock:
            if cls._instance is None:
                inst = object.__new__(cls)
                inst._init_from_runner(runner)
                cls._instance = inst
            else:
                cls._instance._rebind_runner(runner)
            return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        """Drop singleton (unit tests only)."""
        with cls._lock:
            inst = cls._instance
            cls._instance = None
        if inst is not None:
            with contextlib.suppress(Exception):
                inst.shutdown()

    def _init_from_runner(self, runner: Any) -> None:
        ascend = runner.ascend_config
        runtime_config: RuntimeConfig = ascend.runtime_config

        self.runner = runner
        self.runtime_config = runtime_config
        # Leader materializes JSON once. Non-leaders no-op inside ensure_persisted.
        runtime_config.ensure_persisted()
        # Runtime config is solely ``runtime_config`` (JSON).
        self.wave_tracker = WaveTracker()
        # B1: reap must wait for the last async output of a finished request;
        # WaveTracker.pending is the drain signal (replaces the dead store FIFO).
        RequestGuardStore.get().set_drain_probe(self.wave_tracker.pending)
        self.quota = DumpQuota(runtime_config)
        # Detection + report only on last-PP TP0; all ranks share the same report root.
        self.report_writer = ReportWriter(
            runtime_config.report_dir,
            save_sensitive_info=runtime_config.report_save_sensitive_info(),
            max_prompt_token_ids=runtime_config.report_max_prompt_token_ids(),
            max_output_token_ids=runtime_config.report_max_output_token_ids(),
            decode_token_ids=runtime_config.report_decode_token_ids(),
            max_per_req=runtime_config.report_max_per_req(),
            dump_root_provider=runtime_config.dump_root,
        )
        self.action_executor = ActionExecutor(
            runner,
            runtime_config=runtime_config,
            report_writer=self.report_writer,
            quota=self.quota,
        )
        self.action_executor.start()
        # Merged due-bit AR (+ rare due bcasts) overlaps forward on this thread.
        from vllm_ascend.observability.runtime_guard.bus_worker import DueBitsBusWorker

        self._bus_worker = DueBitsBusWorker()
        self._bus_worker.start()
        self._merged_bus_inflight = False
        self._pending_merged_bus_dump_jobs: list[dict[str, Any]] = []
        self._pending_merged_bus_can_dump = False
        self._pending_merged_bus_is_first = False
        self._tokenizer: Any | None = None
        self._tokenizer_failed = False
        self._scheduler_output_for_step: Any | None = None
        self._last_input_batch: Any | None = None
        self._kv_dump_jobs: list[dict[str, Any]] = []
        # Auto dump jobs delivered at wave-head bcast; D2H runs at end-of-wave
        # (after prepare). Same-wave detector arms wait in ``_kv_dump_jobs``
        # until the *next* wave head (accepted +1 wave latency).
        self._deferred_kv_dump_jobs: list[dict[str, Any]] = []
        self._bus_wave_seq = 0
        self._merged_bus_warn_ts = 0.0
        self.detectors = DetectorManager(
            runtime_config=runtime_config,
            runner=runner,
            detection_gate=self.action_executor.can_run_detection,
            detection_skip_reason=self.action_executor.anomaly_check_skip_reason,
        )

    def _rebind_runner(self, runner: Any) -> None:
        """Point nested components at a new runner (same process, rare rebuild)."""
        if runner is self.runner:
            return
        logger.info("[runtime_guard] rebinding processor singleton to a new runner")
        ascend = runner.ascend_config
        runtime_config: RuntimeConfig = ascend.runtime_config
        self.runner = runner
        self.runtime_config = runtime_config
        self.action_executor.rebind_runner(runner, runtime_config=runtime_config)
        self.detectors.rebind_runner(runner)
        # Tokenizer may differ across runners; force lazy reload.
        self._tokenizer = None
        self._tokenizer_failed = False
        self.action_executor.start()
        if not self._bus_worker.started:
            self._bus_worker.start()

    def shutdown(self) -> None:
        """Stop async workers (process teardown / tests)."""
        try:
            # Finish any in-flight merged bus before tearing the worker down.
            # Bounded: a stuck collective (dead peer) must not hang teardown.
            with contextlib.suppress(Exception):
                self._drain_merged_bus(warn_if_pending=False, timeout=_SHUTDOWN_BUS_DRAIN_TIMEOUT_S)
            self._bus_worker.stop()
        except Exception:
            logger.debug("[runtime_guard] bus worker stop failed", exc_info=True)
        try:
            self.action_executor.stop()
        except Exception:
            logger.debug("[runtime_guard] action worker stop failed", exc_info=True)

    # ---- step entry (all ranks) --------------------------------------------

    def refresh_config(
        self,
        *,
        scheduler_output: Any | None = None,
    ) -> bool:
        """All-rank runtime_config sync. Must not be skipped on early PP.

        Manual-dump arming is gated by ``sync_for_step`` /
        ``end_of_wave_sync(allow_manual_dump=…)``, not by this method.
        """
        logger.debug("[runtime_guard sync] enter stage=refresh_config")
        prev_so = self._scheduler_output_for_step
        so = scheduler_output if scheduler_output is not None else prev_so
        self._scheduler_output_for_step = so
        try:
            changed = self._refresh_config_body()
            # Clear IO wave cache only when something may append this step.
            # Runs *after* sync so a hot-reload that turns IO consumers on
            # still clears before sample/detect. Idle A (detectors off) skips.
            if self.runtime_config.needs_cumulative_io():
                RequestIoSnapshotManager.get().clear_wave_cache()
            return changed
        finally:
            self._scheduler_output_for_step = prev_so

    def sync_for_step(
        self,
        *,
        allow_manual_dump: bool = True,
        scheduler_output: Any | None = None,
    ) -> None:
        """Lockstep runtime_guard sync for one engine wave (real step or idle dummy).

        Wave-head: merged config+dump TP0 due-broadcast (idle pays only that).
        End-of-wave: deferred auto D2H + local manual dump (no dump collective).

        Must run on every **last-PP** rank of that EngineCore each wave —
        including idle DP ranks that take ``execute_dummy_batch`` and skip
        ``execute_model``. Non-last PP returns immediately (no cross-PP bus;
        detect/dump already gated elsewhere). Do not put this inside
        ``_dummy_run``: ``execute_model`` may already sync then call
        ``_dummy_run``. Never use a cross-DP full-world collective for config
        hot-reload.

        ``scheduler_output`` (optional): lets MRV2 see scheduled tokens for
        later manual dump at end-of-wave after prepare.
        """
        if not should_run_wave_sync_on_rank():
            return
        # Rank introspection for debug logs only: skip the getattrs and the
        # get_pp_group() try/except when DEBUG is off (guard-all-off zero cost).
        debug_on = logger.isEnabledFor(logging.DEBUG)
        if debug_on:
            runner = self.runner
            dp = getattr(runner, "dp_rank", "?")
            tp = getattr(runner, "tp_rank", "?")
            try:
                pp = get_pp_group().rank_in_group
            except Exception:
                pp = "?"
            logger.debug(
                "[runtime_guard sync] enter sync_for_step allow_manual_dump=%s dp=%s tp=%s pp=%s",
                allow_manual_dump,
                dp,
                tp,
                pp,
            )
        self._scheduler_output_for_step = scheduler_output
        try:
            self.wave_tracker.advance(allow_manual_dump=allow_manual_dump)
            cfg = self.runtime_config
            idle = not cfg.manual_trigger() and not cfg.needs_sample_phase_hooks()
            if idle and not cfg.hot_reload_enabled:
                # Static idle: no config bus. Claim leftover auto jobs only when
                # dump is active (shared gate → all ranks take the same branch).
                if cfg.dump_enabled():
                    self._claim_dump_jobs_to_deferred_via_tp()
                elif self._kv_dump_jobs:
                    # Dump inactive: drop with per-arm refund (quota was
                    # consumed at arm time; no D2H will happen).
                    self._drop_pending_dump_jobs()
                self._end_of_wave_sync_if_no_sample(allow_manual_dump=allow_manual_dump)
                return
            self.refresh_config(scheduler_output=scheduler_output)
            if self.needs_sample_phase_hooks():
                self._cache_prompt_token_ids_from_scheduler_output(scheduler_output)
                finished = getattr(scheduler_output, "finished_req_ids", None)
                if finished and int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0) == 0:
                    self.mark_finished(finished)
                self._reap_finished_requests()
            self._end_of_wave_sync_if_no_sample(allow_manual_dump=allow_manual_dump)
        finally:
            self._scheduler_output_for_step = None
            if debug_on:
                logger.debug(
                    "[runtime_guard sync] leave sync_for_step allow_manual_dump=%s dp=%s tp=%s pp=%s",
                    allow_manual_dump,
                    dp,
                    tp,
                    pp,
                )

    def _end_of_wave_sync_if_no_sample(self, *, allow_manual_dump: bool) -> None:
        """Safety-net end-of-wave when this wave will not ``run_sample_phase``."""
        if allow_manual_dump:
            return
        self.end_of_wave_sync(allow_manual_dump=False)

    def _cache_prompt_token_ids_from_scheduler_output(
        self,
        scheduler_output: Any | None,
    ) -> None:
        """Capture prompt_token_ids from each scheduled new request.

        Bug #9 fix: v2 ``RequestState.all_token_ids`` is a StagedWriteTensor
        whose host mirror stays 0 until ``apply_staged_writes`` commits, and
        ``_remove_request`` pops the id from ``req_id_to_index`` on finish —
        both windows make the snapshot path return zeros or empty. Cache the
        ids here on the first prefill wave so later snapshots (crash or
        finish) read the real prompt. Idempotent per request.
        """
        if scheduler_output is None:
            return
        new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
        if not new_reqs:
            return
        store = RequestGuardStore.get()
        for req in new_reqs:
            req_id = getattr(req, "req_id", None)
            if not req_id:
                continue
            ids = getattr(req, "prompt_token_ids", None)
            if ids is None:
                continue
            store.set_prompt_token_ids(str(req_id), ids)

    # ---- sample / get_output hooks ----------------------------------------

    def mark_finished(self, finished_req_ids: Any) -> None:
        """Mark requests finished; defer Store.clear until last sample is consumed.

        Runner order is ``mark_finished`` → (optional) ``record_sample_waves``
        → ``check_after_sample`` / async ``get_output``. Only Store
        :meth:`~RequestGuardStore.mark_finished` runs here so the last stamp /
        detect / append still see the same state. Sidecars + clear happen in
        :meth:`_reap_finished_requests`.
        """
        if not finished_req_ids:
            return
        RequestGuardStore.get().mark_finished(finished_req_ids, wave=self._current_wave())

    def _current_wave(self) -> int:
        return int(self.wave_tracker.current_wave())

    def _reap_finished_requests(self) -> None:
        """Clear reqs that are finished and drained."""
        store = RequestGuardStore.get()
        wave = self._current_wave()
        reapable = store.list_reapable(current_wave=wave)
        if not reapable:
            return
        store.clear_many(reapable, detectors=self.detectors)
        self.wave_tracker.discard_many(reapable)

    def should_check_after_spec(self) -> bool:
        if not self.action_executor.can_run_detection():
            return False
        return self.detectors.any_enabled_for_spec()

    def needs_sample_phase_hooks(self) -> bool:
        """True when sample-phase runtime_guard hooks must run (else pure ``sample_fn``)."""
        return bool(self.runtime_config.needs_sample_phase_hooks())

    def _soft_fail(self, hook: str, fn: Callable[[], Any]) -> Any:
        # Guard hooks are observational: any exception must stay inside the
        # guard and never reach the engine loop / async copy thread.
        try:
            return fn()
        except Exception:
            logger.exception("[runtime_guard soft-fail] hook=%s raised; skipped this step", hook)
            return None

    def check_after_spec(
        self,
        sampled_tokens: Any,
        accepted_token_nums: Any,
        req_ids: list[str] | None = None,
    ) -> None:
        """Speculative step hook: record accepted tokens + run registered spec detectors.

        Callers should gate with :meth:`should_check_after_spec` when they need
        to skip expensive accepted-count materialization.
        """

        def _run() -> None:
            for alert in self.detectors.check_after_spec(sampled_tokens, accepted_token_nums, req_ids=req_ids):
                self._handle_alert(alert, detector=self.detectors.get(alert.incident_type))

        self._soft_fail("check_after_spec", _run)

    def record_sample_waves(self, req_ids: list[str] | None) -> None:
        self.wave_tracker.record_sample_waves(req_ids)

    def _should_record_sample_waves(self, *, use_async: bool) -> bool:
        """Async: only TP0 (output rank) stamps — matches AscendAsync* wrap.

        Non-TP0 never runs ``get_output`` / ``take_sample_wave`` under async
        scheduling; recording there would leave ``pending`` stamps until
        ``max_deferred_waves`` force-reap.
        """
        if not use_async:
            return True
        try:
            return runner_tp_rank(self.runner) == 0
        except Exception:
            return True

    # ---- single sink for post-pre-sample runtime_guard hooks ---------------

    def run_sample_phase(
        self,
        *,
        sample_fn: Callable[[], SamplePhaseResult],
        speculative_config: Any,
        need_accepted_tokens: bool,
        use_async: bool,
        async_state_update_fn: Callable[[SamplePhaseResult], None] | None = None,
        routed_experts_fn: Callable[[SamplePhaseResult], Any] | None = None,
        accepted_token_nums_fn: Callable[[SamplePhaseResult], Any] | None = None,
    ) -> tuple[SamplePhaseResult, Any]:
        """Single sink for post-pre-sample runtime_guard hooks.

        Replaces 7 inline ``self.runtime_guard.*`` calls scattered across
        ``NPUModelRunner.sample_tokens`` with one orchestration call so
        hook ordering is owned by ``RuntimeGuardProcessor`` rather than the runner.

        Hook 1 (``check_before_sample``) stays on the compute_logits wrap inside
        :func:`~vllm_ascend.observability.runtime_guard.hooks.runtime_guard_sample_tokens`
        so it fires BEFORE grammar bitmask (v2 only).

        Hook sequence (``S1`` golden path):
            2. ``sample_fn()`` returns :class:`SamplePhaseResult`
            3. ``mark_finished``
            -> ``async_state_update_fn`` (only if ``need_accepted_tokens``)
            4. ``check_after_spec`` (spec only; ``accepted_token_nums_fn`` for branch)
            -> ``routed_experts_fn`` (async path: BEFORE wave stamp; sync: AFTER check_after_sample)
            5. ``record_sample_waves``
            6. ``check_after_sample`` (host-ready ids only; see defer note below)
            7. ``end_of_wave_sync`` (config+dump gate; after sync auto arm when possible)

        Native KV capture uses ``dump_kv`` actions only (fully decoupled from
        Ascend/msprobe PrecisionDebugger dump).

        After-sample deferral: when ``use_async`` is set **or** the runner
        returned an :class:`~vllm.v1.outputs.AsyncModelRunnerOutput` (v2 always
        does, even under sync scheduling), hook 6 is skipped here.
        ``AscendAsync*`` ``get_output`` runs it after D2H + ``num_sampled`` trim.
        Appending ``AsyncOutput.sampled_token_ids`` (padded numpy) here inflated
        cumulative IO and false-triggered ``token_repeat`` (W2-3 / D-11).
        """
        # Idle fast-path: detectors all off →
        # skip soft-fail wrappers and observational hooks entirely.
        # Still flush: TP>1 drain collectives must stay lockstep every sample.
        if not self.needs_sample_phase_hooks():
            result = sample_fn()
            if need_accepted_tokens and async_state_update_fn is not None:
                async_state_update_fn(result)
            routed_experts_result = None
            if routed_experts_fn is not None:
                routed_experts_result = routed_experts_fn(result)
            # End-of-wave gate uses collectives — do not soft-fail (desync/hang).
            self.end_of_wave_sync(allow_manual_dump=True)
            return result, routed_experts_result

        # Runner's sample work (sample + draft + bookkeeping + output + profiling + eplb)
        result = sample_fn()
        # Retain wave batch for block_ids after MRV2 clears execute_model_state.
        if result.input_batch is not None:
            self._last_input_batch = result.input_batch
        # Hook 3: mark_finished
        self._soft_fail("mark_finished", lambda: self.mark_finished(result.finished_req_ids))
        # Async state update callback (between mark_finished and check_after_spec)
        if need_accepted_tokens and async_state_update_fn is not None:
            async_state_update_fn(result)
        # Hook 4: check_after_spec (spec only). Gate here to skip the
        # accepted_token_nums_fn callback when detection is off; soft-fail
        # lives inside check_after_spec itself.
        if speculative_config is not None and self.should_check_after_spec():
            if accepted_token_nums_fn is not None:
                accepted_token_nums = accepted_token_nums_fn(result)
            else:
                accepted_token_nums = None
            self.check_after_spec(
                sampled_tokens=result.sampler_output.sampled_token_ids,
                accepted_token_nums=accepted_token_nums,
                req_ids=result.req_ids_output_copy,
            )
        # Async path: routed_experts computed BEFORE wave stamp
        routed_experts_result = None
        if use_async and routed_experts_fn is not None:
            routed_experts_result = routed_experts_fn(result)
        # Hook 5: record_sample_waves (sync: all ranks; async: output-rank TP0 only)
        if self._should_record_sample_waves(use_async=use_async):
            self._soft_fail(
                "record_sample_waves",
                lambda: self.record_sample_waves(result.req_ids_output_copy),
            )
        # Hook 6: check_after_sample only when host ids are already trimmed.
        # v2 returns AsyncOutput under sync scheduling too — pad rows must wait
        # for AscendAsyncOutput.get_output (executor always calls get_output).
        defer_after_sample = use_async or isinstance(
            getattr(result, "model_runner_output", None), AsyncModelRunnerOutput
        )
        if not defer_after_sample:
            self._soft_fail(
                "check_after_sample",
                lambda: self.check_after_sample(
                    sampled_token_ids=result.valid_sampled_token_ids,
                    req_ids=result.req_ids_output_copy,
                ),
            )
        # Sync path: routed_experts computed AFTER check_after_sample
        if not use_async and routed_experts_fn is not None:
            routed_experts_result = routed_experts_fn(result)
        # Hook 7: end-of-wave config+dump gate (collectives — no soft-fail).
        self.end_of_wave_sync(allow_manual_dump=True)
        return result, routed_experts_result

    def check_before_sample(
        self,
        *,
        logits: Any,
        logits_indices: Any = None,
        input_batch: Any = None,
    ) -> None:
        """Pre-sample hook: ``logits_finite`` (and future pre-sample detectors)."""
        self._last_input_batch = input_batch

        def _run() -> None:
            for alert in self.detectors.check_before_sample(
                logits=logits,
                logits_indices=logits_indices,
                input_batch=input_batch,
            ):
                self._handle_alert(alert, detector=self.detectors.get(alert.incident_type))

        self._soft_fail("check_before_sample", _run)

    def check_after_sample(
        self,
        sampled_token_ids: Any,
        req_ids: list[str] | None = None,
    ) -> None:
        """Sample-step hook: drain logits_finite + enqueue CPU detect.

        `logits_finite` already launched / resolved its gate on before-sample
        (async ``.item()`` wait completes here; ``item_sync`` resolved earlier);
        here we drain those incidents. Wave stamp + IO append stay on this
        thread (sync sample or async ``get_output``).
        ``token_repeat`` runs later on ActionQueue;
        request finish does not wait. ``dump_kv`` may still arm after
        ``mark_finished`` (stamps ``request_finished_at_dump``); reaped ids
        are refused at arm.
        """

        def _run() -> None:
            wave_by_req: dict[str, int] = {}
            wave_tracker = self.wave_tracker
            async_sched = bool(getattr(self.runner, "use_async_scheduling", False))
            ids = list(req_ids) if req_ids else []
            if async_sched and not ids:
                logger.warning_once(
                    "[runtime_guard wave] async check_after_sample without req_ids; "
                    "arm_wave will fall back to current_wave (may race advance_wave)"
                )
            for rid in ids:
                if not rid:
                    continue
                rid_s = str(rid)
                stamped = wave_tracker.take_sample_wave(rid_s)
                if stamped is not None:
                    wave_by_req[rid_s] = stamped
                elif async_sched:
                    logger.warning(
                        "[runtime_guard wave] missing sample-wave stamp for req_id=%s under async "
                        "scheduling; arm_wave falls back to current_wave (may be polluted)",
                        rid_s,
                    )
            logits_alerts, snap = self.detectors.after_sample_hot_path(
                sampled_token_ids,
                req_ids=req_ids,
            )
            log_sampling_meta_debug(self.runner, req_ids)
            for alert in logits_alerts:
                arm_wave = wave_by_req.get(alert.req_id) if alert.req_id else None
                self._handle_alert(
                    alert,
                    detector=self.detectors.get(alert.incident_type),
                    arm_wave=arm_wave,
                )
            if snap is not None:
                # Chunked-prefill discard clears sampled rows: skip empty CPU
                # jobs so ActionQueue is not flooded (W1-3 / R-06 long ctx).
                # Store-only token_repeat fold needs a job only when this step
                # produced at least one non-empty row (append already ran).
                if snap.has_nonempty_sampled:
                    self._enqueue_after_sample_cpu(snap, wave_by_req)
            self._reap_finished_requests()

        self._soft_fail("check_after_sample", _run)

    def _enqueue_after_sample_cpu(
        self,
        snap: Any,
        wave_by_req: dict[str, int],
    ) -> None:
        """Submit CPU after-sample detect; never run inline on get_output."""
        req_ids_job = [rid for rid in snap.req_ids if rid]
        store = RequestGuardStore.get()
        store.add_cpu_jobs(req_ids_job)

        def _cpu_job(
            snap: Any = snap,
            wave_by_req: dict[str, int] = wave_by_req,
            req_ids_job: list[str] = req_ids_job,
        ) -> None:
            try:
                for alert in self.detectors.run_after_sample_cpu(snap):
                    arm_wave = wave_by_req.get(alert.req_id) if alert.req_id else None
                    self._handle_alert(
                        alert,
                        detector=self.detectors.get(alert.incident_type),
                        arm_wave=arm_wave,
                    )
            except Exception:
                logger.exception("[runtime_guard] after-sample CPU detect failed")
            finally:
                RequestGuardStore.get().finish_cpu_jobs(req_ids_job)
                self._reap_finished_requests()

        queue = self.action_executor.action_queue
        ok = bool(queue.submit(_cpu_job, heavy=True))
        if not ok:
            logger.warning(
                "[runtime_guard] after-sample CPU detect dropped (queue full or stopping); "
                "token_repeat may miss this step req_ids=%s",
                req_ids_job,
            )
            store.finish_cpu_jobs(req_ids_job)


# ---- sampling meta debug ----


def log_sampling_meta_debug(runner: Any, req_ids: list[str] | None) -> None:
    """DEBUG ``[SamplingMeta]`` for local-batch reqs (TP0 + last PP).

    Gated by logger level only (no JSON switch). Skips ``.item()`` work when
    DEBUG is off. Soft-fail: never raise into the sample path.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    try:
        _emit_sampling_meta_debug(runner, req_ids)
    except Exception as exc:
        logger.debug("[runtime_guard] SamplingMeta log failed: %s", exc, exc_info=True)


def _emit_sampling_meta_debug(runner: Any, req_ids: list[str] | None) -> None:
    # Process-group TP rank (v1 runners have no reliable ``tp_rank`` attr).
    if runner_tp_rank(runner) != 0:
        return
    if not get_pp_group().is_last_rank:
        return

    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return
    sampling_metadata = getattr(input_batch, "sampling_metadata", None)
    if sampling_metadata is None:
        return

    batch_ids = list(getattr(input_batch, "req_ids", None) or [])
    want = {str(r) for r in (req_ids or []) if r} if req_ids else set(batch_ids)
    if not want:
        return

    for req_idx, req_id in enumerate(batch_ids):
        if not req_id or str(req_id) not in want:
            continue

        temp = sampling_metadata.temperature[req_idx].item() if sampling_metadata.temperature is not None else None
        topk = sampling_metadata.top_k[req_idx].item() if sampling_metadata.top_k is not None else None
        topp = sampling_metadata.top_p[req_idx].item() if sampling_metadata.top_p is not None else None

        freq_pen = sampling_metadata.frequency_penalties[req_idx].item()
        pres_pen = sampling_metadata.presence_penalties[req_idx].item()
        rep_pen = sampling_metadata.repetition_penalties[req_idx].item()

        bad_words = sampling_metadata.bad_words_token_ids
        req_bad_words = bad_words.get(req_idx, []) if bad_words else []
        req_output_tokens = (
            sampling_metadata.output_token_ids[req_idx]
            if sampling_metadata.output_token_ids and req_idx < len(sampling_metadata.output_token_ids)
            else []
        )
        req_spec_tokens = (
            sampling_metadata.spec_token_ids[req_idx]
            if sampling_metadata.spec_token_ids and req_idx < len(sampling_metadata.spec_token_ids)
            else None
        )
        if sampling_metadata.logprob_token_ids:
            req_logprob_tokens = sampling_metadata.logprob_token_ids.get(req_idx, [])
        else:
            req_logprob_tokens = None

        logger.debug(
            "[SamplingMeta] req_id=%s req_idx=%d "
            "dp_rank=%d tp_rank=%d "
            "temperature=%.4f top_k=%s top_p=%.4f "
            "freq_pen=%.4f pres_pen=%.4f rep_pen=%.4f "
            "bad_words_group_num=%d output_tokens_len=%d spec_tokens_len=%s logprob_target_tokens_len=%s "
            "all_greedy=%s all_random=%s max_num_logprobs=%s",
            req_id,
            req_idx,
            runner.dp_rank,
            runner.tp_rank,
            temp if temp is not None else -1,
            topk,
            topp if topp is not None else 1.0,
            freq_pen,
            pres_pen,
            rep_pen,
            len(req_bad_words),
            len(req_output_tokens),
            len(req_spec_tokens) if req_spec_tokens else None,
            len(req_logprob_tokens) if req_logprob_tokens else None,
            sampling_metadata.all_greedy,
            sampling_metadata.all_random,
            sampling_metadata.max_num_logprobs,
        )
