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

"""Dumper facade: anomaly dump orchestration over msprobe + pending-OR mixins."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group

from vllm_ascend.dfx.dfx_types import DumpFinishMeta, DumpPhase
from vllm_ascend.dfx.dumper.msprobe import MsprobeBridgeMixin
from vllm_ascend.dfx.dumper.pending import PendingDumpMixin, anomaly_check_rank_skip_reason
from vllm_ascend.dfx.request_state import RequestDfxStore
from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig
from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)
if TYPE_CHECKING:
    from vllm_ascend.dfx.detector.alert import AnomalyAlert
    from vllm_ascend.dfx.detector.base import AnomalyDetector
    from vllm_ascend.dfx.manual_trigger import TriggerEvent
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2


class Dumper(PendingDumpMixin, MsprobeBridgeMixin):
    """Manages msprobe debugger lifecycle and dump arming/activation.

    Detectors live on ``DfxProcessor``; this class reacts to ``AnomalyAlert``
    via :meth:`handle_anomaly_alert`. Runtime knobs come only from the
    injected ``dfx_config`` (JSON). Construct via ``DfxProcessor`` (or tests
    that pass ``dfx_config=`` explicitly).

    Implementation is split across:
    - :class:`MsprobeBridgeMixin` — debugger / dump_enable / start·finalize
    - :class:`PendingDumpMixin` — last-PP TP pending-OR and quota

    Coarse lifecycle is exposed as :attr:`dump_phase` (``DumpPhase``); fine
    forward gating still uses ``_dump_needs_forward`` / ``_dump_forward_seen``.
    """

    def __init__(
        self,
        runner: NPUModelRunner | NPUModelRunnerV2,
        *,
        dfx_config: DfxRuntimeConfig,
    ):
        self.runner = runner
        self.dfx_config = dfx_config
        self._debugger: Any | None = None
        # B6 fix: protect _debugger lifecycle against
        # hot-reload recreate racing in-flight finalize_dump_data.step().
        self._lock = threading.RLock()

        self.sync_dump_limits_from_config()

        self._msprobe_dump_total_count = 0
        self._msprobe_dumped_req_ids: set[str] = set()
        self._msprobe_last_dump_ts: float | None = None
        self._msprobe_dump_active = False
        # After enable: require one start→finalize(dump) pair before disable.
        # Avoids async check enabling dump mid-step then finalize turning it
        # off before any dump-capable forward runs.
        self._dump_needs_forward = False
        self._dump_forward_seen = False
        self._debugger_started = False
        # AclGraphDumper must patch before graph capture (``_running`` stays on).
        # Tensor write is gated by device ``switch`` ↔ msprobe ``dump_enable``
        # (see ``set_msprobe_dump_state`` / ``_sync_aclgraph_dump_enable``).
        self._uses_aclgraph_dumper = False
        self._aclgraph_hooks_installed = False
        # Async cross-rank alignment: check only arms pending; execute_model
        # entry ORs last-PP TP pending (early PP skipped; no PP broadcast).
        self._pending_dump = False
        self._pending_dump_req_id: str | None = None
        # Manual trigger: activate without consuming max_times / cooldown.
        self._pending_dump_skip_quota = False
        # Real-step wave index (advanced on allow_arm sync_for_step only).
        self._wave_index = 0
        # Open arm→activate tracking (committed into RequestDfxStore on activate).
        self._open_dump_arm_wave: int | None = None
        self._open_dump_finish_req_ids: list[str] = []
        self._open_dump_anomaly_type: str | None = None
        self._open_dump_source: str | None = None

        # True after construction-time debugger probe finishes. Late success
        # under ACLGraph may miss already-captured graphs.
        self._startup_debugger_done = False
        # Last path applied to ascend_config / debugger (hot-reload recreate).
        self._applied_msprobe_config_path: str | None = getattr(
            getattr(runner, "ascend_config", None), "dump_config_path", None
        )

        self._apply_observability_switches()

        logger.info_once(
            "DFX ready config=%s report_dir=%s dump.active=%s auto_max_times=%d "
            "ascend_log.level=%s ascend_log.debug=%s ascend_log.modules=%s "
            "spec_check=%s token_logprob_check=%s output_substring_check=%s "
            "token_repeat_check=%s "
            "report.save_sensitive_info=%s",
            str(self.dfx_config.config_path),
            str(self.dfx_config.report_dir),
            self.dfx_config.dump_enabled(),
            self._dump_max_times,
            self.dfx_config.ascend_log_level(),
            # info_once is lru_cached; args must be hashable (not list/dict).
            tuple(self.dfx_config.ascend_log_debug_modules()),
            tuple(sorted(self.dfx_config.ascend_log_modules().items())),
            bool(self.dfx_config.detector_get("spec_acceptance", "enabled", False)),
            bool(self.dfx_config.detector_get("token_logprob", "enabled", False)),
            bool(self.dfx_config.detector_get("output_substring", "enabled", False)),
            bool(self.dfx_config.detector_get("token_repeat", "enabled", False)),
            self.dfx_config.report_save_sensitive_info(),
        )

        # Keep debugger lifecycle fully encapsulated in Dumper. Soft-fail when
        # msprobe / dump_config is missing; force dump off if still active.
        # ACLGraph: path set ⇒ prebuild even when dump capability is off so
        # load_model/start_dump_data can install hooks before capture.
        self._try_init_debugger()
        self._enforce_dump_requires_debugger()
        # Always idle-close when a debugger exists (covers dump-off prebuild).
        self._close_idle_msprobe_dump_gate()
        self._startup_debugger_done = True
        self._applied_msprobe_config_path = getattr(
            getattr(self.runner, "ascend_config", None), "dump_config_path", None
        )

    def _try_init_debugger(self) -> None:
        """Init debugger once when absent (startup or lazy reload retry)."""
        if self._debugger is not None:
            return
        self._init_debugger(self.runner.compilation_config.cudagraph_mode)

    def _teardown_debugger(self) -> None:
        """Drop the current msprobe debugger so it can be reconstructed.

        B6 fix: hold ``self._lock`` so in-flight ``finalize_dump_data.step()``
        on another thread cannot use a debugger we are about to stop/drop.
        """
        with self._lock:
            if bool(getattr(self, "_msprobe_dump_active", False)):
                with suppress(Exception):
                    self.set_msprobe_dump_state(False)
                self._msprobe_dump_active = False
            self._pending_dump = False
            self._pending_dump_req_id = None
            self._pending_dump_skip_quota = False
            self._dump_needs_forward = False
            self._dump_forward_seen = False
            dbg = getattr(self, "_debugger", None)
            if dbg is not None and hasattr(dbg, "stop"):
                with suppress(Exception):
                    dbg.stop()
            self._debugger = None
            self._debugger_started = False
            self._aclgraph_hooks_installed = False
            self._uses_aclgraph_dumper = False

    def recreate_msprobe_debugger(self, *, reason: str = "config") -> bool:
        """Tear down and rebuild PrecisionDebugger / AclGraphDumper.

        Updates ``ascend_config.dump_config_path`` from
        ``dump.msprobe_config_path`` when set. Returns True if a recreate ran.

        B6 fix: serialized against in-flight ``finalize_dump_data.step()``
        via ``self._lock``; teardown + init are one atomic swap.
        """
        with self._lock:
            return self._recreate_msprobe_debugger_locked(reason=reason)

    def _recreate_msprobe_debugger_locked(self, *, reason: str = "config") -> bool:
        """Inner recreate body; caller holds ``self._lock``."""
        cfg_path = self.dfx_config.dump_msprobe_config_path()
        ascend = getattr(self.runner, "ascend_config", None)
        if cfg_path and ascend is not None:
            ascend.dump_config_path = cfg_path
        target = getattr(ascend, "dump_config_path", None) if ascend is not None else None

        was_aclgraph = bool(getattr(self, "_uses_aclgraph_dumper", False)) or (
            bool(getattr(self, "_startup_debugger_done", False))
            and getattr(self.runner, "compilation_config", None) is not None
            and self.runner.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        )
        logger.info(
            "[Anomaly msprobe] recreating debugger reason=%s path=%s %s",
            reason,
            target,
            self.dump_rank_tag(),
        )
        if was_aclgraph:
            logger.warning(
                "[Anomaly msprobe] recreating under ACLGraph/cudagraph — dump "
                "may be empty until worker restart if graphs were already "
                "captured %s",
                self.dump_rank_tag(),
            )
        elif bool(getattr(self, "_startup_debugger_done", False)):
            logger.warning(
                "[Anomaly msprobe] recreating debugger (reason=%s) — dump may be "
                "empty until worker restart OR a forward pass re-installs hooks. "
                "Recommend: edit the msprobe JSON file content directly "
                "(msprobe self-refresh) instead of reload_msprobe=true when the "
                "path is unchanged %s",
                reason,
                self.dump_rank_tag(),
            )
        self._teardown_debugger()
        self._init_debugger(self.runner.compilation_config.cudagraph_mode)
        self._reinstall_msprobe_hooks_after_recreate()
        self._enforce_dump_requires_debugger()
        self._close_idle_msprobe_dump_gate()
        self._applied_msprobe_config_path = getattr(ascend, "dump_config_path", None) if ascend else None
        return True

    def maybe_recreate_msprobe_debugger(self) -> bool:
        """Recreate debugger when path changes or ``reload_msprobe`` is set."""
        reload = self.dfx_config.dump_reload_msprobe()
        cfg_path = self.dfx_config.dump_msprobe_config_path()
        ascend = getattr(self.runner, "ascend_config", None)
        current = getattr(ascend, "dump_config_path", None) if ascend is not None else None
        # Prefer DFX path when set; else keep ascend path.
        desired = cfg_path if cfg_path is not None else current
        applied = getattr(self, "_applied_msprobe_config_path", None)
        path_changed = (desired or "") != (applied or "") or (cfg_path is not None and cfg_path != current)
        if not reload and not path_changed:
            return False
        reason = "reload_msprobe" if reload else "msprobe_config_path"
        ok = self.recreate_msprobe_debugger(reason=reason)
        if reload:
            self.dfx_config.consume_reload_msprobe()
        return ok

    def _close_idle_msprobe_dump_gate(self) -> None:
        """Force msprobe ``dump_enable`` / device switch off while no dump window is open.

        DFX dump *capability* (``auto_max_times`` / ``manual_dump``) can be on
        while idle. Msprobe often defaults ``dump_enable=true`` when the key is
        omitted, so ``AclGraphDumper`` leaves device ``switch=1`` and every
        forward writes staging tensors. Close the gate after debugger init /
        recreate / lazy init; dump windows reopen it via
        ``set_msprobe_dump_state(True)``. Writing JSON (not only syncing
        in-memory switch) keeps a later ``step()`` / ``_refresh_dump_enable``
        from flipping the gate back on.
        """
        if self._debugger is None:
            return
        if self._msprobe_dump_active or self._pending_dump:
            return
        ok = False
        with suppress(Exception):
            ok = bool(self.set_msprobe_dump_state(False))
        if ok:
            logger.info(
                "[Anomaly msprobe] idle dump gate closed (await detector/manual arm) %s",
                self.dump_rank_tag(),
            )

    def _enforce_dump_requires_debugger(self) -> None:
        """If dump is enabled but debugger is unavailable, force dump off.

        On hot-reload with ``dump.enabled=true`` and ``_debugger is None``,
        retries ``_init_debugger`` first (covers install-msprobe-then-enable).
        """
        if not self.dfx_config.dump_enabled():
            return
        was_missing = self._debugger is None
        if was_missing:
            self._try_init_debugger()
        if self._debugger is not None:
            if was_missing and bool(getattr(self, "_startup_debugger_done", False)) and self._is_aclgraph_dumper():
                logger.warning(
                    "[Anomaly msprobe] debugger lazy-initialized after startup under "
                    "ACLGraph; hooks may miss already-captured graphs — restart the "
                    "worker if dump output is empty %s",
                    self.dump_rank_tag(),
                )
            # Capability on ≠ collection on: keep switch/JSON closed until armed.
            self._close_idle_msprobe_dump_gate()
            return
        dump_cfg = getattr(getattr(self.runner, "ascend_config", None), "dump_config_path", None)
        if dump_cfg is None:
            reason = "dump_config_path/dump_config not set (cannot init msprobe debugger)"
        else:
            reason = "msprobe debugger unavailable (install msprobe, then enable dump again)"
        self.dfx_config.disable_dump_unavailable(reason=reason)

    @property
    def dump_phase(self) -> DumpPhase:
        """Coarse dump FSM view: idle → pending → active."""
        if self._msprobe_dump_active:
            return DumpPhase.ACTIVE
        if self._pending_dump:
            return DumpPhase.PENDING
        return DumpPhase.IDLE

    def sync_dump_limits_from_config(self) -> None:
        """Refresh cached dump.max_times / cooldown from live ``dfx_config``."""
        self._dump_cooldown_seconds = self.dfx_config.dump_cooldown_seconds()
        self._dump_max_times = self.dfx_config.dump_max_times()

    def dump_rank_tag(self) -> str:
        """Rank tag for reports / anomaly logs."""
        runner = getattr(self, "runner", None)
        tp = getattr(runner, "tp_rank", "?") if runner is not None else "?"
        dp = getattr(runner, "dp_rank", "?") if runner is not None else "?"
        try:
            pp = get_pp_group().rank_in_group
        except Exception:
            pp = "?"
        return f"dp={dp} tp={tp} pp={pp}"

    def dump_count_snapshot(self, *, dump_armed: bool = False) -> tuple[int, int]:
        """``(dump_count, dump_max_times)`` for report JSON.

        ``dump_count`` is activated dumps so far. When this event just armed
        ``pending_dump`` with quota consumption, activate has not bumped the
        counter yet — report the next activation count (same as msprobe logs).
        Manual trigger (``skip_quota``) does not reserve a slot.
        """
        count = int(getattr(self, "_msprobe_dump_total_count", 0) or 0)
        max_times = int(getattr(self, "_dump_max_times", 0) or 0)
        if (
            dump_armed
            and bool(getattr(self, "_pending_dump", False))
            and not bool(getattr(self, "_pending_dump_skip_quota", False))
        ):
            count += 1
        return count, max_times

    def advance_wave(self, *, allow_arm: bool) -> int:
        """Bump real-step wave index. Dummy waves (``allow_arm=False``) skip.

        Called at the start of each ``sync_for_step``. Returns the current
        wave index after the bump (unchanged on dummy).
        """
        if allow_arm:
            self._wave_index = int(getattr(self, "_wave_index", 0) or 0) + 1
        return int(getattr(self, "_wave_index", 0) or 0)

    def current_wave(self) -> int:
        """Monotonic real-step wave index (0 before the first allow_arm sync)."""
        return int(getattr(self, "_wave_index", 0) or 0)

    def record_sample_waves(self, req_ids: Iterable[str] | None) -> int:
        """Main-thread: append ``current_wave`` for each req after sample.

        Under async scheduling, ``check_after_sample`` runs on the output-copy
        thread while the next ``execute_model`` may already have advanced the
        global wave. Stamping here (before returning AsyncOutput) keeps arm
        wave tied to the sample step that produced the tokens.

        Async: only the consuming rank (last-PP TP0 / ``get_output``) stamps.
        Other async ranks never ``take_sample_wave``, so stamping there only
        grows the FIFO until deferred-wave force-reap (noisy warning logs).
        Sync keeps stamping on every sample rank (they also call
        ``check_after_sample`` and drain the FIFO).
        """
        wave = self.current_wave()
        runner = getattr(self, "runner", None)
        if runner is not None and bool(getattr(runner, "use_async_scheduling", False)):
            if int(getattr(runner, "tp_rank", 0)) != 0:
                return wave
            try:
                if not get_pp_group().is_last_rank:
                    return wave
            except Exception:
                # Unit tests / early init without PP: still stamp on TP0.
                pass
        RequestDfxStore.get().record_sample_waves(req_ids, wave)
        return wave

    def take_sample_wave(self, req_id: str) -> int | None:
        """Pop the oldest main-thread sample wave for ``req_id`` (FIFO)."""
        return RequestDfxStore.get().take_sample_wave(req_id)

    def clear_sample_waves(self, req_id: str) -> None:
        """Drop any queued sample waves for a finished request."""
        RequestDfxStore.get().clear_sample_waves(req_id)

    def dump_arm_wave_for_report(self) -> int | None:
        """Arm-wave stamp for the in-flight open dump, else ``None``."""
        return getattr(self, "_open_dump_arm_wave", None)

    def dump_arm_wave_for_req(self, req_id: str) -> int | None:
        """Arm-wave for ``req_id`` from committed dump_finish meta, if any."""
        return RequestDfxStore.get().dump_arm_wave_for_req(req_id)

    def needs_io_for_dump_finish(self) -> bool:
        """True when TP0 should keep appending cumulative IO for dump_finish.

        Covers pending / active dump windows and any req still waiting for a
        dump_finish sidecar (open arm list or committed meta).
        """
        if bool(getattr(self, "_pending_dump", False)) or bool(getattr(self, "_msprobe_dump_active", False)):
            return True
        if getattr(self, "_open_dump_finish_req_ids", None):
            return True
        return RequestDfxStore.get().has_dump_finish_meta()

    def take_dump_finish_meta(self, req_id: str) -> DumpFinishMeta | None:
        """Pop finish meta for ``req_id`` (one-shot; used by reap / dump_finish write).

        Prefers committed meta (after successful activate). If the req is still
        only on the open arm list (pending, not yet activated), pop it from
        open and return a meta with ``dump_activate_wave=None`` /
        ``dump_waves_after_report=None`` so a dump_finish sidecar can still be
        written without leaving an orphan entry for a later activate.
        """
        if not req_id:
            return None
        committed = RequestDfxStore.get().take_dump_finish(req_id)
        if committed is not None:
            return committed
        open_ids = list(getattr(self, "_open_dump_finish_req_ids", ()) or ())
        if req_id not in open_ids:
            return None
        remaining = [r for r in open_ids if r != req_id]
        self._open_dump_finish_req_ids = remaining
        arm = getattr(self, "_open_dump_arm_wave", None)
        meta = DumpFinishMeta(
            anomaly_type=getattr(self, "_open_dump_anomaly_type", None),
            source=getattr(self, "_open_dump_source", None),
            dump_arm_wave=int(arm) if arm is not None else None,
            dump_activate_wave=None,
            dump_waves_after_report=None,
            dump_count=None,
        )
        if not remaining:
            self._clear_open_dump_wave_tracking()
        return meta

    def _begin_dump_wave_tracking(
        self,
        finish_req_ids: list[str] | None,
        *,
        anomaly_type: str | None,
        source: str,
        arm_wave: int | None = None,
    ) -> None:
        """Stamp arm wave + reqs that should get a dump_finish sidecar later.

        ``arm_wave`` should be the main-thread sample stamp when arming from
        async ``check_after_sample``; omit to use ``current_wave()`` (sync /
        manual paths on the worker main thread).
        """
        reqs = [str(r) for r in (finish_req_ids or ()) if r]
        if arm_wave is None:
            self._open_dump_arm_wave = self.current_wave()
        else:
            self._open_dump_arm_wave = int(arm_wave)
        self._open_dump_finish_req_ids = reqs
        self._open_dump_anomaly_type = anomaly_type
        self._open_dump_source = source

    def _clear_open_dump_wave_tracking(self) -> None:
        self._open_dump_arm_wave = None
        self._open_dump_finish_req_ids = []
        self._open_dump_anomaly_type = None
        self._open_dump_source = None

    def _commit_dump_finish_metas(self, *, consume_quota: bool) -> None:
        """After successful activate: persist per-req wave meta until finish."""
        reqs = list(getattr(self, "_open_dump_finish_req_ids", ()) or ())
        if not reqs:
            self._clear_open_dump_wave_tracking()
            return
        arm = getattr(self, "_open_dump_arm_wave", None)
        activate = self.current_wave()
        waves_after: int | None
        if arm is None:
            waves_after = None
        else:
            waves_after = int(activate) - int(arm)
        dump_count: int | None
        if consume_quota:
            dump_count = int(getattr(self, "_msprobe_dump_total_count", 0) or 0)
        else:
            dump_count = None
        store = RequestDfxStore.get()
        for req_id in reqs:
            store.set_dump_finish(
                req_id,
                DumpFinishMeta(
                    anomaly_type=getattr(self, "_open_dump_anomaly_type", None),
                    source=getattr(self, "_open_dump_source", None),
                    dump_arm_wave=int(arm) if arm is not None else None,
                    dump_activate_wave=int(activate),
                    dump_waves_after_report=waves_after,
                    dump_count=dump_count,
                ),
            )
        self._clear_open_dump_wave_tracking()

    def _apply_observability_switches(self) -> None:
        """Apply ``ascend_log`` from live config."""
        self.dfx_config.apply_ascend_log_level()

    def apply_dfx_config(self) -> None:
        """Pull dump limits / ``ascend_log`` from already-synced ``dfx_config``.

        Runner owns :meth:`~DfxRuntimeConfig.sync_dfx_config`; call this only
        after a successful reload so Dumper never drives config I/O.

        Also lazy-retries msprobe debugger init when ``dump.enabled`` was
        flipped on after install, and forces it back off if still unavailable.
        """
        prev_max = self._dump_max_times
        prev_cd = self._dump_cooldown_seconds
        self.sync_dump_limits_from_config()
        self._apply_observability_switches()
        # Hot-reload only: limits / log level already announced via runtime_config updated.
        if prev_max != self._dump_max_times or prev_cd != self._dump_cooldown_seconds:
            logger.info(
                "[DFX dumper] dump limits applied max_times=%d→%d cooldown=%d→%d %s",
                prev_max,
                self._dump_max_times,
                prev_cd,
                self._dump_cooldown_seconds,
                self.dump_rank_tag(),
            )
        else:
            logger.debug(
                "[DFX dumper] config applied (limits unchanged) %s",
                self.dump_rank_tag(),
            )
        self.maybe_recreate_msprobe_debugger()
        self._enforce_dump_requires_debugger()

    def handle_anomaly_alert(
        self,
        alert: AnomalyAlert,
        *,
        detector: AnomalyDetector | None = None,
        arm_wave: int | None = None,
    ) -> bool:
        """Arm / activate dump from a detector alert (report is runner-owned).

        ``arm_wave``: preferred sample-step stamp from
        :meth:`take_sample_wave` (async-safe). When omitted, uses
        ``current_wave()`` at arm time.
        """
        if alert is None or not alert.is_ill or not alert.req_id:
            return False
        ok = self.enable_msprobe_dump_if_needed(
            alert.req_id,
            req_idx=alert.req_idx,
            skip_related_check=alert.skip_related_check,
            consume_quota=alert.consume_quota,
            finish_req_ids=[alert.req_id],
            anomaly_type=alert.anomaly_type,
            source="anomaly",
            arm_wave=arm_wave,
        )
        if not ok:
            return False
        if detector is not None:
            detector.on_alert_armed(alert)
        return True

    def handle_manual_trigger(
        self,
        trigger: TriggerEvent,
        *,
        finish_req_ids: list[str] | None = None,
    ) -> bool:
        """Arm / activate dump from a control-plane manual trigger event.

        Consumes ``dump.manual_dump`` **after** a newly successful arm so
        ``manual_dump: 1`` still looks active during ``dump_enabled()`` /
        pending-OR (consume-before-arm skipped the dump window).
        """
        if trigger is None or not trigger.req_id:
            return False
        was_busy = bool(self._pending_dump or self._msprobe_dump_active)
        # Manual uses a synthetic req_id; finish sidecars attach to batch reqs.
        reqs = [str(r) for r in (finish_req_ids or ()) if r]
        ok = self.enable_msprobe_dump_if_needed(
            trigger.req_id,
            req_idx=None,
            skip_related_check=True,
            consume_quota=trigger.consume_quota,
            finish_req_ids=reqs,
            anomaly_type=trigger.trigger_type,
            source="manual_trigger",
        )
        if ok and not was_busy:
            # New arm (pending or immediate activate). Continuous ``true`` is
            # a no-op; int N decrements. Failed arm leaves the count for retry.
            self.dfx_config.consume_manual_trigger()
        return ok

    def anomaly_check_skip_reason(self, *, ignore_dump_busy: bool = False) -> str | None:
        """None if detectors may run; otherwise a short skip reason for logs.

        Detect is independent of ``dump.enabled``. While dump is armed /
        active, skip further detect by default to avoid overlapping arms.
        Pass ``ignore_dump_busy=True`` for checks that should still run in the
        same step after another detector already armed dump (e.g. block_kv).
        """
        if not self.dfx_config.any_detector_enabled():
            return "no detector enabled in live DFX config"
        rank_reason = anomaly_check_rank_skip_reason(getattr(self, "runner", None))
        if rank_reason is not None:
            return rank_reason
        if not ignore_dump_busy and self.dfx_config.dump_enabled():
            if self._pending_dump:
                return "pending_dump already armed"
            if self._msprobe_dump_active:
                return "msprobe dump already active"
        return None

    def can_run_anomaly_detection(self, *, ignore_dump_busy: bool = False) -> bool:
        """Whether this rank should invoke detectors this step."""
        return self.anomaly_check_skip_reason(ignore_dump_busy=ignore_dump_busy) is None

    def _dump_state_tag(self) -> str:
        return (
            f"phase={getattr(self, 'dump_phase', '?')} "
            f"active={getattr(self, '_msprobe_dump_active', False)} "
            f"needs_fwd={getattr(self, '_dump_needs_forward', False)} "
            f"fwd_seen={getattr(self, '_dump_forward_seen', False)} "
            f"dbg_started={getattr(self, '_debugger_started', False)} "
            f"pending={getattr(self, '_pending_dump', False)}"
        )

    def is_related_local_request(self, req_id: str, req_idx: int | None = None) -> bool:
        input_batch = getattr(self.runner, "input_batch", None)
        req_ids = getattr(input_batch, "req_ids", None) if input_batch is not None else None

        # v2 (and batch-local) path: req_idx is the position in input_batch.req_ids.
        if req_ids is not None and req_idx is not None:
            if req_idx < 0 or req_idx >= len(req_ids) or req_ids[req_idx] != req_id:
                return False
            requests = getattr(self.runner, "requests", None)
            if requests is not None and req_id not in requests:
                return False
            req_states = getattr(self.runner, "req_states", None)
            req_id_to_index = getattr(req_states, "req_id_to_index", None)
            if req_id_to_index is not None and req_id not in req_id_to_index:
                return False
            discard_request_mask = getattr(self.runner, "discard_request_mask", None)
            if discard_request_mask is not None and hasattr(discard_request_mask, "np"):
                if req_idx < len(discard_request_mask.np) and discard_request_mask.np[req_idx]:
                    return False
            return True

        req_id_to_index = getattr(input_batch, "req_id_to_index", None)
        if req_id_to_index is None:
            req_states = getattr(self.runner, "req_states", None)
            req_id_to_index = getattr(req_states, "req_id_to_index", None)
        if req_id_to_index is None:
            return False

        mapped_idx = req_id_to_index.get(req_id)
        if mapped_idx is None:
            return False

        if req_idx is not None and mapped_idx != req_idx:
            if self.runner.tp_rank == 0:
                logger.warning(
                    "[Anomaly msprobe] req_id=%s skip dump: req_idx mismatch input=%d mapped=%d",
                    req_id,
                    req_idx,
                    mapped_idx,
                )
            return False

        num_reqs = getattr(input_batch, "num_reqs", None)
        if num_reqs is None:
            req_states = getattr(self.runner, "req_states", None)
            num_reqs_np = getattr(req_states, "num_reqs_np", None)
            if num_reqs_np is not None:
                num_reqs = int(num_reqs_np[0])
        if num_reqs is None:
            return False

        if mapped_idx < 0 or mapped_idx >= num_reqs:
            return False

        if req_ids is not None and mapped_idx < len(req_ids) and req_ids[mapped_idx] != req_id:
            return False

        discard_request_mask = getattr(self.runner, "discard_request_mask", None)
        if discard_request_mask is not None and hasattr(discard_request_mask, "np"):
            if mapped_idx < len(discard_request_mask.np) and discard_request_mask.np[mapped_idx]:
                return False

        requests = getattr(self.runner, "requests", None)
        if requests is not None:
            return req_id in requests
        return True
