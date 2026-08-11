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

from typing import TYPE_CHECKING, Any

from vllm.distributed.parallel_state import get_pp_group

from vllm_ascend.dfx.dfx_types import DumpPhase
from vllm_ascend.dfx.dumper.msprobe import MsprobeBridgeMixin
from vllm_ascend.dfx.dumper.pending import PendingDumpMixin, anomaly_check_rank_skip_reason
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

        # True after construction-time debugger probe finishes. Late success
        # under ACLGraph may miss already-captured graphs.
        self._startup_debugger_done = False

        self._apply_observability_switches()

        logger.info_once(
            "DFX ready config=%s report_dir=%s dump.enabled=%s dump.max_times=%d "
            "ascend_log.level=%s ascend_log.debug=%s "
            "spec_check=%s token_logprob_check=%s output_substring_check=%s "
            "token_repeat_check=%s "
            "report.save_sensitive_info=%s",
            str(self.dfx_config.config_path),
            str(self.dfx_config.report_dir),
            self.dfx_config.dump_enabled(),
            self._dump_max_times,
            self.dfx_config.ascend_log_level(),
            # info_once is lru_cached; args must be hashable (not list).
            tuple(self.dfx_config.ascend_log_debug_modules()),
            bool(self.dfx_config.detector_get("spec_acceptance", "enabled", False)),
            bool(self.dfx_config.detector_get("token_logprob", "enabled", False)),
            bool(self.dfx_config.detector_get("output_substring", "enabled", False)),
            bool(self.dfx_config.detector_get("token_repeat", "enabled", False)),
            self.dfx_config.report_save_sensitive_info(),
        )

        # Keep debugger lifecycle fully encapsulated in Dumper. Soft-fail when
        # msprobe / dump_config is missing; force dump.enabled=false if still on.
        self._try_init_debugger()
        self._enforce_dump_requires_debugger()
        self._startup_debugger_done = True

    def _try_init_debugger(self) -> None:
        """Init debugger once when absent (startup or lazy reload retry)."""
        if self._debugger is not None:
            return
        self._init_debugger(self.runner.compilation_config.cudagraph_mode)

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
            return
        dump_cfg = getattr(getattr(self.runner, "ascend_config", None), "dump_config_path", None)
        if dump_cfg is None:
            reason = "dump_config_path/dump_config not set (cannot init msprobe debugger)"
        else:
            reason = "msprobe debugger unavailable (install msprobe, then set dump.enabled=true again)"
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
        self._enforce_dump_requires_debugger()

    def handle_anomaly_alert(
        self,
        alert: AnomalyAlert,
        *,
        detector: AnomalyDetector | None = None,
    ) -> bool:
        """Arm / activate dump from a detector alert (report is runner-owned)."""
        if alert is None or not alert.is_ill or not alert.req_id:
            return False
        ok = self.enable_msprobe_dump_if_needed(
            alert.req_id,
            req_idx=alert.req_idx,
            skip_related_check=alert.skip_related_check,
            consume_quota=alert.consume_quota,
        )
        if not ok:
            return False
        if detector is not None:
            detector.on_alert_armed(alert)
        return True

    def handle_manual_trigger(self, trigger: TriggerEvent) -> bool:
        """Arm / activate dump from a control-plane manual trigger event."""
        if trigger is None or not trigger.req_id:
            return False
        return self.enable_msprobe_dump_if_needed(
            trigger.req_id,
            req_idx=None,
            skip_related_check=True,
            consume_quota=trigger.consume_quota,
        )

    def anomaly_check_skip_reason(self) -> str | None:
        """None if detectors may run; otherwise a short skip reason for logs.

        Detect is independent of ``dump.enabled``. While dump is armed /
        active, skip further detect to avoid overlapping arms.
        """
        if not self.dfx_config.any_detector_enabled():
            return "no detector enabled in live DFX config"
        rank_reason = anomaly_check_rank_skip_reason(getattr(self, "runner", None))
        if rank_reason is not None:
            return rank_reason
        if self.dfx_config.dump_enabled():
            if self._pending_dump:
                return "pending_dump already armed"
            if self._msprobe_dump_active:
                return "msprobe dump already active"
        return None

    def can_run_anomaly_detection(self) -> bool:
        """Whether this rank should invoke detectors this step."""
        return self.anomaly_check_skip_reason() is None

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
