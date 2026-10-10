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

"""Wave-head config+dump bus mixin for RuntimeGuardProcessor."""

from __future__ import annotations

import time
from typing import Any

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.observability.runtime_config.dist import _runtime_config_sync_group_or_none
from vllm_ascend.observability.runtime_guard.bus_worker import MergedBusRequest, MergedBusResult
from vllm_ascend.observability.runtime_guard.rank_gate import should_dump_kv_on_rank

logger = init_logger_ascend(__name__)

_BUS_WARN_INTERVAL_S = 30.0


class RuntimeGuardBusMixin:
    """Wave-head config sync + dump-job delivery (broadcast / file / TP).

    Broadcast path (last-PP × TP only): submit merged due-broadcast(+due bcasts) to
    :class:`DueBitsBusWorker` at wave head; apply results in
    :meth:`_drain_merged_bus` at end-of-wave so the collective overlaps
    forward. Non-last PP and ``tp_size<=1`` poll JSON locally. Missing worker
    is lazily started (no sync fallback).
    """

    # Attributes provided by RuntimeGuardProcessor (mixin composition).
    runner: Any
    runtime_config: Any
    detectors: Any
    report_writer: Any
    action_executor: Any
    _deferred_kv_dump_jobs: list[dict[str, Any]]
    _bus_worker: Any
    _merged_bus_inflight: bool
    _pending_merged_bus_dump_jobs: list[dict[str, Any]]
    _pending_merged_bus_can_dump: bool

    # Defined on RuntimeGuardDumpMixin / RuntimeGuardProcessor.
    _claim_dump_jobs_to_deferred_via_tp: Any
    _drop_pending_dump_jobs: Any
    _refund_dropped_dump_arms: Any

    def _refresh_config_body(self) -> bool:
        # Wave-head: 1×due-broadcast([config_due, dump_due]) + per-lane bcasts.
        # Async path defers apply/stash to end-of-wave (overlap with forward).
        # Auto dump jobs from *previous* wave are bcast into deferred D2H;
        # same-wave detector arms stay queued until the next head (+1 wave).
        return self._wave_head_task_bus()

    def _apply_config_cascade(self) -> None:
        """Re-bind dependents after a successful wave-head config apply."""
        self.action_executor.apply_runtime_config()
        self.runtime_config.apply_ascend_log_level()
        self.detectors.apply_runtime_config()
        self.report_writer.save_sensitive_info = self.runtime_config.report_save_sensitive_info()
        self.report_writer.max_prompt_token_ids = self.runtime_config.report_max_prompt_token_ids()
        self.report_writer.max_output_token_ids = self.runtime_config.report_max_output_token_ids()
        self.report_writer.decode_token_ids = self.runtime_config.report_decode_token_ids()
        self.report_writer.max_per_req = self.runtime_config.report_max_per_req()

    def _wave_head_task_bus(self) -> bool:
        """Wave-head config+dump gate. Returns whether config content changed."""
        cfg = self.runtime_config
        # Hot-reload on last-PP TP → merged due bus; else local file poll + TP dump claim.
        group = _runtime_config_sync_group_or_none() if cfg.hot_reload_enabled else None
        if group is not None and int(getattr(group, "world_size", 1) or 1) > 1:
            return self._wave_head_merged_bus(group)
        # Non-last-PP / tp_size<=1 / reload off: local config; auto dump via TP claim.
        changed = False
        if cfg.hot_reload_enabled:
            try:
                changed = cfg.sync_runtime_config()
            except Exception as exc:
                logger.warning(
                    "[runtime_guard sync] wave-head local config soft-failed error=%s",
                    exc,
                )
                changed = False
            if changed:
                self._apply_config_cascade()
        # Previous-wave auto jobs: claim via TP bus into deferred; D2H at
        # end-of-wave (same path as broadcast merged bus).
        self._claim_dump_jobs_to_deferred_via_tp()
        logger.debug(
            "[runtime_guard sync] leave stage=wave_head_task_bus changed=%s file_or_non_last_pp",
            changed,
        )
        return changed

    def _prepare_merged_bus_locals(self, sync_group: Any) -> tuple[bool, bool, list[dict[str, Any]], bool, bool]:
        """Local due bits + dump-job handoff (clears ``_kv_dump_jobs``)."""
        cfg = self.runtime_config
        config_due_local = bool(cfg.hot_reload_enabled and cfg.config_due_local())

        dump_jobs: list[dict[str, Any]] = []
        dump_due_local = False
        can_dump = should_dump_kv_on_rank()
        if can_dump:
            dump_jobs = list(self._kv_dump_jobs)
            self._kv_dump_jobs.clear()
            try:
                is_src = int(sync_group.rank_in_group) == 0
            except Exception:
                is_src = bool(getattr(sync_group, "is_first_rank", False))
            dump_due_local = bool(is_src and dump_jobs)
        else:
            self._drop_pending_dump_jobs()

        try:
            is_first = bool(sync_group.is_first_rank)
        except Exception:
            is_first = False
        return config_due_local, dump_due_local, dump_jobs, can_dump, is_first

    def _wave_head_merged_bus(self, sync_group: Any) -> bool:
        """Submit 1 due-broadcast([config_due, dump_due]) + due bcasts on
        ``DueBitsBusWorker``.

        Always asynchronous: returns ``False`` immediately; apply/stash happens in
        :meth:`_drain_merged_bus` at end-of-wave (overlaps forward).
        """
        config_due_local, dump_due_local, dump_jobs, can_dump, is_first = (
            RuntimeGuardBusMixin._prepare_merged_bus_locals(self, sync_group)
        )
        worker = self._bus_worker

        # Previous wave must already be drained at end-of-wave; belt-and-suspenders.
        RuntimeGuardBusMixin._drain_merged_bus(self, warn_if_pending=True)
        cfg = self.runtime_config
        build = cfg.build_config_sync_payload if is_first else None
        self._pending_merged_bus_dump_jobs = list(dump_jobs)
        self._pending_merged_bus_can_dump = can_dump
        self._pending_merged_bus_is_first = is_first
        self._bus_wave_seq = int(self._bus_wave_seq) + 1
        wave_seq = self._bus_wave_seq
        try:
            worker.submit(
                MergedBusRequest(
                    sync_group=sync_group,
                    config_due_local=config_due_local,
                    dump_due_local=dump_due_local,
                    dump_jobs=dump_jobs,
                    hot_reload_enabled=bool(cfg.hot_reload_enabled),
                    is_first_rank=is_first,
                    build_config_payload=build,
                    wave_idx=wave_seq,
                )
            )
        except Exception:
            self._refund_dropped_dump_arms(dump_jobs)
            self._pending_merged_bus_dump_jobs = []
            self._pending_merged_bus_can_dump = False
            self._pending_merged_bus_is_first = False
            raise
        self._merged_bus_inflight = True
        logger.debug(
            "[runtime_guard sync] leave stage=wave_head_merged_bus async_submit config_due_local=%s dump_due_local=%s",
            config_due_local,
            dump_due_local,
        )
        return False

    def _drain_merged_bus(self, *, warn_if_pending: bool = False, timeout: float | None = None) -> bool:
        """Wait for async merged-bus result and apply on the inference thread.

        Returns whether config content changed. No-op when nothing inflight.
        ``timeout`` bounds the final ``wait`` for process teardown only —
        regular end-of-wave drains stay unbounded (lockstep collective).
        The not-finished warning is rate-limited to one line per 30s per rank
        (previously every late wave spammed one WARNING per rank).
        """
        if not self._merged_bus_inflight:
            return False
        worker = self._bus_worker

        if warn_if_pending and not worker.poll_ready():
            now = time.monotonic()
            if now - self._merged_bus_warn_ts >= _BUS_WARN_INTERVAL_S:
                self._merged_bus_warn_ts = now
                logger.warning(
                    "[runtime_guard sync] merged bus not finished before end-of-wave; "
                    "waiting on the due broadcast (forward did not fully hide the collective)"
                )
        try:
            result = worker.wait_result(timeout=timeout)
        except Exception:
            jobs = list(self._pending_merged_bus_dump_jobs)
            self._pending_merged_bus_dump_jobs = []
            self._pending_merged_bus_can_dump = False
            self._pending_merged_bus_is_first = False
            self._merged_bus_inflight = False
            self._refund_dropped_dump_arms(jobs)
            raise

        self._merged_bus_inflight = False
        pending_jobs = list(self._pending_merged_bus_dump_jobs)
        can_dump = bool(self._pending_merged_bus_can_dump)
        is_first = bool(self._pending_merged_bus_is_first)
        self._pending_merged_bus_dump_jobs = []
        self._pending_merged_bus_can_dump = False
        self._pending_merged_bus_is_first = False

        if result is None:
            return False
        return RuntimeGuardBusMixin._apply_merged_bus_result(
            self,
            result,
            pending_dump_jobs=pending_jobs,
            can_dump=can_dump,
            is_first_rank=is_first,
        )

    def _apply_merged_bus_result(
        self,
        result: MergedBusResult,
        *,
        pending_dump_jobs: list[dict[str, Any]],
        can_dump: bool,
        is_first_rank: bool,
    ) -> bool:
        if result.error is not None:
            self._refund_dropped_dump_arms(pending_dump_jobs)
            raise result.error

        cfg = self.runtime_config
        changed = False
        if result.config_due and isinstance(result.config_payload, dict):
            try:
                changed = cfg.apply_config_sync_payload(
                    result.config_payload,
                    is_leader=is_first_rank,
                    leader_changed=result.leader_changed,
                )
                if changed:
                    self._apply_config_cascade()
            except Exception as exc:
                logger.warning(
                    "[runtime_guard sync] config apply soft-failed error=%s",
                    exc,
                )
                changed = False

        if result.dump_due and can_dump and result.dump_jobs:
            self._deferred_kv_dump_jobs.extend(list(result.dump_jobs))

        logger.debug(
            "[runtime_guard sync] drained merged bus config_due=%s dump_due=%s changed=%s",
            result.config_due,
            result.dump_due,
            changed,
        )
        return changed
