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

"""Resolve and run configured incident actions."""

from __future__ import annotations

from typing import Any

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.runtime_config.config import RuntimeConfig
from vllm_ascend.runtime_guard.action.actions import Action, ActionContext, get_action
from vllm_ascend.runtime_guard.action.queue import ActionQueue
from vllm_ascend.runtime_guard.incident import Incident
from vllm_ascend.runtime_guard.kv_cache_reader import KvCacheReader
from vllm_ascend.runtime_guard.manual_trigger import MANUAL_TRIGGER_TYPE
from vllm_ascend.runtime_guard.quota import DumpQuota
from vllm_ascend.runtime_guard.rank_gate import dump_rank_tag, should_run_anomaly_check_on_rank
from vllm_ascend.runtime_guard.report import ReportWriter

logger = init_logger_ascend(__name__)

_DEFAULT_ACTIONS = ["report"]


class ActionExecutor:
    def __init__(
        self,
        runner: Any,
        *,
        runtime_config: RuntimeConfig,
        report_writer: ReportWriter,
        quota: DumpQuota,
        action_queue: ActionQueue | None = None,
    ) -> None:
        self._runner = runner
        self._runtime_config = runtime_config
        self._report_writer = report_writer
        self._quota = quota
        self._kv_reader = KvCacheReader(runner)
        self._queue = (
            action_queue
            if action_queue is not None
            else ActionQueue(maxsize=runtime_config.action_queue_max_size())
        )

    @property
    def action_queue(self) -> ActionQueue:
        return self._queue

    def rebind_runner(self, runner: Any, *, runtime_config: RuntimeConfig | None = None) -> None:
        self._runner = runner
        if runtime_config is not None:
            self._runtime_config = runtime_config
        self._kv_reader = KvCacheReader(runner)

    def start(self) -> None:
        self._queue.start()

    def stop(self) -> None:
        self._queue.stop()

    def can_run_detection(self) -> bool:
        return should_run_anomaly_check_on_rank(self._runner)

    def anomaly_check_skip_reason(self) -> str | None:
        from vllm_ascend.runtime_guard.rank_gate import anomaly_check_rank_skip_reason

        return anomaly_check_rank_skip_reason(self._runner)

    def submit_heavy(self, job: Any) -> None:
        """Queue a torch.save-scale job; drop (never inline) when the queue is full."""
        if not self._queue.submit(job, heavy=True):
            logger.warning(
                "[runtime_guard action] skip heavy enqueue (queue full or stopping); KV .pt save may be missing"
            )

    def resolve_actions(
        self,
        incident_type: str,
        *,
        override: list[str] | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        if override is not None:
            names = list(override)
            overrides = self._runtime_config.detector_section(incident_type) or {}
            return names, dict(overrides) if isinstance(overrides, dict) else {}

        defaults = self._runtime_config.actions_default_on_trigger()
        det = self._runtime_config.detector_section(incident_type) or {}
        raw = det.get("on_trigger")
        if raw is None:
            names = list(defaults or _DEFAULT_ACTIONS)
        elif isinstance(raw, str):
            names = [raw]
        elif isinstance(raw, list):
            names = [str(x) for x in raw]
        else:
            names = list(defaults or _DEFAULT_ACTIONS)
        return names, dict(det)

    def handle(
        self,
        incident: Incident,
        *,
        detail: dict[str, Any],
        tokenizer: Any | None = None,
        action_override: list[str] | None = None,
        write_report: bool = True,
        inject_manual_dump_kv: bool = True,
    ) -> None:
        if not self.can_run_detection():
            return
        names, det_cfg = self.resolve_actions(incident.incident_type, override=action_override)
        if not write_report:
            names = [n for n in names if n != "report"]
        # Manual dump: default inject dump_kv (queue+bcast path). End-of-wave
        # local fire passes inject_manual_dump_kv=False — each rank D2H itself.
        if inject_manual_dump_kv and incident.incident_type == MANUAL_TRIGGER_TYPE and "dump_kv" not in names:
            names.append("dump_kv")
        overrides = dict(det_cfg)
        if incident.incident_type == MANUAL_TRIGGER_TYPE and inject_manual_dump_kv:
            dump_cfg = overrides.get("dump_kv")
            if isinstance(dump_cfg, dict):
                overrides["dump_kv"] = {**dump_cfg, "scope": "all_requests"}
            else:
                overrides["dump_kv"] = {"scope": "all_requests"}
        overrides["_actions"] = names
        ctx = ActionContext(
            incident=incident,
            runner=self._runner,
            runtime_config=self._runtime_config,
            report_writer=self._report_writer,
            kv_reader=self._kv_reader,
            quota=self._quota,
            rank_tag=dump_rank_tag(self._runner),
            tokenizer=tokenizer,
            detail=detail,
            action_overrides=overrides,
        )

        # sync_only → prepare → queue commit; prefer report before dump_kv.
        ordered: list[str] = []
        for name in names:
            if name not in ordered:
                ordered.append(name)
        if "report" in ordered and "dump_kv" in ordered:
            ordered = [n for n in ordered if n not in ("report", "dump_kv")]

            def _is_sync_only(n: str) -> bool:
                act = get_action(n)
                return bool(act is not None and act.sync_only)

            head = [n for n in ordered if _is_sync_only(n)]
            tail = [n for n in ordered if not _is_sync_only(n)]
            ordered = head + ["report", "dump_kv"] + [n for n in tail if n not in head]

        for name in ordered:
            action = get_action(name)
            if action is None:
                logger.warning("[runtime_guard action] unknown action=%s incident=%s", name, incident.incident_type)
                continue
            try:
                if action.sync_only:
                    action.run(ctx)
                    continue
                prepared = action.prepare(ctx)
                if prepared is None:
                    continue

                act_bound: Action = action
                prep_bound: Any = prepared

                def _commit(
                    act: Action = act_bound,
                    prep: Any = prep_bound,
                ) -> None:
                    try:
                        act.commit(prep)
                    except Exception:
                        logger.exception(
                            "[runtime_guard action] %s commit failed incident=%s req_id=%s",
                            act.name,
                            incident.incident_type,
                            incident.req_id,
                        )

                dedupe_key = None
                if action.name == "report":
                    # One pending report commit per (wave, req_id).
                    dedupe_key = (
                        "report",
                        incident.wave,
                        str(incident.req_id or ""),
                    )
                ok = self._queue.submit(_commit, heavy=action.heavy, dedupe_key=dedupe_key)
                if not ok and dedupe_key is not None:
                    # Light report: queue-full runs inline (submit still True).
                    # False means same-key dedupe or queue stopping.
                    logger.info(
                        "[runtime_guard action] skip report enqueue (dedupe or stopping) type=%s req_id=%s wave=%s",
                        incident.incident_type,
                        incident.req_id,
                        incident.wave,
                    )
            except Exception as exc:
                logger.exception(
                    "[runtime_guard action] %s prepare failed incident=%s req_id=%s: %s",
                    name,
                    incident.incident_type,
                    incident.req_id,
                    exc,
                )

    def apply_runtime_config(self) -> None:
        self._quota.sync_from_config()
