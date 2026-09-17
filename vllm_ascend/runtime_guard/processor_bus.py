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

from typing import Any

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.runtime_config._dist import (
    SYNC_BROADCAST,
    _runtime_config_sync_group_or_none,
)
from vllm_ascend.runtime_guard.rank_gate import should_dump_kv_on_rank

logger = init_logger_ascend(__name__)


class RuntimeGuardBusMixin:
    """Wave-head config sync + dump-job delivery (broadcast / file / TP)."""

    def _refresh_config_body(
        self,
        *,
        allow_arm: bool,
        scheduler_output: Any | None,
    ) -> bool:
        # Wave-head: 1×AR([config_due, dump_due]) + per-lane bcasts.
        # Config apply here so this wave's detectors see new JSON.
        # Auto dump jobs from *previous* wave are bcast into deferred D2H;
        # same-wave detector arms stay queued until the next head (+1 wave).
        del allow_arm, scheduler_output
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
        use_broadcast = cfg.hot_reload_enabled and cfg.sync_mode == SYNC_BROADCAST
        group = _runtime_config_sync_group_or_none() if use_broadcast else None
        if group is not None and int(getattr(group, "world_size", 1) or 1) > 1:
            return self._wave_head_merged_bus(group)
        # File / PP>1 / no group: local config; auto dump drain (prev-wave jobs).
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
            "[runtime_guard sync] leave stage=wave_head_task_bus changed=%s file_or_solo",
            changed,
        )
        return changed

    def _wave_head_merged_bus(self, sync_group: Any) -> bool:
        """1 AR([config_due, dump_due]) + separate bcasts; stash dump for end D2H."""
        from vllm_ascend.runtime_config._task_bus import (
            broadcast_when_due,
            sync_due_bits,
        )

        cfg = self.runtime_config
        config_due_local = bool(cfg.hot_reload_enabled and cfg.config_due_local())

        dump_jobs: list[dict[str, Any]] = []
        dump_due_local = False
        can_dump = should_dump_kv_on_rank(self.runner)
        if can_dump:
            dump_jobs = list(getattr(self, "_kv_dump_jobs", None) or [])
            if hasattr(self, "_kv_dump_jobs"):
                self._kv_dump_jobs.clear()
            try:
                is_src = int(sync_group.rank_in_group) == 0
            except Exception:
                is_src = bool(getattr(sync_group, "is_first_rank", False))
            dump_due_local = bool(is_src and dump_jobs)
        elif hasattr(self, "_kv_dump_jobs"):
            self._kv_dump_jobs.clear()

        config_due, dump_due = sync_due_bits(
            sync_group, [config_due_local, dump_due_local]
        )

        changed = False
        leader_changed = [False]

        def _build_config() -> dict[str, Any]:
            payload, ch = cfg.build_config_sync_payload()
            leader_changed[0] = ch
            return payload

        if cfg.hot_reload_enabled:
            config_payload = broadcast_when_due(
                sync_group,
                due=config_due,
                build_payload=_build_config if sync_group.is_first_rank else None,
                src=0,
            )
            if config_due and isinstance(config_payload, dict):
                try:
                    changed = cfg.apply_config_sync_payload(
                        config_payload,
                        is_leader=bool(sync_group.is_first_rank),
                        leader_changed=leader_changed[0],
                    )
                    if changed:
                        self._apply_config_cascade()
                except Exception as exc:
                    logger.warning(
                        "[runtime_guard sync] config apply soft-failed error=%s",
                        exc,
                    )
                    changed = False

        try:
            src_rank = int(sync_group.rank_in_group)
        except Exception:
            src_rank = 0 if bool(getattr(sync_group, "is_first_rank", False)) else 1

        jobs = broadcast_when_due(
            sync_group,
            due=dump_due,
            payload=dump_jobs if src_rank == 0 else None,
            src=0,
        )
        if dump_due and can_dump and jobs:
            self._deferred_kv_dump_jobs.extend(list(jobs))

        logger.debug(
            "[runtime_guard sync] leave stage=wave_head_merged_bus "
            "config_due=%s dump_due=%s changed=%s",
            config_due,
            dump_due,
            changed,
        )
        return changed
