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

"""Pending-dump OR sync and dump arming / activation."""

from __future__ import annotations

import time
from typing import Any

import torch
from vllm.distributed.parallel_state import get_pp_group, get_tp_group

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)


def use_pending_dump_sync(runner: Any) -> bool:
    """Whether dump_enable should wait for last-PP TP OR at execute entry.

    Required for async (only TP0 checks). Also used whenever TP>1 so sync
    path never activates debugger mid-sample on a subset of ranks.
    """
    if runner is None:
        return False
    if bool(getattr(runner, "use_async_scheduling", False)):
        return True
    try:
        return get_tp_group().world_size > 1
    except Exception:
        return False


def anomaly_check_rank_skip_reason(runner: Any) -> str | None:
    """None if this rank may run detectors; otherwise a short skip reason.

    Last PP only. When pending-OR is used (async, or sync with TP>1), only TP0.
    """
    if runner is None:
        return "no runner"
    try:
        if not get_pp_group().is_last_rank:
            return "not last PP rank"
    except Exception:
        return "PP group unavailable"
    if use_pending_dump_sync(runner) and int(getattr(runner, "tp_rank", 0)) != 0:
        return "pending-OR: only TP0 runs anomaly check"
    return None


def should_run_anomaly_check_on_rank(runner: Any) -> bool:
    """Whether this rank should evaluate detectors / emit manual-dump alerts."""
    return anomaly_check_rank_skip_reason(runner) is None


class PendingDumpMixin:
    """Mixin: last-PP TP pending-OR, quota, enable / activate."""

    def _anomaly_dump_feature_enabled(self) -> bool:
        """Whether auto dump arming / OR early-out is allowed (needs quota).

        Detect is independent. Manual ``dump.manual_trigger`` does not use this gate.
        """
        if not self.dfx_config.dump_enabled():
            return False
        max_times = self.dfx_config.dump_max_times()
        return not (max_times <= 0 and self._dump_max_times <= 0)

    def _use_pending_dump_sync(self) -> bool:
        return use_pending_dump_sync(getattr(self, "runner", None))

    def _should_run_anomaly_check(self) -> bool:
        return should_run_anomaly_check_on_rank(getattr(self, "runner", None))

    def _clear_pending_dump(self) -> None:
        self._pending_dump = False
        self._pending_dump_req_id = None
        self._pending_dump_skip_quota = False

    def _activate_msprobe_dump(self, req_id: str | None, *, consume_quota: bool = True) -> bool:
        """Turn on dump_enable + reload on this rank (called after sync decide).

        ``consume_quota=False``: ``dump.manual_trigger`` — do not bump count / cooldown.
        """
        if self._debugger is None:
            logger.error(
                "[Anomaly msprobe] skip dump activate req_id=%s: debugger is None",
                req_id,
            )
            return False
        if self._msprobe_dump_active:
            return True
        if not self.set_msprobe_dump_state(True):
            logger.error(
                "[Anomaly msprobe] set dump state failed req_id=%s",
                req_id,
            )
            return False
        self._msprobe_dump_active = True
        self._dump_needs_forward = True
        self._dump_forward_seen = False
        if consume_quota:
            if req_id is not None:
                self._msprobe_dumped_req_ids.add(req_id)
            self._msprobe_dump_total_count += 1
            self._msprobe_last_dump_ts = time.time()
        # Commit arm→activate wave meta for dump_finish sidecars (manual too).
        self._commit_dump_finish_metas(consume_quota=consume_quota)

        logger.info(
            "[Anomaly msprobe] activate ok req_id=%s count=%d/%d consume_quota=%s %s",
            req_id,
            self._msprobe_dump_total_count,
            self._dump_max_times,
            consume_quota,
            self.dump_rank_tag(),
        )
        return True

    def sync_dump_pending_or(self, *, allow_arm: bool = True) -> bool:
        """Align dump among last-PP TP ranks (dump OR only; no config sync).

        Call **after** runner ``dfx.refresh_config()`` / ``sync_dfx_config``.
        Config sync is a per-DP (or file-poll) step and must run on every rank
        of that EngineCore; this method is last-PP TP only — do not fold
        config reload into it.

        Only **last PP** dumps (precision compare usually needs the final stage).
        Early PP skip entirely — no PP / world collective here.

        When pending-OR is enabled (async, or sync with TP>1): OR ``pending_dump``
        across TP; if any rank armed, all last-PP TPs activate together.

        ``allow_arm``: False on dummy/capture — last-PP TPs still join the
        all_reduce (avoid deadlock) but do not activate or clear pending.
        """
        # Fast path — fully-off default service. With hot-reload disabled AND
        # the dump sink off, nothing can ever arm (auto dump, anomaly arm, and
        # manual_trigger all require ``dump.enabled``), so the pending-OR is always 0.
        # Skip the collective entirely: a default run with no DFX params gets
        # zero distributed overhead per step. Safe because with hot-reload off
        # ``dump.enabled`` is a static startup value, identical across last-PP
        # TPs (assumes a consistent startup config per EngineCore). Returns
        # exactly what the OR would (False) — no behavioral change.
        dfx_cfg = getattr(self, "dfx_config", None)
        if dfx_cfg is not None and not dfx_cfg.hot_reload_enabled and not dfx_cfg.dump_enabled():
            return False

        tag = self.dump_rank_tag()
        if not self._use_pending_dump_sync():
            if not self._anomaly_dump_feature_enabled() and not self._pending_dump:
                return False
            return self._msprobe_dump_active

        pp_group = get_pp_group()
        if not pp_group.is_last_rank:
            return False

        tp_group = get_tp_group()
        # Always join OR on last-PP (even if local pending is false /
        # anomaly detectors are off). A peer with manual_trigger pending must not
        # hang alone in all_reduce.
        local_pending = 1 if self._pending_dump else 0
        # Only the arming rank has skip_quota; peers default False — must OR
        # this flag too or manual_trigger peers incorrectly bump max_times.
        local_skip_quota = 1 if (self._pending_dump and self._pending_dump_skip_quota) else 0
        logger.debug(
            "[DFX sync] enter stage=dump_pending_or local=%d skip_quota=%d allow_arm=%s tp_world=%s %s",
            local_pending,
            local_skip_quota,
            allow_arm,
            tp_group.world_size,
            tag,
        )
        # CPU int32 SUM: OR = sum > 0. tp_group.world_size is TP size only
        # (e.g. DP2/PP2/TP2 → 2, not 8).
        flags_t = torch.tensor([local_pending, local_skip_quota], dtype=torch.int32)
        if tp_group.world_size > 1:
            torch.distributed.all_reduce(flags_t, group=tp_group.cpu_group)
        pending_sum = int(flags_t[0].item())
        skip_quota_sum = int(flags_t[1].item())
        logger.debug(
            "[DFX sync] leave stage=dump_pending_or sum=%d skip_quota_sum=%d %s",
            pending_sum,
            skip_quota_sum,
            tag,
        )
        if pending_sum <= 0:
            return False

        if not allow_arm:
            return False

        # Prefer local armed req_id (TP0); peers keep None for manual_trigger.
        req_id = self._pending_dump_req_id
        consume_quota = skip_quota_sum == 0
        logger.debug(
            "[DFX sync] enter stage=dump_activate req_id=%s consume_quota=%s %s",
            req_id,
            consume_quota,
            tag,
        )
        if not self._activate_msprobe_dump(req_id, consume_quota=consume_quota):
            if self._pending_dump:
                logger.error("[Anomaly msprobe] dump activate failed after OR; keep pending")
            logger.debug("[DFX sync] leave stage=dump_activate ok=False %s", tag)
            return False
        self._clear_pending_dump()
        logger.debug("[DFX sync] leave stage=dump_activate ok=True %s", tag)
        return True

    def enable_msprobe_dump_if_needed(
        self,
        req_id: str,
        req_idx: int | None = None,
        *,
        skip_related_check: bool = False,
        consume_quota: bool = True,
        finish_req_ids: list[str] | None = None,
        anomaly_type: str | None = None,
        source: str = "anomaly",
        arm_wave: int | None = None,
    ) -> bool:
        if self._debugger is None:
            logger.error(
                "[Anomaly msprobe] skip dump req_id=%s: debugger is None",
                req_id,
            )
            return False
        if not self.dfx_config.dump_enabled():
            logger.warning(
                "[Anomaly msprobe] skip dump req_id=%s: dump.enabled=false",
                req_id,
            )
            return False
        if not get_pp_group().is_last_rank:
            return False
        if not skip_related_check and not self.is_related_local_request(req_id, req_idx):
            return False
        # Input filters run at detect time via InputFilterManager; manual_trigger
        # bypasses detectors' filter and arms here without re-checking.
        if self._pending_dump or self._msprobe_dump_active:
            # Already armed / dumping this cycle.
            return True
        if consume_quota:
            if req_id in self._msprobe_dumped_req_ids:
                return False
            max_times = self._dump_max_times
            if max_times <= 0 or self._msprobe_dump_total_count >= max_times:
                logger.info_once(
                    "[Anomaly msprobe] skip dump req_id=%s: dump.max_times=%d count=%d",
                    req_id,
                    max_times,
                    self._msprobe_dump_total_count,
                )
                return False
            now_ts = time.time()
            elapsed = None if self._msprobe_last_dump_ts is None else now_ts - self._msprobe_last_dump_ts
            if elapsed is not None and elapsed < self._dump_cooldown_seconds:
                return False

        tracked = list(finish_req_ids) if finish_req_ids is not None else ([req_id] if req_id else [])
        # Async: only arm pending; dump_enable + reload, dumped_req_ids, and
        # cooldown timestamp happen in _activate_msprobe_dump after OR sync so
        # a failed activate does not permanently blacklist the request.
        if self._use_pending_dump_sync():
            self._begin_dump_wave_tracking(
                tracked,
                anomaly_type=anomaly_type,
                source=source,
                arm_wave=arm_wave,
            )
            self._pending_dump = True
            self._pending_dump_req_id = req_id if consume_quota else None
            self._pending_dump_skip_quota = not consume_quota
            logger.info(
                "[Anomaly msprobe] req_id=%s armed pending_dump (await OR sync). "
                "next_activation_count=%d/%d consume_quota=%s",
                req_id,
                self._msprobe_dump_total_count + (1 if consume_quota else 0),
                self._dump_max_times,
                consume_quota,
            )
            return True

        self._begin_dump_wave_tracking(
            tracked,
            anomaly_type=anomaly_type,
            source=source,
            arm_wave=arm_wave,
        )
        return self._activate_msprobe_dump(req_id if consume_quota else None, consume_quota=consume_quota)
