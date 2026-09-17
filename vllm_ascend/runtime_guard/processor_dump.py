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

"""KV dump queue / claim / D2H orchestration mixin for RuntimeGuardProcessor."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from vllm.distributed.parallel_state import get_tp_group

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.runtime_config._task_bus import sync_task_bus
from vllm_ascend.runtime_guard.kv_block_meta import block_ids_for_request
from vllm_ascend.runtime_guard.kv_cache_reader import KvCacheReader
from vllm_ascend.runtime_guard.rank_gate import (
    dump_rank_tag,
    runner_tp_rank,
    should_dump_kv_on_rank,
)
from vllm_ascend.runtime_guard.request_state import RequestGuardStore
from vllm_ascend.runtime_guard.dump_io import kv_dump_wave_dirname, write_kv_dump_skipped

logger = init_logger_ascend(__name__)


class RuntimeGuardDumpMixin:
    """Mixin: queue / claim / run KV dump jobs (wave-deferred D2H)."""

    def queue_kv_dump(self, job: dict[str, Any]) -> bool:
        """TP0: record a dump job for last-PP all TP.

        Delivered at the *next* wave-head dump lane (detector dump is +1 wave),
        then D2H at that wave's end after prepare. Dedupes by ``(wave, req_id)``
        while still pending (same req may arm again on a later wave).
        """
        if not job or not job.get("req_id"):
            return False
        rid = str(job["req_id"])
        wave = job.get("wave")
        pending = getattr(self, "_kv_dump_jobs", None)
        if pending is None:
            self._kv_dump_jobs = []
            pending = self._kv_dump_jobs
        for j in pending:
            if str(j.get("req_id") or "") != rid:
                continue
            if j.get("wave") == wave:
                logger.info(
                    "[runtime_guard dump_kv] skip enqueue: already pending "
                    "req_id=%s wave=%s (same-wave dedupe)",
                    rid,
                    wave,
                )
                return False
        pending.append(dict(job))
        return True

    def end_of_wave_sync(self, *, allow_arm: bool = True) -> None:
        """End-of-wave: deferred auto D2H + local manual dump (no dump AR).

        Wave-head already ran the merged due AR / config apply / dump bcast.
        ``allow_arm``: False on dummy / no-sample (do not burn manual_dump).
        """
        deferred = list(getattr(self, "_deferred_kv_dump_jobs", None) or [])
        if hasattr(self, "_deferred_kv_dump_jobs"):
            self._deferred_kv_dump_jobs.clear()
        if deferred and should_dump_kv_on_rank(self.runner):
            self._run_kv_dumps(deferred)
        self._maybe_fire_manual_local(allow_arm=allow_arm)

    def flush_kv_dumps(self) -> None:
        """Alias for :meth:`end_of_wave_sync` (historical name)."""
        self.end_of_wave_sync(allow_arm=True)

    def _claim_dump_jobs_to_deferred_via_tp(self) -> None:
        """File/PP>1: move pending auto jobs through TP bus into deferred D2H."""
        if not should_dump_kv_on_rank(self.runner):
            if hasattr(self, "_kv_dump_jobs"):
                self._kv_dump_jobs.clear()
            return
        payload = list(getattr(self, "_kv_dump_jobs", None) or [])
        if hasattr(self, "_kv_dump_jobs"):
            self._kv_dump_jobs.clear()
        try:
            tp_group = get_tp_group()
        except Exception:
            if payload:
                self._deferred_kv_dump_jobs.extend(payload)
            return
        tp_size = int(getattr(tp_group, "world_size", 1) or 1)
        if tp_size <= 1:
            if payload:
                self._deferred_kv_dump_jobs.extend(payload)
            return
        try:
            rank = int(tp_group.rank_in_group)
        except Exception:
            rank = 0

        jobs = sync_task_bus(
            tp_group,
            due_local=bool(rank == 0 and payload),
            payload=payload if rank == 0 else None,
            src=0,
        )
        if jobs:
            self._deferred_kv_dump_jobs.extend(list(jobs))

    def _run_kv_dumps(self, jobs: list[dict[str, Any]]) -> None:
        ex = getattr(self, "action_executor", None)
        reader = getattr(ex, "_kv_reader", None) or KvCacheReader(self.runner)
        submit = getattr(ex, "submit_heavy", None)
        dump_root = Path(self.runtime_config.dump_root())
        rank_tag = dump_rank_tag(self.runner)
        store = RequestGuardStore.get()
        quota = getattr(self, "quota", None)
        try:
            is_tp0 = runner_tp_rank(self.runner) == 0
        except Exception:
            is_tp0 = True
        # One prepare may queue N jobs after a single try_consume (arm_id).
        # Refund once per arm if that arm produced no D2H snapshot. Later async
        # torch.save failure does not refund (see ops docs).
        arms: dict[str, dict[str, bool]] = defaultdict(lambda: {"debited": False, "ok": False})
        seen_req: set[str] = set()
        for job in jobs:
            arm_id = str(job.get("arm_id") or f"job-{id(job)}")
            if job.get("consume_quota"):
                arms[arm_id]["debited"] = True
            req_id = str(job.get("req_id") or "")
            if not req_id:
                continue
            if req_id in seen_req:
                continue
            seen_req.add(req_id)
            incident_type = str(job.get("incident_type") or "unknown")
            wave = job.get("wave")
            try:
                wave_i = int(wave) if wave is not None else None
            except (TypeError, ValueError):
                wave_i = None
            wave_dir = kv_dump_wave_dirname(wave_i)
            if not store.kv_dump_allowed(req_id):
                write_kv_dump_skipped(
                    dump_root,
                    req_id=req_id,
                    incident_type=incident_type,
                    reason="finished_or_reaped",
                    stage="drain",
                    rank_tag=rank_tag,
                    wave=wave_i,
                )
                continue
            block_ids = list(block_ids_for_request(self.runner, req_id, None) or [])
            if not block_ids:
                logger.warning(
                    "[runtime_guard dump_kv] skip empty local block_ids req_id=%s rank=%s",
                    req_id,
                    rank_tag,
                )
                write_kv_dump_skipped(
                    dump_root,
                    req_id=req_id,
                    incident_type=incident_type,
                    reason="empty_block_ids",
                    stage="drain",
                    rank_tag=rank_tag,
                    wave=wave_i,
                )
                continue
            out_dir = dump_root / incident_type / req_id / wave_dir / rank_tag
            produced = 0
            try:
                for snap in reader.iter_request_snapshots(
                    req_id=req_id,
                    block_ids=block_ids,
                    out_dir=out_dir,
                ):
                    produced += 1
                    # Stamp incident meta for async save failure markers.
                    snap.payload.setdefault("incident_type", incident_type)
                    if wave_i is not None:
                        snap.payload.setdefault("dump_arm_wave", wave_i)
                    snap.payload.setdefault("dump_root", str(dump_root))
                    if submit is not None:
                        submit(lambda s=snap: KvCacheReader.write_snapshots([s]))
                    else:
                        KvCacheReader.write_snapshots([snap])
            except Exception as exc:
                logger.exception(
                    "[runtime_guard dump_kv] dump failed req_id=%s rank=%s",
                    req_id,
                    rank_tag,
                )
                write_kv_dump_skipped(
                    dump_root,
                    req_id=req_id,
                    incident_type=incident_type,
                    reason="d2h_failed",
                    stage="drain",
                    rank_tag=rank_tag,
                    wave=wave_i,
                    detail={"error": f"{type(exc).__name__}: {exc}"},
                )
                continue
            if produced == 0:
                logger.warning(
                    "[runtime_guard dump_kv] no tensors req_id=%s rank=%s",
                    req_id,
                    rank_tag,
                )
                write_kv_dump_skipped(
                    dump_root,
                    req_id=req_id,
                    incident_type=incident_type,
                    reason="no_tensors",
                    stage="drain",
                    rank_tag=rank_tag,
                    wave=wave_i,
                )
                continue
            arms[arm_id]["ok"] = True
            logger.info(
                "[runtime_guard dump_kv] dumped req_id=%s wave=%s rank=%s tensors=%d dir=%s",
                req_id,
                wave_dir,
                rank_tag,
                produced,
                out_dir,
            )
        if is_tp0 and quota is not None:
            for meta in arms.values():
                if meta["debited"] and not meta["ok"]:
                    quota.refund(consume_quota=True)
