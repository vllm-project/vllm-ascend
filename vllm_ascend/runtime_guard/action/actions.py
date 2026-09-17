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

"""Configurable incident actions.

Hot path: :meth:`Action.prepare` sync-snapshots live tensors. Cold path:
:meth:`Action.commit` writes files on the action worker thread.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.runtime_guard.incident import Incident
from vllm_ascend.runtime_guard.kv_cache_reader import (
    DEFAULT_DUMP_FREE_HEADROOM_BYTES,
    KvCacheReader,
    free_bytes_at,
)
from vllm_ascend.runtime_guard.manual_trigger import MANUAL_TRIGGER_TYPE
from vllm_ascend.runtime_guard.quota import DumpQuota
from vllm_ascend.runtime_guard.rank_gate import (
    dump_rank_tag,
    runner_tp_rank,
    runner_tp_world_size,
)
from vllm_ascend.runtime_guard.report import ReportWriter
from vllm_ascend.runtime_guard.dump_io import (
    write_kv_dump_request_info,
    write_kv_dump_skipped,
)
from vllm_ascend.runtime_config.config import RuntimeConfig

logger = init_logger_ascend(__name__)


@dataclass
class ActionContext:
    incident: Incident
    runner: Any
    runtime_config: RuntimeConfig
    report_writer: ReportWriter
    kv_reader: KvCacheReader
    quota: DumpQuota
    rank_tag: str
    tokenizer: Any | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    action_overrides: dict[str, Any] = field(default_factory=dict)


class Action(ABC):
    name: str
    sync_only: bool = False  # run entirely on the inference thread
    heavy: bool = False  # full queue drops commit (never inline)

    def prepare(self, ctx: ActionContext) -> Any | None:
        """Sync phase: capture CPU-side payloads. Return None to skip commit."""
        return None

    def commit(self, prepared: Any) -> None:
        """Async phase: persist prepared payloads."""
        return None

    def run(self, ctx: ActionContext) -> None:
        prepared = self.prepare(ctx)
        if prepared is not None:
            self.commit(prepared)


@dataclass
class ReportPrepared:
    report_writer: Any
    kwargs: dict[str, Any]


class ReportAction(Action):
    name = "report"

    def prepare(self, ctx: ActionContext) -> ReportPrepared | None:
        # last-PP TP0 only (same gate as detectors); ReportWriter uses shared report_dir.
        dump_count, dump_max_times = ctx.quota.snapshot()
        actions = ctx.action_overrides.get("_actions", [])
        dump_attempted = "dump_kv" in actions
        # Manual per-req reports often run with action_override=["report"] while
        # D2H happens in ``_run_kv_dumps`` / a separate handle — still attempted.
        if ctx.incident.incident_type == MANUAL_TRIGGER_TYPE:
            dump_attempted = True
        return ReportPrepared(
            report_writer=ctx.report_writer,
            kwargs={
                "incident_type": ctx.incident.incident_type,
                "req_id": ctx.incident.req_id,
                "detail": dict(ctx.detail),
                "rank_tag": ctx.rank_tag,
                "tokenizer": ctx.tokenizer,
                "dump_attempted": dump_attempted,
                "dump_count": dump_count,
                "dump_max_times": dump_max_times,
                "dump_arm_wave": ctx.incident.wave,
            },
        )

    def commit(self, prepared: ReportPrepared) -> None:
        prepared.report_writer.write(**prepared.kwargs)


@dataclass
class DumpKvPrepared:
    """Arm-time file payloads; ``commit`` writes them off the inference thread."""

    skip: dict[str, Any] | None = None
    request_infos: list[dict[str, Any]] = field(default_factory=list)


class DumpKvAction(Action):
    """Arm a KV dump: record jobs on TP0; last-PP all TP dump at sample end.

    Default ``scope=request`` uses ``incident.block_ids`` only — saves D2H time
    and disk vs dumping every block. ``incident_type=manual_trigger`` always
    uses ``all_requests`` (``scope`` ignored).

    Detection stays last-PP TP0. This action does **not** D2H; it queues
    ``{req_id, ...}``. At the end of ``run_sample_phase`` (or sync when this
    step has no sample), every last-PP TP rank drains and dumps its shard
    (block_ids resolved then). Finished/reaped requests are always skipped
    (arm and drain). Other PP stages are not dumped.

    Two-phase like every async action: ``prepare`` (inference thread) gates,
    estimates, consumes quota and queues jobs; ``commit`` (action worker)
    writes the arm-time metadata files (``request_info.json`` /
    ``dump_skipped.json``), so no dump-path file I/O runs on the hot path.
    """

    name = "dump_kv"

    def prepare(self, ctx: ActionContext) -> DumpKvPrepared | None:
        cfg = ctx.action_overrides.get("dump_kv") or {}
        if isinstance(cfg, bool):
            cfg = {}
        incident = ctx.incident
        is_manual = incident.incident_type == MANUAL_TRIGGER_TYPE
        # Manual dump always enumerates the live batch; ``scope`` is ignored.
        scope = "all_requests" if is_manual else str(cfg.get("scope", "request"))
        from vllm_ascend.runtime_guard.request_state import RequestGuardStore

        # Synthetic ``__manual_trigger__`` is never a Store/KV owner — skip the
        # per-incident finished gate; ``all_requests`` filters real reqs below.
        if not is_manual and not RequestGuardStore.get().kv_dump_allowed(
            str(incident.req_id or "")
        ):
            return DumpKvPrepared(skip=_dump_skip_kwargs(ctx, reason="finished_or_reaped"))
        base = Path(ctx.runtime_config.dump_root()) / ctx.incident.incident_type

        targets = _resolve_dump_targets(ctx, scope)
        if not targets:
            logger.warning(
                "[runtime_guard dump_kv] skip: no dump targets req_id=%s type=%s scope=%s",
                ctx.incident.req_id,
                ctx.incident.incident_type,
                scope,
            )
            return DumpKvPrepared(
                skip=_dump_skip_kwargs(ctx, reason="no_dump_targets", detail={"scope": scope})
            )
        if all(not bids for _req, bids in targets):
            # Manual arm runs in sync_for_step *before* prepare_inputs, so the
            # first prefill wave often has req ids but empty block tables.
            # Still queue; drain at sample-end flush resolves block_ids. Auto
            # paths usually arm after sample with known blocks — keep the hard skip.
            if not is_manual:
                logger.warning(
                    "[runtime_guard dump_kv] skip: empty block_ids req_id=%s type=%s",
                    ctx.incident.req_id,
                    ctx.incident.incident_type,
                )
                return DumpKvPrepared(
                    skip=_dump_skip_kwargs(
                        ctx,
                        reason="empty_block_ids",
                        detail={"n_targets": len(targets)},
                    )
                )
            logger.info(
                "[runtime_guard dump_kv] arm with empty block_ids "
                "(resolve at sample-end drain) req_id=%s type=%s n_targets=%d",
                ctx.incident.req_id,
                ctx.incident.incident_type,
                len(targets),
            )

        estimated = 0
        for _req_id, block_ids in targets:
            if not block_ids:
                continue
            try:
                piece = ctx.kv_reader.estimate_dump_bytes(block_ids=block_ids)
                estimated += int(piece)
            except (TypeError, ValueError):
                continue
        # last-PP dumps every TP shard; scale single-rank estimate by tp_size.
        tp_size = runner_tp_world_size(ctx.runner)
        estimated *= tp_size
        try:
            headroom = int(
                ctx.runtime_config.dump_get("free_headroom_bytes", DEFAULT_DUMP_FREE_HEADROOM_BYTES)
            )
        except (TypeError, ValueError):
            headroom = DEFAULT_DUMP_FREE_HEADROOM_BYTES
        # Skip free-space gate when estimate is unknown (deferred block_ids).
        if estimated > 0:
            needed = estimated + max(0, headroom)
            free = free_bytes_at(base)
            if isinstance(free, int) and free < needed:
                logger.warning(
                    "[runtime_guard dump_kv] skip: free=%d needed=%d "
                    "(payload=%d tp_size=%d headroom=%d) dir=%s req_id=%s",
                    free,
                    needed,
                    estimated,
                    tp_size,
                    headroom,
                    base,
                    ctx.incident.req_id,
                )
                return DumpKvPrepared(
                    skip=_dump_skip_kwargs(
                        ctx,
                        reason="insufficient_free_space",
                        detail={
                            "free_bytes": free,
                            "needed_bytes": needed,
                            "estimated_payload_bytes": estimated,
                            "tp_size": tp_size,
                            "headroom_bytes": headroom,
                        },
                    )
                )

        if not ctx.quota.try_consume(consume_quota=ctx.incident.consume_quota):
            logger.warning(
                "[runtime_guard dump_kv] quota/cooldown blocked req_id=%s type=%s",
                ctx.incident.req_id,
                ctx.incident.incident_type,
            )
            return DumpKvPrepared(skip=_dump_skip_kwargs(ctx, reason="quota_or_cooldown_blocked"))

        queued_ids = _queue_kv_dumps(ctx, targets)
        if not queued_ids:
            ctx.quota.refund(consume_quota=ctx.incident.consume_quota)
            logger.warning(
                "[runtime_guard dump_kv] queue failed; refunded quota req_id=%s type=%s",
                ctx.incident.req_id,
                ctx.incident.incident_type,
            )
            return DumpKvPrepared(
                skip=_dump_skip_kwargs(
                    ctx,
                    reason="queue_failed",
                    detail={"n_targets": len(targets)},
                )
            )
        infos = _build_request_infos_for_targets(
            ctx,
            [(r, b) for r, b in targets if r in queued_ids],
        )
        if not infos:
            return None
        # manual_dump count is always consumed in ``_handle_manual_trigger``
        # after an armed wave (success or dump_skipped marker).
        return DumpKvPrepared(request_infos=infos)

    def commit(self, prepared: DumpKvPrepared) -> None:
        if prepared.skip is not None:
            write_kv_dump_skipped(**prepared.skip)
        for info in prepared.request_infos:
            write_kv_dump_request_info(**info)


def _dump_skip_kwargs(
    ctx: ActionContext,
    *,
    reason: str,
    detail: dict[str, Any] | None = None,
    stage: str = "arm",
) -> dict[str, Any]:
    """``write_kv_dump_skipped`` kwargs built at arm time (commit writes them)."""
    return {
        "dump_root": ctx.runtime_config.dump_root(),
        "req_id": str(ctx.incident.req_id or "unknown"),
        "incident_type": str(ctx.incident.incident_type or "unknown"),
        "reason": reason,
        "stage": stage,
        "rank_tag": dump_rank_tag(ctx.runner) if ctx.runner is not None else ctx.rank_tag,
        "wave": ctx.incident.wave,
        "detail": detail,
    }


def _resolve_dump_targets(
    ctx: ActionContext,
    scope: str,
) -> list[tuple[str, list[int]]]:
    """Build ``(req_id, block_ids)`` for arm-time estimate / request_info / queue.

    ``scope=all_requests`` lists live local requests via
    ``iter_local_request_rows`` and resolves each req's blocks via
    ``block_ids_for_request`` (never reuses the incident's ``detail.block_ids``
    for other requests).
    """
    from vllm_ascend.runtime_guard.kv_block_meta import block_ids_for_request
    from vllm_ascend.runtime_guard.manual_trigger import iter_local_request_rows
    from vllm_ascend.runtime_guard.request_state import RequestGuardStore

    if scope != "all_requests":
        return [
            (str(ctx.incident.req_id or "unknown"), list(ctx.incident.block_ids or [])),
        ]

    rows = list(iter_local_request_rows(ctx.runner))
    store = RequestGuardStore.get()
    targets: list[tuple[str, list[int]]] = []
    for req_id, req_idx in rows:
        rid = str(req_id)
        if not rid:
            continue
        if not store.kv_dump_allowed(rid):
            continue
        if rid == str(ctx.incident.req_id or "") and ctx.incident.block_ids:
            bids = list(ctx.incident.block_ids)
        else:
            bids = list(block_ids_for_request(ctx.runner, rid, req_idx) or [])
        targets.append((rid, bids))
    return targets


def _detail_for_dump_req(ctx: ActionContext, req_id: str) -> dict[str, Any]:
    """Per-req detail for ``request_info.json`` (manual batch uses ``detail.requests``)."""
    detail = dict(ctx.detail or {})
    requests = detail.get("requests")
    if isinstance(requests, list):
        for entry in requests:
            if isinstance(entry, dict) and str(entry.get("req_id") or "") == str(req_id):
                return dict(entry)
    return detail


def _build_request_infos_for_targets(
    ctx: ActionContext,
    targets: list[tuple[str, list[int]]],
) -> list[dict[str, Any]]:
    """Arm-time TP0: per-target ``write_kv_dump_request_info`` kwargs (commit writes)."""
    if runner_tp_rank(ctx.runner) != 0:
        return []
    rc = ctx.runtime_config
    try:
        save_sensitive = bool(rc.report_save_sensitive_info())
    except Exception:
        save_sensitive = False
    try:
        decode_ids = bool(rc.report_decode_token_ids())
    except Exception:
        decode_ids = True
    try:
        max_prompt = int(rc.report_max_prompt_token_ids())
    except Exception:
        max_prompt = 1000
    try:
        max_output = int(rc.report_max_output_token_ids())
    except Exception:
        max_output = 1000
    dump_root = ctx.runtime_config.dump_root()
    incident_type = str(ctx.incident.incident_type or "unknown")
    wave = ctx.incident.wave
    rank_tag = dump_rank_tag(ctx.runner) if ctx.runner is not None else ctx.rank_tag
    infos: list[dict[str, Any]] = []
    for req_id, block_ids in targets:
        infos.append(
            {
                "dump_root": dump_root,
                "req_id": str(req_id),
                "incident_type": incident_type,
                "detail": _detail_for_dump_req(ctx, str(req_id)),
                "rank_tag": rank_tag,
                "wave": wave,
                "block_ids": list(block_ids) if block_ids else None,
                "tokenizer": ctx.tokenizer,
                "save_sensitive_info": save_sensitive,
                "decode_token_ids": decode_ids,
                "max_prompt_token_ids": max_prompt,
                "max_output_token_ids": max_output,
            }
        )
    return infos


def _queue_kv_dumps(
    ctx: ActionContext,
    targets: list[tuple[str, list[int]]],
) -> set[str]:
    """TP0 records dump jobs; last-PP all TP (incl. TP0) drain at sample end.

    Returns the set of ``req_id`` newly queued. Pending list dedupes by
    ``(wave, req_id)`` (at most one dump job per request per arm wave).
    """
    if runner_tp_rank(ctx.runner) != 0:
        return set()
    queue = getattr(getattr(ctx.runner, "runtime_guard", None), "queue_kv_dump", None)
    if not callable(queue):
        return set()
    import uuid

    arm_id = uuid.uuid4().hex
    consume = bool(ctx.incident.consume_quota)
    wave = ctx.incident.wave
    queued: set[str] = set()
    for req_id, _block_ids in targets:
        rid = str(req_id)
        added = queue(
            {
                "req_id": rid,
                "incident_type": str(ctx.incident.incident_type or "unknown"),
                "consume_quota": consume,
                "arm_id": arm_id,
                "wave": int(wave) if wave is not None else None,
            }
        )
        if added:
            queued.add(rid)
    return queued


class SetLogLevelAction(Action):
    name = "set_log_level"
    sync_only = True

    def run(self, ctx: ActionContext) -> None:
        cfg = ctx.action_overrides.get("set_log_level") or {}
        if not isinstance(cfg, dict):
            return
        level = cfg.get("level")
        modules = cfg.get("modules")
        if level is None and not modules:
            return
        from vllm_ascend.logger import apply_ascend_log_level

        apply_ascend_log_level(
            str(level or ctx.runtime_config.ascend_log_level()),
            module_levels=dict(modules) if isinstance(modules, dict) else None,
        )
        logger.info(
            "[runtime_guard set_log_level] incident=%s level=%s modules=%s",
            ctx.incident.incident_type,
            level,
            modules,
        )


_ACTIONS: dict[str, Action] = {
    ReportAction.name: ReportAction(),
    DumpKvAction.name: DumpKvAction(),
    SetLogLevelAction.name: SetLogLevelAction(),
}


def get_action(name: str) -> Action | None:
    return _ACTIONS.get(name)
