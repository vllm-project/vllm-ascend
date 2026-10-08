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
from vllm_ascend.observability.runtime_config._defaults import _DEFAULTS, DUMP_FREE_HEADROOM_BYTES
from vllm_ascend.observability.runtime_config.config import RuntimeConfig
from vllm_ascend.observability.runtime_guard.action.queue import ActionQueue
from vllm_ascend.observability.runtime_guard.dump import (
    REQUEST_FINISHED_AT_DUMP_KEY,
    KvCacheReader,
    free_bytes_at,
    write_kv_dump_request_info,
    write_kv_dump_skipped,
)
from vllm_ascend.observability.runtime_guard.rank_gate import (
    anomaly_check_rank_skip_reason,
    dump_rank_tag,
    runner_tp_rank,
    runner_tp_world_size,
    should_run_anomaly_check_on_rank,
)
from vllm_ascend.observability.runtime_guard.report import ReportWriter
from vllm_ascend.observability.runtime_guard.state import MANUAL_TRIGGER_TYPE, DumpQuota, Incident

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
        from vllm_ascend.observability.runtime_guard.state import RequestGuardStore

        dump_count, dump_max_times = ctx.quota.snapshot()
        actions = ctx.action_overrides.get("_actions", [])
        dump_attempted = "dump_kv" in actions
        # Manual per-req reports often run with action_override=["report"] while
        # D2H happens in ``_run_kv_dumps`` / a separate handle — still attempted.
        if ctx.incident.incident_type == MANUAL_TRIGGER_TYPE:
            dump_attempted = True
        finished_at_dump = False
        if dump_attempted and ctx.incident.req_id:
            finished_at_dump = RequestGuardStore.get().is_request_finished(str(ctx.incident.req_id))
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
                REQUEST_FINISHED_AT_DUMP_KEY: finished_at_dump,
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
    ``{req_id, ...}``. Drain is +1 wave (detector) or end-of-wave (manual).
    Reaped ids are refused at arm; finished-but-not-reaped may still arm
    (``mark_finished`` can precede ``check_after_sample``) and stamp
    ``request_finished_at_dump``. Armed jobs always attempt D2H. Other PP
    stages are not dumped.

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
        from vllm_ascend.observability.runtime_guard.state import RequestGuardStore

        # Synthetic ``__manual_trigger__`` is never a Store/KV owner — skip the
        # per-incident reaped gate; ``all_requests`` filters real reqs below.
        if not is_manual and not RequestGuardStore.get().kv_dump_allowed(str(incident.req_id or "")):
            return DumpKvPrepared(skip=_dump_skip_kwargs(ctx, reason="reaped"))
        base = Path(ctx.runtime_config.dump_root()) / ctx.incident.incident_type

        targets = _resolve_dump_targets(ctx, scope)
        if not targets:
            logger.warning(
                "[runtime_guard dump_kv] skip: no dump targets req_id=%s type=%s scope=%s",
                ctx.incident.req_id,
                ctx.incident.incident_type,
                scope,
            )
            return DumpKvPrepared(skip=_dump_skip_kwargs(ctx, reason="no_dump_targets", detail={"scope": scope}))
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
        headroom = DUMP_FREE_HEADROOM_BYTES
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

        # Refund on any exception after a successful consume so soft-fail in
        # ActionExecutor does not leave auto quota / cooldown stuck.
        try:
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
            # manual_dump watermark is always bumped in ``_handle_manual_trigger``
            # / ``_maybe_fire_manual_local`` after an armed wave (success or dump_skipped).
            return DumpKvPrepared(request_infos=infos)
        except Exception:
            ctx.quota.refund(consume_quota=ctx.incident.consume_quota)
            logger.exception(
                "[runtime_guard dump_kv] prepare failed after consume; refunded quota req_id=%s type=%s",
                ctx.incident.req_id,
                ctx.incident.incident_type,
            )
            raise

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
    from vllm_ascend.observability.runtime_guard.dump import block_ids_for_request
    from vllm_ascend.observability.runtime_guard.state import RequestGuardStore, iter_local_request_rows

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
    save_sensitive = bool(rc.report_save_sensitive_info())
    decode_ids = bool(rc.report_decode_token_ids())
    max_prompt = int(rc.report_max_prompt_token_ids())
    max_output = int(rc.report_max_output_token_ids())
    dump_root = ctx.runtime_config.dump_root()
    incident_type = str(ctx.incident.incident_type or "unknown")
    wave = ctx.incident.wave
    rank_tag = dump_rank_tag(ctx.runner) if ctx.runner is not None else ctx.rank_tag
    from vllm_ascend.observability.runtime_guard.state import RequestGuardStore

    store = RequestGuardStore.get()
    infos: list[dict[str, Any]] = []
    for req_id, block_ids in targets:
        rid = str(req_id)
        infos.append(
            {
                "dump_root": dump_root,
                "req_id": rid,
                "incident_type": incident_type,
                "detail": _detail_for_dump_req(ctx, rid),
                "rank_tag": rank_tag,
                "wave": wave,
                "block_ids": list(block_ids) if block_ids else None,
                "tokenizer": ctx.tokenizer,
                "save_sensitive_info": save_sensitive,
                "decode_token_ids": decode_ids,
                "max_prompt_token_ids": max_prompt,
                "max_output_token_ids": max_output,
                REQUEST_FINISHED_AT_DUMP_KEY: store.is_request_finished(rid),
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
    Freezes arm-time ``block_ids`` and ``request_finished_at_dump`` into the job.
    """
    if runner_tp_rank(ctx.runner) != 0:
        return set()
    queue = getattr(getattr(ctx.runner, "runtime_guard", None), "queue_kv_dump", None)
    if not callable(queue):
        return set()
    import uuid

    from vllm_ascend.observability.runtime_guard.state import RequestGuardStore

    store = RequestGuardStore.get()
    arm_id = uuid.uuid4().hex
    consume = bool(ctx.incident.consume_quota)
    wave = ctx.incident.wave
    queued: set[str] = set()
    for req_id, block_ids in targets:
        rid = str(req_id)
        bids = list(block_ids) if block_ids else []
        added = queue(
            {
                "req_id": rid,
                "incident_type": str(ctx.incident.incident_type or "unknown"),
                "consume_quota": consume,
                "arm_id": arm_id,
                "wave": int(wave) if wave is not None else None,
                "block_ids": bids,
                REQUEST_FINISHED_AT_DUMP_KEY: store.is_request_finished(rid),
            }
        )
        if added:
            queued.add(rid)
    return queued


_ACTIONS: dict[str, Action] = {
    ReportAction.name: ReportAction(),
    DumpKvAction.name: DumpKvAction(),
}


def get_action(name: str) -> Action | None:
    return _ACTIONS.get(name)


# ---- action executor ----

# Same fallback as ``_DEFAULTS["actions"]["defaults"]["on_trigger"]``.
_DEFAULT_ACTIONS: list[str] = list(_DEFAULTS["actions"]["defaults"]["on_trigger"])


def order_incident_actions(names: list[str]) -> list[str]:
    """Dedup then order: sync_only → report → dump_kv → other async actions."""
    ordered: list[str] = []
    for name in names:
        if name not in ordered:
            ordered.append(name)
    if "report" not in ordered or "dump_kv" not in ordered:
        return ordered
    rest = [n for n in ordered if n not in ("report", "dump_kv")]

    def _is_sync_only(n: str) -> bool:
        act = get_action(n)
        return bool(act is not None and act.sync_only)

    head = [n for n in rest if _is_sync_only(n)]
    tail = [n for n in rest if n not in head]
    return head + ["report", "dump_kv"] + tail


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
            action_queue if action_queue is not None else ActionQueue(maxsize=runtime_config.action_queue_max_size())
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
        ordered = order_incident_actions(names)

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
