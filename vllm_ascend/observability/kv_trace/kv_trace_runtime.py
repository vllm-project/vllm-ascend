# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2-local native binding records, deferred snapshots and output checks."""

from __future__ import annotations

import functools
import logging
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from threading import Lock
from typing import Any

import torch

from vllm_ascend.observability.kv_trace.kv_trace_config import KVTraceConfig
from vllm_ascend.observability.kv_trace.kv_trace_connector import record_output, record_plan
from vllm_ascend.observability.kv_trace.kv_trace_plan import (
    AccessPlan,
    PlanSnapshotter,
    audit_access_plan,
    cache_group,
    cache_layouts,
    metadata_views,
)
from vllm_ascend.observability.kv_trace.kv_trace_state import KVCacheState

_MAX_REPORT_KEYS = 1024


class KVTraceError(RuntimeError):
    """Stop returning model results after a visible address-plan violation."""


class _PendingAudit:
    """One snapshot, consumed once by the output reader or next forward."""

    def __init__(self, complete: Any):
        self._complete = complete
        self._lock = Lock()
        self._error: BaseException | None = None

    def finish(self) -> None:
        with self._lock:
            if self._complete is not None:
                try:
                    self._complete()
                except BaseException as exc:
                    self._error = exc
                finally:
                    self._complete = None
            if self._error is not None:
                raise self._error


class KVTraceRuntime:
    def __init__(self, vllm_config: Any):
        from vllm.logger import logger

        self.logger = logger
        self.config = vllm_config
        self.state = KVCacheState(self.report)
        self.layer_groups: dict[str, int] = {}
        self._layer_layouts: dict[str, dict] = {}
        self._metadata: dict[Any, Any] = {}
        self._view_layouts: dict[Any, dict] = {}
        self._phase: str | None = None
        self._full_graph = False
        self._reported: OrderedDict[tuple, None] = OrderedDict()
        self._scheduled: dict[str, tuple[int | None, int]] | None = None
        self._failure: str | None = None
        options = KVTraceConfig.from_vllm_config(vllm_config)
        self.action = options.action
        self.slot = options.slot
        self._configure_runner: Callable[[], None] | None = None
        self._snapshotter = PlanSnapshotter()
        self._defer_audit = False
        self._pending: _PendingAudit | None = None
        self.layout: dict[int, Any] = {}
        self._batch: dict[str, Any] = {}
        self._parallel: dict[str, Any] = {}
        self._step_id = 0
        self.rank: dict[str, Any] = {}

    def _check_failure(self) -> None:
        if self.action == "raise" and self._failure is not None:
            raise KVTraceError(f"KV Trace stopped model output: {self._failure}; restart the worker before reuse")

    def report(self, event: str, **fields: Any) -> None:
        if self._step_id or event in ("kv_trace.violation", "kv_trace.check"):
            fields.setdefault("step_id", self._step_id)
        if self.rank:
            fields.setdefault("rank", self.rank)
        failed = event == "kv_trace.violation"
        if failed:
            with self.state._lock:
                if self._failure is None:
                    self._failure = str(fields.get("reason", event))
                key = (
                    event,
                    *(
                        fields.get(name)
                        for name in (
                            "operation",
                            "reason",
                            "group_id",
                            "view_id",
                            "field",
                            "block_id",
                            "request_id",
                        )
                    ),
                    tuple(fields.get("request_ids", ())),
                )
                if key in self._reported:
                    self._reported.move_to_end(key)
                    return
                self._reported[key] = None
                if len(self._reported) > _MAX_REPORT_KEYS:
                    self._reported.popitem(last=False)
        level = (
            logging.ERROR
            if failed
            else logging.INFO
            if event in ("kv_trace.enabled", "kv_trace.check")
            else logging.DEBUG
        )
        if self.logger.isEnabledFor(level):
            payload = " ".join(f"{key}={value}" for key, value in fields.items())
            self.logger.log(level, "[KV_TRACE] %s %s", event, payload)

    def gap(self, reason: str, **fields: Any) -> None:
        self.report("kv_trace.check", status="unavailable", reason=reason, **fields)

    def configure(self, runner: Any) -> None:
        config = self.config
        parallel = config.parallel_config
        transfer = config.kv_transfer_config
        sizes = {}
        tables = runner.block_tables
        self.rank = {name: getattr(runner, name, None) for name in ("tp_rank", "pp_rank", "dp_rank", "dcp_rank")}
        self.layout = cache_layouts(runner.kv_cache_config, tables, parallel, self.rank["dcp_rank"])
        self.layer_groups = {
            name: group_id
            for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups)
            for name in group.layer_names
        }
        manager_sizes = getattr(tables, "block_sizes", None)
        for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            sizes[group_id] = spec.block_size
            if getattr(config.model_config, "is_encoder_decoder", False):
                self.layout[group_id]["reason"] = "encoder_decoder_addressing"
            elif manager_sizes is not None and (
                group_id >= len(manager_sizes) or manager_sizes[group_id] != spec.block_size
            ):
                self.layout[group_id]["reason"] = "block_coordinates_unadapted"

        cp_enabled = (
            getattr(parallel, "prefill_context_parallel_size", 1) > 1
            or getattr(parallel, "decode_context_parallel_size", 1) > 1
        )
        self._layer_layouts = {}
        for group_id, groups in enumerate(getattr(runner, "attn_groups", ())):
            layout = self.layout[group_id]
            if layout["reason"]:
                continue
            native_kernel = layout["kernel_block_size"]
            for group in groups:
                kernel = getattr(group, "kernel_block_size", None) or native_kernel
                stride = getattr(group, "kernel_block_stride", 0)
                if cp_enabled and group.backend.get_name() in (
                    "ASCEND_SFA",
                    "ASCEND_SFA_INDEXER",
                    "ASCEND_HISPARSE_INDEXER",
                ):
                    overrides = {"reason": "sfa_cp_metadata_unadapted"}
                elif kernel <= 0 or native_kernel % kernel or stride not in (0, sizes[group_id] // kernel):
                    overrides = {"reason": "block_coordinates_unadapted"}
                elif kernel != native_kernel:
                    overrides = {"kernel_block_size": kernel}
                else:
                    continue
                self._layer_layouts.update(dict.fromkeys(group.layer_names, overrides))
        self.state.configure(sizes, runner.kv_cache_config.num_blocks)
        self._defer_audit = (
            bool(getattr(getattr(config, "scheduler_config", None), "async_scheduling", False))
            and transfer is None
            # Pooling bypasses sample_tokens; non-reply workers may discard outputs.
            and getattr(runner, "is_pooling_model", None) is False
            and getattr(parallel, "tensor_parallel_size", None) == 1
            and getattr(parallel, "pipeline_parallel_size", None) == 1
            and hasattr(runner, "sample_tokens")
            and getattr(parallel, "prefill_context_parallel_size", 1) == 1
            and getattr(parallel, "decode_context_parallel_size", 1) == 1
            and not getattr(parallel, "enable_dbo", False)
            and config.speculative_config is None
        )
        if self._configure_runner is not None:
            self._configure_runner()
        self.report("kv_trace.enabled", failure_action=self.action, slot=self.slot, groups=self.layout)
        if config.speculative_config is not None:
            self.gap("draft_address_not_in_scope", role="draft")

    def observe_metadata(self, metadata: Any) -> None:
        try:
            self._metadata, self._view_layouts = {}, {}
            if metadata is None:
                return
            if not isinstance(metadata, dict):
                self.gap("microbatch_metadata_unadapted")
                return
            self._metadata, self._view_layouts = metadata_views(metadata, self.layer_groups, self._layer_layouts)
            for group in self.state.group_sizes.keys() - {cache_group(key) for key in self._metadata}:
                self.gap("metadata_write_mapping_missing", group_id=group)
        except Exception as exc:
            self._metadata, self._view_layouts = {}, {}
            self.gap(f"metadata_collection_failed:{type(exc).__name__}")

    def _finish_pending(self) -> None:
        if self._pending is not None:
            try:
                self._pending.finish()
            finally:
                self._pending = None
        self._check_failure()

    def _current_plan(self, runner: Any) -> AccessPlan:
        assert self._phase is not None
        ids = list(runner.request_ids) if self._phase == "forward" else []
        dynamic = self.config.speculative_config is not None
        associated = not (
            getattr(self.config.parallel_config, "prefill_context_parallel_size", 1) > 1 and not self._parallel
        )
        global_ids = self._parallel.get("batch", {}).get("request_ids", ids)
        associated = associated and not (dynamic and (not self._batch or len(global_ids) != len(set(global_ids))))
        return AccessPlan(
            phase=self._phase,
            request_ids=ids,
            scheduled=dict(self._scheduled) if self._scheduled is not None and associated else None,
            metadata=dict(self._metadata),
            full_graph=self._full_graph,
            bindings=dict(self.state.requests),
            allow_null_block=getattr(runner.runner.block_tables, "redirect_writes_to_null_block", False),
            step_id=self._step_id,
            role="target",
            layout=dict(self.layout),
            batch=self._batch,
            parallel=self._parallel,
            dynamic=dynamic,
            view_layouts=dict(self._view_layouts),
        )

    def _snapshot_audit(self, plan: AccessPlan) -> None:
        """Retain only this step's evidence, never a copy of the runtime."""
        assert self._pending is None
        plan, ready = self._snapshotter.freeze(plan)

        def complete():
            with torch.inference_mode():

                def gap(reason, **fields):
                    fields.setdefault("step_id", plan.step_id)
                    self.gap(reason, **fields)

                try:
                    if ready is not None:
                        ready.synchronize()
                    audit_access_plan(self.state, plan, gap, slot=self.slot)
                except Exception as exc:
                    gap(f"access_audit_failed:{type(exc).__name__}", detail=str(exc))
                self._check_failure()

        self._pending = _PendingAudit(complete)

    def _guard_output(self, output: Any) -> Any:
        self._check_failure()
        pending = self._pending
        report = functools.partial(self.report, step_id=self._step_id)
        get_output = getattr(output, "get_output", None)

        def complete(result):
            if pending is not None:
                pending.finish()
            self._check_failure()
            record_output(result, report)
            return result

        if get_output is None:
            return complete(output)

        @functools.wraps(get_output)
        def checked(*args, **kwargs):
            return complete(get_output(*args, **kwargs))

        try:
            output.get_output = checked
        except (AttributeError, TypeError):
            if pending is not None:
                pending.finish()
            self._check_failure()
        return output

    def wrap_sample(self, sample: Any):
        @functools.wraps(sample)
        def observed(*args, **kwargs):
            return self._guard_output(sample(*args, **kwargs))

        return observed

    @contextmanager
    def forward_scope(
        self, runner: Any, *, phase: str, metadata: Any = None, full_graph: bool = False, required: bool = True
    ):
        """Audit final per-group metadata at the native return boundary."""
        self._check_failure()
        self._phase, self._full_graph = phase, full_graph
        self._batch, self._parallel = {}, {}
        self.observe_metadata(metadata)
        try:
            yield
            if required or self._metadata:
                try:
                    if phase == "forward":
                        self._snapshot_audit(self._current_plan(runner))
                        if not self._defer_audit:
                            self._finish_pending()
                    else:
                        audit_access_plan(self.state, self._current_plan(runner), self.gap, slot=self.slot)
                except Exception as exc:
                    self.gap(f"access_audit_failed:{type(exc).__name__}")
            self._check_failure()
        finally:
            self._phase = None

    @contextmanager
    def execute_scope(self, output: Any, *, bind: bool = True):
        """Consume the original SchedulerOutput locally, without scheduler hooks."""
        self._check_failure()
        self._step_id += 1
        self._scheduled = self.state.bind(output) if bind else None
        if bind:
            record_plan(output.kv_connector_metadata, self.report)
        try:
            self._check_failure()
            yield
            self._check_failure()
        finally:
            self._scheduled = None

    def zero_blocks(self, block_ids: list[int]) -> None:
        self.report("kv_trace.boundary", boundary="model_runner.update_requests.return", zero_block_ids=block_ids)

    def copy_blocks(self, copies: Any) -> None:
        self.report(
            "kv_trace.boundary",
            boundary="model_runner.update_requests.return",
            block_copies=[copy._asdict() for copy in copies],
        )


def get_trace(vllm_config: Any) -> KVTraceRuntime | None:
    if not KVTraceConfig.from_vllm_config(vllm_config).enabled:
        return None
    guard = getattr(vllm_config, "_ascend_kv_trace", None)
    if guard is None:
        guard = KVTraceRuntime(vllm_config)
        vllm_config._ascend_kv_trace = guard
    return guard
