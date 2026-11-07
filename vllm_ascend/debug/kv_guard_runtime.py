# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker L1 guard with one runner execution-plan collection point.

No KV snapshots, hashes, general-purpose trace or offline reader.
Runner metadata supplies one declared access plan per KV cache group.
Scheduler ranges independently identify each token; attention is not instrumented.
Only small metadata is copied, after the forward and graph parameter updates;
the checker neither inspects KV values nor proves kernel memory accesses.
"""

from __future__ import annotations

import functools
from collections import OrderedDict
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, replace
from threading import Lock
from types import SimpleNamespace
from typing import Any, cast

import torch

from vllm_ascend.debug.kv_guard import INVALID_PREFIX, KVGuard
from vllm_ascend.debug.kv_guard_config import KVGuardConfig
from vllm_ascend.debug.kv_guard_connector import ConnectorGuard

_MAX_REPORT_KEYS = 1024


@dataclass(slots=True)
class ExpectedAccess:
    slots: list[int]
    query_ends: list[int]
    tables: list[list[int]]
    lengths: list[int]


class KVGuardError(RuntimeError):
    """Stop returning model results after a violation or incomplete audit."""


class _PendingAudit:
    """One snapshot, consumed once by the output reader or next forward."""

    def __init__(self, complete: Any):
        self._complete = complete
        self._lock = Lock()
        self._done = False
        self._error: BaseException | None = None

    def finish(self) -> None:
        with self._lock:
            if not self._done:
                try:
                    self._complete()
                except BaseException as exc:
                    self._error = exc
                finally:
                    self._complete = None
                    self._done = True
            if self._error is not None:
                raise self._error


class GuardRuntime:
    def __init__(self, vllm_config: Any):
        from vllm.logger import logger

        self.logger = logger
        self.config = vllm_config
        self.state = KVGuard(self.report)
        self._scratch = KVGuard(self._emit_scratch)
        self.layer_groups: dict[str, int] = {}
        self._metadata: dict[int, Any] = {}
        self._missing_groups: set[int] = set()
        self._phase: str | None = None
        self._full_graph = False
        self._reported: OrderedDict[tuple, None] = OrderedDict()
        self._scheduled: dict[str, tuple[int | None, int]] | None = None
        self._failure: str | None = None
        self.action = KVGuardConfig.from_vllm_config(vllm_config).action
        self._runner_adapter: _V2RunnerAdapter | None = None
        self._host_buffers: list[Any] = []
        self._copy_event: Any = None
        self._connector: ConnectorGuard | None = None
        self._recomputes: list[dict] = []
        self._unready_load_groups: set[int] = set()
        self._defer_audit = False
        self._pending: _PendingAudit | None = None
        self._deferred_state: list[tuple[Any, tuple]] = []

    def _check_failure(self) -> None:
        if self.action == "raise" and self._failure is not None:
            raise KVGuardError(f"KV guard stopped model output: {self._failure}; restart the worker before reuse")

    def _emit_scratch(self, event: str, **fields: Any) -> None:
        if event == "guard.violation":
            self.state.emit(event, evidence="execution_plan", **fields)

    def report(self, event: str, **fields: Any) -> None:
        failed = event in ("guard.violation", "guard.coverage_gap")
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
                            "pool_id",
                            "group_id",
                            "block_id",
                            "alloc_epoch",
                            "layer",
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
        payload = " ".join(f"{key}={value}" for key, value in fields.items())
        log = self.logger.error if failed else self.logger.info
        log("[KV_GUARD] %s %s", event, payload)

    def gap(self, reason: str, **fields: Any) -> None:
        self.report("guard.coverage_gap", reason=reason, **fields)

    def configure(self, runner: Any) -> None:
        config = self.config
        parallel = config.parallel_config
        transfer = config.kv_transfer_config
        unsupported = any(
            (
                getattr(config.model_config, "is_encoder_decoder", False),
                config.speculative_config is not None,
                getattr(parallel, "prefill_context_parallel_size", 1) != 1,
                getattr(parallel, "decode_context_parallel_size", 1) != 1,
                getattr(parallel, "enable_dbo", False),
                transfer is not None and transfer.kv_connector not in ("MooncakeConnector", "MooncakeConnectorV1"),
            )
        )
        sizes = {}
        tables = getattr(getattr(getattr(runner, "input_batch", None), "block_table", None), "block_tables", ())
        for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            sizes[group_id] = spec.block_size
            if tables and tables[group_id].block_size != spec.block_size:
                unsupported = True
            for name in group.layer_names:
                self.layer_groups[name] = group_id
                layer_spec = getattr(spec, "kv_cache_specs", {}).get(name, spec)
                append = (
                    hasattr(layer_spec, "head_size")
                    and getattr(layer_spec, "sliding_window", None) is None
                    and getattr(layer_spec, "compress_ratio", 1) == 1
                    and type(layer_spec).__name__ != "CircularBufferSpec"
                )
                if not append:
                    raise ValueError(f"KV L1 guard: unadapted cache validity rules: {type(layer_spec).__name__}")
                layer = config.compilation_config.static_forward_context.get(name)
                impl = getattr(layer, "impl", None)
                if impl is None or type(impl).__name__ not in ("AscendAttentionBackendImpl", "AscendMLAImpl"):
                    raise ValueError(f"KV L1 guard: unadapted attention implementation: {type(impl).__name__}")
        if unsupported:
            raise ValueError(
                "KV L1 guard: unadapted execution mode or block coordinates "
                "(CP, DBO, speculation, encoder/decoder, connector)"
            )
        self.state.configure(sizes)
        self._defer_audit = (
            bool(getattr(getattr(config, "scheduler_config", None), "async_scheduling", False))
            and transfer is None
            # Pooling bypasses sample_tokens; non-reply workers may discard outputs.
            and getattr(runner, "is_pooling_model", None) is False
            and getattr(parallel, "tensor_parallel_size", None) == 1
            and getattr(parallel, "pipeline_parallel_size", None) == 1
            and hasattr(runner, "sample_tokens")
        )
        if self._runner_adapter is not None:
            self._runner_adapter.configure()
        self.report("guard.enabled", failure_action=self.action)

    @staticmethod
    def _list(value: Any) -> list:
        if isinstance(value, (list, tuple)):
            return list(value)
        return value.detach().cpu().tolist()

    @staticmethod
    def _field(metadata: Any, *names: str) -> Any:
        return next((value for name in names if (value := getattr(metadata, name, None)) is not None), None)

    def collect_metadata(self, metadata: Any) -> None:
        self._metadata = {}
        self._missing_groups = set()
        if metadata is None:
            return
        if not isinstance(metadata, dict):
            self.gap("microbatch_metadata_unadapted")
            self._missing_groups.update(self.state.group_sizes)
            return
        for name, item in metadata.items():
            group = self.layer_groups.get(name)
            if group is not None and item is not None:
                self._metadata.setdefault(group, item)
        for group in self.state.group_sizes.keys() - self._metadata.keys():
            self.gap("metadata_write_mapping_missing", group_id=group)
            self._missing_groups.add(group)

    @staticmethod
    def _identity(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return tuple(GuardRuntime._identity(item) for item in value)
        if isinstance(value, (int, float, str)):
            return value
        return value.data_ptr(), tuple(value.shape), tuple(value.stride())

    def _expected_write_plan(self, runner: Any, group_id: int) -> ExpectedAccess | None:
        """Derive destinations from the scheduler, never from the audited mapping.

        Per-token lists exist only for this forward and are shared by all layers
        in a cache group. No persistent token-ID history or KV tensor is copied.
        """
        if self._phase != "forward" or self._scheduled is None:
            return None
        size = self.state.group_sizes[group_id]
        slots: list[int] = []
        query_ends: list[int] = []
        tables: list[list[int]] = []
        lengths: list[int] = []
        for request_id in runner.input_batch.req_ids:
            computed, count = self._scheduled.get(request_id, (None, 0))
            if computed is None or computed < 0 or count <= 0:
                self.gap("scheduler_write_range_missing", request_id=request_id)
                self._missing_groups.add(group_id)
                return None
            groups = self.state.requests.get(request_id, ())
            refs = groups[group_id] if group_id < len(groups) else ()
            needed = (computed + count + size - 1) // size
            if len(refs) < needed:
                self.gap("scheduler_read_blocks_missing", request_id=request_id, group_id=group_id)
                self._missing_groups.add(group_id)
                return None
            tables.append([block for block, _ in refs[:needed]])
            lengths.append(computed + count)
            for logical in range(computed // size, (computed + count - 1) // size + 1):
                block_id, _ = refs[logical]
                start = max(computed, logical * size) - logical * size
                end = min(computed + count, (logical + 1) * size) - logical * size
                slots.extend(range(block_id * size + start, block_id * size + end))
            query_ends.append(len(slots))
        return ExpectedAccess(slots, query_ends, tables, lengths)

    def _audit(self, runner: Any) -> None:
        phase = self._phase
        assert phase is not None
        if not self._metadata:
            if phase != "warmup":
                self.gap("full_graph_metadata_missing" if self._full_graph else "execution_access_plan_missing")
            return
        request_ids = list(runner.input_batch.req_ids) if phase == "forward" else []
        shadow = self._scratch
        shadow.configure(self.state.group_sizes)
        shadow.emit = self._emit_scratch
        for group, item in self._metadata.items():
            if group in self._unready_load_groups:
                continue
            slots = self._field(item, "slot_mapping")
            table = self._field(item, "block_tables", "block_table", "block_table_tensor")
            lengths = self._field(item, "seq_lens_gpu", "seq_lens", "seq_lens_list", "seq_lens_cpu")
            if slots is None:
                self.gap("metadata_write_mapping_missing", group_id=group)
                self._missing_groups.add(group)
                slots = []
            else:
                count = getattr(item, "num_actual_tokens", None)
                slots = self._list(slots if count is None else slots[:count])
                slots = [slot for row in slots for slot in (row if isinstance(row, list) else [row])]
            size = self.state.group_sizes[group]
            if phase != "forward":
                lengths, table = [], []  # Discarded scratch reads do not establish cache validity.
            elif table is None or lengths is None:
                self.gap("metadata_read_range_missing", group_id=group)
                self._missing_groups.add(group)
                lengths, table = [], []
            else:
                lengths = self._list(lengths)
                columns = max(((length + size - 1) // size for length in lengths), default=0)
                table = (
                    [row[:columns] for row in table[: len(lengths)]]
                    if isinstance(table, (list, tuple))
                    else self._list(table[: len(lengths), :columns])
                )
            entry = {"group_id": group, "slots": slots, "block_table": table, "seq_lens": lengths}
            writes = {(group, slot // size) for slot in slots if slot >= 0}
            keys = writes | {(group, block) for row in table for block in row if block >= 0}
            # Metadata is already on the host. Keep state inspection and commit atomic,
            # sharing read-only blocks and cloning only blocks this step may write.
            with self.state._lock:
                plan = self._expected_write_plan(runner, group)
                if plan is not None:
                    entry.update(
                        expected_slots=plan.slots,
                        query_ends=plan.query_ends,
                        expected_block_table=plan.tables,
                        expected_seq_lens=plan.lengths,
                    )
                    if len(slots) < len(plan.slots):
                        self.gap("write_tokens_not_observed", group_id=group, first_missing_token=len(slots))
                        self._missing_groups.add(group)
                pool = shadow.pool_id = self.state.pool_id
                before = {
                    key: (block, block.epoch, block.valid_end, frozenset(block.owners))
                    for key in writes
                    if (block := self.state.blocks.get(key)) is not None
                }
                shadow.blocks = {
                    key: replace(block) if key in writes else block
                    for key in keys
                    if (block := self.state.blocks.get(key)) is not None
                }
                shadow.observe([entry], request_ids, phase)
                if self.state.pool_id != pool or any(
                    self.state.blocks.get(key) is not block
                    or (block.epoch, block.valid_end, frozenset(block.owners)) != (epoch, valid, owners)
                    for key, (block, epoch, valid, owners) in before.items()
                ):
                    self.gap("allocation_changed_during_access_audit")
                    continue
                for key in before:
                    block, updated = self.state.blocks[key], shadow.blocks[key]
                    if updated.valid_end == INVALID_PREFIX:
                        block.valid_end = INVALID_PREFIX
                    elif phase == "forward" and group not in self._missing_groups and block.valid_end != INVALID_PREFIX:
                        block.valid_end = max(block.valid_end, updated.valid_end)

    def observe_metadata(self, metadata: Any) -> None:
        try:
            self.collect_metadata(metadata)
        except Exception as exc:
            self._metadata = {}
            self._missing_groups = set(self.layer_groups.values())
            self.gap(f"metadata_collection_failed:{type(exc).__name__}")

    def _finish_pending(self) -> None:
        if self._pending is not None:
            try:
                self._pending.finish()
            finally:
                self._pending = None
        operations, self._deferred_state = self._deferred_state, []
        for operation, args in operations:
            operation(*args)
        self._check_failure()

    def _snapshot_audit(self, runner: Any) -> None:
        """Freeze final evidence now; wait only at an output/forward boundary."""
        assert self._pending is None
        job = copy(self)
        job._scheduled = None if self._scheduled is None else dict(self._scheduled)
        job._missing_groups = set(self._missing_groups)
        job._unready_load_groups = set(self._unready_load_groups)
        reporting = cast(Any, job)
        reporting.report, reporting.gap, reporting._check_failure = self.report, self.gap, self._check_failure
        job._scratch = KVGuard(job._emit_scratch)
        frozen_runner = SimpleNamespace(input_batch=SimpleNamespace(req_ids=list(runner.input_batch.req_ids)))
        copies: dict[Any, Any] = {}
        event = None
        device_copies = 0

        def freeze(value):
            nonlocal event, device_copies
            if isinstance(value, torch.Tensor):
                key = value.device, value.dtype, self._identity(value)
                if key not in copies:
                    if value.device.type == "cpu":
                        host = value.clone()
                    else:
                        if self._copy_event is None:
                            self._copy_event = torch.npu.Event()
                        event = self._copy_event
                        index = device_copies
                        device_copies += 1
                        if index >= len(self._host_buffers):
                            self._host_buffers.append(None)
                        host = self._host_buffers[index]
                        if host is None or host.shape != value.shape or host.dtype != value.dtype:
                            host = torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=True)
                            self._host_buffers[index] = host
                        host.copy_(value, non_blocking=True)
                    copies[key] = host
                return copies[key]
            if isinstance(value, (list, tuple)):
                return [freeze(item) for item in value]
            return value

        try:
            job._metadata = {}
            for group, item in self._metadata.items():
                slots = self._field(item, "slot_mapping")
                count = getattr(item, "num_actual_tokens", None)
                job._metadata[group] = SimpleNamespace(
                    slot_mapping=freeze(slots if slots is None or count is None else slots[:count]),
                    block_tables=freeze(self._field(item, "block_tables", "block_table", "block_table_tensor")),
                    seq_lens=freeze(self._field(item, "seq_lens_gpu", "seq_lens", "seq_lens_list", "seq_lens_cpu")),
                )
            if event is not None:
                event.record(torch.npu.current_stream())
        except Exception:
            if event is not None:
                # Even a partially failed snapshot must drain before buffer reuse.
                event.record(torch.npu.current_stream())
                event.synchronize()
            raise

        def complete():
            with torch.inference_mode():
                try:
                    if event is not None:
                        event.synchronize()
                    job._audit(frozen_runner)
                except Exception as exc:
                    self.gap(f"access_audit_failed:{type(exc).__name__}")
                self._check_failure()

        self._pending = _PendingAudit(complete)

    def _guard_output(self, output: Any) -> Any:
        self._check_failure()
        pending = self._pending
        if pending is None:
            return output
        get_output = getattr(output, "get_output", None)
        if get_output is None:
            pending.finish()
            self._check_failure()
            return output

        @functools.wraps(get_output)
        def checked():
            pending.finish()
            self._check_failure()
            return get_output()

        try:
            output.get_output = checked
        except (AttributeError, TypeError):
            pending.finish()
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
        """Audit one completed forward; never advance validity on exceptions."""
        self._check_failure()
        self._phase, self._full_graph = phase, full_graph
        self.observe_metadata(metadata)
        try:
            yield
            if required or self._metadata:
                try:
                    if phase == "forward" and self._connector is not None:
                        self._connector.before_audit()
                    if phase == "forward":
                        for meta in self._recomputes:
                            self.state.recompute(meta)
                        self._recomputes = []
                    if phase == "forward":
                        self._snapshot_audit(runner)
                        if not self._defer_audit:
                            self._finish_pending()
                    else:
                        self._audit(runner)
                except Exception as exc:
                    self.gap(f"access_audit_failed:{type(exc).__name__}")
            self._check_failure()
        finally:
            self._phase = None
            if self._connector is not None:
                self._connector.commit_completions()
                if self._connector.post_forward_complete:
                    self._connector.finish_step()

    def wrap_forward(self, runner: Any, forward: Any, get_context: Any):
        @functools.wraps(forward)
        def observed(num_tokens, *args, **kwargs):
            self._finish_pending()
            context = get_context()
            with self.forward_scope(
                runner,
                phase=kwargs.pop("_kv_guard_phase", "forward"),
                metadata=getattr(context, "attn_metadata", None),
                full_graph="FULL" in str(context.cudagraph_runtime_mode),
            ):
                return forward(num_tokens, *args, **kwargs)

        return observed

    def wrap_update(self, runner: Any, update: Any):
        @functools.wraps(update)
        def observed(output, *args, **kwargs):
            result = update(output, *args, **kwargs)
            binding = getattr(output, "_ascend_kv_guard_context", None)
            if binding is None:
                self.gap("scheduler_binding_missing")
                self._check_failure()
                return result
            for request_id, groups in binding["requests"].items():
                if request_id not in binding.get("scheduled_requests", {}):
                    continue  # Receive-only bindings are not runner batch entries.
                actual = runner.requests.get(request_id)
                expected = [[block for block, _ in refs] for refs in groups]
                if actual is None or [list(g) for g in actual.block_ids] != expected:
                    self.report(
                        "guard.violation",
                        operation="bind",
                        reason="worker_block_mapping_mismatch",
                        request_id=request_id,
                    )
            self._check_failure()
            return result

        return observed

    @contextmanager
    def execute_scope(self, output: Any, *, bind: bool = True, finish_connector: bool = True):
        """Share scheduler binding and cleanup across both runner versions."""
        self._check_failure()
        self._scheduled = None
        self._unready_load_groups.clear()
        self._recomputes = []
        if bind:
            binding = getattr(output, "_ascend_kv_guard_context", None)
            if binding is not None:
                # Receive-only and idle DP steps can return before _update_states.
                uninitialized = set(getattr(output, "new_block_ids_to_zero", None) or ())
                uninitialized.update(
                    copy.dst_block_id for copy in (getattr(output, "kv_cache_block_copies", None) or ())
                )
                context = {**binding, "uninitialized_block_ids": uninitialized}
                if self._pending is not None:
                    self._deferred_state.append((self.state.bind, (context,)))
                else:
                    self.state.bind(context)
                self._recomputes = binding.get("recompute_ranges", [])
                if self._connector is None and getattr(self.config, "kv_transfer_config", None) is not None:
                    from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

                    if has_kv_transfer_group():
                        self._connector = ConnectorGuard(self, get_kv_transfer_group())
                loads = binding.get("loads", [])
                if self._connector is not None:
                    self._connector.stage(loads)
                elif loads:
                    self.gap("load_connector_missing")
                self._scheduled = binding.get("scheduled_requests")
                if self._scheduled is None:
                    self.gap("scheduler_write_plan_missing")
            else:
                self.gap("scheduler_binding_missing")
        try:
            self._check_failure()
            yield
            self._check_failure()
        finally:
            if self._deferred_state:
                self._finish_pending()  # Receive-only, idle and failed steps may never enter a forward.
            if self._connector is not None and finish_connector:
                self._connector.finish_step()
            self._scheduled = None
            self._recomputes = []
        self._check_failure()

    def wrap_execute(self, execute: Any):
        @functools.wraps(execute)
        def observed(output, *args, **kwargs):
            with self.execute_scope(output):
                result = execute(output, *args, **kwargs)
                return self._guard_output(result) if result is not None else result

        return observed

    def zero_blocks(self, block_ids: list[int]) -> None:
        if self._pending is not None:
            self._deferred_state.append((self.zero_blocks, (list(block_ids),)))
            return
        with self.state._lock:
            for group_id in self.state.group_sizes:
                for block_id in block_ids:
                    if block_id == 0:
                        self.state.clear(group_id, block_id)
                        continue
                    block = self.state.blocks.get((group_id, block_id))
                    # A fresh allocation is already UNKNOWN. Clearing it must
                    # not turn its first legitimate write into an invalid write.
                    if block is not None and block.valid_end != 0:
                        if block.valid_end > 0 and block.owners:
                            self.state._violation((group_id, block_id), block, "clear", "clear_live_block")
                        self.state.clear(group_id, block_id)
        self._check_failure()

    def copy_blocks(self, copies: Any) -> None:
        if self._pending is not None:
            self._deferred_state.append((self.copy_blocks, (list(copies),)))
            return
        with self.state._lock:
            for copy in copies:
                if copy.dst_block_id == 0:
                    for group_id in self.state.group_sizes:
                        self.state._violation((group_id, 0), None, "copy", "write_null_block")
                elif not any((group_id, copy.dst_block_id) in self.state.blocks for group_id in self.state.group_sizes):
                    self.gap("copy_destination_unbound", block_id=copy.dst_block_id)
            for group_id in self.state.group_sizes:
                pairs = [
                    (copy.src_block_id, copy.dst_block_id)
                    for copy in copies
                    if (group_id, copy.dst_block_id) in self.state.blocks
                ]
                self.state.copy_blocks(group_id, pairs)
        self._check_failure()


def get_guard(vllm_config: Any) -> GuardRuntime | None:
    if not KVGuardConfig.from_vllm_config(vllm_config).enabled:
        return None
    guard = getattr(vllm_config, "_ascend_kv_guard", None)
    if guard is None:
        guard = GuardRuntime(vllm_config)
        vllm_config._ascend_kv_guard = guard
    return guard


class _V2RunnerAdapter:
    """Only V2 batch ordering, metadata timing and capture need adaptation."""

    def __init__(self, runner: Any, runtime: GuardRuntime):
        self.runner = runner
        self.runtime = runtime
        self.input_batch = SimpleNamespace(req_ids=[])
        self._prepared = False

    def configure(self) -> None:
        tables = self.runner.block_tables
        sizes = [group.kv_cache_spec.block_size for group in self.runner.kv_cache_config.kv_cache_groups]
        if list(tables.block_sizes) != sizes or list(tables.kernel_block_sizes) != sizes:
            raise ValueError("KV guard: unadapted MRV2 allocator/kernel block coordinates")
        prepare = self.runner.model_state.prepare_attn
        if getattr(prepare, "_ascend_kv_guard", False):
            return

        @functools.wraps(prepare)
        def observed(input_batch, cudagraph_mode, *args, **kwargs):
            metadata = prepare(input_batch, cudagraph_mode, *args, **kwargs)
            if self.runtime._phase is not None:
                self.runtime._finish_pending()
                self.input_batch = SimpleNamespace(req_ids=list(input_batch.req_ids[: input_batch.num_reqs]))
                self._prepared = True
                self.runtime._full_graph = "FULL" in str(cudagraph_mode)
                self.runtime.observe_metadata(metadata)
                scheduled = self.runtime._scheduled
                if self.runtime._phase == "forward" and scheduled is not None:
                    ids = self.input_batch.req_ids
                    if len(ids) != len(set(ids)) or set(ids) != set(scheduled):
                        self.runtime.gap("v2_batch_request_mismatch")
                        self.runtime._missing_groups.update(self.runtime.layer_groups.values())
                self.runtime._check_failure()
            return metadata

        cast(Any, observed)._ascend_kv_guard = True
        self.runner.model_state.prepare_attn = observed

    def wrap_execute(self, execute: Any):
        @functools.wraps(execute)
        def observed(scheduler_output, *args, **kwargs):
            # Forward positional and keyword arguments unchanged across vLLM lanes.
            flags = dict(zip(("intermediate_tensors", "dummy_run", "skip_attn_for_dummy_run", "is_profile"), args))
            flags.update(kwargs)
            dummy = bool(flags.get("dummy_run", False))
            profile = bool(flags.get("is_profile", False))
            phase = "warmup" if profile else "dummy" if dummy else "forward"
            required = bool(scheduler_output.total_num_scheduled_tokens) and not (
                profile and flags.get("skip_attn_for_dummy_run")
            )
            if dummy or profile:
                self.runtime._finish_pending()
            self.input_batch = SimpleNamespace(req_ids=[])
            self._prepared = False

            # Synthetic dummy/profile outputs have no scheduler binding, but
            # their writes must still be checked against the shared live KV state.
            with (
                self.runtime.execute_scope(scheduler_output, bind=not (dummy or profile), finish_connector=False),
                self.runtime.forward_scope(self, phase=phase, required=required),
            ):
                result = execute(scheduler_output, *args, **kwargs)
                if phase == "forward" and required and not self._prepared:
                    self.runtime.gap("v2_attention_metadata_missing")
                    self.runtime._missing_groups.update(self.runtime.layer_groups.values())
            return self.runtime._guard_output(result) if result is not None else result

        return observed

    def wrap_update(self, update: Any):
        @functools.wraps(update)
        def observed(scheduler_output, *args, **kwargs):
            result = update(scheduler_output, *args, **kwargs)
            # The parent has completed the physical operations at this point.
            zeroed = getattr(scheduler_output, "new_block_ids_to_zero", None)
            if zeroed:
                self.runtime.zero_blocks(zeroed)
            copies = getattr(scheduler_output, "kv_cache_block_copies", None)
            if copies:
                self.runtime.copy_blocks(copies)
            return result

        return observed

    def wrap_capture(self, capture: Any):
        @functools.wraps(capture)
        def observed(*args, **kwargs):
            self.runtime._check_failure()
            with self.runtime.state._lock:
                if (
                    self.runtime.state.requests
                    or self.runtime.state.loads
                    or any(block.valid_end != 0 for block in self.runtime.state.blocks.values())
                ):
                    raise ValueError(
                        "KV guard: capture requires empty KV state; live/cache-preserving recapture is unsupported"
                    )
            # Startup capture has no request KV to protect. Do not synchronize
            # metadata to the CPU inside a device graph capture. Real replays
            # are audited by wrap_execute using their current metadata.
            return capture(*args, **kwargs)

        return observed


def install_guard(runner: Any) -> GuardRuntime | None:
    """Install one per-engine guard with only the runner-specific hooks."""
    runtime = get_guard(runner.vllm_config)
    if runtime is None:
        return None
    if getattr(runner.vllm_config, "use_v2_model_runner", False):
        adapter = _V2RunnerAdapter(runner, runtime)
        runtime._runner_adapter = adapter
        cast(Any, runner).execute_model = adapter.wrap_execute(runner.execute_model)
        cast(Any, runner).update_requests = adapter.wrap_update(runner.update_requests)
        cast(Any, runner).capture_model = adapter.wrap_capture(runner.capture_model)
    else:
        from vllm.forward_context import get_forward_context

        cast(Any, runner)._model_forward = runtime.wrap_forward(runner, runner._model_forward, get_forward_context)
        cast(Any, runner).execute_model = runtime.wrap_execute(runner.execute_model)
        cast(Any, runner)._update_states = runtime.wrap_update(runner, runner._update_states)
    if hasattr(runner, "sample_tokens"):
        cast(Any, runner).sample_tokens = runtime.wrap_sample(runner.sample_tokens)
    return runtime
