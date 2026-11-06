# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker L1 guard with one runner execution-plan collection point.

No KV snapshots, hashes, general-purpose trace or offline reader.
Adapted per-layer metadata supplies declared access plans. Operator hooks
preserve argument order; scheduler ranges independently identify each token.
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
class WriteAccess:
    layer: str
    slots: Any
    token_offset: int = 0


@dataclass(slots=True)
class ReadAccess:
    layer: str
    table: Any
    lengths: Any
    starts: Any = None
    request_offset: int = 0


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
        self._audit_layer = ""
        self._audit_evidence = ""
        self.layer_groups: dict[str, int] = {}
        self._accesses: list[WriteAccess | ReadAccess] = []
        self._metadata: dict[str, Any] = {}
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
            self.state.emit(event, layer=self._audit_layer, evidence=self._audit_evidence, **fields)

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
                if impl is not None and type(impl).__name__ in ("AscendAttentionBackendImpl", "AscendMLAImpl"):
                    impl._kv_guard = self
                else:
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

    def write(self, layer: str, slots: Any, token_offset: int = 0) -> None:
        if self._phase == "forward" and self._connector is not None and self._connector.sync:
            self._connector.check_ready(layer)
        if self._phase is not None and not self._full_graph:
            self._accesses.append(WriteAccess(layer, slots, token_offset))

    def read(self, layer: str, table: Any, lengths: Any, starts: Any = None, request_offset: int = 0) -> None:
        if self._phase == "forward" and self._connector is not None and self._connector.sync:
            self._connector.check_ready(layer)
        if self._phase is not None and not self._full_graph and table is not None:
            self._accesses.append(ReadAccess(layer, table, lengths, starts, request_offset))

    @staticmethod
    def _list(value: Any) -> list:
        if isinstance(value, (list, tuple)):
            return list(value)
        return value.detach().cpu().tolist()

    @staticmethod
    def _field(metadata: Any, *names: str) -> Any:
        return next((value for name in names if (value := getattr(metadata, name, None)) is not None), None)

    def collect_metadata(self, metadata: Any) -> None:
        """Keep references/coverage only; materialize the selected source later."""
        self._metadata, self._missing_groups = {}, set()
        if metadata is None:
            return  # Profiling and eager operator-only tests may have no attention plan.
        if not isinstance(metadata, dict):
            self.gap("microbatch_metadata_unadapted")
            self._missing_groups.update(self.layer_groups.values())
            return
        for layer, group in self.layer_groups.items():
            item = metadata.get(layer)
            if item is None:
                self.gap(f"metadata_write_mapping_missing:{layer}")
                self._missing_groups.add(group)
                continue
            self._metadata[layer] = item

    def _metadata_accesses(self, layer: str, item: Any) -> list[WriteAccess | ReadAccess]:
        slots = self._field(item, "slot_mapping")
        if slots is None:
            self.gap(f"metadata_write_mapping_missing:{layer}")
            self._missing_groups.add(self.layer_groups[layer])
            return []
        count = getattr(item, "num_actual_tokens", None)
        if count is not None:
            slots = slots[:count]
        accesses: list[WriteAccess | ReadAccess] = [WriteAccess(layer, slots)]
        table = self._field(item, "block_tables", "block_table", "block_table_tensor")
        lengths = self._field(item, "seq_lens_gpu", "seq_lens", "seq_lens_list", "seq_lens_cpu")
        if table is None or lengths is None:
            self.gap(f"metadata_read_range_missing:{layer}")
            self._missing_groups.add(self.layer_groups[layer])
        else:
            accesses.append(ReadAccess(layer, table, lengths))
        return accesses

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

    def _preview_append(self, runner: Any, expected: dict[int, ExpectedAccess]) -> Any:
        """Prepare a silent, unpublished canonical WRITE-then-READ transaction."""
        baseline, proposed, bindings = {}, {}, {}
        with self.state._lock:
            pool = self.state.pool_id
            for group_id, plan in expected.items():
                size = self.state.group_sizes[group_id]
                intervals: dict[int, list[tuple[int, int]]] = {}
                for row, request in enumerate(runner.input_batch.req_ids):
                    groups = self.state.requests[request]
                    bindings[request] = tuple(tuple(refs) for refs in groups)
                    refs = groups[group_id]
                    for block_id, epoch in refs[: len(plan.tables[row])]:
                        key = group_id, block_id
                        block = self.state.blocks.get(key)
                        if (
                            block_id == 0
                            or block is None
                            or epoch != block.epoch
                            or block.valid_end == INVALID_PREFIX
                            or request not in block.owners
                        ):
                            return None
                        baseline[key] = block.epoch, block.valid_end, frozenset(block.owners)
                    computed, count = self._scheduled[request]
                    for logical in range(computed // size, (computed + count - 1) // size + 1):
                        block_id, _ = refs[logical]
                        start = max(computed, logical * size) - logical * size
                        end = min(computed + count, (logical + 1) * size) - logical * size
                        intervals.setdefault(block_id, []).append((start, end))
                for block_id, spans in intervals.items():
                    key = group_id, block_id
                    cursor = baseline[key][1]
                    for start, end in sorted(spans):
                        if start != cursor:
                            return None
                        cursor = end
                    proposed[key] = cursor
                for row, length in enumerate(plan.lengths):
                    for logical, block_id in enumerate(plan.tables[row]):
                        key = group_id, block_id
                        if min(size, length - logical * size) > proposed.get(key, baseline[key][1]):
                            return None
        return pool, baseline, proposed, bindings

    def _try_fast_audit(self, runner: Any) -> bool:
        """Exact final metadata equality permits committing a precomputed state.

        shortcut: only canonical FULL plans; other plans retain the ordered audit.
        """
        if (
            self._phase != "forward"
            or not self._full_graph
            or self._scheduled is None
            or self._missing_groups
            or self._unready_load_groups
            or not self._metadata
            or self._metadata.keys() != self.layer_groups.keys()
        ):
            return False
        request_ids = list(runner.input_batch.req_ids)
        if not request_ids or len(set(request_ids)) != len(request_ids):
            return False
        expected: dict[int, ExpectedAccess] = {}
        inputs, checks, seen = {}, [], set()
        for layer, item in self._metadata.items():
            group_id = self.layer_groups[layer]
            if (group_id, id(item)) in seen:
                continue
            seen.add((group_id, id(item)))
            if group_id not in expected:
                plan = self._expected_write_plan(runner, group_id)
                if plan is None:
                    return False
                expected[group_id] = plan
            plan = expected[group_id]
            accesses = self._metadata_accesses(layer, item)
            if len(accesses) != 2 or self._missing_groups:
                return False
            write, read = accesses
            slots, table, lengths = write.slots, read.table, read.lengths
            if not all(
                isinstance(value, torch.Tensor) and value.dtype in (torch.int32, torch.int64)
                for value in (slots, table, lengths)
            ):
                return False
            rows, columns = len(request_ids), max(map(len, plan.tables), default=0)
            if (
                slots.ndim != 1
                or slots.numel() != len(plan.slots)
                or lengths.shape != (rows,)
                or table.ndim != 2
                or table.shape[0] < rows
                or table.shape[1] < columns
                or any(value.device.type not in ("cpu", "npu") for value in (slots, table, lengths))
            ):
                return False
            keys = []
            for value in (slots, lengths, table[:rows, :columns]):
                key = str(value.device), value.dtype, self._identity(value)
                inputs.setdefault(key, value)
                keys.append(key)
            checks.append((plan, keys))

        # Enqueue every copy before the CPU-only preview. No device values are
        # inspected until the event, at the original post-forward audit point.
        copied = {}
        event = None
        recorded = False
        try:
            for index, (key, value) in enumerate(inputs.items()):
                if value.device.type == "cpu":
                    copied[key] = self._list(value)
                    continue
                if self._copy_event is None:
                    self._copy_event = torch.npu.Event()
                event = self._copy_event
                if index >= len(self._host_buffers):
                    self._host_buffers.extend([None] * (index + 1 - len(self._host_buffers)))
                host = self._host_buffers[index]
                if host is None or host.shape != value.shape or host.dtype != value.dtype:
                    host = torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=True)
                    self._host_buffers[index] = host
                host.copy_(value, non_blocking=True)
                copied[key] = host
            if event is not None:
                event.record(torch.npu.current_stream())
                recorded = True
            candidate = self._preview_append(runner, expected)
        finally:
            if event is not None:
                # A partial enqueue failure must also drain outstanding DMA.
                if not recorded:
                    event.record(torch.npu.current_stream())
                event.synchronize()
        if candidate is None:
            return False
        values = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in copied.items()}
        for plan, (slots, lengths, table) in checks:
            if (
                values[slots] != plan.slots
                or values[lengths] != plan.lengths
                or any(row[: len(want)] != want for row, want in zip(values[table], plan.tables))
            ):
                return False
        pool, baseline, proposed, bindings = candidate
        with self.state._lock:
            unchanged = self.state.pool_id == pool and all(
                tuple(tuple(refs) for refs in self.state.requests.get(request, ())) == groups
                for request, groups in bindings.items()
            )
            for key, before in baseline.items():
                block = self.state.blocks.get(key)
                unchanged = (
                    unchanged
                    and block is not None
                    and (block.epoch, block.valid_end, frozenset(block.owners)) == before
                )
            if not unchanged:
                self.gap("allocation_changed_during_access_audit")
                return True
            for key, valid in proposed.items():
                self.state.blocks[key].valid_end = valid
        return True

    def _audit(self, runner: Any) -> None:
        if self._try_fast_audit(runner):
            return
        phase = self._phase
        assert phase is not None
        # FULL requires this replay's final metadata. Never reuse a captured
        # address plan when the current metadata is missing.
        if self._full_graph and not self._metadata:
            if phase != "warmup":
                self.gap("full_graph_metadata_missing")
            return
        observed: dict[str, list[WriteAccess | ReadAccess]] = {}
        for access in self._accesses:
            observed.setdefault(access.layer, []).append(access)
        if not self._metadata and not observed:
            if phase != "warmup":
                self.gap("execution_access_plan_missing")
            return

        copies: dict[Any, list] = {}
        results: dict[int, list[dict]] = {}
        audited: set[Any] = set()
        templates: dict[tuple[int, int], list[WriteAccess | ReadAccess]] = {}
        baseline: dict = {}
        shadow = self._scratch
        shadow.configure(self.state.group_sizes)
        pool_id = shadow.pool_id = self.state.pool_id
        expected: dict[int, ExpectedAccess | None] = {}
        for layer in dict.fromkeys((*self._metadata, *observed)):
            group_id = self.layer_groups.get(layer)
            if group_id is None:
                self.gap(f"unbound_access_layer:{layer}")
                continue
            if group_id in self._unready_load_groups:
                continue
            actual = observed.get(layer, [])
            if observed and not actual and not self._full_graph:
                self.gap(f"operator_layer_not_observed:{layer}")
                self._missing_groups.add(group_id)
            if actual and not self._full_graph:
                accesses = actual
                evidence = "operator_arguments"
            elif layer in self._metadata:
                item = self._metadata[layer]
                template_key = id(item), group_id
                if template_key not in templates:
                    templates[template_key] = self._metadata_accesses(layer, item)
                accesses = templates[template_key]
                evidence = "execution_plan"
            else:
                accesses = actual
                evidence = "operator_arguments"
            signature = (
                group_id,
                evidence,
                tuple(
                    ("write", self._identity(a.slots), a.token_offset)
                    if isinstance(a, WriteAccess)
                    else (
                        "read",
                        self._identity(a.table),
                        self._identity(a.lengths),
                        self._identity(a.starts),
                        a.request_offset,
                    )
                    for a in accesses
                ),
            )
            if signature in audited:
                continue

            def values(value):
                key = self._identity(value)
                if key not in copies:
                    copies[key] = self._list(value)
                return copies[key]

            groups: list[tuple[dict[str, Any], int]] = []
            write_ranges: list[tuple[int, int]] = []
            keys: set[tuple[int, int]] = set()
            write_keys: set[tuple[int, int]] = set()
            size = self.state.group_sizes[group_id]
            if group_id not in expected:
                expected[group_id] = self._expected_write_plan(runner, group_id)
            plan = expected[group_id]
            for access in accesses:
                entry: dict[str, Any] = {
                    "group_id": group_id,
                    "slots": [],
                    "block_table": [],
                    "seq_lens": [],
                }
                if isinstance(access, WriteAccess):
                    slots = values(access.slots)
                    entry["slots"] = [slot for row in slots for slot in (row if isinstance(row, list) else [row])]
                    if plan is not None:
                        entry["expected_slots"], entry["query_ends"] = plan.slots, plan.query_ends
                        entry["token_offset"] = access.token_offset
                        write_ranges.append((access.token_offset, access.token_offset + len(entry["slots"])))
                    write_keys.update((group_id, slot // size) for slot in entry["slots"] if slot >= 0)
                    keys.update(write_keys)
                    offset = 0
                else:
                    if self._phase in ("dummy", "warmup"):
                        # Dummy outputs are discarded. Their writes can corrupt
                        # live KV, but reading reserved scratch is expected.
                        continue
                    lengths = values(access.lengths)
                    if access.starts is not None:
                        lengths = [a + b for a, b in zip(lengths, values(access.starts))]
                    entry["seq_lens"] = lengths
                    needed = max(((length + size - 1) // size for length in lengths), default=0)
                    if isinstance(access.table, (list, tuple)):
                        entry["block_table"] = [row[:needed] for row in access.table[: len(lengths)]]
                    else:
                        entry["block_table"] = values(access.table[: len(lengths), :needed])
                    keys.update((group_id, block) for row in entry["block_table"] for block in row if block >= 0)
                    offset = access.request_offset
                    if plan is not None:
                        entry["expected_block_table"] = plan.tables[offset:]
                        entry["expected_seq_lens"] = plan.lengths[offset:]
                groups.append((entry, offset))

            if plan is not None:
                covered = 0
                for start, end in sorted(write_ranges):
                    if start > covered:
                        break
                    covered = max(covered, end)
                if covered < len(plan.slots):
                    self.gap("write_tokens_not_observed", layer=layer, first_missing_token=covered)
                    self._missing_groups.add(group_id)

            with self.state._lock:
                # Owners are immutable within the audit. Copy them once per
                # touched block; read-only state can also be shared by layers.
                for key in keys:
                    if key not in baseline and key in self.state.blocks:
                        initial = self.state.blocks[key]
                        baseline[key] = replace(initial, owners=set(initial.owners))
                shadow.blocks = {
                    key: replace(baseline[key]) if key in write_keys else baseline[key]
                    for key in keys
                    if key in baseline
                }
            self._audit_layer, self._audit_evidence = layer, evidence
            request_ids = list(runner.input_batch.req_ids) if self._phase == "forward" else []
            for entry, offset in groups:
                shadow.observe([entry], request_ids[offset:], phase)
            audited.add(signature)
            results.setdefault(group_id, []).append(shadow.blocks)

        # Each layer starts from the same pre-forward validity. Another layer's
        # write cannot legalize an earlier read or hide a missing write. Merge
        # conservatively; no validity is advanced on failed/partial collection.
        with self.state._lock:
            for group_id, versions_by_layer in results.items():
                touched = {key for versions in versions_by_layer for key in versions}
                for key in touched:
                    block = self.state.blocks.get(key)
                    if block is None:
                        continue
                    versions = [versions.get(key, baseline.get(key, block)) for versions in versions_by_layer]
                    if self.state.pool_id != pool_id or any(b.epoch != block.epoch for b in versions):
                        self.gap("allocation_changed_during_access_audit")
                        continue
                    if any(b.valid_end == INVALID_PREFIX for b in versions):
                        block.valid_end = INVALID_PREFIX
                    elif (
                        block.valid_end != INVALID_PREFIX
                        and group_id not in self._missing_groups
                        and self._phase == "forward"
                    ):
                        valid = min(b.valid_end for b in versions)
                        if valid > block.valid_end:
                            block.valid_end = valid

    def observe_metadata(self, metadata: Any) -> None:
        try:
            self.collect_metadata(metadata)
        except Exception as exc:
            self._metadata = {}
            self._missing_groups = set(self.layer_groups.values())
            self.gap(f"metadata_collection_failed:{type(exc).__name__}")

    def _finish_pending(self) -> None:
        if self._pending is not None:
            self._pending.finish()
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
        job.report, job.gap, job._check_failure = self.report, self.gap, self._check_failure
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
            job._accesses = [
                WriteAccess(access.layer, freeze(access.slots), access.token_offset)
                if isinstance(access, WriteAccess)
                else ReadAccess(
                    access.layer,
                    freeze(access.table),
                    freeze(access.lengths),
                    freeze(access.starts),
                    access.request_offset,
                )
                for access in self._accesses
            ]
            observed = {access.layer for access in self._accesses}
            items, job._metadata = {}, {}
            for layer, item in self._metadata.items():
                if not self._full_graph and layer in observed:
                    job._metadata[layer] = None  # Only coverage is used when operator arguments were observed.
                    continue
                key = id(item)
                if key not in items:
                    slots = self._field(item, "slot_mapping")
                    count = getattr(item, "num_actual_tokens", None)
                    if slots is not None and count is not None:
                        slots = slots[:count]
                    items[key] = SimpleNamespace(
                        slot_mapping=freeze(slots),
                        block_tables=freeze(self._field(item, "block_tables", "block_table", "block_table_tensor")),
                        seq_lens=freeze(self._field(item, "seq_lens_gpu", "seq_lens", "seq_lens_list", "seq_lens_cpu")),
                    )
                job._metadata[layer] = items[key]
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
        self._accesses = []
        self.observe_metadata(metadata)
        try:
            yield
            if required or self._metadata or self._accesses:
                try:
                    if phase == "forward" and self._connector is not None:
                        self._connector.before_audit()
                    if phase == "forward":
                        for meta in self._recomputes:
                            self.state.recompute(meta)
                        self._recomputes = []
                    if self._defer_audit and phase == "forward" and self._connector is None:
                        self._snapshot_audit(runner)
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
            context._kv_guard = self
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
