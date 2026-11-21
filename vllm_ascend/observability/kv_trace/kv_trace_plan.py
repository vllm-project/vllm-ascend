# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-step address evidence, independent of subsequent binding updates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from itertools import accumulate
from types import SimpleNamespace
from typing import Any, cast

import torch
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.kv_cache_interface import CircularBufferSpec, iter_layer_specs

from vllm_ascend.observability.kv_trace.kv_trace_state import BlockTable, KVCacheState


@dataclass(slots=True)
class AccessPlan:
    phase: str
    request_ids: list[str]
    scheduled: dict[str, tuple[int | None, int]] | None
    metadata: dict[Any, Any]
    full_graph: bool
    bindings: dict[str, BlockTable]
    allow_null_block: bool = False
    step_id: int = 0
    role: str = "target"
    layout: dict[int, Any] = field(default_factory=dict)
    batch: dict[str, Any] = field(default_factory=dict)
    parallel: dict[str, Any] = field(default_factory=dict)
    dynamic: bool = False


@dataclass(slots=True)
class ExpectedAccess:
    slots: list[int]
    tables: list[list[int]]
    lengths: list[int]


@dataclass(slots=True)
class MetadataSnapshot:
    slot_mapping: Any
    block_tables: Any
    seq_lens: Any
    num_decodes: int = 0
    local_reads: dict[str, Any] = field(default_factory=dict)
    num_actual_tokens: int | None = None
    host_read_lengths: dict[str, Any] = field(default_factory=dict)


def as_list(value: Any) -> list:
    if isinstance(value, (list, tuple)):
        return list(value)
    return value.detach().cpu().tolist() if hasattr(value, "detach") else value.tolist()


def metadata_field(metadata: Any, *names: str) -> Any:
    return next((value for name in names if (value := getattr(metadata, name, None)) is not None), None)


def host_read_lengths(plan: AccessPlan, item: Any, layout: dict) -> dict[str, Any]:
    """The ordinary GQA metadata view publishes three full per-request lengths."""
    if (
        plan.phase != "forward"
        or plan.role != "target"
        or plan.scheduled is None
        or plan.dynamic
        or plan.parallel
        or layout.get("reason")
        or layout.get("cp_size", 1) != 1
        or layout.get("dcp_size", 1) != 1
    ):
        return {}
    if isinstance(item, MetadataSnapshot):
        return item.host_read_lengths
    lengths = getattr(item, "seq_lens", None)
    lengths_list = getattr(item, "seq_lens_list", None)
    if (
        getattr(item, "causal", None) is not True
        or not isinstance(getattr(item, "seq_lens_gpu", None), torch.Tensor)
        or not isinstance(lengths, torch.Tensor)
        or lengths.device.type != "cpu"
        or not isinstance(lengths_list, list)
    ):
        return {}
    # The producer's seq_lens_cpu is an alias of seq_lens.
    return {"seq_lens": lengths, "seq_lens_list": lengths_list}


def local_reads(item: Any) -> dict:
    if isinstance(item, MetadataSnapshot):
        return item.local_reads
    decode = getattr(item, "decode", None)
    history = getattr(getattr(item, "prefill", None), "chunked_context", None)
    return dict(
        causal=getattr(item, "causal", None),
        decode_kind="gqa"
        if hasattr(decode, "num_computed_tokens_of_dcp")
        else "mla"
        if hasattr(decode, "cp_seq_len")
        else None,
        decode_query=getattr(decode, "actual_seq_lengths_q", None),
        decode_history=getattr(decode, "cp_history_seq_len", None),
        decode_full=getattr(decode, "cp_seq_len", None),
        decode_shards=getattr(decode, "num_computed_tokens_of_dcp", None),
        decode_external=getattr(item, "external_flashmla", None) is not None,
        prefill_context=history is not None,
        prefill_history=getattr(history, "local_context_lens", None),
        prefill_cumulative_history=getattr(history, "actual_seq_lengths_kv", None),
    )


def describe_group(spec: Any, kernel_block_size: int, cp_size: int = 1, cp_rank: int = 0, interleave: int = 1) -> dict:
    """Choose checks by storage semantics, independently of feature names."""
    specs = tuple(iter_layer_specs(spec))
    reason = None
    for item in specs:
        if not hasattr(item, "head_size") or not getattr(item, "uses_slot_mapping", True):
            reason = "state_or_unmapped_storage"
        elif any(getattr(item, name, None) for name in ("sliding_window", "attention_chunk_size", "rswa_window")):
            reason = "bounded_history"
        elif getattr(item, "tokens_per_state", getattr(item, "compress_ratio", 1)) != 1:
            reason = "compressed_storage"
        elif isinstance(item, CircularBufferSpec) or getattr(item, "is_circular", False):
            reason = "circular_storage"
    size = spec.block_size
    if (
        size <= 0
        or kernel_block_size <= 0
        or size % kernel_block_size
        or cp_size <= 0
        or not 0 <= cp_rank < cp_size
        or interleave <= 0
    ):
        reason = "block_coordinates_unadapted"
    return dict(
        block_size=size,
        kernel_block_size=kernel_block_size,
        cp_size=cp_size,
        cp_rank=cp_rank,
        interleave=interleave,
        sharded=all(getattr(item, "dcp_sharded", True) for item in specs),
        reason=reason,
    )


def cache_group(key: Any) -> int:
    return key[0] if isinstance(key, tuple) else key


def cache_layouts(config, tables, parallel, dcp_rank):
    kernels = getattr(tables, "kernel_block_sizes", None)
    return {
        group: dict(
            describe_group(
                spec.kv_cache_spec,
                kernels[group] if kernels is not None else spec.kv_cache_spec.block_size,
                getattr(tables, "cp_size", getattr(parallel, "decode_context_parallel_size", 1)),
                getattr(tables, "cp_rank", 0),
                getattr(tables, "cp_interleave", 1),
            ),
            dcp_size=getattr(parallel, "decode_context_parallel_size", 1),
            dcp_rank=dcp_rank,
        )
        for group, spec in enumerate(config.kv_cache_groups)
    }


def local_sequence_lengths(lengths: list[int], cp: int, rank: int | None, interleave: int) -> list:
    if cp == 1 or not lengths:
        return [[length] for length in lengths] if rank is None else lengths
    return get_dcp_local_seq_lens(
        torch.tensor(lengths, dtype=torch.int32),
        dcp_size=cp,
        dcp_rank=rank,
        cp_kv_cache_interleave_size=interleave,
    ).tolist()


def metadata_views(metadata: dict, layer_groups: dict) -> dict:
    """Collect each distinct target metadata view once per cache group."""
    views, seen = {}, set()
    counts: dict[int, int] = {}
    for name, item in metadata.items():
        group = layer_groups.get(name)
        if group is None or item is None or (group, id(item)) in seen:
            continue
        count = counts.get(group, 0)
        views[group if count == 0 else (group, count)] = item
        counts[group] = count + 1
        seen.add((group, id(item)))
    return views


def violation(state, scope, reason, **fields):
    state.emit("kv_trace.violation", reason=reason, evidence="execution_plan", **scope, **fields)


def compare_lengths(state, scope, field, actual, expected, reason="read_length_mismatch") -> bool:
    if actual is None:
        return False
    values = as_list(actual)[: len(expected)]
    if values != expected:
        violation(
            state, dict(scope, operation="read", field=field), reason, expected_lengths=expected, actual_lengths=values
        )
    return True


def audit_read(state, table, lengths, expected, scope, size, host_lengths):
    """Compare the target metadata read table and lengths."""
    rows = len(scope["request_ids"])
    lengths = as_list(lengths)[:rows]
    columns = (
        max((len(row) for row in expected.tables), default=0)
        if expected
        else max(((length + size - 1) // size for length in lengths), default=0)
    )
    table = (
        [row[:columns] for row in table[:rows]] if isinstance(table, (list, tuple)) else as_list(table[:rows, :columns])
    )
    if expected is not None:
        compare_lengths(state, scope, "seq_lens_gpu" if host_lengths else "seq_lens", lengths, expected.lengths)
        for field, actual in host_lengths.items():
            compare_lengths(state, scope, field, actual, expected.lengths)
        if len(table) != len(expected.tables) or any(
            row[: len(want)] != want for row, want in zip(table, expected.tables)
        ):
            violation(
                state,
                dict(scope, operation="read"),
                "read_target_mismatch",
                expected_tables=expected.tables,
                actual_tables=table,
            )
    return table, lengths


class PlanSnapshotter:
    """Reusable D2H buffers; one outstanding snapshot at a time."""

    def __init__(self):
        self._host_buffers: list[Any] = []
        self._copy_event: Any = None
        self._sources: list[Any] = []

    @torch.inference_mode()
    def freeze(self, plan: AccessPlan) -> tuple[AccessPlan, Any]:
        """Copy final values now; return the event that makes them host-readable."""
        copies: dict[Any, Any] = {}
        self._sources.clear()  # The caller drains the previous audit before reusing this snapshotter.
        event = None
        device_copies = 0

        def freeze(value):
            nonlocal event, device_copies
            if isinstance(value, torch.Tensor):
                key = value.device, value.dtype, value.data_ptr(), tuple(value.shape), tuple(value.stride())
                if key not in copies:
                    if value.device.type == "cpu":
                        host = value.clone()
                    else:
                        self._sources.append(value)
                        if self._copy_event is None:
                            self._copy_event = cast(Any, torch).npu.Event()
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
            if isinstance(value, dict):
                return {key: freeze(item) for key, item in value.items()}
            if hasattr(value, "copy"):
                return value.copy()
            return value

        try:
            metadata = {}
            for group, item in plan.metadata.items():
                slots = metadata_field(item, "slot_mapping")
                count = getattr(item, "num_actual_tokens", None)
                table = metadata_field(item, "block_tables", "block_table", "block_table_tensor")
                group_id = cache_group(group)
                layout = plan.layout.get(group_id, {})
                reads = dict(local_reads(item))
                shards = reads.pop("decode_shards", None)
                if shards is not None:
                    rank = layout.get("dcp_rank", layout.get("cp_rank", 0))
                    reads["decode_full"] = (
                        None
                        if rank is None
                        else [row[rank] for row in shards]
                        if isinstance(shards, (list, tuple))
                        else shards[:, rank]
                    )
                if (
                    isinstance(table, torch.Tensor)
                    and table.ndim == 2
                    and plan.scheduled is not None
                    and not plan.parallel
                    and not layout.get("reason")
                ):
                    factor = layout.get("block_size", 1) // layout.get("kernel_block_size", 1)
                    columns = max(
                        (
                            len(groups[group_id]) * factor
                            for request in plan.request_ids
                            if group_id < len(groups := plan.bindings.get(request, ()))
                        ),
                        default=0,
                    )
                    table = table[: len(plan.request_ids), :columns]
                lengths = metadata_field(item, "seq_lens_gpu", "seq_lens", "seq_lens_list", "seq_lens_cpu")
                host_lengths = host_read_lengths(plan, item, layout)
                if host_lengths:
                    rows = len(plan.request_ids)
                    lengths = lengths[:rows]
                    host_lengths = {field: freeze(value[:rows]) for field, value in host_lengths.items()}
                metadata[group] = MetadataSnapshot(
                    slot_mapping=freeze(slots if slots is None or count is None or plan.parallel else slots[:count]),
                    block_tables=freeze(table),
                    seq_lens=freeze(lengths),
                    num_decodes=getattr(item, "num_decodes", 0),
                    local_reads=freeze(reads),
                    num_actual_tokens=count,
                    host_read_lengths=host_lengths,
                )
            batch = freeze(plan.batch)
            parallel = freeze(plan.parallel)
            if event is not None:
                event.record(cast(Any, torch).npu.current_stream())
        except Exception:
            if event is not None:
                # Even a partially failed snapshot must drain before buffer reuse.
                event.record(cast(Any, torch).npu.current_stream())
                event.synchronize()
            raise

        return replace(plan, metadata=metadata, batch=batch, parallel=parallel), event


def expected_access(
    state: KVCacheState, plan: AccessPlan, group_id: int, gap: Callable[..., None], *, slot: bool = True
) -> ExpectedAccess | None:
    """Derive destinations from the scheduler, never from the audited mapping.

    Per-token lists exist only for this forward and are shared by all layers
    in a cache group. No persistent token-ID history or KV tensor is copied.
    """
    if plan.phase != "forward" or plan.scheduled is None:
        return None
    size = state.group_sizes[group_id]
    layout = plan.layout.get(group_id, {})
    cp = layout.get("cp_size", 1) if layout.get("sharded", True) else 1
    rank, interleave = layout.get("cp_rank", 0), layout.get("interleave", 1)
    kernel_block_size = layout.get("kernel_block_size", size)
    factor = size // kernel_block_size
    slots: list[int] = []
    tables: list[list[int]] = []
    lengths: list[int] = []
    use_batch_positions = bool(plan.batch) and (slot or plan.dynamic)
    query_starts = as_list(plan.batch["query_start_loc"]) if use_batch_positions else []
    batch_positions = as_list(plan.batch["positions"]) if use_batch_positions else []
    for row, request_id in enumerate(plan.request_ids):
        scheduled_start, count = plan.scheduled.get(request_id, (None, 0))
        if scheduled_start is None or scheduled_start < 0 or count <= 0:
            gap("scheduler_write_range_missing", request_id=request_id)
            return None
        computed = scheduled_start
        groups = plan.bindings.get(request_id, ())
        refs = groups[group_id] if group_id < len(groups) else ()
        positions: Sequence[int] = range(computed, computed + count)
        if use_batch_positions:
            positions = batch_positions[query_starts[row] : query_starts[row + 1]]
            if plan.dynamic:
                count = len(positions)
                if positions:
                    computed = positions[0]
        needed = (computed + count + size * cp - 1) // (size * cp)
        if len(refs) < needed:
            gap("scheduler_read_blocks_missing", request_id=request_id, group_id=group_id)
            return None
        length = computed + count
        table = [block * factor + offset for block in refs[:needed] for offset in range(factor)]
        tables.append(table)
        lengths.append(length)
        if slot:
            for position in positions:
                logical, offset = divmod(position, size * cp)
                local = cp == 1 or offset // interleave % cp == rank
                offset = offset // (interleave * cp) * interleave + offset % interleave
                slots.append(refs[logical] * size + offset if local else -1)
    local_lengths = local_sequence_lengths(lengths, cp, rank, interleave)
    tables = [
        table[: (length + kernel_block_size - 1) // kernel_block_size] for table, length in zip(tables, local_lengths)
    ]
    return ExpectedAccess(slots, tables, lengths)


def valid_query(starts, rows, tokens):
    return (
        len(starts) == rows + 1
        and bool(starts)
        and starts[0] == 0
        and starts == sorted(starts)
        and starts[-1] <= tokens
    )


def audit_batch(plan: AccessPlan, fail: Callable[..., None]) -> bool:
    """Validate token attribution before using observed positions for addressing."""
    if not plan.batch or plan.scheduled is None:
        return True
    starts = as_list(plan.batch["query_start_loc"])
    positions = as_list(plan.batch["positions"])
    if not valid_query(starts, len(plan.request_ids), len(positions)):
        fail("batch_token_range_mismatch")
        return False
    for row, request in enumerate(plan.request_ids):
        computed, budget = plan.scheduled.get(request, (None, 0))
        actual = positions[starts[row] : starts[row + 1]]
        valid = computed is not None and budget > 0 and bool(actual)
        if valid:
            valid = (
                (len(actual) <= budget and 0 <= actual[0] <= computed)
                if plan.dynamic
                else (len(actual) == budget and actual[0] == computed)
            )
            valid = valid and actual == list(range(actual[0], actual[0] + len(actual)))
        if not valid:
            fail("execution_range_mismatch", request_id=request)
            return False
    return True


def record_plan_only(state: KVCacheState, plan: AccessPlan, item: Any, scope: dict, reason: str) -> None:
    """Keep observable evidence when an address invariant does not apply."""
    emit = partial(state.emit, evidence="execution_plan", **scope)
    fields = {
        name: metadata_field(item, *aliases)
        for name, aliases in {
            "slots": ("slot_mapping",),
            "block_table": ("block_tables", "block_table", "block_table_tensor"),
            "seq_lens": ("seq_lens_gpu", "seq_lens", "seq_lens_list", "seq_lens_cpu"),
        }.items()
    }
    fields = {name: None if value is None else as_list(value) for name, value in fields.items()}
    emit("kv_trace.check", status="not_applicable", reason=reason)
    emit("kv_trace.write", phase=plan.phase, slots=fields.pop("slots"))
    if any(value is not None for value in fields.values()):
        emit("kv_trace.read", **fields)


def audit_dcp_reads(state, plan, item, scope, lengths, gap):
    """Compare only the length fields consumed by the selected DCP attention path."""
    layout = plan.layout.get(scope["group_id"], {})
    size = layout.get("dcp_size", layout.get("cp_size", 1)) if layout.get("sharded", True) else 1
    if plan.phase != "forward" or plan.scheduled is None or size <= 1 or plan.role.endswith(".global"):
        return
    rank = layout.get("dcp_rank", layout.get("cp_rank", 0))
    if rank is None:
        gap("dcp_rank_unavailable", **scope)
        return
    # Load backend policy only for DCP; the ordinary checker needs no backend initialization.
    from vllm_ascend.attention.context_parallel.common_cp import use_history_current_split_decode

    interleave = layout.get("interleave", 1)
    local = partial(local_sequence_lengths, cp=size, rank=rank, interleave=interleave)

    def history(batch, ids):
        if not batch:
            return [plan.scheduled[request][0] for request in ids]
        starts, positions = as_list(batch["query_start_loc"]), as_list(batch["positions"])
        return [positions[start] if start < len(positions) else 0 for start in starts[: len(ids)]]

    ids = plan.request_ids
    source = plan.parallel.get("batch", plan.batch)
    global_ids = source.get("request_ids", ids)
    scheduled = dict(zip(global_ids, history(source, global_ids)))
    decodes = min(getattr(item, "num_decodes", 0), len(ids))
    reads = getattr(item, "local_reads", {})
    checks = {}
    if decodes:
        query = reads.get("decode_query")
        kind = reads.get("decode_kind")
        if reads.get("decode_external") or kind not in ("gqa", "mla"):
            gap("dcp_decode_consumer_unadapted", **scope)
        elif reads.get("causal") is None or query is None:
            gap("dcp_decode_scope_unavailable", **scope)
        else:
            split = use_history_current_split_decode(
                SimpleNamespace(causal=reads["causal"], decode=SimpleNamespace(actual_seq_lengths_q=as_list(query))),
                use_spec_decode=plan.dynamic,
            )
            field = (
                "decode.cp_history_seq_len"
                if split
                else ("decode.num_computed_tokens_of_dcp" if kind == "gqa" else "decode.cp_seq_len")
            )
            checks[field] = (
                ids[:decodes],
                reads.get("decode_history" if split else "decode_full"),
                local(history(plan.batch, ids)[:decodes] if split else lengths[:decodes]),
            )
            if split or kind == "gqa":
                if plan.batch:
                    starts = as_list(plan.batch["query_start_loc"])
                    counts = [end - start for start, end in zip(starts, starts[1:])]
                else:
                    counts = [plan.scheduled[request][1] for request in ids]
                checks["decode.actual_seq_lengths_q"] = (ids[:decodes], query, list(accumulate(counts[:decodes])))
    if reads.get("prefill_context"):
        prefill_ids = ids[decodes:]
        if plan.parallel:
            if source.get("is_prefilling") is None:
                gap("dcp_prefill_rows_unavailable", **scope)
                prefill_ids = None
            else:
                prefill_ids = [
                    request for request, prefill in zip(global_ids, as_list(source["is_prefilling"])) if prefill
                ]
        if prefill_ids is not None:
            values = local([scheduled[request] for request in prefill_ids])
            if reads.get("prefill_history") is None and reads.get("prefill_cumulative_history") is None:
                gap("dcp_prefill_consumer_unadapted", **scope)
            else:
                checks.update(
                    {
                        "prefill.chunked_context.local_context_lens": (
                            prefill_ids,
                            reads.get("prefill_history"),
                            values,
                        ),
                        "prefill.chunked_context.actual_seq_lengths_kv": (
                            prefill_ids,
                            reads.get("prefill_cumulative_history"),
                            list(accumulate(values)),
                        ),
                    }
                )
    checked = []
    for field, (requests, actual, values) in checks.items():
        if compare_lengths(state, dict(scope, request_ids=requests), field, actual, values, "dcp_read_length_mismatch"):
            checked.append(field)
        else:
            gap("dcp_read_consumer_unavailable", field=field, **scope)
    if checked:
        state.emit("kv_trace.check", status="checked", check="dcp_local_reads", fields=checked, **scope)
    elif not decodes and not reads.get("prefill_context"):
        if all(scheduled.get(request) == 0 for request in ids):
            state.emit(
                "kv_trace.check", status="not_applicable", check="dcp_local_reads", reason="no_cached_history", **scope
            )
        else:
            gap("dcp_local_lengths_unavailable", **scope)


def audit_access_plan(state: KVCacheState, plan: AccessPlan, gap: Callable[..., None], *, slot: bool = True) -> None:
    """Compare every final view using one expected address plan per cache group."""
    scope = dict(step_id=plan.step_id, role=plan.role, request_ids=plan.request_ids)
    emit = partial(state.emit, evidence="execution_plan", **scope)
    if plan.parallel:
        audit_parallel(state, plan, gap, slot=slot)
        return
    fail_batch = partial(violation, state, dict(scope, operation="bind"))
    batch_valid = audit_batch(plan, fail_batch)
    if not plan.metadata:
        if plan.phase != "warmup":
            gap("execution_access_plan_missing")
        return
    expectations = {}
    for key, item in plan.metadata.items():
        group = cache_group(key)
        layout = plan.layout.get(group, {})
        view = dict(scope, group_id=group, view_id=str(key))
        emit = partial(state.emit, evidence="execution_plan", full_graph=plan.full_graph, **view)
        if layout.get("reason") or plan.scheduled is None and plan.phase == "forward":
            record_plan_only(state, plan, item, view, layout.get("reason") or "request_mapping_unavailable")
            continue

        def fail(operation, reason, *, view=view, **fields):
            violation(state, dict(view, operation=operation, phase=plan.phase), reason, **fields)

        if group not in expectations:
            expectations[group] = expected_access(state, plan, group, gap, slot=slot) if batch_valid else None
        expected = expectations[group]
        size = state.group_sizes[group]
        slots = metadata_field(item, "slot_mapping")
        if slot and slots is None:
            gap("metadata_write_mapping_missing", group_id=group)
            continue
        slots = (
            None
            if slots is None
            else [value for row in as_list(slots) for value in (row if isinstance(row, list) else [row])]
        )
        if slot:
            for value in slots:
                if value < -1 or value >= state.num_blocks * size:
                    fail("write", "write_outside_cache", slot=value)
                    break
                if 0 <= value < size and not plan.allow_null_block:
                    fail("write", "write_null_block", slot=value)
                    break
            if plan.phase == "dummy":
                if any(value != -1 and not (plan.allow_null_block and 0 <= value < size) for value in slots):
                    fail("write", "dummy_write_target")
            elif expected is not None:
                count = len(expected.slots)
                if slots[:count] != expected.slots or any(value != -1 for value in slots[count:]):
                    fail("write", "write_target_mismatch", expected_slots=expected.slots, actual_slots=slots)
        if expected is not None:
            factor = size // layout.get("kernel_block_size", size)
            for block in {block // factor for row in expected.tables for block in row}:
                if block < 0 or block >= state.num_blocks or (block == 0 and not plan.allow_null_block):
                    fail("write", "invalid_scheduled_block", block_id=block)
                    break
        emit("kv_trace.write", phase=plan.phase, slots=slots)
        if not slot:
            emit("kv_trace.check", status="not_applicable", check="write_plan", reason="slot_disabled")
        if plan.phase != "forward":
            continue
        table = metadata_field(item, "block_tables", "block_table", "block_table_tensor")
        lengths = metadata_field(item, "seq_lens_gpu", "seq_lens", "seq_lens_list", "seq_lens_cpu")
        if table is None or lengths is None:
            gap("metadata_read_range_missing", group_id=group)
            continue
        table, lengths = audit_read(state, table, lengths, expected, view, size, host_read_lengths(plan, item, layout))
        if expected is not None:
            audit_dcp_reads(state, plan, item, view, expected.lengths, gap)
        emit("kv_trace.read", block_table=table, seq_lens=lengths)
        emit(
            "kv_trace.check",
            status="checked" if expected is not None else "unavailable",
            check="address_plan" if slot else "read_plan",
        )


def audit_parallel(state: KVCacheState, plan: AccessPlan, gap: Callable[..., None], *, slot: bool = True) -> None:
    """Check the native global declaration, then its local projection; never repartition."""
    context = plan.parallel
    batch = context["batch"]
    ids = batch["request_ids"]
    global_slots: Any = context.get("slot_mappings")
    global_tables = context["block_tables"]
    metadata = {
        group: MetadataSnapshot(
            None if global_slots is None else global_slots[group][: batch["num_tokens"]], table, batch["seq_lens"]
        )
        for group, table in enumerate(global_tables)
    }
    audit_access_plan(
        state,
        replace(plan, request_ids=ids, metadata=metadata, batch=batch, parallel={}, role=plan.role + ".global"),
        gap,
        slot=slot,
    )

    scope = dict(operation="parallel", step_id=plan.step_id, role=plan.role, request_ids=plan.request_ids)
    fail = partial(violation, state, scope)

    indices = context.get("local_token_indices")
    gather: Any = context.get("padded_gather_idx")
    mask = context.get("gathered_kv_write_mask")
    local = plan.batch
    if not local or indices is None:
        gap("parallel_local_mapping_unavailable", step_id=plan.step_id)
        return
    starts, positions = as_list(batch["query_start_loc"]), as_list(batch["positions"])
    if not valid_query(starts, len(ids), len(positions)):
        return  # The independent global audit already records the invalid range.
    query, actual_positions = as_list(local["query_start_loc"]), as_list(local["positions"])
    indices = as_list(indices)[: local["num_tokens"]]
    if (
        len(query) != len(plan.request_ids) + 1
        or not query
        or query[0] != 0
        or query != sorted(query)
        or (local["num_tokens"] != 0 and query[-1] != local["num_tokens"])
        or len(indices) != local["num_tokens"]
        or any(index < 0 or index >= batch["num_tokens_after_padding"] for index in indices)
        or not set(plan.request_ids).issubset(ids)
    ):
        fail("pcp_local_range_mismatch")
        return
    owners = [request for row, request in enumerate(ids) for _ in range(starts[row + 1] - starts[row])]
    rows = {request: row for row, request in enumerate(ids)}
    row_map = context.get("local_to_global_req_indices")
    if row_map is not None and as_list(row_map) != [rows[request] for request in plan.request_ids]:
        fail("pcp_request_mapping_mismatch")
    local_lengths = as_list(local["seq_lens"])
    for row, request in enumerate(plan.request_ids):
        begin, end = query[row : row + 2]
        selected = indices[begin:end]
        if any(index >= starts[-1] for index in selected):
            continue  # Native graph padding has no scheduled token owner.
        if actual_positions[begin:end] != [positions[index] for index in selected] or any(
            owners[index] != request for index in selected
        ):
            fail("pcp_token_mapping_mismatch", request_id=request)
        actual = actual_positions[begin:end]
        if local_lengths[row] != (actual[-1] + 1 if actual else 0):
            fail("pcp_read_length_mismatch", request_id=request)
    if slot:
        if global_slots is None or gather is None:
            gap("parallel_write_frame_unavailable", step_id=plan.step_id)
            return
        gather = as_list(gather)
        mask = None if mask is None else as_list(mask)
        if mask is not None and len(mask) != len(gather):
            fail("pcp_write_mask_shape_mismatch")
            return
    for key, item in plan.metadata.items():
        group = cache_group(key)
        scope = dict(
            group_id=group, step_id=plan.step_id, role=plan.role, request_ids=plan.request_ids, view_id=str(key)
        )
        if plan.layout.get(group, {}).get("reason"):
            record_plan_only(state, plan, item, scope, plan.layout[group]["reason"])
            continue
        audit_dcp_reads(state, plan, item, scope, local_lengths, gap)
        slots = metadata_field(item, "slot_mapping")
        if slot:
            source = as_list(global_slots[group])
            if any(index < 0 or index >= len(source) for index in gather):
                fail("pcp_gather_index_mismatch", group_id=group)
                continue
            expected = [source[index] if mask is None or mask[token] else -1 for token, index in enumerate(gather)]
            if slots is None:
                gap("metadata_write_mapping_missing", **scope)
                continue
        slots = None if slots is None else as_list(slots)
        if slot and slots != expected:
            fail("pcp_write_frame_mismatch", group_id=group, expected_slots=expected, actual_slots=slots)
        state.emit("kv_trace.write", evidence="execution_plan", slots=slots, full_graph=plan.full_graph, **scope)
        if not slot:
            state.emit(
                "kv_trace.check",
                status="not_applicable",
                check="parallel_write_projection",
                reason="slot_disabled",
                **scope,
            )
        table = metadata_field(item, "block_tables", "block_table", "block_table_tensor")
        if table is not None:
            actual_tables = as_list(table)
            source_tables = as_list(global_tables[group])
            factor = plan.layout[group]["block_size"] // plan.layout[group]["kernel_block_size"]
            for row, request in enumerate(plan.request_ids):
                groups = plan.bindings.get(request, ())
                count = len(groups[group]) * factor if group < len(groups) else 0
                if row >= len(actual_tables) or actual_tables[row][:count] != source_tables[rows[request]][:count]:
                    fail("pcp_read_block_mapping_mismatch", request_id=request, group_id=group)
        else:
            gap("metadata_read_range_missing", **scope)
        state.emit(
            "kv_trace.read",
            evidence="execution_plan",
            block_table=as_list(table) if table is not None else None,
            **scope,
        )
        if table is not None or slot:
            check = "parallel_projection" if table is not None else "parallel_write_projection"
            state.emit(
                "kv_trace.check",
                status="checked",
                check=check if slot else "parallel_read_projection",
                **scope,
            )
