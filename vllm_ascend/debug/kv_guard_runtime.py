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
from dataclasses import dataclass, replace
from typing import Any

from vllm_ascend.debug.kv_guard import INVALID_PREFIX, KVGuard

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
        self.action = "log"

    def _check_failure(self) -> None:
        if self.action == "raise" and self._failure is not None:
            raise KVGuardError(f"KV guard stopped model output: {self._failure}; restart the worker before reuse")

    def _emit_scratch(self, event: str, **fields: Any) -> None:
        if event == "guard.violation":
            self.state.emit(event, layer=self._audit_layer, evidence=self._audit_evidence, **fields)

    def report(self, event: str, **fields: Any) -> None:
        failed = event in ("guard.violation", "guard.coverage_gap")
        if failed:
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
        from vllm_ascend import envs

        self.action = envs.VLLM_ASCEND_KV_GUARD_ACTION
        if self.action not in ("log", "raise"):
            raise ValueError("VLLM_ASCEND_KV_GUARD_ACTION must be 'log' or 'raise'")
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
                transfer is not None and transfer.kv_connector != "MooncakeConnector",
            )
        )
        sizes = {}
        tables = getattr(getattr(runner.input_batch, "block_table", None), "block_tables", ())
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
        self.report("guard.enabled", failure_action=self.action)

    def write(self, layer: str, slots: Any, token_offset: int = 0) -> None:
        if self._phase is not None and not self._full_graph:
            self._accesses.append(WriteAccess(layer, slots, token_offset))

    def read(self, layer: str, table: Any, lengths: Any, starts: Any = None, request_offset: int = 0) -> None:
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

    def _audit(self, runner: Any) -> None:
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
                    if self._phase == "warmup":
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

    def wrap_forward(self, runner: Any, forward: Any, get_context: Any):
        @functools.wraps(forward)
        def observed(num_tokens, *args, **kwargs):
            self._check_failure()
            phase = kwargs.pop("_kv_guard_phase", "forward")
            context = get_context()
            context._kv_guard = self
            self._full_graph = "FULL" in str(context.cudagraph_runtime_mode)
            self._phase = phase
            self._accesses = []
            try:
                self.collect_metadata(getattr(context, "attn_metadata", None))
            except Exception as exc:
                self._metadata = {}
                self._missing_groups = set(self.layer_groups.values())
                self.gap(f"metadata_collection_failed:{type(exc).__name__}")
            try:
                output = forward(num_tokens, *args, **kwargs)
                try:
                    self._audit(runner)
                except Exception as exc:
                    self.gap(f"access_audit_failed:{type(exc).__name__}")
                self._check_failure()
                return output
            finally:
                self._phase = None

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

    def wrap_execute(self, execute: Any):
        @functools.wraps(execute)
        def observed(output, *args, **kwargs):
            self._check_failure()
            binding = getattr(output, "_ascend_kv_guard_context", None)
            if binding is not None:
                # Receive-only and idle DP steps can return before _update_states.
                uninitialized = set(getattr(output, "new_block_ids_to_zero", None) or ())
                uninitialized.update(
                    copy.dst_block_id for copy in (getattr(output, "kv_cache_block_copies", None) or ())
                )
                self.state.bind({**binding, "uninitialized_block_ids": uninitialized})
                self._scheduled = binding.get("scheduled_requests")
                if self._scheduled is None:
                    self.gap("scheduler_write_plan_missing")
            else:
                self.gap("scheduler_binding_missing")
                self._scheduled = None
            try:
                self._check_failure()
                return execute(output, *args, **kwargs)
            finally:
                self._scheduled = None

        return observed

    def zero_blocks(self, block_ids: list[int]) -> None:
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

    def transfer(self, event: str, meta: dict[str, Any]) -> None:
        try:
            if event == "load":
                self.state.load(meta)
            else:
                self.state.fail_load(meta["request_id"])
        except Exception as exc:
            self.gap(f"load_hook_failed:{type(exc).__name__}")


def get_guard(vllm_config: Any) -> GuardRuntime | None:
    from vllm_ascend import envs

    if not envs.VLLM_ASCEND_KV_GUARD:
        return None
    guard = getattr(vllm_config, "_ascend_kv_guard", None)
    if guard is None:
        guard = GuardRuntime(vllm_config)
        vllm_config._ascend_kv_guard = guard
    return guard
