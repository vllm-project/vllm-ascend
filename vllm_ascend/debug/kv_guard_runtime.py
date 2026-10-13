# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker L1 guard with one runner execution-plan collection point.

No KV snapshots, hashes, general-purpose trace or offline reader.
Per-layer final metadata supplies declared access plans, including caches with
no attention implementation. Optional operator hooks preserve argument order.
Only small metadata is copied, after the forward and graph parameter updates;
the checker neither inspects KV values nor proves kernel memory accesses.
"""

from __future__ import annotations

import functools
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any

from vllm_ascend.debug.kv_guard import BlockState, KVGuard


@dataclass(slots=True)
class WriteAccess:
    layer: str
    slots: Any


@dataclass(slots=True)
class ReadAccess:
    layer: str
    table: Any
    lengths: Any
    starts: Any = None
    request_offset: int = 0


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
        self.layer_modes: dict[str, str] = {}
        self.operator_layers: set[str] = set()
        self._accesses: list[WriteAccess | ReadAccess] = []
        self._metadata: dict[str, Any] = {}
        self._missing_groups: set[int] = set()
        self._read_hook_seen = False
        self._plans: dict[tuple[str, int], list[WriteAccess | ReadAccess]] = {}
        self._active = False
        self._phase = "warmup"
        self._key = ("", 0)
        self._reports = 0
        self._max_reports = 10_000
        self._seen_gaps: set[str] = set()
        self._violations: OrderedDict[tuple, None] = OrderedDict()
        self._bound_output: Any = None

    def _emit_scratch(self, event: str, **fields: Any) -> None:
        if event == "guard.violation":
            self.state.emit(event, layer=self._audit_layer, evidence=self._audit_evidence, **fields)

    def _record_operator(self, layer: str) -> bool:
        # FULL uses the current final plan, not redundant parameter callbacks.
        return self._active and not ("FULL" in self._key[0] and layer in self._metadata)

    def report(self, event: str, **fields: Any) -> None:
        if event == "guard.violation":
            key = (
                fields.get("operation"),
                fields.get("reason"),
                fields.get("pool_id"),
                fields.get("group_id"),
                fields.get("block_id"),
                fields.get("alloc_epoch"),
                fields.get("layer"),
                fields.get("request_id"),
                tuple(fields.get("request_ids", ())),
            )
            if key in self._violations:
                self._violations.move_to_end(key)
                return
            self._violations[key] = None
            if len(self._violations) > 1024:
                self._violations.popitem(last=False)
            payload = " ".join(f"{key}={value}" for key, value in fields.items())
            self.logger.error("[KV_GUARD] %s %s", event, payload)
            return
        if event != "guard.state":
            payload = " ".join(f"{key}={value}" for key, value in fields.items())
            self.logger.info("[KV_GUARD] %s %s", event, payload)
            return
        if self._reports == self._max_reports:
            self.logger.warning("[KV_GUARD] state log limit reached; checks and violation reports remain active")
        if self._reports >= self._max_reports:
            self._reports += 1
            return
        self._reports += 1
        payload = " ".join(f"{key}={value}" for key, value in fields.items())
        self.logger.info("[KV_GUARD] %s %s", event, payload)

    def gap(self, reason: str) -> None:
        if reason not in self._seen_gaps:
            self._seen_gaps.add(reason)
            self.report("guard.coverage_gap", reason=reason)

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
                self.layer_modes[name] = "append" if append else "reserved_only"
                if not append:
                    self.gap(f"cache_validity_rules_unadapted:{type(layer_spec).__name__}")
                layer = config.compilation_config.static_forward_context.get(name)
                impl = getattr(layer, "impl", None)
                if impl is not None and type(impl).__name__ in ("AscendAttentionBackendImpl", "AscendMLAImpl"):
                    impl._kv_guard = self
                    self.operator_layers.add(name)
        if unsupported:
            raise ValueError(
                "KV L1 guard: unadapted execution mode or block coordinates "
                "(CP, DBO, speculation, encoder/decoder, connector)"
            )
        self.state.configure(sizes)
        self.report(
            "guard.enabled",
            states=5,
            checksum=False,
            kv_content_copy=False,
            execution_plans=True,
            operator_arguments=True,
            kernel_accesses=False,
            prevent_execution=False,
        )

    def write(self, layer: str, slots: Any) -> None:
        if self._record_operator(layer):
            self._accesses.append(WriteAccess(layer, slots))

    def read(self, layer: str, table: Any, lengths: Any, starts: Any = None, request_offset: int = 0) -> None:
        if self._active:
            self._read_hook_seen = True
            if table is not None and self._record_operator(layer):
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
            return  # Profile/legacy operator-only calls have no attention plan.
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
        if self.layer_modes.get(layer, "append") == "append":
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

    def _audit(self, runner: Any) -> None:
        # FULL replay does not rerun Python writes. Prefer this step's metadata;
        # captured arguments are only a fallback for legacy calls without plans.
        observed: dict[str, list[WriteAccess | ReadAccess]] = {}
        for access in self._accesses:
            observed.setdefault(access.layer, []).append(access)
        fallback: dict[str, list[WriteAccess | ReadAccess]] = {}
        if not self._metadata and "FULL" in self._key[0]:
            for access in self._plans.get(self._key, ()):
                if isinstance(access, WriteAccess):
                    fallback.setdefault(access.layer, []).append(access)
            for layer, accesses in observed.items():
                fallback.setdefault(layer, []).extend(a for a in accesses if isinstance(a, ReadAccess))
        if not self._metadata and not fallback and not observed:
            if self._phase != "warmup":
                self.gap("execution_access_plan_missing")
            return
        if self._phase != "warmup" and "FULL" in self._key[0] and not self._metadata and not self._read_hook_seen:
            self.gap("full_graph_read_hook_missing")

        copies: dict[Any, list] = {}
        results: dict[int, list[dict]] = {}
        audited: set[Any] = set()
        templates: dict[tuple[int, str, int], list[WriteAccess | ReadAccess]] = {}
        baseline: dict = {}
        tainted_nulls: set[int] = set()
        shadow = self._scratch
        shadow.configure(self.state.group_sizes)
        for layer in dict.fromkeys((*self._metadata, *fallback, *observed)):
            group_id = self.layer_groups.get(layer)
            if group_id is None:
                self.gap(f"unbound_access_layer:{layer}")
                continue
            actual = observed.get(layer, [])
            if observed and layer in self.operator_layers and not actual and "FULL" not in self._key[0]:
                self.gap(f"operator_layer_not_observed:{layer}")
                self._missing_groups.add(group_id)
            mode = self.layer_modes.get(layer, "append")
            if actual and "FULL" not in self._key[0]:
                accesses = actual
                evidence = "operator_arguments"
            elif layer in self._metadata:
                item = self._metadata[layer]
                template_key = id(item), mode, group_id
                if template_key not in templates:
                    templates[template_key] = self._metadata_accesses(layer, item)
                accesses = templates[template_key]
                evidence = "execution_plan"
            else:
                accesses = fallback.get(layer, actual)
                evidence = "execution_plan" if layer in fallback else "operator_arguments"
            signature = (
                group_id,
                mode,
                evidence,
                tuple(
                    ("write", self._identity(a.slots))
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
            keys: set[tuple[int, int]] = set()
            write_keys: set[tuple[int, int]] = set()
            size = self.state.group_sizes[group_id]
            for access in accesses:
                entry: dict[str, Any] = {
                    "group_id": group_id,
                    "block_size": size,
                    "write_mode": mode,
                    "slots": [],
                    "block_table": [],
                    "seq_lens": [],
                }
                if isinstance(access, WriteAccess):
                    slots = values(access.slots)
                    entry["slots"] = [slot for row in slots for slot in (row if isinstance(row, list) else [row])]
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
                groups.append((entry, offset))

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
                shadow.tainted_nulls = set(self.state.tainted_nulls)
            self._audit_layer, self._audit_evidence = layer, evidence
            request_ids = list(runner.input_batch.req_ids) if self._phase == "forward" else []
            for entry, offset in groups:
                shadow.observe([entry], request_ids[offset:], self._phase)
            tainted_nulls.update(shadow.tainted_nulls)
            audited.add(signature)
            results.setdefault(group_id, []).append(shadow.blocks)

        # Each layer starts from the same pre-forward validity. Another layer's
        # write cannot legalize an earlier read or hide a missing write. Merge
        # conservatively; no validity is advanced on failed/partial collection.
        with self.state._lock:
            self.state.tainted_nulls.update(tainted_nulls)
            for group_id, versions_by_layer in results.items():
                touched = {key for versions in versions_by_layer for key in versions}
                for key in touched:
                    block = self.state.blocks.get(key)
                    if block is None:
                        continue
                    versions = [versions.get(key, baseline.get(key, block)) for versions in versions_by_layer]
                    if any((b.pool_id, b.epoch) != (block.pool_id, block.epoch) for b in versions):
                        self.gap("allocation_changed_during_access_audit")
                        continue
                    if any(b.state == BlockState.INVALID for b in versions):
                        self.state._transition(key, block, BlockState.INVALID, block.valid_end, "illegal_access")
                    elif (
                        block.state != BlockState.INVALID
                        and group_id not in self._missing_groups
                        and self._phase == "forward"
                    ):
                        valid = min(b.valid_end for b in versions)
                        if valid > block.valid_end:
                            state = BlockState.SEALED_FILL if valid == block.size else BlockState.PARTIAL
                            self.state._transition(key, block, state, valid, "completed_access_plan")

    def wrap_forward(self, runner: Any, forward: Any, get_context: Any):
        @functools.wraps(forward)
        def observed(num_tokens, *args, **kwargs):
            phase = kwargs.pop("_kv_guard_phase", "forward")
            context = get_context()
            context._kv_guard = self
            self._key = str(context.cudagraph_runtime_mode), num_tokens
            self._phase, self._active = phase, True
            self._accesses = []
            self._read_hook_seen = False
            try:
                self.collect_metadata(getattr(context, "attn_metadata", None))
            except Exception as exc:
                self._metadata = {}
                self._missing_groups = set(self.layer_groups.values())
                self.gap(f"metadata_collection_failed:{type(exc).__name__}")
            try:
                output = forward(num_tokens, *args, **kwargs)
                if phase == "warmup" and "FULL" in self._key[0] and self._accesses:
                    self._plans[self._key] = list(self._accesses)
                try:
                    self._audit(runner)
                except Exception as exc:
                    self.gap(f"access_audit_failed:{type(exc).__name__}")
                return output
            finally:
                self._active = False

        return observed

    def wrap_update(self, runner: Any, update: Any):
        @functools.wraps(update)
        def observed(output, *args, **kwargs):
            result = update(output, *args, **kwargs)
            binding = getattr(output, "_ascend_kv_guard_context", None)
            if binding is None:
                self.gap("scheduler_binding_missing")
                return result
            verified = []
            for request in binding["requests"]:
                actual = runner.requests.get(request["request_id"])
                if actual is None or [list(g) for g in actual.block_ids] != request["block_ids"]:
                    self.report(
                        "guard.violation",
                        operation="bind",
                        reason="worker_block_mapping_mismatch",
                        request_id=request["request_id"],
                    )
                else:
                    verified.append(request)
            if self._bound_output is not output or len(verified) != len(binding["requests"]):
                self.state.bind({**binding, "requests": verified})
            return result

        return observed

    def wrap_execute(self, execute: Any):
        @functools.wraps(execute)
        def observed(output, *args, **kwargs):
            binding = getattr(output, "_ascend_kv_guard_context", None)
            if binding is not None:
                # Receive-only and idle DP steps can return before _update_states.
                self.state.bind(binding)
            else:
                self.gap("scheduler_binding_missing")
            self._bound_output = output if binding is not None else None
            try:
                return execute(output, *args, **kwargs)
            finally:
                self._bound_output = None

        return observed

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
