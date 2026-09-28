# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, opt-in KV lifecycle records; no model or sampling modifications.

This module intentionally has no vLLM/torch imports at module scope. Tensor
observation is restricted to the explicitly enabled worker diagnostic path.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceConfig:
    directory: str
    run_id: str = "kv-trace"
    snapshots: bool = False
    layers: tuple[int, ...] = (-1,)
    max_events: int = 100_000
    max_snapshot_bytes: int = 8 * 1024 * 1024

    @classmethod
    def parse(cls, value: str) -> TraceConfig:
        data = json.loads(value)
        if not isinstance(data, dict) or set(data) - cls.__dataclass_fields__.keys():
            raise ValueError("KV trace must be a JSON object containing only documented options")
        if not isinstance(data.get("directory"), str) or not data["directory"]:
            raise ValueError("KV trace requires a non-empty directory")
        run_id = data.get("run_id", "kv-trace")
        if (
            not isinstance(run_id, str)
            or not run_id
            or run_id in (".", "..")
            or any(not (char.isascii() and (char.isalnum() or char in "_.-")) for char in run_id)
        ):
            raise ValueError("KV trace run_id must be a single alphanumeric path component")
        if type(data.get("snapshots", False)) is not bool:
            raise ValueError("KV trace snapshots must be a boolean")
        for field in ("max_events", "max_snapshot_bytes"):
            if field in data and (type(data[field]) is not int or data[field] < 1):
                raise ValueError(f"KV trace {field} must be a positive integer")
        layers = data.get("layers", [-1])
        if not isinstance(layers, list) or not layers or any(type(i) is not int for i in layers):
            raise ValueError("KV trace layers must be a non-empty list of integer group-local layer indices")
        data["layers"] = tuple(dict.fromkeys(layers))
        return cls(**data)


class KVTrace:
    """One writer per component, with per-process identity and thread ordering."""

    def __init__(self, config: TraceConfig, component: str, **actor: Any):
        self.config = config
        self.component = component
        self.actor = actor
        self.host = socket.gethostname()
        self.pid = os.getpid()
        self.trace_id = uuid.uuid4().hex
        self.enabled = True
        self._sequence = 0
        self._lock = threading.Lock()
        self._stream = None

    @classmethod
    def from_env(cls, component: str, vllm_config: Any, **actor: Any) -> KVTrace | None:
        from vllm_ascend import envs

        value = envs.VLLM_ASCEND_KV_TRACE
        if not value:
            return None
        parallel = vllm_config.parallel_config
        transfer = vllm_config.kv_transfer_config
        context = {
            "engine_id": str(transfer.engine_id) if transfer is not None else "local",
            "kv_role": str(transfer.kv_role) if transfer is not None else "local",
            "dp_rank": parallel.data_parallel_rank,
        }
        context.update(actor)
        return cls(TraceConfig.parse(value), component, **context)

    def emit(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        if self.pid != os.getpid():
            # A child must never reuse its parent's descriptor, lock or sequence.
            self.pid = os.getpid()
            self.trace_id = uuid.uuid4().hex
            self._lock = threading.Lock()
            self._stream = None
            self._sequence = 0
        with self._lock:
            if not self.enabled:
                return
            if self._sequence >= self.config.max_events:
                event, fields = "trace.truncated", {"max_events": self.config.max_events}
                self.enabled = False
            self._sequence += 1
            row = {
                "schema_version": 1,
                "run_id": self.config.run_id,
                "host": self.host,
                "pid": self.pid,
                "thread_id": threading.get_ident(),
                "component": self.component,
                "trace_id": self.trace_id,
                "sequence": self._sequence,
                "wall_time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                **self.actor,
                "event": event,
                **fields,
            }
            try:
                if self._stream is None:
                    directory = Path(self.config.directory) / self.config.run_id
                    directory.mkdir(parents=True, exist_ok=True)
                    self._stream = (directory / f"{self.component}-{self.pid}-{self.trace_id}.jsonl").open(
                        "x", encoding="utf-8", buffering=1
                    )
                self._stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
                if not self.enabled:
                    self._stream.close()
            except (OSError, TypeError, ValueError):
                self.enabled = False
                logging.getLogger(__name__).exception("Disabling KV trace after a recording failure")

    def close(self) -> None:
        with self._lock:
            self.enabled = False
            if self._stream is not None:
                self._stream.close()

    def transfer(self, event: str, meta: dict[str, Any], **extra: Any) -> None:
        # Do not serialize payloads, endpoint credentials or transport objects.
        fields = {
            key: meta[key]
            for key in (
                "request_id",
                "remote_request_id",
                "remote_engine_id",
                "local_block_ids",
                "remote_block_ids",
                "num_computed_tokens",
                "all_task_done",
            )
            if key in meta
        }
        if "group_pulls" in meta:
            fields["group_pulls"] = [
                {
                    key: getattr(pull, key, None)
                    for key in (
                        "group_id",
                        "remote_tp_offset",
                        "num_group_pulls",
                        "prefill_pp_rank",
                        "is_group_transfer_end",
                    )
                }
                for pull in meta["group_pulls"]
            ]
        self.emit(event, **fields, **extra)

    def wrap_forward(self, runner: Any, run_model: Any, num_tokens: int, phase: str, context: Any):
        if not self.enabled or phase == "warmup" or getattr(context, "capturing", False):
            return run_model

        @functools.wraps(run_model)
        def observed():
            span_id = uuid.uuid4().hex
            snapshots = []
            try:
                groups = []
                request_ids = list(runner.input_batch.req_ids)
                for group_id, table in enumerate(runner.input_batch.block_table.block_tables):
                    # Actual device metadata is required to expose stale dummy slots.
                    # This opt-in path intentionally synchronizes; normal serving does not.
                    slots = table.slot_mapping.gpu[:num_tokens].cpu().tolist()
                    block_table = table.get_device_tensor()[: max(1, len(request_ids))].cpu().tolist()
                    groups.append(
                        {
                            "group_id": group_id,
                            "block_size": runner.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec.block_size,
                            "kernel_block_size": table.block_size,
                            "slots": slots,
                            "block_table": block_table,
                        }
                    )
                self.emit(
                    "forward.begin",
                    span_id=span_id,
                    phase=phase,
                    attention_state=str(runner.attn_state),
                    graph_mode=str(context.cudagraph_runtime_mode),
                    request_ids=request_ids,
                    groups=groups,
                    num_tokens=num_tokens,
                )
                if self.config.snapshots and self.enabled:
                    snapshots = self._snapshot(runner, groups, span_id)
            except Exception:
                # Observation failure must not substitute for the model result.
                self.emit("trace.observation_error", span_id=span_id, phase=phase, stage="before_forward")
                logging.getLogger(__name__).exception("KV trace could not observe forward inputs")
            try:
                output = run_model()
            except BaseException as exc:
                self.emit("forward.error", span_id=span_id, phase=phase, error_type=type(exc).__name__)
                raise
            try:
                for info, tensor, before in snapshots:
                    after = tensor.detach().cpu().contiguous()
                    diff = compare_snapshot(before, after)
                    offset = info["block_offset"]
                    diff["changed_offsets"] = [offset + i for i in diff["changed_offsets"]]
                    diff["changed_elements_per_offset"] = [
                        [offset + i, count] for i, count in diff["changed_elements_per_offset"]
                    ]
                    self.emit("cache.diff", span_id=span_id, phase=phase, **info, **diff)
            except Exception:
                self.emit("trace.observation_error", span_id=span_id, phase=phase, stage="after_forward")
                logging.getLogger(__name__).exception("KV trace could not compare cache snapshots")
            self.emit(
                "forward.end",
                span_id=span_id,
                phase=phase,
                completion="device_observed" if snapshots else "host_return",
            )
            return output

        return observed

    def observe_forward(self, runner: Any, forward: Any, get_context: Any):
        """Wrap the complete forward, including Ascend graph parameter updates.

        Synchronizing snapshots around only model()/graph.replay() can happen
        before _update_full_graph_params_if_needed and stall graph execution.
        Bind this wrapper once, only when tracing is enabled.
        """

        @functools.wraps(forward)
        def observed(num_tokens, *args, **kwargs):
            call = functools.partial(forward, num_tokens, *args, **kwargs)
            return self.wrap_forward(
                runner, call, num_tokens, kwargs.get("_kv_trace_phase", "forward"), get_context()
            )()

        return observed

    def _snapshot(self, runner: Any, groups: list[dict[str, Any]], span_id: str) -> list:
        snapshots = []
        used_bytes = 0
        if getattr(getattr(runner, "ascend_config", None), "enable_kv_nz", False):
            self.emit("snapshot.skipped", span_id=span_id, reason="packed_nz_layout")
            return snapshots
        for group in groups:
            group_id, block_size = group["group_id"], group["block_size"]
            kernel_size = group["kernel_block_size"]
            if block_size % kernel_size or any(type(slot) is not int for slot in group["slots"]):
                self.emit("snapshot.skipped", span_id=span_id, group_id=group_id, reason="unsupported_slot_layout")
                continue
            scale = block_size // kernel_size
            cache_group = runner.kv_cache_config.kv_cache_groups[group_id]
            layer_names = cache_group.layer_names
            indices = sorted(
                {i % len(layer_names) for i in self.config.layers if -len(layer_names) <= i < len(layer_names)}
            )
            if not indices:
                self.emit("snapshot.skipped", span_id=span_id, group_id=group_id, reason="layer_index_out_of_range")
            # Exclude null block zero; padding -1 has no physical cache write.
            block_ids = sorted({slot // kernel_size for slot in group["slots"] if slot >= block_size})
            for index in indices:
                layer_name = layer_names[index]
                spec = cache_group.kv_cache_spec
                if hasattr(spec, "kv_cache_specs"):
                    spec = spec.kv_cache_specs[layer_name]
                cache = runner._kv_trace_caches.get(layer_name)
                if not hasattr(spec, "head_size") or getattr(spec, "compress_ratio", 1) != 1 or cache is None:
                    self.emit(
                        "snapshot.skipped",
                        span_id=span_id,
                        group_id=group_id,
                        layer=layer_name,
                        reason="unsupported_spec",
                    )
                    continue
                for component, tensor in enumerate(cache_components(cache)):
                    if tensor.ndim < 3 or tensor.shape[1] != kernel_size:
                        self.emit(
                            "snapshot.skipped",
                            span_id=span_id,
                            group_id=group_id,
                            layer=layer_name,
                            reason="unsupported_layout",
                        )
                        continue
                    for block_id in block_ids:
                        if block_id >= tensor.shape[0]:
                            self.emit(
                                "snapshot.skipped",
                                span_id=span_id,
                                group_id=group_id,
                                block_id=block_id,
                                reason="invalid_block",
                            )
                            continue
                        size = tensor[block_id].numel() * tensor.element_size()
                        if used_bytes + size > self.config.max_snapshot_bytes:
                            self.emit(
                                "snapshot.truncated", span_id=span_id, max_snapshot_bytes=self.config.max_snapshot_bytes
                            )
                            return snapshots
                        view = tensor[block_id]
                        # clone is needed even for CPU tests: snapshots must not alias live KV.
                        before = view.detach().clone().cpu().contiguous()
                        used_bytes += size
                        info = {
                            "group_id": group_id,
                            "block_id": block_id // scale,
                            "kernel_block_id": block_id,
                            "kernel_block_size": kernel_size,
                            "block_offset": block_id % scale * kernel_size,
                            "block_size": block_size,
                            "layer": layer_name,
                            "cache_component": component,
                            "shape": list(tensor.shape),
                        }
                        snapshots.append((info, view, before))
        return snapshots


def cache_components(cache: Any) -> list[Any]:
    if isinstance(cache, (tuple, list)):
        return [tensor for tensor in cache if tensor is not None]
    # v1 dense attention often packs K/V in [2, blocks, tokens, heads, dim].
    if cache.ndim == 5 and cache.shape[0] == 2:
        return [cache[0], cache[1]]
    return [cache]


def compare_snapshot(before: Any, after: Any) -> dict[str, Any]:
    """Bitwise element differences; unchanged NaNs must not be called writes."""
    import torch

    before_bytes = before.view(torch.uint8)
    after_bytes = after.view(torch.uint8)
    changed = (before_bytes != after_bytes).reshape(before.shape[0], -1, before.element_size()).any(dim=-1)
    counts = changed.sum(dim=-1).tolist()
    return {
        "changed_offsets": [i for i, count in enumerate(counts) if count],
        "changed_elements": sum(counts),
        "changed_elements_per_offset": [[i, count] for i, count in enumerate(counts) if count],
        "before_sha256": hashlib.sha256(before_bytes.numpy().tobytes()).hexdigest(),
        "after_sha256": hashlib.sha256(after_bytes.numpy().tobytes()).hexdigest(),
    }
