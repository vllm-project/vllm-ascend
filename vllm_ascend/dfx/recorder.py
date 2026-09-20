# SPDX-License-Identifier: Apache-2.0
"""MRV1/MRV2 evidence recorder; optional device work is isolated in DeviceProbe.

Capture is protected by a nonblocking lock: contention drops diagnostics, never
waits for a result thread. The writer only sees owned snapshots.
"""

import base64
import dataclasses
import enum
import json
import logging
import os
import queue
import socket
import threading
import time
import uuid
from collections import OrderedDict, deque
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from vllm_ascend.dfx.config import DfxConfig
from vllm_ascend.dfx.detectors import (
    BlockOwnership,
    OutputContext,
    OutputDetectors,
    check_block_plan,
    check_positions,
    check_transfer_registration,
)

logger = logging.getLogger(__name__)
SCHEMA = "vllm-ascend.dfx.host.v1"
MAX_DEPTH = 12
RECORD_OVERHEAD = 2048
MAX_OMISSIONS = 32
DUMP_QUEUE_SIZE = 1
COMPACT_SEQUENCE_MIN = 8


def _nonblocking(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        if not self._capture_lock.acquire(blocking=False):
            self._contention_drops += 1
            if method.__name__ == "_record_schedule":
                self._reset_generations = True
            if method.__name__ in ("record_output", "output_context"):
                self._history_gap = True
            if method.__name__ == "record_lifecycle":
                self._ownership_gap = True
            return {"status": "contended"} if method.__name__ == "trigger" else None
        try:
            return method(self, *args, **kwargs)
        finally:
            self._capture_lock.release()

    return guarded


class _BudgetExceeded(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class _Array:
    dtype: str
    shape: tuple[int, ...]
    data: bytes


class _Snapshot:
    """Conservative retained-size accounting, not a process RSS guarantee."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = RECORD_OVERHEAD
        self.omissions: list[str] = []

    def charge(self, size: int) -> None:
        self.used += size
        if self.used > self.limit:
            raise _BudgetExceeded

    def copy(self, value: Any, path: str = "payload", depth: int = 0) -> Any:
        self.charge(256)
        if value is None or type(value) in (bool, float):
            return value
        if type(value) is int:
            self.charge(value.bit_length() // 8)
            return value
        if isinstance(value, enum.Enum):
            return self.copy(value.value, path, depth + 1)
        if type(value) is str:
            self.charge(4 * len(value))
            return value
        if isinstance(value, np.generic) and value.dtype.kind in "biuf":
            return self.copy(value.item(), path, depth + 1)
        if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
            self.charge(value.nbytes + 256 * value.ndim)
            return _Array(value.dtype.str, value.shape, value.tobytes(order="C"))
        if depth < MAX_DEPTH:
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                fields = dataclasses.fields(value)
                self.charge(256 * len(fields))
                return {
                    f.name: self.copy(getattr(value, f.name), f"{path}.{f.name}", depth + 1)
                    for f in fields
                    if not f.name.startswith("_")
                }
            if type(value) is dict and all(type(key) is str for key in value):
                self.charge(256 * len(value))
                return {
                    self.copy(key, path, depth + 1): self.copy(item, f"{path}.{key}", depth + 1)
                    for key, item in value.items()
                }
            if type(value) in (list, tuple, set, frozenset):
                # Token/block lists dominate host trace size. Keep integer
                # sequences compact rather than creating one object per token.
                if type(value) in (list, tuple) and len(value) >= COMPACT_SEQUENCE_MIN:
                    if all(type(item) is int for item in value):
                        self.charge(8 * len(value) + 256)
                        try:
                            array = np.asarray(value, dtype=np.int64)
                        except (OverflowError, ValueError):
                            pass  # Preserve arbitrary-size integers via the scalar path.
                        else:
                            return _Array(array.dtype.str, array.shape, array.tobytes())
                self.charge(64 * len(value))
                return tuple(self.copy(item, f"{path}[]", depth + 1) for item in value)
        # No repr(), __dict__, Tensor.cpu(), numpy(), or device scalar extraction.
        if len(self.omissions) < MAX_OMISSIONS:
            self.omissions.append(path[:256])
        return {"unsupported_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _encode(value: Any) -> Any:
    if isinstance(value, _Array):
        return {
            "encoding": "numpy-base64",
            "dtype": value.dtype,
            "shape": value.shape,
            "data": base64.b64encode(value.data).decode("ascii"),
        }
    raise TypeError(type(value).__name__)


class FlightRecorder:
    """Bounded recent host history, exported on a violation or explicit trigger.

    This is evidence collection, not exact device replay or KV checkpointing.
    A caller must not mutate a captured payload; capture takes its own copy.
    """

    def __init__(self, config: DfxConfig, identity: dict[str, Any]):
        self.config = config
        self.worker_id = uuid.uuid4().hex
        self.identity = {
            **identity,
            "run_id": config.run_id or None,
            "worker_epoch": self.worker_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        }
        self.execution_id = 0
        self._capture_lock = threading.RLock()
        self._contention_drops = 0
        self._generations: OrderedDict[str, int] = OrderedDict()
        self._next_generation = 0
        self._reset_generations = False
        self._history_gap = False
        self._output_records = 0
        self._findings = 0
        self._host_check_skips = 0
        self.device_probe = None
        self._manifest = None
        self._late_incidents: OrderedDict[int, str] = OrderedDict()
        self._ownership = BlockOwnership(config)
        self._ownership_gap = False
        self._detectors = OutputDetectors(config)
        self._sequence = 0
        self._records: deque[tuple[int, dict[str, Any]]] = deque()
        self._bytes = 0
        self._evicted = 0
        self._oversized = 0
        self._capture_errors = 0
        self._dump_rejected = 0
        self._dump_errors = 0
        self._submitted = 0
        self._completed = 0
        self._closed = False
        self._queue: queue.Queue = queue.Queue(maxsize=DUMP_QUEUE_SIZE)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._write_loop, name="ascend-dfx-writer", daemon=True)
        # Path validation and directory creation are initialization-only work.
        self.directory = Path(config.output_dir).expanduser().resolve() / f"worker-{self.worker_id}"
        self.directory.mkdir(parents=True, mode=0o700)
        self._thread.start()

    @classmethod
    def create(cls, config: DfxConfig, identity: dict[str, Any]) -> "FlightRecorder | None":
        if not config.enabled:
            return None
        try:
            return cls(config, identity)
        except Exception:
            logger.exception("DFX initialization failed; serving continues without flight recording")
            return None

    def stats(self) -> dict[str, int]:
        return {
            "records": len(self._records),
            "accounted_buffer_bytes": self._bytes,
            "evicted_records": self._evicted,
            "oversized_records": self._oversized,
            "capture_errors": self._capture_errors,
            "dump_rejected": self._dump_rejected,
            "dump_errors": self._dump_errors,
            "dumps_submitted": self._submitted,
            "dumps_completed": self._completed,
            "contention_drops": self._contention_drops,
            "output_records": self._output_records,
            "detector_findings": self._findings,
            "host_check_skips": self._host_check_skips,
            "detector_history_evictions": self._detectors.evictions,
            "out_of_order_outputs": self._detectors.out_of_order,
            "outputs_without_nan_counts": self._detectors.missing_nan_counts,
            "outputs_without_logprobs": self._detectors.missing_logprobs,
            **(
                {"device_" + key: value for key, value in self.device_probe.counters.items()}
                if self.device_probe is not None
                else {}
            ),
        }

    @_nonblocking
    def record_event(self, kind, payload, violations=(), source_execution_id=None):
        if self._closed:
            return
        try:
            self._append(kind, payload, violations, source_execution_id)
            if kind == "device_snapshot" and source_execution_id in self._late_incidents and not violations:
                reason = self._late_incidents.pop(source_execution_id)
                self.trigger("late_device_evidence:" + reason, source_execution_id)
        except Exception:
            self._capture_failed()

    def bind_device(self, runner, caches, cache_layout="NHD"):
        try:
            model = runner.model_config
            values = {
                "model": model.model,
                "revision": model.revision,
                "dtype": str(model.dtype),
                "quantization": model.quantization,
                "hf_config": model.hf_config.to_dict(),
                "parallel": {
                    name: getattr(runner.parallel_config, name, None)
                    for name in (
                        "tensor_parallel_size",
                        "pipeline_parallel_size",
                        "data_parallel_size",
                        "decode_context_parallel_size",
                        "prefill_context_parallel_size",
                    )
                },
            }
            snapshot = _Snapshot(self.config.max_record_bytes)
            self._manifest = snapshot.copy(values)
        except Exception:
            self._capture_failed()
        if not self.config.device_capture_interval:
            return
        try:
            from vllm_ascend.dfx.device_probe import DeviceProbe

            self.device_probe = DeviceProbe(self, runner, caches, cache_layout=cache_layout)
        except Exception:
            self._capture_failed()

    @_nonblocking
    def record_lifecycle(self, runner, scheduler_output):
        if self._closed or not self.config.track_block_ownership:
            return
        try:
            if self._ownership_gap:
                self._ownership.reset()
                self._ownership_gap = False
            cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
            allowed = set(getattr(cached, "resumed_req_ids", ()))
            allowed.update(request.req_id for request in scheduler_output.scheduled_new_reqs)
            payload, violations = self._ownership.observe(
                runner.requests,
                allow_rollback=allowed,
                check_progress=not runner.use_async_scheduling and runner.speculative_config is None,
            )
            payload["finished_req_ids"] = scheduler_output.finished_req_ids
            payload["preempted_req_ids"] = getattr(scheduler_output, "preempted_req_ids", None)
            payload["resumed_req_ids"] = allowed
            self._append("block_ownership", payload, tuple(violations))
        except Exception:
            self._ownership_gap = True
            self._capture_failed()

    def begin_device(self, runner, metadata, input_ids, positions, **kwargs):
        if self.device_probe is None:
            return None
        try:
            return self.device_probe.begin(runner, metadata, input_ids, positions, **kwargs)
        except Exception:
            self._capture_failed()
            return None

    def audit_transfer(self, connector, caches, kv_config):
        if not self.config.audit_transfer_registration:
            return
        try:
            # Explicit adapter for the local Mooncake hybrid worker contract.
            # Other connectors must not silently inherit these assumptions.
            worker = getattr(connector, "connector_worker", None)
            if (
                worker is None
                or type(worker).__module__ != "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector"
            ):
                self.record_event("transfer_registration", {"complete": False, "reason": "unsupported_connector"})
                return
            layer_groups = {
                layer: group_id
                for group_id, group in enumerate(kv_config.kv_cache_groups)
                for layer in group.layer_names
            }
            views = {}
            for layer, cache in caches.items():
                tensors = cache if isinstance(cache, (tuple, list)) else (cache,)
                for tensor in tensors:
                    ptr = tensor.data_ptr()
                    storage = tensor.untyped_storage()
                    view = views.setdefault(
                        ptr,
                        {
                            "blocks": tensor.shape[0],
                            "stride_bytes": tensor.stride(0) * tensor.element_size(),
                            "storage_begin": storage.data_ptr(),
                            "storage_end": storage.data_ptr() + storage.nbytes(),
                            "groups": [],
                        },
                    )
                    view["groups"].append(layer_groups[layer])
            bases = worker.kv_caches_base_addr
            lengths = worker.block_len_per_addr
            strides = worker.block_stride_per_addr or lengths
            groups = worker.addr_group_idx if worker.use_compress else None
            if (
                len(bases) != len(lengths)
                or len(strides) != len(bases)
                or (groups is not None and len(groups) != len(bases))
            ):
                self.record_event("transfer_registration", {"complete": False}, ("transfer_descriptor_count_mismatch",))
                return
            entries = [
                dict(base=base, length=length, stride=stride, groups=groups[index] if groups is not None else None)
                for index, (base, length, stride) in enumerate(zip(bases, lengths, strides))
            ]
            violations = check_transfer_registration(entries, views, worker.num_blocks)
            self.record_event(
                "transfer_registration",
                {
                    "entries": entries,
                    "views": {str(key): value for key, value in views.items()},
                    "complete": True,
                    "scope": "local_registration_not_remote_transfer",
                },
                tuple(violations),
            )
        except Exception:
            self._capture_failed()

    def end_device(self, frame, logits=None):
        if self.device_probe is None or frame is None:
            return
        try:
            self.device_probe.end(frame, logits)
        except Exception:
            self._capture_failed()

    def _append(
        self, kind: str, payload: Any, violations: tuple[str, ...] = (), source_execution_id: int | None = None
    ) -> None:
        snapshot = _Snapshot(self.config.max_record_bytes)
        try:
            captured = snapshot.copy(payload)
            complete = not snapshot.omissions
            size = snapshot.used
        except _BudgetExceeded:
            self._oversized += 1
            captured = None
            complete = False
            size = RECORD_OVERHEAD
            snapshot.omissions = ["record_budget_exceeded"]
        self._sequence += 1
        record = {
            "sequence": self._sequence,
            "execution_id": self.execution_id if source_execution_id is None else source_execution_id,
            "capture_execution_id": self.execution_id,
            "monotonic_ns": time.monotonic_ns(),
            "kind": kind,
            "complete": complete,
            "omissions": snapshot.omissions,
            "violations": violations,
            "payload": captured,
        }
        while self._records and (
            len(self._records) >= self.config.max_records or self._bytes + size > self.config.max_buffer_bytes
        ):
            old_size, _ = self._records.popleft()
            self._bytes -= old_size
            self._evicted += 1
        self._records.append((size, record))
        self._bytes += size
        if violations and self.config.dump_on_violation:
            self.trigger(",".join(violations), record["execution_id"])

    def _capture_failed(self) -> None:
        self._capture_errors += 1
        if self._capture_errors == 1:
            logger.exception("DFX capture failed; serving continues (subsequent failures counted)")

    def record_schedule(self, scheduler_output: Any) -> None:
        # Advance even when capture is dropped, so delayed results never acquire
        # the ID of an unrelated execution. Only the runner calls this method.
        self.execution_id += 1
        self._record_schedule(scheduler_output)

    def record_sync_execution(self):
        self.execution_id += 1
        self.record_event("existing_sync_execution", {"added_model_invocation": False})

    @_nonblocking
    def _record_schedule(self, scheduler_output: Any) -> None:
        if self._closed:
            return
        try:
            for req_id in getattr(scheduler_output, "finished_req_ids", ()):
                self._generations.pop(req_id, None)
            for request in getattr(scheduler_output, "scheduled_new_reqs", ()):
                self._generations.pop(request.req_id, None)
            counts = scheduler_output.num_scheduled_tokens
            violations = []
            if sum(counts.values()) != scheduler_output.total_num_scheduled_tokens:
                violations.append("schedule_total_mismatch")
            if any(count < 0 for count in counts.values()):
                violations.append("negative_scheduled_count")
            self._append("scheduler_received", scheduler_output, tuple(violations))
        except Exception:
            self._capture_failed()

    @_nonblocking
    def record_batch(self, runner: Any, scheduler_output: Any, positions: np.ndarray, counts: np.ndarray) -> None:
        if self._closed:
            return
        try:
            batch = runner.input_batch
            req_ids = batch.req_ids[: batch.num_reqs]
            total = scheduler_output.total_num_scheduled_tokens
            violations = []
            if len(req_ids) != len(set(req_ids)):
                violations.append("duplicate_batch_request")
            expected = scheduler_output.num_scheduled_tokens
            if len(counts) != len(req_ids) or set(req_ids) != set(expected):
                violations.append("batch_request_set_mismatch")
            elif any(int(count) != expected[req_id] for req_id, count in zip(req_ids, counts)):
                violations.append("batch_count_mapping_mismatch")
            if self.config.detect_host_kv and not violations:
                violations.extend(self._check_host_plan(runner, req_ids, counts, positions))
            self._append(
                "host_batch_prepared",
                {
                    "req_ids": req_ids,
                    "num_scheduled_tokens": counts,
                    "input_token_ids": runner.input_ids.cpu[:total].numpy(),
                    "logical_positions": positions,
                    "num_computed_tokens": batch.num_computed_tokens_cpu[: batch.num_reqs],
                    "block_ids": {req_id: runner.requests[req_id].block_ids for req_id in req_ids},
                    "actual_device_inputs_verified": False,
                    "async_scheduling": runner.use_async_scheduling,
                    "speculative": runner.speculative_config is not None,
                    "context_parallel": runner.use_dcp,
                },
                tuple(violations),
            )
        except Exception:
            self._capture_failed()

    def _check_host_plan(self, runner, req_ids, counts, positions):
        if (
            runner.use_async_scheduling
            or runner.speculative_config is not None
            or runner.use_dcp
            or getattr(runner, "pcp_size", 1) != 1
            or getattr(runner, "uses_mrope", False)
            or getattr(runner, "use_compress", False)
            or len(positions) > self.config.max_checked_tokens
        ):
            self._host_check_skips += 1
            return []
        batch = runner.input_batch
        violations = check_positions(positions, counts, batch.num_computed_tokens_cpu[: len(req_ids)])
        if violations:
            return violations
        for group, table in enumerate(batch.block_table.block_tables):
            if table.is_mamba_group or table.is_circular or table.use_hybrid_blocks or table.dcp_world_size != 1:
                self._host_check_skips += 1
                continue
            violations.extend(
                check_block_plan(
                    req_ids,
                    counts,
                    positions,
                    table.get_numpy_array(),
                    table.num_blocks_per_row,
                    table.block_size,
                    {req_id: runner.requests[req_id].block_ids[group] for req_id in req_ids},
                )
            )
        return violations

    @_nonblocking
    def output_context(self, req_ids: list[str], vocab_size: int, spec_tokens=None) -> OutputContext | None:
        if self._closed or not self.config.detect_outputs:
            return None
        try:
            if self._reset_generations:
                self._generations.clear()
                self._reset_generations = False
            requests = []
            for req_id in req_ids:
                generation = self._generations.pop(req_id, None)
                if generation is None:
                    self._next_generation += 1
                    generation = self._next_generation
                self._generations[req_id] = generation
                requests.append((req_id, generation))
            while len(self._generations) > self.config.max_tracked_requests:
                self._generations.popitem(last=False)
            proposed = tuple((req_id, len(tokens)) for req_id, tokens in (spec_tokens or {}).items())
            return OutputContext(self.execution_id, tuple(requests), vocab_size, proposed)
        except Exception:
            self._capture_failed()
            return None

    @_nonblocking
    def record_output(self, output: Any, context: OutputContext | None) -> None:
        if self._closed or context is None:
            return
        try:
            if self._history_gap:
                self._detectors.reset_history()
                self._history_gap = False
            findings = self._detectors.check(output, context)
            self._output_records += 1
            self._findings += len(findings)
            self._append(
                "resolved_output",
                {
                    "req_ids": output.req_ids,
                    "request_generations": dict(context.requests),
                    "proposed_token_counts": dict(context.proposed_tokens),
                    "sampled_token_ids": output.sampled_token_ids,
                    "num_nans_in_logits": output.num_nans_in_logits,
                    "findings": findings,
                },
                tuple(finding["code"] for finding in findings if finding["severity"] != "coverage_gap"),
                context.execution_id,
            )
        except Exception:
            self._history_gap = True
            self._capture_failed()

    @_nonblocking
    def trigger(self, reason: str, source_execution_id: int | None = None) -> dict[str, Any]:
        """Nonblocking export request; accepted does not mean persisted."""
        if self._closed:
            return {"status": "closed"}
        if self._submitted >= self.config.max_dumps:
            self._dump_rejected += 1
            return {"status": "quota_exceeded", "stats": self.stats()}
        try:
            dump_id = self._submitted + 1
            if source_execution_id is not None and self.device_probe is not None and not reason.startswith("late_"):
                self._late_incidents[source_execution_id] = str(reason)[:256]
                while len(self._late_incidents) > self.config.max_records:
                    self._late_incidents.popitem(last=False)
            records = tuple(record for _, record in self._records)
            device_payloads = [
                record["payload"]
                for record in records
                if record["kind"] == "device_snapshot" and record["payload"] is not None
            ]
            tensor_names = {name for payload in device_payloads for name in payload.get("tensors", {})}
            bundle = {
                "schema": SCHEMA,
                "identity": self.identity,
                "manifest": self._manifest,
                "trigger": {
                    "reason": str(reason)[:512],
                    "source_execution_id": source_execution_id,
                    "capture_execution_id": self.execution_id,
                    "source_schedule_retained": any(
                        record["kind"] == "scheduler_received" and record["execution_id"] == source_execution_id
                        for record in records
                    )
                    if source_execution_id is not None
                    else None,
                    "wall_time_ns": time.time_ns(),
                },
                "coverage": {
                    "scheduler": "worker_received_not_scheduler_created",
                    "batch": "host_preparation_not_final_device_input",
                    "device_capture_enabled": self.device_probe is not None,
                    "kv_contents": "sampled_pages" if any(name.startswith("kv.") for name in tensor_names) else False,
                    "device_metadata": "runner_boundary_sample"
                    if any(name.startswith("metadata.") for name in tensor_names)
                    else False,
                    "sampled_outputs": self._output_records > 0,
                    "output_detectors_enabled": self.config.detect_outputs,
                    "raw_logits_finite_checked": "sampled_rows_and_vocab"
                    if "raw_logits_finite" in tensor_names
                    else False,
                    "dummy_forward": any(payload.get("kind") == "sync_only" for payload in device_payloads),
                    "extra_dummy_forward": False,
                    "exact_replay": False,
                    "history_may_start_mid_request": True,
                },
                "stats": self.stats(),
                "records": records,
            }
            self._queue.put_nowait((dump_id, bundle))
            self._submitted += 1
            return {"status": "queued", "path": str(self.directory / f"incident-{dump_id}.json")}
        except queue.Full:
            self._dump_rejected += 1
            return {"status": "busy"}
        except Exception:
            self._capture_failed()
            return {"status": "failed"}

    def _write_loop(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            if self.device_probe is not None:
                self.device_probe.poll()
            try:
                pending_device = self.device_probe is not None and self.device_probe.has_pending()
                dump_id, bundle = self._queue.get(timeout=0.01 if pending_device else 0.1)
            except queue.Empty:
                continue
            temporary = self.directory / f"incident-{dump_id}.partial"
            try:
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as output:
                    json.dump(bundle, output, default=_encode, ensure_ascii=True)
                temporary.replace(self.directory / f"incident-{dump_id}.json")
                self._completed += 1
            except Exception:
                self._dump_errors += 1
                logger.exception("DFX export failed; incomplete incident remains marked .partial")
            finally:
                self._queue.task_done()

    def close(self, timeout: float = 1.0) -> None:
        """Best-effort bounded shutdown; not called on an inference step."""
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=timeout)


class V2FlightRecorder(FlightRecorder):
    """MRV2 adapter: host schedule plans are never presented as device tokens.

    The bounded block mirror is scheduler-derived, not allocator authority. Only
    the execution thread touches it; the writer receives owned event snapshots.
    """

    def __init__(self, config, identity):
        self.requests = {}
        self.schedule = None
        self.frame = None
        self.active_execution = False
        self.sync_only = False
        self.graph_mode = "NONE"
        super().__init__(config, identity)

    def start(self, scheduler_output, *, sync_only=False):
        self.active_execution = True
        self.sync_only = sync_only
        self.schedule = scheduler_output
        self.graph_mode = "UNKNOWN"
        if sync_only:
            self.record_sync_execution()
            return
        self.record_schedule(scheduler_output)
        try:
            for req_id in scheduler_output.finished_req_ids | set(
                getattr(scheduler_output, "preempted_req_ids", ()) or ()
            ):
                self.requests.pop(req_id, None)
            retained_blocks = sum(len(group) for request in self.requests.values() for group in request.block_ids)
            for req in scheduler_output.scheduled_new_reqs:
                old = self.requests.get(req.req_id)
                retained_blocks += sum(map(len, req.block_ids)) - (sum(map(len, old.block_ids)) if old else 0)
                if retained_blocks > self.config.max_tracked_blocks or (
                    old is None and len(self.requests) >= self.config.max_tracked_requests
                ):
                    raise ValueError("block history budget exceeded")
                self.requests[req.req_id] = SimpleNamespace(
                    block_ids=tuple(tuple(group) for group in req.block_ids),
                    num_computed_tokens=req.num_computed_tokens,
                )
            cached = scheduler_output.scheduled_cached_reqs
            for index, req_id in enumerate(cached.req_ids):
                blocks = cached.new_block_ids[index]
                request = self.requests.get(req_id)
                if req_id in cached.resumed_req_ids and blocks is not None:
                    retained_blocks -= sum(map(len, request.block_ids)) if request else 0
                    if request is None and len(self.requests) >= self.config.max_tracked_requests:
                        raise ValueError("request history budget exceeded")
                    request = SimpleNamespace(block_ids=tuple(() for _ in blocks), num_computed_tokens=0)
                    self.requests[req_id] = request
                if request is None:
                    self.record_event("v2_coverage_gap", {"reason": "missing_block_history", "req_id": req_id})
                    continue
                if blocks is not None:
                    retained_blocks += sum(map(len, blocks))
                    if retained_blocks > self.config.max_tracked_blocks:
                        raise ValueError("block history budget exceeded")
                    if len(blocks) != len(request.block_ids):
                        raise ValueError("KV group count changed")
                    request.block_ids = tuple(old + tuple(new) for old, new in zip(request.block_ids, blocks))
                request.num_computed_tokens = cached.num_computed_tokens[index]
        except Exception as error:
            self.requests.clear()
            self._ownership_gap = True
            self.record_event("v2_coverage_gap", {"reason": type(error).__name__, "scope": "block_history_reset"})

    def batch(self, runner, batch, batch_desc):
        try:
            self.graph_mode = str(batch_desc.cg_mode)
            req_ids = batch.req_ids[: batch.num_reqs]
            counts = batch.num_scheduled_tokens
            expected = self.schedule.num_scheduled_tokens
            findings = []
            # PCP local rows and adaptive verification legitimately differ from
            # scheduler upper bounds; retain both, do not report false errors.
            plain = runner.pcp_manager is None and getattr(runner, "adaptive_verification", None) is None
            if plain:
                if len(req_ids) != len(set(req_ids)):
                    findings.append("duplicate_batch_request")
                if set(req_ids) != set(expected) or len(counts) != len(req_ids):
                    findings.append("batch_request_set_mismatch")
                elif any(int(count) != expected[req] for req, count in zip(req_ids, counts)):
                    findings.append("batch_count_mapping_mismatch")
            self.record_event(
                "host_batch_prepared",
                dict(
                    req_ids=req_ids,
                    idx_mapping=batch.idx_mapping_np,
                    num_scheduled_tokens=counts,
                    query_start_loc=batch.query_start_loc_np,
                    num_computed_tokens=batch.num_computed_tokens_np,
                    block_ids={req: self.requests[req].block_ids for req in req_ids if req in self.requests},
                    input_token_ids=None,
                    actual_device_inputs_verified=False,
                    token_source="device_assembled_use_device_snapshot",
                    computed_tokens_semantics="host_upper_bound",
                    graph_mode=self.graph_mode,
                    context_parallel=runner.pcp_manager is not None or runner.use_dcp,
                ),
                findings,
            )
            if self.config.track_block_ownership:
                self.record_lifecycle(
                    SimpleNamespace(requests=self.requests, use_async_scheduling=True, speculative_config=None),
                    self.schedule,
                )
        except Exception:
            self._capture_failed()

    def prepare_device(self, runner, batch, block_tables, slot_mappings):
        if not self.active_execution or self.device_probe is None:
            return
        try:
            self.frame = self.device_probe.begin_v2(
                runner,
                batch,
                block_tables,
                slot_mappings,
                self.requests,
                kind="sync_only" if self.sync_only else "real",
                graph_mode=self.graph_mode,
            )
        except Exception:
            self._capture_failed()

    def finish(self):
        self.end_device(self.frame)
        self.frame = None
        self.active_execution = False

    def bind_v2(self, runner, cache_layout="NHD"):
        try:
            context = runner.compilation_config.static_forward_context
            caches = {
                layer: context[layer].kv_cache
                for group in runner.kv_cache_config.kv_cache_groups
                for layer in group.layer_names
                if layer in context
            }
            self.bind_device(runner, caches, cache_layout=cache_layout)
            self.record_event(
                "v2_coverage",
                {
                    "metadata": "prepare_attn_boundary_not_backend_private",
                    "raw_logits": "not_captured_use_existing_output_nan_counts",
                    "request_history_reference": "unavailable",
                    "host_kv_plan": "not_checked_device_assembled_inputs",
                },
            )
            if self.config.audit_transfer_registration:
                self.record_event("v2_coverage_gap", {"reason": "transfer_registration_adapter_not_supported"})
        except Exception:
            self._capture_failed()
