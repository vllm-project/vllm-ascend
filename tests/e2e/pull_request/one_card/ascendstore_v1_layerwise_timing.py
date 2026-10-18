# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded host timings for one diagnostic v1 Load pass; never inspect KV or synchronize the device."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any

_MAX_RECORDS = 8192
_NS_PER_MS = 1_000_000


@dataclass(frozen=True, slots=True)
class _Interval:
    stage: str
    step: int | None
    layer: int | None
    start: int
    end: int
    failed: bool


class LayerwiseLoadTiming:
    """Observe instance calls under the benchmark's synchronous, single-Worker execution model.

    The main thread owns the active model step; its Load executor must finish before that step returns.
    Lookup runs independently and is deliberately not attributed to a concurrent model step. All aggregation
    happens after the diagnostic requests finish. Nested and parallel stage durations are not additive.
    """

    def __init__(self, worker: Any, backend: Any, lookup_server: Any, model_runner: Any) -> None:
        self._worker = worker
        self._backend = backend
        self._lookup_server = lookup_server
        self._runner = model_runner
        self._restorations: list[tuple[Any, str, bool, Any]] = []
        self._records: list[_Interval] = []
        self._steps: list[dict[str, Any]] = []
        self._active_step: int | None = None
        self._sample_step: int | None = None
        self._layer_context = threading.local()
        self._lock = threading.Lock()
        self._dropped = 0

    def install(self) -> None:
        try:
            self._patch(self._lookup_server, "_lookup", "lookup_service", independent=True)
            self._patch(self._backend, "exists", "lookup_backend_exists", independent=True)
            self._patch(self._worker, "_build_load_batch", "load_batch_prepare")
            self._patch(self._worker, "start_load", "load_start_host")
            self._patch(self._worker, "wait_for_layer_load", "layer_wait_host", layer_argument="layer_name")
            self._patch(self._worker, "_prepare_load_layers", "layer_plan_prepare")
            self._patch(self._worker, "_copy_load_layer", "layer_io_host", layer_argument="layer_id")
            self._patch(self._backend, "start_key_range_load", "session_start")
            self._patch(self._backend, "copy_key_range_load", "range_copy")
            self._patch(self._backend, "finish_key_range_load", "session_end")
            self._replace(self._worker, "_submit_load", self._submit_load_wrapper(self._worker._submit_load))
            self._replace(self._runner, "execute_model", self._execute_wrapper(self._runner.execute_model))
            self._replace(self._runner, "sample_tokens", self._sample_wrapper(self._runner.sample_tokens))
        except BaseException:
            self.restore()
            raise

    def _patch(
        self, owner: Any, name: str, stage: str, *, independent: bool = False, layer_argument: str | None = None
    ) -> None:
        original = getattr(owner, name)

        @wraps(original)
        def timed(*args: Any, **kwargs: Any) -> Any:
            if stage == "lookup_backend_exists" and not getattr(self._layer_context, "lookup", False):
                return original(*args, **kwargs)
            step = None if independent else self._active_step
            if step is None and not independent:
                return original(*args, **kwargs)
            previous_layer = getattr(self._layer_context, "layer", None)
            layer = previous_layer
            if layer_argument is not None:
                value = args[0] if args else kwargs[layer_argument]
                if layer_argument == "layer_name":
                    layer = self._worker._layer_ids_by_name.get(value)
                else:
                    layer = value
            self._layer_context.layer = layer
            previous_lookup = getattr(self._layer_context, "lookup", False)
            if stage == "lookup_service":
                self._layer_context.lookup = True
            try:
                return self._invoke(stage, step, layer, original, args, kwargs)
            finally:
                self._layer_context.layer = previous_layer
                self._layer_context.lookup = previous_lookup

        self._replace(owner, name, timed)

    def _submit_load_wrapper(self, original: Any) -> Any:
        @wraps(original)
        def submit(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            session = self._worker._session
            if session is not None:
                submit_task = session._load_submit

                @wraps(submit_task)
                def submit_timed_task(operation: Any, *task_args: Any, **task_kwargs: Any) -> Any:
                    # Layer tasks receive (layer_id, gate); the finish callback takes no arguments.
                    if not task_args:
                        return submit_task(operation, *task_args, **task_kwargs)

                    @wraps(operation)
                    def run_layer(layer_id: int, gate: Any) -> Any:
                        step = self._active_step
                        if step is None:
                            return operation(layer_id, gate)
                        return self._invoke("layer_task_host", step, layer_id, operation, (layer_id, gate), {})

                    return submit_task(run_layer, *task_args, **task_kwargs)

                # Observe the injected submit function without depending on the task's private name.
                self._replace(session, "_load_submit", submit_timed_task)
            return result

        return submit

    def _execute_wrapper(self, original: Any) -> Any:
        @wraps(original)
        def execute(*args: Any, **kwargs: Any) -> Any:
            scheduler_output = args[0] if args else kwargs["scheduler_output"]
            metadata = scheduler_output.kv_connector_metadata
            commands = metadata.load.commands if metadata is not None else ()
            self._sample_step = None
            if not commands:
                return original(*args, **kwargs)
            step = len(self._steps)
            self._steps.append(
                {
                    "step": step,
                    "request_ids": [command.request_id for command in commands],
                    "scheduled_requests": len(scheduler_output.num_scheduled_tokens),
                    "scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
                }
            )
            self._active_step = step
            checkpoint = len(self._restorations)
            try:
                result = self._invoke("execute_model_host", step, None, original, args, kwargs)
                self._sample_step = step
                return result
            finally:
                # Per-step Session hooks must not retain completed sessions across diagnostic requests.
                self.restore(keep=checkpoint)
                self._active_step = None

        return execute

    def _sample_wrapper(self, original: Any) -> Any:
        @wraps(original)
        def sample(*args: Any, **kwargs: Any) -> Any:
            step, self._sample_step = self._sample_step, None
            if step is None:
                return original(*args, **kwargs)
            return self._invoke("sample_tokens_host", step, None, original, args, kwargs)

        return sample

    def _invoke(self, stage: str, step: int | None, layer: int | None, original: Any, args: Any, kwargs: Any) -> Any:
        failed = False
        start = time.perf_counter_ns()
        try:
            return original(*args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            end = time.perf_counter_ns()
            with self._lock:
                if len(self._records) < _MAX_RECORDS:
                    self._records.append(_Interval(stage, step, layer, start, end, failed))
                else:
                    self._dropped += 1

    def _replace(self, owner: Any, name: str, replacement: Any) -> None:
        attributes = vars(owner)
        self._restorations.append((owner, name, name in attributes, attributes.get(name)))
        setattr(owner, name, replacement)

    def restore(self, *, keep: int = 0) -> None:
        while len(self._restorations) > keep:
            owner, name, existed, value = self._restorations.pop()
            if existed:
                setattr(owner, name, value)
            else:
                delattr(owner, name)

    def report(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)
        steps = []
        for step in self._steps:
            intervals = [record for record in records if record.step == step["step"]]
            stages = _stage_summary(intervals)
            # These two foreground calls are sequential in a synchronous model step. Copies can overlap either.
            foreground = [item for item in intervals if item.stage in ("load_start_host", "layer_wait_host")]
            copies = [item for item in intervals if item.stage == "range_copy"]
            front_ns = _union_ns([(item.start, item.end) for item in foreground])
            copy_ns = _union_ns([(item.start, item.end) for item in copies])
            combined_ns = _union_ns([(item.start, item.end) for item in foreground + copies])
            execute = next((item for item in intervals if item.stage == "execute_model_host"), None)
            sampling = next((item for item in intervals if item.stage == "sample_tokens_host"), None)
            steps.append(
                {
                    **step,
                    "stages": stages,
                    "foreground_load_ms": front_ns / _NS_PER_MS,
                    "range_copy_union_ms": copy_ns / _NS_PER_MS,
                    "copy_overlaps_foreground_ms": (front_ns + copy_ns - combined_ns) / _NS_PER_MS,
                    "execute_outside_load_calls_ms": (
                        None if execute is None else (execute.end - execute.start - front_ns) / _NS_PER_MS
                    ),
                    "execute_to_sample_gap_ms": (
                        None if execute is None or sampling is None else (sampling.start - execute.end) / _NS_PER_MS
                    ),
                }
            )
        layers = sorted({record.layer for record in records if record.layer is not None})
        return {
            "clock": "perf_counter_ns; host wall time only; no added device synchronization",
            "scope": "v1 diagnostic pass only; no KV reads, byte probes, per-key records or hot-path logging",
            "interpretation": {
                "stages": "inclusive host calls; nested and parallel stages must not be summed",
                "layer_task_host": "attention-start gate wait plus layer_io_host and task bookkeeping",
                "layer_io_host": "address/range preparation plus range_copy and completion evidence",
                "range_copy": "Mooncake adapter call, including native copy and return-value normalization",
                "foreground_load_ms": "union of load_start_host and layer_wait_host; already inside execute_model_host",
                "execute_outside_load_calls_ms": "remaining host model call; not isolated NPU compute time",
                "unobserved": "scheduler/client queues, Lookup wire/IPC, async output completion and HTTP delivery",
                "lookup": "service and backend calls are not correlated to individual requests or model steps",
            },
            "record_count": len(records),
            "dropped_records": self._dropped,
            "failed_calls": sum(record.failed for record in records),
            "stages": _stage_summary(records),
            "steps": steps,
            "layers": [
                {"layer": layer, "stages": _stage_summary([record for record in records if record.layer == layer])}
                for layer in layers
            ],
        }


def _stage_summary(records: list[_Interval]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for record in records:
        stage = result.setdefault(record.stage, {"calls": 0, "total_ms": 0.0, "max_ms": 0.0})
        elapsed_ms = (record.end - record.start) / _NS_PER_MS
        stage["calls"] += 1
        stage["total_ms"] += elapsed_ms
        stage["max_ms"] = max(stage["max_ms"], elapsed_ms)
    return result


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    total, previous_end = 0, 0
    for start, end in sorted(intervals):
        total += max(0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    return total
