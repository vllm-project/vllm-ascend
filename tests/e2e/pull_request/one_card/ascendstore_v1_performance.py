# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache control and scoped diagnostic timing for our ordinary serve benchmark."""

import math
import os
import threading
import time
from collections.abc import Callable
from functools import wraps
from typing import Any


class AscendStoreBenchmarkWorkerExtension:
    def start_ascendstore_stage_timing(self) -> dict[str, Any]:
        """Install instance-local hooks after reset, only for the separate C1 cold diagnostic."""
        assert not hasattr(self, "_ascendstore_stage_timing"), "Stage timing is already active"
        timing = _StageTimingCollector(self)
        self._ascendstore_stage_timing = timing
        try:
            timing.install()
        except BaseException:
            try:
                timing.restore()
            finally:
                del self._ascendstore_stage_timing
            raise
        return {"clock": "perf_counter_ns", "pid": timing.pid}

    def stop_ascendstore_stage_timing(self) -> dict[str, Any]:
        """Drain admitted Stores, stop timing, resolve device events once, and always remove the hooks."""
        timing = getattr(self, "_ascendstore_stage_timing", None)
        assert isinstance(timing, _StageTimingCollector), "Stage timing is not active"
        try:
            # The control RPC is serialized after HTTP completion; queued Store jobs still need their tail timed.
            timing.drain_store()
            timing.disable()
            return timing.collect()
        finally:
            try:
                timing.restore()
            finally:
                del self._ascendstore_stage_timing

    def reset_ascendstore_benchmark_pool(self, expected_master_address: str) -> dict[str, Any]:
        """Drain Store and empty only the dedicated loopback master launched by this test."""
        # Resolve Worker-owned resources here, without importing pytest/conftest into the real model Worker.
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        connector = get_kv_transfer_group()
        connector_kind = (type(connector).__module__, type(connector).__name__)
        start = time.perf_counter()
        if connector_kind == (
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
            "AscendStoreConnector",
        ):
            worker = connector.connector_worker
            assert worker is not None
            backend = worker.m_store
            sender = worker.kv_send_thread
            assert sender is not None
            with sender.done_task_lock:
                request_ids = set(sender.stored_requests)
            worker.handle_released_saves(request_ids)
            with sender.done_task_lock:
                assert not sender.stored_requests, "Store must finish before clearing the dedicated pool"
            store = backend.store
        elif connector_kind == (
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector",
            "AscendStoreV1Connector",
        ):
            worker = connector.worker
            assert worker is not None
            backend = worker._resources.backend
            worker.fence_previous_store()
            assert worker._pending_store_batch is None, "Store source release must be confirmed before pool reset"
            store = backend._store
        else:
            raise AssertionError(f"Unsupported connector for the ordinary benchmark: {connector_kind!r}")
        store_drain_ms = (time.perf_counter() - start) * 1000
        assert expected_master_address.startswith("127.0.0.1:"), "Never clear a nonlocal/shared pool"
        assert backend.config.master_server_address == expected_master_address, "Dedicated master address mismatch"
        assert store is not None and callable(getattr(store, "remove_all", None)), "Mooncake remove_all is required"
        start = time.perf_counter()
        try:
            removed_objects = store.remove_all(force=True)
        except TypeError as error:
            raise AssertionError("The benchmark requires Mooncake remove_all(force=True)") from error
        assert type(removed_objects) is int and removed_objects >= 0, "Mooncake pool reset failed"
        # Detect late removable objects; the cold provenance check also rejects readable leftovers.
        assert store.remove_all(force=True) == 0, (
            "Second pool reset failed or found additional objects after Store drain"
        )
        return {
            "removed_objects": removed_objects,
            "store_drain_ms": store_drain_ms,
            "remove_all_ms": (time.perf_counter() - start) * 1000,
        }


class _StageTimingCollector:
    """Own one diagnostic's hooks, asynchronous records, and deferred device events."""

    def __init__(self, worker: Any) -> None:
        self.pid = os.getpid()
        self._worker = worker
        self._enabled = True
        self._lock = threading.Lock()
        self._thread_context = threading.local()
        self._next_step_id = 1
        self._sample_context: dict[str, Any] | None = None
        self._records: list[dict[str, Any]] = []
        self._events: list[tuple[dict[str, Any], Any, Any]] = []
        self._restorations: list[tuple[Any, str, bool, Any]] = []
        self._production_store_contexts: dict[int, dict[str, Any]] = {}
        self._v1_store_contexts: dict[int, dict[str, Any]] = {}
        self._store_worker: Any = None
        self._sender: Any = None
        self._drain_callback: Callable[..., Any] | None = None
        self._production = False
        self._torch: Any = None

    def install(self) -> None:
        # These dependencies belong to the real Worker, not pytest's importing process.
        import torch
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        self._torch = torch
        assert self._worker.vllm_config.model_config.enforce_eager, "Stage timing requires the real eager forward"
        runner = self._worker.model_runner
        connector = get_kv_transfer_group()
        connector_kind = (type(connector).__module__, type(connector).__name__)
        self._patch(self._worker, "execute_model", "execute_model", context_source="step", capture_output=True)
        self._patch(self._worker, "sample_tokens", "sample_tokens", context_source="sample", capture_output=True)
        self._patch(runner.model, "forward", "model_forward", device=True)
        self._patch(runner.model, "compute_logits", "compute_logits", device=True)
        self._patch(connector, "wait_for_save", "store_submit")
        if connector_kind == (
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
            "AscendStoreConnector",
        ):
            self._production = True
            self._store_worker = connector.connector_worker
            assert self._store_worker is not None
            self._sender = self._store_worker.kv_send_thread
            assert self._sender is not None
            self._drain_callback = self._store_worker.handle_released_saves
            self._patch(self._store_worker, "handle_released_saves", "store_fence")
            self._patch(self._sender, "add_save_batch", "store_queue_submit", context_source="production_submit")
            self._patch(self._sender, "_handle_stored_request", "store_operation", context_source="production_store")
            self._patch(self._store_worker.m_store, "put", "backend_store")
        elif connector_kind == (
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector",
            "AscendStoreV1Connector",
        ):
            self._store_worker = connector.worker
            assert self._store_worker is not None
            self._drain_callback = self._store_worker.fence_previous_store
            timeline = self._store_worker._asynchronous_store_timeline
            assert timeline is not None, "The ordinary diagnostic requires a Bulk Store timeline"
            self._patch(self._store_worker, "fence_previous_store", "store_fence")
            self._patch(timeline, "submit", "store_queue_submit", context_source="v1_submit")
            self._patch(timeline, "_operation", "store_operation", context_source="v1_store")
            self._patch(self._store_worker._resources.backend, "store", "backend_store")
        else:
            raise AssertionError(f"Unsupported connector for stage timing: {connector_kind!r}")

    def _patch(
        self,
        owner: Any,
        name: str,
        stage: str,
        *,
        context_source: str = "current",
        device: bool = False,
        capture_output: bool = False,
        fixed_context: dict[str, Any] | None = None,
    ) -> None:
        original = getattr(owner, name)
        assert callable(original), f"Stage timing requires {name}()"
        own_attributes = vars(owner)
        restoration = (owner, name, name in own_attributes, own_attributes.get(name))

        @wraps(original)
        def timed(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                enabled = self._enabled
            if not enabled:
                return original(*args, **kwargs)
            context = self._call_context(context_source, args, kwargs, fixed_context)
            result = self._invoke(stage, original, args, kwargs, context, device=device)
            if capture_output and callable(getattr(result, "get_output", None)):
                # Output resolution runs on another thread: bind its originating step, not the newest one.
                self._patch(result, "get_output", "output_resolve", context_source="fixed", fixed_context=context)
            return result

        setattr(owner, name, timed)
        with self._lock:
            self._restorations.append(restoration)

    def _call_context(
        self,
        source: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        fixed: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        current = getattr(self._thread_context, "current", None)
        if source == "fixed":
            return fixed
        if source == "sample":
            with self._lock:
                return self._sample_context
        if source == "step":
            scheduler_output = args[0] if args else kwargs["scheduler_output"]
            scheduled_tokens = {
                request_id: int(count) for request_id, count in scheduler_output.num_scheduled_tokens.items()
            }
            with self._lock:
                step_context: dict[str, Any] = {
                    "step_id": self._next_step_id,
                    "request_ids": list(scheduled_tokens),
                    "scheduled_tokens": scheduled_tokens,
                }
                self._next_step_id += 1
                self._sample_context = step_context if scheduled_tokens else None
            return step_context
        if source == "production_submit":
            requests = args[0] if args else kwargs["requests"]
            with self._lock:
                for request in requests:
                    submission_context = self._owned_store_context(current, [request.req_id], [])
                    self._production_store_contexts[id(request)] = submission_context
            return current
        if source == "production_store":
            request = args[0] if args else kwargs["req_meta"]
            with self._lock:
                store_context = self._production_store_contexts.pop(id(request), None)
            return store_context or self._owned_store_context(None, [request.req_id], [])
        if source in ("v1_submit", "v1_store"):
            commands = args[0] if args else kwargs["commands"]
            request_ids = list(dict.fromkeys(command.request_id for command in commands))
            job_ids = [command.store_job_id for command in commands if command.store_job_id is not None]
            with self._lock:
                if source == "v1_submit":
                    submission_context = self._owned_store_context(current, request_ids, job_ids)
                    for job_id in job_ids:
                        self._v1_store_contexts[job_id] = submission_context
                    return submission_context
                contexts = [self._v1_store_contexts.pop(job_id, None) for job_id in job_ids]
            # Whole-step submissions normally have one owner. Never infer ownership from another active thread.
            known = [context for context in contexts if context is not None]
            owner = (
                known[0]
                if len(known) == len(commands) and known and len({context["step_id"] for context in known}) == 1
                else None
            )
            return self._owned_store_context(owner, request_ids, job_ids)
        return current

    @staticmethod
    def _owned_store_context(
        context: dict[str, Any] | None, request_ids: list[str], store_job_ids: list[int]
    ) -> dict[str, Any]:
        return {
            "step_id": context["step_id"] if context is not None else None,
            "request_ids": request_ids,
            "scheduled_tokens": dict(context["scheduled_tokens"]) if context is not None else {},
            "store_job_ids": store_job_ids,
        }

    def _invoke(
        self,
        stage: str,
        original: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        context: dict[str, Any] | None,
        *,
        device: bool,
    ) -> Any:
        previous = getattr(self._thread_context, "current", None)
        start_event = end_event = None
        if device:
            start_event = self._torch.npu.Event(enable_timing=True)
            end_event = self._torch.npu.Event(enable_timing=True)
            start_event.record()
        record: dict[str, Any] = {
            "stage": stage,
            "pid": self.pid,
            "thread_id": threading.get_ident(),
            "step_id": context["step_id"] if context is not None else None,
            "request_ids": list(context["request_ids"]) if context is not None else [],
            "scheduled_tokens": dict(context["scheduled_tokens"]) if context is not None else {},
        }
        if context is not None and "store_job_ids" in context:
            record["store_job_ids"] = list(context["store_job_ids"])
        record["start_ns"] = time.perf_counter_ns()
        self._thread_context.current = context
        try:
            return original(*args, **kwargs)
        except BaseException as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["end_ns"] = time.perf_counter_ns()
            record["host_ms"] = (record["end_ns"] - record["start_ns"]) / 1_000_000
            self._thread_context.current = previous
            if end_event is not None:
                end_event.record()
            with self._lock:
                self._records.append(record)
                if start_event is not None:
                    self._events.append((record, start_event, end_event))

    def disable(self) -> None:
        with self._lock:
            self._enabled = False

    def drain_store(self) -> None:
        assert self._drain_callback is not None, "Diagnostic Store drain was not installed"
        if self._production:
            with self._sender.done_task_lock:
                request_ids = set(self._sender.stored_requests)
            self._drain_callback(request_ids)
            with self._sender.done_task_lock:
                assert not self._sender.stored_requests, "Diagnostic Store drain must release its sources"
        else:
            self._drain_callback()
            assert self._store_worker._pending_store_batch is None, "Diagnostic Store source release is unknown"

    def collect(self) -> dict[str, Any]:
        # No per-step sync: complete all main-stream events only after the HTTP diagnostic and Store drain.
        self._torch.npu.synchronize()
        for record, start_event, end_event in self._events:
            device_ms = float(start_event.elapsed_time(end_event))
            assert math.isfinite(device_ms) and device_ms >= 0, "NPU timing returned an invalid duration"
            record["device_ms"] = device_ms
        records = sorted(self._records, key=lambda record: record["start_ns"])
        required_stages = {"execute_model", "sample_tokens", "model_forward"}
        assert required_stages <= {record["stage"] for record in records}, (
            "Diagnostic did not observe real Worker stages"
        )
        assert any(record["stage"] == "model_forward" and record.get("device_ms", 0) > 0 for record in records), (
            "Diagnostic did not collect model-forward NPU timing"
        )
        return {
            "clock": "perf_counter_ns",
            "pid": self.pid,
            "device_timing": "main_stream_elapsed_not_all_kernel_sum",
            "records": records,
        }

    def restore(self) -> None:
        self.disable()
        failures = []
        while self._restorations:
            owner, name, existed, value = self._restorations.pop()
            try:
                if existed:
                    setattr(owner, name, value)
                else:
                    delattr(owner, name)
            except BaseException as error:
                failures.append(error)
        if failures:
            raise RuntimeError(f"Failed to restore {len(failures)} stage timing hooks") from failures[0]
