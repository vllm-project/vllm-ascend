# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-local TP transfer timing; no model or connector classes are patched globally."""

from __future__ import annotations

import os
import threading
import time
from functools import wraps
from typing import Any

from tests.e2e.pull_request.one_card.ascendstore_v1_performance import AscendStoreBenchmarkWorkerExtension


class AscendStoreTPBenchmarkWorkerExtension(AscendStoreBenchmarkWorkerExtension):
    def start_ascendstore_tp_timing(self) -> dict[str, Any]:
        assert not hasattr(self, "_ascendstore_tp_timing"), "TP timing is already active"
        timing = _TPTransferTiming(self)
        self._ascendstore_tp_timing = timing
        try:
            timing.install()
        except BaseException:
            timing.restore()
            del self._ascendstore_tp_timing
            raise
        return timing.identity

    def stop_ascendstore_tp_timing(self) -> dict[str, Any]:
        """Wait for producer publication outside request timing, then remove every instance hook."""
        timing = self._ascendstore_tp_timing
        try:
            drain_start = time.perf_counter_ns()
            timing.drain_stores()
            drain_ms = (time.perf_counter_ns() - drain_start) / 1_000_000
            return {**timing.identity, "store_drain_ms": drain_ms, "records": timing.records()}
        finally:
            timing.restore()
            del self._ascendstore_tp_timing


class _TPTransferTiming:
    def __init__(self, worker: Any) -> None:
        self._worker = worker
        self._store_worker: Any = None
        self._production = False
        self._producer = False
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._restorations: list[tuple[Any, str, bool, Any]] = []
        self.identity: dict[str, Any] = {}

    def install(self) -> None:
        # Worker-only imports keep the pytest/CPU process independent of the NPU runtime.
        import torch
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        connector = get_kv_transfer_group()
        config = self._worker.vllm_config
        role = config.kv_transfer_config.kv_role
        assert role in ("kv_producer", "kv_consumer"), "The TP benchmark requires separate P/D roles"
        self._producer = role == "kv_producer"
        operation = "store" if self._producer else "load"
        kind = (type(connector).__module__, type(connector).__name__)
        if kind == (
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
            "AscendStoreConnector",
        ):
            self._production = True
            worker = connector.connector_worker
            assert worker is not None and worker.tp_mismatch, "Production did not enable TP mismatch"
            backend = worker.m_store
            rank, size, slices = worker.tp_rank, worker.tp_size, worker.num_sub_keys
            self._patch(worker, "_build_tp_mismatch_keys_and_addrs", f"{operation}_arguments")
            self._patch(backend, "put" if self._producer else "get", f"backend_{operation}", buffers=True)
        elif kind == (
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector",
            "AscendStoreV1Connector",
        ):
            worker = connector.worker
            assert worker is not None and worker._topology.tp_partition.tp_mismatch, "v1 did not enable TP mismatch"
            backend = worker._resources.backend
            topology = worker._topology
            rank, size = topology.tp_rank, topology.tp_size
            slices = topology.tp_partition.key_slices_per_rank
            if self._producer:
                self._patch(worker, "_build_store_candidates", "store_candidates")
                self._patch(worker, "_materialize_store_candidates", "store_plan")
            else:
                self._patch(worker, "_build_load_batch", "load_plan")
            self._patch(worker, "_materialize_concrete_bulk_arguments", f"{operation}_arguments")
            self._patch(backend, operation, f"backend_{operation}", buffers=True)
        else:
            raise AssertionError(f"Unsupported TP benchmark connector: {kind!r}")
        self._store_worker = worker
        self._patch(backend, "exists", "backend_exists")
        self.identity = {
            "implementation": "production" if self._production else "v1",
            "role": role,
            "pid": os.getpid(),
            "tp_rank": int(rank),
            "tp_size": int(size),
            "key_slices_per_rank": int(slices),
            "local_device": int(torch.npu.current_device()),
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "clock": "perf_counter_ns",
        }

    def _patch(self, owner: Any, name: str, stage: str, *, buffers: bool = False) -> None:
        original = getattr(owner, name)
        assert callable(original), f"TP timing requires {name}()"
        attributes = vars(owner)
        self._restorations.append((owner, name, name in attributes, attributes.get(name)))

        @wraps(original)
        def timed(*args: Any, **kwargs: Any) -> Any:
            record: dict[str, Any] = {"stage": stage, "thread_id": threading.get_ident()}
            start_ns = time.perf_counter_ns()
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                record["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                end_ns = time.perf_counter_ns()
                record.update(start_ns=start_ns, end_ns=end_ns, host_ms=(end_ns - start_ns) / 1_000_000)
                with self._lock:
                    self._records.append(record)
            # Count descriptors after timing the Backend. These observations still affect the diagnostic HTTP run.
            if buffers:
                keys, addresses, sizes = args[:3]
                assert len(keys) == len(addresses) == len(sizes), "Backend arguments lost key alignment"
                record.update(
                    keys=list(keys),
                    range_count=sum(len(row) for row in addresses),
                    byte_count=sum(sum(row) for row in sizes),
                    result_codes=None if result is None else [int(code) for code in result],
                )
            return result

        setattr(owner, name, timed)

    def drain_stores(self) -> None:
        if not self._producer:
            return
        if self._production:
            sender = self._store_worker.kv_send_thread
            assert sender is not None
            with sender.done_task_lock:
                request_ids = set(sender.stored_requests)
            self._store_worker.handle_released_saves(request_ids)
            with sender.done_task_lock:
                assert not sender.stored_requests, "Producer Store did not drain"
        else:
            self._store_worker.fence_previous_store()
            assert self._store_worker._pending_store_batch is None, "Producer Store source release is unknown"

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(self._records, key=lambda record: record["start_ns"])

    def restore(self) -> None:
        while self._restorations:
            owner, name, existed, value = self._restorations.pop()
            if existed:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
