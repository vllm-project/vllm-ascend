# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker RPC controls for byte accuracy and uninstrumented serving benchmarks."""

import time
from typing import Any

from tests.e2e.common.kv_pool.ascendstore_byte_accuracy import KVByteAccuracy


class AscendStoreBenchmarkWorkerExtension:
    def start_ascendstore_byte_accuracy(self, directory: str) -> dict[str, str]:
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        assert not hasattr(self, "_byte_accuracy"), "Accuracy probe already installed"
        self._drain_stores()
        connector = get_kv_transfer_group()
        production = type(connector).__name__ == "AscendStoreConnector"
        worker = connector.connector_worker if production else connector.worker
        backend = worker.m_store if production else worker._resources.backend
        caches = worker.kv_caches if production else worker._resources.kv_caches
        layerwise = worker.use_layerwise if production else type(worker).__name__ == "KeyRangeLayerwiseWorker"
        self._byte_accuracy = KVByteAccuracy(backend, caches, production, layerwise, directory)
        return {"status": "installed"}

    def finish_ascendstore_byte_accuracy(self, operation: str) -> dict[str, Any]:
        self._drain_stores()
        return self._byte_accuracy.finish(operation)

    def clear_ascendstore_accuracy_kv(self) -> dict[str, str]:
        self._drain_stores()
        self._byte_accuracy.clear_local_kv()
        return {"status": "cleared"}

    def stop_ascendstore_byte_accuracy(self) -> dict[str, str]:
        try:
            self._drain_stores()
        finally:
            self._byte_accuracy.close()
            del self._byte_accuracy
        return {"status": "removed"}

    def drain_ascendstore_benchmark_stores(self) -> dict[str, float]:
        """Wait for admitted Stores before resetting local KV or reading the published external prefixes."""
        _, _, elapsed_ms = self._drain_stores()
        return {"store_drain_ms": elapsed_ms}

    def reset_ascendstore_benchmark_pool(self, expected_master_address: str) -> dict[str, Any]:
        """Drain Store and empty only the dedicated loopback master launched by this test."""
        backend, store, store_drain_ms = self._drain_stores()
        assert expected_master_address.startswith("127.0.0.1:"), "Never clear a nonlocal/shared pool"
        assert backend.config.master_server_address == expected_master_address, "Dedicated master address mismatch"
        assert store is not None and callable(getattr(store, "remove_all", None)), "Mooncake remove_all is required"
        start = time.perf_counter()
        try:
            removed_objects = store.remove_all(force=True)
        except TypeError as error:
            raise AssertionError("The benchmark requires Mooncake remove_all(force=True)") from error
        assert type(removed_objects) is int and removed_objects >= 0, "Mooncake pool reset failed"
        assert store.remove_all(force=True) == 0, "Pool reset found late objects after Store drain"
        return {
            "removed_objects": removed_objects,
            "store_drain_ms": store_drain_ms,
            "remove_all_ms": (time.perf_counter() - start) * 1000,
        }

    def _drain_stores(self) -> tuple[Any, Any, float]:
        # Resolve Worker-owned resources without importing pytest/conftest into the actual model Worker.
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
            if sender is None:
                assert not worker.can_put, "A Store-capable production Worker must own its sender"
            else:
                with sender.done_task_lock:
                    request_ids = set(sender.stored_requests)
                worker.handle_released_saves(request_ids)
                with sender.done_task_lock:
                    assert not sender.stored_requests, "Store must finish before resetting or loading external KV"
            store = backend.store
        elif connector_kind == (
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector",
            "AscendStoreV1Connector",
        ):
            worker = connector.worker
            assert worker is not None
            backend = worker._resources.backend
            worker.fence_previous_store()
            assert worker._pending_store_batch is None, "Store source release must be confirmed before cache control"
            store = backend._store
        else:
            raise AssertionError(f"Unsupported benchmark connector: {connector_kind!r}")
        return backend, store, (time.perf_counter() - start) * 1000
