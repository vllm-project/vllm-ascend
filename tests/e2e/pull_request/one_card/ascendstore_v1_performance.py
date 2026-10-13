# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Untimed cache control for our ordinary AscendStore serve benchmark."""

import time
from typing import Any


class AscendStoreBenchmarkWorkerExtension:
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
