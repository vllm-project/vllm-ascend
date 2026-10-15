# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-local controls for the single-card native Mooncake KeyRange E2E."""

import time
from typing import Any

from tests.e2e.common.kv_pool.ascendstore_layerwise_probe import LayerwiseRangeProbe
from tests.e2e.pull_request.one_card.ascendstore_v1_performance import AscendStoreBenchmarkWorkerExtension


class AscendStoreLayerwiseWorkerExtension(AscendStoreBenchmarkWorkerExtension):
    def start_layerwise_validation(self) -> dict[str, Any]:
        assert not hasattr(self, "_layerwise_probe"), "Layerwise probe already installed"
        connector, worker, backend, caches, implementation = self._layerwise_resources()
        self._drain_stores()
        self._layerwise_probe = LayerwiseRangeProbe(connector, backend, caches, implementation)
        return {"implementation": implementation, "route": type(worker).__name__, "layers": len(caches)}

    def arm_layerwise_destination_poisoning(self) -> dict[str, Any]:
        self._drain_stores()
        return self._layerwise_probe.arm_destination_poisoning()

    def finish_layerwise_validation(self) -> dict[str, Any]:
        self._drain_stores()
        return self._layerwise_probe.verify_complete()

    def remove_layerwise_validation(self) -> dict[str, bool]:
        probe = getattr(self, "_layerwise_probe", None)
        if probe is not None:
            try:
                self._drain_stores()
            finally:
                probe.remove()
                del self._layerwise_probe
        return {"probe_removed": not hasattr(self, "_layerwise_probe")}

    def _drain_stores(self) -> tuple[Any, Any, float]:
        _, worker, backend, _, implementation = self._layerwise_resources()
        start = time.perf_counter()
        if implementation == "production":
            worker._drain_attention_transfers(drain_recv=False, full=True)
            store = backend.store
        else:
            worker.fence_previous_store()
            assert worker._pending_store_batch is None, "Store source release is not confirmed"
            store = backend._store
        return backend, store, (time.perf_counter() - start) * 1000

    def _layerwise_resources(self) -> tuple[Any, Any, Any, dict[str, Any], str]:
        # Worker-process import keeps pytest and server setup out of the spawned model Worker.
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        connector = get_kv_transfer_group()
        kind = (type(connector).__module__, type(connector).__name__)
        if kind == (
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
            "AscendStoreConnector",
        ):
            worker = connector.connector_worker
            assert worker is not None and worker.use_layerwise and worker.use_block_key_layerwise
            return connector, worker, worker.m_store, worker.kv_caches, "production"
        assert kind == (
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector",
            "AscendStoreV1Connector",
        ), f"Unexpected connector: {kind}"
        worker = connector.worker
        assert worker is not None and type(worker).__name__ == "KeyRangeLayerwiseWorker", "Not the KeyRange route"
        return connector, worker, worker._resources.backend, worker._resources.kv_caches, "v1"
