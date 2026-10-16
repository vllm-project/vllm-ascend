# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Store drain controls and opt-in v1 host timing for the single-card Layerwise benchmark."""

import time
from typing import Any

from tests.e2e.pull_request.one_card.ascendstore_v1_layerwise_timing import LayerwiseLoadTiming
from tests.e2e.pull_request.one_card.ascendstore_v1_performance import AscendStoreBenchmarkWorkerExtension


class AscendStoreLayerwiseWorkerExtension(AscendStoreBenchmarkWorkerExtension):
    def start_ascendstore_layerwise_load_timing(self) -> dict[str, str]:
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        assert not hasattr(self, "_layerwise_load_timing"), "Layerwise timing is already active"
        worker, backend, implementation = self._layerwise_resources()
        assert implementation == "v1", "Load diagnostics must not instrument production"
        # vLLM mixes this extension into the real Worker, which owns model_runner.
        timing = LayerwiseLoadTiming(worker, backend, get_kv_transfer_group().lookup_server, self.model_runner)
        timing.install()
        self._layerwise_load_timing = timing
        return {"implementation": implementation, "status": "installed"}

    def stop_ascendstore_layerwise_load_timing(self) -> dict[str, Any]:
        timing = self._layerwise_load_timing
        try:
            # Requests have finished; restore even if the out-of-timing Store drain fails.
            self._drain_stores()
        finally:
            timing.restore()
            del self._layerwise_load_timing
        return timing.report()

    def _drain_stores(self) -> tuple[Any, Any, float]:
        worker, backend, implementation = self._layerwise_resources()
        start = time.perf_counter()
        if implementation == "production":
            worker._drain_attention_transfers(drain_recv=False, full=True)
            store = backend.store
        else:
            worker.fence_previous_store()
            assert worker._pending_store_batch is None, "Store source release is not confirmed"
            store = backend._store
        return backend, store, (time.perf_counter() - start) * 1000

    def _layerwise_resources(self) -> tuple[Any, Any, str]:
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
            return worker, worker.m_store, "production"
        assert kind == (
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector",
            "AscendStoreV1Connector",
        ), f"Unexpected connector: {kind}"
        worker = connector.worker
        assert worker is not None and type(worker).__name__ == "KeyRangeLayerwiseWorker", "Not the KeyRange route"
        return worker, worker._resources.backend, "v1"
