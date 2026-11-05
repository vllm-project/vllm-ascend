# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pool-backed CPD facade; ordinary pool code owns layouts and MemCache I/O."""

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)

from vllm_ascend.distributed.kv_transfer.kv_p2p.pd_pool.metadata import PoolPDMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.pd_pool.scheduler import PoolPDScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.pd_pool.worker import PoolPDWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key import LookupKeyServer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_transfer import (
    validate_shared_pool,
)


class LayerwisePoolConnector(KVConnectorBase_V1, SupportsHMA):
    """Layerwise or bulk P writes, whole-request D reads, one shared pool."""

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        validate_shared_pool(vllm_config, kv_cache_config, worker=role != KVConnectorRole.SCHEDULER)
        self.scheduler = None
        self.worker = None
        self.lookup_server = None
        self._mamba_state = None
        layerwise = bool(vllm_config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", False))
        # Use the same MRV2 extension points as ordinary AscendStore. The
        # runner owns checkpoint src/dst selection and the actual state copy.
        self.requires_mamba_state_copy_after_layer_load = layerwise
        self.requires_mamba_state_copy_after_load = not layerwise
        if role == KVConnectorRole.SCHEDULER:
            pool_scheduler = KVPoolScheduler(vllm_config, False, kv_cache_config)
            groups = pool_scheduler.shared_groups
            self.target_groups = tuple(groups["target_groups"])
            self.prefix_groups = tuple(groups["prefix_groups"])
            self.scheduler = PoolPDScheduler(
                pool_scheduler,
                need_truncate=groups["stateful"],
                mtp_tokens=getattr(vllm_config.speculative_config, "num_speculative_tokens", 0),
            )
        else:
            pool = KVPoolWorker(vllm_config, False, kv_cache_config)
            self.worker = PoolPDWorker(pool)
            if vllm_config.parallel_config.rank == 0:
                self.lookup_server = LookupKeyServer(pool, vllm_config)

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config):
        return None if vllm_config.model_config.use_mla else "HND"

    def get_loaded_kv_cache_group_ids(self, request):
        # Ordinary prefix loads do not restore request-lifetime rings. Retain
        # the selected PD load kind after allocation clears do_remote_prefill;
        # the scheduler also queries this contract when async receive ends.
        if request.request_id in self.scheduler.pd_load_requests:
            return self.target_groups
        return self.prefix_groups

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return self.scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def on_new_request(self, request):
        self.scheduler.on_new_request(request)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output):
        return self.scheduler.build_connector_meta(scheduler_output)

    def request_finished(self, request, block_ids):
        return self.scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(self, request, block_ids):
        return self.scheduler.request_finished(request, block_ids)

    def bind_gpu_block_pool(self, block_pool):
        self.scheduler.pool.bind_gpu_block_pool(block_pool)

    def update_connector_output(self, output):
        self.scheduler.pool.update_connector_output(output)

    def build_connector_worker_meta(self):
        # Pool saves pin committed Mamba checkpoints until every worker has
        # acknowledged publication. Forward the existing event protocol so
        # those references are released even after a request is cancelled.
        return self.worker.pool.build_connector_worker_meta()

    def has_pending_push_work(self):
        return bool(self.scheduler and self.scheduler.pool.has_pending_push_work())

    def register_kv_caches(self, caches):
        self.worker.pool.register_kv_caches(caches)

    def bind_connector_metadata(self, connector_metadata):
        if not isinstance(connector_metadata, PoolPDMetadata):
            raise TypeError("LayerwisePoolConnector requires PoolPDMetadata")
        super().bind_connector_metadata(connector_metadata)
        self.worker.prepare(connector_metadata)

    def start_load_kv(self, forward_context, **kwargs):
        self.worker.start_load(self._connector_metadata)

    def wait_for_layer_load(self, layer_name):
        # CPD loads complete in start_load_kv; ordinary P prefix hits retain
        # the pool's configured per-layer restore timing.
        if self.worker.transfer.layerwise:
            self.worker.transfer.load_layer(layer_name)
            if self._mamba_state is not None:
                self._mamba_state.do_mamba_copy_for_layer(layer_name)

    def prepare_mamba_state_copy(self, model_state):
        if not self.requires_mamba_state_copy_after_layer_load:
            return False
        if not callable(getattr(model_state, "do_mamba_copy_for_layer", None)):
            raise TypeError("Pool PD requires the MRV2 deferred Mamba state-copy interface")
        self._mamba_state = model_state
        return True

    def finish_mamba_state_copy(self):
        self._mamba_state = None

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        if self.worker.transfer.layerwise:
            self.worker.save_layer(layer_name)

    def wait_for_save(self):
        self.worker.finish()

    def get_transfer_results(self, finished_req_ids):
        return self.worker.take_results()

    def get_finished(self, finished_req_ids):
        result = self.worker.take_results()
        return result.finished_sending, result.finished_recving

    def get_block_ids_with_load_errors(self):
        # Block IDs alone do not identify a hybrid cache group. Report the
        # entire failed request through failed_recving instead; the scheduler
        # applies its configured fail/recompute policy across all groups.
        return set()

    def shutdown(self):
        try:
            if self.scheduler is not None:
                pool = self.scheduler.pool
                try:
                    if pool.client is not None:
                        pool.client.close()
                finally:
                    pool.store_scheduler.close()
        finally:
            try:
                if self.lookup_server is not None:
                    self.lookup_server.close()
            finally:
                if self.worker is not None:
                    self.worker.shutdown()
        super().shutdown()
