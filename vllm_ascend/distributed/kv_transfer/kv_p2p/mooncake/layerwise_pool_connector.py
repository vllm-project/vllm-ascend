# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Independent connector facade: pooled storage with CPD request handoff."""

import threading
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from vllm.distributed import (
    get_pcp_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.utils.network_utils import get_ip

from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.core.kv_cache_placement import (
    build_layer_cache_bundles,
    find_draft_layers,
    map_kvpp_layers_to_owners,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import (
    LayerwisePoolConnectorMetadata,
    PoolSourceProfile,
    PoolWorkerMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_scheduler import LayerwisePoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_stats import PoolPDPromMetrics, PoolPDStats
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_worker import LayerwisePoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import get_backend_class
from vllm_ascend.distributed.utils import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
)
from vllm_ascend.utils import preserve_framework_signal_handlers

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


def _create_backend(backend_class, parallel_config, extra_config):
    # MemCache installs native termination handlers without updating Python's
    # signal table. Let vLLM drain workers and leases before closing the SDK;
    # direct SDK signal cleanup can otherwise race the live connector owners.
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Pool backend initialization requires the main thread to preserve framework signal handlers")
    with preserve_framework_signal_handlers():
        return backend_class(parallel_config, extra_config=extra_config)


class LayerwisePoolConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        super().__init__(vllm_config, role, kv_cache_config)
        transfer = vllm_config.kv_transfer_config
        if transfer is None or transfer.is_kv_producer == transfer.is_kv_consumer:
            raise ValueError("LayerwisePoolConnector requires kv_producer or kv_consumer")
        self.config, self.cache_config = vllm_config, kv_cache_config
        self.producer = transfer.is_kv_producer
        extra = getattr(transfer, "kv_connector_extra_config", {})
        if self.producer and extra.get("_pool_pd_role") == "kv_consumer":
            raise ValueError("Pool producer cannot use the bulk-decode graph exemption")
        self.connector_scheduler: LayerwisePoolScheduler | None = None
        self.connector_worker: LayerwisePoolWorker | None = None
        self._handshake = None
        self._bundles: dict[str, tuple[str, ...]] = {}
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = LayerwisePoolScheduler(vllm_config, kv_cache_config)
        elif role == KVConnectorRole.WORKER:
            self._init_worker()
        else:
            raise ValueError(f"Unsupported pool connector role: {role}")

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        # The platform derives this private field from kv_role before graph
        # configuration. Unknown roles conservatively require layer hooks.
        # D uses whole-request asynchronous restore outside graph execution.
        return extra_config.get("_pool_pd_role") != "kv_consumer"

    @classmethod
    def supports_sleep_mode(cls, kv_transfer_config) -> bool:
        # Registered addresses and active native I/O need an explicit remap/
        # drain contract before the connector can participate in sleep mode.
        return False

    def _init_worker(self) -> None:
        config = self.config
        transfer = config.kv_transfer_config
        extra = transfer.kv_connector_extra_config
        backend_class = get_backend_class(extra.get("backend", "memcache"))
        if backend_class is None or backend_class.layerwise_data_plane not in ("gva", "block_key"):
            raise ValueError("Pool connector backend must support layerwise object ranges")
        lease_ttl_ms = extra.get("lease_ttl_ms", 600000)
        retention_seconds = extra.get("retention_seconds", 300)
        if (
            type(lease_ttl_ms) is not int
            or lease_ttl_ms <= 0
            or type(retention_seconds) not in (int, float)
            or not 0 < retention_seconds * 1000 < lease_ttl_ms
        ):
            raise ValueError("Pool retention must be positive and shorter than its native lease TTL")
        parallel = config.parallel_config
        tp_rank, tp_size = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        pcp_size, pcp_rank = get_pcp_group().world_size, get_pcp_group().rank_in_group
        pp_rank = get_pp_group().rank_in_group
        dcp_size, dcp_rank = (
            get_decode_context_model_parallel_world_size(),
            get_decode_context_model_parallel_rank(),
        )
        model_name = extra.get("pool_namespace", config.model_config.model.rstrip("/").split("/")[-1])
        keys = backend_class.bind_layerwise_keys(
            vllm_config=config,
            kv_cache_config=self.cache_config,
            model_name=model_name,
            use_hybrid=len(self.cache_config.kv_cache_groups) > 1,
            grouped_block_size=[
                group.kv_cache_spec.block_size * dcp_size for group in self.cache_config.kv_cache_groups
            ],
        )
        if keys.identity is None:
            raise ValueError("Pool backend must export reconstructible source key identity")
        backend = _create_backend(backend_class, parallel, extra)
        worker_id = f"pp{pp_rank}/pcp{pcp_rank}/tp{tp_rank}"
        try:
            worker = LayerwisePoolWorker(
                backend,
                self.cache_config,
                worker_id=worker_id,
                tp_rank=tp_rank,
                tp_size=tp_size,
                dcp_rank=dcp_rank,
                dcp_size=dcp_size,
                pcp_size=pcp_size,
                pcp_rank=pcp_rank,
                pp_rank=pp_rank,
                total_num_kv_heads=config.model_config.get_total_num_kv_heads(),
                lease_ttl_ms=lease_ttl_ms,
                base_layers=config.model_config.get_total_num_hidden_layers(),
                pool_layer_offset=config.model_config.get_layers_start_end_indices(parallel)[0],
                pool_head_rank=(
                    pcp_rank * tp_size + tp_rank
                    if KVPPConfig.from_vllm_config(config).size > 1
                    else tp_rank
                    // max(
                        1,
                        tp_size // (1 if config.model_config.use_mla else config.model_config.get_total_num_kv_heads()),
                    )
                ),
            )
            profile = PoolSourceProfile(
                uuid4().hex,
                uuid4().hex,
                keys.identity,
                group_block_sizes=tuple(
                    group.kv_cache_spec.block_size * dcp_size for group in self.cache_config.kv_cache_groups
                ),
            )
            if self.producer:
                self._handshake = worker.enable_manifest_control(
                    profile, extra.get("control_host", get_ip()), retention_seconds=retention_seconds
                )
            self.connector_worker = worker
        except Exception:
            backend.close()
            raise

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.hash_block_size = gpu_block_pool.hash_block_size

    def on_new_request(self, request) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.on_new_request(request)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def get_loaded_kv_cache_group_ids(self, request):
        if self.producer and self.connector_scheduler is not None:
            return self.connector_scheduler.prefix_group_ids
        return self.cache_config.transfer_group_ids

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output):
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    def request_finished(self, request, block_ids):
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(self, request, block_ids):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def set_xfer_handshake_metadata(self, metadata):
        assert self.connector_scheduler is not None
        if self.producer:
            self.connector_scheduler.set_xfer_handshake_metadata(metadata)

    def set_xfer_handshake_metadata_pp_aware(self, metadata):
        self.set_xfer_handshake_metadata(metadata)

    def get_handshake_metadata(self):
        return self._handshake

    def register_kv_caches(self, kv_caches):
        assert self.connector_worker is not None
        owned = None
        draft_layers = set()
        if KVPPConfig.from_vllm_config(self.config).size > 1:
            draft_layers = find_draft_layers(self.config, kv_caches)
            owners = map_kvpp_layers_to_owners(self.config, kv_caches)
            rank = get_pcp_group().rank_in_group * self.connector_worker.tp_size + self.connector_worker.tp_rank
            owned = {name for name in kv_caches if owners.get(name) in (None, rank)}
        self.connector_worker.register_kv_caches(kv_caches, owned_layers=owned)
        # KVPP allocates draft caches independently, but their auxiliary
        # indexer/scale caches still share the attention save callback. Only
        # unindexed draft modules need the singleton fallback; numeric draft
        # layers use the same namespace-aware completion grouping as target.
        unindexed_drafts = {name for name in draft_layers if not any(part.isdecimal() for part in name.split("."))}
        self._bundles = build_layer_cache_bundles(self.connector_worker.specs, unindexed_drafts)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        if not isinstance(connector_metadata, LayerwisePoolConnectorMetadata):
            raise TypeError("Layerwise pool received another connector's scheduling metadata")
        super().bind_connector_metadata(connector_metadata)
        assert self.connector_worker is not None
        # Retire cancelled Prefill histories even when V2 Runner performs no
        # model forward and defers/skips its start_load_kv callback.
        self.connector_worker.release_source_requests(connector_metadata.released_sources)
        # V2 Runner may defer start_load_kv until post_forward. Source plans
        # must exist before actual attention/draft cache writes execute.
        if connector_metadata.saves:
            assert self.producer and self._handshake is not None
            for req_id, request in connector_metadata.saves.items():
                self.connector_worker.prepare_save_request(req_id, request, self._handshake.profile)

    def start_load_kv(self, forward_context, **kwargs):
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, LayerwisePoolConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name):
        # D becomes runnable only after the entire BulkReader result succeeds.
        return

    def on_kv_cache_written(self, layer_name: str = "") -> None:
        # Scatter notifications mark the enclosing cache bundle ready, just
        # like the attention-end hook. Worker submission is idempotent when
        # an attention backend emits both notifications in the same forward.
        self.save_kv_layer(layer_name, None, None)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        if not self.producer or not self.has_connector_metadata() or not layer_name:
            return
        assert self.connector_worker is not None
        # Auxiliary indexer/scale caches share the enclosing attention's
        # completion boundary; do not invent model-specific call counters.
        for name in self._bundles.get(layer_name, (layer_name,)):
            self.connector_worker.save_kv_layer(name)

    def wait_for_save(self):
        assert self.connector_worker is not None
        self.connector_worker.wait_for_save()

    def build_connector_worker_meta(self):
        assert self.connector_worker is not None
        reports = tuple(self.connector_worker.take_write_reports())
        return PoolWorkerMetadata(reports) if reports else None

    def get_transfer_results(self, finished_req_ids):
        assert self.connector_worker is not None
        return self.connector_worker.get_transfer_results()

    def get_finished(self, finished_req_ids):
        results = self.get_transfer_results(finished_req_ids)
        return results.finished_sending, results.finished_recving

    def get_block_ids_with_load_errors(self):
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def get_kv_connector_stats(self):
        return None if self.connector_worker is None else self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(cls, data=None):
        return PoolPDStats(data=data or {})

    @classmethod
    def build_prom_metrics(cls, vllm_config, metric_types, labelnames, per_engine_labelvalues):
        return PoolPDPromMetrics(vllm_config, metric_types, labelnames, per_engine_labelvalues)

    def shutdown(self):
        if self.connector_scheduler is not None:
            self.connector_scheduler.shutdown()
        if self.connector_worker is not None:
            self.connector_worker.shutdown()
