from collections.abc import Iterable
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import (
    KVCacheEvent,
    KVConnectorKVEvents,
    KVEventAggregator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.forward_context import ForwardContext
from vllm.logger import logger
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_config import (
    get_layerwise_config,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


class AscendStoreKVEvents(KVConnectorKVEvents):
    def __init__(self, num_workers: int) -> None:
        self._aggregator = KVEventAggregator(num_workers)

    def add_events(self, events: list[KVCacheEvent]) -> None:
        self._aggregator.add_events(events)

    def aggregate(self) -> "AscendStoreKVEvents":
        """
        Aggregate KV events and retain only common events.
        """
        common_events = self._aggregator.get_common_events()
        self._aggregator.clear_events()
        self._aggregator.add_events(common_events)
        self._aggregator.reset_workers()
        return self

    def increment_workers(self, count: int = 1) -> None:
        self._aggregator.increment_workers(count)

    def get_all_events(self) -> list[KVCacheEvent]:
        return self._aggregator.get_all_events()

    def get_number_of_workers(self) -> int:
        return self._aggregator.get_number_of_workers()

    def clear_events(self) -> None:
        self._aggregator.clear_events()
        self._aggregator.reset_workers()

    def __repr__(self) -> str:
        return f"<AscendStoreKVEvents events={self.get_all_events()}>"


class AscendStoreConnector(KVConnectorBase_V1, SupportsHMA):
    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """
        AscendStore requires PIECEWISE CUDA graph mode when layerwise
        operations are enabled.
        """
        return extra_config.get("use_layerwise", False)

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self.kv_role = vllm_config.kv_transfer_config.kv_role

        self.use_layerwise = vllm_config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", False)
        backend_name = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "backend", "mooncake")
        self.backend_name = backend_name.lower()
        self.use_gva_layerwise = self.use_layerwise and self.backend_name == "memcache"
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False
        )

        connector_name = vllm_config.kv_transfer_config.kv_connector
        if connector_name == "MooncakeConnectorStoreV1":
            logger.warning(
                "It is recommended to use the AscendStoreConnector, "
                "as the MoonCakeStoreConnector will be removed in the future."
            )

        if role == KVConnectorRole.SCHEDULER and self.use_gva_layerwise:
            num_layers = vllm_config.model_config.get_num_layers(
                vllm_config.parallel_config)
            extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
            if (get_layerwise_config(num_layers, extra_config).has_layer_reuse
                    and self.kv_role != "kv_producer"):
                logger.warning(
                    "[KV POOL PERFORMANCE WARNING] Layerwise KV cache reuse "
                    "is only expected to perform well on the prefill "
                    "producer node in PD disaggregation. Current kv_role is "
                    "%s, so this mode can have very poor performance.",
                    self.kv_role)

        self.kv_caches: dict[str, torch.Tensor] = {}
        self._kv_cache_events: AscendStoreKVEvents | None = None

        if role == KVConnectorRole.SCHEDULER:
            page_size_bytes = kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
            self.connector_scheduler = KVPoolScheduler(vllm_config, self.use_layerwise, kv_cache_config, page_size_bytes=page_size_bytes)
        else:
            self.connector_worker = KVPoolWorker(
                vllm_config,
                self.use_layerwise,
                kv_cache_config,
            )
            assert self.connector_worker is not None

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished_all_groups(request, block_ids)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side connectors output.
        """
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_finished_recving(
            connector_output.finished_recving)
        self.connector_scheduler.update_finished_sending(
            connector_output.finished_sending)

        # Get the KV events
        kv_cache_events = connector_output.kv_cache_events
        if not kv_cache_events or not isinstance(kv_cache_events, AscendStoreKVEvents):
            return

        if self._kv_cache_events is None:
            self._kv_cache_events = kv_cache_events
        else:
            self._kv_cache_events.add_events(kv_cache_events.get_all_events())
            self._kv_cache_events.increment_workers(kv_cache_events.get_number_of_workers())
        return

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """
        Take the KV cache events from the connector.

        Yields:
            New KV cache events since the last call.
        """
        if self._kv_cache_events is not None:
            self._kv_cache_events.aggregate()
            kv_cache_events = self._kv_cache_events.get_all_events()
            yield from kv_cache_events
            self._kv_cache_events.clear_events()
            self._kv_cache_events = None

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def rebind_kv_transfer_threads(self) -> None:
        assert self.connector_worker is not None
        self.connector_worker.rebind_kv_transfer_threads()

    def init_backend(self) -> None:
        assert self.connector_worker is not None
        self.connector_worker.init_backend()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        logger.debug(
            "KV pool connector start_load_kv metadata_requests=%d specs=%s",
            len(metadata.requests),
            [
                (
                    request.req_id,
                    None if request.load_spec is None else request.load_spec.can_load,
                    None if request.load_spec is None else request.load_spec.vllm_cached_tokens,
                    None if request.load_spec is None else request.load_spec.kvpool_cached_tokens,
                )
                for request in metadata.requests
            ],
        )
        self.connector_worker.start_load_kv(metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self.use_layerwise:
            return
        self.connector_worker.wait_for_layer_load()

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        if not self.use_layerwise:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        self.connector_worker.save_kv_layer(self._get_connector_metadata())

    def wait_for_save(self):
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_put:
            # Don't do save if the role is kv_consumer
            return

        if self.use_layerwise:
            return

        self.connector_worker.wait_for_save(self._get_connector_metadata())

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        done_sending, done_recving = self.connector_worker.get_finished(
            finished_req_ids, self._get_connector_metadata()
        )
        return done_sending, done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return KV block IDs that failed to load on the worker."""
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def get_kv_connector_kv_cache_events(self) -> AscendStoreKVEvents | None:
        """
        Get the KV connector kv cache events collected during the last interval.
        """
        events = self.connector_worker.get_kv_events()
        if not events:
            return None

        ascend_store_kv_events = AscendStoreKVEvents(num_workers=1)
        ascend_store_kv_events.add_events(events)
        return ascend_store_kv_events
