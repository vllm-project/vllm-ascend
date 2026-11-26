from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_events import (
    BlockStored,
    KVCacheEvent,
    KVConnectorKVEvents,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.forward_context import ForwardContext
from vllm.logger import logger
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    get_backend_class,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_cache import DSparkPrefixKeys
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hisparse.layout import has_hisparse_cache_layout
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    has_shared_kv_sources,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import (
    LayerwiseKVPoolScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import (
    LayerwiseKVPoolWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key import LookupKeyServer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendStoreKVConnectorWorkerMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metrics import (
    AscendStoreKVConnectorStats,
    AscendStorePromMetrics,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    BaseKVPoolScheduler,
    KVPoolScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import BaseKVPoolWorker, KVPoolWorker

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorHandshakeMetadata


def create_kv_pool_scheduler(
    vllm_config: VllmConfig, use_layerwise: bool, kv_cache_config: KVCacheConfig | None = None
) -> BaseKVPoolScheduler:
    scheduler_class = LayerwiseKVPoolScheduler if use_layerwise else KVPoolScheduler
    return scheduler_class(vllm_config, use_layerwise, kv_cache_config)


def create_kv_pool_worker(
    vllm_config: VllmConfig,
    use_layerwise: bool,
    kv_cache_config: KVCacheConfig | None = None,
) -> BaseKVPoolWorker:
    worker_class = LayerwiseKVPoolWorker if use_layerwise else KVPoolWorker
    return worker_class(
        vllm_config,
        use_layerwise,
        kv_cache_config,
    )


class AscendStoreKVEvents(KVConnectorKVEvents):
    """Publish a logical block once every rank that writes it has reported.

    ``group_tp_replication_factors[group_idx]`` is the number of aggregated
    workers that own other blocks of that group. A block is ready when
    ``count * factor >= num_workers``. The required count is the number of
    ranks that write its keys. Reports that arrive on later steps stay in
    the counter.
    """

    def __init__(
        self,
        num_workers: int,
        group_tp_replication_factors: Sequence[int] = (1,),
    ) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be greater than zero.")
        if not group_tp_replication_factors or any(factor <= 0 for factor in group_tp_replication_factors):
            raise ValueError("TP replication factors must be positive.")
        self._event_counter: Counter[KVCacheEvent] = Counter()
        self._num_workers = num_workers
        self._group_tp_replication_factors = tuple(group_tp_replication_factors)

    def add_events(self, events: list[KVCacheEvent]) -> None:
        if not isinstance(events, list):
            raise TypeError("events must be a list of KVCacheEvent.")
        self._event_counter.update(events)

    def _replication_factor(self, event: KVCacheEvent) -> int:
        if not isinstance(event, BlockStored) or event.group_idx is None:
            return 1
        if event.group_idx < 0 or event.group_idx >= len(self._group_tp_replication_factors):
            return 1
        return self._group_tp_replication_factors[event.group_idx]

    def _is_ready(self, event: KVCacheEvent, count: int) -> bool:
        return count * self._replication_factor(event) >= self._num_workers

    def aggregate(self) -> "AscendStoreKVEvents":
        """Keep partial quorums. Callers publish through pop_common_events."""
        return self

    def pop_common_events(self) -> list[KVCacheEvent]:
        """Remove and return events whose block writers have all reported."""
        ready = [event for event, count in self._event_counter.items() if self._is_ready(event, count)]
        for event in ready:
            del self._event_counter[event]
        return ready

    def has_events(self) -> bool:
        return bool(self._event_counter)

    def increment_workers(self, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("count must be positive.")
        self._num_workers += count

    def get_all_events(self) -> list[KVCacheEvent]:
        return list(self._event_counter.elements())

    def get_number_of_workers(self) -> int:
        return self._num_workers

    def clear_events(self) -> None:
        self._event_counter.clear()
        self._num_workers = 1

    def __repr__(self) -> str:
        return f"<AscendStoreKVEvents events={self.get_all_events()}>"


class AscendStoreConnector(KVConnectorBase_V1, SupportsHMA):
    @classmethod
    def supports_lookahead_block_hashes(cls, kv_transfer_config: KVTransferConfig) -> bool:
        # AscendStore's external coordinator retains the trailing EAGLE-block
        # drop and prompt-token replay boundaries. It does not propagate the
        # core lookahead length through its store/lookup masks.
        return False

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
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        if "consumer_is_to_put" in extra_config:
            raise ValueError("consumer_is_to_put has been replaced by save_decode_cache")
        if "consumer_is_to_load" in extra_config:
            raise ValueError("consumer_is_to_load has been replaced by enable_lookup")
        self.use_layerwise = extra_config.get("use_layerwise", False)
        self.save_decode_cache = extra_config.get("save_decode_cache", False)
        self.supports_attention_prefetch_gate = bool(self.use_layerwise)
        self.backend_name = extra_config.get("backend", "mooncake").lower()
        if has_hisparse_cache_layout(kv_cache_config):
            incompatibilities = []
            if self.use_layerwise or self.kv_role != "kv_both":
                incompatibilities.append("requires whole-block transfers with kv_role=kv_both")
            if not extra_config.get("load_async", False):
                # HiSparse publishes imported Host Source pages on receive completion.
                incompatibilities.append("requires load_async=true")
            if extra_config.get("discard_partial_chunks", True) is not True:
                incompatibilities.append("requires discard_partial_chunks=true to save only complete blocks")
            # A one-token prefill chunk may take the decode path. A later
            # prefill PUT can publish that row before the deferred page copy.
            hisparse_config = vllm_config.attention_config.hisparse_config
            if hisparse_config is None or not hisparse_config.eager_host_mirror:
                incompatibilities.append("requires eager_host_mirror=true for all whole-block saves")
            try:
                pool_backend_class = get_backend_class(self.backend_name)
                if pool_backend_class is None:
                    raise ValueError(f"unknown pool backend {self.backend_name!r}")
                pool_backend_class.validate_hisparse_colocated_config()
            except (ValueError, OSError, NotImplementedError) as error:
                incompatibilities.append(str(error))
            if incompatibilities:
                raise ValueError(
                    "HiSparse with AscendStore has incompatible configuration:\n- " + "\n- ".join(incompatibilities)
                )
        self.backend_class = get_backend_class(self.backend_name) if self.use_layerwise else None
        self.layerwise_data_plane = self.backend_class.layerwise_data_plane if self.backend_class is not None else None
        self.use_block_key_layerwise = self.use_layerwise and self.layerwise_data_plane == "block_key"
        self.supports_composite_kv_loads = self.use_layerwise and has_shared_kv_sources(vllm_config)
        self.supports_composite_kv_saves = self.supports_composite_kv_loads
        self.can_put = self.kv_role in ("kv_producer", "kv_both") or self.save_decode_cache

        connector_name = vllm_config.kv_transfer_config.kv_connector
        if connector_name == "MooncakeConnectorStoreV1":
            logger.warning(
                "It is recommended to use the AscendStoreConnector, "
                "as the MoonCakeStoreConnector will be removed in the future."
            )

        self._kv_cache_events: AscendStoreKVEvents | None = None
        self._current_step_has_real_forward = False
        # Handle to the (V2) mamba hybrid model state while its per-layer
        # align pre-copy is deferred behind this connector's layerwise loads.
        self._mamba_state: Any = None
        self._loaded_layer: int | None = None
        self.requires_mamba_state_copy_after_layer_load = self.use_layerwise
        # Opt in only for synchronous bulk loads, not background retrieval.
        self.requires_mamba_state_copy_after_load = not self.use_layerwise and not extra_config.get("load_async", False)

        self.connector_scheduler: BaseKVPoolScheduler | None = None
        self.connector_worker: BaseKVPoolWorker | None = None
        self.dspark_prefix_cache_enabled = False

        if role == KVConnectorRole.SCHEDULER:
            assert kv_cache_config is not None
            self.connector_scheduler = create_kv_pool_scheduler(vllm_config, self.use_layerwise, kv_cache_config)
        else:
            self.connector_worker = create_kv_pool_worker(
                vllm_config,
                self.use_layerwise,
                kv_cache_config,
            )
            assert self.connector_worker is not None
            if not self.use_layerwise and vllm_config.parallel_config.rank == 0:
                self.lookup_server = LookupKeyServer(self.connector_worker, vllm_config)

    def configure_dspark_prefix_cache(self, vllm_config: VllmConfig) -> bool:
        """Opt the SFA DSpark producer into joint external prefix reuse.

        MultiConnector passes the parent config because a store child's config
        alone does not identify its SFA PD sibling. Other serving paths do not
        change their keys or layerwise hook layout.
        """
        if not self.use_layerwise or self.layerwise_data_plane != "gva" or self.kv_role != "kv_producer":
            return False
        keys = DSparkPrefixKeys.from_config(vllm_config)
        if keys is None:
            return False
        self.dspark_prefix_cache_enabled = True
        if self.connector_scheduler is not None:
            cast(LayerwiseKVPoolScheduler, self.connector_scheduler).dspark_prefix_keys = keys
            # Retain a final prompt block for safe target/auxiliary recompute.
            self.connector_scheduler.use_eagle = True
        if self.connector_worker is not None:
            cast(LayerwiseKVPoolWorker, self.connector_worker).configure_dspark_prefix_cache(keys)
        return True

    def restore_dspark_prefix(self, metadata: Any, request_id: str, offset: int, block_ids: Any) -> int:
        assert self.connector_worker is not None
        cache = getattr(self.connector_worker, "dspark_prefix_cache", None)
        request = next((request for request in metadata.requests if request.req_id == request_id), None)
        if cache is None or request is None:
            raise RuntimeError("DSpark prefix hit has no registered draft cache or load metadata")
        return cache.restore(request, offset, block_ids)

    def save_dspark_prefix(self, metadata: Any, request_id: str, through_tokens: int, block_ids: Any) -> None:
        assert self.connector_worker is not None
        cache = getattr(self.connector_worker, "dspark_prefix_cache", None)
        request = next((request for request in metadata.requests if request.req_id == request_id), None)
        if cache is not None and request is not None:
            cache.save(request, through_tokens, block_ids)

    def shutdown(self):
        """Stop store users before closing the backend's native resources."""
        try:
            if self.connector_scheduler is not None:
                client = self.connector_scheduler.client
                try:
                    if client is not None:
                        client.close()
                finally:
                    self.connector_scheduler.store_scheduler.close()
        finally:
            if hasattr(self, "lookup_server"):
                self.lookup_server.close()
            if self.connector_worker is not None:
                self.connector_worker.shutdown()
            super().shutdown()

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def set_xfer_handshake_metadata_pp_aware(
        self,
        metadata: dict[tuple[int, int], "KVConnectorHandshakeMetadata"],
    ) -> None:
        """Ignore P/D handshake metadata because AscendStore handles PP via pool keys."""
        pass

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

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        assert self.connector_scheduler is not None
        if self.connector_scheduler.use_layerwise:
            return False
        return self.connector_scheduler.register_finished_partial_tail(request, block_ids, partial_tail_offloads)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side connectors output.
        """
        if self.connector_scheduler is not None:
            self.connector_scheduler.update_connector_output(connector_output)

        # Get the KV events
        kv_cache_events = connector_output.kv_cache_events
        if not kv_cache_events or not isinstance(kv_cache_events, AscendStoreKVEvents):
            return

        if self._kv_cache_events is None:
            self._kv_cache_events = kv_cache_events
        else:
            # The retained object already counted every worker. Adding the next
            # step's worker count would raise the quorum on every poll.
            self._kv_cache_events.add_events(kv_cache_events.get_all_events())

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """Yield blocks whose writers have all reported, and keep the rest."""
        if self._kv_cache_events is not None:
            ready = self._kv_cache_events.pop_common_events()
            if not self._kv_cache_events.has_events():
                self._kv_cache_events = None
            yield from ready

    ############################################################
    # Worker Side Methods
    ############################################################
    def set_external_slot_release_waiter(self, waiter: Callable[[int], None]) -> bool:
        """Pure forwarder: the layerwise transfer gate is evaluated by the worker.

        The connector must not derive the gate itself — the copy here was
        dropped by #14465 while this method still read it (crashing
        MultiConnector init), and restored by #15291. Gating at the
        data-plane consumer, where the flag is already derived, makes that
        class of regression structurally impossible and supersedes the
        connector-side flag entirely.
        """
        if self.connector_worker is None:
            return False
        return self.connector_worker.set_external_slot_release_waiter(waiter)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        """Fence the previous save before this step can reuse KV blocks.

        This hook is temporarily reused for deferred KV cache save
        synchronization and will be replaced by a dedicated mechanism.
        """
        assert self.connector_worker is not None
        self.connector_worker.wait_for_previous_save()

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        if self.use_layerwise:
            assert self.connector_worker is not None
            # Layerwise hooks need this step's tasks before target forward.
            # If scheduler_output.has_sync_kv_loads is False (e.g. save-only
            # steps), V1 calls start_load_kv after target forward, before MTP.
            # Target hooks have advanced current_layer by then, so preparing
            # the step there would reset the current layer before MTP.
            self._loaded_layer = None
            cast(LayerwiseKVPoolWorker, self.connector_worker).prepare_layerwise_step(connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        self._current_step_has_real_forward = forward_context is not None
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

    def wait_for_layer_load_ready(self, layer_name: str) -> None:
        """Let KVPP observe a layer's loads without advancing attention hooks."""
        if not self.use_layerwise:
            return
        assert self.connector_worker is not None
        cast(LayerwiseKVPoolWorker, self.connector_worker).wait_for_layer_load_ready(layer_name)

    def finish_forward(self) -> None:
        if self.use_layerwise and self.has_connector_metadata():
            assert self.connector_worker is not None
            cast(LayerwiseKVPoolWorker, self.connector_worker).load_dflash_draft_kv()

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self.use_layerwise:
            return
        assert self.connector_worker is not None
        worker = cast(LayerwiseKVPoolWorker, self.connector_worker)
        if layer_name in getattr(worker, "dflash_draft_layer_names", ()):
            return
        # PLE and attention can share one physical layer. Load its groups once,
        # then let each cache owner copy its own recurrent state before use.
        layer = worker._extract_physical_layer_index(layer_name)
        if self._loaded_layer != layer:
            worker.wait_for_layer_load()
            self._loaded_layer = layer
        # Mamba state copy must run AFTER this layer's load finishes,
        # otherwise the copy would race with the in-flight layerwise load and
        # read half-loaded state. The model state no-ops for non-mamba layers
        # and for layers whose copy already ran.
        if self._mamba_state is not None:
            self._mamba_state.do_mamba_copy_for_layer(layer_name)

    def prepare_mamba_state_copy(self, mamba_state_or_copy_bufs) -> bool:
        """Take over the mamba align pre-copy for this step.

        Each layer's copy is executed from :meth:`wait_for_layer_load` right
        after that layer's KV load (conv/ssm state included) completes.
        """
        if not self.requires_mamba_state_copy_after_layer_load:
            return False
        if not hasattr(mamba_state_or_copy_bufs, "do_mamba_copy_for_layer"):
            raise TypeError(
                "prepare_mamba_state_copy expects a mamba hybrid model state with "
                f"do_mamba_copy_for_layer, got {type(mamba_state_or_copy_bufs).__name__}"
            )
        self._mamba_state = mamba_state_or_copy_bufs
        return True

    def finish_mamba_state_copy(self) -> None:
        self._mamba_state = None

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        if not self.use_layerwise:
            return

        if not self.can_put:
            # A load-only consumer does not publish KV.
            return
        assert self.connector_worker is not None
        worker = cast(LayerwiseKVPoolWorker, self.connector_worker)
        if layer_name not in getattr(worker, "dflash_draft_layer_names", ()):
            worker.save_kv_layer(self._get_connector_metadata())

    def has_pending_bulk_save(self) -> bool:
        """Let the parent order Host mirroring before this connector's save hook."""
        return (
            self.can_put
            and not self.use_layerwise
            and any(request.can_save for request in self._get_connector_metadata().requests)
        )

    def wait_for_save(self):
        if not self.can_put:
            # A load-only consumer has no store thread to wait for.
            return

        if self.use_layerwise:
            assert self.connector_worker is not None
            worker = cast(LayerwiseKVPoolWorker, self.connector_worker)
            if self.has_connector_metadata():
                worker.save_dflash_draft_kv(self._get_connector_metadata())
            return

        assert self.connector_worker is not None
        cast(KVPoolWorker, self.connector_worker).wait_for_save(self._get_connector_metadata())

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        if self._current_step_has_real_forward:
            try:
                self.connector_worker.m_store.ensure_initialized()
            finally:
                self._current_step_has_real_forward = False
        done_sending, done_recving = self.connector_worker.get_finished(finished_req_ids, metadata)
        return done_sending, done_recving

    def get_transfer_results(self, finished_req_ids: set[str]) -> KVConnectorTransferResults:
        done_sending, done_recving = self.get_finished(finished_req_ids)
        assert self.connector_worker is not None
        failed = self.connector_worker.get_failed_requests()
        return KVConnectorTransferResults(
            finished_sending=done_sending, finished_recving=done_recving, failed_recving=failed
        )

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return KV block IDs that failed to load on the worker."""
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def get_kv_connector_kv_cache_events(self) -> AscendStoreKVEvents | None:
        """Return this poll's events, including an empty batch.

        An empty batch still counts this worker toward the block quorum.
        Workers that cannot publish events stay out of that count.
        """
        assert self.connector_worker is not None
        worker = self.connector_worker
        if not getattr(worker, "enable_kv_events", False) or getattr(worker, "kv_send_thread", None) is None:
            return None
        factors = getattr(worker, "group_tp_replication_factors", (1,))
        if not isinstance(factors, tuple) or not factors:
            factors = (1,)
        events = AscendStoreKVEvents(num_workers=1, group_tp_replication_factors=factors)
        events.add_events(list(worker.get_kv_events()))
        return events

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.bind_gpu_block_pool(gpu_block_pool)

    def bind_kv_cache_manager(self, kv_cache_manager: KVCacheManager) -> None:
        super().bind_kv_cache_manager(kv_cache_manager)
        assert self.connector_scheduler is not None
        self.connector_scheduler.bind_kv_cache_manager(kv_cache_manager)

    def reset_cache(self) -> bool | None:
        """Drop external Mooncake KV after a prefix-cache reset.

        Scheduler only. Drains the rank-0 send queue, then ``remove_all``
        on the Mooncake master. Caller must pause generation first so no
        new puts are enqueued during the drain. Worker role returns None.
        Mooncake layerwise is unsupported and returns False. Other backends
        keep the base-class "not implemented" result of None.
        """
        if self.role != KVConnectorRole.SCHEDULER:
            return None
        if self.backend_name != "mooncake":
            return None
        if self.use_layerwise:
            logger.error("AscendStoreConnector reset_cache does not support Mooncake layerwise.")
            return False
        assert self.connector_scheduler is not None
        self.connector_scheduler.load_specs.clear()
        self._kv_cache_events = None
        return self.connector_scheduler.reset_store()

    def build_connector_worker_meta(self) -> AscendStoreKVConnectorWorkerMetadata | None:
        assert self.connector_worker is not None
        return self.connector_worker.build_connector_worker_meta()

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_stats()

    @classmethod
    def build_kv_connector_stats(cls, data: dict[str, Any] | None = None) -> KVConnectorStats:
        return AscendStoreKVConnectorStats(data=data or {})

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return AscendStorePromMetrics(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )

    def has_pending_push_work(self) -> bool:
        return bool(self.connector_scheduler and self.connector_scheduler.has_pending_push_work())
