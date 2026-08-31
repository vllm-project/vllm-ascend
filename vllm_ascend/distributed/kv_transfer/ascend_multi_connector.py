from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    SupportsHMA,
    supports_hma,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
)
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnector,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class AscendMultiKVConnectorMetadata(MultiKVConnectorMetadata):
    """MultiConnector metadata carrying the expected async-save sources."""

    async_save_sources: dict[str, tuple[int, ...]] | None = None


class AscendMultiConnector(MultiConnector, SupportsHMA):
    """Ascend MultiConnector with idempotent async completion aggregation."""

    _MAX_TRACKED_FINISHED_REQUESTS = 4096

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

        self._all_support_hma = all(supports_hma(c) for c in self._connectors)
        assert vllm_config.scheduler_config.disable_hybrid_kv_cache_manager or self._all_support_hma, (
            "HMA should not be enabled unless all sub-connectors support it"
        )
        self._configure_layerwise_pd_completion()

        # Track the child indexes that have completed each request. The
        # upstream count-only protocol cannot distinguish a duplicate event
        # from one connector from a completion by another connector.
        self._finished_sending_connectors: dict[str, set[int]] = {}
        self._finished_recving_emitted: OrderedDict[str, None] = OrderedDict()
        self._async_save_sources: dict[str, tuple[int, ...]] = {}
        self._eligible_finished_req_ids: set[str] = set()

    def _configure_layerwise_pd_completion(self) -> None:
        self._pd_completion_connector = next(
            (
                connector
                for connector in self._connectors
                if getattr(connector, "is_producer", False)
                and getattr(connector, "connector_worker", None) is not None
                and callable(getattr(connector, "wait_for_layer_send", None))
            ),
            None,
        )
        if self._pd_completion_connector is not None:
            for connector in self._connectors:
                set_waiter = getattr(connector, "set_layerwise_pd_transfer_waiter", None)
                if callable(set_waiter):
                    set_waiter(self._pd_completion_connector.wait_for_layer_send)

    def _pd_connector_first(self):
        provider = getattr(self, "_pd_completion_connector", None)
        if provider is not None:
            yield provider
        yield from (connector for connector in self._connectors if connector is not provider)

    def wait_for_layer_load(self, layer_name: str) -> None:
        for connector in self._pd_connector_first():
            connector.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer,
        attn_metadata: Any,
        **kwargs,
    ) -> None:
        for connector in self._pd_connector_first():
            connector.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def on_kv_cache_written(self, layer_name: str = "") -> None:
        for connector in self._pd_connector_first():
            hook = getattr(connector, "on_kv_cache_written", None)
            if callable(hook):
                hook(layer_name)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        empty_blocks = blocks.new_empty()
        for i, connector in enumerate(self._connectors):
            needs_full_blocks = (
                i == chosen_connector
                or isinstance(connector, MooncakeLayerwiseConnector)
                or bool(getattr(connector, "requires_full_blocks_on_update_after_alloc", False))
            )
            connector.update_state_after_alloc(
                request,
                blocks if needs_full_blocks else empty_blocks,
                num_external_tokens if needs_full_blocks else 0,
            )

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # Recompute offload may contain an unhashed partial block that other
        # prefix-cache connectors cannot restore. Give its request state
        # priority regardless of connector ordering.
        for i, connector in enumerate(self._connectors):
            has_preempted_request = getattr(connector, "has_preempted_request", None)
            if has_preempted_request is None or not has_preempted_request(request.request_id):
                continue
            tokens, load_async = connector.get_num_new_matched_tokens(request, num_computed_tokens)
            if tokens is None:
                return None, False
            if tokens > 0:
                self._requests_to_connector[request.request_id] = i
                return tokens, load_async
            break

        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_before_preempt(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
        num_computed_tokens: int,
    ) -> bool:
        offloaded = False
        for connector in self._connectors:
            hook = getattr(connector, "update_state_before_preempt", None)
            if hook is not None:
                offloaded = bool(hook(request, block_ids, num_computed_tokens)) or offloaded
        return offloaded

    def _remember_finished_request(self, emitted_requests: OrderedDict[str, None], request_id: str) -> None:
        emitted_requests[request_id] = None
        emitted_requests.move_to_end(request_id)
        if len(emitted_requests) > self._MAX_TRACKED_FINISHED_REQUESTS:
            emitted_requests.popitem(last=False)

    def _is_announced_finished_request(self, request_id: str) -> bool:
        if request_id in self._eligible_finished_req_ids:
            return True
        logger.warning("Ignoring KV completion for unannounced request %s", request_id)
        return False

    def bind_connector_metadata(self, connector_metadata: MultiKVConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        if isinstance(connector_metadata, AscendMultiKVConnectorMetadata):
            self._async_save_sources.update(connector_metadata.async_save_sources or {})

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> MultiKVConnectorMetadata:
        metadata = super().build_connector_meta(scheduler_output)
        if not self._async_save_sources:
            return metadata

        ascend_metadata = AscendMultiKVConnectorMetadata(
            metadata=metadata.metadata,
            extra_async_saves=metadata.extra_async_saves,
            async_save_sources=self._async_save_sources,
        )
        self._async_save_sources = {}
        return ascend_metadata

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        """Emit async completions once, after every expected child completes."""
        # Only asynchronous saves need eligibility. This avoids retaining all
        # synchronous terminal request IDs indefinitely.
        self._eligible_finished_req_ids.update(
            request_id for request_id in finished_req_ids if request_id in self._async_save_sources
        )
        finished_sending: set[str] = set()
        finished_recving: set[str] = set()

        for connector_index, connector in enumerate(self._connectors):
            sending, recving = connector.get_finished(finished_req_ids)

            for req_id in sending or ():
                expected_sources = self._async_save_sources.get(req_id)
                if expected_sources is None:
                    logger.warning("Ignoring KV send completion for unknown request %s", req_id)
                    continue
                if connector_index not in expected_sources:
                    logger.warning(
                        "Ignoring KV send completion from unexpected connector %d for request %s",
                        connector_index,
                        req_id,
                    )
                    continue

                completed_connectors = self._finished_sending_connectors.setdefault(req_id, set())
                if connector_index in completed_connectors:
                    logger.debug(
                        "Ignoring duplicate KV send completion from connector %d for request %s",
                        connector_index,
                        req_id,
                    )
                    continue
                completed_connectors.add(connector_index)

                if not self._is_announced_finished_request(req_id):
                    continue
                if not set(expected_sources).issubset(completed_connectors):
                    continue

                self._finished_sending_connectors.pop(req_id, None)
                self._async_save_sources.pop(req_id, None)
                self._extra_async_saves.pop(req_id, None)
                self._eligible_finished_req_ids.discard(req_id)
                finished_sending.add(req_id)

            for req_id in recving or ():
                if req_id in self._finished_recving_emitted:
                    logger.debug("Ignoring duplicate KV receive completion for request %s", req_id)
                    continue
                finished_recving.add(req_id)
                self._remember_finished_request(self._finished_recving_emitted, req_id)

        # A child may finish before the scheduler announces the request. Keep
        # its source set and emit on the first later call that announces it.
        for req_id, completed_connectors in list(self._finished_sending_connectors.items()):
            expected_sources = self._async_save_sources.get(req_id)
            if (
                expected_sources is None
                or req_id not in self._eligible_finished_req_ids
                or not set(expected_sources).issubset(completed_connectors)
            ):
                continue
            self._finished_sending_connectors.pop(req_id, None)
            self._async_save_sources.pop(req_id, None)
            self._extra_async_saves.pop(req_id, None)
            self._eligible_finished_req_ids.discard(req_id)
            finished_sending.add(req_id)

        return finished_sending or None, finished_recving or None

    def shutdown(self) -> None:
        try:
            super().shutdown()
        finally:
            self._finished_sending_connectors.clear()
            self._extra_async_saves.clear()
            self._async_save_sources.clear()
            self._finished_recving_emitted.clear()
            self._eligible_finished_req_ids.clear()

    def _aggregate_request_finished(
        self,
        request: "Request",
        per_connector_fn: Callable[[Any], tuple[bool, dict[str, Any] | None]],
    ) -> tuple[bool, dict[str, Any] | None]:
        async_save_sources: list[int] = []
        kv_txfer_params = None
        for connector_index, connector in enumerate(self._connectors):
            async_save, txfer_params = per_connector_fn(connector)
            if async_save:
                async_save_sources.append(connector_index)
            if txfer_params is not None:
                if kv_txfer_params is not None:
                    raise RuntimeError("Only one connector can produce KV transfer params")
                kv_txfer_params = txfer_params

        if len(async_save_sources) > 1:
            self._extra_async_saves[request.request_id] = len(async_save_sources) - 1
        else:
            self._extra_async_saves.pop(request.request_id, None)
        if async_save_sources:
            self._async_save_sources[request.request_id] = tuple(async_save_sources)
        else:
            self._async_save_sources.pop(request.request_id, None)

        self._requests_to_connector.pop(request.request_id, None)
        return bool(async_save_sources), kv_txfer_params

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if not self._all_support_hma:
            assert len(block_ids) == 1, "HMA with multiple kv_cache_groups requires all sub-connectors to support HMA"
            return self.request_finished(request, block_ids[0])

        return self._aggregate_request_finished(
            request,
            lambda connector: cast(SupportsHMA, connector).request_finished_all_groups(request, block_ids),
        )

    def request_finished(
        self,
        request: "Request",
        blocks: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._aggregate_request_finished(
            request,
            lambda connector: connector.request_finished(request, blocks),
        )
