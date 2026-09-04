from typing import TYPE_CHECKING, Any, cast

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    SupportsHMA,
    supports_hma,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class AscendMultiConnector(MultiConnector, SupportsHMA):
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
        self._configure_layerwise_reuse_completion()

    def _configure_layerwise_reuse_completion(self) -> None:
        # Producers that report when a shared physical KV slot is safe to reuse.
        self._layerwise_slot_release_providers = [
            connector
            for connector in self._connectors
            if getattr(connector, "is_producer", False)
            and getattr(connector, "connector_worker", None) is not None
            and getattr(connector, "supports_layerwise_buffer_reuse", False)
            and callable(getattr(connector, "wait_for_layer_reuse", None))
        ]
        # All remaining connectors, which run after the slot-release providers.
        self._non_slot_release_connectors = [
            connector
            for connector in self._connectors
            if all(connector is not provider for provider in self._layerwise_slot_release_providers)
        ]
        self._external_slot_release_sink_configured = False
        if not self._layerwise_slot_release_providers:
            return
        for connector in self._connectors:
            set_waiter = getattr(connector, "set_external_slot_release_waiter", None)
            if callable(set_waiter) and set_waiter(self._wait_for_external_slot_release) is not False:
                self._external_slot_release_sink_configured = True

    def _wait_for_external_slot_release(self, layer_idx: int) -> None:
        for provider in self._layerwise_slot_release_providers:
            provider.wait_for_layer_reuse(layer_idx)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if getattr(self, "_external_slot_release_sink_configured", False):
            # AscendStore owns the layer-entry reuse wait after accepting the
            # composite waiter, so provider layer-entry waits are redundant.
            connectors = self._non_slot_release_connectors
        else:
            # Without a sink, preserve the original protection by waiting on
            # providers before any sibling connector can write shared slots.
            connectors = [*self._layerwise_slot_release_providers, *self._non_slot_release_connectors]
        for connector in connectors:
            connector.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer,
        attn_metadata: Any,
        **kwargs,
    ) -> None:
        # Phase 1: providers must close any new slot gate before returning.
        for connector in self._layerwise_slot_release_providers:
            connector.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
        # Phase 2: siblings may now publish work that can enable slot reuse.
        for connector in self._non_slot_release_connectors:
            connector.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def on_kv_cache_written(self, layer_name: str = "") -> None:
        # Phase 1: providers close their gates at the earliest cache-write hook.
        for connector in self._layerwise_slot_release_providers:
            hook = getattr(connector, "on_kv_cache_written", None)
            if callable(hook):
                hook(layer_name)
        # Phase 2: only then may sibling hooks publish reuse-enabling work.
        for connector in self._non_slot_release_connectors:
            hook = getattr(connector, "on_kv_cache_written", None)
            if callable(hook):
                hook(layer_name)

    def rebuild_kv_transfer_endpoint(self, local_ip: str, new_engine_id: str | None = None) -> None:
        """Rebuild snapshot-sensitive endpoints owned by sub-connectors."""
        for connector in self._connectors:
            rebuild = getattr(connector, "rebuild_kv_transfer_endpoint", None)
            if callable(rebuild):
                rebuild(local_ip, new_engine_id)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        empty_blocks = blocks.new_empty()
        for i, connector in enumerate(self._connectors):
            needs_full_blocks = i == chosen_connector or bool(
                getattr(connector, "requires_full_blocks_on_update_after_alloc", False)
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
        for c in self._connectors:
            hook = getattr(c, "update_state_before_preempt", None)
            if hook is not None:
                offloaded = bool(hook(request, block_ids, num_computed_tokens)) or offloaded
        return offloaded

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if not self._all_support_hma:
            assert len(block_ids) == 1, "HMA with multiple kv_cache_groups requires all sub-connectors to support HMA"
            return super().request_finished(request, block_ids[0])

        async_saves = 0
        kv_txfer_params = None
        for c in self._connectors:
            async_save, txfer_params = cast(SupportsHMA, c).request_finished_all_groups(request, block_ids)
            if async_save:
                async_saves += 1
            if txfer_params is not None:
                if kv_txfer_params is not None:
                    raise RuntimeError("Only one connector can produce KV transfer params")
                kv_txfer_params = txfer_params
        if async_saves > 1:
            self._extra_async_saves[request.request_id] = async_saves - 1

        self._requests_to_connector.pop(request.request_id, None)

        return async_saves > 0, kv_txfer_params
