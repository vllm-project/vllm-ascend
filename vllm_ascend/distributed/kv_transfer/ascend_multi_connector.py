from typing import TYPE_CHECKING, Any, cast

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    SupportsHMA,
    supports_hma,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import uses_sfa_dspark_kv_transfer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hisparse.layout import has_hisparse_cache_layout

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import KVConnectorKVEvents
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hisparse.draft_host_mirror import (
        HiSparseDraftHostMirrorSaveHook,
    )


class AscendMultiConnector(MultiConnector, SupportsHMA):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        transfer = vllm_config.kv_transfer_config
        assert transfer is not None
        children = transfer.kv_connector_extra_config.get("connectors", ())
        names = {child.get("kv_connector") for child in children}
        colocated_pair = {"HiSparseConnector", "AscendStoreConnector"}
        if "HiSparseConnector" in names and "MooncakeConnectorStoreV1" in names:
            raise ValueError("Use AscendStoreConnector when combining the pool with HiSparseConnector.")
        self._hisparse_store_shutdown_order: tuple[int, int] | None = None
        if colocated_pair <= names:
            # Both children share the local Host Source and Indexer import path.
            # Reject other transfer topologies before creating any child.
            if transfer.kv_role != "kv_both" or any(child.get("kv_role") != "kv_both" for child in children):
                raise ValueError(
                    "HiSparse with AscendStore requires kv_role='kv_both' on MultiConnector and all children. "
                    "PD disaggregation is unsupported."
                )
            if names != colocated_pair:
                raise ValueError(
                    "HiSparse with AscendStore supports only HiSparseConnector and AscendStoreConnector children."
                )
            if len(children) != len(colocated_pair):
                raise ValueError(
                    "HiSparse pooling requires exactly one HiSparseConnector and one AscendStoreConnector."
                )
            if vllm_config.parallel_config.prefill_context_parallel_size != 1:
                raise ValueError("HiSparse with AscendStore requires prefill_context_parallel_size=1.")
            nnodes = vllm_config.parallel_config.nnodes
            if role == KVConnectorRole.WORKER and nnodes == 1:
                from vllm.distributed.parallel_state import get_node_count

                nnodes = get_node_count()
            if nnodes != 1:
                raise ValueError(
                    f"HiSparse with AscendStore requires single-node colocated Prefill and Decode, got {nnodes} nodes."
                )
            child_names = [child["kv_connector"] for child in children]
            self._hisparse_store_shutdown_order = (
                child_names.index("AscendStoreConnector"),
                child_names.index("HiSparseConnector"),
            )

        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

        self._requires_full_dspark_prompt = (
            uses_sfa_dspark_kv_transfer(vllm_config) and transfer.is_kv_producer and not transfer.is_kv_consumer
        )
        self._dspark_prefix_provider_index: int | None = None
        if self._requires_full_dspark_prompt:
            self.configure_dspark_prefix_cache(vllm_config)

        self._all_support_hma = all(supports_hma(c) for c in self._connectors)
        assert vllm_config.scheduler_config.disable_hybrid_kv_cache_manager or self._all_support_hma, (
            "HMA should not be enabled unless all sub-connectors support it"
        )
        self._configure_layerwise_reuse_completion()
        self._draft_host_mirror_save_hook: HiSparseDraftHostMirrorSaveHook | None = None
        if (
            role == KVConnectorRole.WORKER
            and has_hisparse_cache_layout(kv_cache_config)
            and vllm_config.speculative_config is not None
        ):
            self._configure_draft_host_mirror_save_hook()
        # Handle to the (V2) mamba hybrid model state while its per-layer
        # align pre-copy is deferred behind a layerwise child connector.
        self._mamba_state: Any = None
        self.requires_mamba_state_copy_after_layer_load = any(
            getattr(
                connector,
                "requires_mamba_state_copy_after_layer_load",
                False,
            )
            for connector in self._connectors
        )

        self.requires_mamba_state_copy_after_load = any(
            getattr(connector, "requires_mamba_state_copy_after_load", False) for connector in self._connectors
        )

    def _configure_draft_host_mirror_save_hook(self) -> None:
        from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import HiSparseConnector

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hisparse.draft_host_mirror import (
            HiSparseDraftHostMirrorSaveHook,
        )

        source = next((child for child in self._connectors if isinstance(child, HiSparseConnector)), None)
        stores = [child for child in self._connectors if isinstance(child, AscendStoreConnector)]
        if source is not None and stores:
            assert source.connector_worker is not None
            self._draft_host_mirror_save_hook = HiSparseDraftHostMirrorSaveHook(source.connector_worker, stores)

    def wait_for_save(self):
        if self._draft_host_mirror_save_hook is not None:
            self._draft_host_mirror_save_hook.flush_host_mirror_if_needed()
        super().wait_for_save()

    def configure_dspark_prefix_cache(self, vllm_config: "VllmConfig") -> bool:
        """Select a store that restores both target and persistent draft KV."""
        providers = []
        for index, connector in enumerate(self._connectors):
            configure = getattr(connector, "configure_dspark_prefix_cache", None)
            if callable(configure) and configure(vllm_config):
                providers.append(index)
        if len(providers) > 1:
            raise ValueError("DSpark prefix reuse requires exactly one paired target/draft store")
        self._dspark_prefix_provider_index = providers[0] if providers else None
        if providers:
            self._requires_full_dspark_prompt = False
        return bool(providers)

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

    def shutdown(self) -> None:
        if self._hisparse_store_shutdown_order is not None:
            store_index, hisparse_index = self._hisparse_store_shutdown_order
            # Join the Store's transfers before freeing HiSparse Host memory.
            # If Store shutdown raises, the Host registration must stay live.
            self._connectors[store_index].shutdown()
            self._connectors[hisparse_index].shutdown()
            return
        # Release provider waits before joining the store's receive threads.
        self._connectors[:] = [*self._layerwise_slot_release_providers, *self._non_slot_release_connectors]
        super().shutdown()

    def wait_for_layer_load_ready(self, layer_name: str) -> None:
        """Observe child completion without advancing their layer cursors."""
        for connector in self._connectors:
            wait_ready = getattr(connector, "wait_for_layer_load_ready", None)
            if wait_ready is not None:
                wait_ready(layer_name)

    def get_copy_sfa_slot_bindings(self) -> dict[str, int]:
        """Expose request-owned Copy-SFA rows through MultiConnector."""
        bindings: dict[str, int] = {}
        for connector in self._connectors:
            getter = getattr(connector, "get_copy_sfa_slot_bindings", None)
            if not callable(getter):
                continue
            for req_id, pool_slot in getter().items():
                previous = bindings.setdefault(req_id, pool_slot)
                if previous != pool_slot:
                    raise RuntimeError(
                        "MultiConnector children returned conflicting Copy-SFA "
                        f"slots for request {req_id}: {previous} and {pool_slot}"
                    )
        return bindings

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
        # Mamba state copy runs after every child connector finished this
        # layer's load so the copy never reads half-loaded state. The model
        # state no-ops for non-mamba layers and for already-copied layers.
        if (mamba_state := getattr(self, "_mamba_state", None)) is not None:
            mamba_state.do_mamba_copy_for_layer(layer_name)

    def prepare_mamba_state_copy(self, mamba_state_or_copy_bufs) -> bool:
        """Take over the mamba align pre-copy for this step.

        Returns True when any child connector executes per-layer copies from its
        ``wait_for_layer_load``; see ``AscendStoreConnector.prepare_mamba_state_copy``.
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
        # A target-only child cannot initialize the skipped draft context.
        if getattr(self, "_requires_full_dspark_prompt", False):
            return 0, False
        provider_index = getattr(self, "_dspark_prefix_provider_index", None)
        if provider_index is not None:
            tokens, load_async = self._connectors[provider_index].get_num_new_matched_tokens(
                request, num_computed_tokens
            )
            if tokens is not None and tokens > 0:
                self._requests_to_connector[request.request_id] = provider_index
            return tokens, load_async

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

    def get_kv_connector_kv_cache_events(self) -> "KVConnectorKVEvents | None":
        """Collect worker-side KV cache events from all child connectors."""
        combined = None
        for connector in self._connectors:
            events = connector.get_kv_connector_kv_cache_events()
            if events is None:
                continue
            if combined is None:
                combined = events
            else:
                combined.add_events(events.get_all_events())
                combined.increment_workers(events.get_number_of_workers())
        return combined
