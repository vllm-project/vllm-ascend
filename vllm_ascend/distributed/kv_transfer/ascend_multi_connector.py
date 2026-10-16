import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.distributed.parallel_state import get_world_group

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    SupportsHMA,
    supports_hma,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.v1.worker import mamba_utils

from vllm_ascend.utils import get_kv_cache_tensor_layers
from vllm_ascend.attention.cache_layout import (
    LAYERWISE_PREFILL_GET_GROUP_SIZE,
    LAYERWISE_PREFILL_WORKSPACE_ARENA_COUNT,
    LAYERWISE_PREFILL_WORKSPACE_BANK_COUNT,
)
from vllm_ascend.distributed.kv_transfer.layer_workspace_fence import (
    LayerWorkspaceFence,
    WorkspaceKey,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import MooncakeLayerwiseConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import uses_sfa_dspark_kv_transfer

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import KVConnectorKVEvents
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


def _set_layerwise_prefetch_device() -> None:
    """Bind NPU calls in the prefetch worker to this process's local rank."""

    local_rank = get_world_group().local_rank
    torch.npu.set_device(torch.device(f"npu:{local_rank}"))


class AscendMultiConnector(MultiConnector, SupportsHMA):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

        transfer = vllm_config.kv_transfer_config
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
        self._mamba_copy_bufs = None
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

        # Phase 2: only then may sibling hooks publish reuse-enabling work.
        for connector in self._non_slot_release_connectors:
            hook = getattr(connector, "on_kv_cache_written", None)
            if callable(hook):
                hook(layer_name)

        kv_transfer_config = vllm_config.kv_transfer_config
        extra_config = {} if kv_transfer_config is None else (
            kv_transfer_config.kv_connector_extra_config or {}
        )
        self._layerwise_workspace_fences_enabled = bool(
            extra_config.get("layerwise_host_kv_offload", False)
        ) and bool(
            kv_transfer_config
            and kv_transfer_config.is_kv_producer
            and role == KVConnectorRole.WORKER
        )
        self._execution_id = -1
        self._workspace_generation = 0
        self._workspace_fence_lock: Any | None = None
        self._workspace_fences: dict[int, LayerWorkspaceFence] = {}
        self._workspace_layer_order: tuple[str, ...] = ()
        self._workspace_layer_to_index: dict[str, int] = {}
        self._workspace_layer_to_arena: dict[str, int] = {}
        self._layer_load_futures: dict[int, Future[None]] = {}
        self._non_workspace_source_futures: dict[str, list[Future[Any]]] = {}
        self._layer_load_executor: ThreadPoolExecutor | None = None
        self._store_connector: Any | None = None
        if self._layerwise_workspace_fences_enabled:
            self._workspace_fence_lock = threading.Lock()
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
                AscendStoreConnector,
            )

            store_children = [
                connector
                for connector in self._connectors
                if isinstance(connector, AscendStoreConnector)
            ]
            p2p_children = [
                connector
                for connector in self._connectors
                if isinstance(connector, MooncakeLayerwiseConnector)
            ]
            if len(store_children) != 1 or not p2p_children:
                raise ValueError(
                    "Prefill layerwise host offload requires exactly one "
                    "AscendStoreConnector child and at least one "
                    "MooncakeLayerwiseConnector child"
                )
            if any(
                not connector.use_layerwise
                or not connector.use_layerwise_range
                for connector in store_children
            ):
                raise ValueError(
                    "Prefill layerwise host offload requires the Store child "
                    "configuration use_layerwise=true and "
                    "use_layerwise_range=true"
                )

            workspace_tensors = kv_cache_config.kv_cache_tensors[
                :LAYERWISE_PREFILL_WORKSPACE_ARENA_COUNT
            ]
            if len(workspace_tensors) != LAYERWISE_PREFILL_WORKSPACE_ARENA_COUNT:
                raise ValueError(
                    "Prefill layerwise host offload requires "
                    f"{LAYERWISE_PREFILL_WORKSPACE_ARENA_COUNT} grouped-"
                    "prefetch workspace tensors"
                )
            for arena_id, tensor_spec in enumerate(workspace_tensors):
                arena_layers = get_kv_cache_tensor_layers(tensor_spec)
                if not arena_layers:
                    raise ValueError(
                        f"Layerwise workspace arena {arena_id} has no layers"
                    )
                for layer_name in arena_layers:
                    if layer_name in self._workspace_layer_to_arena:
                        raise ValueError(
                            f"Layer {layer_name} is assigned to multiple "
                            "layerwise workspace arenas"
                        )
                    self._workspace_layer_to_arena[layer_name] = arena_id

            ordered_layers = tuple(
                layer_name
                for group in kv_cache_config.kv_cache_groups
                for layer_name in group.layer_names
                if layer_name in self._workspace_layer_to_arena
            )
            if len(ordered_layers) != len(self._workspace_layer_to_arena):
                raise ValueError(
                    "Layerwise workspace layers do not match the KV cache groups"
                )
            for layer_index, layer_name in enumerate(ordered_layers):
                expected_arena = (
                    layer_index % LAYERWISE_PREFILL_WORKSPACE_ARENA_COUNT
                )
                actual_arena = self._workspace_layer_to_arena[layer_name]
                if actual_arena != expected_arena:
                    raise ValueError(
                        f"Layerwise workspace layer {layer_name} uses arena "
                        f"{actual_arena}, expected {expected_arena}"
                    )
            self._workspace_layer_order = ordered_layers
            self._workspace_layer_to_index = {
                layer_name: layer_index
                for layer_index, layer_name in enumerate(ordered_layers)
            }
            self._store_connector = store_children[0]
            self._layer_load_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="layerwise-store-prefetch",
                initializer=_set_layerwise_prefetch_device,
            )




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

    def _wait_for_layer_load_base(self, layer_name: str) -> None:
        """v0.30.0 base routing: provider-first waits plus per-layer mamba copy."""
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
        if (copy_bufs := getattr(self, "_mamba_copy_bufs", None)) is not None:
            mamba_utils.do_mamba_copy_block_for_layer(
                copy_bufs,
                layer_name,
            )
        # Mamba state copy runs after every child connector finished this
        # layer's load so the copy never reads half-loaded state. The model
        # state no-ops for non-mamba layers and for already-copied layers.
        if (mamba_state := getattr(self, "_mamba_state", None)) is not None:
            mamba_state.do_mamba_copy_for_layer(layer_name)

    def prepare_mamba_state_copy(self, mamba_state_or_copy_bufs) -> bool:
        """Take over the mamba align pre-copy for this step.

        The V1 model runner passes its mamba copy buffers; the V2 model runner
        passes its mamba hybrid model state. Returns True when any child
        connector executes per-layer copies from its ``wait_for_layer_load``;
        see ``AscendStoreConnector.prepare_mamba_state_copy``.
        """
        if not self.requires_mamba_state_copy_after_layer_load:
            return False
        if hasattr(mamba_state_or_copy_bufs, "do_mamba_copy_for_layer"):
            self._mamba_state = mamba_state_or_copy_bufs
        else:
            mamba_utils.prepare_mamba_copy_by_layer(mamba_state_or_copy_bufs)
            self._mamba_copy_bufs = mamba_state_or_copy_bufs
        return True

    def finish_mamba_state_copy(self) -> None:
        if self._mamba_copy_bufs is not None:
            try:
                mamba_utils.finish_mamba_copy_by_layer(self._mamba_copy_bufs)
            finally:
                self._mamba_copy_bufs = None
        self._mamba_state = None

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
                or isinstance(connector, MooncakeLayerwiseConnector)
            )
            connector.update_state_after_alloc(
                request,
                blocks if needs_full_blocks else empty_blocks,
                num_external_tokens if needs_full_blocks else 0,
            )

    def _wait_for_workspace_arenas(
        self,
        arena_ids: tuple[int, ...],
    ) -> None:
        assert self._workspace_fence_lock is not None
        with self._workspace_fence_lock:
            fences: list[LayerWorkspaceFence] = []
            for arena_id in arena_ids:
                fence = self._workspace_fences.pop(arena_id, None)
                if fence is not None:
                    fences.append(fence)

        first_error: BaseException | None = None
        for fence in fences:
            try:
                fence.wait_workspace_reusable()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _wait_for_workspace_arena(self, arena_id: int) -> None:
        self._wait_for_workspace_arenas((arena_id,))

    def _workspace_group_layers(
        self,
        group_index: int,
        group_count: int = 1,
    ) -> tuple[str, ...]:
        start = group_index * LAYERWISE_PREFILL_GET_GROUP_SIZE
        end = min(
            start + group_count * LAYERWISE_PREFILL_GET_GROUP_SIZE,
            len(self._workspace_layer_order),
        )
        return self._workspace_layer_order[start:end]

    def _load_workspace_group(
        self,
        group_index: int,
        group_count: int = 1,
    ) -> None:
        layer_names = self._workspace_group_layers(group_index, group_count)
        if not layer_names:
            return
        arena_ids = tuple(
            self._workspace_layer_to_arena[layer_name]
            for layer_name in layer_names
        )
        # Pop every target fence before waiting, then drain all of them before
        # allowing Mooncake to overwrite any arena in this bank.
        self._wait_for_workspace_arenas(arena_ids)
        assert self._store_connector is not None
        self._store_connector.wait_for_layer_loads(layer_names)

    def _schedule_workspace_group_load(
        self,
        group_index: int,
        group_count: int = 1,
    ) -> None:
        group_indices = tuple(
            index
            for index in range(group_index, group_index + group_count)
            if self._workspace_group_layers(index)
        )
        if not group_indices:
            return
        if all(index in self._layer_load_futures for index in group_indices):
            return
        if any(index in self._layer_load_futures for index in group_indices):
            raise RuntimeError(
                "Layerwise Store GET overlaps an already scheduled group"
            )
        assert self._layer_load_executor is not None
        future = self._layer_load_executor.submit(
            self._load_workspace_group,
            group_index,
            group_count,
        )
        for index in group_indices:
            self._layer_load_futures[index] = future

    def _schedule_recycled_workspace_group(self, layer_index: int) -> None:
        next_layer_index = layer_index + 1
        if next_layer_index % LAYERWISE_PREFILL_GET_GROUP_SIZE != 0:
            return
        completed_group_index = layer_index // LAYERWISE_PREFILL_GET_GROUP_SIZE
        recycled_group_index = (
            completed_group_index + LAYERWISE_PREFILL_WORKSPACE_BANK_COUNT
        )
        self._schedule_workspace_group_load(recycled_group_index)

    def _drain_workspace_state(self) -> None:
        first_error: BaseException | None = None
        for group_index in sorted(self._layer_load_futures):
            try:
                self._layer_load_futures[group_index].result()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        self._layer_load_futures.clear()
        for arena_id in range(LAYERWISE_PREFILL_WORKSPACE_ARENA_COUNT):
            try:
                self._wait_for_workspace_arena(arena_id)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    @staticmethod
    def _source_futures(result: Any) -> list[Future[Any]]:
        if result is None:
            return []
        if isinstance(result, Future):
            return [result]
        if isinstance(result, (list, tuple)):
            futures = list(result)
            if not all(isinstance(future, Future) for future in futures):
                raise TypeError("Connector save hook returned a non-Future source handle")
            return futures
        raise TypeError(
            "Connector save hook must return None, Future, or a sequence of Futures"
        )

    def _abort_fence(self, fence: LayerWorkspaceFence) -> None:
        # Close registration before waiting for all readers and compute.
        fence.close_registration()
        fence.wait_workspace_reusable()

    def _drain_non_workspace_source_futures(
        self,
        layer_name: str | None = None,
    ) -> None:
        """Wait until P2P no longer reads cache outside shared arenas.

        MTP cache layers use their own PA blocks rather than a layerwise
        workspace. Their P2P connector still returns source-completion
        futures. A repeated MTP forward must wait for the previous transfer of
        the same physical layer before it can overwrite those PA blocks. The
        final drain also prevents scheduler reuse while a transfer is active.
        """

        if layer_name is None:
            futures = [
                future
                for layer_futures in self._non_workspace_source_futures.values()
                for future in layer_futures
            ]
            self._non_workspace_source_futures.clear()
        else:
            futures = self._non_workspace_source_futures.pop(layer_name, [])
        first_error: BaseException | None = None
        for future in futures:
            try:
                future.result()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    # ==============================
    # Layerwise workspace ordering
    # ==============================
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        if self._layerwise_workspace_fences_enabled:
            first_error: BaseException | None = None
            try:
                self._drain_workspace_state()
            except BaseException as error:
                first_error = error
            try:
                self._drain_non_workspace_source_futures()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            if first_error is not None:
                raise first_error
            self._execution_id += 1
        super().start_load_kv(forward_context, **kwargs)
        if self._layerwise_workspace_fences_enabled and self._workspace_layer_order:
            # Fill both grouped-prefetch banks before model execution. After
            # each bank is consumed, save_kv_layer() rolls that bank forward
            # by another group once all source readers are safe.
            self._schedule_workspace_group_load(
                0,
                group_count=LAYERWISE_PREFILL_WORKSPACE_BANK_COUNT,
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._layerwise_workspace_fences_enabled:
            self._wait_for_layer_load_base(layer_name)
            return

        if layer_name not in self._workspace_layer_to_index:
            # A speculative cache layer may execute the same physical layer
            # repeatedly.
            # Its first invocation is the P2P snapshot that Decode consumes;
            # do not let a later invocation overwrite the source PA blocks
            # until that snapshot is complete. SFA calls this hook before its
            # cache write, so this is the last safe ordering point.
            self._drain_non_workspace_source_futures(layer_name)
            self._wait_for_layer_load_base(layer_name)
            return

        layer_index = self._workspace_layer_to_index[layer_name]
        group_index = layer_index // LAYERWISE_PREFILL_GET_GROUP_SIZE
        future = self._layer_load_futures.get(group_index)
        if future is None:
            raise RuntimeError(
                f"Layerwise Store GET group was not scheduled for {layer_name}"
            )
        for connector in self._connectors:
            if connector is self._store_connector:
                future.result()
                # The group loader normally consumes the arena fence before
                # GET. Keep this check paired with the future as a no-op-safe
                # guard before handing the workspace to Attention.
                self._wait_for_workspace_arena(
                    self._workspace_layer_to_arena[layer_name]
                )
            else:
                connector.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs
    ) -> Any:
        if not self._layerwise_workspace_fences_enabled:
            # v0.30.0 base routes layer saves in two phases: slot-release
            # providers close their gates first, then siblings may publish
            # reuse-enabling work.
            for connector in self._layerwise_slot_release_providers:
                connector.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
            for connector in self._non_slot_release_connectors:
                connector.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
            return None

        layer_index = self._workspace_layer_to_index.get(layer_name)
        if layer_index is None:
            source_futures: list[Future[Any]] = []
            try:
                for connector in self._connectors:
                    result = connector.save_kv_layer(
                        layer_name,
                        kv_layer,
                        attn_metadata,
                        **kwargs,
                    )
                    futures = self._source_futures(result)
                    self._non_workspace_source_futures.setdefault(
                        layer_name, []
                    ).extend(futures)
                    source_futures.extend(futures)
            except BaseException:
                # Do not unwind while an earlier child may still be reading
                # an MTP PA block that the scheduler can subsequently reuse.
                try:
                    self._drain_non_workspace_source_futures()
                except BaseException:
                    pass
                raise
            return source_futures
        arena_id = self._workspace_layer_to_arena[layer_name]

        self._workspace_generation += 1
        key = WorkspaceKey(
            execution_id=self._execution_id,
            layer_id=layer_index,
            arena_id=arena_id,
            workspace_generation=self._workspace_generation,
        )
        fence = LayerWorkspaceFence(key)
        assert self._workspace_fence_lock is not None
        with self._workspace_fence_lock:
            if arena_id in self._workspace_fences:
                raise RuntimeError(
                    f"Layerwise workspace arena {arena_id} is still active "
                    f"before saving {layer_name}"
                )
            self._workspace_fences[arena_id] = fence
        source_futures: list[Future[Any]] = []
        try:
            # This hook runs after the layer attention and output projection
            # have been submitted to the current stream. Store GET is issued
            # by a CPU thread and is not ordered by that stream, so the next
            # layer must not overwrite the shared workspace until this event
            # completes. This is intentionally separate from kv_ready_event,
            # which lets PUT/D2RH overlap with the layer computation.
            compute_release_event = torch.npu.Event()
            compute_release_event.record()
            fence.set_compute_release_event(compute_release_event, key)
            for connector in self._connectors:
                result = connector.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
                # Register each child result before invoking the next child.
                # If a later connector raises, _abort_fence() must still wait
                # for an earlier Store/P2P source reader using the shared
                # workspace.
                for future in self._source_futures(result):
                    fence.add_source_future(future, key)
                    source_futures.append(future)
            fence.close_registration()
        except BaseException:
            self._abort_fence(fence)
            with self._workspace_fence_lock:
                if self._workspace_fences.get(arena_id) is fence:
                    self._workspace_fences.pop(arena_id)
            raise
        self._schedule_recycled_workspace_group(layer_index)
        return source_futures

    def wait_for_save(self):
        if not self._layerwise_workspace_fences_enabled:
            return super().wait_for_save()
        first_error: BaseException | None = None
        try:
            self._drain_workspace_state()
        except BaseException as error:
            first_error = error
        try:
            self._drain_non_workspace_source_futures()
        except BaseException as error:
            if first_error is None:
                first_error = error
        # Even when P2P source completion fails, drain Store child cleanup so
        # a failed range session revokes open objects and releases leases.
        for connector in self._connectors:
            try:
                connector.wait_for_save()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def shutdown(self):
        first_error: BaseException | None = None
        if self._layerwise_workspace_fences_enabled:
            try:
                self._drain_workspace_state()
            except BaseException as error:
                first_error = error
            try:
                self._drain_non_workspace_source_futures()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            assert self._layer_load_executor is not None
            self._layer_load_executor.shutdown(wait=True, cancel_futures=True)
        try:
            super().shutdown()
        except BaseException as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise first_error

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
