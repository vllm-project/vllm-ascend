from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.models.utils import extract_layer_index

from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.core.kv_cache_placement import map_kvpp_layers_to_owners
from vllm_ascend.distributed.kv_transfer.kv_pool.memfabric_mte_transport import (
    KVPPActivePages,
    MemFabricMTEKVPPTransport,
)
from vllm_ascend.distributed.parallel_state import get_kvpp_group

# Dedicated kvpp CPU group: one in-flight prefetch, so tags only distinguish
# the ready/done handshake, not the transformer layer.
_KVPP_READY_TAG = 0
_KVPP_DONE_TAG = 1


def build_layer_cache_bundles(
    layer_owner_ranks: dict[str, int],
    attention_layer_names: tuple[str, ...] | None,
) -> dict[str, tuple[str, ...]]:
    """Group the KV caches consumed by each executable attention layer.

    For example, if layer 0 has a Target KV cache and an indexer cache, both
    are transferred when its attention implementation runs::

        {
            "model.layers.0.self_attn.attn": (
                "model.layers.0.self_attn.attn",
                "model.layers.0.self_attn.indexer.k_cache",
            )
        }
    """
    layers = tuple(attention_layer_names or layer_owner_ranks)
    if attention_layer_names is None:
        return {layer_name: (layer_name,) for layer_name in layers}

    cache_layers_by_index: dict[int, list[str]] = {}
    for cache_layer_name in sorted(
        layer_owner_ranks,
        key=lambda name: (extract_layer_index(name), name),
    ):
        cache_layers_by_index.setdefault(extract_layer_index(cache_layer_name), []).append(cache_layer_name)

    return {
        layer_name: tuple(cache_layers_by_index[extract_layer_index(layer_name)])
        for layer_name in attention_layer_names
    }


@dataclass(frozen=True)
class KVPPCacheLayout:
    """Model-runner cache objects and their physical block layout."""

    layer_caches: dict[str, Any]
    physical_blocks_per_kv_block: Sequence[int]
    tokens_per_block: Sequence[int]


class KVPPRuntime:
    """Model-runner facing KVPP placement and scheduling glue."""

    def __init__(
        self,
        scheduler: KVPPScheduler | None = None,
        managed_cache_group_index: int = 0,
    ) -> None:
        self.scheduler = scheduler
        self.managed_cache_group_index = managed_cache_group_index

    @classmethod
    def create_from_kv_cache(
        cls,
        *,
        vllm_config: Any,
        kv_cache_config: Any,
        block_tables: Any,
        static_forward_context: dict[str, Any],
    ) -> KVPPRuntime:
        if KVPPConfig.from_vllm_config(vllm_config).size <= 1:
            return cls()

        layer_caches: dict[str, Any] = {}
        for cache_group in kv_cache_config.kv_cache_groups:
            for layer_name in cache_group.layer_names:
                module = static_forward_context.get(layer_name)
                if module is not None and hasattr(module, "kv_cache"):
                    layer_caches[layer_name] = module.kv_cache
        return cls.create_from_cache_layout(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            static_forward_context=static_forward_context,
            cache_layout=KVPPCacheLayout(
                layer_caches=layer_caches,
                physical_blocks_per_kv_block=block_tables.blocks_per_kv_block,
                tokens_per_block=block_tables.kernel_block_sizes,
            ),
        )

    @classmethod
    def create_from_cache_layout(
        cls,
        *,
        vllm_config: Any,
        kv_cache_config: Any,
        static_forward_context: dict[str, Any],
        cache_layout: KVPPCacheLayout,
    ) -> KVPPRuntime:
        """Create the runtime after a model runner has normalized its cache layout."""
        layer_names = tuple(
            dict.fromkeys(
                layer_name for cache_group in kv_cache_config.kv_cache_groups for layer_name in cache_group.layer_names
            )
        )
        layer_owner_ranks = map_kvpp_layers_to_owners(vllm_config, layer_names)

        managed_layer_names = set(layer_owner_ranks)
        managed_cache_group_indices = {
            group_index
            for group_index, cache_group in enumerate(kv_cache_config.kv_cache_groups)
            if managed_layer_names.intersection(cache_group.layer_names)
        }
        if len(managed_cache_group_indices) != 1:
            raise ValueError(
                f"KVPP managed layers must belong to one cache group, got {sorted(managed_cache_group_indices)}."
            )
        managed_cache_group_index = managed_cache_group_indices.pop()

        attention_impls: dict[str, Any] = {}
        managed_kv_caches: dict[str, Any] = {}
        for layer_name in layer_owner_ranks:
            if layer_name not in cache_layout.layer_caches:
                raise RuntimeError(f"KVPP could not find the cache bound to layer {layer_name!r}.")
            managed_kv_caches[layer_name] = cache_layout.layer_caches[layer_name]
            module = static_forward_context.get(layer_name)
            impl = getattr(module, "impl", None)
            if impl is not None and hasattr(impl, "layerwise_kv_cache_hook"):
                attention_impls[layer_name] = impl
        if not attention_impls:
            raise RuntimeError("KVPP requires an MLA or SFA attention implementation with a layer cache hook.")

        num_physical_blocks = (
            kv_cache_config.num_blocks * cache_layout.physical_blocks_per_kv_block[managed_cache_group_index]
        )
        tokens_per_block = cache_layout.tokens_per_block[managed_cache_group_index]
        max_blocks_per_request = (vllm_config.model_config.max_model_len + tokens_per_block - 1) // tokens_per_block
        max_active_pages = min(
            num_physical_blocks,
            vllm_config.scheduler_config.max_num_seqs * max_blocks_per_request,
        )
        kvpp_group = get_kvpp_group()
        scheduler = KVPPScheduler(
            kvpp_group=kvpp_group,
            layer_owner_ranks=layer_owner_ranks,
            kv_caches=managed_kv_caches,
            tokens_per_block=tokens_per_block,
            num_physical_blocks=num_physical_blocks,
            max_active_pages=max_active_pages,
            transport=MemFabricMTEKVPPTransport(
                kvpp_group,
                num_physical_blocks,
            ),
            attention_layer_names=tuple(attention_impls),
        )
        for impl in attention_impls.values():
            impl.layerwise_kv_cache_hook = scheduler
        return cls(
            scheduler=scheduler,
            managed_cache_group_index=managed_cache_group_index,
        )

    def prepare_forward(
        self,
        block_tables: tuple[torch.Tensor, ...],
        num_computed_tokens: Any,
    ) -> None:
        if self.scheduler is None:
            return
        self.scheduler.schedule_forward(
            block_tables[self.managed_cache_group_index],
            num_computed_tokens,
        )

    def complete_forward(self) -> None:
        if self.scheduler is None:
            return
        self.scheduler.complete_forward()


def select_active_pages(
    block_table: torch.Tensor,
    num_computed_tokens: Any,
    tokens_per_block: int,
    num_physical_blocks: int,
) -> KVPPActivePages:
    """Return fixed-shape device pages containing computed KV cache.

    The original block table is read only. Invalid columns and duplicate page
    IDs become masked slots instead of being compacted through the host.
    """
    computed_token_counts = torch.as_tensor(
        num_computed_tokens,
        dtype=torch.int64,
        device=block_table.device,
    ).flatten()
    active_block_table = block_table[: computed_token_counts.shape[0]].to(dtype=torch.int64)
    block_columns = torch.arange(
        active_block_table.shape[1],
        dtype=torch.int64,
        device=block_table.device,
    )
    pages_per_request = torch.div(
        computed_token_counts + tokens_per_block - 1,
        tokens_per_block,
        rounding_mode="floor",
    )
    covered_slots = block_columns.unsqueeze(0) < pages_per_request.unsqueeze(1)
    valid_slots = covered_slots & (active_block_table >= 0) & (active_block_table < num_physical_blocks)
    invalid_page_id = torch.full_like(active_block_table, num_physical_blocks)
    physical_page_ids = torch.sort(torch.where(valid_slots, active_block_table, invalid_page_id).flatten()).values
    first_occurrence_mask = torch.ones_like(physical_page_ids, dtype=torch.bool)
    if physical_page_ids.numel() > 1:
        first_occurrence_mask[1:] = physical_page_ids[1:] != physical_page_ids[:-1]
    valid_page_mask = first_occurrence_mask & (physical_page_ids < num_physical_blocks)
    staging_page_indices = torch.cumsum(valid_page_mask.to(dtype=torch.int64), dim=0) - 1
    return KVPPActivePages(
        physical_page_ids=physical_page_ids,
        valid_page_mask=valid_page_mask,
        staging_page_indices=staging_page_indices,
    )


class KVPPScheduler:
    """Schedule stream-ordered layer prefetch over an injected transport.

    Owned layers use persistent KV caches. Non-owned layers are already bound
    by vLLM's planner to one of two alternating full-size scratch caches.
    Active pages are pushed into the same physical block IDs, preserving the
    original block table and slot mapping. The dual buffers let layer N+1 be
    filled while layer N attention still reads its own scratch cache.
    """

    def __init__(
        self,
        kvpp_group: GroupCoordinator,
        layer_owner_ranks: dict[str, int],
        kv_caches: dict[str, Any],
        num_physical_blocks: int,
        tokens_per_block: int,
        max_active_pages: int,
        transport: MemFabricMTEKVPPTransport,
        attention_layer_names: tuple[str, ...] | None = None,
    ) -> None:
        self.kvpp_group = kvpp_group
        self.layer_owner_ranks = layer_owner_ranks
        self.num_physical_blocks = num_physical_blocks
        self.tokens_per_block = tokens_per_block
        self.transport = transport
        self.layer_cache_bundles = build_layer_cache_bundles(
            self.layer_owner_ranks,
            attention_layer_names,
        )
        self.attention_layer_names = tuple(self.layer_cache_bundles)
        self._next_attention_layer_index = 0
        self._active_pages: KVPPActivePages | None = None
        self._kv_transfer_stream: Any | None = None
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._prefetch_future: Future[None] | None = None
        self._npu_device_id: int | None = None
        self.transport.initialize_transport(
            kv_caches,
            tuple(self.layer_cache_bundles.values()),
            max_active_pages,
        )
        if self.kvpp_group.world_size > 1:
            self._npu_device_id = torch.npu.current_device()
            self._kv_transfer_stream = torch.npu.Stream()
            # One transfer may be in flight. Serializing jobs also preserves
            # point-to-point notification order when layer ownership changes.
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kvpp-prefetch")

    def schedule_forward(
        self,
        block_table: torch.Tensor,
        num_computed_tokens: Any,
    ) -> None:
        if self._active_pages is not None:
            raise RuntimeError("KVPP cannot schedule a new forward while the previous forward is active.")
        self._active_pages = select_active_pages(
            block_table,
            num_computed_tokens,
            self.tokens_per_block,
            self.num_physical_blocks,
        )
        self._next_attention_layer_index = 0
        self.start_layer_prefetch(self.attention_layer_names[0])

    def complete_forward(self) -> None:
        if self._active_pages is None:
            return
        self._active_pages = None
        self._next_attention_layer_index = 0

    def wait_for_layer(self, layer_name: str) -> None:
        """Order cache use, then prefetch the next layer before attention."""
        if self._active_pages is None:
            raise RuntimeError("KVPP batch metadata was not prepared before attention.")
        if self._next_attention_layer_index >= len(self.attention_layer_names):
            raise RuntimeError(f"KVPP received an extra attention layer {layer_name!r}.")
        expected_layer = self.attention_layer_names[self._next_attention_layer_index]
        if layer_name != expected_layer:
            raise RuntimeError(f"KVPP expected attention layer {expected_layer!r}, got {layer_name!r}.")
        if self._prefetch_future is not None:
            # Eager path: this blocks only for the residual transfer time
            # because this layer was prefetched while earlier work executed.
            layer_index = extract_layer_index(layer_name)
            with torch.profiler.record_function(f"kvpp.wait.previous_layer.layer_{layer_index}"):
                self._prefetch_future.result()
        self._prefetch_future = None
        self._next_attention_layer_index += 1
        if self._next_attention_layer_index < len(self.attention_layer_names):
            self.start_layer_prefetch(self.attention_layer_names[self._next_attention_layer_index])

    def start_layer_prefetch(self, layer_name: str) -> None:
        self._prefetch_future = None
        if self.kvpp_group.world_size <= 1:
            return
        if self._prefetch_executor is None or self._kv_transfer_stream is None:
            raise RuntimeError("KVPP prefetch resources were not initialized.")
        # All ranks publish a local safe point. Alternating layers use distinct
        # buffers. When a buffer cycles back after two layers, this event is
        # ordered after all earlier attention work on the compute stream, so
        # the owner cannot overwrite a buffer that is still being read.
        scratch_ready = torch.npu.Event()
        scratch_ready.record(torch.npu.current_stream())
        pages = self._active_pages
        if pages is None:
            raise RuntimeError("KVPP active pages were not prepared before prefetch.")
        self._prefetch_future = self._prefetch_executor.submit(
            self.run_layer_prefetch,
            layer_name,
            pages,
            scratch_ready,
        )

    def run_layer_prefetch(
        self,
        layer_name: str,
        active_pages: KVPPActivePages,
        scratch_ready: Any,
    ) -> None:
        """Run safe-point and completion notification off the compute thread."""
        if self._kv_transfer_stream is None:
            raise RuntimeError("KVPP communication stream was not initialized.")
        if self._npu_device_id is not None:
            torch.npu.set_device(self._npu_device_id)

        owner_kvpp_rank = self.layer_owner_ranks[layer_name]
        local_kvpp_rank = self.kvpp_group.rank_in_group
        owner_global_rank = self.kvpp_group.ranks[owner_kvpp_rank]
        layer_index = extract_layer_index(layer_name)
        token = torch.ones(1, dtype=torch.uint8, device="cpu")

        with torch.profiler.record_function(f"kvpp.comm_total.layer_{layer_index}"):
            scratch_ready.synchronize()

            if local_kvpp_rank != owner_kvpp_rank:
                dist.send(
                    token,
                    dst=owner_global_rank,
                    group=self.kvpp_group.cpu_group,
                    tag=_KVPP_READY_TAG,
                )
                dist.recv(
                    token,
                    src=owner_global_rank,
                    group=self.kvpp_group.cpu_group,
                    tag=_KVPP_DONE_TAG,
                )
                with torch.profiler.record_function(f"kvpp.transport_receive.layer_{layer_index}"):
                    with torch.npu.stream(self._kv_transfer_stream):
                        receive_completion = self.transport.copy_active_pages_from_staging(
                            self.layer_cache_bundles[layer_name],
                            active_pages,
                            self._kv_transfer_stream,
                        )
                    receive_completion.synchronize()
                return

            for peer_kvpp_rank, peer_global_rank in enumerate(self.kvpp_group.ranks):
                if peer_kvpp_rank == owner_kvpp_rank:
                    continue
                dist.recv(
                    token,
                    src=peer_global_rank,
                    group=self.kvpp_group.cpu_group,
                    tag=_KVPP_READY_TAG,
                )

            with torch.profiler.record_function(f"kvpp.transport_push.layer_{layer_index}"):
                with torch.npu.stream(self._kv_transfer_stream):
                    completion = self.transport.copy_active_pages_to_staging(
                        self.layer_cache_bundles[layer_name],
                        active_pages,
                        self._kv_transfer_stream,
                    )
                # Only the communication worker waits on the host. The compute
                # thread continues until this layer first writes/reads its
                # paged KV cache.
                completion.synchronize()

            for peer_kvpp_rank, peer_global_rank in enumerate(self.kvpp_group.ranks):
                if peer_kvpp_rank == owner_kvpp_rank:
                    continue
                dist.send(
                    token,
                    dst=peer_global_rank,
                    group=self.kvpp_group.cpu_group,
                    tag=_KVPP_DONE_TAG,
                )
