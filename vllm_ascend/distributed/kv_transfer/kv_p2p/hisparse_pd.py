# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Pull P KV into HiSparse's Host Source and NPU Indexer groups."""

import ctypes
import threading
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakePullConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeTransferMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import (
    MooncakePullRecvingThread,
    _MemFabricPullWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
    as_kv_cache_tensors,
)
from vllm_ascend.utils import get_kv_cache_tensor_layers


class _HiSparsePullThread(MooncakePullRecvingThread):
    def _append_spec_transfer_addresses(
        self,
        spec_index,
        remote_pcp_rank,
        remote_tp_rank,
        remote_pcp_size,
        remote_tp_size,
        remote_dcp_size,
        transfer_entries_by_layer,
        remote_metadata,
        src_list,
        dst_list,
        length_list,
    ) -> Callable[[], None] | None:
        if remote_tp_size != self.tp_size or self.pcp_size != 1 or remote_dcp_size != 1:
            raise ValueError("HiSparse Pull requires equal P/D TP, D PCP=1, and P DCP=1")
        host_layers = {
            pair: entries
            for pair, entries in transfer_entries_by_layer.items()
            if self.kv_cache_config.kv_cache_groups[self.group_indices[pair[0]]].host_resident
        }
        ordinary = {pair: entries for pair, entries in transfer_entries_by_layer.items() if pair not in host_layers}
        if ordinary:
            super()._append_spec_transfer_addresses(
                spec_index,
                remote_pcp_rank,
                remote_tp_rank,
                remote_pcp_size,
                remote_tp_size,
                remote_dcp_size,
                ordinary,
                remote_metadata,
                src_list,
                dst_list,
                length_list,
            )
        if not host_layers:
            return None

        pages_to_reorder: dict[int, tuple[int, int, int]] = {}
        spec = self.kv_cache_specs[spec_index]
        peer = remote_metadata.metadata_by_pcp_rank[remote_pcp_rank].metadata_by_tp_rank[remote_tp_rank]
        for (local_index, remote_index), entries in host_layers.items():
            self._append_host_layer_reads(
                spec,
                local_index,
                remote_index,
                peer.kv_caches_base_addr[remote_index],
                remote_metadata,
                entries,
                local_dsts=src_list,
                remote_srcs=dst_list,
                lengths=length_list,
                pages_to_reorder=pages_to_reorder,
            )
        return partial(self._reorder_host_pages, pages_to_reorder) if pages_to_reorder else None

    def _append_host_layer_reads(
        self,
        spec,
        local_index,
        remote_index,
        remote_bases,
        remote_metadata,
        entries,
        *,
        local_dsts,
        remote_srcs,
        lengths,
        pages_to_reorder,
    ):
        row_bytes, components = self._host_pull_components(
            spec, local_index, remote_index, remote_bases, remote_metadata
        )
        local_base = self.kv_caches_base_addr[local_index][0]
        local_stride = self.block_strides[local_index][0]
        for _, local_blocks, remote_blocks in entries:
            for local_block, remote_block in zip(local_blocks, remote_blocks, strict=True):
                local_page = local_base + local_block * local_stride
                if self._append_host_page_reads(
                    local_page, remote_block, components, spec.block_size, row_bytes, local_dsts, remote_srcs, lengths
                ):
                    pages_to_reorder[local_page] = (spec.block_size, row_bytes, components[0][2])

    @staticmethod
    def _append_host_page_reads(
        local_page, remote_block, components, tokens_per_page, row_bytes, local_dsts, remote_srcs, lengths
    ):
        remote_pages = [
            (base + remote_block * stride, token_bytes, row_offset)
            for base, stride, token_bytes, row_offset in components
        ]
        # MemFabric READ accepts local destinations before remote sources.
        if len(remote_pages) == 1:
            remote_page, _, _ = remote_pages[0]
            local_dsts.append(local_page)
            remote_srcs.append(remote_page)
            lengths.append(tokens_per_page * row_bytes)
            return
        offset = 0
        for remote_page, token_bytes, _ in remote_pages:
            component_bytes = tokens_per_page * token_bytes
            local_dsts.append(local_page + offset)
            remote_srcs.append(remote_page)
            lengths.append(component_bytes)
            offset += component_bytes
        return True

    @staticmethod
    def _reorder_host_pages(pages: dict[int, tuple[int, int, int]]) -> None:
        scratch = np.empty(max(tokens * row_bytes for tokens, row_bytes, _ in pages.values()), dtype=np.uint8)
        for local_page, (tokens, row_bytes, first_component_bytes) in pages.items():
            page_bytes = tokens * row_bytes
            page = np.frombuffer((ctypes.c_uint8 * page_bytes).from_address(local_page), dtype=np.uint8)
            source = scratch[:page_bytes]
            source[:] = page
            rows = page.reshape(tokens, row_bytes)
            rows[:, :first_component_bytes] = source[: tokens * first_component_bytes].reshape(
                tokens, first_component_bytes
            )
            rows[:, first_component_bytes:] = source[tokens * first_component_bytes :].reshape(
                tokens, row_bytes - first_component_bytes
            )

    def _host_pull_components(self, spec, local_index, remote_index, bases, remote_metadata):
        """Describe P's MLA components and their offsets within a D Host row."""
        if len(self.kv_caches_base_addr[local_index]) != 1 or len(bases) not in (1, 2):
            raise ValueError("HiSparse Pull requires one Host Source and one or two P components")

        tokens = spec.block_size
        sizes = remote_metadata.block_lens[remote_index]
        strides = remote_metadata.block_strides[remote_index]
        if len(bases) == 1 and len(remote_metadata.block_shapes[remote_index][0]) == 1:
            # Component-major MLA is advertised as one flat slot.
            hf = self.model_config.hf_text_config
            latent_bytes = tokens * hf.kv_lora_rank * spec.dtype.itemsize
            rope_bytes = tokens * hf.qk_rope_head_dim * spec.dtype.itemsize
            if latent_bytes + rope_bytes > sizes[0]:
                raise ValueError("P MLA slot is smaller than its NoPE/RoPE payload")
            bases = [bases[0], bases[0] + latent_bytes]
            sizes = [latent_bytes, rope_bytes]
            strides = [strides[0], strides[0]]

        page_bytes = self.block_lens[local_index][0]
        if sum(sizes) != page_bytes or any(size % tokens for size in sizes):
            raise ValueError("HiSparse Pull MLA component sizes differ from Host Source")

        row_offset = 0
        components = []
        for base, stride, size in zip(bases, strides, sizes, strict=True):
            token_bytes = size // tokens
            components.append((base, stride, token_bytes, row_offset))
            row_offset += token_bytes
        return row_offset, components


class _HiSparsePDWorker(_MemFabricPullWorker):
    receiving_thread_cls = _HiSparsePullThread

    def _register_transfer_buffers(self, ptrs: list[int], lengths: list[int]) -> None:
        # URMA is communication-based: both P sources and D Host/Indexer destinations need MRs.
        regions: dict[int, int] = {}
        for ptr, length in zip(ptrs, lengths, strict=True):
            for base, size in getattr(self, "_host_pool_regions", {}).items():
                if base <= ptr < base + size:
                    if ptr + length > base + size:
                        raise ValueError("HiSparse transfer region exceeds its Host pool")
                    # Register the owner once instead of overlapping per-layer Source views.
                    ptr, length = base, size
                    break
            regions[ptr] = max(regions.get(ptr, 0), length)
        super()._register_transfer_buffers(list(regions), list(regions.values()))

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]) -> None:
        config = self.kv_cache_config
        transfer_group_ids = set(config.transfer_group_ids)
        if config.hisparse_shared_host_pool and self.tp_rank != 0:
            # TP rank 0 imports the shared Host allocation for all local ranks.
            transfer_group_ids.difference_update(config.host_group_ids)
        excluded_layers = {
            name
            for group_id, group in enumerate(config.kv_cache_groups)
            if group_id not in transfer_group_ids
            for name in group.layer_names
        }
        caches = {name: cache for name, cache in kv_caches.items() if name not in excluded_layers}
        if self.kv_transfer_config.is_kv_consumer:
            self._prepare_destination_caches(caches, transfer_group_ids)

        # Keep group IDs and allocation geometry intact while narrowing the
        # layer list seen by the ordinary Mooncake registration flow.
        tensors = []
        for tensor in config.kv_cache_tensors:
            layers = [name for name in get_kv_cache_tensor_layers(tensor) if name not in excluded_layers]
            if not layers:
                continue
            logical_blocks = config.num_blocks_of(tensor)
            for name in layers:
                for cache in as_kv_cache_tensors(caches.get(name)):
                    if cache.shape[0] % logical_blocks:
                        raise ValueError(
                            f"KV cache layer {name!r} has {cache.shape[0]} physical blocks "
                            f"for {logical_blocks} logical blocks."
                        )
            tensors.append(replace(tensor, layers=layers))

        self.kv_cache_config = replace(config, kv_cache_tensors=tensors)
        try:
            super().register_kv_caches(caches)
        finally:
            self.kv_cache_config = config

    def _prepare_destination_caches(
        self, caches: dict[str, torch.Tensor | list[torch.Tensor]], transfer_group_ids: set[int]
    ) -> None:
        config = self.kv_cache_config
        context = self.vllm_config.compilation_config.static_forward_context
        host_group_ids = transfer_group_ids.intersection(config.host_group_ids)
        source_parts = [
            cache
            for group_id in host_group_ids
            for name in config.kv_cache_groups[group_id].layer_names
            for cache in as_kv_cache_tensors(caches[name])
        ]
        if host_group_ids and (not source_parts or any(cache.device.type != "cpu" for cache in source_parts)):
            raise ValueError("HiSparse PD Source must consist of Host tensors")
        self._host_pool_regions = {
            cache.untyped_storage().data_ptr(): cache.untyped_storage().nbytes() for cache in source_parts
        }
        for group_id in transfer_group_ids:
            group = config.kv_cache_groups[group_id]
            if group.role == "hisparse_indexer":
                for name in group.layer_names:
                    stride = getattr(context[name], "hisparse_block_stride", 1)
                    if stride > 1:
                        caches[name] = [cache[::stride] for cache in as_kv_cache_tensors(caches[name])]

    def _create_recving_thread(self, ready_event: threading.Event) -> MooncakePullRecvingThread:
        metadata = self.xfer_handshake_metadata
        assert isinstance(metadata, MooncakeTransferMetadata)
        config = self.kv_cache_config
        host_groups = set(config.host_group_ids)
        host_blocks = config.hisparse_host_num_blocks
        assert host_blocks is not None
        scales = list(metadata.block_size_scales)
        for index, (group_id, name) in enumerate(zip(metadata.group_indices, metadata.layer_names, strict=True)):
            if group_id in host_groups:
                parts = as_kv_cache_tensors(self.kv_caches[name])
                scales[index] = [cache.shape[0] // host_blocks for cache in parts[: len(scales[index])]]
        self.xfer_handshake_metadata = replace(metadata, block_size_scales=scales)
        return super()._create_recving_thread(ready_event)


def validate_hisparse_pd_config(transfer, kv_cache_config: KVCacheConfig) -> None:
    """Validate the ordinary P -> HiSparse D layout before creating the Worker."""
    if transfer.is_kv_consumer:
        if kv_cache_config.hisparse_host_num_blocks is None:
            raise ValueError("Mooncake memfabric consumer requires HiSparse Host Source")
        transferable = [group for group in kv_cache_config.kv_cache_groups if group.enable_kv_transfer]
        if len(transferable) != 2 or not transferable[0].host_resident or transferable[1].role != "hisparse_indexer":
            raise ValueError("Mooncake memfabric requires Host Source and NPU Indexer transfer groups")
    elif kv_cache_config.hisparse_host_num_blocks is not None:
        raise ValueError("Mooncake memfabric producer must use ordinary Prefill KV")


class _HiSparseMooncakePullConnector(MooncakePullConnector):
    """Use the HiSparse Worker for MemFabric Pull."""

    def _create_worker(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> _HiSparsePDWorker:
        transfer = vllm_config.kv_transfer_config
        assert transfer is not None
        validate_hisparse_pd_config(transfer, kv_cache_config)
        return _HiSparsePDWorker(vllm_config, str(self.engine_id), kv_cache_config)
