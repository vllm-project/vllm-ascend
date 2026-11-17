# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Pull P KV into HiSparse's Host Source and NPU Indexer groups."""

from typing import Any

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import (
    MooncakePullConnectorWorker,
    MooncakePullRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
    as_kv_cache_tensors,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_hisparse_te


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
    ):
        if remote_tp_size != self.tp_size or remote_pcp_size != 1 or remote_dcp_size != 1:
            raise ValueError("HiSparse Pull requires equal P/D TP and PCP=DCP=1")
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
        peer = remote_metadata.metadata_by_pcp_rank[remote_pcp_rank].metadata_by_tp_rank[remote_tp_rank]
        for (local_index, remote_index), entries in host_layers.items():
            targets = self.kv_caches_base_addr[local_index]
            sources = peer.kv_caches_base_addr[remote_index]
            if len(targets) != 1 or len(sources) not in (1, 2):
                raise ValueError("HiSparse Pull requires one Host Source and one or two P components")
            tokens = self.kv_cache_specs[spec_index].block_size
            component_bytes = remote_metadata.block_lens[remote_index]
            source_strides = remote_metadata.block_strides[remote_index]
            if len(sources) == 1 and len(remote_metadata.block_shapes[remote_index][0]) == 1:
                # The common worker publishes component-major MLA slots as one flat descriptor.
                hf = self.model_config.hf_text_config
                itemsize = self.kv_cache_specs[spec_index].dtype.itemsize
                latent_bytes = tokens * hf.kv_lora_rank * itemsize
                rope_bytes = tokens * hf.qk_rope_head_dim * itemsize
                if latent_bytes + rope_bytes > component_bytes[0]:
                    raise ValueError("P MLA slot is smaller than its NoPE/RoPE payload")
                sources = [sources[0], sources[0] + latent_bytes]
                component_bytes = [latent_bytes, rope_bytes]
                source_strides = [source_strides[0], source_strides[0]]
            page_bytes = self.block_lens[local_index][0]
            if sum(component_bytes) != page_bytes or any(size % tokens for size in component_bytes):
                raise ValueError("HiSparse Pull MLA component sizes differ from Host Source")
            row_bytes = [size // tokens for size in component_bytes]
            for _, local_blocks, remote_blocks in entries:
                for local_block, remote_block in zip(local_blocks, remote_blocks, strict=True):
                    target = targets[0] + local_block * self.block_strides[local_index][0]
                    for token in range(tokens):
                        offset = 0
                        for component, size in enumerate(row_bytes):
                            # READ takes local destinations first, then remote P sources.
                            src_list.append(target + token * sum(row_bytes) + offset)
                            dst_list.append(
                                sources[component] + remote_block * source_strides[component] + token * size
                            )
                            length_list.append(size)
                            offset += size


class _HiSparsePDWorker(MooncakePullConnectorWorker):
    receiving_thread_cls = _HiSparsePullThread

    def _get_transfer_engine(self, device_name: str | None):
        extra = self.kv_transfer_config.kv_connector_extra_config or {}
        if extra.get("memfabric_transfer_protocol", "device_urma") != "device_urma":
            raise ValueError("HiSparse Mooncake PD supports only device_urma; SDMA is disabled")
        return global_hisparse_te.get_transfer_engine(self.side_channel_host, device_name)

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
        global_hisparse_te.register_buffer(list(regions), list(regions.values()))

    def _is_transfer_group(self, group_index: int) -> bool:
        return self.kv_cache_config.kv_cache_groups[group_index].enable_kv_transfer

    def _logical_num_blocks(self, group_index: int) -> int:
        group = self.kv_cache_config.kv_cache_groups[group_index]
        if group.host_resident:
            assert self.kv_cache_config.hisparse_host_num_blocks is not None
            return self.kv_cache_config.hisparse_host_num_blocks
        return self.num_blocks

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]) -> None:
        if self.kv_transfer_config.is_kv_consumer:
            context = self.vllm_config.compilation_config.static_forward_context
            caches: dict[str, Any] = dict(kv_caches)
            source_names = [
                name
                for group in self.kv_cache_config.kv_cache_groups
                if group.host_resident and group.enable_kv_transfer
                for name in group.layer_names
            ]
            source_parts = [cache for name in source_names for cache in as_kv_cache_tensors(kv_caches[name])]
            if not source_parts or any(cache.device.type != "cpu" for cache in source_parts):
                raise ValueError("HiSparse PD Source must consist of Host tensors")
            self._host_pool_regions = {
                cache.untyped_storage().data_ptr(): cache.untyped_storage().nbytes() for cache in source_parts
            }
            for group in self.kv_cache_config.kv_cache_groups:
                if group.role != "hisparse_indexer":
                    continue
                for name in group.layer_names:
                    stride = getattr(context[name], "hisparse_block_stride", 1)
                    if stride > 1:
                        caches[name] = tuple(cache[::stride] for cache in as_kv_cache_tensors(kv_caches[name]))
            kv_caches = caches
        super().register_kv_caches(kv_caches)


def validate_hisparse_pd_config(transfer, kv_cache_config: KVCacheConfig) -> None:
    """Validate the ordinary P -> HiSparse D layout before starting either role."""
    if transfer.is_kv_consumer:
        if kv_cache_config.hisparse_host_num_blocks is None:
            raise ValueError("Mooncake memfabric consumer requires HiSparse Host Source")
        transferable = [group for group in kv_cache_config.kv_cache_groups if group.enable_kv_transfer]
        if len(transferable) != 2 or not transferable[0].host_resident or transferable[1].role != "hisparse_indexer":
            raise ValueError("Mooncake memfabric requires Host Source and NPU Indexer transfer groups")
    elif kv_cache_config.hisparse_host_num_blocks is not None:
        raise ValueError("Mooncake memfabric producer must use ordinary Prefill KV")
