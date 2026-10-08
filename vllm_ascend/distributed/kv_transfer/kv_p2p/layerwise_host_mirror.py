# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable P-side Mooncake source for reused DeepSeek-V4 KV buffers.

The D-side hybrid connector pulls a complete request after prefill. Its source
addresses therefore cannot point at P's layerwise-reused NPU slots. This
mirror keeps the original V4 tuple layout in pinned host memory while the P
runner uses a much smaller NPU backing.
"""

from dataclasses import dataclass
from typing import Any

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    get_layerwise_kv_cache_specs,
    get_layerwise_physical_layer_index,
)
from vllm_ascend.utils import get_kv_cache_tensor_layers

_HOST_REGISTRATION_ALIGNMENT = 64 * 1024


@dataclass(frozen=True)
class _Component:
    name: str
    group: int
    page_size: int
    host_offset: int
    npu_pages: torch.Tensor


class DeepSeekV4LayerwiseHostMirror:
    def __init__(
        self,
        config: KVCacheConfig,
        kv_caches: dict[str, Any],
        base_layers: int,
        previous_slot_layer: dict[int, int],
    ):
        self.num_blocks = config.num_blocks
        self.backing_size = max(tensor.size for tensor in config.kv_cache_tensors)
        # HIXL registers page-locked host memory. Keep its registered extent
        # page-aligned while descriptor offsets retain their original values.
        alignment = _HOST_REGISTRATION_ALIGNMENT
        registered_size = (self.backing_size + alignment - 1) // alignment * alignment
        self.backing = torch.empty(registered_size, dtype=torch.uint8, device="cpu", pin_memory=True)
        self.copy_stream = torch.npu.Stream()
        self._copy_done: dict[int, torch.npu.Event] = {}
        self.previous_slot_layer = previous_slot_layer
        self.host_views: dict[str, torch.Tensor] = {}
        self.components: dict[int, list[_Component]] = {}

        layer_specs = get_layerwise_kv_cache_specs(config)
        layer_groups = {
            name: group_idx for group_idx, group in enumerate(config.kv_cache_groups) for name in group.layer_names
        }
        for descriptor in config.kv_cache_tensors:
            for index, name in enumerate(get_kv_cache_tensor_layers(descriptor)):
                spec = layer_specs[name]
                offset = descriptor.offset + index * descriptor.layer_stride
                page_bytes = self.num_blocks * spec.page_size_bytes
                if (
                    descriptor.block_stride != spec.page_size_bytes
                    or offset < 0
                    or offset + page_bytes > self.backing_size
                ):
                    raise ValueError(f"Invalid DeepSeek-V4 host mirror descriptor for {name}.")
                if name in self.host_views:
                    raise ValueError(f"Duplicate DeepSeek-V4 host mirror descriptor for {name}.")
                # A one-byte view is sufficient for the hybrid connector's
                # shared-page reconstruction, which only reads data_ptr().
                self.host_views[name] = self.backing.narrow(0, offset, 1)
                tensors = kv_caches[name]
                if not isinstance(tensors, (tuple, list)):
                    tensors = (tensors,)
                page_base = min(tensor.data_ptr() for tensor in tensors)
                source = tensors[0]
                storage = source.untyped_storage()
                source_offset = page_base - storage.data_ptr()
                if source_offset < 0 or source_offset + page_bytes > storage.nbytes():
                    raise ValueError(f"DeepSeek-V4 NPU page for {name} exceeds its backing storage.")
                raw = torch.empty(0, dtype=torch.uint8, device=source.device).set_(
                    storage, source_offset, (page_bytes,), (1,)
                )
                layer = get_layerwise_physical_layer_index(name, base_layers)
                self.components.setdefault(layer, []).append(
                    _Component(name, layer_groups[name], spec.page_size_bytes, offset, raw.view(self.num_blocks, -1))
                )
        if set(self.host_views) != set(kv_caches):
            raise ValueError("DeepSeek-V4 host mirror must cover every runtime KV cache component.")

    def save_layer(self, layer_name: str, requests: list[Any], base_layers: int) -> None:
        layer = get_layerwise_physical_layer_index(layer_name, base_layers)
        components = self.components.get(layer)
        if not components or not requests:
            return
        self.wait_for_layer_reuse(layer)
        write_done = torch.npu.Event()
        write_done.record(torch.npu.current_stream())
        copy_done = torch.npu.Event()
        with torch.npu.stream(self.copy_stream):
            self.copy_stream.wait_event(write_done)
            for component in components:
                block_ids: set[int] = set()
                for request in requests:
                    groups = request.block_ids_by_group
                    if component.group < len(groups):
                        block_ids.update(groups[component.group])
                host_pages = self.backing.narrow(0, component.host_offset, self.num_blocks * component.page_size).view(
                    self.num_blocks, component.page_size
                )
                for block_id in sorted(block_ids):
                    if block_id < 0 or block_id >= self.num_blocks:
                        raise ValueError(f"DeepSeek-V4 host mirror block ID {block_id} is out of range.")
                    host_pages[block_id].copy_(component.npu_pages[block_id], non_blocking=True)
            copy_done.record(self.copy_stream)
        self._copy_done[layer] = copy_done

    def wait_for_layer_reuse(self, layer: int) -> None:
        for prior in (layer, self.previous_slot_layer.get(layer)):
            if prior is not None and (event := self._copy_done.pop(prior, None)) is not None:
                event.synchronize()

    def wait_for_save(self) -> None:
        for layer in list(self._copy_done):
            self.wait_for_layer_reuse(layer)
