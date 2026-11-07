# SPDX-License-Identifier: Apache-2.0
"""Ascend tensor-view bindings for native Store spec/layout families."""

from dataclasses import replace

import torch
from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import ChunkedTokenDatabase
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.layout import (
    AttentionStoreLayout,
    MambaStoreLayout,
    RankLocalStoreLayout,
)
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import derive_mamba_conv_split
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.kv_cache_interface import MambaAttentionBackendEnum, MLAAttentionSpec, group_kernel_blocks

from vllm_ascend.distributed.kv_transfer.utils.utils import collect_storage_merged_register_regions


def _packed_state_view(tensors, num_blocks):
    """Recover the native byte-page view without copying typed shared storage."""
    first = tensors[0]
    storage = first.untyped_storage()
    page_stride = first.stride(0) * first.element_size()
    offset = first.storage_offset() * first.element_size()
    content = 0
    for tensor in tensors:
        if (
            tensor.untyped_storage().data_ptr() != storage.data_ptr()
            or tensor.data_ptr() != first.data_ptr() + content
            or tensor.stride(0) * tensor.element_size() != page_stride
            or not tensor[0].is_contiguous()
        ):
            return None
        content += tensor[0].numel() * tensor.element_size()
    if content > page_stride or offset + (num_blocks - 1) * page_stride + content > storage.nbytes():
        raise ValueError("Native state byte pages exceed their backing allocation")
    return torch.empty(0, dtype=torch.uint8, device=first.device).set_(
        storage, offset, (num_blocks, 1, 1, content), (page_stride, content, content, 1)
    )


def register_native_cache_views(worker, kv_caches):
    """Bind actual views; native code owns groups, keys, shards and transfers.

    Packed attention/state tensors use native registration directly. Separate
    K/V planes and typed GDN states need views over their existing allocations.
    Full native states, including speculative records, are transferred. No
    model-name dispatch, payload packing or speculative scratch clearing.
    """
    config = worker._kv_cache_config
    context = worker._ascend_vllm_config.compilation_config.static_forward_context
    databases = []
    device_tensors: list[torch.Tensor] = []
    host_tensors: list[torch.Tensor] = []
    for group, db in zip(worker._kv_cache_groups, worker.token_dbs, strict=True):
        old = db.store_layout
        num_blocks = config.hisparse_host_num_blocks if group.host_resident else config.num_blocks
        metadata = old.metadata
        registered = []
        if isinstance(old, RankLocalStoreLayout):
            views = []
            for name in group.layer_names:
                value = kv_caches[name]
                tensors = [value] if isinstance(value, torch.Tensor) else value
                for tensor in tensors:
                    if tensor.numel():
                        views.append(group_kernel_blocks(tensor, num_blocks))
                        registered.append(tensor)
            layout = RankLocalStoreLayout(metadata, old.block_size, old.hash_block_size)
            layout.register_kv_caches(views, num_blocks)
        elif isinstance(old, MambaStoreLayout):
            local_tp = old.store_tp_size // old.shards_per_rank
            rank = old.store_shard_ids[0] // old.shards_per_rank
            states, packed = [], []
            for name, spec in zip(group.layer_names, old.layer_specs, strict=True):
                value = kv_caches[name]
                if isinstance(value, torch.Tensor) and value.dtype in (torch.uint8, torch.int8):
                    packed.append(value.view(num_blocks, 1, 1, -1))
                    registered.append(value)
                    continue
                tensors = context[name].kv_cache if isinstance(value, torch.Tensor) else value
                byte_view = _packed_state_view(tensors, num_blocks)
                if byte_view is not None:
                    packed.append(byte_view)
                    registered.extend(tensors)
                    continue
                if spec.mamba_type not in (MambaAttentionBackendEnum.MAMBA1, MambaAttentionBackendEnum.GDN_ATTN):
                    raise NotImplementedError("This native state family requires its packed byte view")
                if len(tensors) != len(spec.shapes):
                    raise ValueError("State views differ from the registered native spec")
                conv = derive_mamba_conv_split(spec, local_tp, require_dim_first=False)
                states.append((tensors[0], tuple(conv.local_proj_dims), 0 if is_conv_state_dim_first() else 1))
                states.extend((tensor, (tensor.shape[1],), 0) for tensor in tensors[1:])
                registered.extend(tensors)
            if packed and states:
                raise NotImplementedError("One native state group must use consistent packed or typed views")
            layout = MambaStoreLayout(
                metadata, old.block_size, old.hash_block_size, local_tp, old.store_tp_size, rank, old.layer_specs
            )
            if packed:
                layout.register_kv_caches(packed, num_blocks)
            else:
                layout.register_state_views(states, num_blocks)
        elif isinstance(old, AttentionStoreLayout):
            local_tp = old.store_tp_size // old.shards_per_rank
            rank = old.store_shard_ids[0] // old.shards_per_rank
            specs, views = [], []
            planar = False
            for name, spec in zip(group.layer_names, old.layer_specs, strict=True):
                value = kv_caches[name]
                if not isinstance(value, torch.Tensor):
                    value = tuple(tensor for tensor in value if tensor.numel())
                if isinstance(value, torch.Tensor):
                    views.append(group_kernel_blocks(value, num_blocks))
                    specs.append(spec)
                    registered.append(value)
                    continue
                if isinstance(spec, MLAAttentionSpec):
                    if getattr(spec, "scale_dim", 0):
                        raise NotImplementedError("Separate indexer scales need their own native view contract")
                    content_bytes = sum(plane.shape[-1] * plane.element_size() for plane in value)
                    if content_bytes != spec.state_content_size_bytes or any(
                        plane.dtype != spec.dtype for plane in value
                    ):
                        raise ValueError("MLA component views do not match their native content spec")
                    planar |= len(value) > 1
                    for plane in value:
                        registered.append(plane)
                        if plane.ndim == 3:
                            plane = plane.unsqueeze(1)
                        elif plane.ndim == 4 and plane.shape[1] != 1:
                            plane = plane.transpose(1, 2)
                        views.append(group_kernel_blocks(plane, num_blocks))
                        specs.append(replace(spec, state_content_bytes=plane.shape[-1] * plane.element_size()))
                    continue
                if len(value) != 2 or spec.dtype not in (torch.float16, torch.bfloat16):
                    raise NotImplementedError("This attention layout needs a packed view or FP16/BF16 K/V planes")
                planar = True
                for plane, head_size in zip(value, (spec.head_size, spec.head_size_v), strict=True):
                    if plane.dtype != spec.dtype or plane.shape[-1] != head_size:
                        raise ValueError("K/V plane does not match its native attention spec")
                    registered.append(plane)
                    if envs.VLLM_KV_CACHE_LAYOUT not in ("LBHNC", "HND"):
                        plane = plane.transpose(1, 2)
                    views.append(group_kernel_blocks(plane, num_blocks))
                    specs.append(replace(spec, state_content_bytes=head_size * plane.element_size()))
            if planar:
                metadata = replace(metadata, store_namespace=metadata.store_namespace + "@ascend_kv_planes:v1")
            layout = type(old)(
                metadata,
                old.block_size,
                old.hash_block_size,
                local_tp,
                old.store_tp_size,
                rank,
                specs,
                store_chunk_size=old.store_chunk_size,
            )
            layout.register_kv_caches(views, num_blocks)
        else:
            raise NotImplementedError("Unsupported native Store cache layout")
        (host_tensors if group.host_resident else device_tensors).extend(registered)
        databases.append(
            ChunkedTokenDatabase(
                metadata,
                layout.block_size if isinstance(layout, RankLocalStoreLayout) else layout.store_chunk_size,
                hash_block_size=worker.hash_block_size,
                store_layout=layout,
            )
        )
    for tensors, device_resident in ((device_tensors, True), (host_tensors, False)):
        # The runner aligns actual NPU views, not necessarily storage origins.
        # Reuse Ascend's existing span collector and bounded hybrid alignment.
        alignment = 2 * 1024 * 1024 if tensors and tensors[0].device.type == "npu" else None
        regions = collect_storage_merged_register_regions({"caches": tensors}, alignment=alignment)
        for pointer, length in zip(regions.ptrs, regions.lengths, strict=True):
            result = worker.store.register_buffer(pointer, length)
            if result != 0:
                raise RuntimeError(f"Mooncake register_buffer failed: {result}")
            if device_resident:
                worker._device_regions[pointer] = length
    worker.num_blocks = config.num_blocks
    worker.token_dbs = databases
