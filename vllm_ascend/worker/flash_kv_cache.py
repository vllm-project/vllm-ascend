# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import torch


def create_flash_kernel_block_view(raw, spec, kernel_block_size, layout, descriptor, layer_index):
    """Split attention pages inside each manager-owned region of the shared pool.

    BLHNC changes [manager, layer, page] to [manager, subpage, layer,
    kernel_page]. Mamba retains its original manager-page view: the scheduler
    assigns a manager block to only one KV group at a time. Dense replicated
    draft regions and LBHNC keep their per-layer starting offsets.
    """
    # This main-only API must not become an import dependency of non-Flash
    # runners on the supported release branch.
    from vllm.v1.kv_cache_interface import create_kv_cache_views

    if kernel_block_size <= 0 or spec.block_size % kernel_block_size:
        raise ValueError("Flash kernel block size must divide the manager block size.")
    ratio = spec.block_size // kernel_block_size
    if ratio <= 1:
        raise ValueError("Kernel-block view conversion requires a split manager page.")
    if raw.dtype != torch.int8 or raw.ndim != 2:
        raise ValueError("Flash kernel-block views require a two-dimensional byte cache.")
    if raw.stride(0) < raw.shape[1] or raw.stride(0) % ratio or raw.shape[1] % ratio:
        raise ValueError("Flash manager-page stride and payload must divide evenly into kernel pages.")

    offset = raw.storage_offset()
    if raw.stride(0) != raw.shape[1]:
        if layout.name == "BLHNC":
            layer_offset = descriptor.offset + layer_index * descriptor.layer_stride
            if (
                descriptor.block_stride != raw.stride(0)
                or descriptor.layer_stride not in (0, spec.page_size_bytes)
                or layer_offset < 0
                or layer_offset + raw.shape[1] > descriptor.block_stride
                or layer_offset % ratio
                or spec.page_size_bytes % ratio
            ):
                raise ValueError("Unsupported BLHNC manager-page geometry for Flash kernel splitting.")
            offset = raw.storage_offset() - layer_offset + layer_offset // ratio
        elif layout.name == "LBHNC":
            if descriptor.block_stride != raw.stride(0) or (
                len(descriptor.layers) > 1 and descriptor.layer_stride < raw.shape[0] * raw.stride(0)
            ):
                raise ValueError("Unsupported LBHNC manager-page geometry for Flash kernel splitting.")
        else:
            raise ValueError("Flash kernel splitting supports only BLHNC and LBHNC layouts.")

    # The descriptor now describes actual kernel pages; the upstream helper
    # must not subdivide the old manager geometry or replicate its padding.
    kernel_spec = replace(spec, block_size=kernel_block_size, page_size_padded=None)
    kernel_descriptor = replace(
        descriptor,
        layers=[descriptor.layers[layer_index]],
        offset=offset,
        layer_stride=raw.stride(0),
        block_stride=raw.stride(0) // ratio,
    )
    backing = raw.as_strided((raw.untyped_storage().nbytes(),), (1,), storage_offset=0)
    return create_kv_cache_views(backing, kernel_spec, raw.shape[0] * ratio, layout, kernel_descriptor)[0]
