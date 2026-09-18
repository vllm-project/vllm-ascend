# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import torch


def customize_flash_mla_c8_spec(spec, attention):
    """Keep the upstream manager contract, with mixed FP8/BF16 state bytes."""
    if attention.impl.dtype != torch.float8_e4m3fn:
        raise ValueError("A5 FlashMLA C8 requires FP8 E4M3 latent cache.")
    latent_dim, rope_dim = attention.kv_lora_rank, attention.qk_rope_head_dim
    if (latent_dim, rope_dim) != (512, 64):
        raise ValueError("A5 FlashMLA C8 requires 512 latent and 64 BF16 dimensions.")
    return replace(
        spec,
        dtype=torch.float8_e4m3fn,
        cache_dtype_str=None,
        state_content_bytes=latent_dim + rope_dim * torch.bfloat16.itemsize,
    )


def split_flash_mla_c8_cache(cache):
    """Split each kernel page into contiguous FP8 latent and BF16 KR planes."""
    latent_dim, rope_bytes = 512, 64 * torch.bfloat16.itemsize
    if cache.ndim != 3 or cache.dtype != torch.float8_e4m3fn or cache.shape[-1] != latent_dim + rope_bytes:
        raise ValueError("FlashMLA C8 expects a [pages, tokens, 640] FP8 byte view.")
    # The allocator's C=640 byte shape only describes page capacity. FlashMLA
    # requires each component's token rows to be dense: [128,512] FP8 first,
    # then [128,64] BF16. Only physical page starts have gaps between layers.
    pages, tokens, _ = cache.shape
    latent = cache.as_strided((pages, tokens, 1, latent_dim), (cache.stride(0), latent_dim, latent_dim, 1))
    rope_dim = rope_bytes // torch.bfloat16.itemsize
    rope = cache.view(torch.bfloat16).as_strided(
        (pages, tokens, 1, rope_dim),
        (cache.stride(0) // 2, rope_dim, rope_dim, 1),
        storage_offset=(cache.storage_offset() + tokens * latent_dim) // 2,
    )
    return latent, rope


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
