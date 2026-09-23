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


def view_flash_mla_cache(raw, spec, kernel_block_size):
    """View an existing per-layer byte allocation as packed FlashMLA pages."""
    if raw.dtype != torch.int8 or raw.ndim != 1 or not raw.is_contiguous():
        raise ValueError("FlashMLA requires the allocator's contiguous per-layer byte buffer.")
    if kernel_block_size <= 0 or spec.block_size % kernel_block_size:
        raise ValueError("FlashMLA kernel block size must divide the manager block size.")
    if raw.numel() % spec.page_size_bytes:
        raise ValueError("FlashMLA allocation must contain whole manager pages.")
    ratio = spec.block_size // kernel_block_size
    if spec.page_size_bytes % ratio:
        raise ValueError("FlashMLA manager page must divide into equal kernel pages.")
    width = 640 if spec.dtype == torch.float8_e4m3fn else 576
    if spec.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("FlashMLA cache must use BF16 or FP8 E4M3.")
    stride_bytes = spec.page_size_bytes // ratio
    if stride_bytes % spec.dtype.itemsize or stride_bytes < kernel_block_size * width * spec.dtype.itemsize:
        raise ValueError("FlashMLA kernel page is smaller than its cache payload.")
    pages = raw.numel() // spec.page_size_bytes * ratio
    cache = raw.view(spec.dtype).as_strided(
        (pages, kernel_block_size, width),
        (stride_bytes // spec.dtype.itemsize, width, 1),
    )
    return split_flash_mla_c8_cache(cache) if spec.dtype == torch.float8_e4m3fn else cache
