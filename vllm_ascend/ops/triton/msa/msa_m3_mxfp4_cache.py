# SPDX-License-Identifier: Apache-2.0
"""Insert packed MXFP4 index keys and E8M0 scales into managed paged caches."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _scatter_mxfp4_cache_kernel(
    packed,
    scales,
    slots,
    key_cache,
    scale_cache,
    PACKED_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    PACKED_WIDTH: tl.constexpr,
    SCALE_WIDTH: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slots + token)
    if slot >= 0:
        offsets = tl.arange(0, BLOCK_WIDTH)
        key = tl.load(packed + token * PACKED_STRIDE + offsets, offsets < PACKED_WIDTH, other=0)
        tl.store(key_cache + slot * PACKED_WIDTH + offsets, key, offsets < PACKED_WIDTH)
        scale = tl.load(scales + token * SCALE_STRIDE + offsets, offsets < SCALE_WIDTH, other=0)
        tl.store(scale_cache + slot * SCALE_WIDTH + offsets, scale, offsets < SCALE_WIDTH)


def scatter_mxfp4_cache(
    packed: torch.Tensor,
    scales: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_cache: torch.Tensor,
    scale_cache: torch.Tensor,
) -> None:
    """Update current token slots; graph-padding slots (-1) never write cache."""
    tokens = slot_mapping.numel()
    if tokens == 0:
        return
    packed = packed.view(torch.uint8).reshape(tokens, -1)
    scales = scales.view(torch.uint8).reshape(tokens, -1)
    if not (
        packed.is_contiguous() and scales.is_contiguous() and key_cache.is_contiguous() and scale_cache.is_contiguous()
    ):
        raise ValueError("MiniMax M3 MXFP4 cache insertion requires contiguous packed data and scales")
    key_bytes = key_cache.view(torch.uint8)
    scale_bytes = scale_cache.view(torch.uint8)
    if packed.shape[1] != key_bytes.shape[-1]:
        raise ValueError("Packed index-key width differs from the managed cache")
    if scales.shape[1] != key_bytes.shape[-1] // 16:
        raise ValueError("MXFP4 index keys require one E8M0 scale per 32 logical elements")
    _scatter_mxfp4_cache_kernel[(tokens,)](
        packed,
        scales,
        slot_mapping.reshape(-1),
        key_bytes,
        scale_bytes,
        PACKED_STRIDE=packed.stride(0),
        SCALE_STRIDE=scales.stride(0),
        PACKED_WIDTH=packed.shape[1],
        SCALE_WIDTH=scales.shape[1],
        BLOCK_WIDTH=triton.next_power_of_2(packed.shape[1]),
    )
