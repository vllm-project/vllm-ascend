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
    packed_stride: tl.constexpr,
    scale_stride: tl.constexpr,
    packed_width: tl.constexpr,
    scale_width: tl.constexpr,
    block_width: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slots + token)
    if slot >= 0:
        offsets = tl.arange(0, block_width)
        key = tl.load(packed + token * packed_stride + offsets, offsets < packed_width, other=0)
        tl.store(key_cache + slot * packed_width + offsets, key, offsets < packed_width)
        scale = tl.load(scales + token * scale_stride + offsets, offsets < scale_width, other=0)
        tl.store(scale_cache + slot * scale_width + offsets, scale, offsets < scale_width)


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
    key_bytes = key_cache.view(torch.uint8)
    scale_bytes = scale_cache.view(torch.uint8)
    if not (
        packed.is_contiguous() and scales.is_contiguous() and key_bytes.is_contiguous() and scale_bytes.is_contiguous()
    ):
        raise ValueError("MiniMax M3 MXFP4 cache insertion requires contiguous packed data and scales")
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
        packed.stride(0),
        scales.stride(0),
        packed.shape[1],
        scales.shape[1],
        triton.next_power_of_2(packed.shape[1]),
    )
