# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Write packed K scales without copying back unrelated cache pages."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_elements"])
def _scatter_k_scale_kernel(
    source,
    cache,
    block_ids,
    segment_ids,
    fragment_ids,
    num_elements,
    num_heads: tl.constexpr,
    num_groups: tl.constexpr,
    block_stride: tl.constexpr,
    head_stride: tl.constexpr,
    segment_stride: tl.constexpr,
    group_stride: tl.constexpr,
    fragment_stride: tl.constexpr,
    pair_stride: tl.constexpr,
    block_index_stride: tl.constexpr,
    segment_index_stride: tl.constexpr,
    fragment_index_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = index < num_elements
    token = index // (num_heads * num_groups * 2)
    head = index // (num_groups * 2) % num_heads
    group = index // 2 % num_groups
    pair = index % 2
    block = tl.load(block_ids + token * block_index_stride, mask=valid, other=0).to(tl.int64)
    segment = tl.load(segment_ids + token * segment_index_stride, mask=valid, other=0).to(tl.int64)
    fragment = tl.load(fragment_ids + token * fragment_index_stride, mask=valid, other=0).to(tl.int64)
    offset = (
        block * block_stride
        + head.to(tl.int64) * head_stride
        + segment * segment_stride
        + group.to(tl.int64) * group_stride
        + fragment * fragment_stride
        + pair.to(tl.int64) * pair_stride
    )
    value = tl.load(source + index, mask=valid, other=0)
    tl.store(cache + offset, value, mask=valid)


def scatter_k_scales(
    key_scale: torch.Tensor,
    cache: torch.Tensor,
    slot_index: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Update only the addressed token/head/group bytes of a PA_NZ view."""
    block_ids, segment_ids, fragment_ids = slot_index
    if block_ids.numel() == 0:
        return
    num_heads, num_groups = cache.shape[1], cache.shape[3]
    source = key_scale.view(torch.uint8).contiguous()
    num_elements = source.numel()
    expected_elements = block_ids.numel() * num_heads * num_groups * 2
    if num_elements != expected_elements:
        raise ValueError(f"K-scale source contains {num_elements} elements; expected {expected_elements}.")
    block_size = 256
    _scatter_k_scale_kernel[(triton.cdiv(num_elements, block_size),)](
        source,
        cache,
        block_ids,
        segment_ids,
        fragment_ids,
        num_elements,
        num_heads,
        num_groups,
        *cache.stride(),
        block_ids.stride(0),
        segment_ids.stride(0),
        fragment_ids.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
