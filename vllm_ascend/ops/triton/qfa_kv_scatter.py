# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Scatter int8_qfa K/V payloads and K descale into the paged packet views.

The kernels write only the addressed bytes, consuming the packet views'
strides explicitly (the mxfp_k_scale pattern). Design history and the
rationale for the program mapping live in docs/qfa_kv_scatter.md.
"""

import torch
from vllm.triton_utils import tl, triton


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@triton.jit(do_not_specialize=["num_tokens"])
def _scatter_kv_payload_kernel(
    source,
    cache,
    block_ids,
    pos_ids,
    num_tokens,
    num_heads: tl.constexpr,
    num_frags: tl.constexpr,
    block_stride: tl.constexpr,
    head_stride: tl.constexpr,
    frag_stride: tl.constexpr,
    pos_stride: tl.constexpr,
    block_index_stride: tl.constexpr,
    pos_index_stride: tl.constexpr,
):
    # One program moves one (token, head) payload: num_frags lanes of 32
    # bytes. Source bytes are contiguous; the destination breaks into
    # num_frags fragments of 32 bytes at frag_stride apart.
    pid = tl.program_id(0)
    token = pid // num_heads
    head = pid % num_heads
    frag_offs = tl.arange(0, num_frags)
    lane_offs = tl.arange(0, 32)
    block = tl.load(block_ids + token * block_index_stride).to(tl.int64)
    pos = tl.load(pos_ids + token * pos_index_stride).to(tl.int64)
    src_offs = (
        (token * num_heads + head) * (num_frags * 32)
        + frag_offs[:, None] * 32
        + lane_offs[None, :]
    )
    value = tl.load(source + src_offs)
    dst_offs = (
        block * block_stride
        + head.to(tl.int64) * head_stride
        + pos * pos_stride
        + frag_offs[:, None].to(tl.int64) * frag_stride
        + lane_offs[None, :]
    )
    tl.store(cache + dst_offs, value)


@triton.jit(do_not_specialize=["num_tokens"])
def _scatter_k_descale_kernel(
    source,
    cache,
    block_ids,
    pos_ids,
    num_tokens,
    num_heads: tl.constexpr,
    block_stride: tl.constexpr,
    head_stride: tl.constexpr,
    pos_stride: tl.constexpr,
    block_index_stride: tl.constexpr,
    pos_index_stride: tl.constexpr,
):
    # One program scatters one token's per-head FP32 descale row [N] into
    # (blocks, N, Bs).
    token = tl.program_id(0)
    head_offs = tl.arange(0, num_heads)
    block = tl.load(block_ids + token * block_index_stride).to(tl.int64)
    pos = tl.load(pos_ids + token * pos_index_stride).to(tl.int64)
    value = tl.load(source + token * num_heads + head_offs)
    dst_offs = (
        block * block_stride
        + head_offs.to(tl.int64) * head_stride
        + pos * pos_stride
    )
    tl.store(cache + dst_offs, value)


def scatter_kv_payload(
    payload: torch.Tensor,
    cache: torch.Tensor,
    slot_index: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Scatter one quantized payload [T, N, D] into a 5-D PA_NZ packet view.

    ``cache`` is ``(blocks, N, D // 32, block_size, 32)`` with dim0 stride
    equal to the packet pitch (the per-page ``[K | K_descale | V]`` sections
    interleave, so dim0 stride never equals the shape's inner product);
    ``slot_index`` is the step-shared ``(block_ids, pos_ids)`` pair derived
    from the slot mapping. Padded rows arrive clamped to slot 0 upstream and
    land in the null block, which the block pool never hands to a request.

    Capture safe: no host-device sync, no data-dependent shapes.
    """
    block_ids, pos_ids = slot_index
    num_tokens = block_ids.numel()
    if num_tokens == 0:
        return
    num_heads, num_frags = cache.shape[1], cache.shape[2]
    if not (_is_pow2(num_heads) and _is_pow2(num_frags) and _is_pow2(32)):
        raise ValueError(
            "scatter_kv_payload requires power-of-two head and fragment counts "
            f"(got heads={num_heads}, frags={num_frags})."
        )
    source = payload.contiguous().view(torch.int8).view(-1)
    num_elements = source.numel()
    expected_elements = num_tokens * num_heads * num_frags * 32
    if num_elements != expected_elements:
        raise ValueError(
            f"KV payload contains {num_elements} elements; expected {expected_elements}."
        )
    _scatter_kv_payload_kernel[(num_tokens * num_heads,)](
        source,
        cache,
        block_ids,
        pos_ids,
        num_tokens,
        num_heads,
        num_frags,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        cache.stride(3),
        block_ids.stride(0),
        pos_ids.stride(0),
        num_warps=4,
    )


def scatter_k_descale(
    k_descale_tok: torch.Tensor,
    cache: torch.Tensor,
    slot_index: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Scatter per-token K descales [T, N] FP32 into the (blocks, N, Bs) view.

    ``slot_index`` is the same (block_ids, pos_ids) pair the payload scatter
    consumes, so a step's K payload and its descale land in the same slots.
    Capture safe: no host-device sync, no data-dependent shapes.
    """
    block_ids, pos_ids = slot_index
    num_tokens = block_ids.numel()
    if num_tokens == 0:
        return
    num_heads = cache.shape[1]
    if not _is_pow2(num_heads):
        raise ValueError(
            f"scatter_k_descale requires a power-of-two head count (got {num_heads})."
        )
    source = k_descale_tok.contiguous()
    if source.numel() != num_tokens * num_heads:
        raise ValueError(
            f"K descale contains {source.numel()} elements; "
            f"expected {num_tokens * num_heads}."
        )
    _scatter_k_descale_kernel[(num_tokens,)](
        source,
        cache,
        block_ids,
        pos_ids,
        num_tokens,
        num_heads,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        block_ids.stride(0),
        pos_ids.stride(0),
        num_warps=1,
    )
