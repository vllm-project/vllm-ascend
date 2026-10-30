# SPDX-License-Identifier: Apache-2.0
"""Direct gather-input packing and final SFA query unpacking for DCP decode."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit(do_not_specialize=["tokens", "programs", "heads"])
def _prepare_query_head_major(
    nope_ptr,
    rope_ptr,
    packed_ptr,
    nope_stride_token,
    nope_stride_head,
    nope_stride_feature,
    rope_stride_token,
    rope_stride_head,
    rope_stride_feature,
    tokens,
    programs,
    heads,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    n = tl.arange(0, NOPE_DIM)
    r = tl.arange(0, ROPE_DIM)
    for row in tl.range(tl.program_id(0), heads * tokens, programs):
        head, token = row // tokens, row % tokens
        a = tl.load(nope_ptr + token * nope_stride_token + head * nope_stride_head + n * nope_stride_feature)
        b = tl.load(rope_ptr + token * rope_stride_token + head * rope_stride_head + r * rope_stride_feature)
        tl.store(packed_ptr + row * (NOPE_DIM + ROPE_DIM) + n, a)
        tl.store(packed_ptr + row * (NOPE_DIM + ROPE_DIM) + NOPE_DIM + r, b)


@triton.jit(do_not_specialize=["tokens", "programs", "heads"])
def _unpack_query_fragments(
    gathered_ptr,
    nope_ptr,
    rope_ptr,
    tokens,
    programs,
    heads,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    n = tl.arange(0, NOPE_DIM)
    r = tl.arange(0, ROPE_DIM)
    for row in tl.range(tl.program_id(0), tokens * heads, programs):
        token, head = row // heads, row % heads
        base = (head * tokens + token) * (NOPE_DIM + ROPE_DIM)
        a = tl.load(gathered_ptr + base + n)
        b = tl.load(gathered_ptr + base + NOPE_DIM + r)
        tl.store(nope_ptr + row * NOPE_DIM + n, a)
        tl.store(rope_ptr + row * ROPE_DIM + r, b)


def prepare_query_head_major(qn: torch.Tensor, qr: torch.Tensor) -> torch.Tensor:
    tokens = qn.shape[0]
    local_heads = qn.shape[1]
    nope_dim, rope_dim = qn.shape[-1], qr.shape[-1]
    result = torch.empty((local_heads, tokens, nope_dim + rope_dim), dtype=qn.dtype, device=qn.device)
    init_device_properties_triton()
    cores = get_vectorcore_num()
    programs = min(local_heads * tokens, cores)
    _prepare_query_head_major[(programs,)](
        qn,
        qr,
        result,
        qn.stride(0),
        qn.stride(1),
        qn.stride(2),
        qr.stride(0),
        qr.stride(1),
        qr.stride(2),
        tokens,
        programs,
        local_heads,
        nope_dim,
        rope_dim,
    )
    return result


def unpack_query(gathered: torch.Tensor, split_sizes: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = gathered.shape[1]
    heads = gathered.shape[0]
    nope_dim, rope_dim = split_sizes
    qn = torch.empty((tokens, heads, nope_dim), dtype=gathered.dtype, device=gathered.device)
    qr = torch.empty((tokens, heads, rope_dim), dtype=gathered.dtype, device=gathered.device)
    init_device_properties_triton()
    cores = get_vectorcore_num()
    programs = min(tokens * heads, cores)
    _unpack_query_fragments[(programs,)](
        gathered,
        qn,
        qr,
        tokens,
        programs,
        heads,
        nope_dim,
        rope_dim,
    )
    return qn, qr
