# SPDX-License-Identifier: Apache-2.0
"""Direct gather-input packing and final SFA query unpacking for DCP decode."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

_NOPE_DIM = 512
_ROPE_DIM = 64
_PACKED_DIM = _NOPE_DIM + _ROPE_DIM
_MAX_DECODE_TOKENS = 12


@triton.jit(do_not_specialize=["tokens", "programs", "heads"])
def _prepare_query_head_major(
    qn,
    qr,
    out,
    ns0,
    ns1,
    ns2,
    rs0,
    rs1,
    rs2,
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
        a = tl.load(qn + token * ns0 + head * ns1 + n * ns2)
        b = tl.load(qr + token * rs0 + head * rs1 + r * rs2)
        tl.store(out + row * (NOPE_DIM + ROPE_DIM) + n, a)
        tl.store(out + row * (NOPE_DIM + ROPE_DIM) + NOPE_DIM + r, b)


@triton.jit(do_not_specialize=["tokens", "programs", "heads"])
def _unpack_query_fragments(
    gathered,
    qn,
    qr,
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
        a = tl.load(gathered + base + n)
        b = tl.load(gathered + base + NOPE_DIM + r)
        tl.store(qn + row * NOPE_DIM + n, a)
        tl.store(qr + row * ROPE_DIM + r, b)


def can_prepare_query(qn: torch.Tensor, qr: torch.Tensor) -> bool:
    """Select the small BF16 query layout; the attention caller owns input validation."""
    return (
        qn.dtype == torch.bfloat16
        and qn.shape[-1] == _NOPE_DIM
        and qr.shape[-1] == _ROPE_DIM
        and 1 < qn.shape[0] <= _MAX_DECODE_TOKENS
    )


def prepare_query_head_major(qn: torch.Tensor, qr: torch.Tensor) -> torch.Tensor:
    tokens = qn.shape[0]
    local_heads = qn.shape[1]
    result = torch.empty((local_heads, tokens, _PACKED_DIM), dtype=qn.dtype, device=qn.device)
    init_device_properties_triton()
    cores = get_vectorcore_num()
    programs = min(local_heads * tokens, cores)
    _prepare_query_head_major[(programs,)](
        qn,
        qr,
        result,
        *qn.stride(),
        *qr.stride(),
        tokens,
        programs,
        local_heads,
        _NOPE_DIM,
        _ROPE_DIM,
    )
    return result


def unpack_query(gathered: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = gathered.shape[1]
    heads = gathered.shape[0]
    qn = torch.empty((tokens, heads, _NOPE_DIM), dtype=gathered.dtype, device=gathered.device)
    qr = torch.empty((tokens, heads, _ROPE_DIM), dtype=gathered.dtype, device=gathered.device)
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
        _NOPE_DIM,
        _ROPE_DIM,
    )
    return qn, qr
