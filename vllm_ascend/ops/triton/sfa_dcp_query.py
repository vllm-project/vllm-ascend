# SPDX-License-Identifier: Apache-2.0
"""Direct gather-input packing and final SFA query unpacking for DCP8 decode."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_LOCAL_HEADS = 8
_TOTAL_HEADS = 64
_NOPE_DIM = 512
_ROPE_DIM = 64
_PACKED_DIM = _NOPE_DIM + _ROPE_DIM
_MAX_DECODE_TOKENS = 12


if triton is not None:

    @triton.jit
    def _prepare_query_head_major(
        qn,
        qr,
        out,
        ns0: tl.constexpr,
        ns1: tl.constexpr,
        ns2: tl.constexpr,
        rs0: tl.constexpr,
        rs1: tl.constexpr,
        rs2: tl.constexpr,
        tokens: tl.constexpr,
        programs: tl.constexpr,
        heads: tl.constexpr,
        nope_dim: tl.constexpr,
        rope_dim: tl.constexpr,
    ):
        n = tl.arange(0, nope_dim)
        r = tl.arange(0, rope_dim)
        for row in tl.range(tl.program_id(0), heads * tokens, programs):
            head, token = row // tokens, row % tokens
            a = tl.load(qn + token * ns0 + head * ns1 + n * ns2)
            b = tl.load(qr + token * rs0 + head * rs1 + r * rs2)
            tl.store(out + row * (nope_dim + rope_dim) + n, a)
            tl.store(out + row * (nope_dim + rope_dim) + nope_dim + r, b)

    @triton.jit
    def _unpack_query_fragments(
        gathered,
        qn,
        qr,
        tokens: tl.constexpr,
        programs: tl.constexpr,
        heads: tl.constexpr,
        nope_dim: tl.constexpr,
        rope_dim: tl.constexpr,
    ):
        n = tl.arange(0, nope_dim)
        r = tl.arange(0, rope_dim)
        for row in tl.range(tl.program_id(0), tokens * heads, programs):
            token, head = row // heads, row % heads
            base = (head * tokens + token) * (nope_dim + rope_dim)
            a = tl.load(gathered + base + n)
            b = tl.load(gathered + base + nope_dim + r)
            tl.store(qn + row * nope_dim + n, a)
            tl.store(qr + row * rope_dim + r, b)


def can_prepare_query(qn: torch.Tensor, qr: torch.Tensor) -> bool:
    return (
        triton is not None
        and qn.device.type == "npu"
        and qr.device == qn.device
        and qn.dtype == qr.dtype == torch.bfloat16
        and qn.ndim == qr.ndim == 3
        and tuple(qn.shape[1:]) == (_LOCAL_HEADS, _NOPE_DIM)
        and tuple(qr.shape) == (qn.shape[0], _LOCAL_HEADS, _ROPE_DIM)
        and 1 < qn.shape[0] <= _MAX_DECODE_TOKENS
        and all(stride >= 0 for stride in (*qn.stride(), *qr.stride()))
    )


def prepare_query_head_major(qn: torch.Tensor, qr: torch.Tensor) -> torch.Tensor:
    if not can_prepare_query(qn, qr):
        raise ValueError("Unsupported native-DCP8 query preparation")
    tokens = qn.shape[0]
    result = torch.empty((_LOCAL_HEADS, tokens, _PACKED_DIM), dtype=qn.dtype, device=qn.device)
    cores = triton.runtime.driver.active.utils.get_device_properties(qn.device.index)["num_vectorcore"]
    programs = min(_LOCAL_HEADS * tokens, cores)
    _prepare_query_head_major[(programs,)](
        qn,
        qr,
        result,
        *qn.stride(),
        *qr.stride(),
        tokens,
        programs,
        _LOCAL_HEADS,
        _NOPE_DIM,
        _ROPE_DIM,
    )
    return result


def unpack_query(gathered: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        triton is None
        or gathered.device.type != "npu"
        or gathered.dtype != torch.bfloat16
        or gathered.ndim != 3
        or gathered.shape[0] != _TOTAL_HEADS
        or gathered.shape[2] != _PACKED_DIM
        or not 1 < gathered.shape[1] <= _MAX_DECODE_TOKENS
        or not gathered.is_contiguous()
    ):
        raise ValueError("Unsupported native-DCP8 gathered query")
    tokens = gathered.shape[1]
    qn = torch.empty((tokens, _TOTAL_HEADS, _NOPE_DIM), dtype=gathered.dtype, device=gathered.device)
    qr = torch.empty((tokens, _TOTAL_HEADS, _ROPE_DIM), dtype=gathered.dtype, device=gathered.device)
    cores = triton.runtime.driver.active.utils.get_device_properties(gathered.device.index)["num_vectorcore"]
    programs = min(tokens * _TOTAL_HEADS, cores)
    _unpack_query_fragments[(programs,)](
        gathered,
        qn,
        qr,
        tokens,
        programs,
        _TOTAL_HEADS,
        _NOPE_DIM,
        _ROPE_DIM,
    )
    return qn, qr
