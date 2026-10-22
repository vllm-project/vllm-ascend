# SPDX-License-Identifier: Apache-2.0
"""Direct gather-input packing and final SFA query unpacking for DCP decode."""

import torch

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton
except ImportError:
    triton = None
    tl = None


_NOPE_DIM = 512
_ROPE_DIM = 64
_PACKED_DIM = _NOPE_DIM + _ROPE_DIM
_MAX_DECODE_TOKENS = 12


if triton is not None:

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
    return (
        triton is not None
        and qn.device.type == "npu"
        and qr.device == qn.device
        and qn.dtype == qr.dtype == torch.bfloat16
        and qn.ndim == qr.ndim == 3
        and qn.shape[1] > 0
        and qn.shape[2] == _NOPE_DIM
        and tuple(qr.shape) == (qn.shape[0], qn.shape[1], _ROPE_DIM)
        and 1 < qn.shape[0] <= _MAX_DECODE_TOKENS
        and all(stride >= 0 for stride in (*qn.stride(), *qr.stride()))
    )


def prepare_query_head_major(qn: torch.Tensor, qr: torch.Tensor) -> torch.Tensor:
    if not can_prepare_query(qn, qr):
        raise ValueError("Unsupported native-DCP query preparation")
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
    if (
        triton is None
        or gathered.device.type != "npu"
        or gathered.dtype != torch.bfloat16
        or gathered.ndim != 3
        or gathered.shape[0] <= 0
        or gathered.shape[2] != _PACKED_DIM
        or not 1 < gathered.shape[1] <= _MAX_DECODE_TOKENS
        or not gathered.is_contiguous()
    ):
        raise ValueError("Unsupported native-DCP gathered query")
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
