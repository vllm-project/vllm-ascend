# SPDX-License-Identifier: Apache-2.0
"""Raw-bit O/LSE packing for the measured K3 DCP8 decode shapes.

Adapted from PR #16350's bit-preserving exchange. Communication and deferred
combine stay in sfa_cp; no sparse-index or query-layout behavior is changed.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit
def _pack(
    o,
    lse,
    send,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    LS0: tl.constexpr,
    LS1: tl.constexpr,
    T: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    ROW_WORDS: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(T * 96, BLOCK_ROWS)
    percore = tl.cdiv(tiles, tl.num_programs(0))
    for tile in range(tl.program_id(0) * percore, tl.minimum((tl.program_id(0) + 1) * percore, tiles)):
        rows = tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        head = rows // T
        token = rows - head * T
        words = tl.arange(0, 256)
        values = tl.load(
            o + token[:, None] * OS0 + head[:, None] * OS1 + words[None, :], rows[:, None] < T * 96, other=0
        )
        tl.store(send + rows[:, None] * ROW_WORDS + words[None, :], values, rows[:, None] < T * 96)
        stats = tl.load(lse + token * LS0 + head * LS1, rows < T * 96, other=0).to(tl.int32, bitcast=True)
        if ROW_WORDS == 272:
            tail = tl.arange(0, 16)
            tail_values = tl.where(tail[None, :] == 0, stats[:, None], 0)
            tl.store(
                send + rows[:, None] * ROW_WORDS + 256 + tail[None, :],
                tail_values,
                rows[:, None] < T * 96,
            )
        else:
            tl.store(send + rows * ROW_WORDS + 256, stats, rows < T * 96)


def can_use_raw_dcp_exchange(
    output: torch.Tensor,
    lse: torch.Tensor,
    scatter_size: int,
    scatter_dim: int,
    *,
    has_pcp: bool = False,
) -> bool:
    """Shape/dtype-only dispatch: ranks must select the same wire protocol."""
    return (
        not has_pcp
        and scatter_size == 8
        and scatter_dim == 1
        and output.device.type == "npu"
        and output.dtype == torch.bfloat16
        and output.ndim == 3
        and output.shape[0] in (4, 8, 16, 32, 64)
        and output.shape[1:] == (96, 512)
        and lse.device == output.device
        and lse.dtype == torch.float32
        and lse.shape == (*output.shape[:2], 1)
    )


def _view_direct_dcp_wire(output: torch.Tensor) -> torch.Tensor | None:
    """Recover NTD_DCP storage; only the producer contract makes its tail valid."""
    if output.dtype != torch.bfloat16 or output.ndim != 3:
        return None
    tokens, heads, dim = output.shape
    if (
        tokens not in (4, 8, 16, 32, 64)
        or (heads, dim) != (96, 512)
        or output.stride() != (544, tokens * 544, 1)
        or output.storage_offset() != 0
        or output.untyped_storage().nbytes() < heads * tokens * 544 * output.element_size()
    ):
        return None
    # as_strided and dtype views only change metadata. The shared owner remains
    # alive through this alias until All2All completes on the exchange stream.
    physical = output.as_strided((heads, tokens, 544), (tokens * 544, 544, 1))
    return physical.view(torch.int32).view(8, 12, tokens, 272)


def pack_raw_dcp_output_lse(output: torch.Tensor, lse: torch.Tensor, raw_row_words: int = 257) -> torch.Tensor:
    """Use a uniform wire pitch; a local view failure never changes its count."""
    if raw_row_words not in (257, 272):
        raise ValueError("Raw DCP row pitch must be 257 or 272 INT32 words.")
    if not can_use_raw_dcp_exchange(output, lse, 8, 1):
        raise ValueError("Raw DCP packing requires BF16 T4/8/16/32/64 H96 D512 and FP32 LSE.")
    if raw_row_words == 272:
        direct = _view_direct_dcp_wire(output)
        if direct is not None:
            return direct
    # Rank-local strides/capacity only select view versus pack, not wire pitch.
    if output.stride(-1) != 1 or any(stride % 2 for stride in output.stride()[:2]):
        output = output.contiguous()
    if output.storage_offset() % 2:
        output = output.clone()
    words = output.view(torch.int32)
    tokens = output.shape[0]
    send = torch.empty((8, 12, tokens, raw_row_words), dtype=torch.int32, device=output.device)
    init_device_properties_triton()
    block_rows = 32 if tokens == 64 else 16
    programs = min(triton.cdiv(tokens * 96, block_rows), get_vectorcore_num())
    _pack[(programs,)](
        words,
        lse,
        send,
        *words.stride()[:2],
        *lse.stride()[:2],
        tokens,
        block_rows,
        raw_row_words,
        multibuffer=False,
    )
    return send
