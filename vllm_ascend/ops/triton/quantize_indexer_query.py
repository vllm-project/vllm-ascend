# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-head INT8 query quantization with the FP16 scales consumed by QLI."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["num_rows", "blocks_per_core"])
def _quantize_indexer_query_kernel(
    query_ptr,
    quantized_ptr,
    scale_ptr,
    num_rows,
    blocks_per_core,
    BLOCK_ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    MIN_SCALE: tl.constexpr,
):
    first_block = tl.program_id(0) * blocks_per_core
    columns = tl.arange(0, HEAD_DIM)
    for block in range(blocks_per_core):
        rows = (first_block + block) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        offsets = rows[:, None] * HEAD_DIM + columns[None, :]
        query = tl.load(query_ptr + offsets, rows[:, None] < num_rows, other=0).to(tl.float32)
        abs_max = tl.max(tl.abs(query), axis=1)
        # Quantize with the rounded FP16 scale, not the original FP32 value.
        scale = tl.div_rn(abs_max, QUANT_MAX).to(tl.float16).to(tl.float32)
        scale = tl.maximum(scale, MIN_SCALE)
        normalized = tl.div_rn(query, scale[:, None])
        quantized = tl.extra.cann.libdevice.nearbyint(normalized)
        quantized = tl.minimum(tl.maximum(quantized, -QUANT_MAX), QUANT_MAX).to(tl.int8)
        tl.store(quantized_ptr + offsets, quantized, rows[:, None] < num_rows)
        tl.store(scale_ptr + rows, scale, rows < num_rows)


def quantize_indexer_query(query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [tokens, heads, 128] queries to INT8 and FP16 per-head scales.

    Match the indexer's round-to-even and symmetric [-127, 127] range. Zero
    heads use the smallest positive FP16 subnormal so division stays defined.
    """
    assert query.ndim == 3 and query.shape[-1] == 128
    query = query.contiguous()
    quantized = torch.empty_like(query, dtype=torch.int8)
    scale = torch.empty(query.shape[:-1], dtype=torch.float16, device=query.device)
    num_rows = query.numel() // query.shape[-1]
    if num_rows == 0:
        return quantized, scale

    # Each core writes whole 32-byte groups of FP16 scales. Contiguous blocks
    # keep neighboring cores from racing on a partial output cache line.
    block_rows = 16
    num_blocks = triton.cdiv(num_rows, block_rows)
    grid = min(num_blocks, get_vectorcore_num())
    _quantize_indexer_query_kernel[(grid,)](
        query,
        quantized,
        scale,
        num_rows,
        blocks_per_core=triton.cdiv(num_blocks, grid),
        BLOCK_ROWS=block_rows,
        HEAD_DIM=query.shape[-1],
        QUANT_MAX=127.0,
        MIN_SCALE=2.0**-24,
    )
    return quantized, scale
