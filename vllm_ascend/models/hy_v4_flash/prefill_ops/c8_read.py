# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.hisparse.c8 import C8_ROW_BYTES, dequantize_c8_rows

_LATENT = 512
_ROPE = 64


def gather_c8_rows(cache, block_table, block_size, positions):
    """Read selected C8 rows through the page table, zeroing padded positions."""
    if not positions.numel():
        return cache.new_empty((*positions.shape, _LATENT + _ROPE), dtype=torch.float32)
    logical = positions.clamp_min(0).long().flatten()
    pages = block_table.index_select(0, logical // block_size).long()
    slots = pages * block_size + logical % block_size
    packed = cache.index_select(0, slots)[:, :C8_ROW_BYTES].contiguous()
    rows = dequantize_c8_rows(packed).view(*positions.shape, _LATENT + _ROPE)
    return rows.masked_fill(positions.unsqueeze(-1) < 0, 0)


def gather_c8_qk(query, cache, block_table, block_size, positions):
    """Return FP32 QK scores and the selected 512-wide latent values."""
    rows = gather_c8_rows(cache, block_table, block_size, positions)
    scores = torch.bmm(query.float(), rows.transpose(1, 2))
    return scores, rows[..., :_LATENT].contiguous()
