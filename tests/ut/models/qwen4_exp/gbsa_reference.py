# SPDX-License-Identifier: Apache-2.0
"""Reference GBSA metadata preparation used by Qwen4Exp unit tests."""

import torch

GBSA_RESIDUAL_TOKENS = 3
GBSA_COMPRESS_RATIO = 4
GBSA_MAX_SPARSE_BLOCKS = 512


def reference_prepare(
    compact_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    num_kv_heads: int,
) -> tuple[torch.Tensor, ...]:
    sparse_width = compact_indices.shape[1] - GBSA_RESIDUAL_TOKENS
    row_requests = token_to_req.long()
    row_request_lens = seq_lens[row_requests].to(torch.int32)
    current_visible_lens = torch.minimum(
        logical_positions.to(torch.int32).add(1).clamp_min(0),
        row_request_lens,
    )
    tail = compact_indices[:, sparse_width:].to(torch.int32)
    valid_tail = (tail >= 0) & (tail < row_request_lens.unsqueeze(1))
    tail_end = torch.where(valid_tail, tail + 1, 0).amax(dim=1)
    row_visible_lens = torch.where(
        valid_tail.any(dim=1),
        tail_end,
        torch.div(current_visible_lens, GBSA_COMPRESS_RATIO, rounding_mode="floor") * GBSA_COMPRESS_RATIO,
    )
    sparse_idx = compact_indices[:, :sparse_width].to(torch.int32)
    full_blocks = torch.div(
        row_visible_lens,
        GBSA_COMPRESS_RATIO,
        rounding_mode="floor",
    )
    valid = (sparse_idx >= 0) & (sparse_idx < full_blocks.unsqueeze(1))
    order = torch.argsort((~valid).to(torch.int32), dim=1, stable=True)
    sparse_idx = sparse_idx.gather(1, order)
    full_sparse_count = valid.sum(dim=1).to(torch.int32)

    sparse_count = full_sparse_count
    sparse_idx = sparse_idx.unsqueeze(0).expand(num_kv_heads, -1, -1).contiguous()
    sparse_count = sparse_count.unsqueeze(0).expand(num_kv_heads, -1).contiguous()

    cu_q = torch.arange(compact_indices.shape[0] + 1, dtype=torch.int64, device=compact_indices.device)
    row_block_table = block_table[row_requests].contiguous()
    return sparse_idx, sparse_count, row_visible_lens, cu_q, row_block_table
