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
    row_visible_lens = torch.minimum(
        logical_positions.to(torch.int32).add(1).clamp_min(0),
        row_request_lens,
    )
    sparse_idx = compact_indices[:, :sparse_width].to(torch.int32)
    full_blocks = torch.div(
        row_visible_lens,
        GBSA_COMPRESS_RATIO,
        rounding_mode="floor",
    )
    has_residual_block = row_visible_lens.remainder(GBSA_COMPRESS_RATIO).ne(0)
    valid = (sparse_idx >= 0) & (sparse_idx < full_blocks.unsqueeze(1))
    order = torch.argsort((~valid).to(torch.int32), dim=1, stable=True)
    sparse_idx = sparse_idx.gather(1, order)
    full_sparse_count = valid.sum(dim=1).to(torch.int32)

    # A3 GBSA does not accept residual_block_mode=1 in its metadata
    # scheduler. Represent the incomplete visible tail as an explicit sparse
    # block instead. The LI-v2 groups are restricted to full blocks above, so
    # the tail block cannot duplicate an existing valid selection.
    # A3 GBSA accepts at most 512 sparse entries. Normally append one slot so
    # every LI-v2 full-block selection is preserved. If LI-v2 already fills
    # the operator limit, reuse the final slot by dropping the last
    # (lowest-priority) full-block selection.
    if sparse_idx.shape[1] < GBSA_MAX_SPARSE_BLOCKS:
        sparse_idx = torch.cat(
            (sparse_idx, torch.full_like(sparse_idx[:, :1], -1)),
            dim=1,
        )
    sparse_capacity = torch.full_like(full_sparse_count, sparse_idx.shape[1])
    sparse_capacity = sparse_capacity - has_residual_block.to(torch.int32)
    sparse_count = torch.minimum(full_sparse_count, sparse_capacity)
    columns = torch.arange(sparse_idx.shape[1], device=sparse_idx.device)
    tail_slot = columns.unsqueeze(0).eq(sparse_count.long().unsqueeze(1))
    sparse_idx = torch.where(
        tail_slot & has_residual_block.unsqueeze(1),
        full_blocks.unsqueeze(1),
        sparse_idx,
    )
    sparse_count = sparse_count + has_residual_block.to(torch.int32)
    sparse_idx = sparse_idx.unsqueeze(0).expand(num_kv_heads, -1, -1).contiguous()
    sparse_count = sparse_count.unsqueeze(0).expand(num_kv_heads, -1).contiguous()

    cu_q = torch.arange(compact_indices.shape[0] + 1, dtype=torch.int64, device=compact_indices.device)
    row_block_table = block_table[row_requests].contiguous()
    return sparse_idx, sparse_count, row_visible_lens, cu_q, row_block_table
