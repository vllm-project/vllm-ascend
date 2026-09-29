# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _build_dspark_swa_indices_kernel(
    query_start_loc_ptr,
    seq_lens_ptr,
    indices_ptr,
    lengths_ptr,
    num_reqs,
    index_width,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_REQS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    token_idx = tl.program_id(0)
    col_block = tl.program_id(1)

    req_offsets = tl.arange(0, BLOCK_REQS)
    req_ends = tl.load(
        query_start_loc_ptr + req_offsets + 1,
        mask=req_offsets < num_reqs,
        other=0x7FFFFFFF,
    )
    req_idx = tl.sum((token_idx >= req_ends).to(tl.int32), axis=0)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len
    start_pos = tl.maximum(prefix_len - WINDOW_SIZE, 0)
    visible_len = seq_len - start_pos

    cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    valid_col = cols < index_width
    slot = tl.where(cols < visible_len, start_pos + cols, -1)
    tl.store(indices_ptr + token_idx * index_width + cols, slot, mask=valid_col)
    tl.store(lengths_ptr + token_idx, visible_len, mask=col_block == 0)


def build_dspark_swa_indices_triton(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_tokens: int,
    index_width: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build request-local DSpark SWA indices with one kernel launch."""
    indices = torch.empty(
        (num_tokens, 1, index_width),
        dtype=torch.int32,
        device=query_start_loc.device,
    )
    lengths = torch.empty(
        (num_tokens, 1),
        dtype=torch.int32,
        device=query_start_loc.device,
    )
    if num_tokens == 0:
        return indices, lengths

    num_reqs = seq_lens.shape[0]
    block_reqs = triton.next_power_of_2(max(1, num_reqs))
    block_cols = triton.next_power_of_2(index_width)
    grid = (num_tokens, triton.cdiv(index_width, block_cols))
    _build_dspark_swa_indices_kernel[grid](
        query_start_loc,
        seq_lens,
        indices,
        lengths,
        num_reqs,
        index_width,
        WINDOW_SIZE=window_size,
        BLOCK_REQS=block_reqs,
        BLOCK_COLS=block_cols,
    )
    return indices, lengths
