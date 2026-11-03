# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _select_context_window_indices_kernel(
    query_start_loc,
    cu_context_lens,
    indices,
    valid,
    KEEP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    begin = tl.load(query_start_loc + req).to(tl.int64)
    end = tl.load(query_start_loc + req + 1).to(tl.int64)
    dest_begin = tl.load(cu_context_lens + req).to(tl.int64)
    dest_end = tl.load(cu_context_lens + req + 1).to(tl.int64)
    offset = tl.arange(0, BLOCK)
    row = tl.maximum(begin, end - KEEP) + offset
    dest = dest_begin + offset
    mask = dest < dest_end
    is_valid = mask & (row < end)
    tl.store(indices + dest, tl.where(is_valid, row, 0), mask)
    tl.store(valid + dest, is_valid, mask)


@triton.jit
def _select_uniform_context_window_indices_kernel(
    query_start_loc,
    indices,
    valid,
    KEEP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    begin = tl.load(query_start_loc + req).to(tl.int64)
    end = tl.load(query_start_loc + req + 1).to(tl.int64)
    offset = tl.arange(0, BLOCK)
    row = tl.maximum(begin, end - KEEP) + offset
    mask = offset < KEEP
    dest = req.to(tl.int64) * KEEP + offset
    is_valid = mask & (row < end)
    tl.store(indices + dest, tl.where(is_valid, row, 0), mask)
    tl.store(valid + dest, is_valid, mask)


def select_context_window_indices(
    query_start_loc: torch.Tensor,
    cu_context_lens: torch.Tensor,
    num_context_tokens: int,
    keep: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack bounded context suffixes without reading device request lengths."""
    indices = torch.empty(
        num_context_tokens,
        dtype=torch.int64,
        device=query_start_loc.device,
    )
    valid = torch.empty_like(indices, dtype=torch.bool)
    num_reqs = cu_context_lens.numel() - 1
    _select_context_window_indices_kernel[(num_reqs,)](
        query_start_loc,
        cu_context_lens,
        indices,
        valid,
        keep,
        triton.next_power_of_2(keep),
    )
    return indices, valid


def select_uniform_context_window_indices(
    query_start_loc: torch.Tensor,
    num_reqs: int,
    keep: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select suffixes when every request has the full ``keep`` capacity."""
    indices = torch.empty(
        num_reqs * keep,
        dtype=torch.int64,
        device=query_start_loc.device,
    )
    valid = torch.empty_like(indices, dtype=torch.bool)
    _select_uniform_context_window_indices_kernel[(num_reqs,)](
        query_start_loc,
        indices,
        valid,
        keep,
        triton.next_power_of_2(keep),
    )
    return indices, valid
