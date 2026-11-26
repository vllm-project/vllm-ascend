# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Experimental two-part reference for A5 FP8 QSFA decode."""

from collections.abc import Callable

import torch

_S2_TILE_SIZE = 128


def split_sparse_indices(indices: torch.Tensor) -> torch.Tensor:
    """Interleave the two compact candidate ranges of each TND query row.

    QSFA consumes nonnegative rank-local indices followed by -1 padding.
    Keep the token IDs unchanged; only the position in the list is split.
    The fixed output shape also works when valid counts vary across replays.
    """
    num_queries, _, capacity = indices.shape
    half = capacity // 2
    positions = torch.arange(half, device=indices.device, dtype=torch.int32)
    counts = (indices >= 0).sum(dim=-1, keepdim=True, dtype=torch.int32)
    # Put the extra compute tile in the first range for odd tile counts.
    first_size = ((counts + 2 * _S2_TILE_SIZE - 1) // (2 * _S2_TILE_SIZE)) * _S2_TILE_SIZE
    first = indices[..., :half].masked_fill(positions >= first_size, -1)
    offsets = first_size + positions
    second = indices.gather(-1, offsets.to(torch.int64)).masked_fill(offsets >= counts, -1)
    return torch.stack((first, second), dim=1).reshape(num_queries * 2, 1, half)


def merge_split_kv(
    partial_out: torch.Tensor, partial_max: torch.Tensor, partial_sum: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover the normalized output and FP32 max/sum for the whole rank."""
    m0, m1 = partial_max[:, ::2], partial_max[:, 1::2]
    s0, s1 = partial_sum[:, ::2], partial_sum[:, 1::2]
    valid0, valid1 = s0 > 0, s1 > 0
    empty_max = torch.finfo(partial_max.dtype).min
    merged_max = torch.maximum(m0.masked_fill(~valid0, empty_max), m1.masked_fill(~valid1, empty_max))
    # Mask before exp so an empty partition never evaluates exp(inf) or exp(NaN).
    w0 = (m0 - merged_max).masked_fill(~valid0, 0).exp() * s0
    w1 = (m1 - merged_max).masked_fill(~valid1, 0).exp() * s1
    merged_sum = w0 + w1
    denominator = merged_sum.masked_fill(merged_sum == 0, 1)
    weight0 = (w0 / denominator).squeeze(0).unsqueeze(-1)
    weight1 = (w1 / denominator).squeeze(0).unsqueeze(-1)
    result = partial_out[::2].float() * weight0 + partial_out[1::2].float() * weight1
    return result.to(partial_out.dtype), merged_max, merged_sum


def run_qsfa_split_kv(
    query: torch.Tensor,
    sparse_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    attend: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run both parts in one QSFA launch (non-causal, DCP-local candidates)."""
    capacity = sparse_indices.shape[-1]
    if (
        query.shape[0] == 0
        or sparse_indices.ndim != 3
        or sparse_indices.shape[1] != 1
        or capacity < 2 * _S2_TILE_SIZE
        or capacity % (2 * _S2_TILE_SIZE) != 0
    ):
        return attend(query, sparse_indices, cu_seqlens_q)

    split_indices = split_sparse_indices(sparse_indices)
    # Interleave two rows per query. Doubling the cumulative lengths retains
    # the same request/page table, also for zero-length and padded requests.
    partials = attend(query.repeat_interleave(2, dim=0), split_indices, cu_seqlens_q * 2)
    return merge_split_kv(*partials)
