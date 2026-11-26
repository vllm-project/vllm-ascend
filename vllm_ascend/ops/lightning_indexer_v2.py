# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from typing import NamedTuple

import torch


class LightningIndexerV2Window(NamedTuple):
    block_table: torch.Tensor
    valid_pages: torch.Tensor
    visible_lengths: torch.Tensor
    metadata: torch.Tensor


class LightningIndexerV2Plan(NamedTuple):
    cu_query: torch.Tensor
    windows: tuple[LightningIndexerV2Window, ...]


def build_lightning_indexer_v2_plan(
    block_table: torch.Tensor,
    request_rows: torch.Tensor,
    local_visible_lengths: torch.Tensor,
    num_cache_pages: int,
    page_size: int,
    first_valid_page: int,
    num_heads_q: int,
) -> LightningIndexerV2Plan:
    """Represent each query row as one request with its own rank-local causal prefix."""
    rows = request_rows.numel()
    cu_query = torch.arange(rows + 1, dtype=torch.int32, device=block_table.device)
    pages = block_table.index_select(0, request_rows)
    valid_pages = (pages >= first_valid_page) & (pages < num_cache_pages)
    # The kernel may load any page inside the visible prefix. Redirect holes
    # to allocated storage and keep their logical positions for score masking.
    pages = torch.where(valid_pages, pages, 0).to(torch.int32)
    lengths = local_visible_lengths.reshape(rows)
    windows = []
    # Request every score within each window so invalid pages and arbitrary
    # native tie order cannot displace a valid candidate at the local cutoff.
    window_pages = 2048 // page_size
    for start in range(0, pages.shape[1], window_pages):
        stop = min(start + window_pages, pages.shape[1])
        width = (stop - start) * page_size
        visible = (lengths - start * page_size).clamp(0, width)
        metadata = torch.ops._C_ascend.npu_lightning_indexer_v2_metadata(cu_query, visible, num_heads_q, width)
        windows.append(
            LightningIndexerV2Window(pages[:, start:stop].contiguous(), valid_pages[:, start:stop], visible, metadata)
        )
    return LightningIndexerV2Plan(cu_query, tuple(windows))


def has_lightning_indexer_v2() -> bool:
    """Resolve the extension and both CANN operators during worker initialization."""
    try:
        import_module("vllm_ascend.vllm_ascend_C")
    except ImportError:
        return False
    probe = getattr(torch.ops._C_ascend, "is_lightning_indexer_v2_available", None)
    return probe is not None and probe()


def lightning_indexer_v2_candidates(
    query: torch.Tensor,
    weights: torch.Tensor,
    key_cache: torch.Tensor,
    plan: LightningIndexerV2Plan,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score every visible rank-local token with BF16 CANN LightningIndexerV2."""
    from vllm_ascend.ops.triton.sharded_indexer import restore_li_v2_scores

    rows = query.shape[0]
    page_size = key_cache.shape[1]
    capacity = sum(window.block_table.shape[1] * page_size for window in plan.windows)
    # Start in logical token order, including holes. A stable sort then selects
    # the earliest local ID on ties, independent of CANN's native Top-K order.
    dense = torch.full((rows, max(topk, capacity)), -torch.inf, device=query.device, dtype=torch.float32)
    q = query.contiguous()
    w = weights.float().contiguous()
    offset = 0
    for window in plan.windows:
        width = window.block_table.shape[1] * page_size
        indices, scores = torch.ops._C_ascend.npu_lightning_indexer_v2(
            q, key_cache, w, plan.cu_query, window.visible_lengths, window.block_table, window.metadata, width
        )
        restore_li_v2_scores(indices, scores, window.valid_pages, window.visible_lengths, dense, offset, page_size)
        offset += width
    best_scores, order = torch.sort(dense, dim=-1, descending=True, stable=True)
    best_scores = best_scores[:, :topk]
    best_ids = order[:, :topk].to(torch.int32).masked_fill_(best_scores == -torch.inf, -1)
    return best_ids.unsqueeze(1), best_scores.unsqueeze(1)
