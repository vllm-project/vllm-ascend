# SPDX-License-Identifier: Apache-2.0
"""Derived NPU BF16 pages for Decode LightningIndexerV2.

A 64-token physical KV page holds sixteen pooled 128-dimensional keys. Refresh
from packed cache when an eager request arrives, including after a prefix hit
or page reuse. During Decode, mirror newly completed groups with write_pages.
The helper keeps the latest PR's per-query dense history and padding rules.
"""

import cann_ops_transformer  # noqa: F401
import torch
from vllm.triton_utils import tl, triton

from .index_pool_cache import pool_key_addresses

WIDTH = 128
RATIO = 4
BUDGET = 511
COPY_GROUPS = 16


@triton.jit
def _refresh_pages(
    cache,
    pages,
    table,
    END: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    COPY_GROUPS: tl.constexpr = 16,
    WIDTH: tl.constexpr = 128,
    RATIO: tl.constexpr = 4,
):
    group = tl.program_id(0) * COPY_GROUPS + tl.arange(0, COPY_GROUPS)
    lane = tl.arange(0, WIDTH)
    live = group < END
    address = pool_key_addresses(table, group, lane, live, ROW_WIDTH, PAGE_SIZE, RATIO, WIDTH // RATIO)
    block = tl.load(table + group // (PAGE_SIZE // RATIO), live, 0).to(tl.int64)
    row = block * (PAGE_SIZE // RATIO) + group % (PAGE_SIZE // RATIO)
    key = tl.load(cache + address, live[:, None], 0)
    tl.store(pages + row[:, None] * WIDTH + lane[None, :], key, live[:, None])


@triton.jit
def _write_pages(
    keys,
    pages,
    table,
    first,
    end,
    COUNT: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    COPY_GROUPS: tl.constexpr = 16,
    WIDTH: tl.constexpr = 128,
    RATIO: tl.constexpr = 4,
):
    row = tl.arange(0, COPY_GROUPS)
    group = tl.load(first) + row
    live = (row < COUNT) & (group < tl.load(end))
    lane = tl.arange(0, WIDTH)
    block = tl.load(table + group // (PAGE_SIZE // RATIO), live, 0).to(tl.int64)
    target = block * (PAGE_SIZE // RATIO) + group % (PAGE_SIZE // RATIO)
    value = tl.load(keys + row[:, None] * WIDTH + lane[None, :], live[:, None], 0)
    tl.store(pages + target[:, None] * WIDTH + lane[None, :], value, live[:, None])


def refresh_pages(cache, pages, table, block_size, sequence_length):
    groups = sequence_length // RATIO
    if groups:
        _refresh_pages[(triton.cdiv(groups, COPY_GROUPS),)](
            cache.view(torch.bfloat16), pages, table, groups, cache.shape[-1] // 2, block_size, num_warps=1
        )


def write_pages(keys, pages, table, first, end, block_size):
    _write_pages[(1,)](keys, pages, table, first, end, keys.shape[0], block_size, num_warps=1)


def select_pages(query, weights, pages, table, visible, max_groups):
    batch, queries = visible.shape
    lengths = visible[:, -1].clamp_min(0).div(RATIO, rounding_mode="floor").to(torch.int32)
    residual = visible[:, -1].clamp_min(0).remainder(RATIO).to(torch.int32)
    lengths = lengths.clamp_min(BUDGET + 1)
    metadata = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
        16,
        1,
        WIDTH,
        BUDGET,
        seqused_k=lengths,
        cmp_residual_k=residual,
        batch_size=batch,
        max_seqlen_q=queries,
        max_seqlen_k=max_groups,
        layout_q="BSND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=RATIO,
    )
    groups, _ = torch.ops.cann_ops_transformer.lightning_indexer(
        query.to(torch.bfloat16).view(batch, queries, 16, WIDTH),
        pages,
        weights.view(batch, queries, 16),
        BUDGET,
        seqused_k=lengths,
        cmp_residual_k=residual,
        block_table=table.contiguous(),
        metadata=metadata,
        max_seqlen_q=queries,
        layout_q="BSND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=RATIO,
        return_value=0,
    )
    groups = groups.view(batch, queries, BUDGET)
    cutoff = visible // RATIO
    ordered = groups.masked_fill((groups < 0) | (groups >= cutoff[..., None]), max_groups).sort(dim=-1).values
    offsets = torch.arange(RATIO, device=query.device)
    history = ordered[..., None] * RATIO + offsets
    history.masked_fill_(ordered[..., None] == max_groups, -1)
    tail = cutoff[..., None] * RATIO + offsets
    tail.masked_fill_(tail > visible[..., None], -1)
    sparse = torch.cat((history.flatten(-2), tail), dim=-1)
    # Keep the latest PR's dense short-context and padding contract.
    dense = torch.arange((BUDGET + 1) * RATIO, device=query.device).expand(batch, queries, -1)
    dense = dense.masked_fill(dense > visible[..., None], -1)
    return torch.where((cutoff > BUDGET)[..., None], sparse, dense)
