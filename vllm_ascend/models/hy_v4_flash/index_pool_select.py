# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.models.hy_v4_flash.reference import INDEX_DIM
from vllm.triton_utils import tl, triton

from .index_pool_cache import pool_key_addresses

_QUERIES = 6
_GROUP_TILE = 128


@triton.jit
def _score_query_block(
    query,
    head_weight,
    visible,
    scores,
    key,
    group,
    QUERY_START: tl.constexpr,
    QUERY_BLOCK: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    RATIO: tl.constexpr,
    BUDGET: tl.constexpr,
    GROUP_TILE: tl.constexpr,
):
    qid = QUERY_START + tl.arange(0, QUERY_BLOCK)
    q_head = QUERY_START * HEADS + tl.arange(0, QUERY_BLOCK * HEADS)
    lane = tl.arange(0, WIDTH)
    # IndexPool encodes E4M3 values with power-of-two row scales. The
    # reconstructed query values, like the pooled keys, fit in BF16.
    q = tl.load(query + q_head[:, None] * WIDTH + lane[None, :]).to(tl.bfloat16)
    weight = tl.load(head_weight + q_head)
    qk = tl.dot(q, tl.trans(key), out_dtype=tl.float32)
    weighted = tl.maximum(qk, 0.0) * weight[:, None]
    score = tl.sum(weighted.reshape((QUERY_BLOCK, HEADS, GROUP_TILE)), axis=1)
    end_group = tl.load(visible + qid) // RATIO
    recall = (end_group[:, None] > BUDGET) & (group[None, :] < end_group[:, None])
    tl.store(
        scores + qid[:, None] * MAX_GROUPS + group[None, :],
        tl.where(recall, score, float("-inf")),
        group[None, :] < MAX_GROUPS,
    )


@triton.jit
def _score_paged_pool(
    query,
    head_weight,
    cache,
    block_table,
    visible,
    scores,
    MAX_GROUPS: tl.constexpr,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    BUDGET: tl.constexpr,
    GROUP_TILE: tl.constexpr,
    QUERIES: tl.constexpr,
    QUERY_TILE: tl.constexpr,
):
    start = tl.program_id(0) * GROUP_TILE
    group = start + tl.arange(0, GROUP_TILE)
    qid = tl.arange(0, QUERY_TILE)
    end_group = tl.load(visible + qid, qid < QUERIES, -1) // RATIO
    recall = (end_group > BUDGET) & (qid < QUERIES)
    max_visible = tl.max(tl.where(recall, end_group, 0), axis=0)
    if start < max_visible:
        lane = tl.arange(0, WIDTH)
        live = (group < max_visible) & (group < MAX_GROUPS)
        address = pool_key_addresses(block_table, group, lane, live, ROW_WIDTH, PAGE_SIZE, RATIO, WIDTH // RATIO)
        key = tl.load(cache + address, live[:, None], 0)
        # Sharing one key tile and using 4+2 queries keeps a 128-group tile in UB.
        _score_query_block(
            query, head_weight, visible, scores, key, group, 0, 4, MAX_GROUPS, WIDTH, HEADS, RATIO, BUDGET, GROUP_TILE
        )
        _score_query_block(
            query, head_weight, visible, scores, key, group, 4, 2, MAX_GROUPS, WIDTH, HEADS, RATIO, BUDGET, GROUP_TILE
        )
    else:
        tl.store(
            scores + qid[:, None] * MAX_GROUPS + group[None, :],
            float("-inf"),
            (qid[:, None] < QUERIES) & (group[None, :] < MAX_GROUPS),
        )


def write_pool_scores(query, weights, cache, table, visible, page_size, max_groups, ratio, budget, scores):
    """Write one request's scores into a contiguous FP32 destination."""
    cache = cache.view(torch.bfloat16)
    _score_paged_pool[(triton.cdiv(max_groups, _GROUP_TILE),)](
        query,
        weights,
        cache,
        table,
        visible,
        scores,
        max_groups,
        INDEX_DIM,
        query.shape[1],
        cache.shape[-1],
        page_size,
        ratio,
        budget,
        _GROUP_TILE,
        _QUERIES,
        triton.next_power_of_2(_QUERIES),
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
    )


def select_pool_candidates(scores, visible, max_groups, ratio, budget):
    cutoff = visible // ratio
    offsets = torch.arange(ratio, device=visible.device)
    values, selected = scores.topk(budget, dim=-1)
    selected = selected.masked_fill(values == -torch.inf, max_groups).sort(dim=-1).values.unsqueeze(-1)
    history = selected * ratio + offsets
    history = history.masked_fill(selected >= cutoff[:, None, None], -1).flatten(1)
    tail = cutoff[:, None] * ratio + offsets
    return torch.cat((history, tail.masked_fill(tail > visible[:, None], -1)), dim=-1)


def select_paged_pool(query, weights, cache, table, visible, page_size, max_groups, ratio, budget):
    """Score six verification queries against the BF16 keys in paged IndexPool."""
    scores = torch.empty((_QUERIES, max_groups), dtype=torch.float32, device=query.device)
    write_pool_scores(query, weights, cache, table, visible, page_size, max_groups, ratio, budget, scores)
    return select_pool_candidates(scores, visible, max_groups, ratio, budget)
