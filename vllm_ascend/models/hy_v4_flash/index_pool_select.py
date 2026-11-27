# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.models.hy_v4_flash.reference import INDEX_DIM
from vllm.triton_utils import tl, triton

from .index_pool_cache import pool_key_addresses

_QUERIES = 6
_GROUP_TILE = 128
_GROUP_MAJOR_TILE = 64
_GROUP_MAJOR_HEADS = 16
_GROUP_MAJOR_INDEX_DIM = 128
_GROUP_MAJOR_RATIO = 4
_GROUP_MAJOR_BUDGET = 511


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
    GROUP_MAJOR: tl.constexpr,
):
    qid = QUERY_START + tl.arange(0, QUERY_BLOCK)
    q_head = QUERY_START * HEADS + tl.arange(0, QUERY_BLOCK * HEADS)
    lane = tl.arange(0, WIDTH)
    # IndexPool encodes E4M3 values with power-of-two row scales. The
    # reconstructed query values, like the pooled keys, fit in BF16.
    q = tl.load(query + q_head[:, None] * WIDTH + lane[None, :]).to(tl.bfloat16)
    weight = tl.load(head_weight + q_head)
    if GROUP_MAJOR:
        # Keep the head axis contiguous through the vector epilogue. Only the
        # reduced [group, query] tensor is transposed for query-major stores.
        qk = tl.dot(key, tl.trans(q), out_dtype=tl.float32)
        weighted = tl.maximum(qk, 0.0) * weight[None, :]
        score = tl.trans(tl.sum(weighted.reshape((GROUP_TILE, QUERY_BLOCK, HEADS)), axis=2))
    else:
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
    GROUP_MAJOR: tl.constexpr,
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
        # Both layouts share one key tile across the same 4+2 query split.
        _score_query_block(
            query,
            head_weight,
            visible,
            scores,
            key,
            group,
            0,
            4,
            MAX_GROUPS,
            WIDTH,
            HEADS,
            RATIO,
            BUDGET,
            GROUP_TILE,
            GROUP_MAJOR,
        )
        _score_query_block(
            query,
            head_weight,
            visible,
            scores,
            key,
            group,
            4,
            2,
            MAX_GROUPS,
            WIDTH,
            HEADS,
            RATIO,
            BUDGET,
            GROUP_TILE,
            GROUP_MAJOR,
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
    # Specialize the measured six-query/16-head path. Other shapes retain the
    # original query-major layout and 128-group tile. The smaller group-major
    # tile limits live key/accumulator storage without padding extra queries.
    group_major = (
        query.device.type == "npu"
        and query.shape == (_QUERIES, _GROUP_MAJOR_HEADS, INDEX_DIM)
        and weights.shape == (_QUERIES, _GROUP_MAJOR_HEADS)
        and visible.shape == (_QUERIES,)
        and query.is_contiguous()
        and weights.is_contiguous()
        and visible.is_contiguous()
        and scores.is_contiguous()
        and INDEX_DIM == _GROUP_MAJOR_INDEX_DIM
        and ratio == _GROUP_MAJOR_RATIO
        and budget == _GROUP_MAJOR_BUDGET
    )
    group_tile = _GROUP_MAJOR_TILE if group_major else _GROUP_TILE
    _score_paged_pool[(triton.cdiv(max_groups, group_tile),)](
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
        group_tile,
        _QUERIES,
        triton.next_power_of_2(_QUERIES),
        group_major,
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
    sparse = torch.cat((history, tail.masked_fill(tail > visible[:, None], -1)), dim=-1)
    # Short rows have no sparse scores, but still need their complete history.
    dense = torch.arange(sparse.shape[-1], device=visible.device).expand_as(sparse)
    dense = dense.masked_fill(dense > visible[:, None], -1)
    return torch.where((cutoff > budget)[:, None], sparse, dense)


def select_paged_pool(query, weights, cache, table, visible, page_size, max_groups, ratio, budget):
    """Score six verification queries against the BF16 keys in paged IndexPool."""
    scores = torch.empty((_QUERIES, max_groups), dtype=torch.float32, device=query.device)
    write_pool_scores(query, weights, cache, table, visible, page_size, max_groups, ratio, budget, scores)
    return select_pool_candidates(scores, visible, max_groups, ratio, budget)
