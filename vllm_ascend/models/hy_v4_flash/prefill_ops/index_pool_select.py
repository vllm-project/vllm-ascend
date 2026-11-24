# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.models.hy_v4_flash.reference import INDEX_DIM, sparse_candidates
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend.device.device_config import is_950

from .index_pool_cache import gather_pool_keys


def select_paged_pool(query, weights, cache, table, visible, page_size, max_groups, ratio, budget):
    """Score paged IndexPool groups with FP32 accumulation and causal masking."""
    cutoff = visible // ratio
    recall = cutoff > budget
    groups = torch.arange(max_groups, device=query.device)
    last_group = torch.where(recall, cutoff, 0).amax()
    keys = gather_pool_keys(cache, table, groups, groups < last_group, page_size, ratio)

    queries, heads, _ = query.shape
    query = query.bfloat16().float().reshape(-1, INDEX_DIM)
    qk = (query @ keys.float().T).reshape(queries, heads, max_groups)
    scores = (qk.relu() * weights.unsqueeze(-1)).sum(dim=1)
    scores = scores.masked_fill(~(recall[:, None] & (groups < cutoff[:, None])), -torch.inf)

    offsets = torch.arange(ratio, device=visible.device)
    values, selected = scores.topk(budget, dim=-1)
    selected = selected.masked_fill(values == -torch.inf, max_groups).sort(dim=-1).values.unsqueeze(-1)
    history = selected * ratio + offsets
    history = history.masked_fill(selected >= cutoff[:, None, None], -1).flatten(1)
    tail = cutoff[:, None] * ratio + offsets
    return torch.cat((history, tail.masked_fill(tail > visible[:, None], -1)), dim=-1)


@triton.jit(do_not_specialize=["key_rows", "key_tiles", "query_rows"])
def _fuse_partial_index_scores(
    per_head,
    weights,
    positions,
    output,
    key_rows,
    key_tiles,
    query_rows,
    TILE: tl.constexpr,
):
    program = tl.program_id(0)
    query = program // key_tiles
    key = (program % key_tiles) * TILE + tl.arange(0, TILE)
    head = tl.arange(0, 16)
    live = key < key_rows
    head_scores = tl.load(
        per_head + (head[:, None] * query_rows + query) * key_rows + key[None, :],
        live[None, :],
        other=0,
    )
    head_weight = tl.load(weights + query * 16 + head)
    scores = tl.sum(tl.maximum(head_scores, 0.0) * head_weight[:, None], axis=0)
    visible_group = tl.load(positions + query).to(tl.int32) >> 2
    scores = tl.where(key < visible_group, scores, -float("inf"))
    tl.store(output + query * key_rows + key, scores, live)


@triton.jit(do_not_specialize=["count", "sentinel"])
def _mask_partial_topk_indices(values, selected, masked, count, sentinel, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    live = index < count
    score = tl.load(values + index, live, other=0)
    group = tl.load(selected + index, live, other=0)
    tl.store(masked + index, tl.where(score == -float("inf"), sentinel, group), live)


@triton.jit(do_not_specialize=["rows"])
def _expand_partial_index_candidates(ordered, positions, output, rows, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    row = index >> 11
    group = (index & 2047) >> 2
    lane = index & 3
    live = row < rows
    history = group < 511
    position = tl.load(positions + row, live, other=0).to(tl.int64)
    cutoff = position >> 2
    group_id = tl.load(ordered + row * 511 + group, live & history, other=0)
    history_id = tl.where(group_id < cutoff, (group_id << 2) + lane, -1)
    tail = (cutoff << 2) + lane
    candidate = tl.where(history, history_id, tl.where(tail <= position, tail, -1))
    tl.store(output + index, candidate, live)


def _select_partial_candidates(scores, positions, num_keys):
    values, selected = scores.topk(511, dim=-1)
    masked = torch.empty_like(selected)
    rows, budget = selected.shape
    tile = 512
    count = rows * budget
    _mask_partial_topk_indices[(triton.cdiv(count, tile),)](
        values,
        selected,
        masked,
        count,
        num_keys,
        tile,
    )
    ordered = masked.sort(dim=-1).values
    output = torch.empty((rows, 2048), dtype=selected.dtype, device=selected.device)
    _expand_partial_index_candidates[(triton.cdiv(rows * 2048, tile),)](
        ordered,
        positions,
        output,
        rows,
        tile,
    )
    return output


def select_partial_prefill_pool(query, weights, keys, positions, ratio, budget):
    """Select candidates for a partial Prefill chunk with the native TopK and Sort."""
    if (
        HAS_TRITON
        and query.device.type == "npu"
        and is_950()
        and ratio == 4
        and budget == 511
        and query.dtype == weights.dtype == keys.dtype == torch.float32
        and query.ndim == 3
        and query.shape[1:] == (16, INDEX_DIM)
        and 2 <= query.shape[0] < 16
        and query.is_contiguous()
        and weights.shape == query.shape[:2]
        and weights.is_contiguous()
        and keys.ndim == 2
        and keys.shape[1] == INDEX_DIM
        and keys.shape[0] >= 8192
        and keys.is_contiguous()
        and positions.shape == (query.shape[0],)
        and positions.is_contiguous()
        and positions.dtype in (torch.int32, torch.int64)
        and query.device == weights.device == keys.device == positions.device
    ):
        rows, num_keys = query.shape[0], keys.shape[0]
        per_head = torch.matmul(query.transpose(0, 1), keys.T)
        if per_head.is_contiguous():
            scores = torch.empty((rows, num_keys), device=query.device, dtype=torch.float32)
            tile = 512
            tiles = triton.cdiv(num_keys, tile)
            _fuse_partial_index_scores[(rows * tiles,)](
                per_head,
                weights,
                positions,
                scores,
                num_keys,
                tiles,
                rows,
                tile,
                enable_fp_fusion=False,
            )
            return _select_partial_candidates(scores, positions, num_keys)
    return sparse_candidates(query, weights, keys, positions, ratio, budget)
