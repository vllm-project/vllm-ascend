# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.models.hy_v4_flash.reference import INDEX_DIM

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
