# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Portable Ascend fallbacks for Qwen3.8-Flash-Next platform operators."""

import math

import torch
import torch.nn.functional as F


def grouped_gemma_rmsnorm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_groups: int,
) -> torch.Tensor:
    """Apply Gemma RMSNorm independently to each HyperConnection stream."""
    if hidden_states.shape[-1] % num_groups:
        raise ValueError("hidden size must be divisible by the HC stream count")
    group_dim = hidden_states.shape[-1] // num_groups
    grouped = hidden_states.unflatten(-1, (num_groups, group_dim)).float()
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normalized = grouped * torch.rsqrt(variance + eps)
    if weight.numel() == group_dim:
        affine = weight.reshape(1, 1, group_dim)
    elif weight.numel() == hidden_states.shape[-1]:
        affine = weight.reshape(1, num_groups, group_dim)
    else:
        raise ValueError("HC norm weight has an incompatible shape")
    return (normalized * (1.0 + affine.float())).flatten(-2).to(hidden_states.dtype)


def hc_silu(hidden_states: torch.Tensor, hc_count: int) -> torch.Tensor:
    return F.silu(hidden_states.float() / hc_count).to(hidden_states.dtype)


def hc_gate_mix(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if hidden_states.shape != gate.shape:
        raise ValueError("HC input and gate shapes must match")
    group_dim = hidden_states.shape[-1] // hc_count
    hidden_states = hidden_states.unflatten(-1, (hc_count, group_dim))
    gate = torch.sigmoid(gate.float()).unflatten(-1, (hc_count, group_dim))
    return (hidden_states.float() * gate).mean(dim=-2).to(hidden_states.dtype)


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    group_dim = residual.shape[-1] // hc_count
    residual_grouped = residual.unflatten(-1, (hc_count, group_dim))
    injection = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)
    output = residual_grouped.float() + block_output.float().unsqueeze(-2) * injection.unsqueeze(-1)
    return output.flatten(-2).to(residual.dtype)


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = hc_combine(residual, block_output, injection_logits, hc_count)
    normalized = grouped_gemma_rmsnorm(output, norm_weight, eps, hc_count)
    return output, normalized


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store QSA rows using device-native indexed writes."""
    if (
        cache.ndim != 4
        or cache.shape[0] <= 0
        or cache.shape[1] <= 0
        or cache.shape[2] <= 0
    ):
        raise ValueError("QSA cache must be [blocks, block_size, heads, width]")
    if rows.ndim == 2:
        rows = rows.unsqueeze(1)
    if rows.ndim != 3 or rows.shape[1:] != cache.shape[2:]:
        raise ValueError("QSA cache rows have an incompatible shape")
    slots = slot_mapping.reshape(-1).long()
    # Target prefill can have more padded slots than rows, while a draft pass
    # can have more rows than active slots. Only the aligned prefix represents
    # cache updates; the remainder is capacity padding.
    num_updates = min(slots.shape[0], rows.shape[0])
    slots = slots[:num_updates]
    cache_capacity = cache.shape[0] * cache.shape[1]
    valid = (slots >= 0) & (slots < cache_capacity)
    # Keep the update width static for NPU graph capture. ``masked_select``
    # creates a data-dependent output shape. Invalid rows map to slot 0 but
    # contribute a zero delta, so they cannot overwrite a real row there.
    safe_slots = slots.clamp(0, cache_capacity - 1)
    physical_blocks = torch.div(safe_slots, cache.shape[1], rounding_mode="floor")
    block_offsets = safe_slots.remainder(cache.shape[1])
    # Do not flatten the cache here. With an HND cache, ``cache`` is a
    # non-contiguous logical [block, token, head, dim] view and reshape may
    # silently materialize a copy. Two-dimensional indexed assignment keeps
    # writes attached to the original KV-cache allocation for both layouts.
    current_rows = cache[physical_blocks, block_offsets]
    update_rows = rows[:num_updates].to(cache.dtype)
    deltas = (update_rows - current_rows) * valid.view(-1, 1, 1).to(cache.dtype)
    cache.index_put_(
        (physical_blocks, block_offsets),
        deltas,
        accumulate=True,
    )


def qsa_compress_groups_with_ratio(
    raw_keys: torch.Tensor,
    raw_positions: torch.Tensor,
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool completed QSA groups from current rows and the per-request ring."""
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    rows = token_to_req.numel()
    if raw_keys.shape[:2] != (rows, 1):
        raise ValueError("QSA raw keys must be [rows, 1, head_size]")
    if rows == 0:
        return raw_keys.new_empty(raw_keys.shape), raw_positions.new_empty((0, 3))
    row_ids = torch.arange(rows, device=raw_keys.device)
    requests = token_to_req.long()
    request_count = query_start_loc.numel() - 1
    safe_requests = requests.clamp(0, max(request_count - 1, 0))
    query_starts = query_start_loc.index_select(0, safe_requests).long()
    chunk_starts = logical_positions.long() - (row_ids - query_starts)

    group_offsets = torch.arange(compress_ratio, device=raw_keys.device)
    member_positions = logical_positions.long().unsqueeze(1) - (compress_ratio - 1 - group_offsets)
    use_raw = member_positions >= chunk_starts.unsqueeze(1)
    raw_row_ids = query_starts.unsqueeze(1) + member_positions - chunk_starts.unsqueeze(1)
    raw_row_ids = raw_row_ids.clamp(0, max(rows - 1, 0))
    current_members = raw_keys[:, 0].index_select(0, raw_row_ids.reshape(-1)).reshape(rows, compress_ratio, -1)

    state_blocks = compressor_state_block_table[safe_requests, 0].long()
    safe_state_blocks = state_blocks.clamp(0, max(compressor_state_cache.shape[0] - 1, 0))
    state_offsets = member_positions.remainder(compressor_state_cache.shape[1])
    state_members = compressor_state_cache[safe_state_blocks.unsqueeze(1), state_offsets, 0]
    members = torch.where(use_raw.unsqueeze(-1), current_members, state_members)
    pooled = members.float().mean(dim=1, keepdim=True).to(raw_keys.dtype)

    valid = (
        (requests >= 0)
        & (requests < request_count)
        & (state_blocks >= 0)
        & (logical_positions >= compress_ratio - 1)
        & (compressed_slots >= 0)
    )
    pooled = torch.where(valid.reshape(-1, 1, 1), pooled, torch.zeros_like(pooled))

    first_positions = member_positions[:, 0]
    if rope_cache is None:
        first_rope_positions = first_positions.unsqueeze(1).expand(-1, 3)
    else:
        raw_first_rows = raw_row_ids[:, 0]
        raw_first_positions = raw_positions[:, 0].index_select(0, raw_first_rows)
        state_first_positions = rope_cache[safe_state_blocks, first_positions.remainder(rope_cache.shape[1]), 0]
        first_rope_positions = torch.where(use_raw[:, :1], raw_first_positions, state_first_positions)
    first_rope_positions = torch.where(valid.unsqueeze(1), first_rope_positions, torch.zeros_like(first_rope_positions))
    return pooled, first_rope_positions


def _paged_cache_rows(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    logical_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    page_size = cache.shape[1]
    logical_pages = torch.div(logical_positions.clamp_min(0), page_size, rounding_mode="floor")
    valid = (
        (request_ids >= 0)
        & (request_ids < block_table.shape[0])
        & (logical_positions >= 0)
        & (logical_pages < block_table.shape[1])
    )
    safe_requests = request_ids.clamp(0, max(block_table.shape[0] - 1, 0))
    safe_pages = logical_pages.clamp(0, max(block_table.shape[1] - 1, 0))
    physical_pages = block_table[safe_requests, safe_pages].long()
    valid &= (physical_pages >= 0) & (physical_pages < cache.shape[0])
    safe_physical_pages = physical_pages.clamp(0, max(cache.shape[0] - 1, 0))
    page_offsets = logical_positions.remainder(page_size)
    rows = cache[safe_physical_pages, page_offsets]
    return rows, valid


def qsa_select_paged_tokens(
    query: torch.Tensor,
    compressed_key_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select QSA token indices with portable Torch operators."""
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    row_count = query.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((row_count, output_width), dtype=torch.int32, device=query.device)
    compressed_capacity = block_table.shape[1] * compressed_key_cache.shape[1]
    block_topk = token_topk // compress_ratio
    if block_topk > compressed_capacity:
        raise ValueError("QSA top-k exceeds the compressed cache capacity")

    columns = torch.arange(compressed_capacity, device=query.device)
    # Advanced paged indexing materializes one cache row. Bound the complete
    # key-and-score working set rather than only the score tensor; at the
    # 262k context limit a single compressed key row is already tens of MiB.
    bytes_per_row = compressed_capacity * (
        compressed_key_cache.shape[2] * compressed_key_cache.shape[3] * compressed_key_cache.element_size()
        + query.shape[1] * 4
    )
    rows_per_chunk = max(1, (128 * 1024 * 1024) // max(bytes_per_row, 1))
    for row_start in range(0, row_count, rows_per_chunk):
        row_end = min(row_count, row_start + rows_per_chunk)
        row_slice = slice(row_start, row_end)
        request_ids = token_to_req[row_slice].long()
        logical = columns.unsqueeze(0).expand(row_end - row_start, -1)
        requests = request_ids.unsqueeze(1).expand_as(logical)
        keys, valid_cache = _paged_cache_rows(compressed_key_cache, block_table, requests, logical)
        keys = keys[..., 0, :]
        scores = torch.einsum("rhd,rcd->rhc", query[row_slice].float(), keys.float())
        scores = scores.clamp_min_(0).sum(dim=1) / math.sqrt(query.shape[-1])
        safe_requests = request_ids.clamp(0, max(sequence_lengths.shape[0] - 1, 0))
        visible = torch.minimum(
            (query_positions[row_slice].long() + 1) // compress_ratio,
            sequence_lengths.index_select(0, safe_requests).long() // compress_ratio,
        )
        visible_mask = columns.unsqueeze(0) < visible.unsqueeze(1)
        scores.masked_fill_(~(visible_mask & valid_cache), -torch.inf)
        selected_blocks = torch.topk(scores, block_topk, dim=-1).indices

        output_columns = torch.arange(output_width, device=query.device)
        block_rank = torch.div(output_columns, compress_ratio, rounding_mode="floor")
        offsets = output_columns.remainder(compress_ratio)
        safe_rank = block_rank.clamp_max(block_topk - 1)
        expanded = selected_blocks[:, safe_rank] * compress_ratio + offsets
        complete_blocks = visible.clamp_max(block_topk)
        expanded_count = complete_blocks * compress_ratio
        tail_start = ((query_positions[row_slice].long() + 1) // compress_ratio) * compress_ratio
        tail_offset = output_columns.unsqueeze(0) - expanded_count.unsqueeze(1)
        tail_count = query_positions[row_slice].long() + 1 - tail_start
        is_expanded = output_columns.unsqueeze(0) < expanded_count.unsqueeze(1)
        is_tail = (tail_offset >= 0) & (tail_offset < tail_count.unsqueeze(1)) & (tail_offset < compress_ratio - 1)
        tokens = torch.where(is_expanded, expanded, tail_start.unsqueeze(1) + tail_offset)
        request_lengths = sequence_lengths.index_select(0, safe_requests).long()
        valid_tokens = (is_expanded | is_tail) & (tokens < request_lengths.unsqueeze(1))
        out[row_slice].copy_(torch.where(valid_tokens, tokens, -1).to(torch.int32))
    return out


def qsa_sparse_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse paged GQA without CUDA-only kernels."""
    if out is None:
        out = torch.empty_like(query)
    row_count, num_query_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[2]
    if num_query_heads % num_kv_heads:
        raise ValueError("QSA query heads must be divisible by KV heads")
    group_size = num_query_heads // num_kv_heads
    rows_per_chunk = 8
    for row_start in range(0, row_count, rows_per_chunk):
        row_end = min(row_count, row_start + rows_per_chunk)
        row_slice = slice(row_start, row_end)
        logical = logical_indices[row_slice].long()
        request_ids = token_to_req[row_slice].long().unsqueeze(1).expand_as(logical)
        keys, valid = _paged_cache_rows(key_cache, block_table, request_ids, logical)
        values, _ = _paged_cache_rows(value_cache, block_table, request_ids, logical)
        q = query[row_slice].reshape(row_end - row_start, num_kv_heads, group_size, head_dim)
        logits = torch.einsum("rhgd,rkhd->rhgk", q.float(), keys.float())
        logits.mul_(head_dim**-0.5).masked_fill_(~valid[:, None, None, :], -torch.inf)
        probabilities = torch.softmax(logits, dim=-1)
        probabilities = torch.nan_to_num(probabilities)
        result = torch.einsum("rhgk,rkhd->rhgd", probabilities, values.float())
        out[row_slice].copy_(result.reshape(row_end - row_start, num_query_heads, head_dim).to(out.dtype))
    return out


def reshape_and_cache_qsa(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    head_dim: int,
) -> None:
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    qsa_store_cache_rows(key_cache, slot_mapping, key)
    qsa_store_cache_rows(value_cache, slot_mapping, value)
