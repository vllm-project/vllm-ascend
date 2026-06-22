# SPDX-License-Identifier: Apache-2.0
"""Functional PyTorch ops for MiniMax-M3 sparse attention on Ascend."""

from __future__ import annotations

import math

import torch

SPARSE_BLOCK_SIZE = 128

_LOGGED_SPARSE_ATTN = False
_LOGGED_INDEXER = False


def log_sparse_attention_used() -> None:
    global _LOGGED_SPARSE_ATTN
    if not _LOGGED_SPARSE_ATTN:
        from vllm.logger import init_logger

        init_logger(__name__).warning(
            "MiniMax M3 sparse attention path is active (Ascend torch fallback)"
        )
        _LOGGED_SPARSE_ATTN = True


def log_indexer_used() -> None:
    global _LOGGED_INDEXER
    if not _LOGGED_INDEXER:
        from vllm.logger import init_logger

        init_logger(__name__).warning(
            "MiniMax M3 lightning indexer path is active (Ascend torch fallback)"
        )
        _LOGGED_INDEXER = True


def _get_index_cache(index_kv_cache: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    if isinstance(index_kv_cache, (tuple, list)):
        return index_kv_cache[0]
    if index_kv_cache.ndim == 3:
        return index_kv_cache
    if index_kv_cache.ndim == 4:
        return index_kv_cache
    if index_kv_cache.ndim == 5 and index_kv_cache.shape[0] == 2:
        return index_kv_cache[0]
    raise ValueError(f"Unexpected index cache ndim: {index_kv_cache.ndim}")


def _get_index_block(cache: torch.Tensor, page: int) -> torch.Tensor:
    if cache.ndim == 3:
        return cache[page]
    if cache.ndim == 4:
        # Ascend FullAttentionSpec cache layout: [num_blocks, block, 1, head].
        return cache[page, :, 0, :]
    raise ValueError(f"Unexpected index cache ndim: {cache.ndim}")


def _get_main_kv_caches(
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(kv_cache, (tuple, list)):
        return kv_cache[0], kv_cache[1]
    if kv_cache.ndim == 5 and kv_cache.shape[0] == 2:
        return kv_cache[0], kv_cache[1]
    raise ValueError(f"Unexpected main kv cache format: {type(kv_cache)}")


def _num_blocks_for_graph(max_seq_len: int, block_table: torch.Tensor) -> int:
    table_blocks = block_table.shape[-1]
    seq_blocks = max(1, math.ceil(max_seq_len / SPARSE_BLOCK_SIZE))
    return min(table_blocks, seq_blocks)


def _gather_index_cache_blocks(
    cache: torch.Tensor,
    pages: torch.Tensor,
) -> torch.Tensor:
    safe_pages = pages.clamp_min(0).long()
    if cache.ndim == 3:
        return cache[safe_pages]
    if cache.ndim == 4:
        return cache[safe_pages, :, 0, :]
    raise ValueError(f"Unexpected index cache ndim: {cache.ndim}")


def _gather_main_cache_blocks(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pages: torch.Tensor,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_pages = pages.clamp_min(0).long()
    rows = torch.arange(
        SPARSE_BLOCK_SIZE, dtype=torch.long, device=k_cache.device
    ).view(1, 1, 1, 1, SPARSE_BLOCK_SIZE)
    kv_heads = torch.arange(
        num_kv_heads, dtype=torch.long, device=k_cache.device
    ).view(1, num_kv_heads, 1, 1, 1)
    if k_cache.ndim == 4:
        k_sel = k_cache[safe_pages.unsqueeze(-1), rows, kv_heads, :]
        v_sel = v_cache[safe_pages.unsqueeze(-1), rows, kv_heads, :]
        return k_sel, v_sel
    if k_cache.ndim == 5 and k_cache.shape[2] == 1:
        k_sel = k_cache[safe_pages.unsqueeze(-1), rows, 0, kv_heads, :]
        v_sel = v_cache[safe_pages.unsqueeze(-1), rows, 0, kv_heads, :]
        return k_sel, v_sel
    raise ValueError(f"Unexpected main kv cache ndim: {k_cache.ndim}")


def _visible_block_mask(
    seq_lens: torch.Tensor,
    decode_query_len: int,
    max_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_offsets = torch.arange(
        decode_query_len, dtype=seq_lens.dtype, device=seq_lens.device
    )
    q_abs = seq_lens[:, None] - decode_query_len + q_offsets[None, :]
    block_ids = torch.arange(max_blocks, dtype=seq_lens.dtype, device=seq_lens.device)
    visible_blocks = torch.clamp(
        torch.div(q_abs + SPARSE_BLOCK_SIZE, SPARSE_BLOCK_SIZE, rounding_mode="floor"),
        min=0,
        max=max_blocks,
    )
    visible = block_ids.view(1, 1, max_blocks) < visible_blocks[:, :, None]
    return visible, q_abs


def _graph_safe_index_topk_from_scores(
    scores: torch.Tensor,
    valid_blocks: torch.Tensor,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    # scores: [batch, num_heads, decode_query_len, max_blocks]
    # valid_blocks: [batch, decode_query_len, max_blocks]
    masked_scores = scores.masked_fill(~valid_blocks[:, None, :, :], float("-inf"))
    if init_blocks > 0 or local_blocks > 0:
        max_blocks = scores.shape[-1]
        block_ids = torch.arange(max_blocks, dtype=torch.long, device=scores.device)
        forced = torch.zeros_like(valid_blocks)
        if init_blocks > 0:
            forced |= block_ids.view(1, 1, max_blocks) < init_blocks
        if local_blocks > 0:
            visible_counts = valid_blocks.sum(dim=-1, dtype=torch.long)
            local_start = torch.clamp(visible_counts - local_blocks, min=0)
            block_ids_view = block_ids.view(1, 1, max_blocks)
            forced |= (block_ids_view >= local_start[:, :, None]) & (
                block_ids_view < visible_counts[:, :, None]
            )
        masked_scores = masked_scores.masked_fill(
            forced[:, None, :, :] & valid_blocks[:, None, :, :], float("inf")
        )
    effective_topk = min(topk, scores.shape[-1])
    values, indices = torch.topk(masked_scores, k=effective_topk, dim=-1)
    indices = indices.to(torch.int32)
    indices = torch.where(
        torch.isneginf(values),
        torch.full_like(indices, -1),
        indices,
    )
    if effective_topk < topk:
        pad_shape = list(indices.shape)
        pad_shape[-1] = topk - effective_topk
        indices = torch.cat(
            [
                indices,
                torch.full(
                    pad_shape, -1, dtype=indices.dtype, device=indices.device
                ),
            ],
            dim=-1,
        )
    return indices


def _graph_safe_index_decode_torch(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: float,
    decode_query_len: int,
) -> torch.Tensor:
    cache = _get_index_cache(index_kv_cache)
    total_q, num_idx_heads, head_dim = idx_q.shape
    num_reqs = min(seq_lens.shape[0], total_q // decode_query_len)
    max_blocks = _num_blocks_for_graph(max_seq_len, block_table)

    block_table = block_table[:num_reqs, :max_blocks]
    seq_lens = seq_lens[:num_reqs].long()
    pages = block_table.long()
    page_valid = pages >= 0
    k_blocks = _gather_index_cache_blocks(cache, pages).float()

    q = idx_q[: num_reqs * decode_query_len].view(
        num_reqs, decode_query_len, num_idx_heads, head_dim
    )
    q = q.permute(0, 2, 1, 3).float() * sm_scale
    k_flat = k_blocks.reshape(num_reqs, max_blocks * SPARSE_BLOCK_SIZE, head_dim)
    token_scores = torch.matmul(q, k_flat.transpose(1, 2).unsqueeze(1))
    token_scores = token_scores.view(
        num_reqs, num_idx_heads, decode_query_len, max_blocks, SPARSE_BLOCK_SIZE
    )

    block_visible, q_abs = _visible_block_mask(seq_lens, decode_query_len, max_blocks)
    block_ids = torch.arange(max_blocks, dtype=torch.long, device=idx_q.device)
    offsets = torch.arange(SPARSE_BLOCK_SIZE, dtype=torch.long, device=idx_q.device)
    token_positions = (
        block_ids[:, None] * SPARSE_BLOCK_SIZE + offsets[None, :]
    ).view(1, 1, max_blocks, SPARSE_BLOCK_SIZE)
    token_valid = (
        page_valid[:, None, :, None]
        & block_visible[:, :, :, None]
        & (token_positions <= q_abs[:, :, None, None])
    )
    token_scores = token_scores.masked_fill(
        ~token_valid[:, None, :, :, :], float("-inf")
    )
    block_scores = token_scores.max(dim=-1).values
    valid_blocks = block_visible & page_valid[:, None, :]
    topk_idx = _graph_safe_index_topk_from_scores(
        block_scores, valid_blocks, topk, init_blocks, local_blocks
    )
    return topk_idx.permute(1, 0, 2, 3).reshape(
        num_idx_heads, num_reqs * decode_query_len, topk
    )


def _graph_safe_sparse_attn_decode_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
    max_seq_len: int,
) -> None:
    k_cache, v_cache = _get_main_kv_caches(kv_cache)
    total_q, num_heads, head_dim = q.shape
    num_reqs = min(seq_lens.shape[0], total_q // decode_query_len)
    max_blocks = _num_blocks_for_graph(max_seq_len, block_table)
    topk = topk_idx.shape[-1]
    gqa = num_heads // num_kv_heads

    q_view = q[: num_reqs * decode_query_len].view(
        num_reqs, decode_query_len, num_heads, head_dim
    )
    q_view = q_view.view(num_reqs, decode_query_len, num_kv_heads, gqa, head_dim)
    q_view = q_view.permute(0, 2, 1, 3, 4).float()

    selected = topk_idx[:, : num_reqs * decode_query_len].view(
        num_kv_heads, num_reqs, decode_query_len, topk
    )
    selected = selected.permute(1, 0, 2, 3).long()
    safe_selected = selected.clamp(min=0, max=max_blocks - 1)
    block_rows = block_table[:num_reqs, :max_blocks].long()
    gather_rows = block_rows[:, None, None, :].expand(
        num_reqs, num_kv_heads, decode_query_len, max_blocks
    )
    pages = torch.gather(gather_rows, dim=-1, index=safe_selected)
    selected_valid = (selected >= 0) & (selected < max_blocks) & (pages >= 0)

    k_sel, v_sel = _gather_main_cache_blocks(
        k_cache, v_cache, pages, num_kv_heads
    )
    seq_lens = seq_lens[:num_reqs].long()
    q_offsets = torch.arange(
        decode_query_len, dtype=torch.long, device=q.device
    )
    q_abs = seq_lens[:, None] - decode_query_len + q_offsets[None, :]
    offsets = torch.arange(SPARSE_BLOCK_SIZE, dtype=torch.long, device=q.device)
    token_positions = selected[..., None] * SPARSE_BLOCK_SIZE + offsets.view(
        1, 1, 1, 1, SPARSE_BLOCK_SIZE
    )
    token_valid = selected_valid[..., None] & (
        token_positions <= q_abs[:, None, :, None, None]
    )

    k_flat = k_sel.reshape(
        num_reqs, num_kv_heads, decode_query_len, topk * SPARSE_BLOCK_SIZE, head_dim
    ).float()
    v_flat = v_sel.reshape(
        num_reqs, num_kv_heads, decode_query_len, topk * SPARSE_BLOCK_SIZE, head_dim
    ).float()
    flat_valid = token_valid.reshape(
        num_reqs, num_kv_heads, decode_query_len, topk * SPARSE_BLOCK_SIZE
    )

    scores = torch.matmul(q_view, k_flat.transpose(-1, -2)) * sm_scale
    scores = scores.masked_fill(~flat_valid[:, :, :, None, :], -1.0e30)
    probs = torch.softmax(scores, dim=-1)
    probs = probs.masked_fill(~flat_valid[:, :, :, None, :], 0.0)
    out = torch.matmul(probs, v_flat).to(q.dtype)
    out = out.permute(0, 2, 1, 3, 4).reshape(
        num_reqs * decode_query_len, num_heads, head_dim
    )
    output[: num_reqs * decode_query_len] = out


def _forced_block_ids(
    num_blocks_visible: int,
    init_blocks: int,
    local_blocks: int,
) -> list[int]:
    forced: list[int] = []
    for i in range(min(init_blocks, num_blocks_visible)):
        if i not in forced:
            forced.append(i)
    start = max(0, num_blocks_visible - local_blocks)
    for i in range(start, num_blocks_visible):
        if i not in forced:
            forced.append(i)
    return forced


def _topk_from_scores(
    scores: torch.Tensor,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    num_blocks = scores.shape[0]
    if num_blocks <= topk:
        out = torch.full((topk,), -1, dtype=torch.int32, device=scores.device)
        if num_blocks > 0:
            out[:num_blocks] = torch.arange(
                num_blocks, dtype=torch.int32, device=scores.device
            )
        return out

    forced = _forced_block_ids(num_blocks, init_blocks, local_blocks)
    remaining = topk - len(forced)
    if remaining > 0 and num_blocks > 0:
        mask = torch.ones(num_blocks, dtype=torch.bool, device=scores.device)
        if forced:
            mask[torch.tensor(forced, device=scores.device, dtype=torch.long)] = False
        masked_scores = scores.masked_fill(~mask, float("-inf"))
        _, extra = torch.topk(masked_scores, k=remaining)
        forced_tensor = (
            torch.tensor(forced, device=scores.device, dtype=torch.long)
            if forced
            else scores.new_empty(0, dtype=torch.long)
        )
        picked = torch.cat([forced_tensor, extra])[:topk]
        out = torch.full((topk,), -1, dtype=torch.int32, device=scores.device)
        out[:topk] = picked.to(torch.int32)
        return out
    out = torch.full((topk,), -1, dtype=torch.int32, device=scores.device)
    picked = forced[:topk]
    if picked:
        out[: len(picked)] = torch.tensor(picked, dtype=torch.int32, device=scores.device)
    return out


@torch.no_grad()
def minimax_m3_index_score_torch(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
) -> torch.Tensor:
    """Return score [num_kv_heads, total_q, max_block]."""
    log_indexer_used()
    cache = _get_index_cache(index_kv_cache)
    cu_seqlens_q_cpu = cu_seqlens_q.detach().cpu().tolist()
    seq_lens_cpu = seq_lens.detach().cpu().tolist()
    prefix_lens_cpu = prefix_lens.detach().cpu().tolist()
    block_table_cpu = block_table.detach().cpu().tolist()
    total_q, num_idx_heads, _ = idx_q.shape
    batch = len(cu_seqlens_q_cpu) - 1
    max_seq_len = max(seq_lens_cpu) if seq_lens_cpu else 0
    max_block = math.ceil(max_seq_len / SPARSE_BLOCK_SIZE)
    scores = torch.full(
        (num_idx_heads, total_q, max_block),
        float("-inf"),
        dtype=torch.float32,
        device=idx_q.device,
    )

    for b in range(batch):
        q_start = cu_seqlens_q_cpu[b]
        q_end = cu_seqlens_q_cpu[b + 1]
        seq_len = seq_lens_cpu[b]
        prefix_len = prefix_lens_cpu[b]
        bt_row = block_table_cpu[b]
        for t in range(q_end - q_start):
            q_abs = prefix_len + t
            hi_blocks = min(
                math.ceil(seq_len / SPARSE_BLOCK_SIZE),
                math.ceil((q_abs + 1) / SPARSE_BLOCK_SIZE),
            )
            for h in range(num_idx_heads):
                q_heads = idx_q[q_start + t]
                per_head = torch.full(
                    (hi_blocks,), float("-inf"), dtype=torch.float32, device=idx_q.device
                )
                for blk in range(hi_blocks):
                    page = bt_row[blk]
                    if page < 0:
                        continue
                    k_block = _get_index_block(cache, page)
                    pos_end = min(
                        SPARSE_BLOCK_SIZE, max(0, seq_len - blk * SPARSE_BLOCK_SIZE)
                    )
                    if pos_end <= 0:
                        continue
                    k_sel = k_block[:pos_end]
                    per_head[blk] = (q_heads[h] * sm_scale * k_sel).sum(-1).max()
                scores[h, q_start + t, :hi_blocks] = per_head
    return scores


@torch.no_grad()
def minimax_m3_index_topk_torch(
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    log_indexer_used()
    num_idx_heads, total_q, _ = score.shape
    batch = cu_seqlens_q.shape[0] - 1
    cu_seqlens_q_cpu = cu_seqlens_q.detach().cpu().tolist()
    prefix_lens_cpu = prefix_lens.detach().cpu().tolist()
    topk_idx = torch.full(
        (num_idx_heads, total_q, topk), -1, dtype=torch.int32, device=score.device
    )
    for b in range(batch):
        q_start = cu_seqlens_q_cpu[b]
        q_end = cu_seqlens_q_cpu[b + 1]
        prefix_len = prefix_lens_cpu[b]
        for t in range(q_end - q_start):
            q_abs = prefix_len + t
            hi_blocks = math.ceil((q_abs + 1) / SPARSE_BLOCK_SIZE)
            for h in range(num_idx_heads):
                topk_idx[h, q_start + t] = _topk_from_scores(
                    score[h, q_start + t, :hi_blocks], topk, init_blocks, local_blocks
                )
    return topk_idx


@torch.no_grad()
def minimax_m3_index_decode_torch(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    sm_scale: float,
    decode_query_len: int,
) -> torch.Tensor:
    log_indexer_used()
    return _graph_safe_index_decode_torch(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len,
        topk,
        init_blocks,
        local_blocks,
        sm_scale,
        decode_query_len,
    )


def _fill_missing_visible_topk(
    topk_idx: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> torch.Tensor:
    topk = topk_idx.shape[-1]
    filled = topk_idx.clone()
    batch = cu_seqlens_q.shape[0] - 1
    for b in range(batch):
        q_start = int(cu_seqlens_q[b].item())
        q_end = int(cu_seqlens_q[b + 1].item())
        prefix_len = int(prefix_lens[b].item())
        for t in range(q_end - q_start):
            tok = q_start + t
            if bool((filled[:, tok] >= 0).any().item()):
                continue
            q_abs = prefix_len + t
            visible = min(topk, math.ceil((q_abs + 1) / SPARSE_BLOCK_SIZE))
            if visible > 0:
                filled[:, tok, :visible] = torch.arange(
                    visible,
                    dtype=torch.int32,
                    device=filled.device,
                )
    return filled


def _gather_request_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, dtype=torch.long, device=k_cache.device)
    pages = block_table_row[positions // SPARSE_BLOCK_SIZE].long()
    rows = positions % SPARSE_BLOCK_SIZE
    k_req = k_cache[pages, rows]
    v_req = v_cache[pages, rows]
    if k_req.ndim == 3 and k_req.shape[1] == 1:
        k_req = k_req.squeeze(1)
        v_req = v_req.squeeze(1)
    return k_req, v_req


def _sparse_masked_attention_request(
    q_req: torch.Tensor,
    k_req: torch.Tensor,
    v_req: torch.Tensor,
    topk_req: torch.Tensor,
    prefix_len: int,
    sm_scale: float,
) -> torch.Tensor:
    q_len, num_heads, _ = q_req.shape
    seq_len, num_kv_heads, _ = k_req.shape
    gqa = num_heads // num_kv_heads
    out = torch.empty_like(q_req)

    positions = torch.arange(seq_len, dtype=torch.long, device=q_req.device)
    key_blocks = positions // SPARSE_BLOCK_SIZE
    q_pos = prefix_len + torch.arange(q_len, dtype=torch.long, device=q_req.device)
    causal_mask = positions.unsqueeze(0) <= q_pos.unsqueeze(1)

    for kv_h in range(num_kv_heads):
        selected = topk_req[kv_h].long()
        selected_mask = (key_blocks[None, :, None] == selected[:, None, :]).any(-1)
        attn_mask = causal_mask & selected_mask
        heads = slice(kv_h * gqa, (kv_h + 1) * gqa)

        q_heads = q_req[:, heads].transpose(0, 1).float()
        k_head = k_req[:, kv_h].transpose(0, 1).float().expand(gqa, -1, -1)
        scores = torch.bmm(q_heads, k_head).transpose(0, 1) * sm_scale
        scores = scores.masked_fill(~attn_mask[:, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[:, heads] = torch.matmul(
            probs.transpose(0, 1),
            v_req[:, kv_h].float(),
        ).transpose(0, 1).to(q_req.dtype)
    return out


@torch.no_grad()
def minimax_m3_sparse_attn_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
) -> None:
    log_sparse_attention_used()
    k_cache, v_cache = _get_main_kv_caches(kv_cache)
    topk_idx = _fill_missing_visible_topk(topk_idx, cu_seqlens_q, prefix_lens)
    batch = cu_seqlens_q.shape[0] - 1
    cu_seqlens_q_cpu = cu_seqlens_q.detach().cpu().tolist()
    seq_lens_cpu = seq_lens.detach().cpu().tolist()
    prefix_lens_cpu = prefix_lens.detach().cpu().tolist()
    for b in range(batch):
        q_start = cu_seqlens_q_cpu[b]
        q_end = cu_seqlens_q_cpu[b + 1]
        seq_len = seq_lens_cpu[b]
        prefix_len = prefix_lens_cpu[b]
        bt_row = block_table[b]
        k_req, v_req = _gather_request_kv(k_cache, v_cache, bt_row, seq_len)
        if k_req.ndim == 2:
            k_req = k_req.unsqueeze(1)
            v_req = v_req.unsqueeze(1)
        q_req = q[q_start:q_end]
        topk_req = topk_idx[:, q_start:q_end]
        output[q_start:q_end] = _sparse_masked_attention_request(
            q_req, k_req, v_req, topk_req, prefix_len, sm_scale
        )


@torch.no_grad()
def minimax_m3_sparse_attn_decode_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
    max_seq_len: int | None = None,
) -> None:
    log_sparse_attention_used()
    if max_seq_len is None:
        max_seq_len = block_table.shape[-1] * SPARSE_BLOCK_SIZE
    _graph_safe_sparse_attn_decode_torch(
        q,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        num_kv_heads,
        sm_scale,
        output,
        decode_query_len,
        max_seq_len,
    )
