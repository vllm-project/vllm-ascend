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
    seq_lens_cpu: tuple[int, ...] | None = None,
    block_table_cpu: tuple[tuple[int, ...], ...] | None = None,
) -> torch.Tensor:
    log_indexer_used()
    cache = _get_index_cache(index_kv_cache)
    total_q, num_idx_heads, _ = idx_q.shape
    num_reqs = seq_lens.shape[0]
    if seq_lens_cpu is None:
        seq_lens_cpu = tuple(int(x) for x in seq_lens.detach().cpu().tolist())
    if block_table_cpu is None:
        block_table_cpu = tuple(
            tuple(int(x) for x in row)
            for row in block_table.detach().cpu().tolist()
        )
    topk_idx = torch.full(
        (num_idx_heads, total_q, topk), -1, dtype=torch.int32, device=idx_q.device
    )
    for req in range(num_reqs):
        seq_len = seq_lens_cpu[req]
        bt_row = block_table_cpu[req]
        for q_off in range(decode_query_len):
            tok = req * decode_query_len + q_off
            q_abs = seq_len - decode_query_len + q_off
            hi_blocks = min(
                math.ceil(seq_len / SPARSE_BLOCK_SIZE),
                math.ceil((q_abs + 1) / SPARSE_BLOCK_SIZE),
            )
            for h in range(num_idx_heads):
                per_head = torch.full(
                    (hi_blocks,), float("-inf"), dtype=torch.float32, device=idx_q.device
                )
                q_h = idx_q[tok, h]
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
                    per_head[blk] = (q_h * sm_scale * k_sel).sum(-1).max()
                topk_idx[h, tok] = _topk_from_scores(
                    per_head, topk, init_blocks, local_blocks
                )
    return topk_idx


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


def _fill_missing_visible_topk_decode(
    topk_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
) -> torch.Tensor:
    topk = topk_idx.shape[-1]
    filled = topk_idx.clone()
    num_reqs = seq_lens.shape[0]
    for req in range(num_reqs):
        seq_len = int(seq_lens[req].item())
        for q_off in range(decode_query_len):
            tok = req * decode_query_len + q_off
            if tok >= filled.shape[1]:
                continue
            if bool((filled[:, tok] >= 0).any().item()):
                continue
            q_abs = seq_len - decode_query_len + q_off
            visible = min(topk, math.ceil((q_abs + 1) / SPARSE_BLOCK_SIZE))
            if visible > 0:
                filled[:, tok, :visible] = torch.arange(
                    visible,
                    dtype=torch.int32,
                    device=filled.device,
                )
    return filled


def _gather_request_kv_cpu_block_table(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table_row: tuple[int, ...],
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, dtype=torch.long, device=k_cache.device)
    block_ids = positions // SPARSE_BLOCK_SIZE
    pages = torch.tensor(block_table_row, device=k_cache.device, dtype=torch.long)
    page_indices = pages[block_ids]
    rows = positions % SPARSE_BLOCK_SIZE
    k_req = k_cache[page_indices, rows]
    v_req = v_cache[page_indices, rows]
    if k_req.ndim == 3 and k_req.shape[1] == 1:
        k_req = k_req.squeeze(1)
        v_req = v_req.squeeze(1)
    return k_req, v_req


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
    seq_lens_cpu: tuple[int, ...] | None = None,
    block_table_cpu: tuple[tuple[int, ...], ...] | None = None,
) -> None:
    log_sparse_attention_used()
    k_cache, v_cache = _get_main_kv_caches(kv_cache)
    if seq_lens_cpu is None:
        topk_idx = _fill_missing_visible_topk_decode(
            topk_idx, seq_lens, decode_query_len
        )
        seq_lens_cpu = tuple(int(x) for x in seq_lens.detach().cpu().tolist())
    num_reqs = len(seq_lens_cpu)
    for req in range(num_reqs):
        seq_len = seq_lens_cpu[req]
        if block_table_cpu is not None:
            k_req, v_req = _gather_request_kv_cpu_block_table(
                k_cache, v_cache, block_table_cpu[req], seq_len
            )
        else:
            bt_row = block_table[req]
            k_req, v_req = _gather_request_kv(k_cache, v_cache, bt_row, seq_len)
        if k_req.ndim == 2:
            k_req = k_req.unsqueeze(1)
            v_req = v_req.unsqueeze(1)
        for q_off in range(decode_query_len):
            tok = req * decode_query_len + q_off
            q_pos = seq_len - decode_query_len + q_off
            output[tok : tok + 1] = _sparse_masked_attention_request(
                q[tok : tok + 1],
                k_req,
                v_req,
                topk_idx[:, tok : tok + 1],
                q_pos,
                sm_scale,
            )
