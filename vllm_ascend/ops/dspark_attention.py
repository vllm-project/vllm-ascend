# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

DSparkAttentionCustomOp = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        float,
    ],
    torch.Tensor,
]


def _validate_query_block_slots(request_slots: torch.Tensor, block_size: int) -> None:
    for block_offset in range(0, request_slots.numel(), block_size):
        block_slots = request_slots[block_offset : block_offset + block_size].to(torch.long)
        if block_slots.numel() == 0:
            continue
        if torch.any(block_slots != block_slots[0]):
            raise ValueError("DSpark query request_slots must be constant within each draft block")


def _gather_context_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_positions: torch.Tensor,
    cache_valid: torch.Tensor,
    request_slot: int,
    context_start: int,
    context_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context_end < context_start:
        empty = k_cache.new_empty((0,) + k_cache.shape[2:])
        return empty, v_cache.new_empty((0,) + v_cache.shape[2:])

    cache_capacity = k_cache.shape[1]
    ctx_positions = torch.arange(
        context_start,
        context_end + 1,
        dtype=torch.long,
        device=k_cache.device,
    )
    cache_indices = ctx_positions % cache_capacity
    cached_positions = cache_positions[request_slot, cache_indices].to(torch.long)
    valid = cache_valid[request_slot, cache_indices] & (cached_positions == ctx_positions)
    return k_cache[request_slot, cache_indices][valid], v_cache[request_slot, cache_indices][valid]


def _dspark_attention_reference(
    q: torch.Tensor,
    k_ctx: torch.Tensor,
    v_ctx: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    scores = torch.einsum("qhd,khd->qhk", q.float(), k_ctx.float()) * softmax_scale
    sink = attn_sink[: q.shape[1]].float().view(1, q.shape[1], 1)
    scores_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
    exp_scores = torch.exp(scores - scores_max)
    probs = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - scores_max))
    return torch.einsum("qhk,khd->qhd", probs, v_ctx.float()).to(q.dtype)


def _get_dspark_attention_custom_op(q: torch.Tensor) -> DSparkAttentionCustomOp | None:
    if q.device.type == "cpu":
        return None
    try:
        return torch.ops._C_ascend.dspark_attention
    except (AttributeError, RuntimeError):
        return None


def _maybe_call_dspark_attention_custom_op(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_positions: torch.Tensor,
    cache_valid: torch.Tensor,
    draft_k: torch.Tensor,
    draft_v: torch.Tensor,
    request_slots: torch.Tensor,
    positions: torch.Tensor,
    attn_sink: torch.Tensor,
    block_size: int,
    window_size: int,
    softmax_scale: float,
) -> torch.Tensor | None:
    custom_op = _get_dspark_attention_custom_op(q)
    if custom_op is None:
        return None
    return custom_op(
        q,
        k_cache,
        v_cache,
        cache_positions,
        cache_valid,
        draft_k,
        draft_v,
        request_slots,
        positions,
        attn_sink,
        block_size,
        window_size,
        softmax_scale,
    )


def dspark_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_positions: torch.Tensor,
    cache_valid: torch.Tensor,
    draft_k: torch.Tensor,
    draft_v: torch.Tensor,
    request_slots: torch.Tensor,
    positions: torch.Tensor,
    attn_sink: torch.Tensor,
    block_size: int,
    window_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    """DSpark block attention semantic entry point.

    This is the Python reference for the future Ascend C kernel. It keeps the
    kernel-facing API explicit: request-local rolling context cache plus the
    current non-causal draft block.
    """
    if positions.numel() == 0:
        return torch.empty_like(q)
    custom_out = _maybe_call_dspark_attention_custom_op(
        q,
        k_cache,
        v_cache,
        cache_positions,
        cache_valid,
        draft_k,
        draft_v,
        request_slots,
        positions,
        attn_sink,
        block_size,
        window_size,
        softmax_scale,
    )
    if custom_out is not None:
        return custom_out
    if request_slots.numel() != positions.numel():
        raise ValueError(
            "DSpark request_slots length must match query positions: "
            f"request_slots={request_slots.numel()}, positions={positions.numel()}"
        )
    if request_slots.device.type == "cpu":
        _validate_query_block_slots(request_slots, block_size)

    out = torch.empty_like(q)
    pos_long = positions.to(torch.long)
    request_slots_long = request_slots.to(torch.long)
    max_request_slots = k_cache.shape[0]

    for block_offset in range(0, positions.numel(), block_size):
        block_end = min(block_offset + block_size, positions.numel())
        block_pos = pos_long[block_offset:block_end]
        block_start = int(block_pos.min().item())
        context_end = block_start - 1
        context_start = max(0, context_end + 1 - window_size)
        request_slot = int(request_slots_long[block_offset].item())
        if request_slot >= max_request_slots:
            raise ValueError(
                "DSpark request slot exceeds preallocated cache slots: "
                f"slot={request_slot}, capacity={max_request_slots}"
            )

        k_ctx, v_ctx = _gather_context_kv(
            k_cache,
            v_cache,
            cache_positions,
            cache_valid,
            request_slot,
            context_start,
            context_end,
        )
        k_ctx = torch.cat([k_ctx, draft_k[block_offset:block_end]], dim=0)
        v_ctx = torch.cat([v_ctx, draft_v[block_offset:block_end]], dim=0)
        out[block_offset:block_end] = _dspark_attention_reference(
            q[block_offset:block_end],
            k_ctx,
            v_ctx,
            attn_sink,
            softmax_scale,
        )

    return out


__all__ = ["dspark_attention"]
