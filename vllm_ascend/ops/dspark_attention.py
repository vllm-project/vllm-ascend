# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from vllm.logger import logger

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

DSPARK_SAS_MASK_MODE = 4
DSPARK_SAS_CMP_MASK_MODE = 3


def _dspark_sas_window(block_size: int, window_size: int) -> tuple[int, int, int]:
    return (
        DSPARK_SAS_MASK_MODE,
        window_size + block_size - 1,
        block_size - 1,
    )


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


def _get_dspark_sas_ops(q: torch.Tensor) -> tuple[Callable, Callable] | None:
    if q.device.type == "cpu":
        return None
    try:
        return (
            torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata,
            torch.ops._C_ascend.npu_sparse_attn_sharedkv,
        )
    except (AttributeError, RuntimeError):
        return None


def _call_dspark_sas_block(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    block_size: int,
    window_size: int,
    metadata_op: Callable,
    attn_op: Callable,
) -> torch.Tensor:
    _, ori_win_left, ori_win_right = _dspark_sas_window(block_size, window_size)
    cu_seqlens_q = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device)
    cu_seqlens_kv = torch.tensor([0, packed_kv.shape[0]], dtype=torch.int32, device=q.device)
    sinks = attn_sink[: q.shape[1]].float().contiguous()
    metadata = metadata_op(
        num_heads_q=q.shape[1],
        num_heads_kv=1,
        head_dim=q.shape[2],
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_kv,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=None,
        batch_size=1,
        max_seqlen_q=q.shape[0],
        max_seqlen_kv=packed_kv.shape[0],
        cmp_ratio=1,
        ori_mask_mode=DSPARK_SAS_MASK_MODE,
        cmp_mask_mode=DSPARK_SAS_CMP_MASK_MODE,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q="TND",
        layout_kv="TND",
        has_ori_kv=True,
        has_cmp_kv=False,
        device=str(q.device),
    )
    return attn_op(
        q,
        ori_kv=packed_kv,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=softmax_scale,
        cmp_ratio=1,
        ori_mask_mode=DSPARK_SAS_MASK_MODE,
        cmp_mask_mode=DSPARK_SAS_CMP_MASK_MODE,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q="TND",
        layout_kv="TND",
    )[0]


def _maybe_call_dspark_sas_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    cache_positions: torch.Tensor,
    cache_valid: torch.Tensor,
    draft_k: torch.Tensor,
    request_slots: torch.Tensor,
    positions: torch.Tensor,
    attn_sink: torch.Tensor,
    block_size: int,
    window_size: int,
    softmax_scale: float,
) -> torch.Tensor | None:
    ops = _get_dspark_sas_ops(q)
    if ops is None:
        return None
    metadata_op, attn_op = ops
    out = torch.empty_like(q)
    pos_long = positions.to(torch.long)
    request_slots_long = request_slots.to(torch.long)
    max_request_slots = k_cache.shape[0]

    try:
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
            k_ctx, _ = _gather_context_kv(
                k_cache,
                k_cache,
                cache_positions,
                cache_valid,
                request_slot,
                context_start,
                context_end,
            )
            packed_kv = torch.cat(
                [k_ctx[:, :1, :], draft_k[block_offset:block_end, :1, :]],
                dim=0,
            ).contiguous()
            out[block_offset:block_end] = _call_dspark_sas_block(
                q[block_offset:block_end].contiguous(),
                packed_kv,
                attn_sink,
                softmax_scale,
                block_size,
                window_size,
                metadata_op,
                attn_op,
            )
    except RuntimeError as err:
        logger.warning_once("DSpark SAS attention failed; falling back to torch attention: %s", err)
        return None
    return out


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
    *,
    shared_kv: bool = False,
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

    if shared_kv:
        sas_out = _maybe_call_dspark_sas_attention(
            q,
            k_cache,
            cache_positions,
            cache_valid,
            draft_k,
            request_slots,
            positions,
            attn_sink,
            block_size,
            window_size,
            softmax_scale,
        )
        if sas_out is not None:
            return sas_out

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
        if shared_kv:
            v_ctx = k_ctx
            draft_v_block = draft_k[block_offset:block_end]
        else:
            draft_v_block = draft_v[block_offset:block_end]
        k_ctx = torch.cat([k_ctx, draft_k[block_offset:block_end]], dim=0)
        v_ctx = torch.cat([v_ctx, draft_v_block], dim=0)
        out[block_offset:block_end] = _dspark_attention_reference(
            q[block_offset:block_end],
            k_ctx,
            v_ctx,
            attn_sink,
            softmax_scale,
        )

    return out


__all__ = ["dspark_attention"]
