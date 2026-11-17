# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.moe_token_cut_native import (
    MOE_TOKEN_CUT_SCHEMA,
    NATIVE_TOKEN_CUT_CAPACITY,
    invoke_native_op,
)

MAX_MOE_TOKEN_CUT_BATCH_SIZE = 64


@dataclass(frozen=True, slots=True)
class MoETokenCutConfig:
    """Device-side token-cut policy parameters.

    This object intentionally does not decide *when* truncation runs. The
    runner owns that lifecycle because it must update request, KV-cache,
    speculative-decoding, and attention state together.
    """

    level_limit: int = 9
    local_cut_count: int = 0
    cut_count_threshold: int = 4
    small_expert_token_threshold: int = 4
    send_aiv_num: int = 8
    recv_aiv_num: int = 8


@dataclass(frozen=True, slots=True)
class MoETokenRestoreInput:
    """Scheduled cache subset. The live prefix is followed by reserved rows."""

    fresh_rows: int
    hidden: torch.Tensor
    residual: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    request_ids: torch.Tensor


@dataclass(frozen=True, slots=True)
class MoETokenCutInputs:
    hidden: torch.Tensor
    residual: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


@dataclass(frozen=True, slots=True)
class MoETokenCutOutput:
    """Typed bridge from ``moe_token_cut`` to the MC2 dispatcher."""

    dispatch_input: MoETokenDispatchInput
    cut_idx: torch.Tensor
    actual_bs: torch.Tensor
    prefix_index: torch.Tensor
    no_cut_rank_num: torch.Tensor
    # Input scratch after RestoreDiscarded, before compaction. Independent of
    # dispatch output buffers; rebatch may read these on a separate stream.
    inputs: MoETokenCutInputs | None = None


MoeTokenCutOp = Callable[..., None]


def apply_moe_token_cut(
    token_dispatch_input: MoETokenDispatchInput,
    *,
    group_ep: str,
    rank_id: int,
    rank_num: int,
    expert_num: int,
    expert_per_rank: int,
    config: MoETokenCutConfig,
    layer_idx: int = 0,
    residuals: torch.Tensor | None = None,
    allow_cut: bool = True,
    restore: MoETokenRestoreInput | None = None,
    enable_cut: bool = True,
    op: MoeTokenCutOp | None = None,
) -> MoETokenCutOutput:
    """Call the supplied 38-argument, in-place BF16/FP16 kernel in eager mode.

    Outputs retain the original batch shape. ``routing.mc2_mask`` marks the
    compacted valid prefix, allowing MC2 dispatch/combine to stay graph-safe.
    Cross-layer persistence is deliberately outside this function and must be
    handled by the model runner before this adapter is enabled in production.
    """

    hidden_states = token_dispatch_input.hidden_states
    topk_ids = token_dispatch_input.topk_ids
    topk_weights = token_dispatch_input.topk_weights
    batch_size = hidden_states.shape[0]

    if hidden_states.ndim != 2 or topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("moe_token_cut expects 2D hidden_states, topk_ids, and topk_weights")
    if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != batch_size:
        raise ValueError("moe_token_cut inputs must share the same batch and top-k dimensions")
    if batch_size > MAX_MOE_TOKEN_CUT_BATCH_SIZE:
        raise ValueError(f"moe_token_cut supports at most {MAX_MOE_TOKEN_CUT_BATCH_SIZE} tokens, got {batch_size}")
    if expert_num != expert_per_rank * rank_num:
        raise ValueError("expert_num must equal expert_per_rank * rank_num")
    if token_dispatch_input.routing.pertoken_scale is not None:
        raise ValueError("moe_token_cut does not yet compact per-token quantization scales")

    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("native moe_token_cut requires BF16/FP16 hidden states")
    if residuals is not None and (residuals.shape != hidden_states.shape or residuals.dtype != hidden_states.dtype):
        raise ValueError("native moe_token_cut residuals must match hidden states")
    if not group_ep or layer_idx < 0 or not 0 <= rank_id < rank_num:
        raise ValueError("native moe_token_cut requires group_ep and valid rank/layer indices")
    previous_mask = token_dispatch_input.routing.mc2_mask
    if previous_mask is None:
        input_valid_mask = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=hidden_states.device,
        )
    else:
        if previous_mask.numel() < batch_size:
            raise ValueError("moe_token_cut mc2_mask must contain at least one value per input row")
        input_valid_mask = (
            previous_mask[:batch_size]
            .to(
                device=hidden_states.device,
                dtype=torch.bool,
            )
            .contiguous()
        )
    if restore is not None:
        count = restore.hidden.shape[0]
        total = restore.fresh_rows + count
        if count <= 0 or restore.fresh_rows < 0 or total > batch_size:
            raise ValueError("native restore exceeds reserved dispatch rows")
        for value, expected in (
            (restore.hidden, (count, hidden_states.shape[1])),
            (restore.residual, (count, hidden_states.shape[1])),
            (restore.topk_ids, (count, topk_ids.shape[1])),
            (restore.topk_weights, (count, topk_ids.shape[1])),
            (restore.request_ids, (count,)),
        ):
            if value.shape != expected or value.device != hidden_states.device:
                raise ValueError("native restore cache shape/device mismatch")
        if (
            restore.hidden.dtype != hidden_states.dtype
            or restore.residual.dtype != hidden_states.dtype
            or restore.topk_weights.dtype != hidden_states.dtype
            or restore.topk_ids.dtype != torch.int32
            or restore.request_ids.dtype != torch.int32
        ):
            raise ValueError("native restore cache dtype mismatch")
        # The model reserved this exact prefix before MLP. Padding stays last.
        input_valid_mask = torch.arange(batch_size, device=hidden_states.device) < total

    # The kernel has no input mask. Pack valid input rows into a prefix, without
    # a device-to-host count read; undo this order in cut_idx/prefix below.
    order = torch.argsort(input_valid_mask.to(torch.int32), descending=True, stable=True)
    capacity = NATIVE_TOKEN_CUT_CAPACITY
    device = hidden_states.device

    def padded(source, *, dtype=None):
        target = torch.zeros((capacity, *source.shape[1:]), dtype=dtype or source.dtype, device=device)
        target[:batch_size].copy_(source.index_select(0, order))
        return target

    hidden = padded(hidden_states)
    weights = padded(topk_weights, dtype=hidden_states.dtype)
    native_residuals = padded(residuals if residuals is not None else torch.zeros_like(hidden_states))
    cut_idx = torch.zeros(capacity, dtype=torch.int8, device=device)
    prefix_index = torch.zeros(capacity, dtype=torch.int32, device=device)
    active_mask = torch.zeros(capacity, dtype=torch.bool, device=device)
    new_topk = torch.zeros((capacity, topk_ids.shape[1]), dtype=torch.int32, device=device)
    new_topk_weight = torch.zeros_like(weights)
    new_hidden = torch.zeros_like(hidden)
    actual_bs = torch.zeros(1, dtype=torch.int32, device=device)
    # This field is an INPUT restore gate in the supplied kernel, not a rank
    # count output. Only the scheduled native-cache subset enables restoration;
    # Python checkpoints and the legacy separate path leave it disabled here.
    no_cut_rank_num = torch.tensor([int(restore is not None)], dtype=torch.int32, device=device)
    runtime_bs = (
        input_valid_mask.sum(dtype=torch.int32).reshape(1)
        if restore is None
        else torch.tensor([restore.fresh_rows], dtype=torch.int32, device=device)
    )

    def cache(value, example):
        target = torch.zeros_like(example)
        if value is not None:
            target[: value.shape[0]].copy_(value)
        return target

    values = dict(
        topk_ids=padded(topk_ids, dtype=torch.int32),
        topk_weights=weights,
        hidden=hidden,
        cutidx=cut_idx,
        new_topk=new_topk,
        new_topkweight=new_topk_weight,
        new_hidden=new_hidden,
        new_residuals=torch.zeros_like(hidden),
        actual_bs=actual_bs,
        prefix_index=prefix_index,
        no_cut_rank_num=no_cut_rank_num,
        active_mask=active_mask,
        local_cut_count=torch.full((1,), config.local_cut_count, dtype=torch.int32, device=device),
        batch_size=runtime_bs,
        # All EP ranks enter communication together. A locally protected rank
        # uses a no-delete threshold, NOT enable_cut=0 (which skips HCCL).
        enable_cut=torch.tensor([int(enable_cut)], dtype=torch.int32, device=device),
        residuals=native_residuals,
        discard_label=torch.zeros(capacity, dtype=torch.int32, device=device),
        discard_tokens=cache(None if restore is None else restore.hidden, hidden),
        discard_moe_req_id=cache(
            None if restore is None else restore.request_ids, torch.zeros(capacity, dtype=torch.int32, device=device)
        ),
        req_id_list=torch.arange(capacity, dtype=torch.int32, device=device),
        discard_residuals=cache(None if restore is None else restore.residual, hidden),
        discard_topk=cache(None if restore is None else restore.topk_ids, new_topk),
        discard_topk_weight=cache(None if restore is None else restore.topk_weights, weights),
        moe_cnt=torch.tensor([0 if restore is None else restore.hidden.shape[0]], dtype=torch.int32, device=device),
        history_bs=torch.zeros(1, dtype=torch.int32, device=device),
        original_bs=runtime_bs.clone(),
        group_ep=group_ep,
        topk=topk_ids.shape[1],
        rank_id=rank_id,
        layer_idx=layer_idx,
        rank_num=rank_num,
        expert_num=expert_num,
        expert_per_rank=expert_per_rank,
        level_limit=config.level_limit,
        cut_count_threshold=config.cut_count_threshold if allow_cut else capacity,
        small_expert_token_threshold=config.small_expert_token_threshold,
        send_aiv_num=config.send_aiv_num,
        recv_aiv_num=config.recv_aiv_num,
    )
    invoke_native_op(MOE_TOKEN_CUT_SCHEMA, values, op=op)
    source_cut_idx = torch.zeros_like(cut_idx)
    source_prefix = torch.zeros_like(prefix_index)
    source_cut_idx[:batch_size].index_copy_(0, order, cut_idx[:batch_size])
    source_prefix[:batch_size].index_copy_(0, order, prefix_index[:batch_size])

    valid_mask = torch.arange(batch_size, device=hidden_states.device, dtype=torch.int32) < actual_bs.reshape(1)

    routing = replace(token_dispatch_input.routing, mc2_mask=valid_mask)
    compacted_input = replace(
        token_dispatch_input,
        hidden_states=new_hidden[:batch_size],
        topk_ids=new_topk[:batch_size],
        topk_weights=new_topk_weight[:batch_size].to(dtype=topk_weights.dtype),
        routing=routing,
    )
    return MoETokenCutOutput(
        dispatch_input=compacted_input,
        cut_idx=source_cut_idx[:MAX_MOE_TOKEN_CUT_BATCH_SIZE],
        actual_bs=actual_bs,
        prefix_index=source_prefix[:MAX_MOE_TOKEN_CUT_BATCH_SIZE],
        no_cut_rank_num=no_cut_rank_num,
        inputs=MoETokenCutInputs(
            hidden[:batch_size], native_residuals[:batch_size], values["topk_ids"][:batch_size], weights[:batch_size]
        ),
    )


def restore_compacted_rows(
    compacted_rows: torch.Tensor,
    cut_idx: torch.Tensor,
    prefix_index: torch.Tensor,
) -> torch.Tensor:
    """Restore fixed-shape MC2 output to its pre-cut row order.

    Deferred and padding rows become zero. The operation uses only fixed-size
    gather/where primitives, avoiding host reads and dynamic nonzero indices.
    """
    if compacted_rows.ndim < 1:
        raise ValueError("compacted rows must have a batch dimension")
    row_count = compacted_rows.shape[0]
    if cut_idx.ndim != 1 or cut_idx.shape[0] < row_count:
        raise ValueError("cut_idx must contain one value per compacted row")
    if prefix_index.ndim != 1 or prefix_index.shape[0] < row_count:
        raise ValueError("prefix_index must contain one value per compacted row")
    if row_count == 0:
        return compacted_rows.clone()

    source_indices = prefix_index[:row_count].to(dtype=torch.long).sub(1)
    source_indices = source_indices.clamp(min=0, max=row_count - 1)
    gathered = compacted_rows.index_select(0, source_indices)
    kept_mask = cut_idx[:row_count].ne(0)
    kept_mask = kept_mask.reshape(
        row_count,
        *([1] * (compacted_rows.ndim - 1)),
    )
    return torch.where(kept_mask, gathered, torch.zeros_like(gathered))
