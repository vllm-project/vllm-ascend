# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L routing for DeepSeek: form the batch pool, then emit logical IDs and weights."""

import math

import torch
from vllm.triton_utils import tl, triton

# This model-specific operator uses DeepSeek's six routed experts per token.
TOP_K = 6

UNRESTRICTED = 0
CONTRIBUTOR = 1
RESTRICTED = 2


@triton.jit
def _build_pool(
    logits_ptr,
    roles_ptr,
    bias_ptr,
    raw_scores_ptr,
    votes_ptr,
    ROW_STRIDE: tl.constexpr,
    EXPERT_STRIDE: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
    SUFFIX_EXPERTS: tl.constexpr,
    CONTRIBUTOR_ROLE: tl.constexpr,
    RESTRICTED_ROLE: tl.constexpr,
    K: tl.constexpr,
):
    with tl.extra.cann.extension.scope(core_mode="vector"):
        row = tl.program_id(0)
        expert = tl.arange(0, BLOCK)
        valid = expert < EXPERTS
        x = tl.load(logits_ptr + row * ROW_STRIDE + expert * EXPERT_STRIDE, valid, 0).to(tl.float32)
        # Above 20 softplus rounds to x in FP32. Bound exp even in the
        # unselected branch, while preserving small-score rounding and ties.
        SOFTPLUS_THRESHOLD: tl.constexpr = 20.0
        exp_input = tl.where(x > SOFTPLUS_THRESHOLD, SOFTPLUS_THRESHOLD, x)
        softplus = tl.where(x > SOFTPLUS_THRESHOLD, x, tl.log(tl.exp(exp_input) + 1.0))
        raw = tl.sqrt(softplus)
        tl.store(raw_scores_ptr + row * EXPERTS + expert, raw, valid)
        role = tl.load(roles_ptr + row)
        limit = tl.where(role == CONTRIBUTOR_ROLE, K, tl.where(role == RESTRICTED_ROLE, SUFFIX_EXPERTS, 0))
        if limit > 0:
            score = raw + tl.load(bias_ptr + expert, valid, 0.0)
            remaining = valid & (tl.abs(score) < float("inf"))
            selected = tl.full((BLOCK,), False, tl.int1)
            for i in tl.static_range(K):
                best = tl.max(tl.where(remaining, score, -float("inf")), 0)
                picked = tl.min(tl.where(remaining & (score == best), expert, BLOCK), 0)
                chosen = remaining & (expert == picked)
                selected = selected | (chosen & (i < limit))
                remaining = remaining & ~chosen
            tl.atomic_add(votes_ptr + expert, 1, selected, sem="relaxed")


@triton.jit
def _finish_route(
    raw_scores_ptr,
    roles_ptr,
    bias_ptr,
    votes_ptr,
    weights_ptr,
    ids_ptr,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
    RESTRICTED_ROLE: tl.constexpr,
    K: tl.constexpr,
    SCALING: tl.constexpr,
):
    with tl.extra.cann.extension.scope(core_mode="vector"):
        row = tl.program_id(0)
        expert = tl.arange(0, BLOCK)
        valid = expert < EXPERTS
        role = tl.load(roles_ptr + row)
        allowed = tl.load(votes_ptr + expert, valid, 0) > 0
        raw = tl.load(raw_scores_ptr + row * EXPERTS + expert, valid, 0.0)
        score = raw + tl.load(bias_ptr + expert, valid, 0.0)
        remaining = valid & (tl.abs(score) < float("inf")) & ((role != RESTRICTED_ROLE) | allowed)
        # Triton vectors need power-of-two extents; only the first K lanes are stored.
        OUTPUT_BLOCK: tl.constexpr = triton.next_power_of_2(K)
        out = tl.arange(0, OUTPUT_BLOCK)
        ids = tl.full((OUTPUT_BLOCK,), 0, tl.int32)
        for i in tl.static_range(K):
            best = tl.max(tl.where(remaining, score, -float("inf")), 0)
            picked = tl.min(tl.where(remaining & (score == best), expert, BLOCK), 0)
            ids = tl.where(out == i, picked, ids)
            remaining = remaining & (expert != picked)
        # Check all output slots once, outside the unrolled selection loop.
        selected = (out < K) & (ids < EXPERTS)
        safe_ids = tl.where(selected, ids, 0)
        weights = tl.where(selected, tl.gather(raw, safe_ids, axis=0), 0.0)
        # Preserve the reference denominator floor when selected raw scores sum to zero.
        NORMALIZATION_EPS: tl.constexpr = 1e-20
        denom = tl.maximum(tl.sum(weights, 0), NORMALIZATION_EPS)
        tl.store(ids_ptr + row * K + out, safe_ids, out < K)
        tl.store(weights_ptr + row * K + out, weights / denom * SCALING, out < K)


def route_experts(
    logits: torch.Tensor,
    roles: torch.Tensor,
    bias: torch.Tensor,
    raw: torch.Tensor,
    votes: torch.Tensor,
    suffix_experts: int,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route FP32 logits through a shared pool, overwriting only ``raw`` and ``votes``.

    All tensors must share a device. Logits may be strided; roles, bias and
    scratch are contiguous. The caller supplies nonaliasing scratch and roles
    whose votes give every restricted row at least TOP_K candidate experts.
    Metadata is checked without reading device contents or synchronizing.
    """
    if logits.layout != torch.strided or logits.ndim != 2 or logits.dtype != torch.float32:
        raise ValueError("logits must be a strided 2D FP32 tensor")
    rows, experts = logits.shape
    if experts < TOP_K:
        raise ValueError(f"expert-pool routing requires at least {TOP_K} experts")
    for name, tensor, shape, dtype in (
        ("roles", roles, (rows,), torch.int32),
        ("bias", bias, (experts,), torch.float32),
        ("raw", raw, (rows, experts), torch.float32),
        ("votes", votes, (experts,), torch.int32),
    ):
        if (
            tensor.layout != torch.strided
            or tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != logits.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous {dtype} with shape {shape} on the logits device")
    if type(suffix_experts) is not int or not 0 <= suffix_experts <= TOP_K:
        raise ValueError(f"suffix_experts must be an integer in [0, {TOP_K}]")
    if type(scaling) not in (int, float) or not math.isfinite(scaling):
        raise ValueError("scaling must be a finite number")
    weights = torch.empty((rows, TOP_K), dtype=torch.float32, device=logits.device)
    ids = torch.empty((rows, TOP_K), dtype=torch.int32, device=logits.device)
    if rows == 0:
        return weights, ids
    block = triton.next_power_of_2(experts)
    votes.zero_()
    _build_pool[(rows,)](
        logits,
        roles,
        bias,
        raw,
        votes,
        logits.stride(0),
        logits.stride(1),
        experts,
        block,
        suffix_experts,
        CONTRIBUTOR,
        RESTRICTED,
        TOP_K,
        multibuffer=False,
    )
    _finish_route[(rows,)](
        raw,
        roles,
        bias,
        votes,
        weights,
        ids,
        experts,
        block,
        RESTRICTED,
        TOP_K,
        scaling,
        multibuffer=False,
    )
    return weights, ids
