# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L routing for DeepSeek: form the batch pool, then emit logical IDs and weights."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.worker.v2.dspark_expert_pool import CONTRIBUTOR, RESTRICTED, TOP_K


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
        # Preserve the frozen L score rounding, including its effect on ties.
        raw = tl.sqrt(tl.log(tl.exp(x) + 1.0))
        tl.store(raw_scores_ptr + row * EXPERTS + expert, raw, valid)
        role = tl.load(roles_ptr + row)
        limit = tl.where(role == CONTRIBUTOR_ROLE, K, tl.where(role == RESTRICTED_ROLE, SUFFIX_EXPERTS, 0))
        if limit > 0:
            score = raw + tl.load(bias_ptr + expert, valid, 0.0)
            remaining = valid
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
        remaining = valid & ((role != RESTRICTED_ROLE) | allowed)
        # Triton vectors need power-of-two extents; only the first K lanes are stored.
        OUTPUT_BLOCK: tl.constexpr = triton.next_power_of_2(K)
        out = tl.arange(0, OUTPUT_BLOCK)
        ids = tl.full((OUTPUT_BLOCK,), 0, tl.int32)
        for i in tl.static_range(K):
            best = tl.max(tl.where(remaining, score, -float("inf")), 0)
            picked = tl.min(tl.where(remaining & (score == best), expert, BLOCK), 0)
            ids = tl.where(out == i, picked, ids)
            remaining = remaining & (expert != picked)
        weights = tl.where(out < K, tl.gather(raw, ids, axis=0), 0.0)
        # Preserve the reference denominator floor when selected raw scores sum to zero.
        NORMALIZATION_EPS: tl.constexpr = 1e-20
        denom = tl.maximum(tl.sum(weights, 0), NORMALIZATION_EPS)
        tl.store(ids_ptr + row * K + out, ids, out < K)
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
    rows, experts = logits.shape
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
    weights = torch.empty((rows, TOP_K), dtype=torch.float32, device=logits.device)
    ids = torch.empty((rows, TOP_K), dtype=torch.int32, device=logits.device)
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
