# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
# Adapted from vLLM's sample/gumbel.py and spec_decode/dflash2/speculator.py.

import torch
from vllm.triton_utils import tl, triton

_DRAFT_NOISE_SALT = tl.constexpr(1 << 30)


@triton.jit
def _rotl32(value, shift: tl.constexpr):
    return (value << shift) | (value >> (32 - shift))


@triton.jit
def _mix32(state, key):
    key *= 0xCC9E2D51
    key = _rotl32(key, 15)
    key *= 0x1B873593
    state ^= key
    return _rotl32(state, 13) * 5 + 0xE6546B64


@triton.jit
def _draft_uniform32(seed, step, position, token):
    # Match upstream spec_step_seed(seed, step), followed by Murmur3.
    # Split the seed before mixing: every multiply is uint32, avoiding the
    # int64 Philox multiply helper that triton-ascend cannot lower.
    seed = seed.to(tl.uint64)
    state = tl.full((), 0, tl.uint32)
    state = _mix32(state, seed.to(tl.uint32))
    state = _mix32(state, (seed >> 32).to(tl.uint32) ^ step.to(tl.uint32))
    state = _mix32(state, (position + _DRAFT_NOISE_SALT).to(tl.uint32))
    state = _mix32(state, token.to(tl.uint32))
    state ^= 16
    state ^= state >> 16
    state *= 0x85EBCA6B
    state ^= state >> 13
    state *= 0xC2B2AE35
    state ^= state >> 16
    hi = (state >> 16).to(tl.int32).to(tl.float32)
    lo = (state & 0xFFFF).to(tl.int32).to(tl.float32)
    return hi * 1.52587890625e-05 + (lo + 0.5) * 2.3283064365386963e-10


@triton.jit
def _log1p_neg_stable(value):
    # Upstream's degree-8 series preserves the winning Gumbel tail without
    # CUDA libdevice.log1p. Keep the same fp32 evaluation order.
    polynomial = 1.0 / 8.0
    polynomial = 1.0 / 7.0 + value * polynomial
    polynomial = 1.0 / 6.0 + value * polynomial
    polynomial = 1.0 / 5.0 + value * polynomial
    polynomial = 1.0 / 4.0 + value * polynomial
    polynomial = 1.0 / 3.0 + value * polynomial
    polynomial = 1.0 / 2.0 + value * polynomial
    polynomial = 1.0 + value * polynomial
    return tl.where(
        value < 0.25,
        -value * polynomial,
        tl.log(tl.maximum(1.0 - value, 5.960464477539063e-08)),
    )


@triton.jit
def _candidate_walk_kernel(
    scores_ptr,
    candidates_ptr,
    positions_ptr,
    mapping_ptr,
    temperatures_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_ptr,
    NUM_STEPS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PROBABILISTIC: tl.constexpr,
):
    request = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    state = tl.load(mapping_ptr + request * NUM_STEPS)
    valid = state >= 0
    temperature = tl.load(temperatures_ptr + state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + state, mask=valid, other=0)
    previous = 0
    for step in range(NUM_STEPS):
        flat = request * NUM_STEPS + step
        scores = tl.load(
            scores_ptr + (flat * TOP_K + previous) * TOP_K + offsets,
            mask=valid & (offsets < TOP_K),
            other=-float("inf"),
        ).to(tl.float32)
        candidates = tl.load(
            candidates_ptr + flat * TOP_K + offsets,
            mask=valid & (offsets < TOP_K),
            other=0,
        )
        sampled = scores
        if PROBABILISTIC and temperature != 0.0:
            position = tl.load(positions_ptr + flat) - 1
            uniform = _draft_uniform32(seed, step, position, candidates)
            noise = -tl.log(-_log1p_neg_stable(uniform))
            sampled = tl.where(offsets < TOP_K, scores / temperature + noise, -float("inf"))
        # Separate max and min avoids NPU max-with-index lowering issues and
        # retains upstream's first-index tie break.
        maximum = tl.max(sampled, axis=0)
        previous = tl.min(tl.where((offsets < TOP_K) & (sampled == maximum), offsets, BLOCK_K), axis=0)
        tl.store(realized_ptr + flat * TOP_K + offsets, scores, mask=valid & (offsets < TOP_K))
        token = tl.load(candidates_ptr + flat * TOP_K + previous, mask=valid, other=0)
        tl.store(tokens_ptr + flat, token, mask=valid)


@triton.jit
def _cache_candidates_kernel(
    logits_ptr,
    cached_ids_ptr,
    candidates_ptr,
    realized_ptr,
    mapping_ptr,
    stride_request,
    stride_step,
    NUM_STEPS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CLEAR_ONLY: tl.constexpr,
):
    flat = tl.program_id(0)
    state = tl.load(mapping_ptr + flat).to(tl.int64)
    step = flat % NUM_STEPS
    offsets = tl.arange(0, BLOCK_K)
    valid = (state >= 0) & (offsets < TOP_K)
    cache_base = (state * NUM_STEPS + step) * TOP_K
    logits_base = logits_ptr + state * stride_request + step * stride_step
    if CLEAR_ONLY:
        old_ids = tl.load(cached_ids_ptr + cache_base + offsets, mask=valid, other=0)
        tl.store(logits_base + old_ids, -float("inf"), mask=valid)
    else:
        ids = tl.load(candidates_ptr + flat * TOP_K + offsets, mask=valid, other=0)
        scores = tl.load(realized_ptr + flat * TOP_K + offsets, mask=valid, other=0)
        tl.store(logits_base + ids, scores, mask=valid)
        tl.store(cached_ids_ptr + cache_base + offsets, ids, mask=valid)


def sample_candidates(
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    num_reqs: int,
    sample_pos: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    draft_tokens: torch.Tensor,
    realized_scores: torch.Tensor,
    cached_candidate_ids: torch.Tensor,
    draft_logits: torch.Tensor | None,
) -> None:
    num_steps, top_k = candidate_ids.shape[1:]
    candidates = candidate_ids.contiguous()
    mapping = idx_mapping.contiguous()
    block_k = triton.next_power_of_2(top_k)
    _candidate_walk_kernel[(num_reqs,)](
        scores.contiguous(),
        candidates,
        sample_pos.contiguous(),
        mapping,
        temperature,
        seeds,
        draft_tokens,
        realized_scores,
        NUM_STEPS=num_steps,
        TOP_K=top_k,
        BLOCK_K=block_k,
        PROBABILISTIC=draft_logits is not None,
        num_warps=1,
    )
    if draft_logits is not None:
        # Separate launches order the scatters even when old/new IDs overlap;
        # no CUDA barrier primitive or cross-lane ordering is assumed on NPU.
        for clear_only in (True, False):
            _cache_candidates_kernel[(num_reqs * num_steps,)](
                draft_logits,
                cached_candidate_ids,
                candidates,
                realized_scores,
                mapping,
                draft_logits.stride(0),
                draft_logits.stride(1),
                NUM_STEPS=num_steps,
                TOP_K=top_k,
                BLOCK_K=block_k,
                CLEAR_ONLY=clear_only,
                num_warps=1,
            )
