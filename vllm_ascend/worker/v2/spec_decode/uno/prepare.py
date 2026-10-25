# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/uno_prepare.py
# and its eager oracle
# https://github.com/vllm-project/vllm/blob/main/tests/v1/spec_decode/uno_reference.py
"""Eager preparation of the shared-model UNO draft inputs.

Upstream fuses this into a single Triton launch. Ascend keeps the same
semantics in eager Torch for the first stage: the fused kernel relies on
62-bit int64 arithmetic and ``do_not_specialize``, which is not yet validated
on triton-ascend. The eager path is the oracle upstream's own tests compare
against, so it is the correctness anchor here as well.

Three invariants are load-bearing and must not be "optimized" away:

1. ``positions`` are clamped for the model input but the *unclamped* value
   decides KV ownership (``>= max_model_len`` -> ``PAD_SLOT_ID``). The draft
   rows past the context must not write into anyone else's cache block.
2. ``sample_idx_mapping`` is int64. An int32 request slot times a per-request
   state stride overflows past 2**31 rows.
3. The noise hash mixes the flat batch-row offset, so it is deterministic only
   for a fixed request row order.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

# Released UNO noise-hash constants (64-bit mixing, masked to 62 bits).
_NOISE_SEED_MULT = 0x1E3779B185EBCA8
_NOISE_STEP_MULT = 0x11B54A32D192ED0
_NOISE_MIX_1 = 0x3F58476D1CE4E5B9
_NOISE_MIX_2 = 0x14D049BB133111EB
_NOISE_MASK = (1 << 62) - 1

# Inclusive lower bound of the noise token range; the upper bound is the target
# vocabulary size (or ``uno_mask_token_id`` when set).
_NOISE_LOW = 1


def _mix(x: torch.Tensor) -> torch.Tensor:
    x = ((x ^ (x >> 30)) * _NOISE_MIX_1) & _NOISE_MASK
    x = ((x ^ (x >> 27)) * _NOISE_MIX_2) & _NOISE_MASK
    return x ^ (x >> 31)


def fill_uno_noise(
    input_ids: torch.Tensor,
    is_noise: torch.Tensor,
    req_seeds: torch.Tensor,
    step: int,
    low: int,
    high: int,
) -> None:
    """Fill noise rows in place with ids in ``[low, high)``.

    Non-noise rows keep their current value, so the seed row of every request
    keeps the verified bonus token.
    """
    if low >= high:
        raise ValueError("UNO noise requires a nonempty token range")
    slot = torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.int64)
    step_term = (int(step) * _NOISE_STEP_MULT) & _NOISE_MASK
    seeds = (req_seeds.to(torch.int64) * _NOISE_SEED_MULT) & _NOISE_MASK
    h = _mix((seeds + step_term + slot) & _NOISE_MASK)
    noise = (h % (high - low)) + low
    input_ids.copy_(torch.where(is_noise, noise.to(input_ids.dtype), input_ids))


def prepare_uno_inputs(
    buffers: Any,
    slot_mapping: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    input_batch: Any,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor | None,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    seeds: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    k: int,
    max_model_len: int,
    noise_seed: int,
    noise_high: int,
    step: int,
) -> None:
    """Prepare the draft rows: seed token, noise, positions, KV slots.

    Each request contributes ``k`` rows. Row 0 of a request is the verified
    bonus token evaluated against the *base* weights; rows 1..k-1 are random
    noise tokens evaluated through UNO's gate adapter.

    Args:
        buffers: Draft input buffers (``input_ids``, ``positions``,
            ``seq_lens``, ``query_start_loc``).
        slot_mapping: Flat KV slot buffer for the draft rows.
        sample_idx_mapping: Draft row -> request-state slot, or -1 for padding.
        input_batch: The target batch this draft extends.
        num_sampled: Per-request number of tokens the target sampled.
        num_rejected: Per-request number of rejected draft tokens, or None.
        last_sampled: ``[max_num_reqs, 1]`` last sampled token per request.
        next_prefill_tokens: ``[num_prefill_lookahead, max_num_reqs]`` tokens
            whose bonus row is used when the target sampled nothing.
        seeds: Per-request sampler seeds.
        block_table: Batch-local KV block table, one row per request.
        block_size: KV block size in tokens.
        k: Draft rows per request.
        max_model_len: Upper bound for positions and sequence lengths.
        noise_seed: User seed mixed into the noise hash.
        noise_high: Exclusive upper bound of the noise token range.
        step: Draft step counter, mixed into the noise hash.
    """
    if k < 1 or block_size < 1 or max_model_len < 1:
        raise ValueError("UNO preparation requires positive K, block size and max length")
    if noise_high <= _NOISE_LOW:
        raise ValueError("UNO noise range must include at least one token")
    if sample_idx_mapping.dtype != torch.int64:
        raise ValueError("sample_idx_mapping must be int64")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a nonnegative Python integer")

    n = int(input_batch.num_reqs)
    if n < 1:
        raise ValueError("UNO preparation requires at least one request")
    count = n * k
    if count > buffers.input_ids.numel() or count > buffers.positions.numel():
        raise ValueError("draft input buffer is shorter than num_reqs * k")
    if count > slot_mapping.numel() or count > sample_idx_mapping.numel():
        raise ValueError("draft slot/sample buffer is shorter than num_reqs * k")
    if block_table.ndim != 2 or block_table.shape[0] < n:
        raise ValueError("block_table must have one batch-local row per request")

    state_idx = input_batch.idx_mapping.long()
    valid_end = input_batch.query_start_loc[1 : n + 1].long()
    if num_rejected is not None:
        valid_end = valid_end - num_rejected.long()
    first_pos = input_batch.positions[valid_end - 1] + 1

    offsets = torch.arange(k, dtype=torch.int64, device=buffers.device)
    positions = first_pos[:, None] + offsets

    # Row 0 replays the last verified token; rows 1..k-1 are noise. When the
    # target sampled nothing this step, the bonus row comes from the prefill
    # lookahead instead of the sampler.
    seed_tokens = torch.where(
        num_sampled > 0,
        last_sampled.reshape(-1)[state_idx],
        next_prefill_tokens.reshape(-1, seeds.shape[0])[0, state_idx],
    )
    buffers.input_ids[:count].copy_(seed_tokens.repeat_interleave(k))
    is_noise = (offsets != 0).repeat(n)
    req_seeds = (seeds[state_idx] + noise_seed).repeat_interleave(k)
    fill_uno_noise(
        buffers.input_ids[:count],
        is_noise,
        req_seeds,
        step,
        _NOISE_LOW,
        noise_high,
    )

    # Invariant 1: clamp for the model, keep the unclamped value for KV.
    buffers.positions[:count].copy_(positions.clamp(max=max_model_len - 1).flatten())
    buffers.seq_lens[:n].copy_((first_pos + k).clamp(max=max_model_len))
    buffers.seq_lens[n:].zero_()
    torch.arange(n + 1, out=buffers.query_start_loc[: n + 1])
    buffers.query_start_loc[: n + 1].mul_(k)
    buffers.query_start_loc[n + 1 :].fill_(count)
    buffers.input_ids[count:].zero_()
    buffers.positions[count:].zero_()

    # Invariant 2: int64 row -> state-slot mapping; padding stays -1.
    sample_idx_mapping[:count].copy_(state_idx.repeat_interleave(k))
    sample_idx_mapping[count:].fill_(-1)

    # Invariant 1 again: ownership uses the unclamped position. Block 0 is the
    # null block, so non-positive ids are treated as nonresident as well.
    block_numbers = positions // block_size
    block_ids = block_table[:n].gather(1, block_numbers.clamp(max=block_table.shape[1] - 1)).long()
    resident = (
        (positions >= 0)
        & (positions < max_model_len)
        & (block_numbers >= 0)
        & (block_numbers < block_table.shape[1])
        & (block_ids > 0)
    )
    slots = block_ids * block_size + positions % block_size
    slot_mapping[:count].copy_(torch.where(resident, slots, PAD_SLOT_ID).flatten())
    slot_mapping[count:].fill_(PAD_SLOT_ID)


__all__ = [
    "fill_uno_noise",
    "prepare_uno_inputs",
]
