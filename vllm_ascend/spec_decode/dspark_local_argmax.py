# SPDX-License-Identifier: Apache-2.0
"""Greedy DSpark decoding over vocabulary-sharded logits."""

from collections.abc import Callable

import torch

# Candidate token IDs are packed alongside scores into an FP32 collective.
MAX_EXACT_FP32_VOCAB_SIZE = 1 << 24


def sample_local_draft_tokens(
    local_logits: torch.Tensor,
    seed_token_ids: torch.Tensor,
    markov_bias: Callable[[torch.Tensor], torch.Tensor],
    vocab_start: int,
    vocab_size: int,
    world_size: int,
    all_gather: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply Markov correction before reducing each draft position's argmax.

    ``local_logits`` is an owned [requests, draft_positions, local_vocab]
    tensor and is updated in place, matching the full-logit DSpark path.
    Ranks must hold contiguous vocabulary shards in ascending rank order.
    The bias callback deliberately retains the full-vocabulary projection to
    preserve its original rounding behavior; only the LMHead gather changes.
    """
    if vocab_size > MAX_EXACT_FP32_VOCAB_SIZE:
        raise ValueError("Vocabulary exceeds exact FP32 candidate ID range")
    num_requests, num_positions, shard_size = local_logits.shape
    valid_size = max(0, min(shard_size, vocab_size - vocab_start))
    result = torch.empty((num_requests, num_positions), dtype=torch.int64, device=local_logits.device)
    previous = seed_token_ids
    for position in range(num_positions):
        bias = markov_bias(previous)
        scores = local_logits[:, position]
        scores[:, :valid_size].add_(bias[:, vocab_start : vocab_start + valid_size])
        scores[:, valid_size:] = -float("inf")
        values, indices = scores.max(dim=-1)
        indices = indices + vocab_start
        if world_size > 1:
            pairs = torch.stack((values.float(), indices.float()), dim=-1)
            candidates = all_gather(pairs).view(num_requests, world_size, 2)
            # Contiguous shards and first-max semantics preserve global argmax
            # tie breaking, including ties that cross shard boundaries.
            winner = candidates[:, :, 0].argmax(dim=-1, keepdim=True)
            indices = candidates[:, :, 1].gather(-1, winner).squeeze(-1).to(torch.int64)
        result[:, position].copy_(indices)
        previous = result[:, position]
    return result
