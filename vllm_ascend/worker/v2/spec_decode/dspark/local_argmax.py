# SPDX-License-Identifier: Apache-2.0
"""Local-vocabulary argmax reduction for MRV2 DSpark."""

from collections.abc import Callable

import torch

# Token IDs are packed with scores into FP32 for one small collective.
MAX_EXACT_FP32_TOKEN_ID = 1 << 24


def reduce_local_argmax(
    local_scores: torch.Tensor,
    *,
    vocab_start: int,
    vocab_end: int,
    vocab_size: int,
    tp_size: int,
    all_gather: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Select the global argmax from contiguous TP vocabulary shards.

    ``local_scores`` already includes the DSpark Markov correction. Padding
    entries are masked before selecting this rank's candidate. The all-gather
    exchanges only one FP32 ``(score, token_id)`` pair per request.
    """
    if vocab_size > MAX_EXACT_FP32_TOKEN_ID:
        raise ValueError(
            "DSpark local argmax packs token IDs into FP32, but vocabulary "
            f"size {vocab_size} exceeds the exact FP32 integer range."
        )

    valid_size = vocab_end - vocab_start
    if valid_size < local_scores.shape[-1]:
        local_scores[..., valid_size:] = -float("inf")

    local_values, local_indices = local_scores.max(dim=-1)
    global_indices = local_indices + vocab_start
    if tp_size == 1:
        return global_indices.to(torch.int64)

    local_pair = torch.stack(
        (local_values.float(), global_indices.float()),
        dim=-1,
    )
    gathered = all_gather(local_pair)
    gathered = gathered.view(local_scores.shape[0], tp_size, 2)

    # all_gather is rank ordered. PyTorch argmax selects the first maximum,
    # preserving full-vocabulary argmax tie breaking for contiguous shards.
    winner_rank = gathered[..., 0].argmax(dim=-1, keepdim=True)
    return gathered[..., 1].gather(dim=-1, index=winner_rank).squeeze(-1).to(torch.int64)
