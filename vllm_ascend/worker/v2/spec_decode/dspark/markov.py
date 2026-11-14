# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch


def compute_markov_chain_confidence(
    compute_confidence: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    hidden: torch.Tensor,
    markov_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Compute confidence from per-step embeddings returned by the sampled chain."""
    confidence = compute_confidence(hidden, markov_embeddings.flatten(0, 1))
    return confidence.reshape(markov_embeddings.shape[:2])
