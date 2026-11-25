# SPDX-License-Identifier: Apache-2.0
"""Expose a request-sharded q cache without copying its vocabulary rows."""

from types import SimpleNamespace
from typing import Any

import torch


def prepare_rejection_cache(
    cache: Any,
    draft_sampled: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Keep cache storage and request-state indices in their original domains.

    Core owns row assignment and detects rows overwritten by another request.
    Only draft token ids are masked; bonus rows (local position zero) survive.
    No q logits, temperature, seed, or request-index tensor is gathered here.
    """
    if cache.row_mapping is not None:
        batch = SimpleNamespace(
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
        )
        stale = cache.stale_drafts(batch)
        if stale is not None:
            draft_sampled = draft_sampled.masked_fill(stale, -1)
    return cache.values, cache.row_mapping, draft_sampled
