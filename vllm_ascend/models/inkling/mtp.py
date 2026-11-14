# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initial Inkling MTP input preparation; model registration is pending."""

import torch
from torch import nn


def _prepare_mtp_input(
    hidden_states: torch.Tensor,
    token_embeddings: torch.Tensor,
    hidden_norm: nn.Module,
    depth_embed_norm: nn.Module,
    backbone_embed_norm: nn.Module | None = None,
) -> torch.Tensor:
    """Build the depth projection input from hidden states and raw token embeddings."""
    if backbone_embed_norm is not None:
        token_embeddings = backbone_embed_norm(token_embeddings)
    return torch.cat(
        (hidden_norm(hidden_states), depth_embed_norm(token_embeddings)),
        dim=-1,
    )
