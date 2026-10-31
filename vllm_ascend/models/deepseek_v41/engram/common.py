# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gating and token-eligibility helpers for the Ascend Engram port.

The bucket layout and the hashing itself come from upstream, so a checkpoint
lands on the same rows here as it does on the accelerators upstream supports;
the Ascend token history lives in ``hash_state`` next to the SWA slot cache,
which is where upstream keeps it too.
"""

from pathlib import Path

import torch
from safetensors import safe_open

_ROTATION_BLOCK_SIZE = 32


def load_engram_rotation_block(model_root: str, hidden_size: int, rotation_path: Path | None = None) -> torch.Tensor:
    """Use the A3 Quarot basis, or the native A5 identity basis."""
    rotation_path = rotation_path or Path(model_root) / "optional/quarot.safetensors"
    if not rotation_path.is_file():
        return torch.eye(_ROTATION_BLOCK_SIZE)
    with safe_open(rotation_path, framework="pt") as checkpoint:
        rotation = checkpoint.get_tensor("global_rotation")

    num_blocks = hidden_size // _ROTATION_BLOCK_SIZE
    blocks = rotation.reshape(
        num_blocks,
        _ROTATION_BLOCK_SIZE,
        num_blocks,
        _ROTATION_BLOCK_SIZE,
    ).permute(0, 2, 1, 3)
    diag_blocks = blocks[torch.arange(num_blocks), torch.arange(num_blocks)].contiguous()
    if torch.equal(diag_blocks, diag_blocks[:1].expand_as(diag_blocks)):
        return diag_blocks[0]
    return diag_blocks


def engram_enabled(text_config) -> bool:
    """Whether the checkpoint declares Engram n-gram layers."""

    return bool(getattr(text_config, "engram_layer_ids", None))


def valid_engram_token_mask(
    input_ids: torch.Tensor,
    image_token_id: int,
    image_pad_token_id: int,
) -> torch.Tensor:
    """Exclude the complete V4.1 image region from n-gram history."""
    return (input_ids != image_token_id) & (input_ids != image_pad_token_id)


def engram_gate(
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    channel_weight: torch.Tensor,
    rotation_block: torch.Tensor | None,
    token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply original-basis gating, undoing checkpoint rotation when present.

    ``hidden`` and ``key`` have shape [tokens, hc_mult, hidden_size].
    The saved rotation consists of identical diagonal blocks. Restore hidden
    in FP32; the value projection already includes the forward rotation.
    """
    dim = hidden.shape[-1]
    original = hidden.float()
    if rotation_block is not None:
        if rotation_block.ndim == 2:
            original = (original.unflatten(-1, (-1, rotation_block.shape[0])) @ rotation_block.float().T).flatten(-2)
        elif rotation_block.ndim == 3:
            original = (
                torch.matmul(
                    original.unflatten(-1, (rotation_block.shape[0], _ROTATION_BLOCK_SIZE)).unsqueeze(-2),
                    rotation_block.float().transpose(-1, -2),
                )
                .squeeze(-2)
                .flatten(-2)
            )
        else:
            raise ValueError("Engram gate rotation must be a block or block table")
    key = key.float()
    rstd = torch.rsqrt(original.square().mean(-1) + eps)
    rstd *= torch.rsqrt(key.square().mean(-1) + eps)
    dot = (original * channel_weight.float() * key).sum(-1) * rstd * dim**-0.5
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    gate = torch.sigmoid(torch.where(torch.signbit(dot), -magnitude, magnitude))
    gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (hidden.float() + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)
