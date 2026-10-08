# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gating and token-eligibility helpers for the Ascend Engram port.

The bucket layout and the hashing itself come from upstream, so a checkpoint
lands on the same rows here as it does on the accelerators upstream supports;
the Ascend token history lives in ``hash_state`` next to the SWA slot cache,
which is where upstream keeps it too.
"""

import torch


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


def gather_engram_lookback(
    positions,
    query_start_loc,
    request_indices,
    all_token_ids,
    total_lens,
    depth,
    *,
    execution_device=None,
):
    """Read accepted history from MRV2's device request slots.

    Column j is the token at chunk_start - 1 - j. Both prompt and generated
    tokens live in all_token_ids; total_lens excludes unaccepted draft slots.
    Use the current request index mapping, never cached physical KV pages.
    """
    history_device = all_token_ids.device
    if execution_device is None:
        execution_device = next(
            (tensor.device for tensor in (positions, total_lens, request_indices) if tensor.device.type != "cpu"),
            positions.device,
        )
    starts = positions.index_select(0, query_start_loc[:-1].to(device=positions.device, dtype=torch.long))
    lengths = total_lens.index_select(0, request_indices.to(device=total_lens.device, dtype=torch.long))
    # Ascend's registered UVA history is a CPU tensor; without UVA the same
    # attribute is an NPU tensor. Move only per-request control vectors to
    # its actual device, never the potentially multi-GB token table.
    starts = starts.to(device=history_device)
    lengths = lengths.to(device=history_device)
    request_indices = request_indices.to(device=history_device, dtype=torch.long)
    if history_device.type == "cpu" and execution_device.type != "cpu":
        # post_update writes accepted tokens through the UVA pointer on the
        # runner's current stream. Fence before host indexing, including when
        # the caller already provided CPU control vectors and no D2H copy
        # would otherwise make these asynchronous writes visible.
        getattr(torch, execution_device.type).current_stream(execution_device).synchronize()
    previous = starts[:, None] - 1 - torch.arange(depth, device=history_device)
    valid = (previous >= 0) & (previous < lengths[:, None])
    columns = previous.clamp(0, all_token_ids.shape[1] - 1).long()
    tokens = all_token_ids[request_indices[:, None], columns]
    return tokens.masked_fill(~valid, -1)


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
        original = (original.unflatten(-1, (-1, rotation_block.shape[0])) @ rotation_block.float().T).flatten(-2)
    key = key.float()
    rstd = torch.rsqrt(original.square().mean(-1) + eps)
    rstd *= torch.rsqrt(key.square().mean(-1) + eps)
    dot = (original * channel_weight.float() * key).sum(-1) * rstd * dim**-0.5
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    gate = torch.sigmoid(torch.where(torch.signbit(dot), -magnitude, magnitude))
    gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (hidden.float() + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)
