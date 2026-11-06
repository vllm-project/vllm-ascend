"""Compact LM Head: index_select K candidate weight rows -> ``[B, K]``.

This is the device-side win: instead of ``[B, H] @ [V, H]^T -> [B, V]`` and a
tensor-parallel all-gather, we gather the K candidate weight rows
(``index_select`` -> ``[K, H]``) and do a compact ``F.linear`` -> ``[B, K]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .admission import AdmissionDecision, TargetTokenScoringState

if TYPE_CHECKING:
    from torch import nn


def _apply_layer_postprocess(
    logits_processor: object, logits: torch.Tensor
) -> torch.Tensor:
    """Apply the row-wise transforms the LM Head's own ``LogitsProcessor``
    would apply (soft_cap, scale) to the compact ``[B, K]`` logits.

    These transforms are per-row and order-independent, so they are correct on
    the compact tensor. The full ``LogitsProcessor`` is **not** invoked because
    its other branches assume a full-vocab width (``allowed_token_ids`` /
    ``bad_words`` masks index vocab ids); admission rejects every sampler-mask
    semantic precisely so only ``soft_cap``/``scale`` can be active here. This
    mirrors the head's postprocess rather than reusing the call site so the
    compact tensor never reaches a vocab-shaped code path.
    """
    soft_cap = getattr(logits_processor, "soft_cap", None)
    if soft_cap is not None:
        logits = torch.tanh(logits / soft_cap) * soft_cap
    scale = getattr(logits_processor, "scale", 1.0)
    if scale != 1.0:
        logits = logits * scale
    return logits


def project_target_token_logits(
    model: nn.Module,
    hidden_states: torch.Tensor,
    decision: AdmissionDecision,
) -> tuple[torch.Tensor, TargetTokenScoringState] | None:
    """Project ``hidden_states`` onto the K candidate weight rows.

    Args:
        model: The vLLM model exposing ``lm_head`` and ``logits_processor``.
        hidden_states: ``[B, H]`` hidden states at the scoring position.
        decision: A positive (``ok=True``) admission decision.

    Returns:
        ``(compact_logits[B, K], state)`` or ``None`` if the LM Head turns
        out to be non-compact at projection time (caller falls back native).
    """
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return None
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.ndim != 2:
        return None
    # Most models pass no embedding_bias; support it if the head exposes one.
    bias = getattr(lm_head, "bias", None)

    target_ids = tuple(decision.target_token_ids)  # type: ignore[arg-type]
    idx = torch.as_tensor(target_ids, dtype=torch.long, device=weight.device)
    sel_w = weight.index_select(0, idx).contiguous()
    sel_b = None if bias is None else bias.index_select(0, idx).contiguous()

    logits = F.linear(hidden_states, sel_w, sel_b)  # [B, K]

    logits_processor = getattr(model, "logits_processor", None)
    if logits_processor is not None:
        logits = _apply_layer_postprocess(logits_processor, logits)

    state = TargetTokenScoringState.from_ids(list(target_ids), weight.device)
    return logits, state
