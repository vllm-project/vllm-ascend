# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU compat: keep DiffusionGemma's logit soft-cap out of torch.compile.

Upstream ``vllm.model_executor.models.diffusion_gemma`` wraps the final-logit
soft-cap in ``@torch.compile(dynamic=True)``::

    @torch.compile(dynamic=True)
    def _softcap_logits(logits, cap):
        logits = logits.float()
        return torch.tanh(logits / cap) * cap

On CUDA/inductor this fuses cast/div/tanh/mul into one kernel. On Ascend the
torch_npu inductor path *autotunes* the compiled kernel, and one of the tiling
configs it benchmarks hangs the vector core with ``507034 vector core timeout``
during engine warmup — before any token is generated. (The tanh op itself is
fine in isolation; it is the autotune config sweep that trips.)

This patch replaces the module-level ``_softcap_logits`` with an eager function
of identical numerics, so the model-forward graph capture stays intact while the
soft-cap runs eagerly and never enters inductor autotune. NPU-only by
construction (vllm_ascend only loads on Ascend).
"""

from __future__ import annotations

import torch


def _softcap_logits_eager(logits: torch.Tensor, cap: float) -> torch.Tensor:
    # fp32 before tanh for numerical stability (matches HF DiffusionGemma).
    logits = logits.float()
    return torch.tanh(logits / cap) * cap


def install_diffusion_gemma_softcap_compat() -> None:
    """Neutralize the torch.compile on DiffusionGemma's soft-cap for Ascend."""
    try:
        import vllm.model_executor.models.diffusion_gemma as _dg
    except Exception:
        # DiffusionGemma not present in this vllm build; nothing to patch.
        return
    if getattr(_dg, "_softcap_logits", None) is None:
        return
    # Callers reference the module global _softcap_logits, resolved at call
    # time, so replacing the attribute is sufficient.
    _dg._softcap_logits = _softcap_logits_eager
