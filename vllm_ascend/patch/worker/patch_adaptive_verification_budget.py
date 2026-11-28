# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Neutralize invalid adaptive-verification confidences before allocation.

``assign_draft_token_budget`` ranks survival probabilities by the integer bit
pattern of their float values and binary-searches up to ``0x7F800000`` (float
inf). A NaN confidence (``0x7FC00000``) or an out-of-range probability sorts
above every real draft, is admitted unconditionally, and makes the per-request
admitted capacities sum to more than the global draft budget. On Ascend the
device ``query_start_loc`` then no longer matches the adaptive-verification
token count, which trips the FIA padding invariant in ``NPUModelRunner``.

Map every confidence outside ``(0, 1]`` to zero for the slots being allocated,
mirroring the upstream fix (vllm-project/vllm#69440). Zero survival is a valid
tie that the allocator already distributes exactly, so the admitted counts sum
to the budget again. The patch disables itself once upstream applies the clamp
inside the kernel.
"""

import functools
import inspect

import torch
from vllm.v1.worker.gpu.spec_decode import adaptive_verification

_SANITIZER_MARKER = "_vllm_ascend_invalid_confidence_sanitizer"


def _sanitize_invalid_confidence(confidence_probs: torch.Tensor, idx_mapping: torch.Tensor) -> torch.Tensor:
    """Return a copy with every slot probability outside ``(0, 1]`` set to zero.

    NaN fails both comparisons, so the mask also covers NaN and negative or
    infinite values without a separate check.
    """
    sanitized = confidence_probs.clone()
    rows = sanitized[idx_mapping]
    invalid = ~((rows > 0.0) & (rows <= 1.0))
    rows[invalid] = 0.0
    sanitized[idx_mapping] = rows
    return sanitized


def _upstream_sanitizes_invalid_confidence(module) -> bool:
    kernel = getattr(module, "_assign_draft_token_budget_kernel", None)
    fn = getattr(kernel, "fn", kernel)
    if fn is None:
        return False
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return "prob > 0.0" in source and "prob <= 1.0" in source


def _install(module=adaptive_verification) -> None:
    original = getattr(module, "assign_draft_token_budget", None)
    if original is None:
        return
    if getattr(original, _SANITIZER_MARKER, False):
        return
    if _upstream_sanitizes_invalid_confidence(module):
        return

    @functools.wraps(original)
    def assign_draft_token_budget(confidence_probs, idx_mapping, capacities, draft_budget, survival):
        sanitized = _sanitize_invalid_confidence(confidence_probs, idx_mapping)
        return original(sanitized, idx_mapping, capacities, draft_budget, survival)

    setattr(assign_draft_token_budget, _SANITIZER_MARKER, True)
    module.assign_draft_token_budget = assign_draft_token_budget


_install()
