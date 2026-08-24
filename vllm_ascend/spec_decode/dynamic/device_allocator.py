# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Device-side request allocation for hardware-aware speculative decoding.

The hardware profile chooses the total number of draft positions on the host
side because the profile lookup is a small control-plane operation.  Mapping
that total back to requests is data parallel work, however, and should stay on
the accelerator.  This module mirrors the allocation kernel shape used by
upstream adaptive verification: flatten the candidate prefix scores, select
the winning positions on device, then reduce the admission mask by request.

Ascend does not need a custom C kernel for this first implementation.  The
compiled PyTorch path is deliberately kept behind a small fallback boundary;
on a platform where ``torch.compile`` cannot lower ``topk`` or boolean
reductions, the exact same algorithm remains available for correctness and
diagnostics.
"""

from __future__ import annotations

from typing import Callable

import torch

from vllm.logger import logger


def _assign_prefix_budget_compilable(
    ranked_candidates: torch.Tensor,
    mandatory: int,
    extra: int,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Assign a globally selected prefix budget without host row mapping.

    ``ranked_candidates`` is shaped ``[num_reqs, remaining_steps]``.  The
    caller has already added a deterministic tie-break to the scores, so the
    device ``topk`` result is stable for equal confidence values.
    """

    flat = ranked_candidates.reshape(-1)
    winners = flat.topk(extra, largest=True, sorted=False).indices
    admitted = torch.zeros_like(flat, dtype=torch.bool).index_fill_(
        0, winners, True
    )
    per_request = admitted.view_as(ranked_candidates).sum(dim=1)
    lengths.copy_(per_request.to(dtype=lengths.dtype))
    lengths.add_(mandatory)
    return lengths

try:
    _compiled_assign_prefix_budget: Callable[..., torch.Tensor] | None = torch.compile(
        _assign_prefix_budget_compilable,
        dynamic=True,
    )
except (AttributeError, RuntimeError, TypeError):
    # Some supported torch versions expose no compiler, or reject dynamic
    # compilation at import time.  Keep import-time failure out of serving.
    _compiled_assign_prefix_budget = None

_compile_failed = False


def assign_prefix_budget(
    ranked_candidates: torch.Tensor,
    *,
    mandatory: int,
    extra: int,
    lengths: torch.Tensor,
    use_compiled: bool = True,
) -> torch.Tensor:
    """Fill ``lengths`` with the device-side request prefix allocation.

    ``extra`` is the number of positions in addition to ``mandatory`` per
    batch.  The function is intentionally shape-agnostic and works for a
    smaller runtime physical K than the configured maximum.
    """

    if ranked_candidates.ndim != 2:
        raise ValueError("ranked_candidates must be a 2-D tensor")
    if lengths.ndim != 1 or lengths.shape[0] != ranked_candidates.shape[0]:
        raise ValueError("lengths must have one entry per request")
    if mandatory < 0 or extra < 0:
        raise ValueError("mandatory and extra must be non-negative")
    if extra > ranked_candidates.numel():
        raise ValueError("extra exceeds the number of candidate positions")

    if extra == 0:
        lengths.fill_(mandatory)
        return lengths

    global _compile_failed
    can_compile = (
        use_compiled
        and not _compile_failed
        and _compiled_assign_prefix_budget is not None
        and ranked_candidates.device.type in {"npu", "cuda"}
    )
    if can_compile:
        try:
            return _compiled_assign_prefix_budget(
                ranked_candidates,
                mandatory,
                extra,
                lengths,
            )
        except Exception as exc:  # pragma: no cover - hardware dependent
            _compile_failed = True
            logger.warning(
                "Compiled speculative request allocation is unavailable; "
                "falling back to eager device operators: %s",
                exc,
            )

    flat = ranked_candidates.reshape(-1)
    winners = flat.topk(extra, largest=True, sorted=False).indices
    admitted = torch.zeros_like(flat, dtype=torch.bool).index_fill_(
        0, winners, True
    )
    per_request = admitted.view_as(ranked_candidates).sum(dim=1)
    lengths.copy_(per_request.to(dtype=lengths.dtype))
    lengths.add_(mandatory)
    return lengths
