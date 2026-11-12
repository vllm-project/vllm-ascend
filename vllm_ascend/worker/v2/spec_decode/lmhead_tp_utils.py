# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from vllm_ascend.utils import lmhead_tp_enable, lmhead_tp_max_num_logits, lmhead_tp_pad_rows

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class LmheadTPDraftSamplingMixin:
    """Align draft LM-head collectives without exposing padding to sampling.

    Pad hidden states to the group-agreed capacity, then trim logits before
    upstream sampling reads per-row metadata or writes the draft logits cache.
    """

    # Attributes injected by the hosting speculator; annotated for standalone
    # mypy analysis.
    method: str
    vllm_config: "VllmConfig"
    _lmhead_pad_size: int | None
    max_num_reqs: int
    num_speculative_steps: int
    use_local_argmax_reduction: bool
    enable_adaptive_verification: bool

    # Speculators bypassing this compute_draft_logits hook need their own
    # alignment (DSpark wraps its model's compute_draft_logits). Paths without
    # such alignment opt out, e.g. DFlash2's get_top_k_tokens on the sharded head.
    _lmhead_tp_sample_draft_supported = True

    def _lmhead_tp_max_num_logits(self) -> int:
        return lmhead_tp_max_num_logits(self.max_num_reqs, self.num_speculative_steps + 1)

    @contextmanager
    def _lmhead_pad_size_context(self, dp_sync=None, capture_num_reqs: int | None = None):
        # TODO: Integrate this padding logic into the main execution flow.
        # The context manager currently minimizes changes to that flow.
        previous = getattr(self, "_lmhead_pad_size", None)
        if previous is None and lmhead_tp_enable() and self.method == "mtp":
            pad_size = self._lmhead_tp_max_num_logits()
            if self.vllm_config.parallel_config.decode_context_parallel_size == 1:
                # Each draft step samples one hidden-state row per request,
                # regardless of the target's speculative verification length.
                # DP requests must agree across ranks; never substitute a local
                # count when a sync state lacks the shared request bound.
                request_bound = getattr(dp_sync, "num_reqs", None) if dp_sync is not None else capture_num_reqs
                pad_size = self.max_num_reqs
                if request_bound is not None:
                    # Keep a non-empty collective shape for dummy/idle calls.
                    pad_size = min(pad_size, max(1, int(request_bound)))
            self._lmhead_pad_size = pad_size
        try:
            yield
        finally:
            self._lmhead_pad_size = previous

    def _lmhead_tp_validate_draft_sampling(self) -> None:
        """Fail unsupported draft sampling at construction, not first use."""
        if not lmhead_tp_enable():
            return
        if not self._lmhead_tp_sample_draft_supported:
            raise NotImplementedError(
                f"lmhead TP does not support {type(self).__name__}: its draft "
                "sampling does not go through sample_draft, so the "
                "group-aligned row padding cannot be applied."
            )
        if self.use_local_argmax_reduction:
            raise NotImplementedError(
                "lmhead TP does not support use_local_argmax_reduction: "
                "get_top_tokens reduces over the local vocab shard only, which "
                "silently produces wrong tokens under pure-DP lmhead TP."
            )
        if self.enable_adaptive_verification:
            raise NotImplementedError("lmhead TP does not support enable_adaptive_verification.")

    def compute_draft_logits(self, hidden_states: torch.Tensor, spec_step_idx: int) -> torch.Tensor:
        if not lmhead_tp_enable():
            return super().compute_draft_logits(hidden_states, spec_step_idx)  # type: ignore[misc]

        num_logits = hidden_states.shape[0]
        padded = lmhead_tp_pad_rows(
            hidden_states,
            getattr(self, "_lmhead_pad_size", None) or self._lmhead_tp_max_num_logits(),
            "max_num_reqs * (num_speculative_steps + 1)",
        )
        logits = super().compute_draft_logits(padded, spec_step_idx)  # type: ignore[misc]
        # Keep every original row (including DFlash's multiple rows per request),
        # but remove the extra LM-head TP rows before any sampler/cache access.
        return logits[:num_logits]
