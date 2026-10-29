# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch

from vllm_ascend.utils import lmhead_tp_enable, lmhead_tp_max_num_logits, lmhead_tp_pad_rows, vllm_version_is

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class LmheadTPDraftSamplingMixin:
    """Group-aligned draft LM-head sampling for lmhead TP (V1 parity): pad
    the greedy hidden states to the ``lmhead_tp_max_num_logits`` capacity and
    trim back; probabilistic sampling (fixed-size buffers) is rejected at init."""

    # Attributes injected by the hosting speculator; annotated for standalone
    # mypy analysis.
    method: str
    vllm_config: "VllmConfig"
    _lmhead_pad_size: int | None
    max_num_reqs: int
    num_speculative_steps: int
    speculative_config: Any
    use_local_argmax_reduction: bool
    enable_adaptive_verification: bool

    # Speculators whose sampling bypasses sample_draft cannot be row-aligned;
    # they opt out and are rejected at construction (DSpark:
    # compute_draft_logits, DFlash2: get_top_k_tokens on the sharded head).
    _lmhead_tp_sample_draft_supported = True

    def _lmhead_tp_max_num_logits(self) -> int:
        return lmhead_tp_max_num_logits(self.max_num_reqs, self.num_speculative_steps + 1)

    @contextmanager
    def _lmhead_pad_size_context(self, dp_sync=None, capture_num_reqs: int | None = None):
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
        if self.speculative_config.draft_sample_method == "probabilistic":
            raise NotImplementedError(
                "lmhead TP does not support draft_sample_method='probabilistic': "
                "the gumbel path writes into fixed-size draft buffers that cannot "
                "hold the group-aligned padding rows."
            )
        if self.use_local_argmax_reduction:
            raise NotImplementedError(
                "lmhead TP does not support use_local_argmax_reduction: "
                "get_top_tokens reduces over the local vocab shard only, which "
                "silently produces wrong tokens under pure-DP lmhead TP."
            )
        if self.enable_adaptive_verification:
            raise NotImplementedError(
                "lmhead TP does not support enable_adaptive_verification: the "
                "online acceptance estimator sizes its kernels by the draft "
                "logit row count and indexes its per-token features and "
                "confidence buffers by it, so the group-aligned padding rows "
                "read the request mapping and write those buffers out of bounds."
            )

    if vllm_version_is("0.32.0"):

        def sample_draft(
            self,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            idx_mapping: torch.Tensor,
            temperature: torch.Tensor,
            seeds: torch.Tensor,
            draft_step: torch.Tensor,
            draft_logits: torch.Tensor | None,
        ):
            if not lmhead_tp_enable():
                return super().sample_draft(  # type: ignore[misc]
                    hidden_states, positions, idx_mapping, temperature, seeds, draft_step, draft_logits
                )
            if draft_logits is not None:
                raise NotImplementedError(
                    "lmhead TP does not support draft_sample_method='probabilistic': "
                    "the gumbel path writes into fixed-size draft buffers that cannot "
                    "hold the group-aligned padding rows."
                )
            num_logits = hidden_states.shape[0]
            # Shared pad primitive (the runner's sample() pads the target head the
            # same way); zero rows carry no draft token and are trimmed back off.
            padded = lmhead_tp_pad_rows(
                hidden_states,
                getattr(self, "_lmhead_pad_size", None) or self._lmhead_tp_max_num_logits(),
                "max_num_reqs * (num_speculative_steps + 1)",
            )
            out = super().sample_draft(  # type: ignore[misc]
                padded, positions, idx_mapping, temperature, seeds, draft_step, draft_logits
            )
            return out[:num_logits]

    else:
        # vLLM #59554 added `num_steps` to `Speculator.sample_draft`; DFlash
        # passes it so the draft watermarker can key its draw by the step count.
        def sample_draft(  # type: ignore[misc]
            self,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            idx_mapping: torch.Tensor,
            temperature: torch.Tensor,
            seeds: torch.Tensor,
            draft_step: torch.Tensor,
            draft_logits: torch.Tensor | None,
            num_steps: int = 1,
        ):
            if not lmhead_tp_enable():
                return super().sample_draft(  # type: ignore[misc]
                    hidden_states,
                    positions,
                    idx_mapping,
                    temperature,
                    seeds,
                    draft_step,
                    draft_logits,
                    num_steps=num_steps,
                )
            if draft_logits is not None:
                raise NotImplementedError(
                    "lmhead TP does not support draft_sample_method='probabilistic': "
                    "the gumbel path writes into fixed-size draft buffers that cannot "
                    "hold the group-aligned padding rows."
                )
            num_logits = hidden_states.shape[0]
            # Shared pad primitive (the runner's sample() pads the target head the
            # same way); zero rows carry no draft token and are trimmed back off.
            padded = lmhead_tp_pad_rows(
                hidden_states,
                getattr(self, "_lmhead_pad_size", None) or self._lmhead_tp_max_num_logits(),
                "max_num_reqs * (num_speculative_steps + 1)",
            )
            out = super().sample_draft(  # type: ignore[misc]
                padded,
                positions,
                idx_mapping,
                temperature,
                seeds,
                draft_step,
                draft_logits,
                num_steps=num_steps,
            )
            return out[:num_logits]
