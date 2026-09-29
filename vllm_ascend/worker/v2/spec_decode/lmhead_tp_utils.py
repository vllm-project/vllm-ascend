# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

import torch

from vllm_ascend.utils import lmhead_tp_configured, lmhead_tp_max_num_logits, lmhead_tp_pad_rows


class LmheadTPDraftSamplingMixin:
    """Group-aligned draft LM-head sampling for lmhead TP (V1 parity): pad
    the greedy hidden states to the ``lmhead_tp_max_num_logits`` capacity and
    trim back; probabilistic sampling (fixed-size buffers) is rejected at init."""

    # Attributes injected by the hosting speculator; annotated for standalone
    # mypy analysis.
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

    def _lmhead_tp_validate_draft_sampling(self) -> None:
        """Fail unsupported draft sampling at construction, not first use."""
        if not lmhead_tp_configured():
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
        if not lmhead_tp_configured():
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
            self._lmhead_tp_max_num_logits(),
            "max_num_reqs * (num_speculative_steps + 1)",
        )
        out = super().sample_draft(  # type: ignore[misc]
            padded, positions, idx_mapping, temperature, seeds, draft_step, draft_logits
        )
        return out[:num_logits]
