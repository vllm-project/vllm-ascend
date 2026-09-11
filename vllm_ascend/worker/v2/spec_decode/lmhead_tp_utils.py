# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

import torch

from vllm_ascend.utils import lmhead_tp_configured, lmhead_tp_max_num_logits


class LmheadTPDraftSamplingMixin:
    """Group-aligned draft LM-head sampling for lmhead TP (V1 parity).

    The draft LM head is vocab-sharded like the target head, so every
    ``sample_draft`` call must feed the group the same row count: pad the
    hidden states with zero rows up to the ``lmhead_tp_max_num_logits``
    capacity and trim back (V1: ``token_indices_to_sample``). Probabilistic
    sampling writes fixed-size draft buffers that cannot hold the padding
    rows and is rejected at init. Draft and target collectives are
    independent; the capacity convention matches the target side only while
    model states sample one token per step.
    """

    # Attributes injected by the hosting speculator via mixin composition;
    # annotated here so mypy can resolve them when the mixin is analyzed
    # standalone.
    max_num_reqs: int
    num_speculative_steps: int
    speculative_config: Any
    use_local_argmax_reduction: bool

    # Speculators whose draft sampling does not funnel through sample_draft
    # cannot be row-aligned by this mixin; they opt out and are rejected at
    # construction (DSpark: _sample_sequential calls compute_draft_logits
    # directly).
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
        capacity = self._lmhead_tp_max_num_logits()
        num_logits = hidden_states.shape[0]
        if num_logits > capacity:
            # A mismatch would desync the draft LM-head all_gather/all_to_all
            # across the group and hang the collectives. Fail fast instead.
            raise ValueError(
                f"lmhead TP draft rows ({num_logits}) exceed the group-agreed "
                f"capacity ({capacity} = max_num_reqs * (num_speculative_steps + 1))."
            )
        padded = hidden_states
        if num_logits < capacity:
            # Zero rows carry no draft token; they are trimmed back off below.
            padded = torch.nn.functional.pad(hidden_states, (0, 0, 0, capacity - num_logits))
        out = super().sample_draft(  # type: ignore[misc]
            padded, positions, idx_mapping, temperature, seeds, draft_step, draft_logits
        )
        return out[:num_logits]
