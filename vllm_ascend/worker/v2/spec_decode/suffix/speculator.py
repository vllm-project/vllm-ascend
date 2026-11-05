# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Reuse upstream suffix orchestration with Ascend kernel launch adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.suffix import speculator as upstream

from vllm_ascend.worker.v2.spec_decode.suffix.kernel_adapters import (
    ascend_suffix_ingestion,
    ascend_suffix_proposal,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.v1.worker.gpu.dp_utils import DPSyncState


class AscendSuffixSpeculator(upstream.SuffixSpeculator):
    """Inherit buffers, lifecycle and drafting flow; adapt device launches only."""

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        with ascend_suffix_ingestion(upstream):
            return super().on_requests_finished(finished_req_ids)

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: Any,
        slot_mappings: Any,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        # The parent handles dummy runs, but assumes a nonempty launch grid.
        if input_batch.num_reqs == 0:
            return self.drafts[:0]

        with ascend_suffix_proposal(upstream):
            return super().propose(
                input_batch=input_batch,
                attn_metadata=attn_metadata,
                slot_mappings=slot_mappings,
                last_hidden_states=last_hidden_states,
                aux_hidden_states=aux_hidden_states,
                num_sampled=num_sampled,
                num_rejected=num_rejected,
                last_sampled=last_sampled,
                next_prefill_tokens=next_prefill_tokens,
                temperature=temperature,
                seeds=seeds,
                dp_sync_state=dp_sync_state,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                mm_inputs=mm_inputs,
                is_profile=is_profile,
                num_speculative_tokens=num_speculative_tokens,
            )
