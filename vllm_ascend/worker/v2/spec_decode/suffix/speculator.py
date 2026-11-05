# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Reuse upstream suffix orchestration with Ascend kernel launch adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.suffix import speculator as suffix_speculator

from vllm_ascend.ops.triton.spec_decode.suffix import (
    ascend_suffix_ingestion,
    ascend_suffix_proposal,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


class AscendSuffixSpeculator(suffix_speculator.SuffixSpeculator):
    """Inherit buffers, lifecycle and drafting flow; adapt device launches only."""

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        with ascend_suffix_ingestion(suffix_speculator):
            return super().on_requests_finished(finished_req_ids)

    def propose(self, input_batch: InputBatch, *args, **kwargs) -> torch.Tensor:
        # The parent handles dummy runs, but assumes a nonempty launch grid.
        if input_batch.num_reqs == 0:
            return self.drafts[:0]

        with ascend_suffix_proposal(suffix_speculator):
            return super().propose(input_batch, *args, **kwargs)
