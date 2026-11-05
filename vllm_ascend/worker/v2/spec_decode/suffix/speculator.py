# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Reuse upstream suffix orchestration with Ascend kernel launch adapters."""

from __future__ import annotations

import torch
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.suffix import speculator as suffix_speculator


class AscendSuffixSpeculator(suffix_speculator.SuffixSpeculator):
    """Inherit buffers, lifecycle and drafting flow; adapt device launches only."""

    def propose(self, input_batch: InputBatch, *args, **kwargs) -> torch.Tensor:
        # The parent handles dummy runs, but assumes a nonempty launch grid.
        if input_batch.num_reqs == 0:
            return self.drafts[:0]

        return super().propose(input_batch, *args, **kwargs)
