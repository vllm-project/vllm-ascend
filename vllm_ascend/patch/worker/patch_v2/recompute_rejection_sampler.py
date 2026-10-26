# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep physical prefill semantics local to speculative verification."""

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler


class RecomputeAwareRejectionSampler:
    """Adapt an instance without changing graph dispatch or request state.

    A recomputed prompt tail can have a decode-shaped graph while still
    lacking real local draft proposals. Restore its physical prefill flag
    only for the rejection sampler's existing placeholder-token mask.
    """

    def __init__(self, sampler: "RejectionSampler") -> None:
        self._sampler = sampler

    def __getattr__(self, name: str):
        return getattr(self._sampler, name)

    def __call__(
        self,
        logits: "torch.Tensor",
        input_batch: "InputBatch",
        draft_logits: "torch.Tensor | None" = None,
    ):
        sampler_batch = input_batch
        if input_batch.num_draft_tokens:
            physical_prefill = input_batch.is_prefilling_np | (
                input_batch.num_computed_prefill_tokens_np < input_batch.prefill_len_np
            )
            if not np.array_equal(physical_prefill, input_batch.is_prefilling_np):
                sampler_batch = replace(
                    input_batch,
                    is_prefilling_np=physical_prefill,
                    has_prefill=bool(physical_prefill.any()),
                )
        return self._sampler(logits, sampler_batch, draft_logits)
