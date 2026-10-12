# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mask prefill placeholders at the upstream verification boundary."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler


def mask_prefill_placeholder_drafts(input_batch: "InputBatch", draft_sampled: torch.Tensor) -> torch.Tensor:
    """Keep embedding-safe placeholders out of speculative verification.

    Use the actual expanded logits layout, including adaptive verification,
    rather than assuming a fixed number of draft rows per request. Token 0
    remains a valid proposal during decode; only prefill draft rows are masked.
    """
    if not (input_batch.has_prefill and input_batch.num_draft_tokens):
        return draft_sampled

    prefill_mask = None
    for req_idx, is_prefilling in enumerate(input_batch.is_prefilling_np):
        if not is_prefilling:
            continue
        req_state_idx = int(input_batch.idx_mapping_np[req_idx])
        req_mask = input_batch.expanded_idx_mapping == req_state_idx
        prefill_mask = req_mask if prefill_mask is None else prefill_mask | req_mask
    if prefill_mask is None:
        return draft_sampled

    invalid_drafts = prefill_mask & (input_batch.expanded_local_pos > 0)
    return torch.where(invalid_drafts, -1, draft_sampled)


class _PrefillPlaceholderMask:
    """Delegate verification without copying the upstream sampling pipeline."""

    def __init__(self, verify: Callable[..., tuple[torch.Tensor, torch.Tensor, "LogprobsTensors | None"]]) -> None:
        self._verify = verify

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: "InputBatch",
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        max_chunk_logits: int,
        max_num_logprobs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, "LogprobsTensors | None"]:
        return self._verify(
            logits,
            input_batch,
            draft_logits,
            mask_prefill_placeholder_drafts(input_batch, draft_sampled),
            pos,
            max_chunk_logits,
            max_num_logprobs,
        )


def install_prefill_placeholder_mask(sampler: "RejectionSampler") -> None:
    """Install once on this instance, leaving the upstream class untouched.

    The supported upstream __call__ gathers drafts before _verify_in_chunks;
    nothing between those steps consumes their values. Keep this private
    boundary covered by the sampler differential test when updating vLLM.
    Once upstream provides the same mask, this local adapter can be removed.
    """
    if not isinstance(sampler._verify_in_chunks, _PrefillPlaceholderMask):
        sampler._verify_in_chunks = _PrefillPlaceholderMask(sampler._verify_in_chunks)  # type: ignore[method-assign]
