# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Backport vLLM #58784 for the MRV2 rejection sampler on Ascend.

The verified vLLM commit and the matching release still gather draft IDs from
``input_ids``. A prompt-tail step can contain padded draft slots whose values
are zero, although the drafter has never proposed them. Gather and mark those
slots in one kernel, as upstream does, without changing ``input_ids``.
"""

import torch
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs

from vllm_ascend.ops.triton.v2.spec_decode.gather_draft_sampled import gather_draft_sampled


def _call_with_rejected_unproposed_drafts(
    self: rs.RejectionSampler,
    logits: torch.Tensor,
    input_batch: rs.InputBatch,
    draft_logits: torch.Tensor | None = None,
) -> rs.SamplerOutput:
    # Keep the upstream call sequence. Only the draft/position gather differs.
    num_nans = rs.get_num_nans(logits) if self.sampler.compute_nans else None
    draft_sampled, pos = gather_draft_sampled(
        input_batch.input_ids,
        input_batch.positions,
        input_batch.logits_indices,
        input_batch.expanded_idx_mapping,
        input_batch.expanded_local_pos,
        self.sampler.req_states.prefill_len.gpu,
    )
    max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(input_batch.idx_mapping_np)
    sampled, num_sampled, logprobs_tensors = self._verify_in_chunks(
        logits,
        input_batch,
        draft_logits,
        draft_sampled,
        pos,
        rs.get_max_chunk_logits(logits.shape[1]),
        max_num_logprobs,
    )
    num_sampled, num_rejected = rs.get_num_sampled_and_rejected(
        num_sampled,
        input_batch.seq_lens,
        input_batch.cu_num_logits,
        input_batch.idx_mapping,
        self.sampler.req_states.prefill_len.gpu,
    )
    return rs.SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=logprobs_tensors,
        num_nans=num_nans,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
    )


# vLLM #58784 defines this helper. Do not replace the upstream implementation
# once the dependency has advanced past the backport.
if not hasattr(rs, "gather_draft_sampled"):
    rs.RejectionSampler.__call__ = _call_with_rejected_unproposed_drafts
