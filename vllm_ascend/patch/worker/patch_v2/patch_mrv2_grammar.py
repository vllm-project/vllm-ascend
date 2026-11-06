# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[Role] Backport vLLM #54442 worker logic on unmodified vLLM 0.30.0.
[Invariants] Shard limits with requests; only verification sees invalid IDs.
[Timeouts & Locks] No new waits, locks, or explicit device synchronization.
[Side-Effect Policy] Patch sampling methods in memory; never rewrite vLLM files.

Adapted from vLLM ced6857afa and #54442. Remove once the supported release
includes the upstream fix. Keep kernel calls through their upstream module:
patch_triton installs the Ascend verifier there before this module loads.
"""

from collections.abc import Iterable

import numpy as np
import torch
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu import model_runner as _model_runner
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample import batch_shard as _batch_shard
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as _rejection

from vllm_ascend.core.grammar_output import GrammarOutput
from vllm_ascend.worker.v2.structured_outputs import grammar_invalid_drafts

_ORIGINAL_SHARD_GRAMMAR_OUTPUT = _batch_shard._shard_grammar_output


def shard_grammar_output(grammar_output, input_batch, local_req_ids):
    local = _ORIGINAL_SHARD_GRAMMAR_OUTPUT(grammar_output, input_batch, local_req_ids)
    if local is None:
        return None
    limits = getattr(grammar_output, "num_acceptable_drafts", None)
    if limits is not None:
        owned = set(local.structured_output_request_ids)
        limits = [n for req_id, n in zip(grammar_output.structured_output_request_ids, limits) if req_id in owned]
    return GrammarOutput(local.structured_output_request_ids, local.grammar_bitmask, limits)


def sample(
    self,
    hidden_states: torch.Tensor,
    input_batch: InputBatch,
    grammar_output: GrammarOutput | None,
) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
    shard_metadata = None
    global_input_batch = input_batch
    if self.batch_sharder is not None:
        # Shard the inputs along the batch dimension to sample in parallel
        # across TP ranks.
        input_batch, sorted_logits_indices, grammar_output, shard_metadata = self.batch_sharder.shard_sampler_inputs(
            input_batch, grammar_output
        )
        # The hidden states must be gathered in rank-owner-sorted order
        # before computing the partial-vocab logits, so that the all-to-all
        # produces full-vocab logits for just the locally-owned requests.
        sample_hidden_states = hidden_states[sorted_logits_indices]
        local_logits = self.model.compute_logits_local(sample_hidden_states)
        logits = _model_runner.all_to_all_logits(local_logits, shard_metadata)
        logits = logits[:, : self.vocab_size]
    else:
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)

    invalid_drafts = None
    if grammar_output is not None:
        # Apply grammar bitmask to the logits in-place.
        assert self.structured_outputs_worker is not None
        self.structured_outputs_worker.apply_grammar_bitmask(
            logits,
            input_batch,
            grammar_output.structured_output_request_ids,
            grammar_output.grammar_bitmask,
        )
        invalid_drafts = grammar_invalid_drafts(
            input_batch,
            grammar_output.structured_output_request_ids,
            getattr(grammar_output, "num_acceptable_drafts", None),
        )

    sampler_output: SamplerOutput | None
    if input_batch.num_reqs == 0:
        # This rank owns no requests this step. It contributes an
        # all-padding block to the gather below.
        sampler_output = None
    elif input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
        assert self.sampler is not None
        sampler_output = self.sampler(logits, input_batch)
    else:
        # Rejection sampling for spec decoding.
        assert self.rejection_sampler is not None
        assert self.speculator is not None
        sampler_output = self.rejection_sampler(
            logits,
            input_batch,
            # Draft logits are needed for probabilistic rejection sampling.
            self.speculator.draft_logits,
            invalid_drafts,
        )

    if shard_metadata is not None:
        # Gather the sharded sampler outputs from the TP ranks into a single
        # sampler output.
        assert self.sampler is not None
        sampler_output = _model_runner.gather_sampler_output(
            sampler_output,
            shard_metadata,
            device=self.device,
            global_batch=global_input_batch,
            local_batch=input_batch,
            gather_num_nans=self.sampler.compute_nans,
            logprobs_dims=self.sampler.get_logprobs_dims(
                global_input_batch.idx_mapping_np,
                # Rejection sampler does not return logprob token ids.
                include_token_ids=(global_input_batch.num_draft_tokens == 0 or self.rejection_sampler is None),
            ),
        )

    assert sampler_output is not None
    return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected


def _verify(
    self,
    logits: torch.Tensor,
    draft_logits: torch.Tensor | None,
    draft_sampled: torch.Tensor,
    pos: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    verify_draft_sampled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    processed_logits = self.sampler.apply_sampling_params(
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        pos,
        draft_sampled,
        expanded_local_pos,
    )
    sampled, num_sampled = _rejection.rejection_sample(
        processed_logits,
        draft_logits,
        verify_draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        self.sampler.sampling_states.temperature.gpu,
        self.sampler.sampling_states.seeds.gpu,
        self.num_speculative_steps,
        self.synthetic_conditional_rates,
        use_fp64=self.sampler.use_fp64_gumbel,
        use_block_verification=self.use_block_verification,
        **self._watermarking_kwargs(draft_sampled, expanded_idx_mapping, expanded_local_pos),
    )
    return processed_logits, sampled, num_sampled


def _verify_in_chunks(
    self,
    logits: torch.Tensor,
    input_batch: InputBatch,
    draft_logits: torch.Tensor | None,
    draft_sampled: torch.Tensor,
    pos: torch.Tensor,
    max_chunk_logits: int,
    max_num_logprobs: int,
    verify_draft_sampled: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, LogprobsTensors | None]:
    if verify_draft_sampled is None:
        verify_draft_sampled = draft_sampled
    cu_num_logits_np = input_batch.cu_num_logits_np
    use_processed_logits = self.sampler.logprobs_mode in _rejection.PROCESSED_LOGPROBS_MODES
    num_reqs = input_batch.num_reqs

    if logits.shape[0] <= max_chunk_logits:
        # One chunk covers the batch. Adaptive verification compacts the logits
        # without updating cu_num_logits_np (it keeps the pre-compacted layout),
        # so the stale sums must not pick chunk boundaries; its budget cap
        # guarantees the compacted batch always lands here.
        request_chunks: Iterable[tuple[int, int]] = ((0, num_reqs),)
    else:
        assert not self.enable_adaptive_verification
        request_chunks = _rejection._iter_request_chunks(cu_num_logits_np, max_chunk_logits)

    sampled_chunks: list[torch.Tensor] = []
    num_sampled_chunks: list[torch.Tensor] = []
    logprobs_chunks: list[LogprobsTensors] = []

    for start, end in request_chunks:
        lo = int(cu_num_logits_np[start])
        hi = int(cu_num_logits_np[end])
        chunk_cu_num_logits_np = cu_num_logits_np[start : end + 1] - lo
        chunk_cu_num_logits = input_batch.cu_num_logits[start : end + 1] - lo
        # draft_logits uses persistent request-state indices and stays global.
        processed_logits, sampled, num_sampled = self._verify(
            logits[lo:hi],
            draft_logits,
            draft_sampled[lo:hi],
            pos[lo:hi],
            chunk_cu_num_logits,
            input_batch.idx_mapping[start:end],
            input_batch.idx_mapping_np[start:end],
            input_batch.expanded_idx_mapping[lo:hi],
            input_batch.expanded_local_pos[lo:hi],
            verify_draft_sampled[lo:hi],
        )
        chunk_logprobs = self._get_logprobs_tensors(
            sampled,
            num_sampled,
            processed_logits if use_processed_logits else logits[lo:hi],
            chunk_cu_num_logits,
            chunk_cu_num_logits_np,
            max_num_logprobs,
        )
        if chunk_logprobs is not None:
            logprobs_chunks.append(chunk_logprobs)
        del processed_logits
        sampled_chunks.append(sampled)
        num_sampled_chunks.append(num_sampled)

    if len(sampled_chunks) == 1:
        logprobs_tensors = logprobs_chunks[0] if logprobs_chunks else None
        return sampled_chunks[0], num_sampled_chunks[0], logprobs_tensors

    logprobs_tensors = None
    if logprobs_chunks:
        expanded_logits = logits.shape[0] != input_batch.num_reqs
        logprobs_tensors = LogprobsTensors.cat(
            logprobs_chunks,
            cu_num_generated_tokens=(cu_num_logits_np.tolist() if expanded_logits else None),
        )

    sampled = torch.cat(sampled_chunks)
    num_sampled = torch.cat(num_sampled_chunks)
    return sampled, num_sampled, logprobs_tensors


def sample_with_grammar(
    self,
    logits: torch.Tensor,
    input_batch: InputBatch,
    draft_logits: torch.Tensor | None = None,
    invalid_drafts: torch.Tensor | None = None,
) -> SamplerOutput:
    # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
    # that num_nans is computed before applying penalties and temperature.
    num_nans = _rejection.get_num_nans(logits) if self.sampler.compute_nans else None

    draft_sampled = input_batch.input_ids[input_batch.logits_indices]
    pos = input_batch.positions[input_batch.logits_indices]

    # Preserve real IDs for penalties/watermarking; only verification sees -1.
    verify_draft_sampled = draft_sampled
    if invalid_drafts is not None:
        verify_draft_sampled = draft_sampled.masked_fill(invalid_drafts, -1)

    max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(input_batch.idx_mapping_np)
    chunk_logit_limit = _rejection.get_max_chunk_logits(logits.shape[1])
    sampled, num_sampled, logprobs_tensors = self._verify_in_chunks(
        logits,
        input_batch,
        draft_logits,
        draft_sampled,
        pos,
        chunk_logit_limit,
        max_num_logprobs,
        verify_draft_sampled,
    )

    num_sampled, num_rejected = _rejection.get_num_sampled_and_rejected(
        num_sampled,
        input_batch.seq_lens,
        input_batch.cu_num_logits,
        input_batch.idx_mapping,
        self.sampler.req_states.prefill_len.gpu,
    )

    return SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=logprobs_tensors,
        num_nans=num_nans,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
    )


# Rebind existing classes, so already-imported references also receive the fix.
# The 310P runner's own sample/__call__ overrides remain unchanged.
_batch_shard._shard_grammar_output = shard_grammar_output
_model_runner.GPUModelRunner.sample = sample
_rejection.RejectionSampler._verify = _verify
_rejection.RejectionSampler._verify_in_chunks = _verify_in_chunks
_rejection.RejectionSampler.__call__ = sample_with_grammar
