# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""A layer that samples the next tokens from the model's outputs."""
from typing import Dict, List, Optional
import torch_npu
import torch
import os

from vllm.platforms import current_platform
from vllm.model_executor.layers.sampler import (
    _apply_min_p, _build_sampler_output, get_logprobs,
    Sampler, _apply_min_tokens_penalty, SamplerOutput,
    SampleResultArgsType, SampleReturnType,
    SampleResultsDictType, SampleMetadataType, MultinomialSamplesType, get_pythonized_sample_results
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.sampling_metadata import SamplingTensors
from vllm.model_executor.layers.rejection_sampler import RejectionSampler as RejectionSamplerGPU
from vllm.v1.sample.rejection_sampler import RejectionSampler as RejectionSamplerV1
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler as TopKTopPSampler
from vllm.v1.sample.sampler import Sampler as SamplerV1
from vllm.v1.outputs import SamplerOutput as SamplerOutputV1
from vllm.sampling_params import SamplingType
from vllm.sequence import Logprob, VLLM_INVALID_TOKEN_ID

from omni.models.common.config.model_config import model_extra_config


FP32_EPS = 2 ** -24
USE_SORT_OP_MIN_BS = 2
USE_SORT_OP_MAX_BS = 48
flashinfer_top_k_top_p_sampling = None
UNINITIALIZED_CACHED_K_NUM = -1


def _modify_greedy_probs_inplace(logprobs: torch.Tensor, probs: torch.Tensor,
                                 sample_indices: torch.Tensor,
                                 greedy_samples: torch.Tensor) -> None:
    if probs.shape[0] == sample_indices.shape[0]:
        probs.fill_(0)
    else:
        FLOAT32_ZERO_TENSOR = None
        if FLOAT32_ZERO_TENSOR is None:
            FLOAT32_ZERO_TENSOR = torch.tensor(0, dtype=torch.float32, device=current_platform.device_type)
        probs[sample_indices, :] = FLOAT32_ZERO_TENSOR
    FLOAT32_ONE_TENSOR = None
    if FLOAT32_ONE_TENSOR is None:
        FLOAT32_ONE_TENSOR = torch.tensor(1, dtype=torch.float32, device=current_platform.device_type)
    probs[sample_indices, greedy_samples] = FLOAT32_ONE_TENSOR

def _get_logprobs_adapter(need_log_probs, fully_greedy_mode, slice_indexes, logprobs, sampling_metadata,
                          sample_results):
    bs = logprobs.shape[0]
    if need_log_probs or bs != len(sample_results):
        return get_logprobs(
            logprobs, sampling_metadata, sample_results)
    prompt_logprobs = []
    sample_logprobs = []
    if fully_greedy_mode:
        for i in range(bs):
            prompt_logprobs.append(None)
            sample_id = sample_results[i][0][0]
            sample_logprobs.append(
                [
                    {
                        sample_id: Logprob(logprob=0, rank=1)
                    }
                ]
            )
    elif slice_indexes is not None:
        # slice_indexes is not None when do_top_k
        for i in range(len(sample_results)):
            prompt_logprobs.append(None)
            sample_id = sample_results[i][0][0]
            # the real rank is:  rank = (slice_indexes[i] == sample_id).nonzero()[0][0].cpu() + 1
            # the real logprob is : logprob = logprobs[i][rank - 1].cpu()
            # rank * logprob is not need if not need_log_probs, use fake rank to optim time cost
            dummy_rank = 1
            dummy_logprob = 0
            sample_logprobs.append(
                [
                    {
                        sample_id: Logprob(logprob=dummy_logprob, rank=dummy_rank)
                    }
                ]
            )
    else:
        for i in range(bs):
            prompt_logprobs.append(None)
            sample_id = sample_results[i][0][0]
            # the real rank is : rank = torch.sum(logprobs[0] > logprobs[0][sample_id]).item() + 1
            # rank is not need if we not need_log_probs, use fake rank to optim time cost
            rank = 1
            sample_logprobs.append(
                [
                    {
                        sample_id: Logprob(logprob=logprobs[i][sample_id].cpu(), rank=rank)
                    }
                ]
            )
    return prompt_logprobs, sample_logprobs

def _get_greedy_token(probs, logprobs, prob_indexes, long_sample_indices, include_gpu_probs_tensor,
                      modify_greedy_probs, sampled_token_ids_tensor):
    greedy_samples = torch.argmax(logprobs[long_sample_indices],
                                  dim=-1)
    # Adapt: use gather when prob_indexes is not None.
    greedy_samples = gather_tokens(greedy_samples, prob_indexes, long_sample_indices)
    # Adapt end.

    if include_gpu_probs_tensor:
        # Store sampled tokens in output tensor.
        sampled_token_ids_tensor[
            long_sample_indices] = greedy_samples.unsqueeze(-1)

    if modify_greedy_probs:
        # If required, modify the probabilities such that sampling from
        # the modified distribution would always sample the argmax
        # token id.
        _modify_greedy_probs_inplace(logprobs, probs,
                                     long_sample_indices,
                                     greedy_samples)
    return greedy_samples

def gather_tokens(select_tokens, prob_indexes, sample_indices):
    if prob_indexes is not None:
        if select_tokens.dim() == 1:
            select_tokens = select_tokens.unsqueeze(1)
            select_tokens = torch.gather(prob_indexes[sample_indices], dim=-1, index=select_tokens)
            select_tokens = select_tokens.squeeze(1)
        else:
            select_tokens = torch.gather(prob_indexes[sample_indices], dim=-1, index=select_tokens)
    return select_tokens

def _check_top_ks(sampling_tensors, do_top_p_top_k):
    if not sampling_tensors:
        return False
    # The original coda statement "sampling_tensors.top_ks is None" is covered in ./vllm_npu/common/model_executor/sampling_metadata.py
    elif isinstance(sampling_tensors.top_ks, torch.Tensor):
        if not sampling_tensors.top_ks.numel():
            return False
    return do_top_p_top_k

def _apply_penalties(logits: torch.Tensor, prompt_tokens_tensor: torch.Tensor,
                     output_tokens_tensor: torch.Tensor,
                     presence_penalties: torch.Tensor,
                     frequency_penalties: torch.Tensor,
                     repetition_penalties: torch.Tensor,
                     do_repetition_penalties,
                     do_presence_penalties,
                     do_frequency_penalties
                     ) -> torch.Tensor:
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = _get_bin_counts_and_mask(prompt_tokens_tensor, vocab_size,
                                              num_seqs)
    output_bin_counts, output_mask = _get_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs)
    if do_repetition_penalties:
        repetition_penalties = (repetition_penalties - 1)[:, None].repeat(1, vocab_size)
        repetition_penalties = repetition_penalties * (prompt_mask | output_mask) + 1
        logits = torch.where(logits > 0, logits / repetition_penalties,
                             logits * repetition_penalties)
    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    if do_frequency_penalties:
        logits -= frequency_penalties.unsqueeze_(dim=1) * output_bin_counts
    if do_presence_penalties:
        logits -= presence_penalties.unsqueeze_(dim=1) * output_mask
    return logits

def _apply_top_k_top_p_faster(
        logits: torch.Tensor,
        top_ps: List[float],
        top_ks: List[int],
) -> (torch.Tensor, torch.Tensor):
    # Apply top-k.
    bs = logits.shape[0]
    max_k = top_ks.max()
    # 'top_k' operate performance is worse than 'sort' is some case
    # experiment on top_k and sort, sort is better with batch in 2~48
    if USE_SORT_OP_MIN_BS <= bs <= USE_SORT_OP_MAX_BS:
        logits_sort, logits_idx = torch.sort(logits, dim=-1, descending=True)
        logits_sort = logits_sort[:, :max_k].float()
        logits_idx = logits_idx[:, :max_k]
    else:
        logits_sort, logits_idx = torch.topk(logits.float(), max_k)
    top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
    top_k_mask = top_k_mask >= top_ks.unsqueeze(dim=1)
    logits_sort = logits_sort.masked_fill(top_k_mask, -float("inf"))

    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
    logits_sort = logits_sort.masked_fill(top_p_mask, -float("inf"))

    # Return partial tensor with index.
    return logits_sort, logits_idx, top_p_mask
  
def _need_log_probs(sampling_metadata, include_gpu_probs_tensor):
    need_log_probs = False
    for seq_group in sampling_metadata.seq_groups:
        if seq_group.is_prompt and seq_group.sampling_params.prompt_logprobs is not None:
            need_log_probs = True
            break
        if seq_group.sampling_params.logprobs is not None and seq_group.sampling_params.logprobs > 0:
            need_log_probs = True
            break
        if seq_group.sampling_params.n > 1:
            need_log_probs = True
            break
        if include_gpu_probs_tensor:
            need_log_probs = True
    return need_log_probs

def _sample(
        probs: torch.Tensor,
        logprobs: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        sampling_tensors: SamplingTensors,
        include_gpu_probs_tensor: bool,
        modify_greedy_probs: bool,
        prob_indexes=None
) -> SampleReturnType:
    '''Torch-oriented _sample() implementation.

    Single-step scheduling:
    * Perform GPU-side sampling computation
    * Immediately Pythonize sampling result

    Multi-step scheduling:
    * Perform GPU-side sampling computation
    * Defer Pythonization & preserve GPU-side
      tensors required for Pythonization
    '''

    categorized_seq_group_ids: Dict[SamplingType,
                                    List[int]] = {t: []
                                                  for t in SamplingType}
    categorized_sample_indices = sampling_metadata.categorized_sample_indices
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        sampling_params = seq_group.sampling_params
        sampling_type = sampling_params.sampling_type
        categorized_seq_group_ids[sampling_type].append(i)

    sample_results_dict: SampleResultsDictType = {}
    sample_metadata: SampleMetadataType = {}
    multinomial_samples: MultinomialSamplesType = {}
    greedy_samples: Optional[torch.Tensor] = None
    beam_search_logprobs: Optional[torch.Tensor] = None

    # Create output tensor for sampled token ids.
    if include_gpu_probs_tensor:
        sampled_token_ids_tensor = torch.full((logprobs.shape[0], 1),
                                               VLLM_INVALID_TOKEN_ID,
                                               dtype=torch.long,
                                               device=logprobs.device)
    else:
        sampled_token_ids_tensor = None

    # Counterintiutively, having two loops here is actually faster.
    # The first loop can run without waiting on GPU<->CPU sync.
    for sampling_type in SamplingType:
        sample_indices = categorized_sample_indices[sampling_type]
        num_tokens = len(sample_indices)
        if num_tokens == 0:
            continue

        long_sample_indices = sample_indices.long()
        seq_group_id = categorized_seq_group_ids[sampling_type]
        seq_groups = [sampling_metadata.seq_groups[i] for i in seq_group_id]
        sample_metadata[sampling_type] = (seq_group_id, seq_groups)
        if sampling_type == SamplingType.GREEDY:
            greedy_samples = _get_greedy_token(probs, logprobs, prob_indexes, long_sample_indices,
                                               include_gpu_probs_tensor, modify_greedy_probs, sampled_token_ids_tensor)
        elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            max_n_in_batch = 1
            for seq_group in seq_groups:
                if seq_group.is_prompt:
                    sampling_params = seq_group.sampling_params
                    max_n_in_batch = max(max_n_in_batch, sampling_params.n)
            seq_groups_arg = (None if sampling_type == SamplingType.RANDOM else
                              seq_groups)

            multinomial_samples[sampling_type] = _multinomial(
                probs[long_sample_indices],
                max_n_in_batch,
                seq_groups=seq_groups_arg)
            # Adapt: use gather when prob_indexes is not None.
            multinomial_samples[sampling_type] = gather_tokens(multinomial_samples[sampling_type],
                                                               prob_indexes, long_sample_indices)
            # Adapt end.

            if sampled_token_ids_tensor is not None:
                # Store sampled tokens in output tensor.
                sampled_token_ids_tensor[long_sample_indices] = \
                    multinomial_samples[sampling_type].to(torch.long)

        elif sampling_type == SamplingType.BEAM:
            beam_search_logprobs = logprobs[sample_indices]
        else:
            raise ValueError(f"Unsupported sampling type: {sampling_type}")

    # Encapsulate arguments for computing Pythonized sampler
    # results, whether deferred or otherwise.
    maybe_deferred_args = SampleResultArgsType(
        sampling_metadata=sampling_metadata,
        sample_metadata=sample_metadata,
        multinomial_samples=multinomial_samples,
        greedy_samples=greedy_samples,
        beam_search_logprobs=beam_search_logprobs,
        sample_results_dict=sample_results_dict)

    if not sampling_metadata.skip_sampler_cpu_output:
        # GPU<->CPU sync happens here.
        # This also converts the sampler output to a Python object.
        # Return Pythonized sampler result & sampled token ids
        return get_pythonized_sample_results(
            maybe_deferred_args), sampled_token_ids_tensor
    else:
        # Defer sampler result Pythonization; return deferred
        # Pythonization args & sampled token ids
        return (
            maybe_deferred_args,
            sampled_token_ids_tensor,
        )

class AscendSampler(Sampler):

    def _init_sampling_tensors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ):
        """The goal here is to reuse sampling tensors between similar decode
        runs. This is possible because sampling logic does not change between
        decodes of the same sequences.
        """
        _, vocab_size = logits.shape

        # First free any existing stored sampling tensors.
        # This is necessary because some sampling tensors may
        # have pinned memory.
        self._sampling_tensors = None

        # Initialize new sampling tensors
        (sampling_tensors, do_temperature, do_penalties, do_repetition_penalties,
         do_presence_penalties, do_frequency_penalties, do_top_p_top_k, do_min_p) = \
            SamplingTensors.from_sampling_metadata(sampling_metadata, vocab_size, logits.device, logits.dtype)

        self._sampling_tensors = sampling_tensors
        self._do_temperature = do_temperature
        self._do_penalties = do_penalties
        self._do_repetition_penalties = do_repetition_penalties
        self._do_presence_penalties = do_presence_penalties
        self._do_frequency_penalties = do_frequency_penalties
        self._do_top_p_top_k = do_top_p_top_k
        self._do_min_p = do_min_p


    def forward(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """
        Single-step scheduling:
        * Perform GPU-side sampling computation & compute
          GPU-side logprobs tensor
        * Pythonize sampling result & logprobs tensor

        Multi-step scheduling:
        * Perform GPU-side sampling computation & compute
          GPU-side logprobs tensor
        * Defer Pythonization of sampling result & logprobs
          tensor
        * Encapsulate arguments required for deferred Pythonization
          in the :class:`SamplerOutput` structure

        Args:
            logits: (num_tokens, vocab_size).
            sampling_metadata: Metadata for sampling.
        """
        _, vocab_size = logits.shape

        logits = _apply_min_tokens_penalty(logits, sampling_metadata)

        # Adapt start: Disable tensor generation in pure greedy mode
        fully_greedy_mode = True
        for i, seq_group in enumerate(sampling_metadata.seq_groups):
            sampling_params = seq_group.sampling_params
            sampling_type = sampling_params.sampling_type
            do_penalty = not (sampling_params.presence_penalty == 0.0 and
                              sampling_params.frequency_penalty == 0.0 and
                              sampling_params.repetition_penalty == 1.0)
            if sampling_type != SamplingType.GREEDY or do_penalty:
                fully_greedy_mode = False
                break

        if fully_greedy_mode:
            sampling_tensors = None
            do_temperature = False
            do_penalties = False
            do_top_p_top_k = False
            do_min_p = False
        else:
            # Prepare sampling tensors with pinned memory to avoid blocking.
            if not sampling_metadata.reuse_sampling_tensors:
                self._init_sampling_tensors(logits, sampling_metadata)
            elif self._do_penalties:
                # In this case, the sampling tensors logic depends on
                # "output_tokens" of a sequence. As a result, we cannot
                # reuse sampling tensors, since "output_tokens" changes
                # between decode runs.
                self._init_sampling_tensors(logits, sampling_metadata)
            sampling_tensors = self._sampling_tensors
            do_temperature = self._do_temperature
            do_penalties = self._do_penalties
            do_repetition_penalties = self._do_repetition_penalties
            do_presence_penalties = self._do_presence_penalties
            do_frequency_penalties = self._do_frequency_penalties
            do_top_p_top_k = self._do_top_p_top_k
            do_min_p = self._do_min_p
            do_top_p_top_k = _check_top_ks(sampling_tensors, do_top_p_top_k)

        # Adapt end: Disable tensor generation in pure greedy mode
        # Apply presence and frequency penalties.
        if do_penalties:
            logits = _apply_penalties(logits, sampling_tensors.prompt_tokens,
                                      sampling_tensors.output_tokens,
                                      sampling_tensors.presence_penalties,
                                      sampling_tensors.frequency_penalties,
                                      sampling_tensors.repetition_penalties,
                                      do_repetition_penalties,
                                      do_presence_penalties,
                                      do_frequency_penalties)

        # Use in-place division to avoid creating a new tensor.
        if do_temperature:
            logits.div_(sampling_tensors.temperatures.unsqueeze(dim=1))
        # Adapt: using _apply_top_k_top_p_faster instead of _apply_top_k_top_p
        if do_top_p_top_k:
            logits, slice_indexes, top_p_mask = _apply_top_k_top_p_faster(logits, sampling_tensors.top_ps,
                                                                          sampling_tensors.top_ks)
        else:
            slice_indexes = None
        # Adapt end
        if do_min_p:
            logits = _apply_min_p(logits, sampling_tensors.min_ps)

        # adapt optim in greedy case
        need_log_probs = _need_log_probs(sampling_metadata, self.include_gpu_probs_tensor)
        logprobs, probs = self.get_probs_and_logprobs(logits, fully_greedy_mode and not need_log_probs)
        # adapt optim in greedy case END

        # Sample the next tokens.
        maybe_deferred_sample_results, maybe_sampled_tokens_tensor = _sample(
            probs,
            logprobs,
            sampling_metadata,
            sampling_tensors,
            include_gpu_probs_tensor=self.include_gpu_probs_tensor,
            modify_greedy_probs=self._should_modify_greedy_probs_inplace,
            prob_indexes=slice_indexes
        )

        # Put logp prob in the correct location when do_top_p_top_k
        if do_top_p_top_k and need_log_probs:
            src = torch.arange(slice_indexes.shape[0], device=slice_indexes.device).unsqueeze(-1).expand_as(
                slice_indexes)
            indices = src * vocab_size + slice_indexes
            indices = indices[~top_p_mask].flatten()
            logprobs_ori = torch.full((logprobs.shape[0], vocab_size), float('-inf'), dtype=logprobs.dtype,
                                      device=logprobs.device)
            cast_logprobs_ori = logprobs_ori.view(-1)
            torch_npu.npu_scatter_nd_update_(cast_logprobs_ori, indices.unsqueeze(1), logprobs[~top_p_mask])
            logprobs = logprobs_ori

            if self.include_gpu_probs_tensor:
                probs_ori = torch.full((probs.shape[0], vocab_size), 0.0, dtype=probs.dtype, device=probs.device)
                cast_probs_ori = probs_ori.view(-1)
                torch_npu.npu_scatter_nd_update_(cast_probs_ori, indices.unsqueeze(1), probs[~top_p_mask])
                probs = probs_ori

        if self.include_gpu_probs_tensor:
            # maybe_sampled_tokens_tensor is generated in function `_sample` according to `include_gpu_probs_tensor`.
            on_device_tensors = (probs, logprobs, maybe_sampled_tokens_tensor)
        else:
            # Since Pythonization has already happened, don't preserve
            # GPU-side tensors.
            on_device_tensors = None

        # Get the logprobs query results.
        prompt_logprobs = None
        sample_logprobs = None
        if not sampling_metadata.skip_sampler_cpu_output:
            # Pythonize logprobs now (GPU -> CPU); do not defer.
            prompt_logprobs, sample_logprobs = _get_logprobs_adapter(need_log_probs, fully_greedy_mode, slice_indexes,
                                                                     logprobs, sampling_metadata,
                                                                     maybe_deferred_sample_results)

        return _build_sampler_output(
            maybe_deferred_sample_results,
            sampling_metadata,
            prompt_logprobs,
            sample_logprobs,
            on_device_tensors=on_device_tensors,
            skip_sampler_cpu_output=sampling_metadata.skip_sampler_cpu_output)

    def get_probs_and_logprobs(self, logits, not_need_softmax):
        if not_need_softmax:
            if self.include_gpu_probs_tensor:
                logprobs = logits.float()
            else:
                logprobs = logits
            probs = logprobs
        else:
            # We use float32 for probabilities and log probabilities.
            # Compute the probabilities.
            probs = torch.softmax(logits, dim=-1, dtype=torch.float)
            # Compute the log probabilities.
            # Use log_softmax to ensure numerical stability.
            logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        return logprobs, probs

class RejectionSampler(RejectionSamplerGPU):

    def __init__(self, *args, **kwargs):
        super(RejectionSampler, self).__init__(*args, **kwargs)
        self.int64_neg_one = torch.tensor(-1, device=current_platform.device_type, dtype=self.token_id_dtype)
        self.cached_indices = None

        self.cached_k_tensor = None
        self.cached_k = UNINITIALIZED_CACHED_K_NUM


    def _get_accepted(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
        seeded_seqs: Optional[Dict[int, torch.Generator]],
    ) -> torch.Tensor:

        batch_size, k, _ = draft_probs.shape

        uniform_rand = self._create_uniform_samples(seeded_seqs, batch_size,
                                                    k - 1, target_probs.device)

        # adapt: replace index_select with gather.
        draft_token_ids = draft_token_ids.view(batch_size, k, 1)
        selected_draft_probs = torch.gather(draft_probs, dim=-1, index=draft_token_ids).view(batch_size, k)
        selected_target_probs = torch.gather(target_probs, dim=-1, index=draft_token_ids).view(batch_size, k)
        selected_target_probs.div_(selected_draft_probs).clamp_max_(1)

        accepted = uniform_rand < selected_target_probs
        return accepted


    def _get_recovered_probs(
            self,
            target_probs: torch.Tensor,  # [k, vocab_size]
            draft_probs: torch.Tensor,  # [k, vocab_size]
    ) -> torch.Tensor:

        _, k, _ = draft_probs.shape

        # adapt: use inplace ops.
        target_probs.sub_(draft_probs).clamp_min_(self._smallest_positive_value)
        recovered_probs = target_probs / torch.sum(target_probs, dim=-1).view(-1, k, 1)

        return recovered_probs

    def _create_output(
            self,
            accepted: torch.Tensor,  # [batch_size, k]
            substitute_token_ids: torch.Tensor,  # [batch_size, k]
            draft_token_ids: torch.Tensor,  # [batch_size, k]
            bonus_token_ids: torch.Tensor,  # [batch_size]
    ) -> torch.Tensor:

        batch_size, k = substitute_token_ids.shape
        bonus_token_ids = bonus_token_ids.squeeze()
        # Determine the index of the first False value for each row.
        accepted_eqeal_zero_mask = accepted == 0
        limits = accepted_eqeal_zero_mask.max(1).indices
        # Adapt: opt "limits[~(accepted == 0).any(1)] = k"
        mask = accepted_eqeal_zero_mask.any(1)
        if self.cached_k_tensor is None or self.cached_k != k:
            self.cached_k_tensor = torch.tensor(k, dtype=limits.dtype, device=limits.device)
            self.cached_k = k
        limits = torch.where(mask, limits, self.cached_k_tensor)

        # Create masks using the indices.
        if self.cached_indices is None or self.cached_indices.shape[1] != k:
            self.cached_indices = torch.arange(k, device=accepted.device).unsqueeze(0)
        accepted_mask = self.cached_indices < limits.unsqueeze(1)
        after_false_mask = self.cached_indices == limits.unsqueeze(1)

        # Create an extended output tensor
        output_with_bonus_tokens = torch.full(
            (batch_size, k + self._num_bonus_tokens),
            fill_value=-1,
            dtype=self.token_id_dtype,
            device=accepted.device)
        output = output_with_bonus_tokens[:, :k]

        # Fill in the first k columns of the output tensor using masks and data
        # tensors.
        # Adapt: remove index select
        torch.where(accepted_mask,
                    draft_token_ids,
                    self.int64_neg_one,
                    out=output)

        # Fill the last column.
        # We check output directly as accepted may have True values inconsistent
        # with causal acceptance.
        # Adapt: avoid mem copy
        output_with_bonus_tokens[:, -1] = torch.where(output[:, -1] != self.int64_neg_one,
                                                      bonus_token_ids, self.int64_neg_one)

        # Fill the recovered token ids.
        output.mul_(~after_false_mask).add_(
            substitute_token_ids.mul(after_false_mask))

        # Adapt: disable log metric when disable_logprobs is True.
        if getattr(self, "enable_spec_metric", True):
            self.num_accepted_tokens += accepted.sum()
            self.num_emitted_tokens += (output_with_bonus_tokens != -1).sum()
            self.num_draft_tokens += batch_size * k

        return output_with_bonus_tokens

def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits, None

        # Avoid sorting vocab for top-k only case.
        return apply_top_k_only(logits, k), None

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    return logits_sort, logits_idx


def apply_top_k_only(
    logits: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits.masked_fill_(logits < top_k_mask, -float("inf"))
    return logits

def random_sample(
    probs: torch.Tensor,
    idx: Optional[torch.Tensor],
    generators: dict[int, torch.Generator],
    stream
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    with torch_npu.npu.stream(stream) :
        q = torch.empty_like(probs)
        # NOTE(woosuk): To batch-process the requests without their own seeds,
        # which is the common case, we first assume that every request does
        # not have its own seed. Then, we overwrite the values for the requests
        # that have their own seeds.
        if len(generators) != probs.shape[0]:
            q.exponential_()
        if generators:
            # TODO(woosuk): This can be slow because we handle each request
            # one by one. Optimize this.
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.default_stream().wait_stream(stream)
    res = probs.div_(q).argmax(dim=-1).view(-1)
    if idx == None:
        return res
    else:
        return idx[torch.arange(res.shape[0]), res]
    
def generate_random_sequence(
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        stream
) -> torch.Tensor:
    with torch_npu.npu.stream(stream):
        q = torch.empty_like(logits, dtype=torch.float32)
        if len(generators) != logits.shape[0]:
            q.exponential_()
        if generators:
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.default_stream().wait_stream(stream)
    return q

class AscendTopKTopPSamplerV1(TopKTopPSampler):
    """
    Module that performs optional top-k and top-p filtering followed by
    weighted random sampling of logits.

    Implementations may update the logits tensor in-place.
    """

    def __init__(self):
        super().__init__()
        self.dsa_stream = torch_npu.npu.Stream()
    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        PyTorch-native implementation of top-k and top-p sampling.

        The logits tensor may be updated in-place.
        """
        use_npu_top_k_top_p_sample = os.environ.get("OMNI_USE_NPU_TOP_K_TOP_P_SAMPLE", False)
        if use_npu_top_k_top_p_sample == False or p is None or k is None:
            logits, idx = apply_top_k_top_p(logits, k, p)
            probs = logits.softmax(dim=-1, dtype=torch.float32)
            return random_sample(probs, idx, generators, self.dsa_stream)
        else:
            logits = logits.type(torch.bfloat16)
            p = p.type(torch.bfloat16)
            k = k.type(torch.int32)
            q = generate_random_sequence(logits, generators, self.dsa_stream)
            res = torch_npu.npu_top_k_top_p_sample(logits, k, p, q)
            return res[0]

class AscendSamplerV1(SamplerV1):
    def __init__(self):
        super().__init__()
        self.topk_topp_sampler = AscendTopKTopPSamplerV1()

class SimpleSampler(RejectionSamplerV1):
 
    def __init__(self, main_sampler, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.previous_frequency_penalties = []
        self.previous_repetition_penalties = [] 
        self.previous_presence_penalties = []
        self.main_sampler = main_sampler
        self.main_sampler.topk_topp_sampler = AscendTopKTopPSamplerV1()
        self.minus_one = None

    # TODO to support mixture case
    def forward(self, input_ids, logits, logits_indices, sampling_metadata, num_decodes, num_prefills):

        if num_decodes != 0 and num_prefills != 0:
            raise ("Chunked prefill is not supported in current version.")
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            raise ("Logprobs gathered is not supported in current version")
        if self.minus_one is None:
            # prepare const on npu
            self.minus_one = -torch.ones(1, 1, device=input_ids.device, dtype=input_ids.dtype)

        batch_size = num_decodes + num_prefills
        logits_indices = logits_indices.to(torch.int32)
        num_sampling_tokens_per_req = (logits_indices.numel() // batch_size)
        if self.main_sampler is None or sampling_metadata.all_greedy:
            forward_tokens = logits.argmax(dim=-1).to(dtype = input_ids.dtype, device=input_ids.device).view(batch_size, -1)
        else:
            start_indices = torch.arange(batch_size, device=logits.device) * num_sampling_tokens_per_req
            forward_tokens = torch.empty_like(logits_indices, dtype=input_ids.dtype, device=input_ids.device).view(batch_size, -1)
            for i in range(num_sampling_tokens_per_req):
                sampler_output = self.main_sampler(
                    logits=logits[start_indices],
                    sampling_metadata=sampling_metadata,
                )
                start_indices += 1
                forward_tokens[:, i] = sampler_output.sampled_token_ids.view(-1)
                sampler_output.sampled_token_ids = None

        # Create output buffer.
        # output_token_ids: 
        # if accepted [input_ids[-1], forward_tokens_result]
        # else [forward_tokens_result, -1]

        # all prefill
        if num_decodes == 0:
            last_accepted_index = torch.arange(batch_size, dtype=torch.int32, device = logits_indices.device)
            output_token_ids = forward_tokens.view(-1, 1)
            accepted_num = 0
        else:
            accepted =  input_ids[logits_indices].view(batch_size, -1)[:,1:] == forward_tokens.view(batch_size, -1)[:,:-1]  # bool [batch_size, 1]
            if model_extra_config.operator_opt_config.control_accept_rate >= 0 and model_extra_config.operator_opt_config.control_accept_rate <= 1:
                accepted = torch.empty_like(accepted, dtype=torch.float32).uniform_() < model_extra_config.operator_opt_config.control_accept_rate

            padding_zero = torch.zeros((batch_size, 1), dtype=torch.int32, device=input_ids.device)
            accepted_mask = accepted.to(dtype=torch.int32)
            accepted_mask = torch.cat((accepted_mask, padding_zero), dim=1)
            accepted_num = accepted_mask.argmin(dim=1).to(dtype=torch.int32)
            offset = torch.arange(num_sampling_tokens_per_req, device = accepted_num.device, dtype = torch.int32)
            output_token_ids = torch.where(offset[None, :] <= accepted_num[:, None], forward_tokens, self.minus_one)
            last_accepted_index = torch.arange(batch_size, device=input_ids.device, dtype=torch.int32) * num_sampling_tokens_per_req + accepted_num

        sampler_output = SamplerOutputV1(
            sampled_token_ids = output_token_ids,
            logprobs_tensors = None
        )

        return sampler_output, forward_tokens, last_accepted_index, accepted_num

def _multinomial(
    probs: torch.Tensor,
    num_samples: int,
    k: int,
    seeded_seqs: Dict[int, torch.Generator],
) -> torch.Tensor:

    if num_samples > 1:
        # This is equivalent to torch.repeat_interleaved (which also
        # forces a GPU<->CPU sync).
        probs = probs[:, None, :].expand(probs.shape[0], num_samples,
                                         probs.shape[1]).contiguous().view(
                                             -1, probs.shape[1])
    q = torch.empty_like(probs)
    if not seeded_seqs:
        q.exponential_(1.0)
    else:
        non_seeded_indices: List[int] = []
        start = 0
        for idx in range(len(q) // k):
            end = start + k
            generator = seeded_seqs.get(idx)
            if generator is None:
                non_seeded_indices.extend(list(range(start, end)))
            else:
                q[start:end].exponential_(1.0, generator=generator)
            start = end
        q[non_seeded_indices].exponential_(1.0)
    # adaptor: add FP32_EPS to avoid div zero
    q.add_(FP32_EPS)
    return probs.div_(q).argmax(dim=1).view(-1, num_samples)
