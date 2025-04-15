# SPDX-License-Identifier: Apache-2.0
"""A layer that samples the next tokens from the model's outputs."""
from typing import Optional

import torch
from vllm.model_executor.layers.sampler import (Sampler, SampleResultArgsType,
                                                SamplerOutput, _apply_min_p,
                                                _apply_min_tokens_penalty,
                                                _build_sampler_output, _sample,
                                                get_logprobs)
from vllm.model_executor.sampling_metadata import SamplingMetadata

from vllm_ascend.sample.ops.penalties import apply_penalties


class AscendSampler(Sampler):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        assert logits is not None
        _, vocab_size = logits.shape

        # Prepare sampling tensors with pinned memory to avoid blocking.
        if not sampling_metadata.reuse_sampling_tensors:
            self._init_sampling_tensors(logits, sampling_metadata)
        elif self._do_penalties:
            # In this case, the sampling tensors logic depends on
            # "output_tokens" of a sequence. As a result, we cannot
            # reuse sampling tensors, since "output_tokens" changes
            # between decode runs.
            self._init_sampling_tensors(logits, sampling_metadata)

        assert self._sampling_tensors is not None
        sampling_tensors = self._sampling_tensors
        do_penalties = self._do_penalties
        do_top_p_top_k = self._do_top_p_top_k
        do_min_p = self._do_min_p

        logits = _apply_min_tokens_penalty(logits, sampling_metadata)

        # Apply presence and frequency penalties.
        if do_penalties:
            logits = apply_penalties(logits, sampling_tensors.prompt_tokens,
                                     sampling_tensors.output_tokens,
                                     sampling_tensors.presence_penalties,
                                     sampling_tensors.frequency_penalties,
                                     sampling_tensors.repetition_penalties)

        # Use float32 to apply temperature scaling.
        # Use in-place division to avoid creating a new tensor.
        logits = logits.to(torch.float)
        logits.div_(sampling_tensors.temperatures.unsqueeze(dim=1))

        if do_top_p_top_k:
            logits = _apply_top_k_top_p_npu(logits, sampling_tensors.top_ps,
                                            sampling_tensors.top_ks)

        if do_min_p:
            logits = _apply_min_p(logits, sampling_tensors.min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        maybe_deferred_sample_results, maybe_sampled_tokens_tensor = _sample(
            probs,
            logprobs,
            sampling_metadata,
            sampling_tensors,
            include_gpu_probs_tensor=self.include_gpu_probs_tensor,
            modify_greedy_probs=self._should_modify_greedy_probs_inplace,
        )

        if self.include_gpu_probs_tensor:
            assert maybe_sampled_tokens_tensor is not None
            on_device_tensors = (probs, logprobs, maybe_sampled_tokens_tensor)
        else:
            on_device_tensors = None

        # Get the logprobs query results.
        prompt_logprobs = None
        sample_logprobs = None
        if not sampling_metadata.skip_sampler_cpu_output:
            assert not isinstance(maybe_deferred_sample_results,
                                  SampleResultArgsType)
            prompt_logprobs, sample_logprobs = get_logprobs(
                logprobs, sampling_metadata, maybe_deferred_sample_results)

        return _build_sampler_output(
            maybe_deferred_sample_results,
            sampling_metadata,
            prompt_logprobs,
            sample_logprobs,
            on_device_tensors=on_device_tensors,
            skip_sampler_cpu_output=sampling_metadata.skip_sampler_cpu_output)


def _apply_top_k_top_p_npu(
    logits: torch.Tensor,
    p: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """Apply top-k and top-p optimized for NPU.

    This algorithm avoids using torch.scatter which is time-consuming on NPU.
    """
    batch_size, vocab_size = logits.shape
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    boundary = logits_sort.gather(1, (vocab_size - k).unsqueeze(dim=1))
    top_k_mask = logits_sort < boundary
    logits_sort.masked_fill_(top_k_mask, -float("inf"))
    cutoff = top_k_mask.sum(dim=-1).min()
    probs_sort = logits_sort.softmax(dim=-1)[:, cutoff:]
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = probs_sum > 1 - p.unsqueeze(dim=1)
    top_p_mask[:, -1] = True
    strides = torch.arange(0,
                           batch_size * vocab_size,
                           vocab_size,
                           device=logits.device)
    flatten_idx = logits_idx[:, cutoff:] + strides.unsqueeze(dim=1)
    valid_idx = torch.masked_select(flatten_idx, top_p_mask)
    logits_flatten = logits.flatten()
    valid_logits = torch.index_select(logits_flatten, 0, valid_idx)
    logits = torch.empty_like(logits_flatten).fill_(-float("inf"))
    logits[valid_idx] = valid_logits
    return logits.reshape(batch_size, vocab_size)
