# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import torch
from vllm.triton_utils import HAS_TRITON, triton
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (
    GREEDY_TEMPERATURE,
    MAX_SPEC_LEN,
    PLACEHOLDER_TOKEN_ID,
    RejectionSampler,
    generate_uniform_probs,
)

from vllm.distributed.parallel_state import get_tp_group
from vllm_ascend.ops.triton.reject_sample import (
    cal_grid_and_block_size,
    expand_triton,
    rejection_greedy_sample_with_triton,
    rejection_random_sample_block_verify_kernel,
    rejection_random_sample_kernel,
    sample_recovered_tokens_kernel,
)
from vllm_ascend.sample.penalties import apply_all_penalties
from vllm_ascend.sample.sampler import apply_top_k_top_p

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, SamplerOutput
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words_with_drafts
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm_ascend.ascend_config import get_ascend_config

class AscendRejectionSampler(RejectionSampler):
    """Ascend-optimized rejection sampler for speculative decoding.

    This class overrides key methods from the base RejectionSampler to provide
    Ascend-specific optimizations:
    - Optimized greedy sampling with reduced communication
    - Distributed top-k/top-p sampling
    - Efficient batch expansion operations
    """
    
    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
        repeat_indices: torch.Tensor,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        if sampling_metadata.no_penalties:
            return logits
        
        """Use Triton-Ascend penalties on NPU when Triton is available; else vLLM default."""
        if not HAS_TRITON:
            return Sampler.apply_penalties(logits, sampling_metadata, output_token_ids)

        assert sampling_metadata.prompt_token_ids is not None
        prompt_token_ids = sampling_metadata.prompt_token_ids[repeat_indices]
        presence_penalties = sampling_metadata.presence_penalties[repeat_indices]
        frequency_penalties = sampling_metadata.frequency_penalties[repeat_indices]
        repetition_penalties = sampling_metadata.repetition_penalties[repeat_indices]
        return apply_all_penalties(
            logits,
            prompt_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            output_token_ids,
        )
    
    def prepare_sampling(self, top_k):
        if top_k is not None:
            self.top_k = top_k
        else:
            self.top_k = None
            
    def __init__(self, sampler):
        super().__init__(sampler)
        # Store Ascend-specific optimizations
        self._ascend_optimizations_enabled = True
        self.top_k = None
    
    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens + batch_size, vocab_size]. Here,
                probabilities from different requests are flattened into a
                single tensor because this is the shape of the output logits.
                NOTE: `logits` can be updated in place to save memory.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            SamplerOutput:
                Contains the final output token IDs and their logprobs if
                requested.
        """
        assert metadata.max_spec_len <= MAX_SPEC_LEN
        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        # When indexing with a tensor (bonus_logits_indices), PyTorch
        # creates a new tensor with separate storage from the original
        # logits tensor. This means any in-place operations on bonus_logits
        # won't affect the original logits tensor.
        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
            predict_bonus_token=True,
            # Override the logprobs mode to return logits because they are
            # needed later to compute the accepted token logprobs.
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # Just like `bonus_logits`, `target_logits` is a new tensor with
        # separate storage from the original `logits` tensor. Therefore,
        # it is safe to update `target_logits` in place.
        raw_target_logits = logits[target_logits_indices]
        # Use float32 for the target_logits.
        raw_target_logits = raw_target_logits.to(torch.float32)
        target_logits = raw_target_logits
        if not self.is_processed_logprobs_mode:
            # Clone raw_target_logits before applying processors to preserve
            # the original raw logits for logprobs computation, since
            # apply_logits_processors modifies the tensor in-place.
            target_logits = target_logits.clone()
        target_logits = self.apply_logits_processors(
            target_logits, sampling_metadata, metadata
        )
        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `apply_sampling_constraints` function.
        target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
            self.top_k
        )

        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
        )

        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            logprobs_tensors = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                target_logits if self.is_processed_logprobs_mode else raw_target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
            )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )

def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    tp_group = get_tp_group()
    B, V_local = logits.shape
    rank = tp_group.rank_in_group
    world_size = tp_group.world_size

    local_max_logits, local_max_indices = logits.max(dim=-1)

    local_global_idx = local_max_indices + rank * V_local  # [B]

    # [B, world_size]
    gathered_logits = tp_group.all_gather(local_max_logits.unsqueeze(-1), dim=-1)
    gathered_global_idx = tp_group.all_gather(local_global_idx.unsqueeze(-1), dim=-1) # [B, world_size]

    global_max_rank = gathered_logits.argmax(dim=-1)  # [B]

    target_argmax = gathered_global_idx.gather(
        dim=-1,
        index=global_max_rank.unsqueeze(-1)
    ).squeeze(-1)  # [B]
    return target_argmax
    
def apply_sampling_constraints(
    logits: torch.Tensor,  # [num_tokens, vocab_size//tp_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
    top_k,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Process logits based on sampling metadata for distributed scenario.

    This function applies temperature scaling to the logits,
    then top-k, allgather, and top-p. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be processed (local vocab partition).
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]:
            - Processed logits of shape [num_tokens, top_k*tp_size] or
                [num_tokens, vocab_size//tp_size] for greedy
            - Indices tensor of shape [num_tokens, top_k*tp_size] or None for greedy
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        # return logits
        return logits, None

    num_tokens = logits.shape[0]
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    # Get expanded top_k and top_p tensors.
    k = None
    if sampling_metadata.top_k is not None:
        k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        )
    p = None
    if sampling_metadata.top_p is not None:
        p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )

    # New flow: top_k -> allgather -> top_p
    # Returns processed logits and indices
    if get_ascend_config().enable_reduce_sample:
        return apply_top_k_top_p(logits, top_k, k, p)
    else:
        return apply_top_k_top_p(logits, k, p)

def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size//tp_size] or tuple of (logits, indices)
    # For greedy: Tensor [num_tokens, vocab_size//tp_size]
    # For random: tuple of (logits [num_tokens, top_k*tp_size], indices [num_tokens, top_k*tp_size])
    target_logits_or_tuple: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Rejection sampling for speculative decoding in distributed setting.

    Args:
        draft_token_ids: Draft token IDs [num_tokens]
        num_draft_tokens: Number of draft tokens per request
        max_spec_len: Maximum speculative length
        cu_num_draft_tokens: Cumulative draft tokens [batch_size]
        draft_probs: Draft probabilities [num_tokens, vocab_size] or None for ngram
        target_logits_or_tuple: Target logits or tuple of (logits, indices)
            - For greedy: Tensor [num_tokens, vocab_size//tp_size]
            - For random: tuple of (compressed_logits, indices) where
                - compressed_logits: [num_tokens, top_k*tp_size]
                - indices: [num_tokens, top_k*tp_size] global vocabulary indices or None
        bonus_token_ids: Bonus token IDs [batch_size, 1]
        sampling_metadata: Sampling metadata

    Returns:
        output_token_ids: [batch_size, max_spec_len + 1]
    """
    # Unpack target_logits_or_tuple
    if isinstance(target_logits_or_tuple, tuple):
        target_logits, target_indices = target_logits_or_tuple
    else:
        target_logits = target_logits_or_tuple
        target_indices = None

    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_logits.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    device = target_logits.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_logits.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_logits.shape[0] == num_tokens

    # When num_speculative_tokens>=3, using block verify.
    # Skip block verify when draft_probs is None (suffix/ngram methods)
    # to avoid incorrect verification results.
    using_block_verify = max_spec_len >= 3 and draft_probs is not None

    # Create output buffer.
    output_token_ids = torch.empty(
        (batch_size, max_spec_len + 1),
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if HAS_TRITON:
        grid, block_size = cal_grid_and_block_size(batch_size)
    # For greedy sampling, we need to do allgather first to get global argmax
    if not sampling_metadata.all_random:
        if get_ascend_config().enable_reduce_sample: 
            target_argmax = greedy_sample(target_logits)
        else:
            target_argmax = target_logits.argmax(dim=-1).view(-1)

        if HAS_TRITON:
            rejection_greedy_sample_with_triton(
                output_token_ids,
                num_draft_tokens,
                cu_num_draft_tokens,
                draft_token_ids,
                target_argmax,
                bonus_token_ids,
                is_greedy,
                max_spec_len,
                grid,
                block_size,
            )
        else:
            if min(num_draft_tokens) == 1 and max(num_draft_tokens) == 1 and sampling_metadata.all_greedy:
                rejection_greedy_sample_spec_len_1_pytorch(
                    output_token_ids,
                    draft_token_ids,
                    target_argmax,
                    bonus_token_ids,
                )
            else:
                rejection_greedy_sample_pytorch(
                    output_token_ids,
                    cu_num_draft_tokens,
                    draft_token_ids,
                    target_argmax,
                    bonus_token_ids,
                    num_draft_tokens,
                    max_spec_len,
                    is_greedy,
                )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # For random sampling with compressed logits
    # target_logits is [num_tokens, top_k*tp_size] with indices [num_tokens, top_k*tp_size]
    if target_indices is not None:
        # Compressed mode: logits are [num_tokens, top_k*tp_size]
        # We need to handle rejection sampling with compressed vocab
        compressed_vocab_size = target_logits.shape[-1]
        global_vocab_size = draft_probs.shape[-1] if draft_probs is not None else compressed_vocab_size

        # Compute probability distribution from target logits
        target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
        assert target_probs.is_contiguous()

        # Generate uniform probabilities for rejection sampling
        uniform_probs = generate_uniform_probs(
            num_tokens,
            num_draft_tokens,
            sampling_metadata.generators,
            device,
        )

        # Sample recovered tokens for each position
        recovered_token_ids = sample_recovered_tokens(
            max_spec_len,
            num_draft_tokens,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            sampling_metadata,
            device,
            target_indices=target_indices,
            global_vocab_size=global_vocab_size,
            compressed_mode=True,
        )

        if not using_block_verify:
            # Rejection sampling for random sampling requests with compressed logits
            if HAS_TRITON:
                rejection_random_sample_kernel[(grid,)](
                    output_token_ids,
                    cu_num_draft_tokens,
                    draft_token_ids,
                    draft_probs,
                    target_probs,
                    target_indices,
                    bonus_token_ids,
                    recovered_token_ids,
                    uniform_probs.to(torch.float32),
                    is_greedy,
                    max_spec_len,
                    compressed_vocab_size,
                    global_vocab_size,
                    batch_size,
                    NO_DRAFT_PROBS=draft_probs is None,
                    COMPRESSED_MODE=True,
                    BLOCK_SIZE=block_size,
                )
            else:
                rejection_random_sample_pytorch(
                    output_token_ids,
                    cu_num_draft_tokens,
                    draft_token_ids,
                    draft_probs,
                    target_probs,
                    bonus_token_ids,
                    recovered_token_ids,
                    uniform_probs,
                    is_greedy,
                    max_spec_len,
                    compressed_vocab_size,
                    IS_NGRAM=draft_probs is None,
                    target_indices=target_indices,
                    compressed_mode=True,
                )
        else:
            if HAS_TRITON:
                rejection_random_sample_block_verify_kernel[(grid,)](
                    output_token_ids,
                    cu_num_draft_tokens,
                    draft_token_ids,
                    draft_probs,
                    target_probs,
                    target_indices,
                    bonus_token_ids,
                    recovered_token_ids,
                    uniform_probs.to(torch.float32),
                    is_greedy,
                    max_spec_len,
                    compressed_vocab_size,
                    global_vocab_size,
                    batch_size,
                    NO_DRAFT_PROBS=draft_probs is None,
                    COMPRESSED_MODE=True,
                    BLOCK_SIZE=block_size,
                )
            else:
                rejection_random_sample_block_verify_pytorch(
                    output_token_ids,
                    cu_num_draft_tokens,
                    draft_token_ids,
                    draft_probs,
                    target_probs,
                    bonus_token_ids,
                    recovered_token_ids,
                    uniform_probs,
                    is_greedy,
                    max_spec_len,
                    compressed_vocab_size,
                    IS_NGRAM=draft_probs is None,
                    target_indices=target_indices,
                    compressed_mode=True,
                )
    else:
        # Fallback to original uncompressed mode
        # This path should not be used in the new distributed flow
        vocab_size = target_logits.shape[-1]

        # Compute probability distribution from target logits
        target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
        assert target_probs.is_contiguous()

        # Generate uniform probabilities for rejection sampling
        uniform_probs = generate_uniform_probs(
            num_tokens,
            num_draft_tokens,
            sampling_metadata.generators,
            device,
        )

        # Sample recovered tokens for each position
        recovered_token_ids = sample_recovered_tokens(
            max_spec_len,
            num_draft_tokens,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            sampling_metadata,
            device,
            target_indices=None,
            global_vocab_size=vocab_size,
            compressed_mode=False,
        )

        if not using_block_verify:
            if HAS_TRITON:
                rejection_random_sample_kernel[(grid,)](
                    output_token_ids,
                    cu_num_draft_tokens,
                    draft_token_ids,
                    draft_probs,
                    target_probs,
                    None,  # target_indices
                    bonus_token_ids,
                    recovered_token_ids,
                    uniform_probs.to(torch.float32),
                    is_greedy,
                    max_spec_len,
                    vocab_size,
                    vocab_size,  # global_vocab_size
                    batch_size,
                    NO_DRAFT_PROBS=draft_probs is None,
                    COMPRESSED_MODE=False,
                    BLOCK_SIZE=block_size,
                )
            else:
                rejection_random_sample_pytorch(
                    output_token_ids,
                    cu_num_draft_tokens,
                    draft_token_ids,
                    draft_probs,
                    target_probs,
                    bonus_token_ids,
                    recovered_token_ids,
                    uniform_probs,
                    is_greedy,
                    max_spec_len,
                    vocab_size,
                    IS_NGRAM=draft_probs is None,
                    target_indices=None,
                    compressed_mode=False,
                )
        else:
            if HAS_TRITON:
                rejection_random_sample_block_verify_kernel[(grid,)](
                    output_token_ids,
                    cu_num_draft_tokens,
                    draft_token_ids,
                    draft_probs,
                    target_probs,
                    None,  # target_indices
                    bonus_token_ids,
                    recovered_token_ids,
                    uniform_probs.to(torch.float32),
                    is_greedy,
                    max_spec_len,
                    vocab_size,
                    vocab_size,  # global_vocab_size
                    batch_size,
                    NO_DRAFT_PROBS=draft_probs is None,
                    COMPRESSED_MODE=False,
                    BLOCK_SIZE=block_size,
                )
            else:
                rejection_random_sample_block_verify_pytorch(
                    output_token_ids,
                    cu_num_draft_tokens,
                    draft_token_ids,
                    draft_probs,
                    target_probs,
                    bonus_token_ids,
                    recovered_token_ids,
                    uniform_probs,
                    is_greedy,
                    max_spec_len,
                    vocab_size,
                    IS_NGRAM=draft_probs is None,
                    target_indices=None,
                    compressed_mode=False,
                )

    return output_token_ids

def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    if HAS_TRITON:
        expand_triton(batch_size, expanded_x, x, cu_num_tokens, replace_from, replace_to, max_num_tokens=MAX_SPEC_LEN)
    else:
        expand_pytorch(
            expanded_x,
            x,
            cu_num_tokens,
            replace_from,
            replace_to,
            MAX_NUM_TOKENS=MAX_SPEC_LEN,  # To avoid recompilation.
        )
    return expanded_x


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
    target_indices: torch.Tensor | None = None,
    global_vocab_size: int | None = None,
    compressed_mode: bool = False,
) -> torch.Tensor:
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]

    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()

    num_draft_tensor = torch.tensor(num_draft_tokens, pin_memory=True).to(device, non_blocking=True)
    has_draft_mask = num_draft_tensor > 0

    for i, generator in sampling_metadata.generators.items():
        temp_q = torch.empty_like(q[i])
        temp_q.exponential_(generator=generator)
        q[i] = torch.where(has_draft_mask[i], temp_q, q[i])

    recovered_token_ids = torch.empty_like(draft_token_ids)
    if HAS_TRITON:
        sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
            recovered_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            target_indices,  # None for uncompressed mode
            q,
            vocab_size,
            global_vocab_size if global_vocab_size is not None else vocab_size,
            triton.next_power_of_2(vocab_size),
            NO_DRAFT_PROBS=draft_probs is None,
            COMPRESSED_MODE=compressed_mode,
            SUB_BLOCK=512,
            # TODO: enable multibuffer when accuracy problem is solved.
            multibuffer=False,
        )
    else:
        sample_recovered_tokens_pytorch(
            recovered_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            IS_NGRAM=draft_probs is None,
            target_indices=target_indices,
            compressed_mode=compressed_mode,
        )
    return recovered_token_ids


def rejection_greedy_sample_spec_len_1_pytorch(
    output_token_ids,  # [batch_size, 2]
    draft_token_ids,  # [num_tokens]
    target_argmax,  # [num_tokens]
    bonus_token_ids,  # [batch_size]
):
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)
    assert batch_size == num_tokens
    accept_req_mask = draft_token_ids == target_argmax
    output_token_ids[:, 0] = target_argmax
    bonus_token_ids = bonus_token_ids.squeeze(1)
    output_token_ids[:, 1] = torch.where(accept_req_mask, bonus_token_ids, output_token_ids[:, 1])


def rejection_greedy_sample_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    target_argmax,  # [num_tokens]
    bonus_token_ids,  # [batch_size]
    draft_tokens_per_req,  # [batch_size], list
    max_spec_len,
    is_greedy=None,  # [batch_size] or None
):
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)
    device = output_token_ids.device
    draft_tokens_per_req = torch.tensor(draft_tokens_per_req).to(device, non_blocking=True)
    if is_greedy is None:
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)

    start_indices = cu_num_draft_tokens - draft_tokens_per_req
    req_ids = torch.arange(batch_size, device=device)
    token_req_ids = torch.repeat_interleave(req_ids, draft_tokens_per_req)
    token_positions = torch.arange(num_tokens, device=device) - start_indices[token_req_ids]

    # Find the first mismatch position of each request.
    mismatch_global = draft_token_ids != target_argmax
    if max_spec_len == 0:
        first_mismatch_pos_per_req = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:
        # [bs, max_spec_len]
        pos_matrix = torch.full((batch_size, max_spec_len), -1, dtype=torch.long, device=device)
        pos_matrix[token_req_ids, token_positions] = token_positions
        mismatch_matrix = torch.full((batch_size, max_spec_len), False, dtype=torch.bool, device=device)
        mismatch_matrix[token_req_ids, token_positions] = mismatch_global
        mismatch_positions = torch.where(mismatch_matrix, pos_matrix, max_spec_len * 2)
        first_mismatch_pos_per_req, _ = torch.min(mismatch_positions, dim=1)
        no_mismatch_mask = first_mismatch_pos_per_req == max_spec_len * 2
        first_mismatch_pos_per_req[no_mismatch_mask] = draft_tokens_per_req[no_mismatch_mask]

    # Copy matched target tokens into output.
    copy_len = torch.minimum(first_mismatch_pos_per_req + 1, draft_tokens_per_req)
    copy_indices = torch.arange(max_spec_len + 1, device=device).expand(batch_size, -1)
    copy_mask = copy_indices < copy_len.unsqueeze(1)
    greedy_mask = is_greedy.unsqueeze(1)
    final_copy_mask = copy_mask & greedy_mask
    global_idx = start_indices.unsqueeze(1) + copy_indices
    output_token_ids[final_copy_mask] = target_argmax[global_idx[final_copy_mask]].to(output_token_ids.dtype)
    # Fill bonus token.
    needs_bonus = is_greedy & (first_mismatch_pos_per_req >= draft_tokens_per_req)
    if torch.any(needs_bonus):
        bonus_rows = torch.where(needs_bonus)[0]
        bonus_cols = draft_tokens_per_req[bonus_rows]
        bonus_token_ids = bonus_token_ids.squeeze(1)
        output_token_ids[bonus_rows, bonus_cols] = bonus_token_ids[bonus_rows]


def rejection_random_sample_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size] or [num_tokens, compressed_vocab_size]
    bonus_token_ids,  # [batch_size]
    recovered_token_ids,  # [num_tokens]
    uniform_probs,  # [num_tokens]
    is_greedy,  # [batch_size]
    max_spec_len,
    vocab_size,
    IS_NGRAM=False,
    target_indices=None,  # [num_tokens, compressed_vocab_size] global vocab indices
    compressed_mode=False,
):
    """
    This function implements the Speculative Decoding rejection sampling step.
    Instead of looping through each request and each token (which causes high
    overhead), it uses a fully vectorized approach:

    1.  **Index Mapping**: Converts the flattened 1D token arrays into a 2D
        [batch_size, max_draft_len] grid using 'cu_num_draft_tokens' to handle
        variable-length sequences in the batch.
    2.  **Parallel Validation**: Calculates the acceptance condition
        (target_prob / draft_prob >= uniform_sample) for ALL draft tokens
        simultaneously across the entire batch.
    3.  **Short-circuit Simulation**: In the loop version, once a token is rejected,
        subsequent tokens are ignored. Here, we simulate this by finding the
        'first_reject_pos' using argmax on the rejection mask and creating a
        'should_skip' mask for all indices after the first failure.
    4.  **Token Selection**: Uses 'torch.where' to select:
        - Draft tokens (if accepted)
        - Recovered tokens (at the point of first rejection)
        - Bonus tokens (if all tokens in a sequence were accepted)
    5.  **Masking**: Ensures operations only apply to non-greedy requests and
        within valid sequence lengths.
    """

    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device

    zero_cpu = torch.tensor([0], pin_memory=True)
    zero_device = zero_cpu.to(device, non_blocking=True)

    cu_start = torch.cat([zero_device, cu_num_draft_tokens[:-1]])
    cu_end = cu_num_draft_tokens
    num_draft_per_batch = cu_end - cu_start

    max_draft_len = max_spec_len
    pos_indices_cpu = torch.arange(max_draft_len, pin_memory=True)
    pos_indices = pos_indices_cpu.to(device, non_blocking=True)[None, :]

    valid_mask = pos_indices < num_draft_per_batch[:, None]
    global_token_indices = cu_start[:, None] + pos_indices
    global_token_indices = global_token_indices.clamp(0, draft_token_ids.shape[0] - 1)
    draft_tokens = draft_token_ids[global_token_indices]  # [batch_size, max_draft_len]

    if IS_NGRAM:
        ones_cpu = torch.ones(1, pin_memory=True, dtype=torch.float32)
        draft_token_probs = ones_cpu.to(device, non_blocking=True).expand_as(draft_tokens)
    else:
        flat_indices = global_token_indices.flatten()
        flat_draft_tokens = draft_tokens.flatten()
        flat_draft_probs = draft_probs[flat_indices, flat_draft_tokens]
        draft_token_probs = flat_draft_probs.view(batch_size, max_draft_len)

    # Get target token probs
    if compressed_mode:
        # For compressed mode, need to search for draft token in candidates
        flat_global_indices = global_token_indices.flatten()
        flat_draft_tokens = draft_tokens.flatten()

        flat_target_indices = target_indices[flat_global_indices]
        flat_target_probs = target_probs[flat_global_indices]

        # Check if draft token is in candidates
        draft_expanded = flat_draft_tokens.unsqueeze(1)
        is_in_candidates = (flat_target_indices == draft_expanded)

        # Get the probability of draft token from target (if present)
        target_token_probs_flat = torch.where(
            is_in_candidates,
            flat_target_probs,
            torch.tensor(0.0, device=device)
        ).sum(dim=1)

        target_token_probs = target_token_probs_flat.view(batch_size, max_draft_len)
    else:
        flat_indices = global_token_indices.flatten()
        flat_draft_tokens = draft_tokens.flatten()
        flat_target_probs = target_probs[flat_indices, flat_draft_tokens]
        target_token_probs = flat_target_probs.view(batch_size, max_draft_len)

    uniform_token_probs = uniform_probs[global_token_indices]
    recovered_tokens = recovered_token_ids[global_token_indices]

    zero_threshold_cpu = torch.tensor([0.0], pin_memory=True, dtype=torch.float32)
    zero_threshold = zero_threshold_cpu.to(device, non_blocking=True)

    acceptance_condition = (draft_token_probs > zero_threshold) & (
        target_token_probs / draft_token_probs >= uniform_token_probs
    )

    first_rejection = (~acceptance_condition) & valid_mask

    default_pos_cpu = torch.full([batch_size, 1], max_draft_len, pin_memory=True)
    default_pos = default_pos_cpu.to(device, non_blocking=True)

    first_reject_pos = torch.where(
        first_rejection.any(dim=1, keepdim=True), first_rejection.float().argmax(dim=1, keepdim=True), default_pos
    )
    pos_mask = pos_indices >= first_reject_pos
    should_skip = pos_mask & valid_mask

    final_acceptance = acceptance_condition & (~should_skip)
    non_greedy_mask = ~is_greedy
    update_mask = non_greedy_mask[:, None] & valid_mask & (~should_skip)

    first_reject_mask = (pos_indices == first_reject_pos) & valid_mask & non_greedy_mask[:, None]
    final_update_mask = update_mask | first_reject_mask
    final_tokens = torch.where(
        first_reject_mask,
        recovered_tokens,
        torch.where(final_acceptance, draft_tokens, output_token_ids[:, :max_draft_len]),
    )

    output_token_ids[:, :max_draft_len] = torch.where(
        final_update_mask, final_tokens, output_token_ids[:, :max_draft_len]
    )

    no_rejection = first_reject_pos.squeeze(1) >= num_draft_per_batch
    should_add_bonus = non_greedy_mask & no_rejection

    bonus_positions = num_draft_per_batch  # [batch_size]

    seq_len = output_token_ids.shape[1]
    all_positions_cpu = torch.arange(seq_len, pin_memory=True)
    all_positions = all_positions_cpu.to(device, non_blocking=True)[None, :]  # [1, seq_len]

    batch_bonus_positions = bonus_positions[:, None]  # [batch_size, 1]

    max_spec_len_cpu = torch.tensor([max_spec_len], pin_memory=True)
    max_spec_len_device = max_spec_len_cpu.to(device, non_blocking=True)

    valid_bonus_pos = bonus_positions < (max_spec_len_device + 1)
    final_bonus_mask = should_add_bonus & valid_bonus_pos

    bonus_pos_match = all_positions == batch_bonus_positions
    bonus_pos_mask = bonus_pos_match & final_bonus_mask[:, None]

    bonus_values_expanded = bonus_token_ids.view(-1, 1).expand(-1, seq_len)
    output_token_ids[:] = torch.where(bonus_pos_mask, bonus_values_expanded, output_token_ids)


def expand_pytorch(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS,
):
    """
    This function broadcasts batch-level values (input_ptr) to token-level
    positions (output_ptr) based on cumulative token offsets. It acts like
    a "scatter" or "repeat_interleave" operation but with custom logic:

    1.  **Range Broadcasting**: It creates a boolean matrix 'in_range' of size
        [num_tokens, batch_size] that identifies which batch index each token
        belongs to by checking if the token index falls between cu_start and cu_end.
    2.  **Conditional Replacement**: Before expansion, it replaces specific values
        (e.g., padding or special markers) in the input to prepare the data.
    3.  **Matrix-based Mapping**: It uses 'torch.einsum' to perform a weighted
        sum that effectively "picks" the correct batch value for every token position
        simultaneously, avoiding a Python loop over the batch.
    """
    device = cu_num_tokens_ptr.device
    batch_size = input_ptr.shape[0]
    num_tokens = output_ptr.shape[0]

    if batch_size == 0 or num_tokens == 0:
        return

    cu_start = torch.cat([torch.tensor([0], pin_memory=True).to(device, non_blocking=True), cu_num_tokens_ptr[:-1]])
    cu_end = cu_num_tokens_ptr

    token_indices = torch.arange(num_tokens, device=device)[:, None]  # [num_tokens, 1]
    cu_start_exp = cu_start[None, :]  # [1, batch_size]
    cu_end_exp = cu_end[None, :]  # [1, batch_size]

    in_range = (token_indices >= cu_start_exp) & (token_indices < cu_end_exp)

    replaced_input = torch.where(input_ptr == replace_from, replace_to, input_ptr).float()

    token_values = torch.einsum("tb,b->t", in_range.float(), replaced_input)

    needs_update = in_range.any(dim=1)

    output_ptr[:] = torch.where(needs_update, token_values, output_ptr)


def sample_recovered_tokens_pytorch(
    output_token_ids,  # [num_tokens]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size] or [num_tokens, compressed_vocab_size]
    q,  # [batch_size, vocab_size] or [batch_size, compressed_vocab_size]
    vocab_size,
    IS_NGRAM=False,
    target_indices=None,  # [num_tokens, compressed_vocab_size] global vocab indices
    compressed_mode=False,
):
    """
    When a draft token is rejected, we must sample a "recovered" token from
    a modified distribution. This function calculates that distribution across
    the entire flattened batch.

    1.  **Token-to-Batch Mapping**: Using the cumulative draft token counts, it
        determines which request in the batch each token belongs to. This is
        necessary because 'q' (normalization factor) is stored per-request.
    2.  **Probability Adjustment**:
        - If N-GRAM: It zeroes out the draft token's probability in the target.
        - If Probabilistic: It calculates max(0, target_probs - draft_probs)
          as per the standard speculative decoding algorithm.
    3.  **Normalization & Sampling**: It divides the adjusted probabilities
        by the normalization distribution 'q'. To remain vectorized, it
        broadcasts 'q' from [batch_size, vocab] to [num_tokens, vocab].
    4.  **Argmax Selection**: It selects the best recovery token for every
        position in one pass using torch.argmax.
    """
    device = output_token_ids.device
    num_tokens = output_token_ids.shape[0]

    if num_tokens == 0:
        return

    cu_start = torch.cat(
        [
            torch.tensor([0], pin_memory=True).to(device, non_blocking=True),
            cu_num_draft_tokens[:-1],
        ]
    )
    cu_end = cu_num_draft_tokens

    token_indices = torch.arange(num_tokens, device=device)  # [num_tokens]

    token_indices_expanded = token_indices[:, None]  # [num_tokens, 1]
    cu_start_expanded = cu_start[None, :]  # [1, batch_size]
    cu_end_expanded = cu_end[None, :]  # [1, batch_size]

    in_range_mask = (token_indices_expanded >= cu_start_expanded) & (token_indices_expanded < cu_end_expanded)

    token_to_batch = torch.argmax(in_range_mask.int(), dim=1)

    has_match = in_range_mask.any(dim=1)
    token_to_batch = torch.where(has_match, token_to_batch, 0)

    if compressed_mode:
        # Compressed mode: target_probs is [num_tokens, compressed_vocab_size]
        # target_indices maps compressed indices to global vocab indices
        if IS_NGRAM:
            # Zero out the draft token in target_probs
            prob = target_probs.clone()
            for i in range(num_tokens):
                draft_id = draft_token_ids[i]
                mask = target_indices[i] == draft_id
                prob[i, mask] = 0
        else:
            # Gather draft probs at candidate indices
            flat_indices = target_indices.flatten()
            token_offsets = torch.arange(num_tokens, device=device)[:, None] * draft_probs.shape[1]
            flat_token_offsets = token_offsets.expand_as(target_indices).flatten()

            draft_probs_flat = draft_probs.flatten()
            valid_mask = flat_indices < draft_probs.shape[1]
            flat_draft_probs_at_indices = torch.where(
                valid_mask,
                draft_probs_flat[flat_token_offsets + flat_indices],
                torch.tensor(0.0, device=device)
            )
            draft_probs_at_indices = flat_draft_probs_at_indices.view(num_tokens, vocab_size)

            prob = torch.maximum(
                target_probs - draft_probs_at_indices,
                torch.tensor(0.0, device=device),
            )
    else:
        # Uncompressed mode
        if IS_NGRAM:
            token_indices = torch.arange(num_tokens, device=device)

            modified_target_probs = target_probs.clone()
            modified_target_probs[token_indices, draft_token_ids] = 0
            prob = modified_target_probs

        else:
            prob = torch.maximum(
                target_probs - draft_probs,
                torch.tensor(0.0, pin_memory=True).to(device, non_blocking=True),
            )

    q_values = q[token_to_batch]  # [num_tokens, vocab_size]

    epsilon = 1e-10
    q_values_safe = torch.where(q_values == 0, epsilon, q_values)
    q_values_safe = torch.where(torch.isinf(q_values), epsilon, q_values_safe)

    prob_over_q = prob / q_values_safe

    prob_over_q = torch.where((q_values == 0) | torch.isinf(q_values), -1e10, prob_over_q)

    if compressed_mode:
        # Get the index in compressed vocab
        compressed_indices = torch.argmax(prob_over_q, dim=1)
        # Convert to global vocabulary indices
        output_token_ids[:] = target_indices[torch.arange(num_tokens, device=device), compressed_indices]
    else:
        recovered_ids = torch.argmax(prob_over_q, dim=1)
        output_token_ids[:] = recovered_ids


def rejection_random_sample_block_verify_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size] or [num_tokens, compressed_vocab_size]
    bonus_token_ids,  # [batch_size]
    recovered_token_ids,  # [num_tokens]
    uniform_probs,  # [num_tokens]
    is_greedy,  # [batch_size]
    max_spec_len,
    vocab_size,
    IS_NGRAM=False,
    target_indices=None,  # [num_tokens, compressed_vocab_size] global vocab indices
    compressed_mode=False,
):
    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device

    zero_cpu = torch.tensor([0], pin_memory=True)
    zero_device = zero_cpu.to(device, non_blocking=True)

    cu_start = torch.cat([zero_device, cu_num_draft_tokens[:-1]])
    cu_end = cu_num_draft_tokens
    num_draft_per_batch = (cu_end - cu_start)[:, None]
    pos_indices_cpu = torch.arange(max_spec_len, pin_memory=True)
    pos_indices = pos_indices_cpu.to(device, non_blocking=True)[None, :]
    valid_mask = pos_indices < num_draft_per_batch
    global_token_indices = cu_start[:, None] + pos_indices
    global_token_indices = global_token_indices.clamp(0, draft_token_ids.shape[0] - 1)
    draft_tokens = draft_token_ids[global_token_indices]

    if IS_NGRAM:
        ones_cpu = torch.ones(1, pin_memory=True, dtype=torch.float32)
        draft_token_probs = ones_cpu.to(device, non_blocking=True).expand_as(draft_tokens)
    else:
        flat_indices = global_token_indices.flatten()
        flat_draft_tokens = draft_tokens.flatten()
        flat_draft_probs = draft_probs[flat_indices, flat_draft_tokens]
        draft_token_probs = flat_draft_probs.view(batch_size, max_spec_len)

    # Get target token probs
    if compressed_mode:
        # For compressed mode, need to search for draft token in candidates
        flat_global_indices = global_token_indices.flatten()
        flat_draft_tokens = draft_tokens.flatten()

        flat_target_indices = target_indices[flat_global_indices]
        flat_target_probs = target_probs[flat_global_indices]

        # Check if draft token is in candidates
        draft_expanded = flat_draft_tokens.unsqueeze(1)
        is_in_candidates = (flat_target_indices == draft_expanded)

        # Get the probability of draft token from target (if present)
        target_token_probs_flat = torch.where(
            is_in_candidates,
            flat_target_probs,
            torch.tensor(0.0, device=device)
        ).sum(dim=1)

        target_token_probs = target_token_probs_flat.view(batch_size, max_spec_len)
    else:
        flat_indices = global_token_indices.flatten()
        flat_draft_tokens = draft_tokens.flatten()
        flat_target_probs = target_probs[flat_indices, flat_draft_tokens]
        target_token_probs = flat_target_probs.view(batch_size, max_spec_len)

    uniform_token_probs = uniform_probs[global_token_indices]
    recovered_tokens = recovered_token_ids[global_token_indices]

    pi = target_token_probs / draft_token_probs
    pi = pi.clamp(max=1.0)
    pi = torch.cumprod(pi, dim=-1)
    uniform_token_probs = torch.cumprod(uniform_token_probs, dim=-1)
    legal_mask = (draft_token_probs > 0) & (pi >= uniform_token_probs)
    legal_mask = legal_mask & valid_mask

    last_accept_pos = torch.where(
        legal_mask.any(dim=-1, keepdim=True),
        (max_spec_len - legal_mask.flip(dims=[-1]).float().argmax(dim=-1, keepdim=True) - 1),
        -1,
    )
    non_greedy_mask = (~is_greedy)[:, None]

    accept_mask = (pos_indices <= last_accept_pos) & valid_mask & non_greedy_mask
    output_token_ids[:, :max_spec_len] = torch.where(accept_mask, draft_tokens, output_token_ids[:, :max_spec_len])

    reject_mask = (pos_indices == last_accept_pos + 1) & valid_mask & non_greedy_mask
    output_token_ids[:, :max_spec_len] = torch.where(reject_mask, recovered_tokens, output_token_ids[:, :max_spec_len])

    bonus_mask = (last_accept_pos + 1 >= num_draft_per_batch) & non_greedy_mask
    all_positions_cpu = torch.arange(max_spec_len + 1, pin_memory=True)
    all_positions = all_positions_cpu.to(device, non_blocking=True)[None, :]
    bonus_pos_match = all_positions == num_draft_per_batch
    bonus_mask = bonus_mask & bonus_pos_match
    bonus_values_expanded = bonus_token_ids.view(-1, 1).expand(-1, max_spec_len + 1)
    output_token_ids[:] = torch.where(bonus_mask, bonus_values_expanded, output_token_ids)