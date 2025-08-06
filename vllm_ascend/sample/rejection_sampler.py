# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
import torch.nn as nn
import vllm.v1.sample.rejection_sampler as rs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (RejectionSampler, compute_probs,
                                              generate_uniform_probs)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

PLACEHOLDER_TOKEN_ID = -1
GREEDY_TEMPERATURE = -1
# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 32


class AscendRejectionSampler(RejectionSampler, nn.Module):
    """
    The implementation strictly follows the algorithm described in
        https://arxiv.org/abs/2211.17192.
    However, we want to clarify the terminology used in the implementation:
    accepted tokens: tokens that are accepted based on the relationship
            between the "raw" draft and target probabilities.
    recovered tokens: tokens that are sampled based on the adjusted probability
        distribution, which is derived from both the draft and target
        probabilities.
    bonus tokens:
        If all proposed tokens are accepted, the bonus token is added to the
        end of the sequence. The bonus token is only sampled from the target
        probabilities. We pass in the bonus tokens instead of sampling them
        in the rejection sampler to allow for more flexibility in the
        sampling process. For example, we can use top_p, top_k sampling for
        bonus tokens, while spec decode does not support these sampling
        strategies.
    output tokens:
        Tokens are finally generated with the rejection sampler.
        output tokens = accepted tokens + recovered tokens + bonus tokens
    """

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: Optional[torch.Tensor],
        # [num_tokens, vocab_size]
        target_logits: torch.Tensor,
        # [batch_size, 1]
        bonus_token_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        '''
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            target_logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens, vocab_size]. Here, probabilities from
                different requests are flattened into a single tensor because
                this is the shape of the output logits.
                NOTE: `target_logits` can be updated in place to save memory.
            bonus_token_ids_tensor (torch.Tensor):
                A tensor containing bonus tokens. Shape is [batch_size, 1].
                Bonus tokens are added to the end of the sequence if all
                proposed tokens are accepted. We generate the bonus tokens
                outside of the rejection sampler with the default sampling
                strategy. It allows for more flexibility in the sampling
                process such as top_p, top_k sampling.
            sampling_metadata (SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            output_token_ids (torch.Tensor):
                A tensor containing the final output token IDs.
        '''
        assert metadata.max_spec_len <= MAX_SPEC_LEN
        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `compute_probs` function.
        target_probs = compute_probs(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )

        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_probs,
            bonus_token_ids,
            sampling_metadata,
        )
        return output_token_ids


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_probs.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_probs.shape == (num_tokens, vocab_size)

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
    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_probs.argmax(dim=-1)
        rejection_greedy_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
            # num_warps=1,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Generate uniform probabilities for rejection sampling.
    # [num_tokens]
    uniform_probs = generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    # Sample recovered tokens for each position.
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # Rejection sampling for random sampling requests.
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
        # num_warps=1,
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
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    recovered_token_ids = torch.empty_like(draft_token_ids)
    sample_recovered_tokens_pytorch(
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        IS_NGRAM=draft_probs is None,
    )
    return recovered_token_ids


def rejection_greedy_sample_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    target_argmax,  # [num_tokens]
    bonus_token_ids,  # [batch_size]
    is_greedy=None,  # [batch_size] or None
    max_spec_len=None,
):
    batch_size = output_token_ids.shape[0]

    if is_greedy is None:
        is_greedy = torch.ones(batch_size,
                               dtype=torch.bool,
                               device=output_token_ids.device)

    for req_idx in range(batch_size):
        if not is_greedy[req_idx]:
            continue

        if req_idx == 0:
            start_idx = 0
        else:
            start_idx = cu_num_draft_tokens[req_idx - 1].item()
        end_idx = cu_num_draft_tokens[req_idx].item()
        num_draft_tokens = end_idx - start_idx

        rejected = False
        for pos in range(num_draft_tokens):
            if not rejected:
                draft_token_id = draft_token_ids[start_idx + pos].item()
                target_argmax_id = target_argmax[start_idx + pos].item()

                output_token_ids[req_idx, pos] = target_argmax_id

                if draft_token_id != target_argmax_id:
                    rejected = True

        if not rejected:
            bonus_token_id = bonus_token_ids[req_idx].item()
            output_token_ids[req_idx, num_draft_tokens] = bonus_token_id


def rejection_random_sample_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size]
    bonus_token_ids,  # [batch_size]
    recovered_token_ids,  # [num_tokens]
    uniform_probs,  # [num_tokens]
    is_greedy,  # [batch_size]
    max_spec_len,
    vocab_size,
    IS_NGRAM=False,
):
    batch_size = output_token_ids.shape[0]

    for req_idx in range(batch_size):
        if is_greedy[req_idx]:
            continue

        if req_idx == 0:
            start_idx = 0
        else:
            start_idx = cu_num_draft_tokens[req_idx - 1].item()
        end_idx = cu_num_draft_tokens[req_idx].item()
        num_draft_tokens = end_idx - start_idx

        rejected = False
        for pos in range(num_draft_tokens):
            if not rejected:
                draft_token_id = draft_token_ids[start_idx + pos].item()

                if IS_NGRAM:
                    draft_prob = 1.0
                else:
                    draft_prob = draft_probs[start_idx + pos,
                                             draft_token_id].item()

                target_prob = target_probs[start_idx + pos,
                                           draft_token_id].item()
                uniform_prob = uniform_probs[start_idx + pos].item()

                if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                    token_id = draft_token_id
                else:
                    rejected = True
                    token_id = recovered_token_ids[start_idx + pos].item()

                output_token_ids[req_idx, pos] = token_id

        if not rejected:
            bonus_token_id = bonus_token_ids[req_idx].item()
            output_token_ids[req_idx, num_draft_tokens] = bonus_token_id


def expand_pytorch(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS,
):
    batch_size = len(input_ptr)

    for req_idx in range(batch_size):
        start_idx = 0 if req_idx == 0 else cu_num_tokens_ptr[req_idx - 1]
        end_idx = cu_num_tokens_ptr[req_idx]
        num_tokens = end_idx - start_idx

        src_val = input_ptr[req_idx]
        src_val = replace_to if src_val == replace_from else src_val

        offset = torch.arange(MAX_NUM_TOKENS, device=num_tokens.device)
        mask = offset < num_tokens

        output_slice = start_idx + offset[mask]
        output_ptr[output_slice] = src_val


def sample_recovered_tokens_pytorch(
    output_token_ids,  # [num_tokens]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size]
    q,  # [batch_size, vocab_size]
    vocab_size,
    IS_NGRAM=False,
):
    batch_size = len(cu_num_draft_tokens)

    for req_idx in range(batch_size):
        start_idx = 0 if req_idx == 0 else cu_num_draft_tokens[req_idx - 1]
        end_idx = cu_num_draft_tokens[req_idx]
        num_draft_tokens = end_idx - start_idx

        for pos in range(num_draft_tokens):
            token_idx = start_idx + pos

            if IS_NGRAM:
                draft_token_id = draft_token_ids[token_idx]
                orig_prob = target_probs[token_idx, draft_token_id].item()
                target_probs[token_idx, draft_token_id] = 0
                prob = target_probs[token_idx].clone()
            else:
                draft_p = draft_probs[token_idx].clone()
                target_p = target_probs[token_idx].clone()
                prob = torch.maximum(target_p - draft_p,
                                     torch.tensor(0.0, device=target_p.device))

            q_values = torch.full((vocab_size, ),
                                  float('-inf'),
                                  device=q.device)
            q_values[:vocab_size] = q[req_idx, :vocab_size]

            recovered_id = torch.argmax(prob / q_values).item()
            output_token_ids[token_idx] = recovered_id

            if IS_NGRAM:
                target_probs[token_idx, draft_token_id] = orig_prob


rs.expand_batch_to_tokens = expand_batch_to_tokens
