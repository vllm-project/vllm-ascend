# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.model_executor.layers.utils import get_token_bin_counts_and_mask
from vllm.v1.sample.ops.penalties import _convert_to_tensors


def apply_penalties(logits: torch.Tensor, prompt_tokens_tensor: torch.Tensor,
                    output_tokens_tensor: torch.Tensor,
                    presence_penalties: torch.Tensor,
                    frequency_penalties: torch.Tensor,
                    repetition_penalties: torch.Tensor) -> torch.Tensor:
    """Optimized implementation of repetition penalties on NPU.

    Applies penalties in place to the logits tensor
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts 
        are padded to the maximum prompt length within the batch using 
        `vocab_size` as the padding value. The value `vocab_size` is used 
        for padding because it does not correspond to any valid token ID 
        in the vocabulary.
    output_tokens_tensor: The output tokens tensor.
    presence_penalties: The presence penalties of shape (num_seqs, )
    frequency_penalties: The frequency penalties of shape (num_seqs, )
    repetition_penalties: The repetition penalties of shape (num_seqs, )
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(prompt_tokens_tensor,
                                                   vocab_size, num_seqs)
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs)

    repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(
        1, vocab_size)

    # Avoid IndexPut operations in original apply_penalties function which are extremely time-consuming on NPU.
    sequence_mask = prompt_mask | output_mask
    logits = torch.where(sequence_mask & torch.lt(logits, 0),
                         logits * repetition_penalties,
                         logits).to(logits.dtype)
    logits = torch.where(sequence_mask & torch.ge(logits, 0),
                         logits / repetition_penalties,
                         logits).to(logits.dtype)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    """
    Applies presence, frequency and repetition penalties to the logits.
    """
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size,
                                          logits.device)
    return apply_penalties(logits, prompt_token_ids, output_tokens_t,
                           presence_penalties, frequency_penalties,
                           repetition_penalties)
