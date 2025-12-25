import pytest
import torch
from torch.testing import assert_close

from vllm_ascend.sample.rejection_sampler import (
    rejection_random_sample_block_verify_kernel,
    rejection_random_sample_block_verify_pytorch)

DEVICE = "npu"
BATCH_SIZE = 3
MAX_SPEC_LEN = 3
VOCAB_SIZE = 5
NUM_TOKENS = BATCH_SIZE * MAX_SPEC_LEN
CU_NUM_DRAFT_TOKENS = torch.arange(start=MAX_SPEC_LEN,
                                   end=NUM_TOKENS + 1,
                                   step=MAX_SPEC_LEN,
                                   dtype=torch.int32,
                                   device=DEVICE)
DRAFT_TOKEN_IDS = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2],
                               dtype=torch.int64,
                               device=DEVICE)
DRAFT_PROBS = None
TARGET_PROBS = torch.tensor(
    [
        [0.2, 0.1, 0.2, 0.4, 0.1],  # 0
        [0.1, 0.4, 0.1, 0.1, 0.3],  # 0
        [0.2, 0.1, 0.4, 0.1, 0.2],  # 0
        [0.4, 0.2, 0.1, 0.2, 0.1],  # 0
        [0.1, 0.6, 0.1, 0.1, 0.1],  # 1
        [0.2, 0.2, 0.2, 0.3, 0.1],  # 0
        [0.4, 0.4, 0.1, 0.0, 0.1],  # 1
        [0.4, 0.3, 0.1, 0.1, 0.1],  # 0
        [0.4, 0.0, 0.5, 0.0, 0.1],  # 1
    ],
    dtype=torch.float32,
    device=DEVICE)
UNIFORM_PROBS = torch.tensor([
    0.9,
    0.7,
    0.8,
    0.5,
    0.45,
    1.0,
    0.39,
    0.4,
    0.1,
],
                             dtype=torch.float32,
                             device=DEVICE)
BONUS_TOKEN_IDS = torch.full((BATCH_SIZE, ),
                             MAX_SPEC_LEN + 1,
                             dtype=torch.int64,
                             device=DEVICE)
IS_GREEDY = torch.zeros(NUM_TOKENS, dtype=torch.bool, device=DEVICE)


@pytest.mark.parametrize("cu_num_draft_tokens", [CU_NUM_DRAFT_TOKENS])
@pytest.mark.parametrize("draft_token_ids", [DRAFT_TOKEN_IDS])
@pytest.mark.parametrize("draft_probs", [DRAFT_PROBS])
@pytest.mark.parametrize("target_probs", [TARGET_PROBS])
@pytest.mark.parametrize("bonus_token_ids", [BONUS_TOKEN_IDS])
@pytest.mark.parametrize("uniform_probs", [UNIFORM_PROBS])
@pytest.mark.parametrize("is_greedy", [IS_GREEDY])
@pytest.mark.parametrize("batch_size", [BATCH_SIZE])
@pytest.mark.parametrize("max_spec_len", [MAX_SPEC_LEN])
@pytest.mark.parametrize("vocab_size", [VOCAB_SIZE])
@torch.inference_mode()
def test_rejection_sampler_block_verify_triton_kernel(
        cu_num_draft_tokens,  # [batch_size]
        draft_token_ids,  # [num_tokens]
        draft_probs,  # [num_tokens, vocab_size] or None
        target_probs,  # [num_tokens, vocab_size]
        bonus_token_ids,  # [batch_size]
        uniform_probs,  # [num_tokens]
        is_greedy,  # [batch_size]
        batch_size,  # int
        max_spec_len,  # int
        vocab_size,  # int
) -> None:
    output_token_ids_ref = torch.full((batch_size, max_spec_len + 1),
                                      -1,
                                      dtype=torch.int64,
                                      device=DEVICE)

    output_token_ids_triton = output_token_ids_ref.clone()

    rejection_random_sample_block_verify_pytorch(
        output_token_ids=output_token_ids_ref,
        cu_num_draft_tokens=cu_num_draft_tokens,
        draft_token_ids=draft_token_ids,
        draft_probs=draft_probs,
        target_probs=target_probs,
        bonus_token_ids=bonus_token_ids,
        uniform_probs=uniform_probs,
        is_greedy=is_greedy,
        max_spec_len=max_spec_len,
        vocab_size=vocab_size,
        IS_NGRAM=draft_probs is None)

    rejection_random_sample_block_verify_kernel[(batch_size, )](
        output_token_ids_ptr=output_token_ids_triton,
        cu_num_draft_tokens_ptr=cu_num_draft_tokens,
        draft_token_ids_ptr=draft_token_ids,
        draft_probs_ptr=draft_probs,
        target_probs_ptr=target_probs,
        bonus_token_ids_ptr=bonus_token_ids,
        uniform_probs_ptr=uniform_probs,
        is_greedy_ptr=is_greedy,
        max_spec_len=max_spec_len,
        vocab_size=vocab_size,
        NO_DRAFT_PROBS=draft_probs is None,
        multibuffer=True)

    assert_close(output_token_ids_ref, output_token_ids_triton)
