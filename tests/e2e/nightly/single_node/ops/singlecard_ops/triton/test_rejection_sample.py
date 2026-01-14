import pytest
import torch
from vllm.v1.sample.rejection_sampler import \
    rejection_random_sample_kernel as original_rejection_random_sample_kernel

from vllm_ascend.ops.triton.reject_sample import (
    cal_grid_and_block_size, rejection_random_sample_block_verify_kernel,
    rejection_random_sample_kernel)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.sample.rejection_sampler import \
    rejection_random_sample_block_verify_pytorch


@pytest.fixture(scope="function", autouse=True)
def setup_device_properties():
    init_device_properties_triton()
    yield


@pytest.mark.parametrize("max_spec_len", [1, 2, 3])
@pytest.mark.parametrize("vocab_size", [151_936])
@pytest.mark.parametrize("batch_size", [1, 8, 32, 64, 128, 256, 512, 1024])
@torch.inference_mode()
def test_rejection_random_sample(max_spec_len, vocab_size, batch_size):
    device = 'npu'
    torch.manual_seed(0)
    draft_probs = torch.rand(batch_size * max_spec_len,
                             vocab_size,
                             dtype=torch.float32,
                             device=device)
    target_probs = torch.rand(batch_size * max_spec_len,
                              vocab_size,
                              dtype=torch.float32,
                              device=device)
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64,
                                    device=device)
    draft_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size * max_spec_len, ),
                                    dtype=torch.int64,
                                    device=device)
    output_token_ids = torch.empty((batch_size, max_spec_len + 1),
                                   dtype=torch.int64,
                                   device=device)
    original_output_token_ids = output_token_ids.clone()
    num_tokens = draft_token_ids.shape[0]
    uniform_probs = torch.rand((num_tokens, ),
                               dtype=torch.float32,
                               device=device)
    num_draft_tokens = [max_spec_len] * batch_size
    num_draft_tokens = torch.tensor(num_draft_tokens,
                                    dtype=torch.int32,
                                    device=device)
    cu_num_draft_tokens = torch.cumsum(num_draft_tokens,
                                       dim=0,
                                       dtype=torch.int32)
    is_greedy_ptr = torch.full((batch_size, ),
                               False,
                               dtype=torch.bool,
                               device=device)
    recovered_ids = torch.zeros_like(draft_token_ids,
                                     dtype=torch.int64,
                                     device=device)
    grid, block_size = cal_grid_and_block_size(batch_size)
    original_rejection_random_sample_kernel[(batch_size, )](
        original_output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_ids,
        uniform_probs,
        is_greedy_ptr,
        max_spec_len,
        vocab_size,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    rejection_random_sample_kernel[(grid, )](output_token_ids,
                                             cu_num_draft_tokens,
                                             draft_token_ids,
                                             draft_probs,
                                             target_probs,
                                             bonus_token_ids,
                                             recovered_ids,
                                             uniform_probs,
                                             is_greedy_ptr,
                                             max_spec_len,
                                             vocab_size,
                                             batch_size,
                                             NO_DRAFT_PROBS=draft_probs
                                             is None,
                                             BLOCK_SIZE=block_size)
    torch.npu.synchronize()
    assert torch.equal(original_output_token_ids, output_token_ids)


DEVICE = "npu"
BATCH_SIZE = 7
MAX_SPEC_LEN = 3
VOCAB_SIZE = 5
CU_NUM_DRAFT_TOKENS = torch.tensor([2, 2, 5, 8, 11, 14, 15],
                                   dtype=torch.int32,
                                   device=DEVICE)
DRAFT_TOKEN_IDS = torch.tensor([0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
                               dtype=torch.int64,
                               device=DEVICE)
NUM_TOKENS = DRAFT_TOKEN_IDS.shape[0]
DRAFT_PROBS = None
TARGET_PROBS = torch.tensor(
    [
        [0.4, 0.3, 0.1, 0.1, 0.1],  # 0
        [0.1, 0.9, 0.0, 0.0, 0.0],  # 1
        [0.2, 0.1, 0.2, 0.4, 0.1],  # 0
        [0.1, 0.4, 0.1, 0.1, 0.3],  # 0
        [0.2, 0.1, 0.4, 0.1, 0.2],  # 0
        [0.4, 0.2, 0.1, 0.2, 0.1],  # 0
        [0.1, 0.6, 0.1, 0.1, 0.1],  # 1
        [0.2, 0.2, 0.2, 0.3, 0.1],  # 0
        [0.4, 0.2, 0.1, 0.2, 0.1],  # 0
        [0.1, 0.6, 0.1, 0.1, 0.1],  # 1
        [0.2, 0.2, 0.2, 0.3, 0.1],  # 0
        [0.4, 0.4, 0.1, 0.0, 0.1],  # 1
        [0.4, 0.3, 0.1, 0.1, 0.1],  # 0
        [0.4, 0.0, 0.5, 0.0, 0.1],  # 1
        [0.4, 0.1, 0.3, 0.1, 0.1],  # 1
    ],
    dtype=torch.float32,
    device=DEVICE)
UNIFORM_PROBS = torch.tensor([
    0.9,
    0.0,
    0.9,
    0.7,
    0.8,
    0.5,
    0.45,
    1.0,
    0.5,
    0.45,
    1.0,
    0.39,
    0.4,
    0.1,
    0.3,
],
                             dtype=torch.float32,
                             device=DEVICE)
BONUS_TOKEN_IDS = torch.full((BATCH_SIZE, ),
                             MAX_SPEC_LEN + 1,
                             dtype=torch.int64,
                             device=DEVICE)
RECOVERED_TOKEN_IDS = torch.full((NUM_TOKENS, ),
                                 MAX_SPEC_LEN,
                                 dtype=torch.int64,
                                 device=DEVICE)
IS_GREEDY = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=DEVICE)
IS_GREEDY[4] = True


@pytest.mark.parametrize("cu_num_draft_tokens", [CU_NUM_DRAFT_TOKENS])
@pytest.mark.parametrize("draft_token_ids", [DRAFT_TOKEN_IDS])
@pytest.mark.parametrize("draft_probs", [DRAFT_PROBS])
@pytest.mark.parametrize("target_probs", [TARGET_PROBS])
@pytest.mark.parametrize("bonus_token_ids", [BONUS_TOKEN_IDS])
@pytest.mark.parametrize("recovered_token_ids", [RECOVERED_TOKEN_IDS])
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
        recovered_token_ids,  # [num_tokens]
        uniform_probs,  # [num_tokens]
        is_greedy,  # [batch_size]
        batch_size,  # int
        max_spec_len,  # int
        vocab_size,  # int
) -> None:

    grid, block_size = cal_grid_and_block_size(batch_size)

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
        recovered_token_ids=recovered_token_ids,
        uniform_probs=uniform_probs,
        is_greedy=is_greedy,
        max_spec_len=max_spec_len,
        vocab_size=vocab_size,
        IS_NGRAM=draft_probs is None)

    rejection_random_sample_block_verify_kernel[(grid, )](
        output_token_ids_ptr=output_token_ids_triton,
        cu_num_draft_tokens_ptr=cu_num_draft_tokens,
        draft_token_ids_ptr=draft_token_ids,
        draft_probs_ptr=draft_probs,
        target_probs_ptr=target_probs,
        bonus_token_ids_ptr=bonus_token_ids,
        recovered_token_ids_ptr=recovered_token_ids,
        uniform_probs_ptr=uniform_probs,
        is_greedy_ptr=is_greedy,
        max_spec_len=max_spec_len,
        vocab_size=vocab_size,
        vec_len=batch_size,
        NO_DRAFT_PROBS=draft_probs is None,
        BLOCK_SIZE=block_size)
    torch.npu.synchronize()
    assert torch.equal(output_token_ids_ref, output_token_ids_triton)
