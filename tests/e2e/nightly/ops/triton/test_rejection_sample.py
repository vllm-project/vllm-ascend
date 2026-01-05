import pytest
import torch
from vllm.v1.sample.rejection_sampler import \
    rejection_random_sample_kernel as original_rejection_random_sample_kernel

from vllm_ascend.ops.triton.reject_sample import (
    cal_grid_and_block_size, rejection_random_sample_kernel)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


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
