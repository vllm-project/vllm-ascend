import pytest
import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore
from vllm.v1.sample.rejection_sampler import \
    rejection_random_sample_kernel as original_rejection_random_sample_kernel

from vllm_ascend.ops.triton.reject_sample import (
    cal_grid_and_block_size, rejection_random_sample_block_verify_kernel,
    rejection_random_sample_kernel, sample_recovered_tokens_kernel)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.sample.rejection_sampler import \
    rejection_random_sample_block_verify_pytorch


@pytest.fixture(scope="function", autouse=True)
def setup_device_properties():
    init_device_properties_triton()
    yield


@triton.jit
def original_sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
    SUB_BLOCK: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr +
                                               req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    loop = (vocab_size + SUB_BLOCK - 1) // SUB_BLOCK
    global_recovered_id = -1
    global_max_p = -1.0
    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
        orig_prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size +
                            draft_token_id)
        # Temporarily zero out the probability of the draft token.
        # This is essentially the same as target_prob - draft_prob, except that
        # n-gram does not have draft_prob. We regard it as 1.
        tl.store(
            target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id,
            0)
        for loop_i in range(loop):
            vocab_start = loop_i * SUB_BLOCK
            vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
            prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size +
                           vocab_offset,
                           mask=vocab_offset < vocab_size,
                           other=0)
            q = tl.load(q_ptr + req_idx * vocab_size + vocab_offset,
                        mask=vocab_offset < vocab_size,
                        other=float("-inf"))
            new_p = prob / q
            recovered_id = tl.argmax(new_p, axis=-1)
            max_p = tl.get_element(new_p, (recovered_id, ))
            if max_p > global_max_p:
                global_max_p = max_p
                global_recovered_id = vocab_start + recovered_id
    else:
        for loop_i in range(loop):
            vocab_start = loop_i * SUB_BLOCK
            vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
            draft_prob = tl.load(draft_probs_ptr +
                                 (start_idx + pos) * vocab_size + vocab_offset,
                                 mask=vocab_offset < vocab_size,
                                 other=0)
            target_prob = tl.load(target_probs_ptr +
                                  (start_idx + pos) * vocab_size +
                                  vocab_offset,
                                  mask=vocab_offset < vocab_size,
                                  other=0)
            prob = tl.maximum(target_prob - draft_prob, 0)
            # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
            # `tl.argmax` will select the maximum value.

            q = tl.load(q_ptr + req_idx * vocab_size + vocab_offset,
                        mask=vocab_offset < vocab_size,
                        other=float("-inf"))
            new_p = prob / q
            recovered_id = tl.argmax(new_p, axis=-1)
            max_p = tl.get_element(new_p, (recovered_id, ))
            if max_p > global_max_p:
                global_max_p = max_p
                global_recovered_id = vocab_start + recovered_id

    tl.store(output_token_ids_ptr + start_idx + pos, global_recovered_id)

    if NO_DRAFT_PROBS:
        # Restore the original probability.
        tl.store(
            target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id,
            orig_prob)


@pytest.mark.parametrize("max_spec_len", [1, 2, 3])
@pytest.mark.parametrize("vocab_size", [151_936])
@pytest.mark.parametrize("batch_size", [1, 8, 32, 64, 128, 256, 512, 1024])
@torch.inference_mode()
def test_sample_recovered_tokens_kernel(max_spec_len, vocab_size, batch_size):
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
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()

    draft_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size * max_spec_len, ),
                                    dtype=torch.int64,
                                    device=device)
    recovered_token_ids = torch.zeros_like(draft_token_ids,
                                           dtype=torch.int64,
                                           device=device)

    original_recovered_token_ids = recovered_token_ids.clone()

    num_draft_tokens = [max_spec_len] * batch_size
    num_draft_tokens = torch.tensor(num_draft_tokens,
                                    dtype=torch.int32,
                                    device=device)
    cu_num_draft_tokens = torch.cumsum(num_draft_tokens,
                                       dim=0,
                                       dtype=torch.int32)

    original_sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        original_recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        triton.next_power_of_2(vocab_size),
        NO_DRAFT_PROBS=draft_probs is None,
        SUB_BLOCK=4 * 1024,
        multibuffer=False,
    )
    grid, block_size = cal_grid_and_block_size(batch_size)
    sample_recovered_tokens_kernel[(grid, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        batch_size,
        block_size,
        triton.next_power_of_2(vocab_size),
        NO_DRAFT_PROBS=draft_probs is None,
        VOCAB_BLOCK_SIZE=4 * 1024,
        multibuffer=False,
    )

    torch.npu.synchronize()
    assert torch.equal(original_recovered_token_ids, recovered_token_ids)


@pytest.mark.parametrize("max_spec_len", [1, 2, 3])
@pytest.mark.parametrize("vocab_size", [151_936])
@pytest.mark.parametrize("batch_size", [1, 8, 32, 64, 128, 256, 512, 1024])
@torch.inference_mode()
def test_rejection_random_sample_kernel(max_spec_len, vocab_size, batch_size):
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
