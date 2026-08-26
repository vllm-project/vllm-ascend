import gc

import pytest
import torch

from vllm_ascend.ops.triton.reject_sample import (
    cal_grid_and_block_size,
    expand_kernel,
    rejection_greedy_sample_spec_len_1_triton,
    rejection_greedy_sample_triton,
    rejection_random_sample_block_verify_kernel,
    rejection_random_sample_kernel,
    sample_recovered_tokens_kernel,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.sample.rejection_sampler import (
    expand_pytorch,
    rejection_greedy_sample_pytorch,
    rejection_greedy_sample_spec_len_1_pytorch,
    rejection_random_sample_block_verify_pytorch,
    rejection_random_sample_pytorch,
    sample_recovered_tokens_pytorch,
)

KERNEL_TEST_ITERS = 1


@pytest.fixture(scope="function", autouse=True)
def setup_device_properties():
    init_device_properties_triton()
    yield


@torch.inference_mode()
def test_expand_kernel():
    device = "npu"
    batch_size = 5
    max_num_tokens = 3

    x = torch.tensor([4.0, -1.0, -1.0, 7.0, 8.0], dtype=torch.float32, device=device)
    cu_num_tokens = torch.tensor([2, 2, 5, 6, 9], dtype=torch.int32, device=device)
    output = torch.full((9,), -5.0, dtype=torch.float32, device=device)
    grid, block_size = cal_grid_and_block_size(batch_size)

    for i in range(KERNEL_TEST_ITERS):
        output_ref = output.clone()
        output_triton = output.clone()

        expand_pytorch(
            output_ref,
            x,
            cu_num_tokens,
            -1.0,
            99.0,
            MAX_NUM_TOKENS=max_num_tokens,
        )
        expand_kernel[(grid,)](
            output_triton,
            x,
            cu_num_tokens,
            -1.0,
            99.0,
            batch_size,
            MAX_NUM_TOKENS=max_num_tokens,
            BLOCK_SIZE=block_size,
        )
        torch.npu.synchronize()
        assert torch.equal(output_ref, output_triton), f"iteration {i}"

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_sample_recovered_tokens_kernel():
    device = "npu"
    batch_size = 4
    max_spec_len = 3
    vocab_size = 5

    cu_num_draft_tokens = torch.tensor([2, 2, 5, 6], dtype=torch.int32, device=device)
    draft_token_ids = torch.tensor([1, 3, 0, 2, 4, 1], dtype=torch.int64, device=device)
    draft_probs = torch.tensor(
        [
            [0.05, 0.10, 0.10, 0.10, 0.10],
            [0.01, 0.01, 0.01, 0.01, 0.01],
            [0.05, 0.05, 0.05, 0.05, 0.05],
            [0.10, 0.09, 0.19, 0.29, 0.39],
            [0.09, 0.10, 0.19, 0.29, 0.39],
            [0.09, 0.19, 0.29, 0.39, 0.10],
        ],
        dtype=torch.float32,
        device=device,
    )
    target_probs = torch.tensor(
        [
            [0.10, 0.20, 0.90, 0.15, 0.12],
            [0.02, 0.02, 0.02, 0.03, 0.70],
            [0.10, 0.20, 0.30, 0.80, 0.40],
            [0.90, 0.10, 0.20, 0.30, 0.40],
            [0.10, 0.90, 0.20, 0.30, 0.40],
            [0.10, 0.20, 0.30, 0.40, 0.95],
        ],
        dtype=torch.float32,
        device=device,
    )
    q = torch.ones((batch_size, vocab_size), dtype=torch.float32, device=device)
    output_token_ids = torch.full_like(draft_token_ids, -1)

    for i in range(KERNEL_TEST_ITERS):
        output_token_ids_ref = output_token_ids.clone()
        output_token_ids_triton = output_token_ids.clone()

        sample_recovered_tokens_pytorch(
            output_token_ids_ref,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            IS_NGRAM=False,
        )
        sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
            output_token_ids_triton,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            NO_DRAFT_PROBS=False,
            SUB_BLOCK=8,
        )
        torch.npu.synchronize()
        assert torch.equal(output_token_ids_ref, output_token_ids_triton), f"iteration {i}"

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("synthetic_mode", [False, True])
@pytest.mark.parametrize("max_spec_len", [1, 2, 3])
@pytest.mark.parametrize("vocab_size", [1024])
@pytest.mark.parametrize("batch_size", [1, 256, 512, 1024])
@torch.inference_mode()
def test_rejection_random_sample(synthetic_mode, max_spec_len, vocab_size, batch_size):
    device = "npu"
    torch.manual_seed(0)
    draft_probs = torch.rand(batch_size * max_spec_len, vocab_size, dtype=torch.float32, device=device)
    target_probs = torch.rand(batch_size * max_spec_len, vocab_size, dtype=torch.float32, device=device)
    bonus_token_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, 1), dtype=torch.int64, device=device)
    draft_token_ids = torch.randint(
        low=0, high=vocab_size, size=(batch_size * max_spec_len,), dtype=torch.int64, device=device
    )
    output_token_ids = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int64, device=device)
    output_token_ids_ref = output_token_ids.clone()
    num_tokens = draft_token_ids.shape[0]
    uniform_probs = torch.rand((num_tokens,), dtype=torch.float32, device=device)
    num_draft_tokens = [max_spec_len] * batch_size
    num_draft_tokens = torch.tensor(num_draft_tokens, dtype=torch.int32, device=device)
    cu_num_draft_tokens = torch.cumsum(num_draft_tokens, dim=0, dtype=torch.int32)
    is_greedy_ptr = torch.full((batch_size,), False, dtype=torch.bool, device=device)
    recovered_ids = torch.zeros_like(draft_token_ids, dtype=torch.int64, device=device)
    grid, block_size = cal_grid_and_block_size(batch_size)
    # Synthetic mode accepts each draft token at position i with probability
    # rates[i], bypassing the draft/target prob comparison. The rates must be
    # length == max_spec_len and (per vLLM config validation) non-increasing.
    if synthetic_mode:
        synthetic_conditional_rates = torch.tensor(
            [0.9 - 0.3 * i for i in range(max_spec_len)],
            dtype=torch.float32,
            device=device,
        )
    else:
        synthetic_conditional_rates = None
    rejection_random_sample_pytorch(
        output_token_ids_ref,
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
        IS_NGRAM=draft_probs is None,
        synthetic_mode=synthetic_mode,
        synthetic_conditional_rates=synthetic_conditional_rates,
    )
    rejection_random_sample_kernel[(grid,)](
        output_token_ids,
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
        vocab_size,  # global_vocab_size
        batch_size,
        None,  # ori_target_probs
        synthetic_conditional_rates,  # synthetic_conditional_rates (None when off)
        NO_ORI_TARGET_PROBS=True,
        NO_DRAFT_PROBS=draft_probs is None,
        SYNTHETIC_MODE=synthetic_mode,
        ENTROPY_VERIFY=False,
        BLOCK_SIZE=block_size,
    )
    torch.npu.synchronize()
    assert torch.equal(output_token_ids_ref, output_token_ids), f"synthetic_mode={synthetic_mode}"
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("synthetic_mode", [False, True])
@torch.inference_mode()
def test_rejection_greedy_sample_spec_len_1_triton_kernel(synthetic_mode):
    device = "npu"
    batch_size = 5

    draft_token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device=device)
    target_argmax = torch.tensor([1, 0, 3, 7, 5], dtype=torch.int64, device=device)
    bonus_token_ids = torch.tensor([[11], [12], [13], [14], [15]], dtype=torch.int64, device=device)
    output_token_ids = torch.full((batch_size, 2), -1, dtype=torch.int64, device=device)
    grid, block_size = cal_grid_and_block_size(batch_size)

    # Synthetic mode accepts position 0 with prob rates[0] (uniform < rate),
    # independent of the target match. Mix accept/reject across the batch:
    # u=[0.3,0.6,0.4,0.7,0.2], rate=0.5 -> accept,reject,accept,reject,accept.
    if synthetic_mode:
        uniform_probs_t = torch.tensor([0.3, 0.6, 0.4, 0.7, 0.2], dtype=torch.float32, device=device)
        rates_t = torch.tensor([0.5], dtype=torch.float32, device=device)
    else:
        uniform_probs_t = None
        rates_t = None

    for i in range(KERNEL_TEST_ITERS):
        output_token_ids_ref = output_token_ids.clone()
        output_token_ids_triton = output_token_ids.clone()

        rejection_greedy_sample_spec_len_1_pytorch(
            output_token_ids_ref,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            uniform_probs=uniform_probs_t,
            synthetic_conditional_rates=rates_t,
            synthetic_mode=synthetic_mode,
        )
        rejection_greedy_sample_spec_len_1_triton[(grid,)](
            output_token_ids_triton,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            batch_size,
            uniform_probs_t,
            rates_t,
            SYNTHETIC_MODE=synthetic_mode,
            BLOCK_SIZE=block_size,
        )
        torch.npu.synchronize()
        assert torch.equal(output_token_ids_ref, output_token_ids_triton), (
            f"iteration {i}, synthetic_mode={synthetic_mode}"
        )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("synthetic_mode", [False, True])
@torch.inference_mode()
def test_rejection_greedy_sample_triton_kernel(synthetic_mode):
    device = "npu"
    batch_size = 6
    max_spec_len = 3
    draft_tokens_per_req = [3, 2, 1, 0, 3, 2]

    cu_num_draft_tokens = torch.tensor([3, 5, 6, 6, 9, 11], dtype=torch.int32, device=device)
    draft_token_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.int64, device=device)
    target_argmax = torch.tensor([1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11], dtype=torch.int64, device=device)
    bonus_token_ids = torch.tensor([[21], [22], [23], [24], [25], [26]], dtype=torch.int64, device=device)
    is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)
    output_token_ids = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int64, device=device)
    grid, block_size = cal_grid_and_block_size(batch_size)

    # Synthetic: rates = [0.8, 0.5, 0.2] (non-increasing). The 11 uniform
    # values mix accepts/rejects to exercise first-rejection: a rejected
    # position emits target_argmax and stops the request; all-accepted
    # requests append the bonus token.
    if synthetic_mode:
        uniform_probs_t = torch.tensor(
            [0.2, 0.4, 0.1, 0.6, 0.3, 0.9, 0.1, 0.7, 0.3, 0.1, 0.5],
            dtype=torch.float32,
            device=device,
        )
        rates_t = torch.tensor([0.8, 0.5, 0.2], dtype=torch.float32, device=device)
    else:
        uniform_probs_t = None
        rates_t = None

    for i in range(KERNEL_TEST_ITERS):
        output_token_ids_ref = output_token_ids.clone()
        output_token_ids_triton = output_token_ids.clone()

        rejection_greedy_sample_pytorch(
            output_token_ids_ref,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            draft_tokens_per_req,
            max_spec_len,
            is_greedy,
            uniform_probs=uniform_probs_t,
            synthetic_conditional_rates=rates_t,
            synthetic_mode=synthetic_mode,
        )
        rejection_greedy_sample_triton[(grid,)](
            output_token_ids_triton,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            batch_size,
            max_spec_len,
            uniform_probs_t,
            rates_t,
            SYNTHETIC_MODE=synthetic_mode,
            BLOCK_SIZE=block_size,
        )
        torch.npu.synchronize()
        assert torch.equal(output_token_ids_ref, output_token_ids_triton), (
            f"iteration {i}, synthetic_mode={synthetic_mode}"
        )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


# Boundary scenarios for the cu_num_draft_tokens[offset - 1] read in
# rejection_greedy_sample_triton. The old tl.where formulation evaluated the
# load unconditionally, so lane offset == 0 read one element before the
# buffer whenever it was active. Cover every lane-0 activation mode:
# - "all_greedy_none": is_greedy=None makes lane 0 always active (previously
#   an out-of-bounds read on every launch).
# - "all_greedy": is_greedy all True, same lane-0 activation via the tensor
#   path.
# - "mixed_first_greedy": request 0 greedy (lane 0 active, OOB in old code).
# - "mixed_first_random": request 0 sampling (lane 0 masked by is_greedy_mask).
# The batch mixes all-match (bonus), first-token reject, mid reject, and
# zero-draft-token requests.
@pytest.mark.parametrize(
    "is_greedy_pattern",
    ["all_greedy_none", "all_greedy", "mixed_first_greedy", "mixed_first_random"],
)
@torch.inference_mode()
def test_rejection_greedy_sample_triton_boundary(is_greedy_pattern):
    device = "npu"
    batch_size = 8
    max_spec_len = 3
    # Varying draft lengths, including a zero-draft-token request (index 4).
    draft_tokens_per_req = [3, 1, 2, 3, 0, 2, 1, 3]
    cu_num_draft_tokens = torch.tensor([3, 4, 6, 9, 9, 11, 12, 15], dtype=torch.int32, device=device)
    draft_token_ids = torch.tensor(
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=torch.int64, device=device
    )
    # Requests: 0 all-match, 1 match, 2 first-token reject, 3 mid reject,
    # 4 no tokens, 5 all-match, 6 match, 7 first-token reject.
    target_argmax = torch.tensor(
        [5, 6, 7, 8, 0, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19], dtype=torch.int64, device=device
    )
    bonus_token_ids = torch.arange(21, 29, dtype=torch.int64, device=device).unsqueeze(1)

    if is_greedy_pattern == "all_greedy_none":
        is_greedy = None
        greedy_rows = torch.ones(batch_size, dtype=torch.bool)
    elif is_greedy_pattern == "all_greedy":
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)
        greedy_rows = is_greedy.cpu()
    elif is_greedy_pattern == "mixed_first_greedy":
        is_greedy = torch.tensor([True, False, True, False, True, False, True, False], device=device)
        greedy_rows = is_greedy.cpu()
    else:  # mixed_first_random
        is_greedy = torch.tensor([False, True, False, True, False, True, False, True], device=device)
        greedy_rows = is_greedy.cpu()

    output_token_ids = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int64, device=device)
    grid, block_size = cal_grid_and_block_size(batch_size)

    output_ref = output_token_ids.clone()
    output_triton = output_token_ids.clone()

    rejection_greedy_sample_pytorch(
        output_ref,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        draft_tokens_per_req,
        max_spec_len,
        is_greedy,
    )
    rejection_greedy_sample_triton[(grid,)](
        output_triton,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy,
        batch_size,
        max_spec_len,
        None,  # uniform_probs (standard greedy path)
        None,  # synthetic_conditional_rates
        SYNTHETIC_MODE=False,
        BLOCK_SIZE=block_size,
    )
    torch.npu.synchronize()
    assert torch.equal(output_ref, output_triton), f"is_greedy_pattern={is_greedy_pattern}"

    # Non-greedy rows are owned by the random-sampling kernel and must stay
    # untouched (sentinel preserved). The all-greedy pattern has no such rows,
    # so the check is skipped.
    if is_greedy is not None:
        non_greedy_rows = ~greedy_rows
        if non_greedy_rows.any():
            assert (output_triton[non_greedy_rows.to(device)] == -1).all()

    # Spot-check greedy rows: an all-match request emits the bonus token; a
    # first-token rejection emits target_argmax and leaves the tail untouched.
    if greedy_rows[0]:
        assert output_triton[0].tolist() == [5, 6, 7, 21]
    if greedy_rows[7]:
        assert output_triton[7].tolist() == [0, -1, -1, -1]

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


# Randomized mixed batch large enough to force BLOCK_SIZE > 1, so the
# offset == 0 lane shares a program block with other lanes. Covers the
# multi-lane layout where only lane 0 of the first block needs masking.
@pytest.mark.parametrize("first_req_greedy", [True, False])
@torch.inference_mode()
def test_rejection_greedy_sample_triton_boundary_multilane(first_req_greedy):
    device = "npu"
    torch.manual_seed(1234)
    batch_size = 256  # > vector core count on all Ascend hardware
    max_spec_len = 4
    vocab_size = 32

    lens = torch.randint(0, max_spec_len + 1, (batch_size,), dtype=torch.int32)
    lens[0] = max_spec_len  # force an all-match request for the bonus path
    cu_num_draft_tokens = torch.cumsum(lens, dim=0, dtype=torch.int32).to(device)
    draft_tokens_per_req = lens.tolist()
    num_tokens = int(lens.sum())

    draft_token_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64, device=device)
    target_argmax = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64, device=device)
    # Make request 0 all-match so its bonus path is exercised.
    end0 = int(cu_num_draft_tokens[0])
    target_argmax[:end0] = draft_token_ids[:end0]
    bonus_token_ids = torch.randint(0, vocab_size, (batch_size, 1), dtype=torch.int64, device=device)
    is_greedy = (torch.rand(batch_size) < 0.5).to(device)
    is_greedy[0] = first_req_greedy

    output_token_ids = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int64, device=device)
    grid, block_size = cal_grid_and_block_size(batch_size)
    assert block_size > 1

    output_ref = output_token_ids.clone()
    output_triton = output_token_ids.clone()

    rejection_greedy_sample_pytorch(
        output_ref,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        draft_tokens_per_req,
        max_spec_len,
        is_greedy,
    )
    rejection_greedy_sample_triton[(grid,)](
        output_triton,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy,
        batch_size,
        max_spec_len,
        None,
        None,
        SYNTHETIC_MODE=False,
        BLOCK_SIZE=block_size,
    )
    torch.npu.synchronize()
    assert torch.equal(output_ref, output_triton), f"first_req_greedy={first_req_greedy}"
    # All-match request 0 appends the bonus token when it is greedy.
    if first_req_greedy:
        assert output_triton[0, max_spec_len].item() == bonus_token_ids[0, 0].item()

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


# Boundary scenarios for the cu_num_draft_tokens[offsets - 1] read in
# rejection_random_sample_kernel. The mask there is not_greedy_mask, so lane 0
# was active (and the old tl.where formulation read one element before the
# buffer) whenever request 0 was a sampling request. Cover:
# - "all_random": every request sampling (lane 0 always active).
# - "mixed_first_random": request 0 sampling, greedy rows interleaved.
# - "mixed_first_greedy": request 0 greedy (lane 0 masked by not_greedy_mask).
# Also exercises a -1 draft placeholder (always rejected -> recovered token),
# a forced all-accept request (bonus), and a zero-draft-token request.
@pytest.mark.parametrize("is_greedy_pattern", ["all_random", "mixed_first_random", "mixed_first_greedy"])
@torch.inference_mode()
def test_rejection_random_sample_boundary(is_greedy_pattern):
    device = "npu"
    torch.manual_seed(5678)
    batch_size = 8
    max_spec_len = 3
    vocab_size = 5
    cu_num_draft_tokens = torch.tensor([3, 4, 6, 9, 9, 11, 12, 15], dtype=torch.int32, device=device)
    num_tokens = 15

    draft_token_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64, device=device)
    draft_probs = torch.rand(num_tokens, vocab_size, dtype=torch.float32, device=device) + 0.01
    target_probs = torch.rand(num_tokens, vocab_size, dtype=torch.float32, device=device)
    uniform_probs = torch.rand(num_tokens, dtype=torch.float32, device=device)
    recovered_token_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64, device=device)
    bonus_token_ids = torch.randint(0, vocab_size, (batch_size, 1), dtype=torch.int64, device=device)

    # Force deterministic paths on top of the random data:
    # - Request 0 (tokens 0..2): all-accept -> bonus token at position 3.
    for t in range(3):
        d = draft_token_ids[t]
        draft_probs[t, :] = 0.01
        draft_probs[t, d] = 0.5
        target_probs[t, :] = 0.0
        target_probs[t, d] = 0.5
        uniform_probs[t] = 0.1  # 0.5 / 0.5 = 1.0 >= 0.1 -> accept
    # - Token 6 (request 3, position 0): -1 placeholder -> always rejected,
    #   emits the recovered token.
    draft_token_ids[6] = -1

    if is_greedy_pattern == "all_random":
        is_greedy = torch.zeros(batch_size, dtype=torch.bool, device=device)
    elif is_greedy_pattern == "mixed_first_random":
        is_greedy = torch.tensor([False, True, False, True, False, True, False, True], device=device)
    else:  # mixed_first_greedy
        is_greedy = torch.tensor([True, False, True, False, True, False, True, False], device=device)

    output_token_ids = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int64, device=device)
    grid, block_size = cal_grid_and_block_size(batch_size)

    output_ref = output_token_ids.clone()
    output_triton = output_token_ids.clone()

    rejection_random_sample_pytorch(
        output_ref,
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
    )
    rejection_random_sample_kernel[(grid,)](
        output_triton,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        None,  # target_indices
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        vocab_size,  # global_vocab_size
        batch_size,
        None,  # ori_target_probs
        None,  # synthetic_conditional_rates
        NO_ORI_TARGET_PROBS=True,
        NO_DRAFT_PROBS=False,
        ENABLE_REDUCE_SAMPLING=False,
        SYNTHETIC_MODE=False,
        ENTROPY_VERIFY=False,
        BLOCK_SIZE=block_size,
    )
    torch.npu.synchronize()
    assert torch.equal(output_ref, output_triton), f"is_greedy_pattern={is_greedy_pattern}"

    # Greedy rows are owned by the greedy kernel and must stay untouched.
    greedy_rows = is_greedy.cpu()
    if greedy_rows.any():
        assert (output_triton[greedy_rows.to(device)] == -1).all()

    # Spot checks, only meaningful when the row is a sampling request.
    if not greedy_rows[0]:
        # Request 0: all three draft tokens accepted + bonus at position 3.
        assert output_triton[0].tolist() == [
            draft_token_ids[0].item(),
            draft_token_ids[1].item(),
            draft_token_ids[2].item(),
            bonus_token_ids[0, 0].item(),
        ]
    if not greedy_rows[3]:
        # Request 3 starts with a -1 placeholder: rejected, emits recovered.
        assert output_triton[3, 0].item() == recovered_token_ids[6].item()

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


DEVICE = "npu"
BATCH_SIZE = 7
MAX_SPEC_LEN = 3
VOCAB_SIZE = 5
CU_NUM_DRAFT_TOKENS = torch.tensor([2, 2, 5, 8, 11, 14, 15], dtype=torch.int32, device=DEVICE)
DRAFT_TOKEN_IDS = torch.tensor([0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=torch.int64, device=DEVICE)
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
    device=DEVICE,
)
UNIFORM_PROBS = torch.tensor(
    [
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
    device=DEVICE,
)
BONUS_TOKEN_IDS = torch.full((BATCH_SIZE,), MAX_SPEC_LEN + 1, dtype=torch.int64, device=DEVICE)
RECOVERED_TOKEN_IDS = torch.full((NUM_TOKENS,), MAX_SPEC_LEN, dtype=torch.int64, device=DEVICE)
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

    output_token_ids_ref = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int64, device=DEVICE)

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
        IS_NGRAM=draft_probs is None,
    )

    rejection_random_sample_block_verify_kernel[(grid,)](
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
        global_vocab_size=vocab_size,
        vec_len=batch_size,
        ori_target_probs_ptr=None,
        NO_ORI_TARGET_PROBS=True,
        NO_DRAFT_PROBS=draft_probs is None,
        BLOCK_SIZE=block_size,
        ENTROPY_VERIFY=False,
        SUB_BLOCK=32,
    )
    torch.npu.synchronize()
    assert torch.equal(output_token_ids_ref, output_token_ids_triton)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
