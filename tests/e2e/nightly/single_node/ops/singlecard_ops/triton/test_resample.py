# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Numerical tests for MRV2 categorical resampling on Ascend NPU.
#
# The tests exercise the public resample() API and use independent
# PyTorch/analytic references; no Triton implementation is reproduced here.

import gc
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.v2.spec_decode.resample import resample
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample

DEVICE = "npu"
STAT_TRIALS = 4096
STAT_ATOL = 0.03
VOCAB_BLOCK_SIZE = 8192


@pytest.fixture(scope="module", autouse=True)
def _npu_env():
    init_device_properties_triton()
    yield
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()


def _cu_num_logits(lengths: list[int]) -> torch.Tensor:
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return torch.tensor(cu, dtype=torch.int32, device=DEVICE)


def _expanded_mapping(lengths: list[int], req_state_rows: list[int] | None = None) -> torch.Tensor:
    if req_state_rows is None:
        req_state_rows = list(range(len(lengths)))
    mapping = torch.empty(sum(lengths), dtype=torch.int32)
    offset = 0
    for length, row in zip(lengths, req_state_rows):
        mapping[offset : offset + length] = row
        offset += length
    return mapping.to(DEVICE)


def _request_state_tensors(req_state_rows: list[int], temperatures: list[float]):
    max_num_reqs = max(req_state_rows) + 1
    temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=DEVICE)
    seed = torch.arange(max_num_reqs, dtype=torch.int64, device=DEVICE) * 104729 + 17
    for row, value in zip(req_state_rows, temperatures):
        temperature[row] = value
    return temperature, seed


def _position_tensor(num_logits: int) -> torch.Tensor:
    # Keep positions well inside int32 because the NPU RNG path casts them to int32.
    return torch.arange(num_logits, dtype=torch.int64, device=DEVICE) * 7 + 11


def _assert_distribution(samples: torch.Tensor, expected: torch.Tensor, atol: float = STAT_ATOL) -> None:
    samples_cpu = samples.cpu()
    assert int(samples_cpu.min()) >= 0
    assert int(samples_cpu.max()) < expected.numel()
    counts = torch.bincount(samples_cpu, minlength=expected.numel()).to(torch.float64)
    observed = counts / counts.sum()
    torch.testing.assert_close(observed, expected.to(torch.float64).cpu(), rtol=0.0, atol=atol)


def _repeat_rows(rows: torch.Tensor, repeats: int) -> torch.Tensor:
    return rows.unsqueeze(0).expand(repeats, -1).contiguous()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_random_bonus_single_support_accepts_logits_dtypes(dtype):
    """The operator loads fp16/bf16/fp32 logits and accumulates categorical mass in fp32."""
    vocab_size = 1025
    lengths = [1, 1, 1]
    req_state_rows = [2, 0, 1]
    expected_ids = [1024, 17, 1023]
    expected = torch.tensor(expected_ids, dtype=torch.int64, device=DEVICE)

    target_logits = torch.full((3, vocab_size), float("-inf"), dtype=dtype, device=DEVICE)
    target_logits[0, expected_ids[0]] = 0.0
    target_logits[1, expected_ids[1]] = 1.0
    target_logits[2, expected_ids[2]] = -1.0

    sampled = torch.full((3, 1), -1, dtype=torch.int64, device=DEVICE)
    num_sampled = torch.zeros(3, dtype=torch.int32, device=DEVICE)
    cu_num_logits = _cu_num_logits(lengths)
    expanded_idx_mapping = _expanded_mapping(lengths, req_state_rows)
    draft_sampled = torch.zeros(3, dtype=torch.int32, device=DEVICE)
    temperature, seed = _request_state_tensors(req_state_rows, [1.0, 0.7, 1.3])
    lse = torch.zeros(3, dtype=torch.float32, device=DEVICE)

    resample(
        sampled,
        num_sampled,
        target_logits,
        lse,
        None,
        lse,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        _position_tensor(3),
    )
    torch.npu.synchronize()

    assert torch.equal(sampled[:, 0], expected)
    assert torch.equal(num_sampled, torch.ones_like(num_sampled))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_full_draft_residual_accepts_logits_dtypes(dtype):
    """Full-draft residual supports fp16/bf16/fp32 target and draft logits with fp32 mass computation."""
    vocab_size = 4
    target_row = torch.tensor([2.0, -10.0, -10.0, -2.0], dtype=dtype, device=DEVICE)
    draft_row = torch.tensor([-2.0, -10.0, -10.0, 2.0], dtype=dtype, device=DEVICE)
    target_logits = torch.stack([target_row, target_row])
    draft_logits = draft_row.reshape(1, 1, vocab_size)
    target_lse = torch.logsumexp(target_row.float(), dim=0).reshape(1)
    draft_lse = torch.logsumexp(draft_row.float(), dim=0).reshape(1)
    sampled = torch.full((1, 2), -1, dtype=torch.int64, device=DEVICE)
    num_sampled = torch.zeros(1, dtype=torch.int32, device=DEVICE)

    resample(
        sampled,
        num_sampled,
        target_logits,
        target_lse,
        draft_logits,
        draft_lse,
        _cu_num_logits([2]),
        _expanded_mapping([2]),
        torch.zeros(2, dtype=torch.int32, device=DEVICE),
        torch.ones(1, dtype=torch.float32, device=DEVICE),
        torch.tensor([211], dtype=torch.int64, device=DEVICE),
        _position_tensor(2),
        has_draft_logits=True,
    )
    torch.npu.synchronize()

    assert sampled[0, 0].item() == 0
    assert num_sampled[0].item() == 1


@torch.inference_mode()
def test_partial_negative_inf_has_zero_probability_mass():
    """Masked -inf vocabulary entries are valid zero-mass tokens and are never sampled."""
    probs = torch.tensor([0.25, 0.0, 0.75, 0.0, 0.0], dtype=torch.float32)
    logits = torch.where(probs > 0, torch.log(probs), torch.tensor(float("-inf")))
    target_logits = _repeat_rows(logits, STAT_TRIALS).to(DEVICE)
    sampled = torch.full((STAT_TRIALS, 1), -1, dtype=torch.int64, device=DEVICE)
    num_sampled = torch.zeros(STAT_TRIALS, dtype=torch.int32, device=DEVICE)
    zeros = torch.zeros(STAT_TRIALS, dtype=torch.float32, device=DEVICE)

    resample(
        sampled,
        num_sampled,
        target_logits,
        zeros,
        None,
        zeros,
        _cu_num_logits([1] * STAT_TRIALS),
        _expanded_mapping([1] * STAT_TRIALS),
        torch.zeros(STAT_TRIALS, dtype=torch.int32, device=DEVICE),
        torch.ones(STAT_TRIALS, dtype=torch.float32, device=DEVICE),
        torch.arange(STAT_TRIALS, dtype=torch.int64, device=DEVICE) + 401,
        _position_tensor(STAT_TRIALS),
    )
    torch.npu.synchronize()

    assert not bool(((sampled[:, 0] == 1) | (sampled[:, 0] == 3) | (sampled[:, 0] == 4)).any())
    _assert_distribution(sampled[:, 0], probs)


@torch.inference_mode()
def test_ragged_requests_and_shuffled_mapping_select_correct_rows():
    """Ragged cu_num_logits and shuffled request-state mappings select the intended resample row per request."""
    lengths = [2, 4, 3]
    rejected_step_values = [0, 2, 2]
    rejected_steps = torch.tensor(rejected_step_values, dtype=torch.int32, device=DEVICE)
    req_state_rows = [4, 1, 3]
    vocab_size = 19
    expected_ids = [3, 11, 18]
    target_logits = torch.full((sum(lengths), vocab_size), float("-inf"), dtype=torch.float32, device=DEVICE)
    cu = [0, 2, 6, 9]
    for req_idx, token_id in enumerate(expected_ids):
        target_logits[cu[req_idx] + rejected_step_values[req_idx], token_id] = 0.0

    sampled = torch.full((3, 5), -1, dtype=torch.int64, device=DEVICE)
    draft_sampled = torch.zeros(sum(lengths), dtype=torch.int32, device=DEVICE)
    temperature, seed = _request_state_tensors(req_state_rows, [0.7, 1.0, 1.3])
    zeros = torch.zeros(3, dtype=torch.float32, device=DEVICE)

    resample(
        sampled,
        rejected_steps,
        target_logits,
        zeros,
        None,
        zeros,
        _cu_num_logits(lengths),
        _expanded_mapping(lengths, req_state_rows),
        draft_sampled,
        temperature,
        seed,
        _position_tensor(sum(lengths)),
    )
    torch.npu.synchronize()

    for req_idx, (step, token_id) in enumerate(zip(rejected_step_values, expected_ids)):
        assert sampled[req_idx, step].item() == token_id
    assert torch.equal(rejected_steps, torch.tensor([1, 3, 3], dtype=torch.int32, device=DEVICE))


@pytest.mark.parametrize("vocab_size", [1023, 1024, 1025])
@torch.inference_mode()
def test_greedy_bonus_handles_block_tail(vocab_size):
    """Cover BLOCK_SIZE - 1 / BLOCK_SIZE / BLOCK_SIZE + 1 and exact global argmax selection."""
    target_logits = torch.arange(vocab_size, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    sampled = torch.full((1, 1), -1, dtype=torch.int64, device=DEVICE)
    num_sampled = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    zeros = torch.zeros(1, dtype=torch.float32, device=DEVICE)

    resample(
        sampled,
        num_sampled,
        target_logits,
        zeros,
        None,
        zeros,
        _cu_num_logits([1]),
        _expanded_mapping([1]),
        torch.zeros(1, dtype=torch.int32, device=DEVICE),
        torch.zeros(1, dtype=torch.float32, device=DEVICE),
        torch.tensor([19], dtype=torch.int64, device=DEVICE),
        _position_tensor(1),
    )
    torch.npu.synchronize()

    assert sampled[0, 0].item() == vocab_size - 1
    assert num_sampled[0].item() == 1


@torch.inference_mode()
def test_greedy_non_bonus_preserves_verification_result():
    """Greedy rejection must not read unwritten workspaces or overwrite the target argmax from verification."""
    lengths = [2, 3]
    num_reqs = len(lengths)
    vocab_size = 33
    target_logits = torch.randn(sum(lengths), vocab_size, dtype=torch.float32, device=DEVICE)
    sampled = torch.full((num_reqs, 3), -777, dtype=torch.int64, device=DEVICE)
    num_sampled = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    sampled[0, 0] = 11
    sampled[1, 1] = 22
    zeros = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)

    resample(
        sampled,
        num_sampled,
        target_logits,
        zeros,
        None,
        zeros,
        _cu_num_logits(lengths),
        _expanded_mapping(lengths),
        torch.zeros(sum(lengths), dtype=torch.int32, device=DEVICE),
        torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE),
        torch.tensor([101, 103], dtype=torch.int64, device=DEVICE),
        _position_tensor(sum(lengths)),
    )
    torch.npu.synchronize()

    assert sampled[0, 0].item() == 11
    assert sampled[1, 1].item() == 22
    assert torch.equal(num_sampled, torch.tensor([1, 2], dtype=torch.int32, device=DEVICE))


@torch.inference_mode()
def test_one_hot_none_and_dummy_tensor_are_semantically_identical():
    """Support both direct draft_logits=None and the upper-layer dummy-tensor compatibility path."""
    vocab_size = 5
    probs = torch.tensor([0.0, 0.8, 0.0, 0.2, 0.0], dtype=torch.float32)
    target_row = torch.where(probs > 0, torch.log(probs), torch.tensor(float("-inf")))
    target_logits = torch.stack([target_row, torch.zeros_like(target_row)]).to(DEVICE)
    draft_sampled = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    cu_num_logits = _cu_num_logits([2])
    expanded_idx_mapping = _expanded_mapping([2])
    temperature = torch.ones(1, dtype=torch.float32, device=DEVICE)
    seed = torch.tensor([123], dtype=torch.int64, device=DEVICE)
    pos = _position_tensor(2)
    lse = torch.zeros(1, dtype=torch.float32, device=DEVICE)

    sampled_none = torch.full((1, 2), -1, dtype=torch.int64, device=DEVICE)
    count_none = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    resample(
        sampled_none,
        count_none,
        target_logits,
        lse,
        None,
        lse,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        pos,
    )

    sampled_dummy = torch.full((1, 2), -1, dtype=torch.int64, device=DEVICE)
    count_dummy = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    dummy = target_logits.new_empty(1, 1, 1)
    resample(
        sampled_dummy,
        count_dummy,
        target_logits,
        lse,
        dummy,
        lse,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        pos,
        has_draft_logits=False,
    )
    torch.npu.synchronize()

    assert sampled_none[0, 0].item() == 3
    assert torch.equal(sampled_none, sampled_dummy)
    assert torch.equal(count_none, count_dummy)


@torch.inference_mode()
def test_full_draft_residual_uses_request_state_mapping():
    """req_idx and req_state_idx are different address spaces; shuffled mappings must select the right draft row."""
    lengths = [2, 2]
    req_state_rows = [3, 1]
    vocab_size = 4
    p = torch.tensor([[0.2, 0.5, 0.2, 0.1], [0.1, 0.2, 0.6, 0.1]], dtype=torch.float32)
    q = torch.tensor([[0.3, 0.4, 0.2, 0.1], [0.1, 0.3, 0.5, 0.1]], dtype=torch.float32)
    expected = torch.tensor([1, 2], dtype=torch.int64, device=DEVICE)

    target_logits = torch.empty(sum(lengths), vocab_size, dtype=torch.float32, device=DEVICE)
    target_logits[0] = p[0].log().to(DEVICE)
    target_logits[1].zero_()
    target_logits[2] = p[1].log().to(DEVICE)
    target_logits[3].zero_()

    draft_logits = torch.zeros(4, 1, vocab_size, dtype=torch.float32, device=DEVICE)
    draft_logits[3, 0] = q[0].log().to(DEVICE)
    draft_logits[1, 0] = q[1].log().to(DEVICE)
    sampled = torch.full((2, 2), -1, dtype=torch.int64, device=DEVICE)
    num_sampled = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    zeros = torch.zeros(2, dtype=torch.float32, device=DEVICE)
    temperature, seed = _request_state_tensors(req_state_rows, [1.0, 1.0])

    resample(
        sampled,
        num_sampled,
        target_logits,
        zeros,
        draft_logits,
        zeros,
        _cu_num_logits(lengths),
        _expanded_mapping(lengths, req_state_rows),
        torch.zeros(sum(lengths), dtype=torch.int32, device=DEVICE),
        temperature,
        seed,
        _position_tensor(sum(lengths)),
        has_draft_logits=True,
    )
    torch.npu.synchronize()

    assert torch.equal(sampled[:, 0], expected)
    assert torch.equal(num_sampled, torch.ones_like(num_sampled))


@torch.inference_mode()
def test_random_bonus_distribution_matches_softmax():
    probs = torch.tensor([0.05, 0.15, 0.40, 0.25, 0.15], dtype=torch.float32)
    target_logits = _repeat_rows(probs.log(), STAT_TRIALS).to(DEVICE)
    lengths = [1] * STAT_TRIALS
    sampled = torch.full((STAT_TRIALS, 1), -1, dtype=torch.int64, device=DEVICE)
    num_sampled = torch.zeros(STAT_TRIALS, dtype=torch.int32, device=DEVICE)
    zeros = torch.zeros(STAT_TRIALS, dtype=torch.float32, device=DEVICE)
    temperature = torch.ones(STAT_TRIALS, dtype=torch.float32, device=DEVICE)
    seed = torch.arange(STAT_TRIALS, dtype=torch.int64, device=DEVICE) + 1009

    resample(
        sampled,
        num_sampled,
        target_logits,
        zeros,
        None,
        zeros,
        _cu_num_logits(lengths),
        _expanded_mapping(lengths),
        torch.zeros(STAT_TRIALS, dtype=torch.int32, device=DEVICE),
        temperature,
        seed,
        _position_tensor(STAT_TRIALS),
    )
    torch.npu.synchronize()
    _assert_distribution(sampled[:, 0], probs)


@torch.inference_mode()
def test_full_draft_residual_distribution_matches_positive_p_minus_q():
    p = torch.tensor([0.35, 0.30, 0.20, 0.10, 0.05], dtype=torch.float32)
    q = torch.tensor([0.10, 0.35, 0.15, 0.15, 0.25], dtype=torch.float32)
    residual = torch.clamp(p - q, min=0.0)
    expected = residual / residual.sum()

    lengths = [2] * STAT_TRIALS
    total_logits = 2 * STAT_TRIALS
    target_logits = torch.empty(total_logits, p.numel(), dtype=torch.float32, device=DEVICE)
    target_logits[0::2] = p.log().to(DEVICE)
    target_logits[1::2] = p.log().to(DEVICE)
    draft_logits = _repeat_rows(q.log(), STAT_TRIALS).unsqueeze(1).to(DEVICE)
    sampled = torch.full((STAT_TRIALS, 2), -1, dtype=torch.int64, device=DEVICE)
    num_sampled = torch.zeros(STAT_TRIALS, dtype=torch.int32, device=DEVICE)
    zeros = torch.zeros(STAT_TRIALS, dtype=torch.float32, device=DEVICE)
    temperature = torch.ones(STAT_TRIALS, dtype=torch.float32, device=DEVICE)
    seed = torch.arange(STAT_TRIALS, dtype=torch.int64, device=DEVICE) + 2017

    resample(
        sampled,
        num_sampled,
        target_logits,
        zeros,
        draft_logits,
        zeros,
        _cu_num_logits(lengths),
        _expanded_mapping(lengths),
        torch.zeros(total_logits, dtype=torch.int32, device=DEVICE),
        temperature,
        seed,
        _position_tensor(total_logits),
        has_draft_logits=True,
    )
    torch.npu.synchronize()
    _assert_distribution(sampled[:, 0], expected)


@torch.inference_mode()
def test_one_hot_residual_distribution_excludes_rejected_token_and_renormalizes():
    probs = torch.tensor([0.10, 0.45, 0.20, 0.15, 0.10], dtype=torch.float32)
    rejected_token = 1
    expected = probs.clone()
    expected[rejected_token] = 0.0
    expected /= expected.sum()

    lengths = [2] * STAT_TRIALS
    total_logits = 2 * STAT_TRIALS
    target_logits = torch.empty(total_logits, probs.numel(), dtype=torch.float32, device=DEVICE)
    target_logits[0::2] = probs.log().to(DEVICE)
    target_logits[1::2] = probs.log().to(DEVICE)
    draft_sampled = torch.zeros(total_logits, dtype=torch.int32, device=DEVICE)
    draft_sampled[1::2] = rejected_token
    sampled = torch.full((STAT_TRIALS, 2), -1, dtype=torch.int64, device=DEVICE)
    num_sampled = torch.zeros(STAT_TRIALS, dtype=torch.int32, device=DEVICE)
    zeros = torch.zeros(STAT_TRIALS, dtype=torch.float32, device=DEVICE)
    temperature = torch.ones(STAT_TRIALS, dtype=torch.float32, device=DEVICE)
    seed = torch.arange(STAT_TRIALS, dtype=torch.int64, device=DEVICE) + 3001

    resample(
        sampled,
        num_sampled,
        target_logits,
        zeros,
        None,
        zeros,
        _cu_num_logits(lengths),
        _expanded_mapping(lengths),
        draft_sampled,
        temperature,
        seed,
        _position_tensor(total_logits),
    )
    torch.npu.synchronize()

    assert not bool((sampled[:, 0] == rejected_token).any())
    _assert_distribution(sampled[:, 0], expected)


@torch.inference_mode()
def test_business_vocab_shape_single_support():
    """Exercise the current business vocabulary shape with multiple tasks per Vector Core worker."""
    num_reqs = 8
    vocab_size = 151936
    expected_ids = [0, 1023, 1024, 8191, 8192, 65535, 100001, vocab_size - 1]
    target_logits = torch.full((num_reqs, vocab_size), float("-inf"), dtype=torch.float32, device=DEVICE)
    for req_idx, token_id in enumerate(expected_ids):
        target_logits[req_idx, token_id] = float(req_idx)

    sampled = torch.full((num_reqs, 1), -1, dtype=torch.int64, device=DEVICE)
    num_sampled = torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE)
    zeros = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)
    temperature = torch.ones(num_reqs, dtype=torch.float32, device=DEVICE)
    seed = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE) + 701

    resample(
        sampled,
        num_sampled,
        target_logits,
        zeros,
        None,
        zeros,
        _cu_num_logits([1] * num_reqs),
        _expanded_mapping([1] * num_reqs),
        torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        temperature,
        seed,
        _position_tensor(num_reqs),
    )
    torch.npu.synchronize()

    expected = torch.tensor(expected_ids, dtype=torch.int64, device=DEVICE)
    assert torch.equal(sampled[:, 0], expected)
    assert torch.equal(num_sampled, torch.ones_like(num_sampled))


@torch.inference_mode()
def test_mixed_greedy_bonus_random_bonus_and_residual_in_one_launch():
    """Inactive workspace cells must stay masked when different request branches share one launch."""
    lengths = [2, 1, 1, 2]
    vocab_size = 9
    target_logits = torch.zeros(sum(lengths), vocab_size, dtype=torch.float32, device=DEVICE)

    # req0: greedy non-bonus, sampled[0, 0] already belongs to the verification kernel.
    # req1: greedy bonus with exact argmax 5.
    target_logits[2] = torch.arange(vocab_size, dtype=torch.float32, device=DEVICE)
    target_logits[2, 5] = 100.0
    # req2: random bonus with one supported token 7.
    target_logits[3].fill_(float("-inf"))
    target_logits[3, 7] = 0.0
    # req3: random full-draft residual with one positive residual token 2.
    p = torch.tensor([0.1, 0.1, 0.4, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05], dtype=torch.float32)
    q = torch.tensor([0.1, 0.1, 0.3, 0.1, 0.1, 0.10, 0.05, 0.05, 0.10], dtype=torch.float32)
    target_logits[4] = p.log().to(DEVICE)

    draft_logits = torch.zeros(4, 1, vocab_size, dtype=torch.float32, device=DEVICE)
    draft_logits[3, 0] = q.log().to(DEVICE)
    sampled = torch.full((4, 2), -777, dtype=torch.int64, device=DEVICE)
    sampled[0, 0] = 4
    num_sampled = torch.zeros(4, dtype=torch.int32, device=DEVICE)
    zeros = torch.zeros(4, dtype=torch.float32, device=DEVICE)
    temperature = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32, device=DEVICE)
    seed = torch.tensor([31, 37, 41, 43], dtype=torch.int64, device=DEVICE)

    resample(
        sampled,
        num_sampled,
        target_logits,
        zeros,
        draft_logits,
        zeros,
        _cu_num_logits(lengths),
        _expanded_mapping(lengths),
        torch.zeros(sum(lengths), dtype=torch.int32, device=DEVICE),
        temperature,
        seed,
        _position_tensor(sum(lengths)),
        has_draft_logits=True,
    )
    torch.npu.synchronize()

    assert torch.equal(sampled[:, 0], torch.tensor([4, 5, 7, 2], dtype=torch.int64, device=DEVICE))
    assert torch.equal(num_sampled, torch.ones_like(num_sampled))


@torch.inference_mode()
def test_full_draft_moderate_near_cancellation_keeps_positive_residual():
    """A small but fp32-resolvable positive residual must not collapse to zero mass."""
    p = torch.tensor([0.5001, 0.4999], dtype=torch.float32)
    q = torch.tensor([0.5000, 0.5000], dtype=torch.float32)
    target_logits = torch.stack([p.log(), p.log()]).to(DEVICE)
    draft_logits = q.log().reshape(1, 1, -1).to(DEVICE)
    sampled = torch.full((1, 2), -1, dtype=torch.int64, device=DEVICE)
    num_sampled = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    zeros = torch.zeros(1, dtype=torch.float32, device=DEVICE)

    resample(
        sampled,
        num_sampled,
        target_logits,
        zeros,
        draft_logits,
        zeros,
        _cu_num_logits([2]),
        _expanded_mapping([2]),
        torch.zeros(2, dtype=torch.int32, device=DEVICE),
        torch.ones(1, dtype=torch.float32, device=DEVICE),
        torch.tensor([4099], dtype=torch.int64, device=DEVICE),
        _position_tensor(2),
        has_draft_logits=True,
    )
    torch.npu.synchronize()

    assert sampled[0, 0].item() == 0
    assert num_sampled[0].item() == 1


@torch.inference_mode()
def test_same_seed_and_position_are_deterministic():
    vocab_size = 17
    lengths = [1] * 32
    target_logits = torch.randn(32, vocab_size, dtype=torch.float32, device=DEVICE)
    zeros = torch.zeros(32, dtype=torch.float32, device=DEVICE)
    temperature = torch.ones(32, dtype=torch.float32, device=DEVICE)
    seed = torch.arange(32, dtype=torch.int64, device=DEVICE) + 901
    pos = _position_tensor(32)

    def run_once():
        sampled = torch.full((32, 1), -1, dtype=torch.int64, device=DEVICE)
        num_sampled = torch.zeros(32, dtype=torch.int32, device=DEVICE)
        resample(
            sampled,
            num_sampled,
            target_logits,
            zeros,
            None,
            zeros,
            _cu_num_logits(lengths),
            _expanded_mapping(lengths),
            torch.zeros(32, dtype=torch.int32, device=DEVICE),
            temperature,
            seed,
            pos,
        )
        return sampled.clone(), num_sampled.clone()

    sampled_a, count_a = run_once()
    sampled_b, count_b = run_once()
    torch.npu.synchronize()
    assert torch.equal(sampled_a, sampled_b)
    assert torch.equal(count_a, count_b)


# ---------------------------------------------------------------------------
# End-to-end through the patched rejection_sample entry point
# ---------------------------------------------------------------------------


@torch.inference_mode()
def test_rejection_sample_greedy_end_to_end():
    """The upper rejection path and categorical resample preserve greedy semantics."""
    num_reqs = 5
    num_spec_steps = 3
    num_logits_per_req = num_spec_steps + 1
    vocab_size = 2 * VOCAB_BLOCK_SIZE + 37
    max_num_reqs = 11
    torch.manual_seed(2026)

    cu = [i * num_logits_per_req for i in range(num_reqs + 1)]
    num_logits = cu[-1]
    cu_num_logits = torch.tensor(cu, dtype=torch.int32, device=DEVICE)
    rows = torch.randperm(max_num_reqs)[:num_reqs].to(torch.int32)
    idx_mapping = rows.to(DEVICE)
    expanded_idx_mapping = rows.repeat_interleave(num_logits_per_req).to(DEVICE)
    expanded_local_pos = torch.arange(num_logits_per_req, dtype=torch.int32).repeat(num_reqs).to(DEVICE)

    target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=DEVICE)
    draft_sampled = torch.randint(0, vocab_size, (num_logits,), dtype=torch.int32, device=DEVICE)
    target_argmax = target_logits.argmax(dim=1)

    # Request r accepts min(r, num_spec_steps) draft tokens.  The batch
    # therefore includes immediate rejection, later rejection, and bonus paths.
    for req_idx in range(num_reqs):
        accept = min(req_idx, num_spec_steps)
        for step in range(num_spec_steps):
            draft_slot = cu[req_idx] + step + 1
            target_token = int(target_argmax[cu[req_idx] + step])
            draft_sampled[draft_slot] = target_token if step < accept else (target_token + 1) % vocab_size

    temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=DEVICE)
    seed = torch.randint(1, 2**30, (max_num_reqs,), dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_logits, dtype=torch.int64, device=DEVICE) + 5

    sampled, num_sampled = rejection_sample(
        target_logits,
        None,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        num_spec_steps,
    )
    torch.npu.synchronize()

    target_argmax_cpu = target_argmax.cpu().tolist()
    draft_sampled_cpu = draft_sampled.cpu().tolist()

    for req_idx in range(num_reqs):
        expected = []
        accepted = 0
        for step in range(num_spec_steps):
            target_token = target_argmax_cpu[cu[req_idx] + step]
            expected.append(target_token)
            if target_token != draft_sampled_cpu[cu[req_idx] + step + 1]:
                break
            accepted += 1
        else:
            expected.append(target_argmax_cpu[cu[req_idx + 1] - 1])

        expected_count = accepted + 1
        assert int(num_sampled[req_idx]) == expected_count
        assert sampled[req_idx, :expected_count].cpu().tolist() == expected

    counts = num_sampled.cpu().tolist()
    assert min(counts) == 1
    assert max(counts) == num_logits_per_req


@torch.inference_mode()
def test_empty_batch_is_noop():
    target_logits = torch.empty((0, 8), dtype=torch.float32, device=DEVICE)
    sampled = torch.empty((0, 1), dtype=torch.int64, device=DEVICE)
    num_sampled = torch.empty((0,), dtype=torch.int32, device=DEVICE)
    empty_i32 = torch.empty((0,), dtype=torch.int32, device=DEVICE)
    empty_i64 = torch.empty((0,), dtype=torch.int64, device=DEVICE)
    cu_num_logits = torch.zeros(1, dtype=torch.int32, device=DEVICE)

    resample(
        sampled,
        num_sampled,
        target_logits,
        torch.empty((0,), dtype=torch.float32, device=DEVICE),
        None,
        torch.empty((0,), dtype=torch.float32, device=DEVICE),
        cu_num_logits,
        empty_i32,
        empty_i32,
        torch.empty((0,), dtype=torch.float32, device=DEVICE),
        empty_i64,
        empty_i64,
    )

    assert sampled.numel() == 0
    assert num_sampled.numel() == 0


@torch.inference_mode()
def test_zero_vocab_is_rejected():
    with pytest.raises(ValueError, match="vocab_size must be greater than 0"):
        resample(
            torch.empty((1, 1), dtype=torch.int64, device=DEVICE),
            torch.zeros(1, dtype=torch.int32, device=DEVICE),
            torch.empty((1, 0), dtype=torch.float32, device=DEVICE),
            torch.zeros(1, dtype=torch.float32, device=DEVICE),
            None,
            torch.zeros(1, dtype=torch.float32, device=DEVICE),
            _cu_num_logits([1]),
            _expanded_mapping([1]),
            torch.zeros(1, dtype=torch.int32, device=DEVICE),
            torch.ones(1, dtype=torch.float32, device=DEVICE),
            torch.tensor([1], dtype=torch.int64, device=DEVICE),
            _position_tensor(1),
        )


@torch.inference_mode()
def test_explicit_full_draft_requires_draft_logits():
    with pytest.raises(ValueError, match="draft_logits cannot be None"):
        resample(
            torch.full((1, 2), -1, dtype=torch.int64, device=DEVICE),
            torch.zeros(1, dtype=torch.int32, device=DEVICE),
            torch.zeros((2, 8), dtype=torch.float32, device=DEVICE),
            torch.zeros(1, dtype=torch.float32, device=DEVICE),
            None,
            torch.zeros(1, dtype=torch.float32, device=DEVICE),
            _cu_num_logits([2]),
            _expanded_mapping([2]),
            torch.zeros(2, dtype=torch.int32, device=DEVICE),
            torch.ones(1, dtype=torch.float32, device=DEVICE),
            torch.tensor([1], dtype=torch.int64, device=DEVICE),
            _position_tensor(2),
            has_draft_logits=True,
        )


@torch.inference_mode()
def test_target_vocab_dimension_must_be_contiguous():
    vocab_size = 8
    base = torch.zeros((1, vocab_size * 2), dtype=torch.float32, device=DEVICE)
    target_logits = base[:, ::2]
    assert target_logits.stride(-1) == 2

    with pytest.raises(ValueError, match="vocabulary dimension must be contiguous"):
        resample(
            torch.full((1, 1), -1, dtype=torch.int64, device=DEVICE),
            torch.zeros(1, dtype=torch.int32, device=DEVICE),
            target_logits,
            torch.zeros(1, dtype=torch.float32, device=DEVICE),
            None,
            torch.zeros(1, dtype=torch.float32, device=DEVICE),
            _cu_num_logits([1]),
            _expanded_mapping([1]),
            torch.zeros(1, dtype=torch.int32, device=DEVICE),
            torch.ones(1, dtype=torch.float32, device=DEVICE),
            torch.tensor([1], dtype=torch.int64, device=DEVICE),
            _position_tensor(1),
        )


@torch.inference_mode()
def test_draft_vocab_dimension_must_be_contiguous():
    vocab_size = 8
    target_logits = torch.zeros((2, vocab_size), dtype=torch.float32, device=DEVICE)
    draft_base = torch.zeros((1, 1, vocab_size * 2), dtype=torch.float32, device=DEVICE)
    draft_logits = draft_base[..., ::2]
    assert draft_logits.stride(-1) == 2

    with pytest.raises(ValueError, match="vocabulary dimension must be contiguous"):
        resample(
            torch.full((1, 2), -1, dtype=torch.int64, device=DEVICE),
            torch.zeros(1, dtype=torch.int32, device=DEVICE),
            target_logits,
            torch.zeros(1, dtype=torch.float32, device=DEVICE),
            draft_logits,
            torch.zeros(1, dtype=torch.float32, device=DEVICE),
            _cu_num_logits([2]),
            _expanded_mapping([2]),
            torch.zeros(2, dtype=torch.int32, device=DEVICE),
            torch.ones(1, dtype=torch.float32, device=DEVICE),
            torch.tensor([1], dtype=torch.int64, device=DEVICE),
            _position_tensor(2),
            has_draft_logits=True,
        )
