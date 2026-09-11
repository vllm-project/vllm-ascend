# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch

from vllm_ascend.ops.triton.v2.sample.categorical_sample import categorical_sample

DEVICE = "npu"
VOCAB_SIZE = 151936


@pytest.fixture(autouse=True)
def cleanup_npu():
    yield
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def _seed_and_pos(num_tokens: int, num_reqs: int) -> tuple[torch.Tensor, torch.Tensor]:
    seed = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE) * 104729 + 17
    pos = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE) + 23
    return seed, pos


@pytest.mark.parametrize("num_tokens", [1, 16, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_categorical_sample_greedy(num_tokens, dtype):
    """temperature=0 must match the exact argmax on the long-vocab inference shape."""
    torch.manual_seed(0)
    logits = torch.randn(num_tokens, VOCAB_SIZE, dtype=dtype, device=DEVICE)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    temperature = torch.zeros(num_tokens, dtype=torch.float32, device=DEVICE)
    seed, pos = _seed_and_pos(num_tokens, num_tokens)

    sampled = categorical_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False)
    torch.npu.synchronize()

    assert sampled.dtype == torch.int64
    assert sampled.shape == (num_tokens,)
    torch.testing.assert_close(sampled, logits.argmax(dim=-1), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_categorical_sample_mixed_temperature_and_hierarchy_boundaries(dtype):
    """Mixed greedy/random rows must work across coarse/fine/tail boundaries."""
    torch.manual_seed(1)
    num_tokens = 16
    logits = torch.randn(num_tokens, VOCAB_SIZE, dtype=dtype, device=DEVICE)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    temperature = torch.ones(num_tokens, dtype=torch.float32, device=DEVICE)
    temperature[:8] = 0.0
    seed, pos = _seed_and_pos(num_tokens, num_tokens)

    expected = torch.empty(num_tokens, dtype=torch.int64, device=DEVICE)
    expected[:8] = logits[:8].argmax(dim=-1)

    support = torch.tensor([0, 1023, 1024, 8191, 8192, 65535, 131071, VOCAB_SIZE - 1], dtype=torch.int64, device=DEVICE)
    logits[8:] = float("-inf")
    logits[torch.arange(8, 16, device=DEVICE), support] = 3.0
    expected[8:] = support

    sampled = categorical_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=True)
    torch.npu.synchronize()

    torch.testing.assert_close(sampled, expected, rtol=0, atol=0)


@pytest.mark.parametrize("is_drafting", [False, True])
def test_categorical_sample_deterministic_same_seed_and_pos(is_drafting):
    """The categorical implementation must be deterministic for identical inputs."""
    torch.manual_seed(2)
    num_tokens = 16
    logits = torch.randn(num_tokens, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    temperature = torch.tensor([0.5, 1.0] * (num_tokens // 2), dtype=torch.float32, device=DEVICE)
    seed, pos = _seed_and_pos(num_tokens, num_tokens)

    sampled_1 = categorical_sample(
        logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=True, is_drafting=is_drafting
    )
    sampled_2 = categorical_sample(
        logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=True, is_drafting=is_drafting
    )
    torch.npu.synchronize()

    torch.testing.assert_close(sampled_1, sampled_2, rtol=0, atol=0)


def test_categorical_sample_apply_temperature_matches_prescaled_logits():
    """In-kernel temperature scaling must match sampling from pre-scaled FP32 logits."""
    torch.manual_seed(3)
    num_tokens = 16
    logits = torch.randint(-32, 33, (num_tokens, VOCAB_SIZE), dtype=torch.int32, device=DEVICE).to(torch.float32) / 8
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    temperature = torch.tensor([0.5, 1.0, 2.0, 1.0] * (num_tokens // 4), dtype=torch.float32, device=DEVICE)
    seed, pos = _seed_and_pos(num_tokens, num_tokens)

    sampled_scaled_in_kernel = categorical_sample(
        logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=True
    )
    scaled_logits = logits / temperature[:, None]
    sampled_prescaled = categorical_sample(
        scaled_logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False
    )
    torch.npu.synchronize()

    torch.testing.assert_close(sampled_scaled_in_kernel, sampled_prescaled, rtol=0, atol=0)


@pytest.mark.parametrize("per_token_col", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_categorical_sample_logits_cache(per_token_col, dtype):
    """logits_cache must store raw pre-temperature logits at request/column indices."""
    torch.manual_seed(4)
    num_tokens = 4
    max_num_reqs = 8
    num_cols = 3
    logits = torch.randn(num_tokens, VOCAB_SIZE, dtype=dtype, device=DEVICE)
    expanded_idx_mapping = torch.tensor([2, 5, 7, 0], dtype=torch.int32, device=DEVICE)
    temperature = torch.tensor([0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 0.5, 1.0], dtype=torch.float32, device=DEVICE)
    seed, pos = _seed_and_pos(num_tokens, max_num_reqs)
    logits_cache = torch.zeros(max_num_reqs, num_cols, VOCAB_SIZE, dtype=dtype, device=DEVICE)

    if per_token_col:
        logits_cache_col = torch.tensor([0, 1, 2, 1], dtype=torch.int32, device=DEVICE)
        expected_cols = [0, 1, 2, 1]
    else:
        logits_cache_col = torch.tensor(1, dtype=torch.int32, device=DEVICE)
        expected_cols = [1] * num_tokens

    categorical_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=True,
        logits_cache=logits_cache,
        logits_cache_col=logits_cache_col,
    )
    torch.npu.synchronize()

    used = set()
    for token_idx, col in enumerate(expected_cols):
        req = expanded_idx_mapping[token_idx].item()
        used.add((req, col))
        torch.testing.assert_close(logits_cache[req, col], logits[token_idx], rtol=0, atol=0)

    for req in range(max_num_reqs):
        for col in range(num_cols):
            if (req, col) not in used:
                assert torch.count_nonzero(logits_cache[req, col]).item() == 0


def test_categorical_sample_random_distribution_sanity():
    """Equal finite logits should produce samples across all supported tokens."""
    num_tokens = 128
    support = torch.tensor([7, 1027, 8199, VOCAB_SIZE - 1], dtype=torch.int64, device=DEVICE)
    logits = torch.full((num_tokens, VOCAB_SIZE), float("-inf"), dtype=torch.float32, device=DEVICE)
    logits[:, support] = 0.0
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    temperature = torch.ones(num_tokens, dtype=torch.float32, device=DEVICE)
    seed, pos = _seed_and_pos(num_tokens, num_tokens)

    sampled = categorical_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=True)
    torch.npu.synchronize()

    sampled_cpu = sampled.cpu()
    support_cpu = support.cpu()
    assert set(sampled_cpu.tolist()).issubset(set(support_cpu.tolist()))
    counts = torch.tensor([(sampled_cpu == token).sum().item() for token in support_cpu])
    assert (counts >= 12).all() and (counts <= 52).all(), f"unexpected counts for equal-mass support: {counts.tolist()}"


def test_categorical_sample_use_fp64_is_not_supported():
    logits = torch.zeros(1, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
    expanded_idx_mapping = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    temperature = torch.ones(1, dtype=torch.float32, device=DEVICE)
    seed, pos = _seed_and_pos(1, 1)

    with pytest.raises(NotImplementedError, match="FP64 categorical sampling is not supported on NPU"):
        categorical_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=True, use_fp64=True)
