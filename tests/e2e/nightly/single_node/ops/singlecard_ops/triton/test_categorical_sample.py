# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.v2.sample.categorical_sample import categorical_sample

DEVICE = "npu"
VOCAB_SIZE = 151936
SUPPORTED_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.fixture(scope="module", autouse=True)
def _npu_env():
    init_device_properties_triton()
    yield
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def _seed_and_pos(
    num_tokens: int,
    num_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seed = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE) * 104729 + 17
    pos = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE) + 23
    return seed, pos


def _sample(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    *,
    apply_temperature: bool,
    is_drafting: bool = False,
    logits_cache: torch.Tensor | None = None,
    logits_cache_col: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor:
    """Call the version-compatible public categorical entry explicitly."""
    return categorical_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=apply_temperature,
        is_drafting=is_drafting,
        logits_cache=logits_cache,
        logits_cache_col=logits_cache_col,
        use_fp64=use_fp64,
    )


@pytest.mark.parametrize("num_tokens", [1, 16, 64])
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_categorical_sample_greedy(num_tokens, dtype):
    """temperature=0 must match exact argmax on the long-vocab shape."""
    torch.manual_seed(0)
    logits = torch.randn(
        num_tokens,
        VOCAB_SIZE,
        dtype=dtype,
        device=DEVICE,
    )
    expanded_idx_mapping = torch.arange(
        num_tokens,
        dtype=torch.int32,
        device=DEVICE,
    )
    temperature = torch.zeros(
        num_tokens,
        dtype=torch.float32,
        device=DEVICE,
    )
    seed, pos = _seed_and_pos(num_tokens, num_tokens)

    sampled = _sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=False,
    )
    torch.npu.synchronize()

    assert sampled.dtype == torch.int64
    assert sampled.shape == (num_tokens,)
    torch.testing.assert_close(
        sampled,
        logits.argmax(dim=-1),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_categorical_sample_mixed_temperature_and_hierarchy_boundaries(dtype):
    """Mixed greedy/random rows must cross fine/coarse/tail boundaries."""
    torch.manual_seed(1)
    num_tokens = 16
    logits = torch.randn(
        num_tokens,
        VOCAB_SIZE,
        dtype=dtype,
        device=DEVICE,
    )
    expanded_idx_mapping = torch.arange(
        num_tokens,
        dtype=torch.int32,
        device=DEVICE,
    )
    temperature = torch.ones(
        num_tokens,
        dtype=torch.float32,
        device=DEVICE,
    )
    temperature[:8] = 0.0
    seed, pos = _seed_and_pos(num_tokens, num_tokens)

    expected = torch.empty(
        num_tokens,
        dtype=torch.int64,
        device=DEVICE,
    )
    expected[:8] = logits[:8].argmax(dim=-1)

    support = torch.tensor(
        [
            0,
            1023,
            1024,
            8191,
            8192,
            65535,
            131071,
            VOCAB_SIZE - 1,
        ],
        dtype=torch.int64,
        device=DEVICE,
    )
    logits[8:] = float("-inf")
    logits[torch.arange(8, 16, device=DEVICE), support] = 3.0
    expected[8:] = support

    sampled = _sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=True,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(sampled, expected, rtol=0, atol=0)


@pytest.mark.parametrize("is_drafting", [False, True])
def test_categorical_sample_deterministic_same_seed_and_pos(is_drafting):
    """Identical input, seed and position must replay deterministically."""
    torch.manual_seed(2)
    num_tokens = 16
    logits = torch.randn(
        num_tokens,
        VOCAB_SIZE,
        dtype=torch.float32,
        device=DEVICE,
    )
    expanded_idx_mapping = torch.arange(
        num_tokens,
        dtype=torch.int32,
        device=DEVICE,
    )
    temperature = torch.tensor(
        [0.5, 1.0] * (num_tokens // 2),
        dtype=torch.float32,
        device=DEVICE,
    )
    seed, pos = _seed_and_pos(num_tokens, num_tokens)

    sampled_1 = _sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=True,
        is_drafting=is_drafting,
    )
    sampled_2 = _sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=True,
        is_drafting=is_drafting,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(sampled_1, sampled_2, rtol=0, atol=0)


def test_categorical_sample_draft_rng_uses_position_salt():
    """Draft RNG must be equivalent to the target stream at pos + 2**30."""
    torch.manual_seed(20)
    num_tokens = 32
    vocab_size = 1031
    logits = torch.randn(
        num_tokens,
        vocab_size,
        dtype=torch.float32,
        device=DEVICE,
    )
    expanded_idx_mapping = torch.arange(
        num_tokens,
        dtype=torch.int32,
        device=DEVICE,
    )
    temperature = torch.ones(
        num_tokens,
        dtype=torch.float32,
        device=DEVICE,
    )
    seed, pos = _seed_and_pos(num_tokens, num_tokens)

    draft = _sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=True,
        is_drafting=True,
    )
    shifted_target = _sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos + (1 << 30),
        apply_temperature=True,
        is_drafting=False,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(draft, shifted_target, rtol=0, atol=0)


def test_categorical_sample_apply_temperature_matches_prescaled_logits():
    """In-kernel scaling must match sampling from pre-scaled FP32 logits."""
    torch.manual_seed(3)
    num_tokens = 16
    logits = (
        torch.randint(
            -32,
            33,
            (num_tokens, VOCAB_SIZE),
            dtype=torch.int32,
            device=DEVICE,
        ).to(torch.float32)
        / 8
    )
    expanded_idx_mapping = torch.arange(
        num_tokens,
        dtype=torch.int32,
        device=DEVICE,
    )
    temperature = torch.tensor(
        [0.5, 1.0, 2.0, 1.0] * (num_tokens // 4),
        dtype=torch.float32,
        device=DEVICE,
    )
    seed, pos = _seed_and_pos(num_tokens, num_tokens)

    sampled_scaled_in_kernel = _sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=True,
    )
    scaled_logits = logits / temperature[:, None]
    sampled_prescaled = _sample(
        scaled_logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=False,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        sampled_scaled_in_kernel,
        sampled_prescaled,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("per_token_col", [False, True])
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_categorical_sample_logits_cache(per_token_col, dtype):
    """Cache must store raw pre-temperature logits at mapped slots."""
    torch.manual_seed(4)
    num_tokens = 4
    max_num_reqs = 8
    num_cols = 3

    logits = torch.randn(
        num_tokens,
        VOCAB_SIZE,
        dtype=dtype,
        device=DEVICE,
    )
    expanded_idx_mapping = torch.tensor(
        [2, 5, 7, 0],
        dtype=torch.int32,
        device=DEVICE,
    )
    temperature = torch.tensor(
        [0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 0.5, 1.0],
        dtype=torch.float32,
        device=DEVICE,
    )
    seed, pos = _seed_and_pos(num_tokens, max_num_reqs)
    logits_cache = torch.zeros(
        max_num_reqs,
        num_cols,
        VOCAB_SIZE,
        dtype=dtype,
        device=DEVICE,
    )

    if per_token_col:
        logits_cache_col = torch.tensor(
            [0, 1, 2, 1],
            dtype=torch.int32,
            device=DEVICE,
        )
        expected_cols = [0, 1, 2, 1]
    else:
        logits_cache_col = torch.tensor(
            1,
            dtype=torch.int32,
            device=DEVICE,
        )
        expected_cols = [1] * num_tokens

    _sample(
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
        torch.testing.assert_close(
            logits_cache[req, col],
            logits[token_idx],
            rtol=0,
            atol=0,
        )

    for req in range(max_num_reqs):
        for col in range(num_cols):
            if (req, col) not in used:
                assert torch.count_nonzero(logits_cache[req, col]).item() == 0


def test_categorical_sample_padding_mapping_does_not_write_cache():
    """CUDAGraph padding request index -1 must not write logits_cache."""
    torch.manual_seed(5)
    num_tokens = 4
    max_num_reqs = 4

    logits = torch.randn(
        num_tokens,
        VOCAB_SIZE,
        dtype=torch.float32,
        device=DEVICE,
    )
    expanded_idx_mapping = torch.tensor(
        [0, 2, -1, -1],
        dtype=torch.int32,
        device=DEVICE,
    )
    temperature = torch.ones(
        max_num_reqs,
        dtype=torch.float32,
        device=DEVICE,
    )
    seed, pos = _seed_and_pos(num_tokens, max_num_reqs)
    logits_cache = torch.zeros(
        max_num_reqs,
        1,
        VOCAB_SIZE,
        dtype=torch.float32,
        device=DEVICE,
    )

    sampled = _sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=True,
        logits_cache=logits_cache,
    )
    torch.npu.synchronize()

    assert sampled.shape == (num_tokens,)
    torch.testing.assert_close(
        logits_cache[0, 0],
        logits[0],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        logits_cache[2, 0],
        logits[1],
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(logits_cache[1]).item() == 0
    assert torch.count_nonzero(logits_cache[3]).item() == 0


def test_categorical_sample_shared_request_mapping():
    """Rows sharing a request must share its seed and temperature stream."""
    torch.manual_seed(6)
    logits_row_0 = torch.randn(
        1,
        VOCAB_SIZE,
        dtype=torch.float32,
        device=DEVICE,
    )
    logits_row_1 = torch.randn(
        1,
        VOCAB_SIZE,
        dtype=torch.float32,
        device=DEVICE,
    )
    logits = torch.cat(
        [logits_row_0, logits_row_0, logits_row_1, logits_row_1],
        dim=0,
    )
    expanded_idx_mapping = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.int32,
        device=DEVICE,
    )
    temperature = torch.tensor(
        [0.7, 1.3],
        dtype=torch.float32,
        device=DEVICE,
    )
    seed = torch.tensor(
        [12345, 67890],
        dtype=torch.int64,
        device=DEVICE,
    )
    pos = torch.tensor(
        [11, 11, 19, 19],
        dtype=torch.int64,
        device=DEVICE,
    )

    sampled = _sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=True,
    )
    torch.npu.synchronize()

    assert sampled[0].item() == sampled[1].item()
    assert sampled[2].item() == sampled[3].item()


def test_categorical_sample_cache_does_not_change_sampled_tokens():
    """Raw-logit cache writes must not change sampled token IDs."""
    torch.manual_seed(7)
    num_tokens = 16
    logits = torch.randn(
        num_tokens,
        VOCAB_SIZE,
        dtype=torch.float32,
        device=DEVICE,
    )
    expanded_idx_mapping = torch.arange(
        num_tokens,
        dtype=torch.int32,
        device=DEVICE,
    )
    temperature = torch.ones(
        num_tokens,
        dtype=torch.float32,
        device=DEVICE,
    )
    seed, pos = _seed_and_pos(num_tokens, num_tokens)
    logits_cache = torch.zeros(
        num_tokens,
        1,
        VOCAB_SIZE,
        dtype=torch.float32,
        device=DEVICE,
    )

    sampled_without_cache = _sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=True,
    )
    sampled_with_cache = _sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=True,
        logits_cache=logits_cache,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        sampled_without_cache,
        sampled_with_cache,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        logits_cache[:, 0],
        logits,
        rtol=0,
        atol=0,
    )


def test_categorical_sample_random_distribution_sanity():
    """Equal finite logits should reach every supported token."""
    num_tokens = 128
    support = torch.tensor(
        [7, 1027, 8199, VOCAB_SIZE - 1],
        dtype=torch.int64,
        device=DEVICE,
    )
    logits = torch.full(
        (num_tokens, VOCAB_SIZE),
        float("-inf"),
        dtype=torch.float32,
        device=DEVICE,
    )
    logits[:, support] = 0.0
    expanded_idx_mapping = torch.arange(
        num_tokens,
        dtype=torch.int32,
        device=DEVICE,
    )
    temperature = torch.ones(
        num_tokens,
        dtype=torch.float32,
        device=DEVICE,
    )
    seed, pos = _seed_and_pos(num_tokens, num_tokens)

    sampled = _sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=True,
    )
    torch.npu.synchronize()

    sampled_cpu = sampled.cpu()
    support_cpu = support.cpu()
    assert set(sampled_cpu.tolist()).issubset(set(support_cpu.tolist()))
    counts = torch.tensor([(sampled_cpu == token).sum().item() for token in support_cpu])
    assert (counts >= 12).all() and (counts <= 52).all(), f"unexpected counts for equal-mass support: {counts.tolist()}"


def test_categorical_sample_business_shape_distribution_accuracy():
    """B64/V151936 samples must match the torch.softmax reference."""
    num_tokens = 64
    num_trials = 256
    support = torch.tensor(
        [
            7,
            1023,
            1024,
            8191,
            8192,
            65535,
            131071,
            VOCAB_SIZE - 1,
        ],
        dtype=torch.int64,
        device=DEVICE,
    )
    support_logits = torch.tensor(
        [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        dtype=torch.float32,
        device=DEVICE,
    )

    logits = torch.full(
        (num_tokens, VOCAB_SIZE),
        float("-inf"),
        dtype=torch.float32,
        device=DEVICE,
    )
    logits[:, support] = support_logits
    expanded_idx_mapping = torch.arange(
        num_tokens,
        dtype=torch.int32,
        device=DEVICE,
    )
    temperature = torch.ones(
        num_tokens,
        dtype=torch.float32,
        device=DEVICE,
    )
    base_seed, pos = _seed_and_pos(num_tokens, num_tokens)

    counts = torch.zeros(
        len(support),
        dtype=torch.int64,
        device=DEVICE,
    )
    for trial in range(num_trials):
        seed = base_seed + trial * 1000003
        sampled = _sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature=True,
        )
        for idx, token in enumerate(support):
            counts[idx] += (sampled == token).sum()

    torch.npu.synchronize()

    actual_probs = counts.to(torch.float32).cpu() / (num_tokens * num_trials)
    expected_probs = torch.softmax(support_logits, dim=0).cpu()
    abs_error = torch.abs(actual_probs - expected_probs)
    max_abs_error = torch.max(abs_error).item()
    tv_distance = 0.5 * torch.sum(abs_error).item()

    assert max_abs_error <= 0.02, (
        f"categorical probability max abs error "
        f"{max_abs_error:.6f} exceeds 0.02; "
        f"actual={actual_probs.tolist()}, "
        f"expected={expected_probs.tolist()}"
    )
    assert tv_distance <= 0.04, (
        f"categorical probability TV distance "
        f"{tv_distance:.6f} exceeds 0.04; "
        f"actual={actual_probs.tolist()}, "
        f"expected={expected_probs.tolist()}"
    )


def test_categorical_sample_empty_input():
    """Empty expanded batches must return an empty int64 tensor."""
    logits = torch.empty(
        0,
        32,
        dtype=torch.float32,
        device=DEVICE,
    )
    expanded_idx_mapping = torch.empty(
        0,
        dtype=torch.int32,
        device=DEVICE,
    )
    temperature = torch.ones(
        1,
        dtype=torch.float32,
        device=DEVICE,
    )
    seed = torch.zeros(
        1,
        dtype=torch.int64,
        device=DEVICE,
    )
    pos = torch.empty(
        0,
        dtype=torch.int64,
        device=DEVICE,
    )

    sampled = _sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=True,
    )

    assert sampled.dtype == torch.int64
    assert sampled.shape == (0,)


def test_categorical_sample_use_fp64_is_not_supported():
    logits = torch.zeros(
        1,
        VOCAB_SIZE,
        dtype=torch.float32,
        device=DEVICE,
    )
    expanded_idx_mapping = torch.zeros(
        1,
        dtype=torch.int32,
        device=DEVICE,
    )
    temperature = torch.ones(
        1,
        dtype=torch.float32,
        device=DEVICE,
    )
    seed, pos = _seed_and_pos(1, 1)

    with pytest.raises(
        NotImplementedError,
        match="FP64 categorical sampling is not supported on NPU",
    ):
        _sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature=True,
            use_fp64=True,
        )
