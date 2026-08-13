# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Production correctness tests for the Ascend categorical sampling operator.

The operator is intentionally tested on a real NPU.  The three groups below
cover its stateless RNG semantics, FP32 accumulation accuracy, and sampled
distribution under ACLGraph replay.
"""

from __future__ import annotations

import math
import struct
import subprocess
import sys
import textwrap
from dataclasses import dataclass

import pytest
import torch
import torch_npu  # noqa: F401  # Registers the NPU and ACLGraph APIs.

from tests.ut.sample.custom_op_utils import require_categorical_sampling_operator

DEVICE = torch.device("npu")
MAX_VOCAB_SIZE = 1_048_576
STATISTICAL_ALPHA = 0.001
STATISTICAL_BATCH_SIZE = 256
STATISTICAL_SAMPLE_COUNT = 131_072

UINT32_MASK = (1 << 32) - 1
PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85


def _run_categorical_sampling(
    logits: torch.Tensor,
    mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    return_lse: bool = False,
    apply_temperature: bool = False,
    output_processed_logits: torch.Tensor | None = None,
    output_processed_logits_col: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C_ascend.npu_categorical_sample(
        logits,
        mapping,
        temperature,
        seed,
        pos,
        return_lse,
        apply_temperature,
        output_processed_logits,
        output_processed_logits_col,
        use_fp64,
    )


@pytest.fixture(scope="module", autouse=True)
def _require_categorical_sampling_operator() -> None:
    require_categorical_sampling_operator()


def _to_npu(value: torch.Tensor) -> torch.Tensor:
    return value.to(DEVICE)


def _float32(value: float) -> float:
    """Round a Python float exactly as an IEEE FP32 scalar operation does."""
    return struct.unpack("=f", struct.pack("=f", value))[0]


def _mul_hi(lhs: int, rhs: int) -> int:
    return ((lhs * rhs) >> 32) & UINT32_MASK


def _mul_lo(lhs: int, rhs: int) -> int:
    return (lhs * rhs) & UINT32_MASK


def _philox_words(seed: int, position: int) -> tuple[int, int]:
    counter0 = position & UINT32_MASK
    counter1 = (position >> 32) & UINT32_MASK
    counter2 = 0
    counter3 = 0
    key0 = seed & UINT32_MASK
    key1 = (seed >> 32) & UINT32_MASK

    for _ in range(10):
        hi0 = _mul_hi(PHILOX_M0, counter0)
        lo0 = _mul_lo(PHILOX_M0, counter0)
        hi1 = _mul_hi(PHILOX_M1, counter2)
        lo1 = _mul_lo(PHILOX_M1, counter2)
        counter0 = (hi1 ^ counter1 ^ key0) & UINT32_MASK
        counter1 = lo1
        counter2 = (hi0 ^ counter3 ^ key1) & UINT32_MASK
        counter3 = lo0
        key0 = (key0 + PHILOX_W0) & UINT32_MASK
        key1 = (key1 + PHILOX_W1) & UINT32_MASK

    return counter0, counter1


def _philox_uniform(seed: int, position: int) -> float:
    """CPU definition of the operator's Philox4x32-10 FP32 draw."""
    counter0, _ = _philox_words(seed, position)

    random24 = counter0 >> 8
    shifted = _float32(_float32(float(random24)) + 0.5)
    return _float32(shifted * (1.0 / (1 << 24)))


def _philox_random64(seed: int, position: int) -> int:
    random0, random1 = _philox_words(seed, position)
    return (random1 << 32) | random0


def _uniform_support_choice(support: list[int], seed: int, position: int) -> int:
    """Select from equal weights using the kernel's inclusive CDF semantics."""
    target = _float32(_philox_uniform(seed, position) * len(support))
    cumulative = 0.0
    for token_id in support:
        cumulative = _float32(cumulative + 1.0)
        if cumulative >= target:
            return token_id
    return support[-1]


def _assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_categorical_sampling_stateless_philox_matches_cpu_reference(dtype: torch.dtype) -> None:
    """Seed and logical position fully define sampling, including padding."""
    supports = [
        [0, 3, 7, 19, 32],
        [1, 2, 5, 8, 13, 21, 31],
        [4, 9, 16],
        [0, 3, 7, 19, 32],
        [1, 2, 5, 8, 13, 21, 31],
        [4, 9, 16],
        [],
        [0, 3, 7, 19, 32],
    ]
    mapping_cpu = torch.tensor([0, 1, 2, 0, 1, 2, -1, 0], dtype=torch.int32)
    seed_values = [0x0123456789ABCDEF, -37, (1 << 40) + 17]
    positions = [0, 1, 2, (1 << 32) + 3, (1 << 32) + 9, 99, 123, (1 << 40) + 5]

    logits_cpu = torch.full((len(supports), 33), -float("inf"), dtype=dtype)
    for row, support in enumerate(supports):
        if support:
            logits_cpu[row, support] = 0

    outputs = _run_categorical_sampling(
        _to_npu(logits_cpu),
        _to_npu(mapping_cpu),
        _to_npu(torch.ones(3, dtype=torch.float32)),
        _to_npu(torch.tensor(seed_values, dtype=torch.int64)),
        _to_npu(torch.tensor(positions, dtype=torch.int64)),
    )
    expected = []
    for row, request_index in enumerate(mapping_cpu.tolist()):
        if request_index == -1:
            expected.append(0)
        else:
            expected.append(_uniform_support_choice(supports[row], seed_values[request_index], positions[row]))

    _assert_tensor_equal(outputs[0], torch.tensor(expected, dtype=torch.int64))
    assert outputs[1].numel() == 0


def test_categorical_sampling_fp64_resolves_sub_fp32_probability_interval() -> None:
    """FP64 mode must resolve a winner inside an interval narrower than 2^-24."""
    seed_value = 0x0123456789ABCDEF
    position = 482_600
    num_rare_tokens = 32_768
    vocab_size = num_rare_tokens + 1
    rare_logit = -26.0 * math.log(2.0)
    logits = torch.full((1, vocab_size), rare_logit, dtype=torch.float32, device=DEVICE)
    logits[0, 0] = 0.0

    sampled = _run_categorical_sampling(
        logits,
        torch.tensor([0], dtype=torch.int32, device=DEVICE),
        torch.ones(1, dtype=torch.float32, device=DEVICE),
        torch.tensor([seed_value], dtype=torch.int64, device=DEVICE),
        torch.tensor([position], dtype=torch.int64, device=DEVICE),
        use_fp64=True,
    )[0]

    # The kernel represents exp(logit - max) with 42 fractional bits.  Each
    # rare token therefore has mass 2^16 while the dominant token has 2^42.
    dominant_mass = 1 << 42
    rare_mass = 1 << 16
    total_mass = dominant_mass + num_rare_tokens * rare_mass
    random64 = _philox_random64(seed_value, position)
    target = (random64 * total_mass) >> 64
    expected = 1 + (target - dominant_mass) // rare_mass

    # Quantizing the same draw to a 24-bit midpoint lands two probability
    # intervals away. This makes the assertion sensitive to actual precision,
    # instead of merely checking that the use_fp64 branch runs.
    random24_midpoint = ((random64 >> 40) << 40) + (1 << 39)
    fp32_grid_target = (random24_midpoint * total_mass) >> 64
    fp32_grid_choice = 1 + (fp32_grid_target - dominant_mass) // rare_mass
    assert expected == 2310
    assert fp32_grid_choice == 2312
    _assert_tensor_equal(sampled, torch.tensor([expected], dtype=torch.int64))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_categorical_sampling_greedy_accepted_special_values_are_exact(dtype: torch.dtype) -> None:
    logits_cpu = torch.full((3, 33), -4.0, dtype=dtype)
    logits_cpu[0, [5, 17]] = 3
    logits_cpu[1, [7, 21]] = float("inf")
    logits_cpu[2] = float("nan")  # Padding rows do not inspect logits.
    mapping = torch.tensor([0, 1, -1], dtype=torch.int32)
    temperature = torch.zeros(2, dtype=torch.float32)
    pos = torch.arange(3, dtype=torch.int64)

    first = _run_categorical_sampling(
        _to_npu(logits_cpu),
        _to_npu(mapping),
        _to_npu(temperature),
        _to_npu(torch.tensor([11, 22], dtype=torch.int64)),
        _to_npu(pos),
    )
    second = _run_categorical_sampling(
        _to_npu(logits_cpu),
        _to_npu(mapping),
        _to_npu(temperature),
        _to_npu(torch.tensor([101, 202], dtype=torch.int64)),
        _to_npu(pos + 1000),
    )
    expected_tokens = torch.tensor([5, 7, 0], dtype=torch.int64)

    for outputs in (first, second):
        _assert_tensor_equal(outputs[0], expected_tokens)
        assert outputs[1].numel() == 0


def test_categorical_sampling_is_invariant_to_batch_layout() -> None:
    vocab_size = 33
    logits = torch.zeros((4, vocab_size), dtype=torch.float32, device=DEVICE)
    mapping = torch.tensor([0, 1, 2, 0], dtype=torch.int32, device=DEVICE)
    temperature = torch.ones(4, dtype=torch.float32, device=DEVICE)
    seed = torch.tensor([101, 202, 303, 404], dtype=torch.int64, device=DEVICE)
    pos = torch.tensor([5, 7, 11, 13], dtype=torch.int64, device=DEVICE)

    baseline = _run_categorical_sampling(logits, mapping, temperature, seed, pos)[0]
    repeated = _run_categorical_sampling(logits, mapping, temperature, seed, pos)[0]

    order = torch.tensor([2, 0, 3, 1], dtype=torch.int64, device=DEVICE)
    inverse = torch.argsort(order)
    permuted = _run_categorical_sampling(
        logits.index_select(0, order),
        mapping.index_select(0, order),
        temperature,
        seed,
        pos.index_select(0, order),
    )[0].index_select(0, inverse)

    unrelated_logits = torch.cat((torch.full((1, vocab_size), 1.0, device=DEVICE), logits), dim=0)
    unrelated_mapping = torch.cat((torch.tensor([3], dtype=torch.int32, device=DEVICE), mapping), dim=0)
    unrelated_pos = torch.cat((torch.tensor([17], dtype=torch.int64, device=DEVICE), pos), dim=0)
    with_unrelated_request = _run_categorical_sampling(
        unrelated_logits,
        unrelated_mapping,
        temperature,
        seed,
        unrelated_pos,
    )[0][1:]

    padded_logits = torch.cat((logits, torch.zeros((3, vocab_size), device=DEVICE)), dim=0)
    padded_mapping = torch.cat((mapping, torch.full((3,), -1, dtype=torch.int32, device=DEVICE)), dim=0)
    padded_pos = torch.cat((pos, torch.arange(3, dtype=torch.int64, device=DEVICE)), dim=0)
    with_padding = _run_categorical_sampling(padded_logits, padded_mapping, temperature, seed, padded_pos)[0][:4]

    torch.npu.synchronize()
    for actual in (repeated, permuted, with_unrelated_request, with_padding):
        _assert_tensor_equal(actual, baseline.cpu())


@pytest.mark.parametrize("use_fp64", [False, True])
def test_categorical_sampling_supports_zero_stride_logits(use_fp64: bool) -> None:
    """A broadcast row remains a view and is sampled as independent logical rows."""
    vocab_size = 257
    num_rows = 8
    base_logits = torch.arange(vocab_size, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    logits = base_logits.expand(num_rows, vocab_size)
    assert logits.stride() == (0, 1)
    assert logits.untyped_storage().nbytes() == base_logits.untyped_storage().nbytes()

    sampled = _run_categorical_sampling(
        logits,
        torch.arange(num_rows, dtype=torch.int32, device=DEVICE),
        torch.zeros(num_rows, dtype=torch.float32, device=DEVICE),
        torch.arange(num_rows, dtype=torch.int64, device=DEVICE),
        torch.arange(num_rows, dtype=torch.int64, device=DEVICE),
        use_fp64=use_fp64,
    )[0]

    _assert_tensor_equal(sampled, torch.full((num_rows,), vocab_size - 1, dtype=torch.int64))


def test_categorical_sampling_rejects_nonzero_overlapping_row_stride() -> None:
    """Only a zero-stride broadcast is allowed; arbitrary overlapping rows are not."""
    num_rows = 2
    vocab_size = 17
    backing = torch.arange(vocab_size + num_rows - 1, dtype=torch.float32, device=DEVICE)
    logits = torch.as_strided(backing, (num_rows, vocab_size), (1, 1))
    assert 0 < logits.stride(0) < vocab_size

    with pytest.raises(RuntimeError, match="processed_logits row stride is invalid"):
        _run_categorical_sampling(
            logits,
            torch.arange(num_rows, dtype=torch.int32, device=DEVICE),
            torch.zeros(num_rows, dtype=torch.float32, device=DEVICE),
            torch.arange(num_rows, dtype=torch.int64, device=DEVICE),
            torch.arange(num_rows, dtype=torch.int64, device=DEVICE),
        )


def test_categorical_sampling_aclgraph_replay_matches_eager_position_sequence() -> None:
    batch_size = 8
    logits = torch.zeros((batch_size, 20), dtype=torch.float32, device=DEVICE)
    mapping = torch.tensor([0, 1, 0, 2, 1, 2, 0, 1], dtype=torch.int32, device=DEVICE)
    temperature = torch.ones(3, dtype=torch.float32, device=DEVICE)
    seed = torch.tensor([101, 202, 303], dtype=torch.int64, device=DEVICE)
    pos = torch.arange(batch_size, dtype=torch.int64, device=DEVICE)

    for _ in range(2):
        _run_categorical_sampling(logits, mapping, temperature, seed, pos)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_outputs = _run_categorical_sampling(logits, mapping, temperature, seed, pos)
    torch.npu.synchronize()

    position_sequence = [torch.arange(batch_size, dtype=torch.int64) + offset for offset in (0, 1009, (1 << 32) + 27)]
    eager_results = []
    for positions in position_sequence:
        pos.copy_(_to_npu(positions))
        eager_results.append(_run_categorical_sampling(logits, mapping, temperature, seed, pos)[0].cpu())

    for _ in range(2):
        replay_results = []
        for positions in position_sequence:
            pos.copy_(_to_npu(positions))
            graph.replay()
            replay_results.append(graph_outputs[0].cpu())
        for actual, expected in zip(replay_results, eager_results, strict=True):
            _assert_tensor_equal(actual, expected)


def test_categorical_sampling_aclgraph_padding_count_does_not_change_real_samples() -> None:
    real_rows = 4
    vocab_size = 20
    real_logits = torch.zeros((real_rows, vocab_size), dtype=torch.float32, device=DEVICE)
    real_mapping = torch.tensor([0, 1, 0, 2], dtype=torch.int32, device=DEVICE)
    real_pos = torch.tensor([5, 7, 11, 13], dtype=torch.int64, device=DEVICE)
    temperature = torch.ones(3, dtype=torch.float32, device=DEVICE)
    seed = torch.tensor([101, 202, 303], dtype=torch.int64, device=DEVICE)
    retained_graph_state = []
    real_outputs = []

    for padding_rows in (0, 7):
        logits = torch.cat(
            (real_logits, torch.zeros((padding_rows, vocab_size), dtype=torch.float32, device=DEVICE)),
            dim=0,
        )
        mapping = torch.cat(
            (real_mapping, torch.full((padding_rows,), -1, dtype=torch.int32, device=DEVICE)),
            dim=0,
        )
        pos = torch.cat(
            (real_pos, torch.arange(padding_rows, dtype=torch.int64, device=DEVICE)),
            dim=0,
        )
        for _ in range(2):
            _run_categorical_sampling(logits, mapping, temperature, seed, pos)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs = _run_categorical_sampling(logits, mapping, temperature, seed, pos)
        graph.replay()
        real_outputs.append(outputs[0][:real_rows].cpu())
        retained_graph_state.append((graph, outputs, logits, mapping, pos))

    _assert_tensor_equal(real_outputs[1], real_outputs[0])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_categorical_sampling_lse_matches_fp64_reference(dtype: torch.dtype) -> None:
    vocab_size = 4097
    logits_fp32 = torch.empty((4, vocab_size), dtype=torch.float32)
    logits_fp32[0] = torch.linspace(-12.0, 4.0, vocab_size)
    logits_fp32[1] = -float("inf")
    logits_fp32[1, [0, 20, 2048, 4096]] = torch.tensor([2.0, -1.0, 0.5, 2.0])
    logits_fp32[2] = torch.linspace(-80.0, 0.0, vocab_size)
    logits_fp32[2, 3333] = 18.0
    logits_fp32[3] = torch.sin(torch.arange(vocab_size, dtype=torch.float32) * 0.013) * 7.0
    logits_cpu = logits_fp32.to(dtype)

    outputs = _run_categorical_sampling(
        _to_npu(logits_cpu),
        torch.arange(4, dtype=torch.int32, device=DEVICE),
        torch.ones(4, dtype=torch.float32, device=DEVICE),
        torch.tensor([11, 22, 33, 44], dtype=torch.int64, device=DEVICE),
        torch.arange(4, dtype=torch.int64, device=DEVICE),
        True,
    )
    expected_lse = torch.logsumexp(logits_cpu.to(torch.float64), dim=-1)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-3

    torch.testing.assert_close(
        outputs[1].cpu().to(torch.float64),
        expected_lse,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("vocab_size", [1, 20, 33, 4097, MAX_VOCAB_SIZE])
def test_categorical_sampling_lse_vocabulary_boundaries(vocab_size: int) -> None:
    logits = torch.zeros((1, vocab_size), dtype=torch.float32, device=DEVICE)
    outputs = _run_categorical_sampling(
        logits,
        torch.zeros(1, dtype=torch.int32, device=DEVICE),
        torch.ones(1, dtype=torch.float32, device=DEVICE),
        torch.tensor([12345], dtype=torch.int64, device=DEVICE),
        torch.tensor([67890], dtype=torch.int64, device=DEVICE),
        True,
    )

    assert 0 <= int(outputs[0].cpu()) < vocab_size
    torch.testing.assert_close(
        outputs[1].cpu().to(torch.float64),
        torch.tensor([math.log(vocab_size)], dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_categorical_sampling_positive_infinity_lse_and_padding(dtype: torch.dtype) -> None:
    logits = torch.tensor(
        [
            [0.0, float("inf"), -1.0, float("inf")],
            [float("nan"), float("nan"), float("nan"), float("nan")],
        ],
        dtype=dtype,
        device=DEVICE,
    )
    outputs = _run_categorical_sampling(
        logits,
        torch.tensor([0, -1], dtype=torch.int32, device=DEVICE),
        torch.ones(1, dtype=torch.float32, device=DEVICE),
        torch.tensor([3], dtype=torch.int64, device=DEVICE),
        torch.tensor([11, 13], dtype=torch.int64, device=DEVICE),
        True,
    )
    tokens, lse = (output.cpu() for output in outputs)

    assert int(tokens[0]) in (1, 3)
    assert int(tokens[1]) == 0
    assert math.isinf(float(lse[0])) and float(lse[0]) > 0
    assert float(lse[1]) == 0.0


_ASSERTION_SUBPROCESS = textwrap.dedent(
    """
    import sys

    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.utils import enable_categorical_sample_op

    if not enable_categorical_sample_op():
        raise RuntimeError("the categorical custom operator is unavailable")

    case, execution = sys.argv[1:3]
    device = torch.device("npu")
    logits = torch.tensor([[0.0, 1.0, -1.0, 2.0]], dtype=torch.float32, device=device)
    mapping = torch.zeros(1, dtype=torch.int32, device=device)
    temperature = torch.ones(1, dtype=torch.float32, device=device)
    seed = torch.ones(1, dtype=torch.int64, device=device)
    pos = torch.zeros(1, dtype=torch.int64, device=device)
    cache = None
    cache_col = None
    if case == "cache_column":
        cache = torch.full((1, 2, 4), 17.0, dtype=torch.float32, device=device)
        cache_col = torch.zeros((), dtype=torch.int32, device=device)

    def sample():
        return torch.ops._C_ascend.npu_categorical_sample(
            logits,
            mapping,
            temperature,
            seed,
            pos,
            False,
            False,
            cache,
            cache_col,
            False,
        )

    if execution == "aclgraph":
        sample()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            graph_outputs = sample()
        torch.npu.synchronize()

    if case == "nan":
        logits[0, 0] = float("nan")
    elif case == "all_negative_infinity":
        logits.fill_(-float("inf"))
    elif case == "mapping":
        mapping.fill_(-2)
    elif case == "cache_column":
        cache_col.fill_(2)
    else:
        raise AssertionError(f"unknown case: {case}")

    if execution == "eager":
        sample()
    else:
        graph.replay()
    torch.npu.synchronize()
    """
)


@pytest.mark.parametrize("execution", ["eager", "aclgraph"])
@pytest.mark.parametrize(
    "case,expected_message",
    [
        ("nan", "CategoricalSample processed logits must not contain NaN"),
        ("all_negative_infinity", "CategoricalSample processed logits row must not be all -inf"),
        ("mapping", "CategoricalSample expanded index mapping is outside request state"),
        ("cache_column", "CategoricalSample output processed logits column is outside cache bounds"),
    ],
)
def test_categorical_sampling_asserts_invalid_device_values(
    execution: str,
    case: str,
    expected_message: str,
) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _ASSERTION_SUBPROCESS, case, execution],
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0, f"{case}/{execution} unexpectedly succeeded"
    assert expected_message in output, output


@dataclass(frozen=True)
class CategoricalSamplingDistributionCase:
    name: str
    logits: torch.Tensor
    expected_probabilities: torch.Tensor


def _categorical_sampling_distribution_cases() -> list[CategoricalSamplingDistributionCase]:
    uniform_logits = torch.zeros(20, dtype=torch.float32)
    uniform_probabilities = torch.full((20,), 1 / 20, dtype=torch.float64)

    skewed_probabilities = torch.tensor([0.70, 0.20, 0.09, 0.01], dtype=torch.float64)
    skewed_logits = skewed_probabilities.log().to(torch.float32)
    skewed_reference = torch.softmax(skewed_logits.to(torch.float64), dim=0)

    masked_logits = torch.full((33,), -float("inf"), dtype=torch.float32)
    masked_indices = torch.tensor([1, 7, 18, 31], dtype=torch.int64)
    masked_values = torch.tensor([0.50, 0.30, 0.15, 0.05], dtype=torch.float64)
    masked_logits[masked_indices] = masked_values.log().to(torch.float32)
    masked_reference = torch.zeros(33, dtype=torch.float64)
    masked_reference[masked_indices] = torch.softmax(masked_logits[masked_indices].to(torch.float64), dim=0)

    positive_infinity_logits = torch.zeros(20, dtype=torch.float32)
    positive_infinity_indices = torch.tensor([0, 5, 11, 19], dtype=torch.int64)
    positive_infinity_logits[positive_infinity_indices] = float("inf")
    positive_infinity_reference = torch.zeros(20, dtype=torch.float64)
    positive_infinity_reference[positive_infinity_indices] = 0.25

    return [
        CategoricalSamplingDistributionCase("uniform", uniform_logits, uniform_probabilities),
        CategoricalSamplingDistributionCase("skewed", skewed_logits, skewed_reference),
        CategoricalSamplingDistributionCase("masked", masked_logits, masked_reference),
        CategoricalSamplingDistributionCase("positive_infinity", positive_infinity_logits, positive_infinity_reference),
    ]


def _merge_small_expected_buckets(
    observed: torch.Tensor,
    expected: torch.Tensor,
    minimum_expected: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    large = expected >= minimum_expected
    merged_observed = observed[large]
    merged_expected = expected[large]
    if bool((~large).any()):
        small_expected = expected[~large].sum()
        if float(small_expected) < minimum_expected:
            raise AssertionError("sample count is too small for a valid chi-squared test")
        merged_observed = torch.cat((merged_observed, observed[~large].sum().reshape(1)))
        merged_expected = torch.cat((merged_expected, small_expected.reshape(1)))
    return merged_observed, merged_expected


def _chi_squared_p_value(observed: torch.Tensor, probabilities: torch.Tensor) -> tuple[float, float]:
    positive = probabilities > 0
    observed_support = observed[positive].to(torch.float64)
    expected_support = probabilities[positive] * observed.sum()
    observed_support, expected_support = _merge_small_expected_buckets(observed_support, expected_support)
    statistic = float((((observed_support - expected_support) ** 2) / expected_support).sum())
    degrees_of_freedom = observed_support.numel() - 1
    p_value = float(
        torch.special.gammaincc(
            torch.tensor(degrees_of_freedom / 2, dtype=torch.float64),
            torch.tensor(statistic / 2, dtype=torch.float64),
        )
    )
    return statistic, p_value


def _collect_categorical_sampling_distribution_with_aclgraph(logits_row: torch.Tensor) -> torch.Tensor:
    vocab_size = logits_row.numel()
    logits = logits_row.repeat(STATISTICAL_BATCH_SIZE, 1).to(DEVICE)
    mapping = torch.zeros(STATISTICAL_BATCH_SIZE, dtype=torch.int32, device=DEVICE)
    temperature = torch.ones(1, dtype=torch.float32, device=DEVICE)
    seed = torch.tensor([0x0123456789ABCDEF], dtype=torch.int64, device=DEVICE)
    pos = torch.arange(STATISTICAL_BATCH_SIZE, dtype=torch.int64, device=DEVICE)

    for _ in range(2):
        _run_categorical_sampling(logits, mapping, temperature, seed, pos)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_outputs = _run_categorical_sampling(logits, mapping, temperature, seed, pos)
    torch.npu.synchronize()

    samples = torch.empty(STATISTICAL_SAMPLE_COUNT, dtype=torch.int64, device=DEVICE)
    replay_count = STATISTICAL_SAMPLE_COUNT // STATISTICAL_BATCH_SIZE
    for replay in range(replay_count):
        graph.replay()
        offset = replay * STATISTICAL_BATCH_SIZE
        samples[offset : offset + STATISTICAL_BATCH_SIZE].copy_(graph_outputs[0])
        pos.add_(STATISTICAL_BATCH_SIZE)
    torch.npu.synchronize()
    samples_cpu = samples.cpu()
    assert bool(((samples_cpu >= 0) & (samples_cpu < vocab_size)).all()), "sampled token is out of range"
    return torch.bincount(samples_cpu, minlength=vocab_size)


@pytest.mark.parametrize("case", _categorical_sampling_distribution_cases(), ids=lambda case: case.name)
def test_categorical_sampling_aclgraph_distribution(case: CategoricalSamplingDistributionCase) -> None:
    observed = _collect_categorical_sampling_distribution_with_aclgraph(case.logits)
    masked = case.expected_probabilities == 0
    assert int(observed[masked].sum()) == 0, f"{case.name}: sampled a zero-probability token"

    statistic, p_value = _chi_squared_p_value(observed, case.expected_probabilities)
    assert p_value >= STATISTICAL_ALPHA, (
        f"{case.name}: chi-squared distribution check failed: "
        f"statistic={statistic:.6f}, p_value={p_value:.6g}, alpha={STATISTICAL_ALPHA}"
    )
