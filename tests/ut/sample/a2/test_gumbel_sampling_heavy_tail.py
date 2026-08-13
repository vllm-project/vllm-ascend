# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Heavy-tail accuracy coverage for the AscendC-backed Gumbel wrapper."""

import math

import pytest
import torch
import torch_npu  # noqa: F401  # Registers the NPU custom-device APIs.

from tests.ut.sample.custom_op_utils import require_categorical_sampling_operator
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample

DEVICE = torch.device("npu")
VOCAB_SIZE = 200_000
NUM_SAMPLES = 131_072
SAMPLE_CHUNK_SIZE = 4_096
HEAD_LOG_GAP = 18.0
Z_TOLERANCE = 10.0


@pytest.fixture(scope="module", autouse=True)
def _require_categorical_sampling_operator() -> None:
    require_categorical_sampling_operator()


def _make_heavy_tailed_counts() -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234)
    counts = torch.randint(1, 4, (VOCAB_SIZE,), generator=generator, dtype=torch.int64)
    counts[0] = round(math.exp(HEAD_LOG_GAP))
    return counts


def _z_score(observed: int, expected: float, num_trials: int) -> float:
    probability = expected / num_trials
    return (observed - expected) / math.sqrt(num_trials * probability * (1.0 - probability))


def _sample_broadcast_logits(logits_1d: torch.Tensor, use_fp64: bool) -> torch.Tensor:
    sampled_chunks = []
    temperature = torch.ones(1, dtype=torch.float32, device=DEVICE)
    seed = torch.tensor([0xABCD], dtype=torch.int64, device=DEVICE)
    for start in range(0, NUM_SAMPLES, SAMPLE_CHUNK_SIZE):
        chunk_size = min(SAMPLE_CHUNK_SIZE, NUM_SAMPLES - start)
        logits = logits_1d.unsqueeze(0).expand(chunk_size, VOCAB_SIZE)
        assert logits.stride() == (0, 1)
        sampled_chunks.append(
            gumbel_sample(
                logits,
                torch.zeros(chunk_size, dtype=torch.int32, device=DEVICE),
                temperature,
                seed,
                torch.arange(start, start + chunk_size, dtype=torch.int64, device=DEVICE),
                apply_temperature=True,
                use_fp64=use_fp64,
            )
        )
    return torch.cat(sampled_chunks)


@pytest.mark.parametrize("use_fp64", [False, True])
def test_heavy_tail_sampling_matches_target_distribution(use_fp64: bool) -> None:
    """The aggregate tail remains reachable beyond the naive FP32 Gumbel cap."""
    counts = _make_heavy_tailed_counts()
    total = counts.sum().item()
    logits = counts.to(torch.float64).log().to(torch.float32).to(DEVICE)

    sampled = _sample_broadcast_logits(logits, use_fp64)
    tail_count = torch.count_nonzero(sampled != 0).item()
    tail_probability = (total - counts[0].item()) / total
    z_score = _z_score(tail_count, NUM_SAMPLES * tail_probability, NUM_SAMPLES)

    assert sampled.min().item() >= 0
    assert sampled.max().item() < VOCAB_SIZE
    assert abs(z_score) < Z_TOLERANCE, (
        f"sampled tail mass {tail_count / NUM_SAMPLES:.3e} != target {tail_probability:.3e} (z={z_score:.2f})"
    )
