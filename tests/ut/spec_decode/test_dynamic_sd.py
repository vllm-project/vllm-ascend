# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Regression tests for the dynamic speculative-decoding schedule helpers."""

import pytest
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup


def _make_lookup(
    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]],
    vllm_max_batch_size: int = 256,
    vllm_num_speculative_tokens: int = 3,
) -> list[int]:
    return build_dynamic_sd_schedule_lookup(
        num_speculative_tokens_per_batch_size=num_speculative_tokens_per_batch_size,
        vllm_max_batch_size=vllm_max_batch_size,
        vllm_num_speculative_tokens=vllm_num_speculative_tokens,
    )


@pytest.mark.parametrize(
    ("batch_size", "expected_num_spec_tokens"),
    [
        (0, 0),
        (1, 3),
        (16, 3),
        (17, 3),
        (31, 3),
        (32, 2),
        (128, 2),
        (129, 2),
        (255, 2),
        (256, 0),
    ],
)
def test_dynamic_sd_schedule_boundaries_and_gaps(
    batch_size: int,
    expected_num_spec_tokens: int,
) -> None:
    """A gap inherits the speculative-token count from the prior range."""
    lookup = _make_lookup([(1, 16, 3), (32, 128, 2), (256, 256, 0)])

    assert lookup[batch_size] == expected_num_spec_tokens


def test_dynamic_sd_schedule_clamps_to_runtime_max() -> None:
    lookup = _make_lookup(
        [(1, 16, 4)],
        vllm_max_batch_size=16,
        vllm_num_speculative_tokens=3,
    )

    assert lookup[1:] == [3] * 16


@pytest.mark.parametrize(
    ("schedule", "error"),
    [
        ([(2, 16, 3)], "must start at 1"),
        ([(1, 16, 3), (16, 32, 2)], "non-overlapping and sorted"),
        ([(1, 16, -1)], "values must be >= 0"),
        ([], "must not be empty"),
    ],
)
def test_dynamic_sd_schedule_rejects_invalid_ranges(
    schedule: list[tuple[int, int, int]],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _make_lookup(schedule)


def test_dynamic_sd_schedule_rejects_malformed_entry() -> None:
    with pytest.raises(ValueError, match="3-item sequence"):
        _make_lookup([(1, 16)])  # type: ignore[list-item]


def test_dynamic_sd_schedule_is_required() -> None:
    with pytest.raises(ValueError, match="is required"):
        build_dynamic_sd_schedule_lookup(
            num_speculative_tokens_per_batch_size=None,
            vllm_max_batch_size=256,
            vllm_num_speculative_tokens=5,
        )


def test_dynamic_sd_lookup_rejects_batch_size_above_max() -> None:
    lookup = _make_lookup([(1, 256, 3)])

    with pytest.raises(IndexError):
        _ = lookup[257]
