#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""Unit tests for _compute_rope_block_size_head.

These tests verify that the dynamic tile sizing logic correctly:
  1. Produces BLOCK_SIZE_HEAD values that fit within the UB safety budget.
  2. Respects the power-of-2 constraint required by Triton.
  3. Caps results at the default max for each kernel variant.
  4. Falls back gracefully when rope_dim is unknown (-1).
  5. Returns at least _ROPE_MIN_BLOCK_SIZE (1) for extreme inputs.

The tests do NOT require NPU hardware; get_ub_size_bytes() is mocked to
a controlled value so the arithmetic is deterministic.
"""

from unittest.mock import patch

import pytest

from vllm_ascend.ops.triton.rope import (
    _ROPE_DEFAULT_BLOCK_SIZE_NEOX,
    _ROPE_DEFAULT_BLOCK_SIZE_NON_NEOX,
    _ROPE_FLOAT32_BYTES,
    _ROPE_MIN_BLOCK_SIZE,
    _ROPE_NUM_LIVE_TENSORS_NEOX,
    _ROPE_NUM_LIVE_TENSORS_NON_NEOX,
    _ROPE_UB_SAFETY_FACTOR,
    _compute_rope_block_size_head,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _expected_block(
    head_dim: int,
    rope_dim: int,
    is_neox_style: bool,
    ub_bytes: int,
) -> int:
    """Reimplement the tile-sizing formula independently for cross-checking."""
    effective = rope_dim if rope_dim > 0 else head_dim
    # Triton's next_power_of_2(n) returns the smallest 2^k >= n, with 2^k=1
    # when n <= 1.
    pad = 1
    while pad < effective:
        pad *= 2
    half = pad // 2
    half = max(half, 1)

    if is_neox_style:
        num_live = _ROPE_NUM_LIVE_TENSORS_NEOX
        bytes_per_head_per_tensor = half * _ROPE_FLOAT32_BYTES
        default_max = _ROPE_DEFAULT_BLOCK_SIZE_NEOX
    else:
        num_live = _ROPE_NUM_LIVE_TENSORS_NON_NEOX
        bytes_per_head_per_tensor = half * 2 * _ROPE_FLOAT32_BYTES
        default_max = _ROPE_DEFAULT_BLOCK_SIZE_NON_NEOX

    shared = 2 * half * _ROPE_FLOAT32_BYTES
    available = int(ub_bytes * _ROPE_UB_SAFETY_FACTOR) - shared
    bytes_per_head = num_live * bytes_per_head_per_tensor
    if bytes_per_head <= 0:
        return default_max
    max_block = max(available // bytes_per_head, _ROPE_MIN_BLOCK_SIZE)
    block_size = min(max_block, default_max)
    result = 1
    while result * 2 <= block_size:
        result *= 2
    return result


def _ub_footprint_bytes(
    block_size: int,
    rope_dim: int,
    is_neox_style: bool,
) -> int:
    """Compute the actual UB footprint for a given (block_size, rope_dim)."""
    effective = rope_dim if rope_dim > 0 else 0
    pad = 1
    while pad < effective:
        pad *= 2
    half = max(pad // 2, 1)

    if is_neox_style:
        num_live = _ROPE_NUM_LIVE_TENSORS_NEOX
        bytes_per_head_per_tensor = half * _ROPE_FLOAT32_BYTES
    else:
        num_live = _ROPE_NUM_LIVE_TENSORS_NON_NEOX
        bytes_per_head_per_tensor = half * 2 * _ROPE_FLOAT32_BYTES

    shared = 2 * half * _ROPE_FLOAT32_BYTES
    return shared + block_size * num_live * bytes_per_head_per_tensor


# ──────────────────────────────────────────────────────────────────────────────
# Test fixtures
# ──────────────────────────────────────────────────────────────────────────────

# Default UB size used by most tests: 192 KB (Ascend 910B/C).
DEFAULT_UB_BYTES = 192 * 1024

# Realistic (head_dim, rope_dim) pairs seen in production models.
# Each entry is (head_dim, rope_dim, is_neox, description).
REALISTIC_CASES = [
    # Full RoPE, power-of-two
    (64, 64, True, "small head, full rope, neox"),
    (128, 128, True, "medium head, full rope, neox (Qwen/Llama)"),
    (256, 256, True, "large head, full rope, neox"),
    # Partial RoPE (rope_dim < head_dim)
    (128, 64, True, "partial rope, neox (Llama-2 style)"),
    (256, 128, True, "large head, partial rope, neox"),
    # Non-power-of-two rotary_dim (triggers sin-offset path)
    (128, 96, True, "non-pow2 rope, neox"),
    (256, 192, True, "large non-pow2 rope, neox"),
    # Non-NeoX (GPT-J style) variants
    (64, 64, False, "small head, full rope, non-neox"),
    (128, 128, False, "medium head, full rope, non-neox"),
    (128, 96, False, "non-pow2 rope, non-neox"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_ub_size():
    """Mock get_ub_size_bytes to return DEFAULT_UB_BYTES."""
    with patch(
        "vllm_ascend.ops.triton.rope.get_ub_size_bytes",
        return_value=DEFAULT_UB_BYTES,
    ):
        yield


class TestComputeBlockSizeHead:
    """Functional correctness of _compute_rope_block_size_head."""

    @pytest.mark.parametrize(
        "head_dim, rope_dim, is_neox, desc",
        REALISTIC_CASES,
        ids=[c[3] for c in REALISTIC_CASES],
    )
    def test_result_matches_independent_formula(self, mock_ub_size, head_dim, rope_dim, is_neox, desc):
        """Cross-check against an independent reimplementation of the formula."""
        actual = _compute_rope_block_size_head(head_dim, rope_dim, is_neox)
        expected = _expected_block(head_dim, rope_dim, is_neox, DEFAULT_UB_BYTES)
        assert actual == expected, (
            f"{desc}: got {actual}, expected {expected} (head_dim={head_dim}, rope_dim={rope_dim}, neox={is_neox})"
        )

    @pytest.mark.parametrize(
        "head_dim, rope_dim, is_neox, desc",
        REALISTIC_CASES,
        ids=[c[3] for c in REALISTIC_CASES],
    )
    def test_result_is_power_of_two(self, mock_ub_size, head_dim, rope_dim, is_neox, desc):
        """Triton requires BLOCK_SIZE_HEAD to be a power-of-2 constexpr."""
        result = _compute_rope_block_size_head(head_dim, rope_dim, is_neox)
        assert result >= 1
        assert (result & (result - 1)) == 0, f"{desc}: {result} is not a power of 2"

    @pytest.mark.parametrize(
        "head_dim, rope_dim, is_neox, desc",
        REALISTIC_CASES,
        ids=[c[3] for c in REALISTIC_CASES],
    )
    def test_result_respects_default_cap(self, mock_ub_size, head_dim, rope_dim, is_neox, desc):
        """Result must not exceed the default max for the kernel variant."""
        result = _compute_rope_block_size_head(head_dim, rope_dim, is_neox)
        cap = _ROPE_DEFAULT_BLOCK_SIZE_NEOX if is_neox else _ROPE_DEFAULT_BLOCK_SIZE_NON_NEOX
        assert result <= cap, f"{desc}: {result} > cap {cap}"

    @pytest.mark.parametrize(
        "head_dim, rope_dim, is_neox, desc",
        REALISTIC_CASES,
        ids=[c[3] for c in REALISTIC_CASES],
    )
    def test_result_fits_within_ub_budget(self, mock_ub_size, head_dim, rope_dim, is_neox, desc):
        """The computed tile must fit within (ub_size * safety_factor)."""
        result = _compute_rope_block_size_head(head_dim, rope_dim, is_neox)
        footprint = _ub_footprint_bytes(result, rope_dim, is_neox)
        budget = int(DEFAULT_UB_BYTES * _ROPE_UB_SAFETY_FACTOR)
        assert footprint <= budget, (
            f"{desc}: footprint={footprint} > budget={budget} (block={result}, rope_dim={rope_dim}, neox={is_neox})"
        )


class TestBlockSizeWithinBudget:
    """Verify that doubling the block would exceed the UB budget.

    This guards against the tile being needlessly conservative: if 2*block
    still fits in UB, the power-of-2 search should have picked it instead.
    """

    @pytest.mark.parametrize(
        "head_dim, rope_dim, is_neox, desc",
        REALISTIC_CASES,
        ids=[c[3] for c in REALISTIC_CASES],
    )
    def test_next_power_of_two_would_overflow_ub(self, mock_ub_size, head_dim, rope_dim, is_neox, desc):
        result = _compute_rope_block_size_head(head_dim, rope_dim, is_neox)
        cap = _ROPE_DEFAULT_BLOCK_SIZE_NEOX if is_neox else _ROPE_DEFAULT_BLOCK_SIZE_NON_NEOX
        # If already at the default cap, doubling is capped away — skip.
        if result >= cap:
            return
        doubled = result * 2
        footprint_doubled = _ub_footprint_bytes(doubled, rope_dim, is_neox)
        budget = int(DEFAULT_UB_BYTES * _ROPE_UB_SAFETY_FACTOR)
        assert footprint_doubled > budget, (
            f"{desc}: doubled block {doubled} fits ({footprint_doubled} <= {budget}) but function returned {result}"
        )


class TestRopeDimFallback:
    """rope_dim=-1 (unknown) should fall back to head_dim."""

    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    @pytest.mark.parametrize("is_neox", [True, False])
    def test_unknown_rope_dim_uses_head_dim(self, mock_ub_size, head_dim, is_neox):
        """When rope_dim=-1, the function uses head_dim as an upper bound."""
        result_unknown = _compute_rope_block_size_head(head_dim, -1, is_neox)
        result_full = _compute_rope_block_size_head(head_dim, head_dim, is_neox)
        assert result_unknown == result_full, (
            f"head_dim={head_dim}, neox={is_neox}: rope_dim=-1 gave {result_unknown}, full gave {result_full}"
        )


class TestMinimumBlockSize:
    """Extreme small UB should still yield at least _ROPE_MIN_BLOCK_SIZE."""

    def test_tiny_ub_returns_minimum(self):
        """With a 1 KB UB, no realistic tile fits — should return 1."""
        with patch(
            "vllm_ascend.ops.triton.rope.get_ub_size_bytes",
            return_value=1024,
        ):
            result = _compute_rope_block_size_head(128, 128, True)
        assert result >= _ROPE_MIN_BLOCK_SIZE

    def test_zero_ub_returns_minimum(self):
        """With a zero UB, the function must not crash and return >= 1."""
        with patch(
            "vllm_ascend.ops.triton.rope.get_ub_size_bytes",
            return_value=0,
        ):
            result = _compute_rope_block_size_head(128, 128, True)
        assert result >= _ROPE_MIN_BLOCK_SIZE


class TestUbScaling:
    """Tile size should scale (weakly) up with available UB."""

    @pytest.mark.parametrize("is_neox", [True, False])
    def test_larger_ub_never_reduces_tile(self, is_neox):
        """Doubling UB should never produce a smaller tile."""
        with patch(
            "vllm_ascend.ops.triton.rope.get_ub_size_bytes",
            return_value=128 * 1024,
        ):
            small_ub = _compute_rope_block_size_head(128, 128, is_neox)
        with patch(
            "vllm_ascend.ops.triton.rope.get_ub_size_bytes",
            return_value=256 * 1024,
        ):
            big_ub = _compute_rope_block_size_head(128, 128, is_neox)
        assert big_ub >= small_ub


class TestLiveTensorRegression:
    """Guard against accidentally lowering the live-tensor counts.

    The counts (8/10) were calibrated by inspecting the Triton Ascend
    compiler IR and verified empirically on Ascend 910B. Lowering them
    would cause UB overflow at the original BLOCK_SIZE_HEAD=64.
    """

    def test_neox_live_tensor_count(self):
        assert _ROPE_NUM_LIVE_TENSORS_NEOX == 8, (
            "NeoX live-tensor count was calibrated to 8 from compiler IR; "
            "do not lower without re-running UB overflow verification."
        )

    def test_non_neox_live_tensor_count(self):
        assert _ROPE_NUM_LIVE_TENSORS_NON_NEOX == 10, (
            "Non-NeoX live-tensor count was calibrated to 10 from compiler "
            "IR; do not lower without re-running UB overflow verification."
        )

    def test_neox_count_higher_than_original_undercount(self):
        """The original code had 4/6 — the fix raises to 8/10."""
        assert _ROPE_NUM_LIVE_TENSORS_NEOX > 4
        assert _ROPE_NUM_LIVE_TENSORS_NON_NEOX > 6
