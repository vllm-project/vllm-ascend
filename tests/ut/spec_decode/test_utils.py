# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm_ascend.spec_decode.utils.

These exercise the CPU/GPU correction helpers used by the async spec-decode
path on Ascend.
"""

from __future__ import annotations

import numpy as np
import torch

from vllm_ascend.spec_decode.utils import (
    correct_optimistic_seq_lens_cpu,
    update_num_computed_tokens_for_batch_change,
)


def _build_optimistic_seq_lens(
    prev_step_computed: np.ndarray,
    prev_drafts: np.ndarray,
    num_scheduled_step_n: np.ndarray,
) -> np.ndarray:
    """Recreate the value the scheduler would write into ``optimistic_seq_lens_cpu``.

    The scheduler advances ``num_computed_tokens_cpu`` by the previous step's
    scheduled-token count (``prev_drafts + 1``), which is the optimistic count
    assuming all drafts were accepted.
    """
    optimistic_num_computed = prev_step_computed + (prev_drafts + 1)
    return optimistic_num_computed + num_scheduled_step_n


def _ground_truth_seq_lens(
    prev_step_computed: np.ndarray,
    valid_count: np.ndarray,
    num_scheduled_step_n: np.ndarray,
) -> np.ndarray:
    """``self.seq_lens`` (GPU) carries this exact value after correction."""
    return prev_step_computed + valid_count + num_scheduled_step_n


def test_correct_optimistic_seq_lens_cpu_matches_gpu_seq_lens():
    """CPU correction must match the seq_lens that the GPU path computes."""
    num_reqs = 5
    # prev_step_computed is C[N-2], i.e. num_computed at end of step N-2
    prev_step_computed = np.array([100, 200, 50, 0, 300], dtype=np.int32)
    # Drafts scheduled in step N-1; participating reqs have prev_drafts > 0
    prev_drafts = np.array([2, 4, 0, 0, 3], dtype=np.int32)
    # Drafts actually accepted; valid_count == 1 + accepted (bonus + accepted)
    accepted = np.array([2, 0, 0, 0, 1], dtype=np.int32)
    valid_count = accepted + 1
    # Step N's scheduled count
    num_scheduled_step_n = np.array([3, 5, 1, 1, 4], dtype=np.int32)
    # prev_positions: -1 for new requests, otherwise gather index
    prev_positions = np.array([0, 1, 2, -1, 4], dtype=np.int32)

    optimistic = _build_optimistic_seq_lens(prev_step_computed, prev_drafts, num_scheduled_step_n).astype(np.int32)

    correct_optimistic_seq_lens_cpu(
        optimistic,
        prev_positions,
        prev_drafts,
        valid_count.astype(np.int32),
        num_reqs,
    )

    expected = _ground_truth_seq_lens(prev_step_computed, valid_count, num_scheduled_step_n)
    # Non-participating requests (prev_drafts == 0 or prev_positions < 0) keep
    # the optimistic value, which already coincides with the truth because
    # there were no drafts to reject.
    non_participating = (prev_drafts == 0) | (prev_positions < 0)
    expected[non_participating] = _build_optimistic_seq_lens(
        prev_step_computed[non_participating],
        prev_drafts[non_participating],
        num_scheduled_step_n[non_participating],
    )
    np.testing.assert_array_equal(optimistic, expected)


def test_correct_optimistic_seq_lens_cpu_no_participants():
    """No participating reqs → optimistic_seq_lens unchanged."""
    optimistic = np.array([10, 20, 30], dtype=np.int32)
    correct_optimistic_seq_lens_cpu(
        optimistic,
        np.array([-1, -1, -1], dtype=np.int32),
        np.zeros(3, dtype=np.int32),
        np.zeros(3, dtype=np.int32),
        3,
    )
    np.testing.assert_array_equal(optimistic, np.array([10, 20, 30]))


def test_correct_optimistic_seq_lens_cpu_all_accepted():
    """When every draft was accepted, the correction is a no-op."""
    num_reqs = 3
    prev_drafts = np.array([2, 3, 1], dtype=np.int32)
    valid_count = (prev_drafts + 1).astype(np.int32)  # all accepted
    prev_positions = np.array([0, 1, 2], dtype=np.int32)
    optimistic = np.array([105, 208, 51], dtype=np.int32)
    expected = optimistic.copy()
    correct_optimistic_seq_lens_cpu(optimistic, prev_positions, prev_drafts, valid_count, num_reqs)
    np.testing.assert_array_equal(optimistic, expected)


def test_correct_optimistic_seq_lens_cpu_all_rejected():
    """When every draft was rejected (only bonus token kept), correction == prev_drafts."""
    num_reqs = 3
    prev_drafts = np.array([2, 3, 1], dtype=np.int32)
    valid_count = np.array([1, 1, 1], dtype=np.int32)  # only bonus kept
    prev_positions = np.array([0, 1, 2], dtype=np.int32)
    optimistic = np.array([105, 208, 51], dtype=np.int32)
    expected = optimistic - prev_drafts
    correct_optimistic_seq_lens_cpu(optimistic, prev_positions, prev_drafts, valid_count, num_reqs)
    np.testing.assert_array_equal(optimistic, expected)


def test_correct_optimistic_seq_lens_cpu_in_place_mutation():
    """The function must modify the input array in place."""
    optimistic = np.array([100, 200], dtype=np.int64)
    prev_drafts = np.array([2, 0], dtype=np.int32)
    valid_count = np.array([2, 0], dtype=np.int32)  # 1 of 2 accepted, prefill
    prev_positions = np.array([0, -1], dtype=np.int32)
    pre_id = id(optimistic)
    correct_optimistic_seq_lens_cpu(optimistic, prev_positions, prev_drafts, valid_count, 2)
    assert id(optimistic) == pre_id
    # req 0: correction = prev_drafts + 1 - valid_count = 2 + 1 - 2 = 1
    # req 1: not participating → unchanged
    np.testing.assert_array_equal(optimistic, np.array([99, 200]))


def test_correct_optimistic_seq_lens_cpu_partial_batch():
    """Only the first num_reqs entries of the buffer are touched."""
    optimistic = np.array([100, 200, 999, 999], dtype=np.int32)
    prev_drafts = np.array([2, 1, 0, 0], dtype=np.int32)
    valid_count = np.array([2, 2, 0, 0], dtype=np.int32)
    prev_positions = np.array([0, 1, -1, -1], dtype=np.int32)

    correct_optimistic_seq_lens_cpu(optimistic, prev_positions, prev_drafts, valid_count, 2)
    # req 0: correction = 2 + 1 - 2 = 1, → 99
    # req 1: correction = 1 + 1 - 2 = 0, → 200 unchanged
    # tail (idx 2,3) untouched
    np.testing.assert_array_equal(optimistic, np.array([99, 200, 999, 999]))


def test_cpu_and_gpu_corrections_agree():
    """The CPU helper must agree with ``update_num_computed_tokens_for_batch_change``.

    They live on different sides of the device boundary, but both implement
    the same correction. We compare the post-correction seq_lens (= corrected
    num_computed_tokens + num_scheduled_step_n) on each path.
    """
    num_reqs = 6
    prev_step_computed = np.array([100, 250, 80, 0, 410, 5], dtype=np.int32)
    prev_drafts = np.array([2, 4, 0, 0, 3, 1], dtype=np.int32)
    accepted = np.array([2, 1, 0, 0, 0, 1], dtype=np.int32)
    valid_count = (accepted + 1).astype(np.int32)
    num_scheduled_step_n = np.array([3, 5, 1, 1, 4, 2], dtype=np.int32)
    prev_positions = np.array([0, 1, 2, -1, 4, 5], dtype=np.int32)

    # CPU path
    optimistic = _build_optimistic_seq_lens(prev_step_computed, prev_drafts, num_scheduled_step_n).astype(np.int32)
    correct_optimistic_seq_lens_cpu(optimistic, prev_positions, prev_drafts, valid_count, num_reqs)

    # GPU path on CPU device for portability
    cpu_num_computed = torch.from_numpy(
        prev_step_computed + prev_drafts + 1  # scheduler-bumped optimistic
    ).to(torch.int32)
    num_computed_gpu = torch.from_numpy(prev_step_computed.copy()).to(torch.int32)
    num_accepted_gpu = torch.zeros(num_reqs, dtype=torch.int32)
    valid_count_t = torch.from_numpy(valid_count)
    prev_positions_t = torch.from_numpy(prev_positions)
    prev_drafts_t = torch.from_numpy(prev_drafts)
    update_num_computed_tokens_for_batch_change(
        num_computed_gpu,
        num_accepted_gpu,
        prev_positions_t,
        valid_count_t,
        prev_drafts_t,
        cpu_num_computed,
    )

    gpu_seq_lens = num_computed_gpu.numpy() + num_scheduled_step_n

    np.testing.assert_array_equal(optimistic, gpu_seq_lens)
