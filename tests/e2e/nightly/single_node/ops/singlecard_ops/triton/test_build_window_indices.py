# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical regression coverage for the explicit A5 SWA KV prefix."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.dsa_v41 import build_replay_swa_visible_lens
from vllm_ascend.ops.triton.build_window_indices import build_window_indices_triton
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

WINDOW = 128
DEVICE = "npu:0"

pytestmark = pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="Requires Ascend A5")


def _assert_rows(indices, lengths, expected, window_size):
    rows = indices.squeeze(1).cpu().tolist()
    counts = lengths.flatten().cpu().tolist()
    assert indices.dtype == lengths.dtype == torch.int32
    assert counts == [len(row) for row in expected]
    assert rows == [row + [-1] * (window_size - len(row)) for row in expected]


def test_replay_boundary_mixed_requests_and_padding():
    query_positions = list(range(33792, 33920)) + [0, 1, 639]
    positions = torch.tensor(query_positions + [0, 0], dtype=torch.int64, device=DEVICE)
    visible_lengths = build_replay_swa_visible_lens(
        WINDOW,
        torch.tensor([0, 128, 130, 131], dtype=torch.int32, device=DEVICE),
        torch.tensor([33920, 2, 640], dtype=torch.int32, device=DEVICE),
        torch.tensor([33792, 0, 639], dtype=torch.int32, device=DEVICE),
        len(query_positions),
    )
    normal_indices, normal_lengths = build_window_indices_triton(positions, WINDOW)
    expected_normal = [list(range(max(0, p - WINDOW + 1), p + 1)) for p in query_positions + [0, 0]]
    _assert_rows(normal_indices, normal_lengths, expected_normal, WINDOW)

    expected_replay = [list(range(33792, p + 1)) for p in range(33792, 33920)]
    expected_replay += [[0], [0, 1], [639], [], []]
    indices_output = torch.full_like(normal_indices, 12345)
    lengths_output = torch.full_like(normal_lengths, 12345)
    indices, lengths = build_window_indices_triton(
        positions,
        WINDOW,
        visible_lengths=visible_lengths,
        indices_output=indices_output,
        lengths_output=lengths_output,
    )
    assert indices is indices_output and lengths is lengths_output
    _assert_rows(indices, lengths, expected_replay, WINDOW)
    assert normal_indices[0, 0, 0].item() == 33665
    assert indices[0, 0, 0].item() == 33792


@pytest.mark.parametrize("window_size", [1, 7, WINDOW])
@pytest.mark.parametrize("num_tokens", [0, 1, 6])
def test_visible_length_limits_and_empty_input(window_size, num_tokens):
    query_positions = [0, 1, 3, 128, 256, 639][:num_tokens]
    positions = torch.tensor(query_positions, dtype=torch.int64, device=DEVICE)
    # The sliced input is noncontiguous. Its missing tail denotes padding.
    visible = torch.tensor([0, 99, -2, 99, 1000, 99, 1, 99], dtype=torch.int32, device=DEVICE)[::2]
    visible = visible[:num_tokens]
    indices, lengths = build_window_indices_triton(positions, window_size, visible_lengths=visible)
    floors = [1, 2, 0, 128, 257, 640][:num_tokens]
    expected = [list(range(max(0, p - window_size + 1, floor), p + 1)) for p, floor in zip(query_positions, floors)]
    _assert_rows(indices, lengths, expected, window_size)
