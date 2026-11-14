# SPDX-License-Identifier: Apache-2.0
"""CPU validation of persistent 310P MRoPE host-buffer updates."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend._310p.worker.v2.rope import Ascend310PRopeState


@pytest.mark.parametrize("num_dims", [3, 4])
def test_mrope_mixed_prefill_decode_preserves_padding_and_dummy_column(num_dims):
    state = object.__new__(Ascend310PRopeState)
    state.num_dims = num_dims
    state.positions_cpu = torch.full((num_dims, 9), -71, dtype=torch.int64)
    state.positions_np = state.positions_cpu.numpy()
    state.positions = torch.full_like(state.positions_cpu, -90)
    state.prefill_delta = SimpleNamespace(np=np.array([5, 0, -2]))
    state.prefill_positions = SimpleNamespace(cpu=torch.arange(3 * num_dims * 12).view(3 * num_dims, 12))
    state.prepare_positions_cpu(
        np.array([2, 0, 1]), np.array([0, 2, 5, 6]), np.array([0, 10, 0]), np.array([7, 2, 4]), 8
    )
    expected = torch.zeros((num_dims, 9), dtype=torch.int64)
    expected[:, :2] = torch.tensor([2, 3])
    expected[:, 2:5] = torch.tensor([12, 13, 14])
    expected[:, 5] = state.prefill_positions.cpu[num_dims : 2 * num_dims, 2]
    expected[:, 8] = -71
    torch.testing.assert_close(state.positions_cpu, expected, rtol=0, atol=0)
    torch.testing.assert_close(state.positions, expected, rtol=0, atol=0)
