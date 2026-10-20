# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops.triton.v2.mamba import aligned_indices


@pytest.fixture(autouse=True)
def _patch_missing_next_power_of_2(monkeypatch):
    # vLLM's CPU Triton placeholder omits this host arithmetic helper.
    if not hasattr(aligned_indices.triton, "next_power_of_2"):
        monkeypatch.setattr(
            aligned_indices.triton,
            "next_power_of_2",
            lambda value: 1 << (value - 1).bit_length(),
            raising=False,
        )


@pytest.mark.parametrize("num_reqs", [1, 2, 17, 32, 33, 128])
@pytest.mark.parametrize("slots", [1, 3, 4, 7, 8, 16, 17])
def test_window_launch_covers_rows_without_partial_write_overlap(num_reqs, slots):
    config = aligned_indices._aligned_index_launch_config(24, num_reqs, 128, slots, 48)
    grid, block_rows, window, tiles_per_program = config
    assert grid[0] == 24 and grid[1] <= 2
    assert block_rows * slots * 4 % 32 == 0
    assert block_rows * window <= 4096
    assert window >= slots + 7
    assert grid[1] * tiles_per_program * block_rows >= num_reqs
    assert (grid[1] - 1) * tiles_per_program * block_rows < num_reqs


@pytest.mark.parametrize("num_reqs", [0, 1, 17, 33])
@pytest.mark.parametrize("max_reqs,slots", [(64, 4), (64, 3), (33, 4), (33, 3), (33, 1)])
def test_npu_gather_dispatches_once_and_keeps_output_storage(monkeypatch, num_reqs, max_reqs, slots):
    window_kernel, flat_kernel = MagicMock(), MagicMock()
    monkeypatch.setattr(aligned_indices, "_aligned_indices_window_kernel", window_kernel)
    monkeypatch.setattr(aligned_indices, "_aligned_indices_flat_kernel", flat_kernel)
    initialize = MagicMock()
    monkeypatch.setattr(aligned_indices, "init_device_properties_triton", initialize)
    monkeypatch.setattr(aligned_indices, "get_vectorcore_num", lambda: 48)
    table_ptrs = torch.zeros(24, dtype=torch.int64)
    seq_lens = torch.ones(max_reqs * 2, dtype=torch.int32)[::2]
    output = torch.full((24, max_reqs, slots), -99, dtype=torch.int32)
    actual = aligned_indices.compute_aligned_state_indices(table_ptrs, seq_lens, output, num_reqs, 64, 32, 16)
    assert actual.shape == (24, num_reqs, slots)
    assert torch.all(output == -99)
    if not num_reqs:
        initialize.assert_not_called()
        window_kernel.__getitem__.assert_not_called()
        flat_kernel.__getitem__.assert_not_called()
        return
    assert actual.data_ptr() == output.data_ptr()
    selected, unused = (window_kernel, flat_kernel) if max_reqs * slots % 8 == 0 else (flat_kernel, window_kernel)
    selected.__getitem__.assert_called_once()
    selected.__getitem__.return_value.assert_called_once()
    unused.__getitem__.assert_not_called()
    args, kwargs = selected.__getitem__.return_value.call_args
    assert args[0] is table_ptrs and args[1] is seq_lens and args[2] is output
    assert args[3:5] == (64, 2)
    assert kwargs["multibuffer"] is False and kwargs["unit_flag"] is False


def test_npu_gather_rejects_non_contiguous_output(monkeypatch):
    monkeypatch.setattr(aligned_indices, "init_device_properties_triton", lambda: pytest.fail("must validate first"))
    output = torch.empty((24, 64, 8), dtype=torch.int32)[:, :, ::2]
    with pytest.raises(AssertionError):
        aligned_indices.compute_aligned_state_indices(
            torch.zeros(24, dtype=torch.int64), torch.ones(64, dtype=torch.int32), output, 1, 32, 32, 16
        )
