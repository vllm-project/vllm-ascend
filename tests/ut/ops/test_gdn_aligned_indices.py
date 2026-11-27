# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops.triton.mamba import aligned_indices


def _context(output):
    return SimpleNamespace(
        is_initialized=True,
        aligned_state_indices=output,
        block_table_ptrs=torch.zeros(output.shape[0], dtype=torch.int64),
        block_table_stride_req=64,
        block_size=16,
    )


@pytest.mark.parametrize("num_reqs,separate_tables", [(0, False), (17, False), (17, True)])
def test_gather_uses_existing_output(monkeypatch, num_reqs, separate_tables):
    kernel = MagicMock()
    monkeypatch.setattr(aligned_indices, "_aligned_indices_kernel", kernel)
    seq_lens = torch.ones(66, dtype=torch.int32)[::2]
    output = torch.full((24, 33, 3), -99, dtype=torch.int32)
    ctx = _context(output)
    table_ptrs = ctx.block_table_ptrs
    if separate_tables:
        ctx.aligned_index_block_table_ptrs = table_ptrs.clone()
        table_ptrs = ctx.aligned_index_block_table_ptrs
    actual = aligned_indices.compute_aligned_state_indices(ctx, seq_lens, num_reqs)
    assert actual.shape == (24, num_reqs, 3)
    assert actual.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
    if not num_reqs:
        kernel.__getitem__.assert_not_called()
        return
    kernel.__getitem__.assert_called_once()
    launch = kernel.__getitem__.return_value
    launch.assert_called_once()
    args, _ = launch.call_args
    assert args[0] is table_ptrs and args[1] is seq_lens and args[2] is output
    assert args[3:] == (64, 2, num_reqs)


def test_npu_gather_rejects_non_contiguous_output():
    output = torch.empty((24, 64, 8), dtype=torch.int32)[:, :, ::2]
    with pytest.raises(AssertionError):
        aligned_indices.compute_aligned_state_indices(_context(output), torch.ones(64, dtype=torch.int32), 1)
