# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState


def _state():
    return AscendMambaHybridModelState.__new__(AscendMambaHybridModelState)


def test_cached_views_read_current_gather_values_with_new_tensor_view_objects():
    state = _state()
    backing = torch.arange(4 * 8 * 3, dtype=torch.int32).view(4, 8, 3)
    rows = state._get_aligned_state_index_views(backing[:, :2])
    backing.add_(100)
    current = state._get_aligned_state_index_views(backing[:, :2])
    assert current is rows
    for i, row in enumerate(current):
        torch.testing.assert_close(row, backing[i, :2], rtol=0, atol=0)
        assert row.data_ptr() == backing[i].data_ptr()


@pytest.mark.parametrize("counts", [(2, 1, 2), (1, 8, 3), (0, 1, 0)])
def test_request_count_changes_rebuild_shape_without_retaining_stale_rows(counts):
    state = _state()
    backing = torch.arange(4 * 8 * 3, dtype=torch.int32).view(4, 8, 3)
    last = None
    for count in counts:
        rows = state._get_aligned_state_index_views(backing[:, :count])
        assert all(row.shape == (count, 3) for row in rows)
        if last is not None:
            assert rows is not last
        for i, row in enumerate(rows):
            torch.testing.assert_close(row, backing[i, :count], rtol=0, atol=0)
        last = rows
        backing.add_(10)


def test_same_shape_replaced_storage_rebinds_to_new_current_indices():
    state = _state()
    original = torch.arange(4 * 2 * 3, dtype=torch.int32).view(4, 2, 3)
    rows = state._get_aligned_state_index_views(original)
    replacement = original.clone() + 123
    current = state._get_aligned_state_index_views(replacement)
    assert current is not rows
    for i, row in enumerate(current):
        assert row.data_ptr() != rows[i].data_ptr()
        torch.testing.assert_close(row, replacement[i], rtol=0, atol=0)


def test_same_pointer_and_shape_with_changed_stride_does_not_reuse_old_views():
    state = _state()
    backing = torch.arange(4 * 3 * 3, dtype=torch.int32).view(4, 3, 3)
    rows = state._get_aligned_state_index_views(backing)
    transposed = backing.transpose(1, 2)
    current = state._get_aligned_state_index_views(transposed)
    assert current is not rows
    for i, row in enumerate(current):
        torch.testing.assert_close(row, transposed[i], rtol=0, atol=0)


def test_same_pointer_shape_stride_with_changed_dtype_rebuilds():
    state = _state()
    backing = torch.arange(4 * 3 * 3, dtype=torch.int32).view(4, 3, 3)
    rows = state._get_aligned_state_index_views(backing)
    interpreted = backing.view(torch.float32)
    current = state._get_aligned_state_index_views(interpreted)
    assert current is not rows
    assert all(row.dtype == torch.float32 for row in current)
