# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the UVA host-shadow dirty-row tracking.

These buffers are sized like the flattened block table (millions of rows), so
recording dirty rows per element used to make a single ``np[:n] = x`` cost
hundreds of milliseconds of host time. The tracker must stay O(1) for the
contiguous-prefix writes and still copy scattered writes correctly.
"""

import numpy as np
import pytest
import torch

from vllm_ascend.patch.worker.patch_uva import (
    MonitoredNumPyArray,
    MonitoredTorchTensor,
    UvaBufferWrapper,
    _DirtyRows,
)


def _fake_buffer(rows: int, cols: int = 2, dtype=torch.int32) -> UvaBufferWrapper:
    """A wrapper backed by two CPU tensors, so no NPU is required."""
    buf = UvaBufferWrapper.__new__(UvaBufferWrapper)
    buf._cpu = torch.zeros((rows, cols), dtype=dtype)
    buf._np = buf._cpu.numpy()
    buf._dirty = _DirtyRows(rows)
    buf._uva = torch.zeros((rows, cols), dtype=dtype)
    buf._use_real_uva = False
    buf._cpu_view = MonitoredTorchTensor(buf._cpu, buf._dirty.mark)
    buf._np_view = MonitoredNumPyArray(buf._np, buf._dirty.mark)
    return buf


def test_prefix_write_is_tracked_without_allocating_a_mask():
    dirty = _DirtyRows(2_097_152)

    dirty.mark(slice(0, 8192))

    assert dirty.take() == [(0, 8192)]
    # The whole point: no per-row set and no 2 MB mask for a prefix write.
    assert dirty._mask is None


def test_contiguous_growth_merges_into_one_run():
    dirty = _DirtyRows(16)

    dirty.mark(slice(0, 3))
    dirty.mark(slice(3, 5))

    assert dirty.take() == [(0, 5)]


def test_scattered_writes_stay_separate_runs():
    dirty = _DirtyRows(16)

    dirty.mark(slice(0, 2))
    dirty.mark(7)

    assert dirty.take() == [(0, 2), (7, 8)]


def test_scattered_run_touching_the_prefix_is_coalesced():
    dirty = _DirtyRows(16)

    dirty.mark(4)
    dirty.mark(5)
    dirty.mark(slice(0, 4))

    # Rows 0-5 are dirty and contiguous, so one H2D copy is enough.
    assert dirty.take() == [(0, 6)]


def test_fancy_index_writes_cover_exact_rows():
    dirty = _DirtyRows(16)

    dirty.mark(np.array([1, 3, 4]))

    assert dirty.take() == [(1, 2), (3, 5)]


def test_boolean_mask_write_covers_true_positions():
    dirty = _DirtyRows(8)
    mask = np.array([False, True, True, False, False, False, False, True])

    dirty.mark(mask)

    assert dirty.take() == [(1, 3), (7, 8)]


def test_multi_dimensional_boolean_key_refreshes_the_whole_buffer():
    # ``arr[arr > 0] = v`` is a legal write that no row index describes. The
    # data is already written when the tracker sees the key, so it must fall
    # back to a full refresh rather than raise.
    dirty = _DirtyRows(4)

    dirty.mark(np.zeros((4, 2), dtype=bool))

    assert dirty.take() == [(0, 4)]


def test_repeated_marking_never_loses_rows():
    dirty = _DirtyRows(16)

    dirty.mark(slice(0, 10))
    dirty.mark(slice(5, 8))
    dirty.mark(slice(20, 24))

    assert dirty.take() == [(0, 10)]


def test_take_resets_the_tracker():
    dirty = _DirtyRows(4)
    dirty.mark(slice(0, 4))

    assert dirty.take() == [(0, 4)]
    assert dirty.take() == []


def test_mask_is_not_scanned_until_a_scattered_write_marks_it():
    dirty = _DirtyRows(64)

    dirty.mark(5)  # the only write that needs the mask
    assert dirty.take() == [(5, 6)]
    # The allocation is kept, but nothing is pending, so later syncs skip it.
    assert dirty._mask is not None
    assert dirty._mask_pending is False

    dirty.mark(slice(0, 4))
    assert dirty._mask_pending is False
    assert dirty.take() == [(0, 4)]

    dirty.mark(9)
    assert dirty._mask_pending is True
    assert dirty.take() == [(9, 10)]


def test_out_of_bounds_index_is_rejected():
    dirty = _DirtyRows(8)

    with pytest.raises(IndexError):
        dirty.mark(-10)
    with pytest.raises(IndexError):
        dirty.mark(8)

    assert dirty.take() == []


def test_negative_index_within_bounds_is_resolved():
    dirty = _DirtyRows(8)

    dirty.mark(-1)

    assert dirty.take() == [(7, 8)]


def test_empty_tuple_key_marks_the_whole_buffer():
    # ``arr[()] = value`` is NumPy's "everything" index.
    dirty = _DirtyRows(8)

    dirty.mark(())

    assert dirty.take() == [(0, 8)]


def test_negative_step_slice_marks_the_reversed_rows():
    dirty = _DirtyRows(8)
    rows = slice(5, 2, -1)  # rows 5, 4, 3

    dirty.mark(rows)

    assert dirty.take() == [(3, 6)]


def test_empty_negative_step_slice_marks_nothing():
    dirty = _DirtyRows(8)

    dirty.mark(slice(2, 5, -1))

    assert dirty.take() == []


def test_numpy_write_reaches_the_device_shadow():
    buf = _fake_buffer(8)

    buf.np[2:5] = 7
    buf.uva(8)

    assert buf._uva[:2].eq(0).all()
    assert buf._uva[2:5].eq(7).all()
    assert buf._uva[5:].eq(0).all()


def test_scattered_numpy_writes_reach_the_device_shadow():
    buf = _fake_buffer(8)

    buf.np[6] = 3
    buf.np[1] = 5
    buf.uva(8)

    assert buf._uva[1].eq(5).all()
    assert buf._uva[6].eq(3).all()
    assert buf._uva[0].eq(0).all()


def test_multi_dimensional_boolean_write_reaches_the_device_shadow():
    buf = _fake_buffer(4)
    mask = np.array([[True, False], [False, False], [False, True], [False, False]])

    buf.np[mask] = 5
    buf.uva(4)

    assert buf._uva.tolist() == [[5, 0], [0, 0], [0, 5], [0, 0]]


def test_torch_view_write_reaches_the_device_shadow():
    buf = _fake_buffer(8)

    buf.cpu[3] = torch.tensor([1, 2], dtype=torch.int32)
    buf.uva(8)

    assert buf._uva[3].tolist() == [1, 2]
    assert buf._uva[4].tolist() == [0, 0]


def test_uint64_shadow_copies_through_int64_storage():
    buf = _fake_buffer(4, cols=1, dtype=torch.uint64)

    buf.np[0:2] = np.array([[2**63 + 1], [7]], dtype=np.uint64)
    buf.uva(4)

    assert buf._uva[0].item() == 2**63 + 1
    assert buf._uva[1].item() == 7


def test_repeated_sync_only_copies_new_rows():
    buf = _fake_buffer(8)

    buf.np[0:2] = 4
    buf.uva(8)
    buf.np[5] = 9
    buf.uva(8)

    assert buf._uva[0:2].eq(4).all()
    assert buf._uva[5].eq(9).all()
