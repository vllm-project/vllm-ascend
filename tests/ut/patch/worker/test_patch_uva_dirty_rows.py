# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the UVA host-shadow dirty-row tracking.

Recording dirty rows one row at a time used to make a single ``np[:n] = x``
cost hundreds of milliseconds of host time, and that prefix write runs on every
step. The tracker must stay O(1) for the contiguous-prefix writes and still
copy scattered writes correctly.
"""

from unittest import mock

import numpy as np
import pytest
import torch

from vllm_ascend.patch.worker import patch_uva as patch_uva_module
from vllm_ascend.patch.worker.patch_uva import (
    UvaBufferWrapper,
    _DirtyRows,
)


def _make_uva_buffer(rows: int, cols: int = 2, dtype: torch.dtype = torch.int32) -> UvaBufferWrapper:
    """Build a real wrapper on CPU: the UT has no NPU and no pinned allocator."""
    pinned_zeros = torch.zeros

    def _unpinned_zeros(*args, **kwargs):
        kwargs["pin_memory"] = False
        return pinned_zeros(*args, **kwargs)

    with (
        mock.patch.object(patch_uva_module, "is_uva_available", lambda: False),
        mock.patch.object(patch_uva_module, "_zeros_on_npu", torch.zeros_like),
        mock.patch.object(torch, "zeros", _unpinned_zeros),
    ):
        return UvaBufferWrapper((rows, cols), dtype)


class _FailingDevice:
    """A device shadow whose Nth row-range copy fails."""

    def __init__(self, shadow: torch.Tensor, fail_on: int) -> None:
        self._shadow = shadow
        self._fail_on = fail_on
        self._calls = 0

    def __getitem__(self, key):
        self._calls += 1
        if self._calls == self._fail_on:
            raise RuntimeError("H2D copy failed")
        return self._shadow[key]


def test_prefix_write_is_tracked_without_allocating_a_mask():
    dirty = _DirtyRows(2_097_152)

    dirty.mark(slice(0, 8192))

    assert dirty.take() == [(0, 8192)]
    # The whole point: no per-row set and no mask at all for a prefix write.
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


def test_zero_dimensional_integer_key_marks_the_row():
    # ``arr[np.array(3)] = v`` is a scalar index, not a row list.
    dirty = _DirtyRows(8)

    dirty.mark(np.array(3))

    assert dirty.take() == [(3, 4)]


def test_two_dimensional_integer_key_marks_every_listed_row():
    # ``arr[pairs] = v`` writes elements, but every row it names is dirty.
    dirty = _DirtyRows(8)

    dirty.mark(np.array([[0, 1], [1, 5]]))

    assert dirty.take() == [(0, 2), (5, 6)]


def test_whole_buffer_write_clears_the_pending_mask():
    dirty = _DirtyRows(8)

    dirty.mark(5)
    dirty.mark(slice(None))

    assert dirty.take() == [(0, 8)]
    assert dirty._mask_pending is False


def test_single_row_write_below_the_prefix_allocates_no_mask():
    dirty = _DirtyRows(8)

    dirty.mark(slice(0, 4))
    dirty.mark(2)

    assert dirty._mask is None
    assert dirty.take() == [(0, 4)]


def test_numpy_write_reaches_the_device_shadow():
    buf = _make_uva_buffer(8)

    buf.np[2:5] = 7
    buf.uva(8)

    assert buf._uva[:2].eq(0).all()
    assert buf._uva[2:5].eq(7).all()
    assert buf._uva[5:].eq(0).all()


def test_scattered_numpy_writes_reach_the_device_shadow():
    buf = _make_uva_buffer(8)

    buf.np[6] = 3
    buf.np[1] = 5
    buf.uva(8)

    assert buf._uva[1].eq(5).all()
    assert buf._uva[6].eq(3).all()
    assert buf._uva[0].eq(0).all()


def test_multi_dimensional_boolean_write_reaches_the_device_shadow():
    buf = _make_uva_buffer(4)
    mask = np.array([[True, False], [False, False], [False, True], [False, False]])

    buf.np[mask] = 5
    buf.uva(4)

    assert buf._uva.tolist() == [[5, 0], [0, 0], [0, 5], [0, 0]]


def test_torch_view_write_reaches_the_device_shadow():
    buf = _make_uva_buffer(8)

    buf.cpu[3] = torch.tensor([1, 2], dtype=torch.int32)
    buf.uva(8)

    assert buf._uva[3].tolist() == [1, 2]
    assert buf._uva[4].tolist() == [0, 0]


def test_numpy_view_writes_are_refused_instead_of_going_stale():
    # A write through a view never reaches the callback, so the mirror hands
    # out read-only views and lets the write fail here instead of leaving the
    # device copy stale.
    buf = _make_uva_buffer(8)

    with pytest.raises(ValueError):
        buf.np[0:2].fill(7)
    with pytest.raises(ValueError):
        buf.np[0:2] += 5
    with pytest.raises(ValueError):
        np.asarray(buf.np)[0, 0] = 1
    with pytest.raises(ValueError):
        buf.np.reshape(-1)[0] = 1

    # The mirror itself must still be writable through __setitem__.
    buf.np[0:2] = 7
    buf.uva(8)
    assert buf._uva[0:2].eq(7).all()


def test_numpy_in_place_method_publishes_the_whole_buffer():
    buf = _make_uva_buffer(8)

    # ndarray's in-place methods mutate the receiver and return None, so the
    # tracker cannot tell which rows changed and publishes all of them.
    buf.np.fill(3)
    buf.uva(8)

    assert buf._uva.eq(3).all()


def test_numpy_put_is_observed():
    buf = _make_uva_buffer(8)

    # np.put() dispatches to the mirror's own put(), which is intercepted.
    # Flat indices 8 and 9 are the two columns of row 4.
    np.put(buf.np, [8, 9], 6)
    buf.uva(8)

    assert buf._uva[4].eq(6).all()
    assert buf._uva[5].eq(0).all()


def test_torch_in_place_method_publishes_the_whole_buffer():
    buf = _make_uva_buffer(8)

    buf.cpu.copy_(torch.full((8, 2), 4, dtype=torch.int32))
    buf.uva(8)

    assert buf._uva.eq(4).all()


def test_torch_numpy_bridge_is_read_only():
    buf = _make_uva_buffer(8)

    array = buf.cpu.numpy()

    assert array.flags.writeable is False
    with pytest.raises(ValueError):
        array[0, 0] = 1


def test_torch_data_handle_is_refused():
    buf = _make_uva_buffer(8)

    with pytest.raises(AttributeError, match="dirty tracking"):
        _ = buf.cpu.data


@pytest.mark.parametrize(
    "alias",
    [
        lambda cpu: cpu.to(torch.int32),  # no-op cast: returns the mirror itself
        lambda cpu: cpu.detach(),
        lambda cpu: cpu.view(-1),
        lambda cpu: cpu.reshape(-1),
        lambda cpu: cpu.contiguous(),
        lambda cpu: cpu.t(),
        lambda cpu: cpu.narrow(0, 0, 2),
        lambda cpu: cpu.split(2)[0],
    ],
)
def test_torch_alias_writes_reach_the_device_shadow(alias):
    # Every one of these hands out a tensor over the mirror's storage, so a
    # write through it must publish the whole buffer.
    buf = _make_uva_buffer(8)

    alias(buf.cpu).fill_(9)
    buf.uva(8)

    assert torch.equal(buf._uva, buf._cpu)


def test_numpy_explicit_inplace_dunder_reaches_the_device_shadow():
    # ``buf.np += 5`` fails on the proxy (no __iadd__ on the type), but an
    # explicit __iadd__ call mutates the mirror in place and must publish it.
    buf = _make_uva_buffer(8)

    buf.np.__iadd__(5)
    buf.uva(8)

    assert torch.equal(buf._uva, buf._cpu)


def test_mirrors_report_their_length():
    buf = _make_uva_buffer(8)

    assert len(buf.np) == 8
    assert len(buf.cpu) == 8


def test_uint64_shadow_copies_through_int64_storage():
    buf = _make_uva_buffer(4, cols=1, dtype=torch.uint64)

    buf.np[0:2] = np.array([[2**63 + 1], [7]], dtype=np.uint64)
    buf.uva(4)

    assert buf._uva[0].item() == 2**63 + 1
    assert buf._uva[1].item() == 7


def test_repeated_sync_only_copies_new_rows():
    buf = _make_uva_buffer(8)

    buf.np[0:2] = 4
    buf.uva(8)
    buf.np[5] = 9
    buf.uva(8)

    assert buf._uva[0:2].eq(4).all()
    assert buf._uva[5].eq(9).all()


def test_failed_copy_leaves_the_remaining_rows_dirty():
    buf = _make_uva_buffer(8)
    buf.np[5] = 9
    buf.np[1] = 3

    shadow = buf._uva
    buf._uva = _FailingDevice(shadow, fail_on=2)
    with pytest.raises(RuntimeError):
        buf.uva(8)

    # The range whose copy failed must be retried instead of being dropped.
    buf._uva = shadow
    buf.uva(8)

    assert buf._uva[1].eq(3).all()
    assert buf._uva[5].eq(9).all()


def test_failed_prefix_copy_leaves_the_prefix_dirty():
    buf = _make_uva_buffer(8)
    buf.np[0:3] = 4

    shadow = buf._uva
    buf._uva = _FailingDevice(shadow, fail_on=1)
    with pytest.raises(RuntimeError):
        buf.uva(8)

    buf._uva = shadow
    buf.uva(8)

    assert buf._uva[0:3].eq(4).all()


def test_sync_refreshes_rows_past_the_requested_prefix():
    # ``uva(n)`` slices what the caller sees; it does not clip the sync.
    buf = _make_uva_buffer(8)
    buf.np[0:5] = 9

    buf.uva(2)

    assert buf._uva[4].eq(9).all()


def test_uint64_scattered_write_reaches_the_device_shadow():
    buf = _make_uva_buffer(8, cols=1, dtype=torch.uint64)

    buf.np[6] = np.array([2**63 + 1], dtype=np.uint64)
    buf.np[1] = np.array([7], dtype=np.uint64)
    buf.uva(8)

    assert buf._uva[1].item() == 7
    assert buf._uva[6].item() == 2**63 + 1


def test_empty_index_list_marks_nothing():
    # ``arr[[]] = v`` is a valid no-op write in NumPy; ``np.asarray([])`` is
    # float64, so the tracker must not try to use it as an index.
    dirty = _DirtyRows(8)

    dirty.mark([])
    # An empty multi-dimensional boolean mask selects no elements either.
    dirty.mark(np.zeros((0, 2), dtype=bool))

    assert dirty.take() == []


def test_empty_index_write_is_a_noop():
    buf = _make_uva_buffer(8)

    buf.np[[]] = 1
    buf.uva(8)

    assert buf._uva.eq(0).all()
