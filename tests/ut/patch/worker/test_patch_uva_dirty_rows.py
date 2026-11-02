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


def _count_copied_rows(monkeypatch) -> list[int]:
    """Record how many rows each H2D copy moves.

    ``_sync_dirty_rows`` copies one slice per contiguous run, so this is what
    distinguishes "publish what changed" from "publish everything": contents
    alone look the same in both cases.
    """
    rows: list[int] = []
    original = torch.Tensor.copy_

    def _copy(self, src, *args, **kwargs):
        rows.append(int(self.shape[0]))
        return original(self, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", _copy)
    return rows


def _spy_int64_storage(monkeypatch) -> list[torch.dtype]:
    """Record the dtypes handed to ``_int64_storage`` and check its contract.

    The int64 view exists only because the NPU copy kernels reject ``uint64``,
    and a CPU copy accepts either dtype, so the tensor contents cannot catch a
    missing view. Assert the call site instead.
    """
    seen: list[torch.dtype] = []
    original = patch_uva_module._int64_storage

    def _storage(tensor):
        seen.append(tensor.dtype)
        result = original(tensor)
        if tensor.dtype == torch.uint64:
            assert result.dtype == torch.int64
        return result

    monkeypatch.setattr(patch_uva_module, "_int64_storage", _storage)
    return seen


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
    dirty.mark(12)

    # Rows 2-6 and 8-11 were not written; merging the runs would copy them.
    assert dirty.take() == [(0, 2), (7, 8), (12, 13)]


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


def test_repeated_marking_never_loses_rows():
    dirty = _DirtyRows(16)

    dirty.mark(slice(0, 10))
    dirty.mark(slice(5, 8))
    dirty.mark(slice(20, 24))

    assert dirty.take() == [(0, 10)]


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

    dirty.mark(slice(5, 2, -1))  # rows 5, 4, 3

    assert dirty.take() == [(3, 6)]

    # With a negative step, ``stop > start`` is the empty form - the opposite
    # of the forward reading, which would mark rows 2-4 here.
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

    # One run for the whole buffer: the scattered bit is subsumed, not copied
    # as a second range.
    assert dirty.take() == [(0, 8)]
    assert dirty._mask_pending is False

    # take() retired both markers: a later write starts from a clean tracker
    # instead of re-publishing the whole buffer.
    dirty.mark(3)
    assert dirty.take() == [(3, 4)]


def test_appending_indices_extends_the_prefix():
    # The 310P staged-write path marks one row per request and tensor, so
    # consecutive appends must stay in the prefix instead of allocating the
    # mask.
    dirty = _DirtyRows(8)

    dirty.mark(0)
    dirty.mark(1)

    assert dirty._prefix == 2
    assert dirty._mask is None
    assert dirty.take() == [(0, 2)]


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


def test_index_only_key_publishes_the_whole_buffer():
    class _RowIndex:
        def __index__(self) -> int:
            return 3

    buf = _make_uva_buffer(8)

    # NumPy accepts any object with __index__ as a row index; the tracker
    # cannot read it as rows, so it must publish everything.
    buf.np[_RowIndex()] = 5
    buf.uva(8)

    assert buf._uva[3].eq(5).all()
    assert buf._uva[4].eq(0).all()


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
    with pytest.raises(ValueError):
        np.put(buf.np, [8, 9], 6)
    with pytest.raises(ValueError):
        buf.np.flat[0] = 1
    with pytest.raises(ValueError):
        buf.np.fill(3)
    with pytest.raises(ValueError):
        buf.np.__iadd__(5)

    # The seal cannot be undone: NumPy refuses to re-enable writeability,
    # which a plain read-only view of a writable base would allow.
    with pytest.raises(ValueError):
        buf.np[0:2].flags.writeable = True
    with pytest.raises(ValueError):
        np.asarray(buf.np).flags.writeable = True

    # The mirror itself must still be writable through __setitem__.
    buf.np[0:2] = 7
    buf.uva(8)
    assert buf._uva[0:2].eq(7).all()
    assert buf._uva[2:].eq(0).all()


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
        lambda cpu: cpu.view(-1),  # returns a new view over the same storage
    ],
)
def test_torch_alias_writes_reach_the_device_shadow(alias):
    # Both "returns the receiver" and "returns a view" shapes hand out a
    # writable tensor over the mirror's storage, so a write through either must
    # publish the whole buffer.
    buf = _make_uva_buffer(8)

    alias(buf.cpu).fill_(9)
    buf.uva(8)

    assert torch.equal(buf._uva, buf._cpu)


def test_torch_attribute_views_publish_the_whole_buffer():
    # Attribute views are not calls, but they still hand out a writable view of
    # the mirror's storage (``.T`` stands for the ``.mT``/``.H``/``.real``
    # family here).
    buf = _make_uva_buffer(8)

    buf.cpu.T[0] = 9
    buf.uva(8)

    assert torch.equal(buf._uva, buf._cpu)


def test_torch_storage_handle_publishes_the_whole_buffer(monkeypatch):
    # A raw storage handle can write any element, and handing it out cannot be
    # told apart from a plain read, so it publishes everything. ``storage``
    # shares the same branch and the same publish; it is left out because
    # torch deprecates ``TypedStorage``.
    buf = _make_uva_buffer(8)
    rows = _count_copied_rows(monkeypatch)

    buf.cpu.untyped_storage()
    buf.uva(8)

    assert rows == [8]


def test_attribute_assignment_cannot_shadow_the_mirror():
    buf = _make_uva_buffer(8)

    with pytest.raises(AttributeError):
        buf.np.shape = (16, 1)
    with pytest.raises(AttributeError):
        buf.np.flat = 9
    with pytest.raises(AttributeError):
        buf.cpu.shape = (16, 1)

    assert buf.np.shape == (8, 2)
    assert buf.cpu.shape == (8, 2)


def test_mirrors_report_their_length():
    buf = _make_uva_buffer(8)

    assert len(buf.np) == 8
    assert len(buf.cpu) == 8


@pytest.mark.parametrize("write", ["prefix", "scattered"])
def test_uint64_copies_through_int64_storage(write, monkeypatch):
    # The NPU copy kernel rejects DT_UINT64, so every copy goes through an
    # int64 view of both sides. A CPU copy accepts either dtype, so the tensor
    # contents cannot catch a missing view: assert the call site.
    buf = _make_uva_buffer(8, cols=1, dtype=torch.uint64)
    seen = _spy_int64_storage(monkeypatch)

    if write == "prefix":
        buf.np[0:2] = np.array([[2**63 + 1], [7]], dtype=np.uint64)
        runs = 1
    else:
        buf.np[6] = np.array([2**63 + 1], dtype=np.uint64)
        buf.np[1] = np.array([7], dtype=np.uint64)
        runs = 2
    buf.uva(8)

    # One destination and one source per run, both seen as uint64.
    assert seen == [torch.uint64] * (2 * runs)
    assert buf._uva[1].item() == 7
    assert buf._uva[6 if write == "scattered" else 0].item() == 2**63 + 1


def test_repeated_sync_only_copies_new_rows(monkeypatch):
    buf = _make_uva_buffer(8)
    copied = _count_copied_rows(monkeypatch)

    buf.np[0:2] = 4
    buf.uva(8)
    buf.np[5] = 9
    buf.uva(8)

    # The second sync must move the one new row, not repeat the first batch and
    # not fall back to a whole-buffer copy.
    assert copied == [2, 1]
    assert buf._uva[0:2].eq(4).all()
    assert buf._uva[5].eq(9).all()


@pytest.mark.parametrize("fail_on", [1, 2])
def test_failed_copy_leaves_the_remaining_rows_dirty(fail_on):
    # Two disjoint runs: the prefix [0, 3) and the single row 5. Failing on
    # either one must keep the ranges that were not refreshed dirty.
    buf = _make_uva_buffer(8)
    buf.np[0:3] = 4
    buf.np[5] = 9

    shadow = buf._uva
    buf._uva = _FailingDevice(shadow, fail_on=fail_on)
    with pytest.raises(RuntimeError):
        buf.uva(8)

    # The range whose copy failed must be retried instead of being dropped.
    buf._uva = shadow
    buf.uva(8)

    assert buf._uva[0:3].eq(4).all()
    assert buf._uva[5].eq(9).all()


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


def test_empty_index_write_is_a_noop(monkeypatch):
    # ``arr[[]] = v`` is a valid no-op write in NumPy; ``np.asarray([])`` is
    # float64, so the tracker must not try to use it as an index.
    dirty = _DirtyRows(8)

    dirty.mark([])
    # An empty multi-dimensional boolean mask selects no elements either.
    dirty.mark(np.zeros((0, 2), dtype=bool))

    assert dirty.take() == []

    # End to end: an empty write must not refresh the whole buffer.
    buf = _make_uva_buffer(8)
    copied = _count_copied_rows(monkeypatch)

    buf.np[[]] = 1
    buf.uva(8)

    assert copied == []
    assert buf._uva.eq(0).all()
