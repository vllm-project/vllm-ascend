# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prefix-write cases for the emulated UVA host shadow.

These cases came in with upstream #20541, which first tracked a contiguous
prefix with a single bound instead of one index per row. ``_DirtyRows`` in this
tree generalises that: the prefix stays O(1), scattered writes fall back to a
NumPy mask, and the sync copies one run per contiguous range. The cases are
kept here so the behaviour #20541 pinned stays pinned.
"""

import sys
import types
from unittest import mock

import numpy as np
import pytest
import torch

from vllm_ascend.patch.worker import patch_uva


def _make_buffer(
    shape: int | tuple[int, ...],
    dtype: torch.dtype = torch.int64,
    *,
    real_uva: bool = False,
) -> patch_uva.UvaBufferWrapper:
    """Build a real wrapper on CPU: the UT has no NPU and no pinned allocator."""
    pinned_zeros = torch.zeros

    def _unpinned_zeros(*args, **kwargs):
        kwargs["pin_memory"] = False
        return pinned_zeros(*args, **kwargs)

    with (
        mock.patch.object(patch_uva, "is_uva_available", lambda: real_uva),
        mock.patch.object(patch_uva, "_zeros_on_npu", torch.zeros_like),
        mock.patch.object(torch, "zeros", _unpinned_zeros),
    ):
        return patch_uva.UvaBufferWrapper(shape, dtype)


def _assert_idle(buffer: patch_uva.UvaBufferWrapper) -> None:
    """Nothing is left pending, so the next sync copies no row."""
    assert buffer._dirty._prefix == 0
    assert not buffer._dirty._aliased
    mask = buffer._dirty._mask
    assert mask is None or not mask.any()


@pytest.mark.parametrize("view", ["np", "cpu"])
def test_prefix_skips_per_element_tracking(view):
    buffer = _make_buffer(16)
    values = np.arange(8) if view == "np" else torch.arange(8)
    getattr(buffer, view)[:8] = values
    getattr(buffer, view)[:3] = 9
    # One bound and no mask: the prefix write allocates nothing per row.
    assert buffer._dirty._prefix == 8
    assert buffer._dirty._mask is None
    torch.testing.assert_close(buffer.uva(), buffer._cpu)
    _assert_idle(buffer)
    # A shorter subsequent use must not copy stale, untouched rows.
    buffer._uva[10] = 99
    getattr(buffer, view)[:2] = 5
    result = buffer.uva(2)
    assert result.tolist() == [5, 5]
    assert buffer._uva[10] == 99


@pytest.mark.parametrize("prefix_first", [True, False])
def test_mixed_sparse_and_prefix_writes(prefix_first):
    buffer = _make_buffer(16)

    def sparse():
        buffer.np[2] = 20
        buffer.cpu[12] = 120
        buffer.np[9:15:2] = 30

    if prefix_first:
        buffer.np[:6] = 7
        sparse()
    else:
        sparse()
        buffer.np[:6] = 7
    torch.testing.assert_close(buffer.uva(), buffer._cpu)
    _assert_idle(buffer)


def test_empty_failed_and_negative_stop_prefix():
    buffer = _make_buffer(16)
    buffer.np[:0] = []
    assert buffer._dirty._prefix == 0
    with pytest.raises(ValueError):
        buffer.np[:3] = [1, 2]
    assert buffer._dirty._prefix == 0
    buffer.np[:-2] = 4
    assert buffer._dirty._prefix == 14
    torch.testing.assert_close(buffer.uva(), buffer._cpu)


def test_uint64_prefix_preserves_high_bits():
    buffer = _make_buffer(16, dtype=torch.uint64)
    buffer.np[:2] = np.array([2**63 + 3, 2**64 - 1], dtype=np.uint64)
    assert buffer.uva(2).tolist() == [2**63 + 3, 2**64 - 1]


def test_real_uva_does_not_track(monkeypatch):
    # With ``pinned_mem_register`` the pinned host buffer is mapped for the
    # device, so ``uva()`` hands back that mapping and there is nothing to
    # track: the mirrors are the raw objects.
    monkeypatch.setitem(sys.modules, "vllm_ascend.vllm_ascend_C", types.ModuleType("vllm_ascend.vllm_ascend_C"))
    monkeypatch.setattr(torch.ops._C_ascend, "get_npu_view_from_cpu_tensor", lambda cpu: cpu, raising=False)

    buffer = _make_buffer(16, real_uva=True)

    assert buffer._use_real_uva is True
    assert buffer.np is buffer._np
    assert buffer.cpu is buffer._cpu

    buffer.np[:4] = 3
    assert buffer.uva(4).tolist() == [3] * 4
    _assert_idle(buffer)


@pytest.mark.parametrize("view", ["np", "cpu"])
@pytest.mark.parametrize("column", [slice(None), 0])
def test_multidimensional_prefix(view, column):
    buffer = _make_buffer((16, 3))
    getattr(buffer, view)[:5, column] = 7
    # Only the first axis picks rows, so a prefix over it stays a prefix.
    assert buffer._dirty._prefix == 5
    assert buffer._dirty._mask is None
    torch.testing.assert_close(buffer.uva(), buffer._cpu)


@pytest.mark.parametrize("prefix", [0, 4])
def test_nonzero_contiguous_range_avoids_indexed_copy(prefix, monkeypatch):
    buffer = _make_buffer(16)

    def unexpected(*args):
        pytest.fail("contiguous rows must not use indexed copy or re-pin")

    monkeypatch.setattr(torch.Tensor, "pin_memory", unexpected)
    buffer._uva.fill_(-1)
    if prefix:
        buffer.np[:prefix] = 3
    buffer.np[8:12] = 9
    actual = buffer.uva()
    assert actual[8:12].tolist() == [9] * 4
    assert actual[4:8].tolist() == [-1] * 4
    if prefix:
        assert actual[:prefix].tolist() == [3] * prefix
    _assert_idle(buffer)
