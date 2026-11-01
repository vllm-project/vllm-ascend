# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm_ascend.patch.worker import patch_uva


@pytest.fixture
def buffer(monkeypatch):
    buf = patch_uva.UvaBufferWrapper.__new__(patch_uva.UvaBufferWrapper)
    buf._cpu = torch.zeros(16, dtype=torch.int64)
    buf._np = buf._cpu.numpy()
    buf._uva = torch.zeros_like(buf._cpu)
    buf._use_real_uva = False
    buf._modified_indices = set()
    buf._dirty_prefix_end = 0
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)

    def cpu_to(self, *args, **kwargs):
        if kwargs.get("device") == "npu":
            kwargs["device"] = "cpu"
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    return buf


@pytest.mark.parametrize("view", ["np", "cpu"])
def test_prefix_skips_per_element_tracking(buffer, monkeypatch, view):
    def unexpected(*args):
        pytest.fail("prefix writes must not enumerate row indices")

    monkeypatch.setattr(patch_uva, "get_row_indices_from_key", unexpected)
    values = np.arange(8) if view == "np" else torch.arange(8)
    getattr(buffer, view)[:8] = values
    getattr(buffer, view)[:3] = 9
    assert buffer._dirty_prefix_end == 8
    assert not buffer._modified_indices
    torch.testing.assert_close(buffer.uva(), buffer._cpu)
    assert buffer._dirty_prefix_end == 0
    assert not buffer._modified_indices
    # A shorter subsequent use must not copy stale, untouched rows.
    buffer._uva[10] = 99
    getattr(buffer, view)[:2] = 5
    result = buffer.uva(2)
    assert result.tolist() == [5, 5]
    assert buffer._uva[10] == 99


@pytest.mark.parametrize("prefix_first", [True, False])
def test_mixed_sparse_and_prefix_writes(buffer, prefix_first):
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
    assert not buffer._modified_indices
    assert buffer._dirty_prefix_end == 0


def test_empty_failed_and_negative_stop_prefix(buffer):
    buffer.np[:0] = []
    assert buffer._dirty_prefix_end == 0
    with pytest.raises(ValueError):
        buffer.np[:3] = [1, 2]
    assert buffer._dirty_prefix_end == 0
    buffer.np[:-2] = 4
    assert buffer._dirty_prefix_end == 14
    torch.testing.assert_close(buffer.uva(), buffer._cpu)


def test_uint64_prefix_preserves_high_bits(buffer):
    buffer._cpu = torch.zeros(16, dtype=torch.uint64)
    buffer._np = buffer._cpu.numpy()
    buffer._uva = torch.zeros_like(buffer._cpu)
    buffer.np[:2] = np.array([2**63 + 3, 2**64 - 1], dtype=np.uint64)
    assert buffer.uva(2).tolist() == [2**63 + 3, 2**64 - 1]


def test_real_uva_does_not_track(buffer):
    buffer._use_real_uva = True
    buffer._uva = buffer._cpu
    buffer.np[:4] = 3
    assert buffer.np is buffer._np
    assert buffer.cpu is buffer._cpu
    assert buffer.uva(4).tolist() == [3] * 4
    assert not buffer._modified_indices
    assert buffer._dirty_prefix_end == 0


@pytest.mark.parametrize("view", ["np", "cpu"])
@pytest.mark.parametrize("column", [slice(None), 0])
def test_multidimensional_prefix(buffer, monkeypatch, view, column):
    buffer._cpu = torch.zeros((16, 3), dtype=torch.int64)
    buffer._np = buffer._cpu.numpy()
    buffer._uva = torch.zeros_like(buffer._cpu)

    def unexpected(*args):
        pytest.fail("multidimensional prefix must not enumerate rows")

    monkeypatch.setattr(patch_uva, "get_row_indices_from_key", unexpected)
    getattr(buffer, view)[:5, column] = 7
    assert buffer._dirty_prefix_end == 5
    torch.testing.assert_close(buffer.uva(), buffer._cpu)


@pytest.mark.parametrize("prefix", [0, 4])
def test_nonzero_contiguous_range_avoids_indexed_copy(buffer, monkeypatch, prefix):
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
    assert buffer._dirty_prefix_end == 0
    assert not buffer._modified_indices
