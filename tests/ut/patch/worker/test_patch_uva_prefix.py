# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_ascend.patch.worker.patch_v2 import patch_uva


@pytest.mark.parametrize("key", [slice(None), slice(0, 131155), (slice(None), slice(1, 3))])
def test_prefix_tracking_does_not_enumerate_rows(monkeypatch, key):
    array = np.zeros((131155, 4), dtype=np.int32)
    per_row = Mock()
    prefix = Mock()
    enumerate_rows = Mock(side_effect=AssertionError("prefix writes must not enumerate rows"))
    monkeypatch.setattr(patch_uva, "get_row_indices_from_key", enumerate_rows)

    monitored = patch_uva.MonitoredNumPyArray(array, per_row, prefix)
    monitored[key] = 7

    prefix.assert_called_once_with(131155)
    per_row.assert_not_called()
    enumerate_rows.assert_not_called()
    np.testing.assert_array_equal(array[key], 7)


@pytest.mark.parametrize("key", [slice(2, 7), slice(None, None, 2), -1, [1, 5]])
def test_nonprefix_writes_keep_row_tracking(key):
    array = np.zeros(9, dtype=np.int32)
    per_row = Mock()
    prefix = Mock()
    monitored = patch_uva.MonitoredNumPyArray(array, per_row, prefix)

    monitored[key] = 3

    assert {call.args[0] for call in per_row.call_args_list} == patch_uva.get_row_indices_from_key(key, 9)
    prefix.assert_not_called()


def test_legacy_callback_and_failed_assignment():
    array = np.zeros(8, dtype=np.int32)
    per_row = Mock()
    monitored = patch_uva.MonitoredNumPyArray(array, per_row)
    monitored[:3] = 1
    assert [call.args[0] for call in per_row.call_args_list] == [0, 1, 2]

    prefix = Mock()
    monitored = patch_uva.MonitoredNumPyArray(array, per_row, prefix)
    per_row.reset_mock()
    with pytest.raises(ValueError):
        monitored[:3] = np.ones(4)
    per_row.assert_not_called()
    prefix.assert_not_called()


def make_cpu_buffer(shape, dtype):
    """Exercise real dirty-copy code using CPU tensors instead of allocating NPU memory."""
    buffer = patch_uva.UvaBufferWrapper.__new__(patch_uva.UvaBufferWrapper)
    buffer._cpu = torch.zeros(shape, dtype=dtype)
    buffer._np = buffer._cpu.numpy()
    buffer._uva = torch.zeros_like(buffer._cpu)
    buffer._use_real_uva = False
    buffer._modified_indices = set()
    buffer._modified_prefix = 0
    return buffer


@pytest.mark.parametrize("shape", [(31,), (31, 4)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.uint64])
def test_prefix_copy_accumulates_and_resets(shape, dtype):
    buffer = make_cpu_buffer(shape, dtype)
    buffer.np[:19] = 3
    buffer.np[:7] = 8
    buffer.cpu[2:9] = 9
    assert buffer._modified_prefix == 19

    result = buffer.uva(10)
    assert result.shape[0] == 10
    assert buffer._uva.tolist() == buffer._cpu.tolist()
    assert buffer._modified_prefix == 0
    assert not buffer._modified_indices

    buffer.np[:3] = 11
    buffer.uva()
    assert buffer._uva.tolist() == buffer._cpu.tolist()
    buffer.np[:0] = 7
    buffer.np[:-100] = 7
    assert buffer._modified_prefix == 0
    assert not buffer._modified_indices


@pytest.mark.parametrize("dtype", [torch.int32, torch.uint64])
def test_prefix_and_sparse_writes_preserve_all_values(monkeypatch, dtype):
    buffer = make_cpu_buffer((31, 4), dtype)
    original_to = torch.Tensor.to

    def to_cpu(tensor, *args, **kwargs):
        if kwargs.get("device") == "npu":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    monkeypatch.setattr(torch.Tensor, "to", to_cpu)
    buffer.np[:11, 1:3] = 6
    buffer.np[25] = 17
    buffer.cpu[4] = 13
    if dtype == torch.uint64:
        buffer.np[25, 0] = 2**63 + 3
    buffer.uva()

    assert buffer._uva.tolist() == buffer._cpu.tolist()
    assert not buffer._modified_indices
    assert buffer._modified_prefix == 0


def test_real_uva_keeps_direct_views():
    buffer = make_cpu_buffer((8,), torch.int32)
    buffer._use_real_uva = True
    buffer._uva = buffer._cpu
    assert buffer.cpu is buffer._cpu
    assert buffer.np is buffer._np
    buffer.np[:4] = 3
    assert buffer.uva().tolist() == buffer._cpu.tolist()
    assert buffer._modified_prefix == 0
    assert not buffer._modified_indices
