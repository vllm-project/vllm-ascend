# SPDX-License-Identifier: Apache-2.0
"""Exercise owner-only checkpoint publication into real host memory."""

import ctypes
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def mapping():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram/shared_host.py"
    spec = importlib.util.spec_from_file_location("engram_shared_host_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    buffer = (ctypes.c_ubyte * 24)()
    obj = module.MappedHostBuffer.__new__(module.MappedHostBuffer)
    obj.rows, obj.width, obj.dtype = 3, 2, torch.float32
    obj.row_bytes, obj.owner, obj.closed = 8, True, False
    obj.allocation = module._Allocation()
    obj.allocation.host_ptr = ctypes.addressof(buffer)
    return obj, buffer


def test_publish_writes_only_requested_rows(mapping):
    obj, buffer = mapping
    obj.publish(1, torch.tensor([[1.25, -2.5], [3.0, 4.0]]))
    assert list((ctypes.c_float * 6).from_buffer(buffer)) == [0, 0, 1.25, -2.5, 3, 4]


def test_publish_rejects_unowned_closed_and_invalid_ranges(mapping):
    obj, buffer = mapping
    row = torch.tensor([[1.0, 2.0]])
    obj.owner = False
    with pytest.raises(RuntimeError, match="owner"):
        obj.publish(0, row)
    obj.owner, obj.closed = True, True
    with pytest.raises(RuntimeError, match="owner"):
        obj.publish(0, row)
    obj.closed = False
    with pytest.raises(ValueError, match="outside"):
        obj.publish(3, row)
    with pytest.raises(ValueError, match="dtype"):
        obj.publish(0, row.to(torch.int8))
    assert bytes(buffer) == bytes(24)
