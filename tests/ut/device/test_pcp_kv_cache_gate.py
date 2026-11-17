# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/device/pcp_kv_cache.py"


def load_copy(monkeypatch, supported):
    utils = ModuleType("vllm_ascend.utils")
    utils.__dict__["supports_triton"] = lambda: supported
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    return runpy.run_path(str(SOURCE))["copy_pcp_kv_cache"]


@pytest.mark.parametrize("packed_dtype", [None, torch.uint8, torch.float8_e4m3fn])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("values", [[], [-1], [0, 7, -1, 3, 0], [-1, -1]])
def test_native_copy_preserves_rows_and_packed_bits_without_triton(monkeypatch, packed_dtype, strided, values):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("vllm_ascend.ops.triton"):
            raise AssertionError(f"Native copy imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    copy = load_copy(monkeypatch, False)
    storage = torch.arange(3 * 4 * 1 * 16).reshape(3, 4, 1, 16)
    key = storage.to(torch.float16 if packed_dtype is None else torch.uint8)
    if packed_dtype is not None:
        key = key.view(packed_dtype)
    if strided:
        key = key[..., ::2]
    cache = (key, key[..., :2]) if packed_dtype is None else (key,)
    cache_before = [part.contiguous().view(torch.uint8).clone() for part in cache]
    slots = torch.tensor(values, dtype=torch.int32)
    output = copy(cache, slots)
    parts = cache if packed_dtype is None else (key.view(torch.int8),)
    expected_rows = []
    for slot in values:
        if slot < 0:
            expected_rows.append(torch.zeros(sum(part.shape[-1] for part in parts), dtype=parts[0].dtype))
        else:
            expected_rows.append(torch.cat([part[slot // 4, slot % 4, 0] for part in parts]))
    expected = (
        torch.stack(expected_rows)
        if expected_rows
        else torch.empty(0, sum(part.shape[-1] for part in parts), dtype=parts[0].dtype)
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.is_contiguous()
    assert all(torch.equal(part.contiguous().view(torch.uint8), saved) for part, saved in zip(cache, cache_before))


def test_supported_copy_invokes_existing_triton_implementation(monkeypatch):
    copy = load_copy(monkeypatch, True)
    kernel = ModuleType("vllm_ascend.ops.triton.pcp_kv_cache")
    dispatch = Mock(return_value=object())
    kernel.__dict__["copy_pcp_kv_cache"] = dispatch
    monkeypatch.setitem(sys.modules, kernel.__name__, kernel)
    cache, slots = object(), object()
    assert copy(cache, slots) is dispatch.return_value
    dispatch.assert_called_once_with(cache, slots)
