# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bulk ndarray writes must commit emulated UVA without per-token dirty sets."""

import array
import ast
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import ArrayLike


@pytest.fixture
def wrapper_class():
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_uva.py"
    names = {
        "get_row_indices_from_key",
        "MonitoredNumPyArray",
        "MonitoredTorchTensor",
        "_int64_storage",
        "UvaBufferWrapper",
    }
    selected = [
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    namespace = dict(torch=torch, np=np, Callable=Callable, Sequence=Sequence, ArrayLike=ArrayLike)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(source), "exec"), namespace)
    return namespace["UvaBufferWrapper"]


def make_buffer(cls, size, dtype, real=False):
    result = cls.__new__(cls)
    result._cpu = torch.zeros(size, dtype=dtype)
    result._np = result._cpu.numpy()
    result._uva = result._cpu if real else torch.zeros(size, dtype=dtype)
    result._use_real_uva = real
    result._modified_indices = set()
    result._modified_prefix_end = 0
    return result


@pytest.mark.parametrize("real", [False, True])
def test_bulk_chunks_preserve_current_values_and_alias(wrapper_class, real):
    buf = make_buffer(wrapper_class, 32, torch.int32, real)
    for delta in (0, 11, 3):
        chunks = [np.arange(8, dtype=np.int64)[::2] + delta, array.array("i", [91 + delta, 92 + delta])]
        actual = buf.copy_chunks_to_uva(chunks, 6)
        assert actual.data_ptr() == buf._uva.data_ptr()
        assert actual.tolist() == np.concatenate(chunks).tolist()
        assert buf._modified_prefix_end == 0 and not buf._modified_indices


def test_bulk_prefix_and_sparse_modifications_both_reach_device(wrapper_class, monkeypatch):
    # Keep this sparse-copy contract portable; the separate NPU run exercises
    # real pinned storage and asynchronous device copies.
    original_to = torch.Tensor.to

    def cpu_to(self, *args, **kwargs):
        if kwargs.get("device") == "npu":
            kwargs["device"] = "cpu"
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    buf = make_buffer(wrapper_class, 32, torch.int32)
    buf.np[20] = 123
    assert buf.copy_chunks_to_uva([np.arange(7)], 7).tolist() == list(range(7))
    assert buf.uva()[20] == 123
    # A later device-side write is authoritative until another CPU modification.
    buf._uva[:3] = 456
    assert buf.uva(3).tolist() == [456] * 3


def test_bulk_tracking_is_constant_size_for_long_prompt(wrapper_class):
    buf = make_buffer(wrapper_class, 65536, torch.int32)
    original = buf._sync_dirty_rows
    pending = []

    def synchronize():
        pending.append((buf._modified_prefix_end, len(buf._modified_indices)))
        original()

    buf._sync_dirty_rows = synchronize
    result = buf.copy_chunks_to_uva([np.arange(65536, dtype=np.int32)], 65536)
    assert pending == [(65536, 0)]
    assert result[-1] == 65535


def test_opaque_uint64_values_keep_all_bits(wrapper_class):
    buf = make_buffer(wrapper_class, 32, torch.uint64)
    expected = np.arange(17, dtype=np.uint64) + np.uint64(2**63 + 19)
    result = buf.copy_chunks_to_uva([expected], len(expected))
    assert np.array_equal(result.numpy(), expected)
