# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def prepare(monkeypatch):
    # No kernel emulation: these tests exercise only allocation and ABI checks.
    triton_utils = types.ModuleType("vllm.triton_utils")
    triton_utils.tl = types.SimpleNamespace(constexpr=object)
    triton_utils.triton = types.SimpleNamespace(jit=lambda fn: fn)
    monkeypatch.setitem(sys.modules, "vllm.triton_utils", triton_utils)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/qwen4_exp/lightning_prepare.py"
    spec = importlib.util.spec_from_file_location("lightning_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_qsa_lightning_inputs


@pytest.mark.parametrize("heads, padded", [(4, 8), (8, 8), (17, 24), (64, 64)])
def test_empty_lightning_inputs_preserve_abi(prepare, heads, padded):
    query = torch.empty(0, heads, 128, dtype=torch.bfloat16)
    metadata = torch.empty(0, dtype=torch.int64)
    outputs = prepare(query, metadata, metadata, metadata, torch.empty(0, 7, dtype=torch.int32))
    assert [tuple(t.shape) for t in outputs] == [(0, padded, 128), (0, padded), (0,), (0,), (0, 7)]
    assert [t.dtype for t in outputs] == [torch.bfloat16, torch.bfloat16, torch.int32, torch.int32, torch.int32]
    if heads == padded:
        assert outputs[0] is query


@pytest.mark.parametrize(
    "case",
    ["zero_heads", "too_many_heads", "head_width", "query_dtype", "rows", "float_metadata", "empty_table", "requests"],
)
def test_lightning_rejects_invalid_abi_without_launch(prepare, case):
    heads = 0 if case == "zero_heads" else 65 if case == "too_many_heads" else 4
    query = torch.empty(
        2, heads, 64 if case == "head_width" else 128, dtype=torch.float32 if case == "query_dtype" else torch.bfloat16
    )
    positions = torch.empty(
        1 if case == "rows" else 2, dtype=torch.float32 if case == "float_metadata" else torch.int64
    )
    lengths = torch.empty(3 if case == "requests" else 2, dtype=torch.int32)
    owners = torch.empty(2, dtype=torch.int32)
    table = torch.empty(2, 0 if case == "empty_table" else 7, dtype=torch.int32)
    with pytest.raises(ValueError):
        prepare(query, positions, lengths, owners, table)
