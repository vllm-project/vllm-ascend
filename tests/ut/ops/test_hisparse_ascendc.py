# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Resolver launch planning only; these CPU tests do not execute NPU kernels."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def resolver():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/hisparse_ascendc.py"
    spec = importlib.util.spec_from_file_location("hisparse_resolver_launch_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("hot,union", [(16, 8), (8192, 8192), (12288, 64), (32768, 32768)])
def test_scratch_tracks_capacity_instead_of_model_shape(resolver, hot, union):
    hash_size, nbytes = resolver.scratch_layout(hot, union)
    assert hash_size > union and hash_size & (hash_size - 1) == 0
    expected_words = 2 * hash_size + hot + 2 * ((hot + 31) // 32) + 64
    assert nbytes == (expected_words + 15) // 16 * 64 + 512


@pytest.mark.parametrize("hot,union", [(0, 0), (16, 17), (32769, 1), (16, 0)])
def test_scratch_rejects_invalid_slot_representation(resolver, hot, union):
    with pytest.raises(ValueError, match="int16 LRU"):
        resolver.scratch_layout(hot, union)


def arguments(topk, hot_size, dtype, width, block):
    rows = 2
    args = {
        name: None
        for name in (
            "miss_mask",
            "resolved_global_indices",
            "valid_counts",
            "swap_host_physical_rows",
            "swap_device_physical_rows",
            "swap_counts",
            "resident_block_table",
            "num_valid_rows",
        )
    }
    args.update(
        host_cache=torch.empty(4096, width, dtype=dtype),
        hot_cache=torch.empty(2, block, width, dtype=dtype),
        hot_block_table=torch.zeros(1, (hot_size + block - 1) // block, dtype=torch.int32),
        global_indices=torch.zeros(rows, topk, dtype=torch.int32),
        hot_indices=torch.empty(rows, topk, dtype=torch.int32),
        device_global_indices=torch.full((1, hot_size), -1, dtype=torch.int32),
        lru_slots=torch.arange(hot_size, dtype=torch.int32).to(torch.int16)[None],
        request_state_indices=torch.zeros(1, dtype=torch.int32),
        region_stride=hot_size,
        max_union_rows=min(hot_size, topk * rows),
        stats=torch.zeros(2, dtype=torch.int64),
        attention_indices=torch.empty(rows, topk, dtype=torch.int32),
        attention_block_stride=block * 2,
        request_ids=torch.zeros(rows, dtype=torch.int32),
        source_block_table=torch.ones(1, topk, dtype=torch.int32),
        source_block_size=block,
        resident_block_size=block,
        resident_null_block=0,
    )
    return args


@pytest.mark.parametrize(
    "topk,hot,dtype,width,block",
    [(8, 16, torch.bfloat16, 576, 4), (4096, 12288, torch.float32, 32, 64)],
)
def test_native_dispatch_has_no_model_or_capture_gate(monkeypatch, resolver, topk, hot, dtype, width, block):
    native = SimpleNamespace(scratch_capacity=lambda: 2**20, resolve_tensors=MagicMock())
    monkeypatch.setattr(resolver, "_load_resolver", lambda: native)
    monkeypatch.setattr(torch.ops, "hisparse_fused60103", native)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            is_current_stream_capturing=MagicMock(side_effect=AssertionError("must not branch on capture"))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        torch.Tensor, "record_stream", MagicMock(side_effect=AssertionError("C++ owns stream recording"))
    )
    workspace = SimpleNamespace(
        first_refs=torch.empty(2 * hot, dtype=torch.int32), errors=torch.empty(2, 8, dtype=torch.int32)
    )
    owner = SimpleNamespace(use_fused_index_ops=False, workspace=workspace)
    args = arguments(topk, hot, dtype, width, block)
    assert resolver.try_resolve(owner, args)
    assert resolver.try_resolve(owner, args)
    assert native.resolve_tensors.call_count == 2
    passed = native.resolve_tensors.call_args.args
    assert passed[0] is args["global_indices"]
    assert passed[6] is args["device_global_indices"]
    assert passed[10].data_ptr() == workspace.first_refs.data_ptr()
    assert passed[17].data_ptr() == workspace.errors.data_ptr()
    assert passed[19:] == (4096, block, block, block, 0, block * 2, args["max_union_rows"])
    assert set(vars(owner)) == {"use_fused_index_ops", "workspace"}


def test_resource_shortfall_falls_back_without_binding(monkeypatch, resolver):
    native = SimpleNamespace(scratch_capacity=lambda: 512)
    monkeypatch.setattr(resolver, "_load_resolver", lambda: native)
    owner = SimpleNamespace()
    assert not resolver.try_resolve(owner, arguments(8, 16, torch.bfloat16, 576, 4))
    assert not hasattr(owner, "_ascendc_resolvers")
