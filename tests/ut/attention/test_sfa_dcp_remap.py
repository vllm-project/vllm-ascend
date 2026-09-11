# SPDX-License-Identifier: Apache-2.0
"""Fallback/routing checks without constructing a serving model."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("dcp_remap_under_test", ROOT / "vllm_ascend/ops/triton/sfa_dcp_remap.py")
remap = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = remap
spec.loader.exec_module(remap)


def method():
    source = ROOT / "vllm_ascend/attention/context_parallel/sfa_cp.py"
    tree = ast.parse(source.read_text())
    function = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_remap_sparse_indices"
    )
    namespace = {
        "torch": torch,
        "HAS_TRITON": False,
        "can_fuse_remap": remap.can_fuse_remap,
        "fused_remap": remap.fused_remap,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_remap_sparse_indices"], namespace


def instance(count=8):
    return SimpleNamespace(
        dcp_size=8,
        dcp_rank=0,
        enable_dsa_cp=False,
        _dcp_interleave_size=128,
        _dcp_index_topk=count,
        _remap_order=torch.arange(count, dtype=torch.float32),
        _remap_invalid_index=torch.tensor(-1.0),
    )


@pytest.mark.parametrize("decode_only", [False, True])
def test_cpu_fallback_preserves_order_duplicates_and_invalid_suffix(decode_only):
    run, _ = method()
    indices = torch.tensor([[0, 128, 1024, 1, 1025, -1, 2, 0]], dtype=torch.int32)
    expected = torch.tensor([[0, 128, 1, 129, 2, 0, -1, -1]], dtype=torch.int32)
    assert torch.equal(run(instance(), indices, decode_only=decode_only), expected)


def test_prefill_never_enters_fusion_even_if_layout_qualifies():
    run, namespace = method()
    namespace["can_fuse_remap"] = lambda _: True

    def forbidden(*_):
        raise AssertionError("prefill entered decode fusion")

    namespace["fused_remap"] = forbidden
    indices = torch.tensor([[0, 128, 1024, 1, 1025, -1, 2, 0]], dtype=torch.int32)
    assert run(instance(), indices, decode_only=False).shape == indices.shape


@pytest.mark.parametrize("attribute,value", [("enable_dsa_cp", True), ("dcp_size", 2), ("_dcp_interleave_size", 1)])
def test_other_ownership_contracts_never_enter_fusion(attribute, value):
    run, namespace = method()
    namespace["can_fuse_remap"] = lambda _: True

    def forbidden(*_):
        raise AssertionError("unsupported ownership entered decode fusion")

    namespace["fused_remap"] = forbidden
    state = instance()
    setattr(state, attribute, value)
    indices = torch.tensor([[0, 128, 1024, 1, 1025, -1, 2, 0]], dtype=torch.int32)
    assert run(state, indices, decode_only=True).shape == indices.shape


def test_empty_fallback_and_configured_topk_limit():
    run, _ = method()
    empty = torch.empty((0, 1, 2048), dtype=torch.int32)
    assert not remap.can_fuse_remap(empty)
    assert run(instance(2048), empty, decode_only=True).shape == empty.shape
    with pytest.raises(RuntimeError, match="exceeds configured index_topk"):
        run(instance(8), torch.zeros((1, 9), dtype=torch.int32), decode_only=True)


def test_single_rank_keeps_original_tensor():
    run, _ = method()
    state = instance()
    state.dcp_size = 1
    indices = torch.tensor([[1, -1, 2]], dtype=torch.int32)
    assert run(state, indices, decode_only=True) is indices


def test_decode_and_prefill_both_select_two_kernel_remap(monkeypatch):
    run, namespace = method()
    calls = []
    indices = SimpleNamespace(shape=(1, 2048), numel=lambda: 2048, is_npu=True)
    namespace["HAS_TRITON"] = True
    namespace["can_fuse_remap"] = lambda _: True
    namespace["fused_remap"] = lambda *args: calls.append(("donor", args))
    upstream = SimpleNamespace(remap_sparse_indices_triton=lambda *args: calls.append(("upstream", args)))
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.triton.sparse_index_remap", upstream)
    state = instance(2048)
    run(state, indices, decode_only=True)
    assert calls == [("upstream", (indices, 8, 0, 128))]
    calls.clear()
    run(state, indices, decode_only=False)
    assert calls == [("upstream", (indices, 8, 0, 128))]
