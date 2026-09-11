# SPDX-License-Identifier: Apache-2.0
"""Exercise exchange routing and invalid-rank semantics without an NPU."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_function(path, name, namespace):
    source = ROOT / path
    tree = ast.parse(source.read_text())
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + [method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "tokens,ranks,scatter,supported",
    [
        (1, 8, 1, True),
        (6, 8, 1, True),
        (12, 8, 1, True),
        (13, 8, 1, False),
        (6, 2, 1, True),
        (8, 8, 0, True),
    ],
)
def test_registered_op_keeps_upstream_fallback_and_output_dtype(tokens, ranks, scatter, supported):
    calls = []
    group = object()
    output = torch.zeros(tokens, 64, 512, dtype=torch.bfloat16)
    lse = torch.zeros(tokens, 64, 1)
    shape = (tokens, 64 // ranks, 512) if scatter else (tokens // ranks, 64, 512)

    def exchange(out, stats, selected_group):
        assert selected_group is group and out is output and stats is lse
        calls.append("exchange")
        return torch.ones(shape, dtype=torch.float32)

    def transfer(recv, send, *, group):
        calls.append("upstream_collective")

    namespace = {
        "torch": torch,
        "dist": SimpleNamespace(all_to_all_single=transfer),
        "can_exchange": lambda *_: supported,
        "exchange": exchange,
        "pack_sfa_dcp_output_lse": lambda *_: torch.empty(8, 16),
        "fused_sfa_dcp_lse_combine": lambda *_: torch.ones(shape, dtype=output.dtype),
    }
    run = load_function("vllm_ascend/ops/triton/sfa_cp.py", "sfa_dcp_a2a_fused_combine", namespace)
    result = run(output, lse, ranks, scatter, group)
    assert result.dtype == output.dtype and result.shape == shape
    assert calls == (["exchange"] if ranks == 8 and scatter == 1 and supported else ["upstream_collective"])


@pytest.mark.parametrize("tokens,expected", [(0, False), (1, True), (6, True), (12, True), (13, False)])
def test_exchange_gate(tokens, expected):
    namespace = {"torch": torch, "triton": object(), "_MAX_DECODE_TOKENS": 12, "_HEADS": 64, "_HEAD_DIM": 512}
    run = load_function("vllm_ascend/ops/triton/sfa_dcp_exchange.py", "can_exchange", namespace)
    device = SimpleNamespace(type="npu")
    output = SimpleNamespace(device=device, dtype=torch.bfloat16, ndim=3, shape=(tokens, 64, 512))
    lse = SimpleNamespace(device=device, dtype=torch.float32, shape=(tokens, 64, 1))
    assert run(output, lse) is expected


@pytest.mark.parametrize("token_dim", [1, 2])
def test_merge_masks_invalid_rank_outputs_before_weighting(token_dim):
    spec = importlib.util.spec_from_file_location("merge_under_test", ROOT / "vllm_ascend/ops/triton/sfa_dcp_merge.py")
    merge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge)
    parts = torch.full((8, 2, 3, 4), float("nan"))
    lse = torch.full((8, 2, 3), float("-inf"))
    lse[0] = float("nan")
    lse[1] = float("inf")
    parts[2] = 2.0
    lse[2] = 0.0
    parts[3] = 4.0
    lse[3] = 0.0
    # One position has no valid ranks; even NaN payloads must yield zero.
    lse[:, 0, 0] = float("-inf")
    expected = torch.full((2, 3, 4), 3.0)
    expected[0, 0] = 0.0
    expected = expected.movedim(token_dim - 1, 0).contiguous()
    assert torch.equal(merge.fused_merge(parts, lse, token_dim), expected)


def test_custom_op_fake_retains_shape_dtype_and_does_not_collect():
    run = load_function("vllm_ascend/ops/triton/sfa_cp.py", "sfa_dcp_a2a_fused_fake", {"torch": torch})
    output = torch.empty(6, 64, 512, dtype=torch.bfloat16)
    result = run(output, torch.empty(6, 64, 1), 8, 1, "unused")
    assert result.shape == (6, 8, 512) and result.dtype == output.dtype
