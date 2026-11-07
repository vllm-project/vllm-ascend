# SPDX-License-Identifier: Apache-2.0
"""CPU launch contract and output-gate integration; numerical tests use NPU."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


class Kernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, args, kwargs))


def load_gate():
    path = ROOT / "vllm_ascend/ops/triton/flash_attention_output.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign) or getattr(node, "name", "") == "flash_attention_gate"
    ]
    kernel = Kernel()
    scope = dict(
        torch=torch,
        triton=SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y),
        init_device_properties_triton=lambda: None,
        get_vectorcore_num=lambda: 64,
        _flash_attention_gate_kernel=kernel,
    )
    exec(compile(tree, str(path), "exec"), scope)
    return scope["flash_attention_gate"], kernel


@pytest.mark.parametrize("tokens,hidden", [(1, 768), (8, 1536), (32, 3072), (1024, 12288), (17, 129)])
def test_launch_preserves_storage_and_row_strides(tokens, hidden):
    gate_op, kernel = load_gate()
    a = torch.empty(tokens, hidden + 16, dtype=torch.bfloat16)[:, :hidden]
    g = torch.empty(tokens, hidden + 8, dtype=torch.bfloat16)[:, :hidden]
    assert gate_op(a, g) is a
    grid, args, kwargs = kernel.calls[0]
    assert args[0] is a and args[1] is g
    assert args[2] == tokens * hidden
    assert kwargs["PROJECTED_STRIDE"] == hidden + 16
    assert kwargs["GATE_STRIDE"] == hidden + 8
    assert kwargs["HIDDEN"] == hidden
    assert 0 < grid[0] <= 64


@pytest.mark.parametrize("shape", [(0, 1536), (8, 0)])
def test_empty_does_not_launch(shape):
    gate_op, kernel = load_gate()
    a = torch.empty(shape)
    assert gate_op(a, a) is a
    assert not kernel.calls


@pytest.mark.parametrize(
    "a,g,message",
    [
        (torch.empty(128), torch.empty(128), "2-D"),
        (torch.empty(2, 128), torch.empty(1, 128), "Gate shape"),
        (torch.empty(2, 128), torch.empty(2, 128, dtype=torch.bfloat16), "dtypes"),
        (torch.empty(2, 256)[:, ::2], torch.empty(2, 128), "columns"),
        (torch.empty(2, 128), torch.empty(2, 256)[:, ::2], "columns"),
    ],
)
def test_invalid_inputs_fail_before_launch(a, g, message):
    gate_op, kernel = load_gate()
    with pytest.raises(ValueError, match=message):
        gate_op(a, g)
    assert not kernel.calls


@pytest.mark.parametrize("path,owner", [("mla_v1.py", "AscendMLAImpl"), ("sfa_v1.py", "AscendSFAImpl")])
@pytest.mark.parametrize("enabled", [False, True])
def test_production_output_gate_branch(path, owner, enabled):
    source = ROOT / "vllm_ascend/attention" / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    branch = next(
        n
        for n in forward.body
        if isinstance(n, ast.If)
        and any(
            isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "flash_attention_gate"
            for c in ast.walk(n)
        )
    )
    projected, gate = torch.randn(4, 1536), torch.randn(4, 1536)
    original = projected.clone()
    hidden = torch.randn(4, 32)[:, ::2]
    calls = []

    def project(value):
        assert value.is_contiguous()
        calls.append("g_proj")
        return gate, None

    def fused(value, logits):
        calls.append("fused")
        assert value is projected and logits is gate
        value.mul_(torch.sigmoid(logits))

    impl = SimpleNamespace(g_proj=Mock(side_effect=project))
    scope = dict(
        self=impl,
        gate=gate if enabled else None,
        gate_hidden_states=hidden if enabled else None,
        attn_output=projected,
        o_proj_input=projected,
        flash_attention_gate=fused,
    )
    exec(compile(ast.Module(body=[branch], type_ignores=[]), str(source), "exec"), scope)
    expected = original * torch.sigmoid(gate) if enabled else original
    torch.testing.assert_close(projected, expected, atol=0, rtol=0)
    assert calls == ((["g_proj", "fused"] if path == "sfa_v1.py" else ["fused"]) if enabled else [])
    # SFA must keep the gating projection after V-up projection and before O projection.
    if path == "sfa_v1.py":
        vup = [
            n.lineno
            for n in ast.walk(forward)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_v_up_proj"
        ]
        finalize = [
            n.lineno
            for n in ast.walk(forward)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr.startswith("_finalize")
        ]
        assert max(vup) < branch.lineno < min(finalize)
