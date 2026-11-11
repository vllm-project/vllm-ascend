# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the production selector with real Dynamo and CPU-only operators.

Run with --confcutdir=tests/ut/models. This does not validate NPU kernels.
"""

import ast
import importlib.util
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "kimi_compile_prepared_input", ROOT / "vllm_ascend/quantization/prepared_linear_input.py"
)
assert spec is not None and spec.loader is not None
prepared_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepared_module)
PreparedLinearInput = prepared_module.PreparedLinearInput


class CUDAGraphMode(Enum):
    NONE = 0


def unpack(x):
    if isinstance(x, PreparedLinearInput):
        return x.quantized
    return x[0] if isinstance(x, tuple) else x


class MXFP8Method:
    def apply_single_group_gmm(self, layer, x):
        return unpack(x) * 3

    def apply(self, layer, x, bias=None):
        output = unpack(x) * 2
        return output.unsqueeze(0) if layer._mxfp8_gmm_nz_layout else output


@pytest.fixture
def selector():
    tree = ast.parse((ROOT / "vllm_ascend/models/kimi_k3/kimi_k3.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendKimiK3RoutedLinear")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "apply_projection")
    method.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    scope = dict(
        torch=torch,
        PreparedLinearInput=PreparedLinearInput,
        AscendW8A8MXFP8DynamicLinearMethod=MXFP8Method,
        CUDAGraphMode=CUDAGraphMode,
        is_forward_context_available=lambda: True,
        get_forward_context=lambda: context,
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), "kimi_k3.py", "exec"),
        scope,
    )
    return scope["apply_projection"]


@pytest.mark.parametrize("input_kind", ["tensor", "tuple", "prepared"])
@pytest.mark.parametrize("nz_layout", [False, True])
def test_fullgraph_compile_never_queries_npu_capture(selector, monkeypatch, input_kind, nz_layout):
    def forbidden_capture_query():
        raise AssertionError("The NPU capture query must not enter the compiled graph")

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(is_current_stream_capturing=forbidden_capture_query, is_available=lambda: False),
        raising=False,
    )
    layer = SimpleNamespace(quant_method=MXFP8Method(), _mxfp8_gmm_nz_layout=nz_layout)

    def forward(x, scale):
        if input_kind == "tuple":
            x = (x, scale)
        elif input_kind == "prepared":
            x = PreparedLinearInput(x, scale, x.dtype)
        return selector(layer, x)

    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    for value in (1.0, 2.0):
        x = torch.full((4, 128), value)
        result = compiled(x, torch.ones(4, 2, 2))
        torch.testing.assert_close(result, x * 2, rtol=0, atol=0)
        assert result.shape == x.shape


@pytest.mark.parametrize("capturing", [False, True])
def test_python_eager_uses_actual_capture_state(selector, monkeypatch, capturing):
    calls = []

    def capture_query():
        calls.append(capturing)
        return capturing

    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=capture_query), raising=False)
    x = torch.ones(4, 128)
    layer = SimpleNamespace(quant_method=MXFP8Method(), _mxfp8_gmm_nz_layout=False)
    result = selector(layer, x)
    torch.testing.assert_close(result, x * (2 if capturing else 3), rtol=0, atol=0)
    assert calls == [capturing]
