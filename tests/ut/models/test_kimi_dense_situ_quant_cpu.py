# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU routing tests for the production MLP wrapper and linear quantization.

The upstream MLP/SiTU contract is represented by CPU fixtures; NPU operations
are mocked. These tests check activation/quantization boundaries, not device math.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_ascend.quantization.quant_type import QuantType

ROOT = Path(__file__).resolve().parents[3]


def load_nodes(path, names, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), path, "exec"), scope)


class Situ(nn.Module):
    def __init__(self, beta, linear_beta, calls):
        super().__init__()
        self.beta, self.linear_beta = beta, linear_beta
        self.calls = calls
        self.output = None

    def forward(self, x):
        self.calls.append("activation")
        gate, up = x.float().chunk(2, dim=-1)
        gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        self.output = (gate * up).to(x.dtype)
        return self.output


class MLPBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        config = kwargs["quant_config"]
        self.act_fn = Situ(kwargs["activation_situ_beta"], kwargs["activation_situ_linear_beta"], config.calls)
        self.reduce_results = kwargs["reduce_results"]
        self.down_proj = config.down_proj
        self.gate_up_proj = config.gate_up_proj

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        return self.down_proj(self.act_fn(gate_up))[0]


@pytest.fixture
def runtime(monkeypatch):
    calls = []
    scope = dict(__name__=__name__, torch=torch, nn=nn, math=math, KimiMLP=MLPBase)
    exec((ROOT / "vllm_ascend/quantization/prepared_linear_input.py").read_text(encoding="utf-8"), scope)
    scope.update(
        AscendLinearScheme=object,
        QuantType=QuantType,
        WeightSwitchGatherSpec=lambda *a, **kw: None,
        register_scheme=lambda *a: lambda cls: cls,
    )

    def quantize(x, *, dst_type, **kwargs):
        calls.append("quantize")
        return x.to(dst_type), torch.ones(x.shape[0])

    def matmul(q, *args, output_dtype, **kwargs):
        calls.append("matmul")
        return q.to(output_dtype)

    npu = SimpleNamespace(
        float8_e8m0fnu=object(),
        npu_dynamic_quant=Mock(side_effect=quantize),
        npu_dynamic_mx_quant=Mock(side_effect=quantize),
        npu_quant_matmul=Mock(side_effect=matmul),
    )
    scope["torch_npu"] = npu
    for file, cls in [
        ("w8a8_dynamic", "AscendW8A8DynamicLinearMethod"),
        ("w8a8_mxfp8", "AscendW8A8MXFP8DynamicLinearMethod"),
    ]:
        load_nodes(f"vllm_ascend/quantization/methods/w8a8/{file}.py", [cls], scope)
    kernels = {}
    for name in ("situ_mx_quant", "dequant_situ_quant"):
        kernels[name] = Mock(side_effect=AssertionError("dense MLP must retain the upstream SiTU path"))
        monkeypatch.setattr(torch.ops._C_ascend, name, kernels[name], raising=False)
    load_nodes("vllm_ascend/models/kimi_k3/kimi_k3.py", ["AscendKimiMLP"], scope)
    return SimpleNamespace(scope=scope, npu=npu, kernels=kernels, calls=calls)


def make_mlp(runtime, scheme_kind, sequence_parallel, *, linear_beta=25.0, reduce_results=True):
    cls = runtime.scope[
        "AscendW8A8DynamicLinearMethod" if scheme_kind == "int8" else "AscendW8A8MXFP8DynamicLinearMethod"
    ]
    scheme = cls.__new__(cls)
    scheme.group_size, scheme.dynamic_mx_quant_scale_alg = 32, 0
    layer = SimpleNamespace(weight=object(), weight_scale=object())

    def gate_up(x):
        runtime.calls.append("gate_up")
        return x, None

    model = runtime.scope["AscendKimiMLP"](
        128,
        33792,
        "situ",
        reduce_results=reduce_results,
        use_sequence_parallel=sequence_parallel,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=linear_beta,
        quant_config=SimpleNamespace(
            calls=runtime.calls,
            gate_up_proj=Mock(side_effect=gate_up),
            down_proj=Mock(side_effect=lambda x: (scheme.apply(layer, x), None)),
        ),
    )
    return model


@pytest.mark.parametrize("scheme_kind", ["int8", "mxfp8"])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("linear_beta", [None, 25.0])
@pytest.mark.parametrize("input_width", [128, 33792])
def test_dense_situ_quantizes_after_activation_once(runtime, scheme_kind, sequence_parallel, linear_beta, input_width):
    # 33792 is the gate/up width of the existing K3 TP2 smoke model. The
    # whole-row dequant_situ_quant kernel cannot fit this input in A3 UB.
    model = make_mlp(runtime, scheme_kind, sequence_parallel, linear_beta=linear_beta)
    full = torch.linspace(-3, 3, 4 * input_width, dtype=torch.float32).view(4, input_width).to(torch.bfloat16)

    def gather(x):
        runtime.calls.append("gather")
        return full

    def scatter(x):
        runtime.calls.append("scatter")
        return x[2:3]

    runtime.scope["sp_all_gather"] = Mock(side_effect=gather)
    runtime.scope["sp_reduce_scatter"] = Mock(side_effect=scatter)
    x = full[2:3] if sequence_parallel else full
    result = model(x)
    quantizer = runtime.npu.npu_dynamic_quant if scheme_kind == "int8" else runtime.npu.npu_dynamic_mx_quant
    quantizer.assert_called_once()
    quant_input = quantizer.call_args.args[0]
    assert quant_input is model.act_fn.output
    assert quant_input.dtype == torch.bfloat16
    assert quant_input.shape == (4, input_width // 2)
    model.gate_up_proj.assert_called_once()
    assert model.gate_up_proj.call_args.args[0] is full
    model.down_proj.assert_called_once()
    assert model.down_proj.call_args.args[0] is quant_input
    runtime.npu.npu_quant_matmul.assert_called_once()
    assert runtime.npu.npu_quant_matmul.call_args.kwargs["output_dtype"] == torch.bfloat16
    other_quantizer = runtime.npu.npu_dynamic_mx_quant if scheme_kind == "int8" else runtime.npu.npu_dynamic_quant
    other_quantizer.assert_not_called()
    quant_dtype = torch.int8 if scheme_kind == "int8" else torch.float8_e4m3fn
    expected = quant_input.to(quant_dtype).to(torch.bfloat16)
    torch.testing.assert_close(result, expected[2:3] if sequence_parallel else expected, rtol=0, atol=0)
    expected_calls = ["gate_up", "activation", "quantize", "matmul"]
    assert runtime.calls == (["gather", *expected_calls, "scatter"] if sequence_parallel else expected_calls)
    for kernel in runtime.kernels.values():
        kernel.assert_not_called()
    assert model.reduce_results is (not sequence_parallel)


@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("reduce_results", [False, True])
def test_dense_mlp_preserves_reduce_results_contract(runtime, sequence_parallel, reduce_results):
    model = make_mlp(runtime, "int8", sequence_parallel, reduce_results=reduce_results)
    assert model.reduce_results is (reduce_results and not sequence_parallel)
