# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU routing tests; execute production MLP/linear methods with NPU ops mocked.

Run with --confcutdir=tests/ut/models. These tests do not validate device math.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]


def load_nodes(path, names, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), path, "exec"), scope)


class Situ(nn.Module):
    def __init__(self, beta=4.0, linear_beta=25.0):
        super().__init__()
        self.beta, self.linear_beta = beta, linear_beta
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x[..., : x.shape[-1] // 2]


class MLPBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.act_fn = Situ(kwargs["activation_situ_beta"], kwargs["activation_situ_linear_beta"])
        self.reduce_results = kwargs["reduce_results"]
        self.down_proj = kwargs["quant_config"].down_proj
        self.gate_up_proj = kwargs["quant_config"].gate_up_proj

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        return self.down_proj(self.act_fn(gate_up))[0]


@pytest.fixture
def runtime(monkeypatch):
    scope = dict(__name__=__name__, torch=torch, nn=nn, math=math, KimiMLP=MLPBase, SituAndMul=Situ)
    # Use the real hardware mapping without importing vLLM or torch-npu.
    exec((ROOT / "vllm_ascend/device/hardware.py").read_text(), scope)
    scope["get_device_config"] = lambda: SimpleNamespace(_device_type=scope["AscendDeviceType"].A5)
    tree = ast.parse((ROOT / "vllm_ascend/device/hardware_profile.py").read_text())
    tree.body = [
        n
        for n in tree.body
        if not (isinstance(n, ast.ImportFrom) and n.module is not None and n.module.startswith("vllm_ascend"))
    ]
    exec(compile(tree, "hardware_profile.py", "exec"), scope)
    exec((ROOT / "vllm_ascend/quantization/prepared_linear_input.py").read_text(), scope)
    config = SimpleNamespace(lora_config=None)
    scope.update(
        get_current_vllm_config=lambda: config,
        AscendLinearScheme=object,
        WeightSwitchGatherSpec=lambda *a, **kw: None,
        register_scheme=lambda *a: lambda cls: cls,
    )
    npu = SimpleNamespace(
        float8_e8m0fnu=object(),
        npu_dynamic_quant=Mock(side_effect=AssertionError("unexpected requantization")),
        npu_dynamic_mx_quant=Mock(side_effect=AssertionError("unexpected requantization")),
        npu_quant_matmul=Mock(side_effect=lambda q, *a, output_dtype, **kw: q.to(output_dtype)),
    )
    scope["torch_npu"] = npu
    for file, cls in [
        ("w8a8_dynamic", "AscendW8A8DynamicLinearMethod"),
        ("w8a8_mxfp8", "AscendW8A8MXFP8DynamicLinearMethod"),
    ]:
        load_nodes(f"vllm_ascend/quantization/methods/w8a8/{file}.py", [cls], scope)
    kernels = {}
    for name, dtype in [("situ_mx_quant", torch.float8_e4m3fn), ("dequant_situ_quant", torch.int8)]:

        def fused(x, *, _dtype=dtype, **kwargs):
            q = x[:, : x.shape[-1] // 2].to(_dtype)
            scale_shape = (x.shape[0], x.shape[-1] // 128, 2) if _dtype == torch.float8_e4m3fn else (x.shape[0],)
            return q, torch.ones(scale_shape)

        kernels[name] = Mock(side_effect=fused)
        monkeypatch.setattr(torch.ops._C_ascend, name, kernels[name], raising=False)
    load_nodes("vllm_ascend/models/kimi_k3/kimi_k3.py", ["AscendKimiMLP"], scope)
    return SimpleNamespace(scope=scope, config=config, npu=npu, kernels=kernels)


def make_mlp(
    runtime,
    hardware="A5",
    sequence_parallel=False,
    *,
    width=64,
    beta=4.0,
    linear_beta=25.0,
    hidden_act="situ",
    scheme_kind=None,
    input_shard=True,
    group_size=32,
    scale_alg=0,
):
    s = runtime.scope
    s["get_current_hardware_profile"] = Mock(
        side_effect=lambda: s["get_hardware_profile"](getattr(s["AscendDeviceType"], hardware))
    )
    cls = s["AscendW8A8MXFP8DynamicLinearMethod" if hardware == "A5" else "AscendW8A8DynamicLinearMethod"]
    if scheme_kind == "subclass":
        cls = type("OtherScheme", (cls,), {})
    scheme = object() if scheme_kind == "scheme" else cls.__new__(cls)
    if scheme_kind != "scheme":
        scheme.group_size, scheme.dynamic_mx_quant_scale_alg = group_size, scale_alg
    layer = SimpleNamespace(
        weight=object(),
        weight_scale=object(),
        input_is_parallel=input_shard,
        quant_method=SimpleNamespace(quant_method=scheme),
    )
    down = Mock(wraps=lambda x: (scheme.apply(layer, x), None))
    down.input_is_parallel = input_shard
    down.input_size_per_partition = width
    down.quant_method = layer.quant_method
    gate_up = Mock(side_effect=lambda x: (x, None))
    model = s["AscendKimiMLP"](
        128,
        2 * width,
        hidden_act,
        use_sequence_parallel=sequence_parallel,
        activation_situ_beta=beta,
        activation_situ_linear_beta=linear_beta,
        quant_config=SimpleNamespace(down_proj=down, gate_up_proj=gate_up),
    )
    return model, scheme


@pytest.mark.parametrize("hardware", ["A3", "A5"])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("linear_beta", [None, 25.0])
def test_fused_input_reaches_real_linear_once_and_preserves_sp(runtime, hardware, sequence_parallel, linear_beta):
    model, _ = make_mlp(runtime, hardware, sequence_parallel, linear_beta=linear_beta)
    full = torch.arange(4, dtype=torch.bfloat16)[:, None].expand(4, 128).contiguous()
    runtime.scope["sp_all_gather"] = Mock(return_value=full)
    runtime.scope["sp_reduce_scatter"] = Mock(side_effect=lambda x: x[2:3])
    x = full[2:3] if sequence_parallel else full
    result = model(x)
    torch.testing.assert_close(result, x[:, :64], rtol=0, atol=0)
    selected = "situ_mx_quant" if hardware == "A5" else "dequant_situ_quant"
    kernel = runtime.kernels[selected]
    kernel.assert_called_once()
    assert kernel.call_args.kwargs["activate_left"] is True
    assert kernel.call_args.kwargs["beta"] == 4.0
    assert kernel.call_args.kwargs["linear_beta"] == (linear_beta or 0.0)
    if hardware == "A3":
        assert kernel.call_args.kwargs == dict(
            beta=4.0, linear_beta=linear_beta or 0.0, activate_left=True, quant_mode="dynamic"
        )
    assert model.act_fn.calls == 0
    model.gate_up_proj.assert_called_once()
    model.down_proj.assert_called_once()
    prepared = model.down_proj.call_args.args[0]
    assert isinstance(prepared, runtime.scope["PreparedLinearInput"])
    assert runtime.npu.npu_quant_matmul.call_args.kwargs["pertoken_scale"] is prepared.scale
    assert prepared.output_dtype == torch.bfloat16
    runtime.npu.npu_dynamic_quant.assert_not_called()
    runtime.npu.npu_dynamic_mx_quant.assert_not_called()
    assert runtime.scope["sp_all_gather"].call_count == int(sequence_parallel)
    assert runtime.scope["sp_reduce_scatter"].call_count == int(sequence_parallel)
    assert model.reduce_results is (not sequence_parallel)


@pytest.mark.parametrize("hardware", ["A3", "A5"])
@pytest.mark.parametrize("unsupported", ["scheme", "subclass", "activation"])
def test_other_schemes_and_activations_keep_original_path(runtime, hardware, unsupported):
    options = {"hidden_act": "silu"} if unsupported == "activation" else {"scheme_kind": unsupported}
    model, _ = make_mlp(runtime, hardware, **options)
    assert isinstance(model._prepare_down_input(torch.zeros(4, 128, dtype=torch.bfloat16)), torch.Tensor)
    assert model.act_fn.calls == 1
    for kernel in runtime.kernels.values():
        kernel.assert_not_called()


@pytest.mark.parametrize("unsupported", ["group_size", "scale_alg"])
def test_mx_quantization_contract_is_not_bypassed(runtime, unsupported):
    options = {"group_size": 64} if unsupported == "group_size" else {"scale_alg": 1}
    model, _ = make_mlp(runtime, **options)
    assert isinstance(model._prepare_down_input(torch.zeros(4, 128, dtype=torch.bfloat16)), torch.Tensor)
    runtime.kernels["situ_mx_quant"].assert_not_called()


@pytest.mark.parametrize("hardware", ["A3", "A5"])
@pytest.mark.parametrize(
    "case", ["dtype", "rank", "empty", "strides", "alignment", "wide", "beta", "linear_beta", "input_shard"]
)
def test_selected_kernel_owns_input_validation(runtime, hardware, case):
    options: dict[str, Any] = {}
    if case == "alignment":
        options["width"] = 63
    elif case == "wide":
        options["width"] = 16896
    elif case == "beta":
        options["beta"] = -1.0
    elif case == "linear_beta":
        options["linear_beta"] = -2.0
    elif case == "input_shard":
        options["input_shard"] = False
    model, _ = make_mlp(runtime, hardware, **options)
    x = torch.zeros(4, 2 * options.get("width", 64), dtype=torch.bfloat16)
    if case == "dtype":
        x = x.float()
    elif case == "rank":
        x = x.unsqueeze(0)
    elif case == "empty":
        x = x[:0]
    elif case == "strides":
        x = torch.zeros(128, 4, dtype=x.dtype).t()
    kernel = runtime.kernels["situ_mx_quant" if hardware == "A5" else "dequant_situ_quant"]
    kernel.side_effect = RuntimeError("kernel input contract")
    with pytest.raises(RuntimeError, match="kernel input contract"):
        model._prepare_down_input(x)
    assert kernel.call_args.args[0] is x
    assert model.act_fn.calls == 0


@pytest.mark.parametrize("hardware", ["A2", "_310P", "A3", "A5"])
def test_hardware_and_lora_exclusions(runtime, hardware):
    runtime.config.lora_config = object() if hardware in ("A3", "A5") else None
    model, _ = make_mlp(runtime, hardware)
    assert not model._fuse_situ_mx_quant
    assert not model._fuse_situ_int8_quant


@pytest.mark.parametrize("hardware", ["A3", "A5"])
def test_operator_registration_can_happen_after_model_construction(runtime, monkeypatch, hardware):
    name = "dequant_situ_quant" if hardware == "A3" else "situ_mx_quant"
    kernel = runtime.kernels[name]
    monkeypatch.delattr(torch.ops._C_ascend, name)
    model, _ = make_mlp(runtime, hardware)
    x = torch.zeros(1, 128, dtype=torch.bfloat16)
    # Resolve the registered op when executing, not while constructing the model.
    monkeypatch.setattr(torch.ops._C_ascend, name, kernel, raising=False)
    assert isinstance(model._prepare_down_input(x), runtime.scope["PreparedLinearInput"])


@pytest.mark.parametrize("hardware", ["A3", "A5"])
def test_fixed_selection_is_not_repeated_in_forward(runtime, hardware):
    model, _ = make_mlp(runtime, hardware)
    select_profile = runtime.scope["get_current_hardware_profile"]
    assert select_profile.call_count == 1
    # These constructor inputs must no longer be read on the hot path.
    model.down_proj.quant_method = None
    model.act_fn.beta = float("nan")
    model.act_fn.linear_beta = float("nan")
    for _ in range(2):
        result = model._prepare_down_input(torch.zeros(4, 128, dtype=torch.bfloat16))
        assert isinstance(result, runtime.scope["PreparedLinearInput"])
    assert select_profile.call_count == 1
    kernel = runtime.kernels["situ_mx_quant" if hardware == "A5" else "dequant_situ_quant"]
    assert kernel.call_args.kwargs["beta"] == 4.0
    assert kernel.call_args.kwargs["linear_beta"] == 25.0
