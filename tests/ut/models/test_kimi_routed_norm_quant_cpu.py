# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the actual latent transform and norm bias loader on CPU.

The CANN kernels and projection are mocked; this verifies routing and parameter
handoff, not NPU numerical accuracy. Run with --confcutdir=tests/ut/models.
"""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]


def load_nodes(path, names, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), path, "exec"), scope)


class NormBase(nn.Module):
    def __init__(self, hidden_size, eps, var_hidden_size, has_weight, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype or torch.bfloat16), requires_grad=False)
        self.variance_epsilon = eps

    def forward(self, x):
        return self.forward_oot(x)


class TransformBase(nn.Module):
    def __init__(self, norm, up_proj):
        super().__init__()
        self.norm, self.up_proj = norm, up_proj

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        return self.up_proj(x)[0]


@pytest.fixture
def runtime(monkeypatch):
    norm_kernel = Mock(side_effect=lambda x, *_: (x.clone(), None))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rms_norm=norm_kernel))
    fused_kernel = Mock()
    monkeypatch.setattr(torch.ops.npu, "npu_rms_norm_dynamic_mx_quant", fused_kernel, raising=False)
    int8_kernel = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "npu_rms_norm_dynamic_quant", int8_kernel, raising=False)
    config = SimpleNamespace(quant_config=SimpleNamespace(quant_description={}), lora_config=None)
    profile = SimpleNamespace(supports=Mock(return_value=True))
    scheme_cls = type("MXFP8Linear", (), {})
    scope = dict(
        torch=torch,
        nn=nn,
        RMSNorm=NormBase,
        KimiRoutedOutputTransform=TransformBase,
        AscendW8A8MXFP8DynamicLinearMethod=scheme_cls,
        _NORM_BIAS_IN_QUANT_DESCRIPTION={},
        get_current_vllm_config=lambda: config,
        get_current_hardware_profile=lambda: profile,
        HardwareCapability=SimpleNamespace(DYNAMIC_MX_QUANT_FUSION=object(), RMS_NORM_DYNAMIC_QUANT=object()),
        AscendLinearScheme=object,
        WeightSwitchGatherSpec=lambda *a, **kw: None,
        register_scheme=lambda *a: lambda cls: cls,
        __name__=__name__,
    )
    exec((ROOT / "vllm_ascend/quantization/prepared_linear_input.py").read_text(), scope)
    load_nodes("vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py", ["AscendW8A8DynamicLinearMethod"], scope)
    load_nodes("vllm_ascend/ops/layernorm.py", ["_quant_description_has_norm_bias", "AscendRMSNorm"], scope)
    load_nodes("vllm_ascend/models/kimi_k3/kimi_k3.py", ["AscendKimiRoutedOutputTransform"], scope)
    return SimpleNamespace(
        scope=scope,
        config=config,
        profile=profile,
        norm_kernel=norm_kernel,
        fused_kernel=fused_kernel,
        scheme_cls=scheme_cls,
        int8_kernel=int8_kernel,
    )


def make_transform(runtime, *, indexer_bias=True, hidden_size=3584, scale_alg=1):
    if indexer_bias:
        runtime.config.quant_config.quant_description = {
            "language_model.model.layers.3.self_attn.indexer.k_norm.bias": "FLOAT"
        }
    norm = runtime.scope["AscendRMSNorm"](hidden_size, eps=1e-6)
    scheme = runtime.scheme_cls()
    scheme.group_size, scheme.dynamic_mx_quant_scale_alg = 32, scale_alg
    up_proj = Mock(quant_method=SimpleNamespace(quant_method=scheme), return_value=(object(), None))
    transform = runtime.scope["AscendKimiRoutedOutputTransform"](norm, up_proj)
    return transform, norm, up_proj


@pytest.mark.parametrize("indexer_bias", [False, True])
@pytest.mark.parametrize("scale_alg", [0, 1])
def test_unloaded_bias_uses_fused_norm_quant_without_requantizing(runtime, indexer_bias, scale_alg):
    transform, norm, up_proj = make_transform(runtime, indexer_bias=indexer_bias, scale_alg=scale_alg)
    assert (norm.bias is not None) is indexer_bias
    assert not norm.bias_loaded
    x = torch.randn(4, 3584, dtype=torch.bfloat16)
    quantized = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(4, 56, 2, dtype=torch.uint8)
    runtime.fused_kernel.return_value = (quantized, scales, torch.empty(4, 1))

    result = transform(x)

    runtime.fused_kernel.assert_called_once_with(
        x, norm.weight, epsilon=norm.variance_epsilon, scale_alg=scale_alg, dst_type=torch.float8_e4m3fn
    )
    runtime.norm_kernel.assert_not_called()
    up_proj.assert_called_once()
    passed = up_proj.call_args.args[0]
    assert passed[0] is quantized and passed[1] is scales
    assert result is up_proj.return_value[0]


def test_loaded_bias_preserves_native_addition(runtime):
    transform, norm, up_proj = make_transform(runtime)
    bias = torch.full((3584,), 0.25, dtype=torch.bfloat16)
    norm._bias_weight_loader(norm.bias, bias)
    assert norm.bias_loaded
    x = torch.randn(4, 3584, dtype=torch.bfloat16)
    transform(x)
    runtime.fused_kernel.assert_not_called()
    runtime.norm_kernel.assert_called_once()
    torch.testing.assert_close(up_proj.call_args.args[0], x + bias)


@pytest.mark.parametrize("unsupported", ["hardware", "group_size", "dtype", "width", "scheme", "unknown_bias"])
def test_existing_fusion_constraints_remain_native(runtime, unsupported):
    width = 3552 if unsupported == "width" else 3584
    if unsupported == "hardware":
        runtime.profile.supports.return_value = False
    transform, norm, up_proj = make_transform(runtime, hidden_size=width)
    if unsupported == "group_size":
        up_proj.quant_method.quant_method.group_size = 64
    elif unsupported == "scheme":
        up_proj.quant_method.quant_method = object()
    elif unsupported == "unknown_bias":
        # A norm without Ascend's loader flag must not silently drop its bias.
        del norm.bias_loaded
        norm.forward = Mock(side_effect=lambda x: x)
    dtype = torch.float16 if unsupported == "dtype" else torch.bfloat16
    x = torch.randn(4, width, dtype=dtype)
    transform(x)
    runtime.fused_kernel.assert_not_called()
    assert isinstance(up_proj.call_args.args[0], torch.Tensor)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("indexer_bias", [False, True])
def test_int8_norm_reaches_actual_projection_without_requantizing(runtime, dtype, indexer_bias):
    transform, norm, up_proj = make_transform(runtime, indexer_bias=indexer_bias)
    norm.weight.data = norm.weight.data.to(dtype)
    scheme = runtime.scope["AscendW8A8DynamicLinearMethod"]()
    up_proj.quant_method.quant_method = scheme
    x = torch.randn(4, 3584, dtype=dtype)
    q, scale = torch.ones_like(x, dtype=torch.int8), torch.ones(4)
    runtime.int8_kernel.return_value = q, scale
    matmul = Mock(return_value=torch.zeros(4, 7168, dtype=dtype))
    quantize = Mock(side_effect=AssertionError("prepared norm output must not be quantized again"))
    runtime.scope["torch_npu"] = SimpleNamespace(npu_dynamic_quant=quantize, npu_quant_matmul=matmul)
    layer = SimpleNamespace(weight=object(), weight_scale=object())
    up_proj.side_effect = lambda value: (scheme.apply(layer, value), None)

    result = transform(x)

    runtime.int8_kernel.assert_called_once_with(x, norm.weight, epsilon=norm.variance_epsilon)
    runtime.norm_kernel.assert_not_called()
    runtime.fused_kernel.assert_not_called()
    quantize.assert_not_called()
    matmul.assert_called_once_with(
        q, layer.weight, layer.weight_scale, pertoken_scale=scale, bias=None, output_dtype=dtype
    )
    assert result is matmul.return_value


@pytest.mark.parametrize(
    "unsupported",
    [
        "hardware",
        "lora",
        "fp8",
        "subclass",
        "dtype",
        "gamma_dtype",
        "rank",
        "empty",
        "loaded_bias",
        "unknown_bias",
        "no_norm",
    ],
)
def test_int8_ineligible_cases_keep_split_path(runtime, unsupported):
    if unsupported == "hardware":
        runtime.profile.supports.return_value = False
    if unsupported == "lora":
        runtime.config.lora_config = object()
    transform, norm, up_proj = make_transform(runtime)
    cls = runtime.scope["AscendW8A8DynamicLinearMethod"]
    scheme = cls()
    up_proj.quant_method.quant_method = scheme
    x = torch.randn(4, 3584, dtype=torch.bfloat16)
    if unsupported == "fp8":
        scheme.act_quant_type = torch.float8_e4m3fn
    elif unsupported == "subclass":
        up_proj.quant_method.quant_method = type("FP8Subclass", (cls,), {})()
    elif unsupported == "dtype":
        x = x.float()
    elif unsupported == "gamma_dtype":
        norm.weight.data = norm.weight.data.float()
    elif unsupported == "rank":
        x = x.unsqueeze(0)
    elif unsupported == "empty":
        x = x[:0]
    elif unsupported == "loaded_bias":
        norm._bias_weight_loader(norm.bias, torch.ones_like(norm.bias))
    elif unsupported == "unknown_bias":
        del norm.bias_loaded
        norm.forward = Mock(side_effect=lambda value: value)
    elif unsupported == "no_norm":
        transform.norm = None
    transform(x)
    runtime.int8_kernel.assert_not_called()
    runtime.fused_kernel.assert_not_called()
    assert isinstance(up_proj.call_args.args[0], torch.Tensor)
