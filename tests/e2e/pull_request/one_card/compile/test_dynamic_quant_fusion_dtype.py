# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm_ascend.ops.register_custom_ops  # noqa: F401
from vllm_ascend.compilation.passes.norm_quant_fusion_pass import (
    AddRMSNormDynamicQuantPattern,
    AddRMSNormDynamicQuantPatternWithBias,
)
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

try:
    import npugraph_ex as nge

    _COMPILER_MODE = "npugraph_ex"
except ImportError:
    import torchair as nge

    _COMPILER_MODE = "reduce-overhead"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("quant_kind", ["default_int8", "explicit_int8", "fp8"])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("use_bias", [False, True])
def test_dynamic_quant_fusion_preserves_dtype(dtype, quant_kind, eps, use_bias):
    quant_dtype = torch.float8_e4m3fn if quant_kind == "fp8" else torch.int8
    if quant_dtype == torch.float8_e4m3fn and get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("FP8 dynamic quantization is unsupported on this NPU")
    if use_bias and not hasattr(torch.ops._C_ascend, "npu_add_rms_norm_bias"):
        pytest.skip("The custom add-RMSNorm-bias operator is unavailable")

    config = SimpleNamespace(model_config=SimpleNamespace(dtype=dtype))
    pattern_cls = AddRMSNormDynamicQuantPatternWithBias if use_bias else AddRMSNormDynamicQuantPattern
    pattern_cls(config, eps=eps).register(PatternMatcherPass())
    torch.manual_seed(42)
    x = torch.randn(32, 256, dtype=dtype, device="npu")
    residual = torch.randn_like(x)
    weight = torch.randn(256, dtype=dtype, device="npu")
    bias = torch.randn_like(weight)

    def model(x, residual, weight, bias):
        if use_bias:
            norm, _, updated_residual = torch.ops._C_ascend.npu_add_rms_norm_bias(x, residual, weight, bias, eps)
        else:
            norm, _, updated_residual = torch.ops.npu.npu_add_rms_norm(x, residual, weight, eps)
        if quant_kind == "default_int8":
            quantized, scale = torch.ops.npu.npu_dynamic_quant(norm)
        else:
            quantized, scale = torch.ops.npu.npu_dynamic_quant(norm, dst_type=quant_dtype)
        return quantized, scale, updated_residual

    optimized_targets: list[object] = []
    original_optimize = nge.npu_fx_compiler._optimize_fx

    def inspect_graph(gm, *args, **kwargs):
        result = original_optimize(gm, *args, **kwargs)
        optimized_targets.extend(node.target for node in gm.graph.nodes if node.op == "call_function")
        return result

    torch._dynamo.reset()
    compiler_config = nge.CompilerConfig()
    compiler_config.mode = _COMPILER_MODE
    backend = nge.get_npu_backend(compiler_config=compiler_config)
    with torch.no_grad(), patch.object(nge.npu_fx_compiler, "_optimize_fx", inspect_graph):
        expected = model(x, residual, weight, bias)
        actual = torch.compile(model, backend=backend, fullgraph=True, dynamic=True)(x, residual, weight, bias)
        torch.npu.synchronize()

    fused_op = torch.ops.npu.npu_add_rms_norm_dynamic_quant.default
    assert actual[0].dtype == expected[0].dtype == quant_dtype
    assert torch.equal(actual[2], expected[2])
    if quant_dtype == torch.float8_e4m3fn:
        assert fused_op not in optimized_targets
        assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
        assert torch.equal(actual[1], expected[1])
    else:
        assert fused_op in optimized_targets
        # The fused operator avoids the intermediate model-dtype rounding.
        torch.testing.assert_close(actual[0].float(), expected[0].float(), atol=1, rtol=0)
        torch.testing.assert_close(actual[1], expected[1], atol=1e-4, rtol=0.01)
