# SPDX-License-Identifier: Apache-2.0
"""NPU regression for the Kimi latent RMSNorm/MXFP8/up-projection path."""

from unittest.mock import patch

import pytest
import torch
import torch_npu
from torch import nn
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.kimi_k3.amd.linear import KimiRoutedOutputTransform

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.models.kimi_k3 import AscendKimiRoutedOutputTransform
from vllm_ascend.ops.linear import AscendReplicatedLinear
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod


@pytest.fixture(autouse=True)
def _require_mx_norm_fusion():
    if not get_current_hardware_profile().supports(HardwareCapability.DYNAMIC_MX_QUANT_FUSION) or not hasattr(
        torch.ops.npu, "npu_rms_norm_dynamic_mx_quant"
    ):
        pytest.skip("Requires A5 and RmsNormDynamicMxQuant")


class _Norm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, device="npu", dtype=torch.bfloat16), requires_grad=False)
        self.variance_epsilon = 1e-6

    def forward(self, x):
        return torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)[0]


def _make_projection(scale_alg, quantized=True):
    # Avoid model/config/distributed setup while exercising the real linear
    # forward, adapter, MXFP8 apply(), and (q, scale) input contract.
    layer = AscendReplicatedLinear.__new__(AscendReplicatedLinear)
    nn.Module.__init__(layer)
    layer.custom_op = None
    layer.bias = None
    layer.skip_bias_add = False
    layer.return_bias = True
    weight = torch.randn(7168, 3584, device="npu", dtype=torch.bfloat16) * 0.02
    if quantized:
        scheme = AscendW8A8MXFP8DynamicLinearMethod.__new__(AscendW8A8MXFP8DynamicLinearMethod)
        scheme.group_size = 32
        scheme.dynamic_mx_quant_scale_alg = scale_alg
        layer.quant_method = AscendLinearMethod(scheme)
        q, scale = torch_npu.npu_dynamic_mx_quant(weight, dst_type=torch.float8_e4m3fn, scale_alg=scale_alg)
        layer.weight = nn.Parameter(q.t().contiguous(), requires_grad=False)
        layer.weight_scale = nn.Parameter(scale.transpose(0, 1).contiguous(), requires_grad=False)
        layer.mxfp8_tp_padding = (0, 0)
    else:
        layer.quant_method = UnquantizedLinearMethod()
        layer.weight = nn.Parameter(weight, requires_grad=False)
    return layer


@pytest.mark.parametrize("scale_alg", [0, 1])
def test_kimi_rmsnorm_mxquant_projection_and_graph(scale_alg):
    torch.manual_seed(921)
    norm = _Norm(3584)
    projection = _make_projection(scale_alg)
    baseline = KimiRoutedOutputTransform(norm, projection)
    fused = AscendKimiRoutedOutputTransform(norm, projection)
    x = torch.randn(8, 3584, device="npu", dtype=torch.bfloat16)
    op = torch.ops.npu.npu_rms_norm_dynamic_mx_quant
    with torch.inference_mode(), patch.object(torch.ops.npu, "npu_rms_norm_dynamic_mx_quant", wraps=op) as called:
        actual = fused(x)
        torch.testing.assert_close(actual, baseline(x), atol=0, rtol=0)
        assert called.call_count == 1
        assert called.call_args.kwargs["scale_alg"] == scale_alg
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = fused(x)
        for multiplier in [0.0, 0.1, 10.0]:
            x.copy_(torch.randn_like(x) * multiplier)
            graph.replay()
            torch.testing.assert_close(actual, baseline(x), atol=0, rtol=0)


@pytest.mark.parametrize("fallback", ["non_mx", "no_norm", "unsupported_runtime"])
def test_kimi_rmsnorm_mxquant_fallback(fallback):
    projection = _make_projection(0, quantized=fallback != "non_mx")
    norm = None if fallback == "no_norm" else _Norm(3584)
    baseline = KimiRoutedOutputTransform(norm, projection)
    fused = AscendKimiRoutedOutputTransform(norm, projection)
    if fallback == "unsupported_runtime":
        fused._supports_mx_norm_fusion = False
    x = torch.randn(8, 3584, device="npu", dtype=torch.bfloat16)
    with torch.inference_mode(), patch.object(torch.ops.npu, "npu_rms_norm_dynamic_mx_quant") as called:
        torch.testing.assert_close(fused(x), baseline(x), atol=0, rtol=0)
        called.assert_not_called()
