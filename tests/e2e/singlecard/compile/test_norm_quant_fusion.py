#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import List

import pytest
import torch
import torch.nn as nn
import torch_npu
import vllm.config
from vllm.compilation.fx_utils import OpOverload
from vllm.config import ModelConfig, VllmConfig

from tests.e2e.singlecard.compile.backend import TestBackend
from vllm_ascend.compilation.passes.norm_quant_fusion_pass import \
    AddRMSNormQuantFusionPass


class TestModelWithoutBias(nn.Module):
    """
    A minimal test model that simulates the pattern:
        AddRMSNorm → Quantization (without bias)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, device="npu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.rms_norm_weight = nn.Parameter(
            torch.randn(hidden_size, device=device))
        self.quant_scale = torch.tensor([1.0], device=device)
        self.quant_offset = torch.tensor([0.0], device=device)

    def forward(self, x):
        """
        Forward pass:
          1. Perform npu_add_rms_norm
          2. Quantize the normalized output to int8
        Returns both quantized output and updated residual.
        """
        residual = torch.zeros_like(x)

        norm_output, _, new_residual = torch_npu.npu_add_rms_norm(
            x, residual, self.rms_norm_weight, self.eps)

        quantized_output = torch_npu.npu_quantize(norm_output,
                                                  self.quant_scale,
                                                  self.quant_offset,
                                                  torch.qint8, -1, False)

        return quantized_output, new_residual

    def ops_in_model_before(self) -> List[OpOverload]:
        """Return the list of expected operators BEFORE fusion."""
        return [
            torch.ops.npu.npu_add_rms_norm.default,
            torch.ops.npu.npu_quantize.default
        ]

    def ops_in_model_after(self) -> List[OpOverload]:
        """Return the list of expected operators AFTER successful fusion."""
        return [torch.ops.npu.npu_add_rms_norm_quant.default]


class TestModelWithBias(nn.Module):
    """
    A test model that simulates the pattern:
        AddRMSNorm → Add Bias → Quantization (with bias)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, device="npu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.rms_norm_weight = nn.Parameter(
            torch.randn(hidden_size, device=device))
        self.bias = nn.Parameter(torch.randn(hidden_size, device=device))
        self.quant_scale = torch.tensor([1.0], device=device)
        self.quant_offset = torch.tensor([0.0], device=device)

    def forward(self, x):
        """
        Forward pass:
          1. Perform npu_add_rms_norm
          2. Add bias
          3. Quantize to int8
        Returns both quantized output and updated residual.
        """
        residual = torch.zeros_like(x)

        norm_output, _, new_residual = torch_npu.npu_add_rms_norm(
            x, residual, self.rms_norm_weight, self.eps)

        # Add bias
        norm_output_with_bias = norm_output + self.bias

        quantized_output = torch_npu.npu_quantize(norm_output_with_bias,
                                                  self.quant_scale,
                                                  self.quant_offset,
                                                  torch.qint8, -1, False)

        return quantized_output, new_residual

    def ops_in_model_before(self) -> List[OpOverload]:
        """Return the list of expected operators BEFORE fusion."""
        return [
            torch.ops.npu.npu_add_rms_norm.default,
            torch.ops.aten.add.Tensor,  # Add bias operation
            torch.ops.npu.npu_quantize.default
        ]

    def ops_in_model_after(self) -> List[OpOverload]:
        """Return the list of expected operators AFTER successful fusion."""
        return [torch.ops.npu.npu_add_rms_norm_quant.default]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("use_bias", [False, True])
def test_rmsnorm_quant_fusion(dtype: torch.dtype, hidden_size: int,
                              num_tokens: int, eps: float, use_bias: bool):
    """
    End-to-end test for AddRMSNorm+Quantize fusion.
    Compares: Operator presence/absence before and after graph transformation
    """
    torch.set_default_dtype(dtype)
    torch.manual_seed(1)

    vllm_config = VllmConfig(model_config=ModelConfig(dtype=dtype))

    with vllm.config.set_current_vllm_config(vllm_config):
        backend = TestBackend(
            custom_passes=[AddRMSNormQuantFusionPass(vllm_config=vllm_config)])
        if use_bias:
            model = TestModelWithBias(hidden_size, eps, device="npu")
        else:
            model = TestModelWithoutBias(hidden_size, eps, device="npu")
        model = model.to("npu")

        x = torch.rand(num_tokens,
                       hidden_size,
                       device="npu",
                       dtype=dtype,
                       requires_grad=False)

        result_unfused = model(x)
        print("Unfused result:", [t.shape for t in result_unfused])
        model_fused = torch.compile(model, backend=backend)
        result_fused = model_fused(x)
        print("Fused result:", [t.shape for t in result_fused])

        print("=== Checking operator fusion ===")
        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())
