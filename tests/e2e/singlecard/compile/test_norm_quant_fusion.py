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

import pytest
import torch
import torch.nn as nn
import vllm.config
from vllm.compilation.passes.fx_utils import OpOverload
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
from vllm.utils.system_utils import update_environment_variables

import vllm_ascend.ops.register_custom_ops  # noqa
from tests.e2e.singlecard.compile.backend import TestBackend
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.compilation.passes.norm_quant_fusion_pass import AddRMSNormQuantFusionPass
from vllm_ascend.utils import enable_custom_op

# Cache backend to avoid duplicate pattern registration
_backend_cache = None


def get_or_create_backend(vllm_config):
    """Get or create backend with fusion passes (cached to avoid duplicate pattern registration)."""
    global _backend_cache
    if _backend_cache is None:
        _backend_cache = TestBackend(custom_passes=[AddRMSNormQuantFusionPass(vllm_config=vllm_config)])
    return _backend_cache


class TestModelWithoutBias(nn.Module):
    """
    A minimal test model that simulates the pattern:
        AddRMSNorm → Quantization (without bias)
    """

    def __init__(self, hidden_size: int, dtype: torch.dtype, eps: float = 1e-6, device="npu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.rms_norm_weight = nn.Parameter(torch.randn(hidden_size, device=device))
        self.quant_scale = torch.ones(hidden_size, dtype=dtype, device=device)
        self.quant_scale_reciprocal = torch.ones(hidden_size, dtype=dtype, device=device)
        self.quant_offset = torch.zeros(hidden_size, dtype=dtype, device=device)

    def forward(self, x):
        """
        Forward pass:
          1. Perform npu_add_rms_norm
          2. Quantize the normalized output to int8
        Returns both quantized output and updated residual.
        """
        residual = torch.zeros_like(x)

        norm_output, _, new_residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
            x, residual, self.rms_norm_weight, None, self.eps
        )

        quantized_output = torch.ops.vllm.quantize(
            norm_output, self.quant_scale, self.quant_scale_reciprocal, self.quant_offset
        )

        return quantized_output, new_residual

    def ops_in_model_before(self) -> list[OpOverload]:
        """Return the list of expected operators BEFORE fusion."""
        return [torch.ops._C_ascend.npu_add_rms_norm_bias.default, torch.ops.vllm.quantize.default]

    def ops_in_model_after(self) -> list[OpOverload]:
        """Return the list of expected operators AFTER successful fusion."""
        return [torch.ops.npu.npu_add_rms_norm_quant.default]


class TestModelWithBias(nn.Module):
    """
    A test model that simulates the pattern:
        AddRMSNorm → Add Bias → Quantization (with bias)
    """

    def __init__(self, hidden_size: int, dtype: torch.dtype, eps: float = 1e-6, device="npu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.rms_norm_weight = nn.Parameter(torch.randn(hidden_size, device=device))
        self.bias = nn.Parameter(torch.randn(hidden_size, device=device))
        self.quant_scale = torch.ones(hidden_size, dtype=dtype, device=device)
        self.quant_scale_reciprocal = torch.ones(hidden_size, dtype=dtype, device=device)
        self.quant_offset = torch.zeros(hidden_size, dtype=dtype, device=device)

    def forward(self, x):
        """
        Forward pass:
          1. Perform npu_add_rms_norm
          2. Add bias
          3. Quantize to int8
        Returns both quantized output and updated residual.
        """
        residual = torch.zeros_like(x)

        norm_output_with_bias, _, new_residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
            x, residual, self.rms_norm_weight, self.bias, self.eps
        )

        quantized_output = torch.ops.vllm.quantize(
            norm_output_with_bias, self.quant_scale, self.quant_scale_reciprocal, self.quant_offset
        )

        return quantized_output, new_residual

    def ops_in_model_before(self) -> list[OpOverload]:
        """Return the list of expected operators BEFORE fusion."""
        return [torch.ops._C_ascend.npu_add_rms_norm_bias.default, torch.ops.vllm.quantize.default]

    def ops_in_model_after(self) -> list[OpOverload]:
        """Return the list of expected operators AFTER successful fusion."""
        return [torch.ops.npu.npu_add_rms_norm_quant.default]


class TestModelSPWithoutBias(nn.Module):
    """
    A minimal test model that simulates the pattern:
        AddRMSNorm → maybe_allgather → Quantization (without bias)
    """

    def __init__(self, hidden_size: int, dtype: torch.dtype, eps: float = 1e-6, device="npu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.rms_norm_weight = nn.Parameter(torch.randn(hidden_size, device=device))
        self.quant_scale = torch.ones(hidden_size, dtype=dtype, device=device)
        self.quant_scale_reciprocal = torch.ones(hidden_size, dtype=dtype, device=device)
        self.quant_offset = torch.zeros(hidden_size, dtype=dtype, device=device)

    def forward(self, x):
        """
        Forward pass:
          1. Perform npu_add_rms_norm
          2. Perform a fake maybe_all_gather_and_maybe_unpad
          3. Quantize the normalized output to int8
        Returns both quantized output and updated residual.
        """
        residual = torch.zeros_like(x)

        norm_output, _, new_residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
            x, residual, self.rms_norm_weight, None, self.eps
        )

        norm_output = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(norm_output, True)

        quantized_output = torch.ops.vllm.quantize(
            norm_output, self.quant_scale, self.quant_scale_reciprocal, self.quant_offset
        )

        return quantized_output, new_residual

    def ops_in_model_before(self) -> list[OpOverload]:
        """Return the list of expected operators BEFORE fusion."""
        return [
            torch.ops._C_ascend.npu_add_rms_norm_bias.default,
            torch.ops.vllm.maybe_all_gather_and_maybe_unpad.default,
            torch.ops.vllm.quantize.default,
        ]

    def ops_in_model_after(self) -> list[OpOverload]:
        """Return the list of expected operators AFTER successful fusion."""
        return [torch.ops.npu.npu_add_rms_norm_quant.default, torch.ops.vllm.maybe_all_gather_and_maybe_unpad.default]


class TestModelSPWithBias(nn.Module):
    """
    A minimal test model that simulates the pattern:
        AddRMSNorm → Add bias → maybe_allgather → Quantization (without bias)
    """

    def __init__(self, hidden_size: int, dtype: torch.dtype, eps: float = 1e-6, device="npu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.rms_norm_weight = nn.Parameter(torch.randn(hidden_size, device=device))
        self.bias = nn.Parameter(torch.randn(hidden_size, device=device))
        self.quant_scale = torch.ones(hidden_size, dtype=dtype, device=device)
        self.quant_scale_reciprocal = torch.ones(hidden_size, dtype=dtype, device=device)
        self.quant_offset = torch.zeros(hidden_size, dtype=dtype, device=device)

    def forward(self, x):
        """
        Forward pass:
          1. Perform npu_add_rms_norm
          2. Add bias
          3. Perform a fake maybe_all_gather_and_maybe_unpad
          4. Quantize the normalized output to int8
        Returns both quantized output and updated residual.
        """
        residual = torch.zeros_like(x)

        norm_output_with_bias, _, new_residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
            x, residual, self.rms_norm_weight, self.bias, self.eps
        )

        norm_output_with_bias = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(norm_output_with_bias, True)

        quantized_output = torch.ops.vllm.quantize(
            norm_output_with_bias, self.quant_scale, self.quant_scale_reciprocal, self.quant_offset
        )

        return quantized_output, new_residual

    def ops_in_model_before(self) -> list[OpOverload]:
        """Return the list of expected operators BEFORE fusion."""
        return [
            torch.ops._C_ascend.npu_add_rms_norm_bias.default,
            torch.ops.vllm.maybe_all_gather_and_maybe_unpad.default,
            torch.ops.vllm.quantize.default,
        ]

    def ops_in_model_after(self) -> list[OpOverload]:
        """Return the list of expected operators AFTER successful fusion."""
        return [torch.ops.npu.npu_add_rms_norm_quant.default, torch.ops.vllm.maybe_all_gather_and_maybe_unpad.default]


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("sp_enable", [False, True])
def test_rmsnorm_quant_fusion(
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
    eps: float,
    use_bias: bool,
    sp_enable: bool,
):
    """
    End-to-end test for AddRMSNorm+Quantize fusion.
    Compares: Operator presence/absence before and after graph transformation
    """
    torch.set_default_dtype(dtype)
    torch.manual_seed(1)

    vllm_config = VllmConfig(model_config=ModelConfig(dtype=dtype))

    with vllm.config.set_current_vllm_config(vllm_config):
        update_environment_variables(
            {
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12345",
            }
        )
        init_distributed_environment()
        ensure_model_parallel_initialized(1, 1)

    with vllm.config.set_current_vllm_config(vllm_config), set_ascend_forward_context(None, vllm_config):
        backend = get_or_create_backend(vllm_config)
        if use_bias:
            if not enable_custom_op():
                return
            if sp_enable:
                model = TestModelSPWithBias(hidden_size, dtype, eps, device="npu")
            else:
                model = TestModelWithBias(hidden_size, dtype, eps, device="npu")
        else:
            if sp_enable:
                model = TestModelSPWithoutBias(hidden_size, dtype, eps, device="npu")
            else:
                model = TestModelWithoutBias(hidden_size, dtype, eps, device="npu")
        model = model.to("npu")

        x = torch.rand(num_tokens, hidden_size, device="npu", dtype=dtype, requires_grad=False)

        result_unfused = model(x)
        print("Unfused result:", [t.shape for t in result_unfused])
        model_fused = torch.compile(model, backend=backend)
        result_fused = model_fused(x)
        print("Fused result:", [t.shape for t in result_fused])

        print("=== Checking operator fusion ===")
        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=not sp_enable)
        backend.check_after_ops(model.ops_in_model_after())
