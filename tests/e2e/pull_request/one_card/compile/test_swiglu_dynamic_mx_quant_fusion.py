#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import torch_npu
import vllm.config
from vllm.compilation.passes.fx_utils import OpOverload
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
from vllm.utils.system_utils import update_environment_variables

import vllm_ascend.ops.register_custom_ops  # noqa
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.compilation.passes.swiglu_dynamic_mx_quant_fusion_pass import (
    SwiGluDynamicMXQuantFusionPass,
)

from .backend import TestBackend

# Cache backend to avoid duplicate pattern registration
_backend_cache = None


def get_or_create_backend(vllm_config):
    """Get or create backend with fusion passes (cached to avoid duplicate pattern registration)."""
    global _backend_cache
    if _backend_cache is None:
        _backend_cache = TestBackend(
            custom_passes=[SwiGluDynamicMXQuantFusionPass(vllm_config=vllm_config)]
        )
    return _backend_cache


class TestSwiGluDynamicMXQuantModel(nn.Module):
    """
    A minimal test model that simulates the pattern:
        SwiGlu → DynamicMXQuant
    """

    def __init__(self, hidden_size: int, dtype: torch.dtype, device="npu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype

    def forward(self, x):
        """
        Forward pass:
          1. Perform SwiGlu activation (input is split in half: gate and up)
          2. Dynamic MX quantization
        Returns both quantized output and scale.
        """
        # SwiGlu splits input into gate and up, applies SiLU to gate, then multiplies
        swiglu_out = torch.ops.npu.npu_swiglu(x)

        # Dynamic MX quantization
        quantized_output = torch.ops.npu.npu_dynamic_mx_quant(swiglu_out, dst_type=torch.float8_e4m3fn)

        return quantized_output[0], quantized_output[1]

    def ops_in_model_before(self) -> list[OpOverload]:
        """Return the list of expected operators BEFORE fusion."""
        return [torch.ops.npu.npu_swiglu.default, torch.ops.npu.npu_dynamic_mx_quant.default]

    def ops_in_model_after(self) -> list[OpOverload]:
        """Return the list of expected operators AFTER successful fusion."""
        return [torch.ops.npu.npu_swiglu_mx_quant.default]


class TestSwiGluDynamicMXQuantSPModel(nn.Module):
    """
    A test model that simulates the pattern:
        SwiGlu → maybe_allgather → DynamicMXQuant
    """

    def __init__(self, hidden_size: int, dtype: torch.dtype, device="npu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype

    def forward(self, x):
        """
        Forward pass:
          1. Perform SwiGlu activation
          2. Maybe all_gather (sequence parallelism)
          3. Dynamic MX quantization
        Returns both quantized output and scale.
        """
        # SwiGlu splits input into gate and up, applies SiLU to gate, then multiplies
        swiglu_out = torch.ops.npu.npu_swiglu(x)

        # Maybe all_gather for sequence parallelism
        swiglu_out = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(swiglu_out, True)

        # Dynamic MX quantization
        quantized_output = torch.ops.npu.npu_dynamic_mx_quant(swiglu_out, dst_type=torch.float8_e4m3fn)

        return quantized_output[0], quantized_output[1]

    def ops_in_model_before(self) -> list[OpOverload]:
        """Return the list of expected operators BEFORE fusion."""
        return [
            torch.ops.npu.npu_swiglu.default,
            torch.ops.vllm.maybe_all_gather_and_maybe_unpad.default,
            torch.ops.npu.npu_dynamic_mx_quant.default,
        ]

    def ops_in_model_after(self) -> list[OpOverload]:
        """Return the list of expected operators AFTER successful fusion."""
        return [
            torch.ops.npu.npu_swiglu_mx_quant.default,
            torch.ops.vllm.maybe_all_gather_and_maybe_unpad.default,
        ]


class TestSwiGluDynamicMXQuantFloat4Model(nn.Module):
    """
    A test model that simulates the pattern:
        SwiGlu → DynamicMXQuant with float4_e2m1fn_x2
    """

    def __init__(self, hidden_size: int, dtype: torch.dtype, device="npu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype

    def forward(self, x):
        """
        Forward pass:
          1. Perform SwiGlu activation (input is split in half: gate and up)
          2. Dynamic MX quantization with float4
        Returns both quantized output and scale.
        """
        # SwiGlu splits input into gate and up, applies SiLU to gate, then multiplies
        swiglu_out = torch.ops.npu.npu_swiglu(x)

        # Dynamic MX quantization with float4
        quantized_output = torch.ops.npu.npu_dynamic_mx_quant(swiglu_out, dst_type=torch_npu.float4_e2m1fn_x2)

        return quantized_output[0], quantized_output[1]

    def ops_in_model_before(self) -> list[OpOverload]:
        """Return the list of expected operators BEFORE fusion."""
        return [torch.ops.npu.npu_swiglu.default, torch.ops.npu.npu_dynamic_mx_quant.default]

    def ops_in_model_after(self) -> list[OpOverload]:
        """Return the list of expected operators AFTER successful fusion."""
        return [torch.ops.npu.npu_swiglu_mx_quant.default]


class TestSwiGluDynamicMXQuantSPFloat4Model(nn.Module):
    """
    A test model that simulates the pattern:
        SwiGlu → maybe_allgather → DynamicMXQuant with float4_e2m1fn_x2
    """

    def __init__(self, hidden_size: int, dtype: torch.dtype, device="npu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype

    def forward(self, x):
        """
        Forward pass:
          1. Perform SwiGlu activation
          2. Maybe all_gather (sequence parallelism)
          3. Dynamic MX quantization with float4
        Returns both quantized output and scale.
        """
        # SwiGlu splits input into gate and up, applies SiLU to gate, then multiplies
        swiglu_out = torch.ops.npu.npu_swiglu(x)

        # Maybe all_gather for sequence parallelism
        swiglu_out = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(swiglu_out, True)

        # Dynamic MX quantization with float4
        quantized_output = torch.ops.npu.npu_dynamic_mx_quant(swiglu_out, dst_type=torch_npu.float4_e2m1fn_x2)

        return quantized_output[0], quantized_output[1]

    def ops_in_model_before(self) -> list[OpOverload]:
        """Return the list of expected operators BEFORE fusion."""
        return [
            torch.ops.npu.npu_swiglu.default,
            torch.ops.vllm.maybe_all_gather_and_maybe_unpad.default,
            torch.ops.npu.npu_dynamic_mx_quant.default,
        ]

    def ops_in_model_after(self) -> list[OpOverload]:
        """Return the list of expected operators AFTER successful fusion."""
        return [
            torch.ops.npu.npu_swiglu_mx_quant.default,
            torch.ops.vllm.maybe_all_gather_and_maybe_unpad.default,
        ]


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [128])  # Input size is 2*hidden_size for gate+up
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("sp_enable", [False, True])
def test_swiglu_dynamic_mx_quant_fusion(
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
    sp_enable: bool,
):
    """
    End-to-end test for SwiGlu+DynamicMXQuant fusion with float8.
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

        if sp_enable:
            model = TestSwiGluDynamicMXQuantSPModel(hidden_size, dtype, device="npu")
        else:
            model = TestSwiGluDynamicMXQuantModel(hidden_size, dtype, device="npu")
        model = model.to("npu")

        # Input is 2*hidden_size for gate and up projections
        x = torch.rand(num_tokens, hidden_size * 2, device="npu", dtype=dtype, requires_grad=False)

        result_unfused = model(x)
        print("Unfused result:", [t.shape for t in result_unfused])
        model_fused = torch.compile(model, backend=backend)
        result_fused = model_fused(x)
        print("Fused result:", [t.shape for t in result_fused])

        print("=== Checking operator fusion ===")
        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=not sp_enable)
        backend.check_after_ops(model.ops_in_model_after())


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [128])  # Input size is 2*hidden_size for gate+up
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("sp_enable", [False, True])
def test_swiglu_dynamic_mx_quant_fusion_float4(
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
    sp_enable: bool,
):
    """
    End-to-end test for SwiGlu+DynamicMXQuant fusion with float4_e2m1fn_x2.
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

        if sp_enable:
            model = TestSwiGluDynamicMXQuantSPFloat4Model(hidden_size, dtype, device="npu")
        else:
            model = TestSwiGluDynamicMXQuantFloat4Model(hidden_size, dtype, device="npu")
        model = model.to("npu")

        # Input is 2*hidden_size for gate and up projections
        x = torch.rand(num_tokens, hidden_size * 2, device="npu", dtype=dtype, requires_grad=False)

        result_unfused = model(x)
        print("Unfused result:", [t.shape for t in result_unfused])
        model_fused = torch.compile(model, backend=backend)
        result_fused = model_fused(x)
        print("Fused result:", [t.shape for t in result_fused])

        print("=== Checking operator fusion ===")
        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=not sp_enable)
        backend.check_after_ops(model.ops_in_model_after())