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
import torch.nn.functional as F
import vllm.config
from vllm.compilation.passes.fx_utils import OpOverload
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.utils.system_utils import update_environment_variables

import vllm_ascend.ops.register_custom_ops  # noqa
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.compilation.passes.swiglu_fusion_pass import SwigluFusionPass

from .backend import TestBackend

# Cache backend to avoid duplicate pattern registration
_backend_cache = None


def get_or_create_backend(vllm_config):
    """Get or create backend with fusion passes (cached to avoid duplicate pattern registration)."""
    global _backend_cache
    if _backend_cache is None:
        _backend_cache = TestBackend(custom_passes=[SwigluFusionPass(vllm_config=vllm_config)])
    return _backend_cache


class TestSwigluModel(nn.Module):
    """Minimal model that writes the bare SwiGLU sequence:

        gate, up = x.chunk(2, dim=-1); out = F.silu(gate) * up

    Mirrors ``DeepseekV4VisionMLP.forward`` (which bypasses
    ``AscendSiluAndMul``/``npu_swiglu``).
    """

    def __init__(self, hidden_size: int, dtype: torch.dtype, device="npu"):
        super().__init__()
        # w1 outputs the concatenated gate_up tensor (2 * hidden_size).
        self.w1 = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.w2 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)

    def ops_in_model_before(self) -> list[OpOverload]:
        """Operators present before fusion (post-decomposition aten chain).

        chunk(2, dim=-1) decomposes to aten.split.Tensor; F.silu(gate)
        decomposes to aten.sigmoid + aten.mul; the final `* up` is another
        aten.mul.
        """
        return [
            torch.ops.aten.split.Tensor.default,
            torch.ops.aten.sigmoid.default,
            torch.ops.aten.mul.Tensor.default,
        ]

    def ops_in_model_after(self) -> list[OpOverload]:
        """Operator present after successful fusion."""
        return [torch.ops.npu.npu_swiglu.default]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("num_tokens", [257])
def test_swiglu_fusion(dtype: torch.dtype, hidden_size: int, num_tokens: int):
    """End-to-end test for chunk->silu->mul SwiGLU fusion into npu_swiglu.

    Verifies operator presence/absence before and after graph transformation
    plus numerical equivalence of the fused vs unfused result.
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

    with vllm.config.set_current_vllm_config(vllm_config), set_ascend_forward_context(
        None, vllm_config
    ):
        backend = get_or_create_backend(vllm_config)
        model = TestSwigluModel(hidden_size, dtype, device="npu")
        model = model.to("npu")

        x = torch.rand(num_tokens, hidden_size, device="npu", dtype=dtype, requires_grad=False)

        # no_grad keeps aot_autograd in fw-only mode so intermediate tensors
        # of the aten op chain are not retained for backward; the pattern
        # matcher requires the matched subgraph's intermediates to be dead.
        with torch.no_grad():
            result_unfused = model(x)
            model_fused = torch.compile(model, backend=backend)
            result_fused = model_fused(x)

        assert torch.allclose(result_fused, result_unfused, atol=1e-5, rtol=1e-3)

        print("=== Checking operator fusion ===")
        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=True)
        backend.check_after_ops(model.ops_in_model_after())
