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
# This file is a part of the vllm-ascend project.
#

from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import set_current_vllm_config

from vllm_ascend.ops.activation import (
    AscendQuickGELU,
    AscendSiluAndMul,
    AscendSwigluOAIAndMul,
    AscendSwigluStepAndMul,
)


@pytest.fixture
def default_vllm_config():
    mock_config = MagicMock()

    mock_config.compilation_config.dispatch_forward_backend = "eager"

    mock_config.compilation_config.custom_ops = ["all"]

    with set_current_vllm_config(mock_config):
        yield mock_config


def _swiglu_oai_reference(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    gate = x[..., ::2].clamp(max=limit)
    up = x[..., 1::2].clamp(min=-limit, max=limit)
    return (up + 1) * gate * torch.sigmoid(gate * alpha)


def _quick_gelu_reference(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


def _silu_and_mul_reference(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :d]) * x[..., d:]


def _swiglustep_and_mul_reference(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
    # Independent of the chunk-based implementation under test: slice the gate
    # (first half) and up (second half) halves and apply silu + symmetric clamp.
    d = x.shape[-1] // 2
    gate = torch.nn.functional.silu(x[..., :d]).clamp(max=limit)
    up = x[..., d:].clamp(min=-limit, max=limit)
    return gate * up


class TestAscendSwigluOAIAndMul:
    def test_swiglu_oai_forward_matches_reference_formula(self):
        x = torch.tensor([[8.0, 9.0, -2.0, -8.0, 3.0, 0.5, -9.0, 10.0]], dtype=torch.float32)
        result = AscendSwigluOAIAndMul.swiglu_oai_forward(x)
        expected = _swiglu_oai_reference(x)

        assert result.shape == (1, x.shape[-1] // 2)
        assert result.dtype == x.dtype
        assert torch.allclose(result, expected)

    def test_swiglu_oai_forward_uses_interleaved_gate_and_up_layout(self):
        x = torch.tensor([[1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]], dtype=torch.float32)
        result = AscendSwigluOAIAndMul.swiglu_oai_forward(x, alpha=1.5, limit=100.0)
        expected = _swiglu_oai_reference(x, alpha=1.5, limit=100.0)
        chunk_based = (x[..., 4:] + 1) * x[..., :4] * torch.sigmoid(x[..., :4] * 1.5)

        assert torch.allclose(result, expected)
        assert not torch.allclose(result, chunk_based)

    def test_swiglu_oai_forward_with_custom_alpha_and_limit_matches_reference(self):
        x = torch.tensor([[9.0, 8.0, -5.0, -9.0]], dtype=torch.float32)
        alpha = 2.0
        limit = 5.0
        result = AscendSwigluOAIAndMul.swiglu_oai_forward(x, alpha=alpha, limit=limit)
        expected = _swiglu_oai_reference(x, alpha=alpha, limit=limit)

        assert torch.allclose(result, expected)

    def test_swiglu_oai_forward_clamps_gate_and_up_values(self):
        x = torch.tensor([[100.0, 100.0, -100.0, -100.0]], dtype=torch.float32)
        result = AscendSwigluOAIAndMul.swiglu_oai_forward(x)
        expected = _swiglu_oai_reference(x)

        assert torch.allclose(result, expected)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_swiglu_oai_forward_large_input(self):
        x = torch.randn(64, 128, dtype=torch.float32)
        result = AscendSwigluOAIAndMul.swiglu_oai_forward(x)
        expected = _swiglu_oai_reference(x)

        assert result.shape == (64, 64)
        assert torch.allclose(result, expected)
        assert not torch.isnan(result).any()


class TestSwiglustepAndMul:
    def test_swiglustep_and_mul_matches_reference_formula(self):
        # last dim 16 => N=8 satisfies the triton kernel's N%(32/elem_size)==0
        # UB alignment on NPU (fp32: N%8==0), so this runs the fused kernel.
        x = torch.tensor(
            [[1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, -8.0, -1.0, -2.0, 3.0, -4.0, -5.0, 6.0, -7.0, 8.0]],
            dtype=torch.float32,
            device="npu",
        )
        result = AscendSwigluStepAndMul.swiglustep_forward(x)
        expected = _swiglustep_and_mul_reference(x.cpu())

        assert result.shape == (1, x.shape[-1] // 2)
        assert result.dtype == x.dtype
        assert torch.allclose(result.cpu(), expected, atol=1e-5)

    def test_swiglustep_and_mul_uses_contiguous_gate_up_layout(self):
        # gate = first half, up = second half (contiguous split via chunk),
        # NOT the interleaved layout used by SwigluOAI.
        x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]],
            dtype=torch.float32,
            device="npu",
        )
        result = AscendSwigluStepAndMul.swiglustep_forward(x, limit=100.0)
        expected = _swiglustep_and_mul_reference(x.cpu(), limit=100.0)
        interleaved = torch.nn.functional.silu(x.cpu()[..., ::2]) * x.cpu()[..., 1::2]

        assert torch.allclose(result.cpu(), expected, atol=1e-5)
        assert not torch.allclose(result.cpu(), interleaved, atol=1e-5)

    def test_swiglustep_and_mul_with_custom_limit_matches_reference(self):
        x = torch.tensor(
            [[9.0, 8.0, -5.0, -9.0, 7.0, -3.0, 6.0, -2.0, 1.0, -9.0, 8.0, -5.0, 4.0, -7.0, 3.0, -1.0]],
            dtype=torch.float32,
            device="npu",
        )
        limit = 3.0
        result = AscendSwigluStepAndMul.swiglustep_forward(x, limit=limit)
        expected = _swiglustep_and_mul_reference(x.cpu(), limit=limit)

        assert torch.allclose(result.cpu(), expected, atol=1e-5)

    def test_swiglustep_and_mul_clamps_gate_and_up_values(self):
        # gate = [100]*8 -> silu(~100) clamped to 7.0;
        # up   = [-100]*8 -> clamped to -7.0  =>  7.0 * -7.0 = -49.0
        x = torch.tensor([[100.0] * 8 + [-100.0] * 8], dtype=torch.float32, device="npu")
        result = AscendSwigluStepAndMul.swiglustep_forward(x)
        expected = _swiglustep_and_mul_reference(x.cpu())

        assert torch.allclose(result.cpu(), expected, atol=1e-5)
        assert torch.allclose(result.cpu(), torch.full((1, 8), -49.0), atol=1e-4)
        assert not torch.isnan(result.cpu()).any()
        assert not torch.isinf(result.cpu()).any()

    def test_swiglustep_and_mul_large_input(self):
        x = torch.randn(64, 128, dtype=torch.float32, device="npu")
        result = AscendSwigluStepAndMul.swiglustep_forward(x)
        expected = _swiglustep_and_mul_reference(x.cpu())

        assert result.shape == (64, 64)
        assert torch.allclose(result.cpu(), expected, atol=1e-5)
        assert not torch.isnan(result.cpu()).any()

    def test_swiglustep_and_mul_validates_limit(self):
        x = torch.tensor([[8.0, 9.0, -2.0, -8.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="requires limit"):
            AscendSwigluStepAndMul.swiglustep_forward(x, limit=None)


class TestActivationNPUPrecision:
    @pytest.mark.parametrize(
        "dtype,atol,rtol",
        [
            (torch.float32, 1e-4, 1e-4),
            (torch.float16, 5e-3, 5e-3),
            (torch.bfloat16, 2e-2, 2e-2),
        ],
    )
    def test_ascend_quick_gelu_matches_cpu_reference_on_npu(self, dtype, atol, rtol, default_vllm_config):
        x_cpu = torch.linspace(-6, 6, steps=128, dtype=torch.float32).reshape(16, 8)
        x_npu = x_cpu.to(dtype=dtype, device="npu")

        result = AscendQuickGELU().forward_oot(x_npu).cpu()
        expected = _quick_gelu_reference(x_cpu.to(dtype=dtype)).float()

        assert torch.allclose(result.float(), expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "dtype,atol,rtol",
        [
            (torch.float32, 1e-4, 1e-4),
            (torch.float16, 5e-3, 5e-3),
            (torch.bfloat16, 2e-2, 2e-2),
        ],
    )
    def test_ascend_silu_and_mul_matches_cpu_reference_on_npu(
        self,
        dtype,
        atol,
        rtol,
        default_vllm_config,
    ):
        x_cpu = torch.randn(16, 16, dtype=torch.float32)
        x_npu = x_cpu.to(dtype=dtype, device="npu")

        result = AscendSiluAndMul().forward_oot(x_npu).cpu()
        expected = _silu_and_mul_reference(x_cpu.to(dtype=dtype)).float()

        assert torch.allclose(result.float(), expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "dtype,atol,rtol",
        [
            (torch.float32, 1e-5, 1e-5),
            (torch.float16, 5e-3, 5e-3),
            (torch.bfloat16, 2e-2, 2e-2),
        ],
    )
    def test_ascend_swiglu_oai_matches_cpu_reference_on_npu(self, dtype, atol, rtol):
        x_cpu = torch.randn(16, 16, dtype=torch.float32) * 4
        x_npu = x_cpu.to(dtype=dtype, device="npu")

        result = AscendSwigluOAIAndMul.swiglu_oai_forward(x_npu).cpu()
        expected = _swiglu_oai_reference(x_cpu.to(dtype=dtype)).float()

        assert result.shape == (16, 8)
        assert torch.allclose(result.float(), expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "dtype,atol,rtol",
        [
            (torch.float32, 1e-5, 1e-5),
            (torch.float16, 5e-3, 5e-3),
            (torch.bfloat16, 2e-2, 2e-2),
        ],
    )
    def test_swiglustep_and_mul_matches_cpu_reference_on_npu(self, dtype, atol, rtol):
        # last dim 32 => N=16 satisfies the triton kernel's N%16==0 UB
        # alignment, so bf16/fp16 exercise the fused kernel (not native).
        x_cpu = torch.randn(16, 32, dtype=torch.float32) * 4
        x_npu = x_cpu.to(dtype=dtype, device="npu")

        result = AscendSwigluStepAndMul.swiglustep_forward(x_npu).cpu()
        expected = _swiglustep_and_mul_reference(x_cpu.to(dtype=dtype)).float()

        assert result.shape == (16, 16)
        assert torch.allclose(result.float(), expected, atol=atol, rtol=rtol)
