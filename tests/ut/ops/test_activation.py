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

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.activation import QuickGELU, SiluAndMul

from vllm_ascend.ops.activation import AscendQuickGELU, AscendSiluAndMul, AscendSwigluOAIAndMul
from vllm_ascend.utils import is_310p as is_310p_hw


@pytest.fixture
def dummy_tensor():
    return torch.randn(4, 8, dtype=torch.float16)


@pytest.fixture
def default_vllm_config():
    mock_config = MagicMock()

    mock_config.compilation_config.dispatch_forward_backend = "eager"

    mock_config.compilation_config.custom_ops = ["all"]

    with set_current_vllm_config(mock_config):
        yield mock_config


@patch("torch_npu.npu_fast_gelu", side_effect=lambda x: x + 1)
def test_QuickGELU_forward(mock_gelu, dummy_tensor, default_vllm_config):
    layer = QuickGELU()
    out = layer.forward(dummy_tensor)

    expected_out = dummy_tensor + 1
    assert torch.allclose(out, expected_out)

    mock_gelu.assert_called_once()


@patch("torch_npu.npu_fast_gelu", side_effect=lambda x: x + 1)
def test_AscendQuickGELU_forward_oot(mock_gelu, dummy_tensor, default_vllm_config):
    layer = AscendQuickGELU()
    out = layer.forward_oot(dummy_tensor)

    assert torch.allclose(out, dummy_tensor + 1)
    mock_gelu.assert_called_once_with(dummy_tensor)


@pytest.mark.skipif(is_310p_hw(), reason="non_310P device unittest case.")
@patch("vllm_ascend.ops.activation.get_weight_prefetch_method", return_value=MagicMock())
@patch("torch_npu.npu_swiglu", side_effect=lambda x: x + 1)
def test_SiluAndMul_forward(
    mock_swiglu,
    mock_get_weight_prefetch_method,
    dummy_tensor,
    default_vllm_config,
):
    layer = SiluAndMul()
    out = layer.forward(dummy_tensor)
    expected_arg = dummy_tensor

    # assert mock_swiglu.call_count == 1
    mock_swiglu.assert_called_once()

    actual_arg = mock_swiglu.call_args[0][0]
    assert torch.allclose(actual_arg, expected_arg), "npu_swiglu called with unexpected input"

    expected_out = dummy_tensor + 1
    assert torch.allclose(out, expected_out)


@pytest.mark.skipif(is_310p_hw(), reason="non_310P device unittest case.")
@patch("vllm_ascend.ops.activation.get_weight_prefetch_method")
@patch("torch_npu.npu_swiglu", side_effect=lambda x: x + 1)
def test_AscendSiluAndMul_forward_oot_prefetch(
    mock_swiglu,
    mock_get_weight_prefetch_method,
    dummy_tensor,
    default_vllm_config,
):
    weight_prefetch_method = MagicMock()
    weight_prefetch_method.MLP_DOWN = "mlp_down"
    mock_get_weight_prefetch_method.return_value = weight_prefetch_method

    layer = AscendSiluAndMul()
    out = layer.forward_oot(dummy_tensor)

    weight_prefetch_method.maybe_prefetch_mlp_weight_preprocess.assert_called_once_with(
        weight_prefetch_method.MLP_DOWN, dummy_tensor
    )
    weight_prefetch_method.maybe_prefetch_mlp_weight_postprocess.assert_called_once_with(out)
    mock_swiglu.assert_called_once_with(dummy_tensor)
    assert torch.allclose(out, dummy_tensor + 1)


@pytest.mark.skipif(not is_310p_hw(), reason="310P device unittest case.")
@patch("torch.nn.functional.silu", side_effect=lambda x: x + 1)
def test_SiluAndMul_forward_310p(
    mock_silu,
    dummy_tensor,
    default_vllm_config,
):
    layer = SiluAndMul()
    out = layer.forward(dummy_tensor)
    h = dummy_tensor.shape[-1] // 2
    expected_arg = dummy_tensor[..., :h]

    # assert mock_silu.call_count == 1
    mock_silu.assert_called_once()

    actual_arg = mock_silu.call_args[0][0]
    assert torch.allclose(actual_arg, expected_arg), "swiglu called with unexpected input"

    expected_out = (dummy_tensor[..., :h] + 1) * dummy_tensor[..., h:]
    assert torch.allclose(out, expected_out)


def _swiglu_oai_reference(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    gate = x[..., ::2].clamp(max=limit)
    up = x[..., 1::2].clamp(min=-limit, max=limit)
    return (up + 1) * gate * torch.sigmoid(gate * alpha)


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

    def test_swiglu_oai_forward_different_dtypes(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(2, 8, dtype=dtype)
            reference_x = x.to(torch.float32)
            result = AscendSwigluOAIAndMul.swiglu_oai_forward(x)
            expected = _swiglu_oai_reference(reference_x).to(dtype)

            assert result.shape == (2, 4)
            assert result.dtype == dtype
            assert torch.allclose(result.float(), expected.float(), atol=1e-3, rtol=1e-3)

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
