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
from vllm.model_executor.layers.activation import QuickGELU, SiluAndMul, SiluAndMulWithClamp

from vllm_ascend.ops.activation import AscendSiluAndMulWithClamp


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


@patch("torch_npu.npu_swiglu", side_effect=lambda x: x + 1)
def test_SiluAndMul_forward(
    mock_swiglu,
    dummy_tensor,
    default_vllm_config,
):
    layer = SiluAndMul()
    out = layer.forward(dummy_tensor)
    expected_arg = dummy_tensor

    mock_swiglu.assert_called_once()

    actual_arg = mock_swiglu.call_args[0][0]
    assert torch.allclose(actual_arg, expected_arg), "npu_swiglu called with unexpected input"

    expected_out = dummy_tensor + 1
    assert torch.allclose(out, expected_out)


@patch("torch_npu.npu_clipped_swiglu", side_effect=lambda x, **kwargs: x[..., : x.shape[-1] // 2] + 1)
def test_SiluAndMulWithClamp_forward(
    mock_clipped_swiglu,
    dummy_tensor,
    default_vllm_config,
):
    layer = SiluAndMulWithClamp(swiglu_limit=7.0, alpha=1.702, beta=1.0)

    assert isinstance(layer, AscendSiluAndMulWithClamp)

    out = layer.forward(dummy_tensor)

    mock_clipped_swiglu.assert_called_once()

    actual_arg = mock_clipped_swiglu.call_args[0][0]
    assert torch.allclose(actual_arg, dummy_tensor), "npu_clipped_swiglu called with unexpected input"
    assert mock_clipped_swiglu.call_args[1] == {
        "interleaved": False,
        "alpha": 1.702,
        "limit": 7.0,
        "bias": 1.0,
    }

    expected_out = dummy_tensor[..., : dummy_tensor.shape[-1] // 2] + 1
    assert torch.allclose(out, expected_out)
