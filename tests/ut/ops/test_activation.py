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

from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.layers.activation import QuickGELU, SiluAndMul


@pytest.fixture
def dummy_tensor():
    return torch.randn(4, 8, dtype=torch.float16)


@patch("torch_npu.npu_fast_gelu", side_effect=lambda x: x + 1)
def test_QuickGELU_forward(mock_gelu, dummy_tensor):
    layer = QuickGELU()
    out = layer.forward(dummy_tensor)

    expected_out = dummy_tensor + 1
    assert torch.allclose(out, expected_out)

    mock_gelu.assert_called_once()


@pytest.mark.parametrize("is_310p_return", [True, False])
@patch("torch_npu.npu_swiglu", side_effect=lambda x: x + 1)
@patch("torch.ops.vllm.maybe_wait_prefetch_done", side_effect=lambda x: None)
@patch("torch.ops.vllm.maybe_prefetch_mlp_down_proj",
       side_effect=lambda x: None)
def test_SiluAndMul_forward(mock_maybe_prefetch_mlp_down_proj,
                            mock_maybe_wait_prefetch_done, mock_swiglu,
                            is_310p_return, dummy_tensor):

    with patch("vllm_ascend.utils.is_310p", return_value=is_310p_return):
        layer = SiluAndMul()
        out = layer.forward(dummy_tensor)

        if is_310p_return:
            expected_arg = dummy_tensor.to(torch.float32)
        else:
            expected_arg = dummy_tensor

        # assert mock_maybe_prefetch_mlp_down_proj.call_count == 1
        mock_maybe_prefetch_mlp_down_proj.assert_called_once()

        # assert mock_swiglu.call_count == 1
        mock_swiglu.assert_called_once()

        # assert mock_maybe_wait_prefetch_done.call_count == 1
        mock_maybe_wait_prefetch_done.assert_called_once()

        actual_arg = mock_swiglu.call_args[0][0]
        assert torch.allclose(
            actual_arg,
            expected_arg), "npu_swiglu called with unexpected input"

        expected_out = dummy_tensor + 1
        assert torch.allclose(out, expected_out)
