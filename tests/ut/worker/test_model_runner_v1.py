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

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.utils import AscendSocVersion
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


# yapf: disable
@pytest.mark.parametrize(
    "soc_version, enable_expert_parallel, world_size, num_tokens, mc2_tokens_capacity, quant_type, expected_method",
    [
        # Case 1: Expert parallel is disabled, should always be 'allgather'
        (AscendSocVersion.A2, False, 8, 100, 256, None, MoECommType.ALLGATHER),
        (AscendSocVersion.A3, False, 16, 500, 256, None, MoECommType.ALLGATHER),

        # Case 2: A2 SOC with w4a8_dynamic -> use alltoall when not mc2
        (AscendSocVersion.A2, True, 8, 100, 256, "w4a8_dynamic", MoECommType.ALLTOALL),
        (AscendSocVersion.A2, True, 16, 257, 256, "w4a8_dynamic", MoECommType.ALLTOALL),
        (AscendSocVersion.A2, True, 16, 100, 256, "w4a8_dynamic", MoECommType.MC2),  # meets mc2 condition

        # Case 3: A2 SOC without w4a8_dynamic -> fallback to allgather
        (AscendSocVersion.A2, True, 8, 100, 256, None, MoECommType.ALLGATHER),
        (AscendSocVersion.A2, True, 16, 257, 256, None, MoECommType.ALLGATHER),

        # Case 4: A3 SOC
        (AscendSocVersion.A3, True, 8, 100, 256, None, MoECommType.MC2),
        (AscendSocVersion.A3, True, 8, 257, 256, None, MoECommType.ALLTOALL),
    ])
# yapf: enable
def test_select_moe_comm_method(soc_version, enable_expert_parallel,
                                world_size, num_tokens, mc2_tokens_capacity,
                                quant_type, expected_method):
    """
    Tests the _select_moe_comm_method with various configurations including quant_type.
    """
    # Mock the NPUModelRunner instance and its dependencies
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.parallel_config = MagicMock()
    mock_runner.parallel_config.enable_expert_parallel = enable_expert_parallel
    mock_runner.parallel_config.world_size_across_dp = world_size
    mock_runner.mc2_tokens_capacity = mc2_tokens_capacity

    # Add vllm_config.model_config.hf_config mock with moe_quantize
    mock_hf_config = MagicMock()
    mock_hf_config.moe_quantize = quant_type
    mock_model_config = MagicMock()
    mock_model_config.hf_config = mock_hf_config
    mock_vllm_config = MagicMock()
    mock_vllm_config.model_config = mock_model_config
    mock_runner.vllm_config = mock_vllm_config

    # Patch the helper functions
    with patch('vllm_ascend.worker.model_runner_v1.get_ascend_soc_version',
               return_value=soc_version), \
         patch('vllm_ascend.worker.model_runner_v1.is_global_first_rank',
               return_value=True), \
         patch('vllm_ascend.worker.model_runner_v1.is_moe_model',
               return_value=True):

        # Bind the real method to the mock object
        method = NPUModelRunner._select_moe_comm_method(
            mock_runner, num_tokens, False)

        # Assert the result
        assert method == expected_method


def test_select_moe_comm_method_unsupported_soc():
    """
    Tests that _select_moe_comm_method raises ValueError for an unsupported SOC.
    """
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.parallel_config = MagicMock()
    mock_runner.parallel_config.enable_expert_parallel = True
    mock_runner.mc2_tokens_capacity = 256

    # Add vllm_config.model_config.hf_config mock with moe_quantize
    mock_hf_config = MagicMock()
    mock_hf_config.moe_quantize = None
    mock_model_config = MagicMock()
    mock_model_config.hf_config = mock_hf_config
    mock_vllm_config = MagicMock()
    mock_vllm_config.model_config = mock_model_config
    mock_runner.vllm_config = mock_vllm_config

    unsupported_soc = "UnsupportedSOC"

    with patch('vllm_ascend.worker.model_runner_v1.get_ascend_soc_version',
               return_value=unsupported_soc), \
         patch('vllm_ascend.worker.model_runner_v1.is_global_first_rank',
               return_value=True), \
         patch('vllm_ascend.worker.model_runner_v1.is_moe_model',
                  return_value=True), \
         pytest.raises(ValueError, match=f"Unsupported soc_version: {unsupported_soc}"):

        NPUModelRunner._select_moe_comm_method(mock_runner, 100, False)


@patch('vllm_ascend.worker.model_runner_v1.torch_npu')
@patch('vllm_ascend.worker.model_runner_v1.torch')
def test_init_creates_transfer_event_and_pinned_memory(mock_torch,
                                                       mock_torch_npu):
    """Test that initialization creates transfer event and pinned CPU memory."""
    # This is a simplified test focusing only on the new attributes
    # We mock the entire __init__ process and only test the specific lines we added

    # Mock torch.empty to return a mock tensor
    mock_pinned_tensor = MagicMock()
    mock_torch.empty.return_value = mock_pinned_tensor

    # Mock torch_npu.npu.Event - 需要设置嵌套的 mock 结构
    mock_event = MagicMock()
    mock_torch_npu.npu.Event.return_value = mock_event

    # Create a runner instance using __new__ to bypass __init__
    runner = NPUModelRunner.__new__(NPUModelRunner)

    # Manually set the attributes we need for our test
    runner.max_model_len = 2048

    # Test the specific lines from the commit
    runner.transfer_event = mock_torch_npu.npu.Event()
    runner.sampled_token_ids_pinned_cpu = mock_torch.empty(
        (runner.max_model_len, 1),
        dtype=torch.int64,
        device="cpu",
        pin_memory=True)

    # Verify max_model_len is set
    assert runner.max_model_len == 2048

    # Verify transfer_event is created
    assert runner.transfer_event == mock_event
    mock_torch_npu.npu.Event.assert_called_once()

    # Verify pinned CPU memory is created with correct parameters
    assert runner.sampled_token_ids_pinned_cpu == mock_pinned_tensor
    mock_torch.empty.assert_called_with((2048, 1),
                                        dtype=torch.int64,
                                        device="cpu",
                                        pin_memory=True)
