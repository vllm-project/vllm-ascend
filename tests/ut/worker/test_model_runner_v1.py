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

from vllm_ascend.utils import AscendSocVersion
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


# yapf: disable
@pytest.mark.parametrize(
    "soc_version, enable_expert_parallel, world_size, num_tokens, mc2_tokens_capacity, expected_method",
    [
        # Case 1: Expert parallel is disabled, should always be 'allgather'
        (AscendSocVersion.A2, False, 8, 100, 256, "allgather"),
        (AscendSocVersion.A3, False, 16, 500, 256, "allgather"),

        # Case 2: A2 SOC
        # 2.1: MC2 conditions met (tokens <= capacity, world_size >= 16)
        (AscendSocVersion.A2, True, 16, 100, 256, "mc2"),
        (AscendSocVersion.A2, True, 32, 256, 256, "mc2"),
        # 2.2: MC2 token capacity exceeded
        (AscendSocVersion.A2, True, 16, 257, 256, "allgather"),
        # 2.3: MC2 world size not met
        (AscendSocVersion.A2, True, 8, 100, 256, "allgather"),
        (AscendSocVersion.A2, True, 15, 100, 256, "allgather"),

        # Case 3: A3 SOC
        # 3.1: MC2 condition met (tokens <= capacity)
        (AscendSocVersion.A3, True, 8, 100, 256, "mc2"),
        (AscendSocVersion.A3, True, 16, 256, 256, "mc2"),
        # 3.2: MC2 token capacity exceeded
        (AscendSocVersion.A3, True, 8, 257, 256, "alltoall"),
        (AscendSocVersion.A3, True, 16, 500, 256, "alltoall"),

    ])
# yapf: enable
def test_select_moe_comm_method(soc_version, enable_expert_parallel,
                                world_size, num_tokens, mc2_tokens_capacity,
                                expected_method):
    """
    Tests the _select_moe_comm_method with various configurations.
    """
    # Mock the NPUModelRunner instance and its dependencies
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.parallel_config = MagicMock()
    mock_runner.parallel_config.enable_expert_parallel = enable_expert_parallel
    mock_runner.parallel_config.world_size = world_size
    mock_runner.mc2_tokens_capacity = mc2_tokens_capacity

    # Patch the helper functions
    with patch('vllm_ascend.worker.model_runner_v1.get_ascend_soc_version',
               return_value=soc_version), \
         patch('vllm_ascend.worker.model_runner_v1.is_global_first_rank',
               return_value=True):

        # Call the method under test
        method = NPUModelRunner._select_moe_comm_method(
            mock_runner, num_tokens)

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

    unsupported_soc = "UnsupportedSOC"

    with patch('vllm_ascend.worker.model_runner_v1.get_ascend_soc_version',
               return_value=unsupported_soc), \
         patch('vllm_ascend.worker.model_runner_v1.is_global_first_rank',
               return_value=True), \
         pytest.raises(ValueError, match=f"Unsupported soc_version: {unsupported_soc}"):

        NPUModelRunner._select_moe_comm_method(mock_runner, 100)


class TestNPUModelRunnerInit:
    """Tests for NPUModelRunner initialization including new event sync feature."""

    def test_patch_torch_npu_structure(self):
        """Test to verify torch_npu mock structure works correctly."""
        with patch('vllm_ascend.worker.model_runner_v1.torch_npu'
                   ) as mock_torch_npu:
            # 设置 mock 的嵌套结构
            mock_event = MagicMock()
            mock_torch_npu.npu.Event.return_value = mock_event

            # 验证 mock 结构
            assert mock_torch_npu.npu.Event() == mock_event
            mock_torch_npu.npu.Event.assert_called_once()

    @patch('vllm_ascend.worker.model_runner_v1.torch_npu')
    @patch('vllm_ascend.worker.model_runner_v1.torch')
    def test_init_creates_transfer_event_and_pinned_memory(
            self, mock_torch, mock_torch_npu):
        """Test that initialization creates transfer event and pinned CPU memory."""
        # Mock torch.empty to return a mock tensor
        mock_pinned_tensor = MagicMock()
        mock_torch.empty.return_value = mock_pinned_tensor

        # Mock torch_npu.npu.Event - 需要设置嵌套的 mock 结构
        mock_event = MagicMock()
        mock_torch_npu.npu.Event.return_value = mock_event

        # Create mock vllm_config with necessary attributes
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 2048
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_batched_tokens = 1024
        mock_vllm_config.scheduler_config.max_num_seqs = 32
        mock_vllm_config.parallel_config.data_parallel_size = 1
        mock_vllm_config.parallel_config.pipeline_parallel_size = 1
        mock_vllm_config.parallel_config.tensor_parallel_size = 1
        mock_vllm_config.parallel_config.world_size = 1
        mock_vllm_config.parallel_config.enable_expert_parallel = False
        mock_vllm_config.speculative_config = None
        mock_vllm_config.observability_config = None
        mock_vllm_config.lora_config = None
        mock_vllm_config.prompt_adapter_config = None
        mock_vllm_config.decoding_config = None

        # Mock other required attributes
        mock_vllm_config.cache_config.enable_prefix_caching = False
        mock_vllm_config.model_config.dtype = torch.float16

        with patch.multiple(
            'vllm_ascend.worker.model_runner_v1',
            get_ascend_soc_version=MagicMock(return_value=AscendSocVersion.A2),
            is_global_first_rank=MagicMock(return_value=True),
            _check_env_vars_for_multiprocess=MagicMock(),
            STR_DTYPE_TO_TORCH_DTYPE={'float16': torch.float16},
        ), \
        patch('vllm_ascend.worker.model_runner_v1.VocabParallelEmbedding'), \
        patch('vllm_ascend.worker.model_runner_v1.ParallelLMHead'), \
        patch('vllm_ascend.worker.model_runner_v1.get_model'), \
        patch('vllm_ascend.worker.model_runner_v1.CudagraphDispatcher'):

            # Create NPUModelRunner instance
            runner = NPUModelRunner(vllm_config=mock_vllm_config)

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


class TestNPUModelRunnerToList:
    """Tests for the _to_list method in NPUModelRunner."""

    def test_to_list_converts_tensor_correctly(self):
        """Test that _to_list correctly converts tensor to list using event sync."""
        # Create a mock runner with required attributes
        mock_runner = MagicMock(spec=NPUModelRunner)

        # Mock the pinned CPU tensor
        mock_pinned_tensor = MagicMock()
        mock_pinned_tensor.tolist.return_value = [[1], [2], [3]]
        mock_runner.sampled_token_ids_pinned_cpu = MagicMock()
        mock_runner.sampled_token_ids_pinned_cpu.__getitem__.return_value = mock_pinned_tensor

        # Mock the transfer event
        mock_event = MagicMock()
        mock_runner.transfer_event = mock_event

        # Create a mock input tensor
        mock_input_tensor = MagicMock()
        mock_input_tensor.shape = [3, 1]  # 3 tokens, 1 dimension

        # Call the method
        result = NPUModelRunner._to_list(mock_runner, mock_input_tensor)

        # Verify the result
        assert result == [[1], [2], [3]]

        # Verify the pinned tensor slice was accessed correctly
        mock_runner.sampled_token_ids_pinned_cpu.__getitem__.assert_called_once_with(
            slice(None, 3))

        # Verify copy operation was called
        mock_pinned_tensor.copy_.assert_called_once_with(mock_input_tensor,
                                                         non_blocking=True)

        # Verify event operations were called
        mock_event.record.assert_called_once()
        mock_event.synchronize.assert_called_once()

        # Verify tolist was called on the pinned tensor
        mock_pinned_tensor.tolist.assert_called_once()

    def test_to_list_handles_different_tensor_shapes(self):
        """Test that _to_list handles tensors of different shapes correctly."""
        # Create a mock runner
        mock_runner = MagicMock(spec=NPUModelRunner)

        # Mock pinned tensor for different sizes
        mock_pinned_tensor = MagicMock()
        mock_pinned_tensor.tolist.return_value = [[10], [20]]
        mock_runner.sampled_token_ids_pinned_cpu = MagicMock()
        mock_runner.sampled_token_ids_pinned_cpu.__getitem__.return_value = mock_pinned_tensor

        # Mock the transfer event
        mock_event = MagicMock()
        mock_runner.transfer_event = mock_event

        # Test with a smaller tensor (2 tokens)
        mock_input_tensor = MagicMock()
        mock_input_tensor.shape = [2, 1]

        result = NPUModelRunner._to_list(mock_runner, mock_input_tensor)

        # Verify the correct slice was used
        mock_runner.sampled_token_ids_pinned_cpu.__getitem__.assert_called_with(
            slice(None, 2))
        assert result == [[10], [20]]

    def test_to_list_event_synchronization_flow(self):
        """Test that _to_list follows the correct event synchronization flow."""
        # Create a mock runner
        mock_runner = MagicMock(spec=NPUModelRunner)

        # Mock pinned tensor
        mock_pinned_tensor = MagicMock()
        mock_pinned_tensor.tolist.return_value = [[42]]
        mock_runner.sampled_token_ids_pinned_cpu = MagicMock()
        mock_runner.sampled_token_ids_pinned_cpu.__getitem__.return_value = mock_pinned_tensor

        # Mock the transfer event
        mock_event = MagicMock()
        mock_runner.transfer_event = mock_event

        # Create a mock input tensor
        mock_input_tensor = MagicMock()
        mock_input_tensor.shape = [1, 1]

        # Call the method
        NPUModelRunner._to_list(mock_runner, mock_input_tensor)

        # Verify the order of operations: copy -> record -> synchronize -> tolist
        # We can't easily verify exact order without more complex mocking,
        # but we can verify all operations were called
        mock_pinned_tensor.copy_.assert_called_once()
        mock_event.record.assert_called_once()
        mock_event.synchronize.assert_called_once()
        mock_pinned_tensor.tolist.assert_called_once()
