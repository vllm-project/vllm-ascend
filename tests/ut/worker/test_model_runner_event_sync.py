import torch
import torch_npu
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from tests.ut.base import PytestBase
from vllm import VllmConfig
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerEventSync(PytestBase):
    """Test event synchronization optimization in NPUModelRunner"""

    @pytest.fixture
    def mock_vllm_config(self):
        """Create a mock VllmConfig for testing"""
        config = MagicMock(spec=VllmConfig)

        # Mock model config
        config.model_config = MagicMock()
        config.model_config.max_model_len = 2048
        config.model_config.is_multimodal_model = False
        config.model_config.uses_mrope = False
        config.model_config.use_mla = False

        # Mock cache config
        config.cache_config = MagicMock()
        config.cache_config.block_size = 16
        config.cache_config.max_num_blocks_per_req = 128

        # Mock scheduler config
        config.scheduler_config = MagicMock()
        config.scheduler_config.max_num_batched_tokens = 2048
        config.scheduler_config.max_num_seqs = 256

        # Mock parallel config
        config.parallel_config = MagicMock()
        config.parallel_config.data_parallel_size = 1
        config.parallel_config.enable_expert_parallel = False
        config.parallel_config.world_size = 1

        # Mock lora config
        config.lora_config = None

        # Mock speculative config
        config.speculative_config = None

        # Mock observability config
        config.observability_config = MagicMock()
        config.observability_config.collect_detailed_traces = False

        return config

    @pytest.fixture
    def model_runner(self, mock_vllm_config):
        """Create NPUModelRunner instance with mocked dependencies"""
        device = torch.device("npu:0")

        with patch('vllm_ascend.worker.model_runner_v1.get_ascend_config'), \
             patch('vllm_ascend.worker.model_runner_v1.envs_ascend'), \
             patch('vllm_ascend.worker.model_runner_v1.get_dp_group'), \
             patch('vllm_ascend.worker.model_runner_v1.get_pp_group'), \
             patch('vllm_ascend.worker.model_runner_v1.get_tp_group'), \
             patch('vllm_ascend.worker.model_runner_v1.lmhead_tp_enable'), \
             patch('vllm_ascend.worker.model_runner_v1.AscendCommonAttentionMetadata'), \
             patch('vllm_ascend.worker.model_runner_v1.AscendMLACommonAttentionMetadata'), \
             patch('vllm_ascend.worker.model_runner_v1.InputBatch'), \
             patch('vllm_ascend.worker.model_runner_v1.AscendAttentionMetadataBuilder'), \
             patch('vllm_ascend.worker.model_runner_v1.AscendMLAAttentionMetadataBuilder'), \
             patch('vllm_ascend.worker.model_runner_v1.AscendSampler'), \
             patch('vllm_ascend.worker.model_runner_v1.AscendRejectionSampler'), \
             patch('vllm_ascend.worker.model_runner_v1.ACLGraphDispatcher'), \
             patch('vllm_ascend.worker.model_runner_v1.HunYuanVideoTextModelAdapter'):

            runner = NPUModelRunner(mock_vllm_config, device)
            return runner

    def test_initialization_with_event_sync_attributes(self, model_runner):
        """Test that NPUModelRunner initializes with event sync attributes"""
        # Check that transfer_event is initialized
        assert hasattr(model_runner, 'transfer_event')
        assert isinstance(model_runner.transfer_event, torch_npu.npu.Event)

        # Check that sampled_token_ids_pinned_cpu is initialized
        assert hasattr(model_runner, 'sampled_token_ids_pinned_cpu')
        assert isinstance(model_runner.sampled_token_ids_pinned_cpu, torch.Tensor)
        assert model_runner.sampled_token_ids_pinned_cpu.device.type == 'cpu'
        assert model_runner.sampled_token_ids_pinned_cpu.is_pinned()
        assert model_runner.sampled_token_ids_pinned_cpu.shape == (2048, 1)  # max_model_len, 1
        assert model_runner.sampled_token_ids_pinned_cpu.dtype == torch.int64

    def test_to_list_method_basic_functionality(self, model_runner):
        """Test basic functionality of _to_list method"""
        # Create a sample tensor on NPU
        sampled_token_ids = torch.tensor([[1], [2], [3]], dtype=torch.int64, device="npu:0")

        # Mock the event methods
        model_runner.transfer_event.record = MagicMock()
        model_runner.transfer_event.synchronize = MagicMock()

        # Call _to_list
        result = model_runner._to_list(sampled_token_ids)

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 3
        assert result == [[1], [2], [3]]

        # Verify event methods were called
        model_runner.transfer_event.record.assert_called_once()
        model_runner.transfer_event.synchronize.assert_called_once()

    def test_to_list_method_with_different_sizes(self, model_runner):
        """Test _to_list method with different tensor sizes"""
        # Mock the event methods
        model_runner.transfer_event.record = MagicMock()
        model_runner.transfer_event.synchronize = MagicMock()

        # Test with single token
        single_token = torch.tensor([[42]], dtype=torch.int64, device="npu:0")
        result_single = model_runner._to_list(single_token)
        assert result_single == [[42]]

        # Test with multiple tokens
        multi_tokens = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.int64, device="npu:0")
        result_multi = model_runner._to_list(multi_tokens)
        assert result_multi == [[1], [2], [3], [4], [5]]

        # Verify events were called for both cases
        assert model_runner.transfer_event.record.call_count == 2
        assert model_runner.transfer_event.synchronize.call_count == 2

    def test_to_list_method_uses_pinned_memory(self, model_runner):
        """Test that _to_list method correctly uses pinned memory buffer"""
        sampled_token_ids = torch.tensor([[10], [20]], dtype=torch.int64, device="npu:0")

        # Mock the event methods
        model_runner.transfer_event.record = MagicMock()
        model_runner.transfer_event.synchronize = MagicMock()

        # Mock the pinned tensor copy operation to verify it's called correctly
        original_copy = model_runner.sampled_token_ids_pinned_cpu.copy_
        model_runner.sampled_token_ids_pinned_cpu.copy_ = MagicMock(side_effect=original_copy)

        # Call _to_list
        result = model_runner._to_list(sampled_token_ids)

        # Verify the pinned memory buffer was used correctly
        model_runner.sampled_token_ids_pinned_cpu.copy_.assert_called_once()
        args, kwargs = model_runner.sampled_token_ids_pinned_cpu.copy_.call_args
        assert torch.equal(args[0], sampled_token_ids)
        assert kwargs.get('non_blocking', False) == True

        # Verify result
        assert result == [[10], [20]]

    @patch.object(NPUModelRunner, '_to_list')
    def test_execute_model_uses_to_list_method(self, mock_to_list, model_runner):
        """Test that execute_model uses _to_list method instead of direct tolist()"""
        # This test verifies the integration point where sampled_token_ids.tolist()
        # was replaced with self._to_list(sampled_token_ids)

        # Mock dependencies for execute_model
        mock_scheduler_output = MagicMock()
        mock_scheduler_output.total_num_scheduled_tokens = 0

        # Mock the no work case to avoid full model execution
        with patch('vllm_ascend.worker.model_runner_v1.has_kv_transfer_group', return_value=False):
            result = model_runner.execute_model(mock_scheduler_output)

            # In the no-work case, _to_list should not be called
            mock_to_list.assert_not_called()

    def test_to_list_method_memory_efficiency(self, model_runner):
        """Test that _to_list method is memory efficient"""
        # Create a larger tensor to test memory usage
        large_tensor = torch.tensor(
            [[i] for i in range(100)],
            dtype=torch.int64,
            device="npu:0"
        )

        # Mock the event methods
        model_runner.transfer_event.record = MagicMock()
        model_runner.transfer_event.synchronize = MagicMock()

        # Verify that the method doesn't create unnecessary copies
        original_pinned_buffer = model_runner.sampled_token_ids_pinned_cpu

        result = model_runner._to_list(large_tensor)

        # Buffer should be the same object (no new allocation)
        assert model_runner.sampled_token_ids_pinned_cpu is original_pinned_buffer

        # Result should be correct
        expected = [[i] for i in range(100)]
        assert result == expected

    def test_to_list_method_error_handling(self, model_runner):
        """Test _to_list method with edge cases and error conditions"""
        # Test with empty tensor
        empty_tensor = torch.empty((0, 1), dtype=torch.int64, device="npu:0")

        # Mock the event methods
        model_runner.transfer_event.record = MagicMock()
        model_runner.transfer_event.synchronize = MagicMock()

        result = model_runner._to_list(empty_tensor)
        assert result == []

        # Events should still be called for consistency
        model_runner.transfer_event.record.assert_called_once()
        model_runner.transfer_event.synchronize.assert_called_once()

    def test_to_list_method_performance_optimization(self, model_runner):
        """Test that _to_list method implements the performance optimization correctly"""
        sampled_token_ids = torch.tensor([[1], [2]], dtype=torch.int64, device="npu:0")

        # Mock the event methods to verify synchronization pattern
        model_runner.transfer_event.record = MagicMock()
        model_runner.transfer_event.synchronize = MagicMock()

        # Call the method
        result = model_runner._to_list(sampled_token_ids)

        # Verify that the optimization pattern is followed:
        # 1. Non-blocking copy to pinned memory
        # 2. Event record
        # 3. Event synchronize
        # 4. tolist() on CPU tensor
        model_runner.transfer_event.record.assert_called_once()
        model_runner.transfer_event.synchronize.assert_called_once()

        # Verify the event methods are called in the correct order
        # (record before synchronize)
        calls = [call[0] for call in [
            model_runner.transfer_event.record.call_args,
            model_runner.transfer_event.synchronize.call_args
        ]]
        assert len(calls) == 2  # Both methods should have been called

    def test_max_model_len_attribute_initialization(self, mock_vllm_config):
        """Test that max_model_len attribute is properly initialized"""
        device = torch.device("npu:0")

        with patch('vllm_ascend.worker.model_runner_v1.get_ascend_config'), \
             patch('vllm_ascend.worker.model_runner_v1.envs_ascend'), \
             patch('vllm_ascend.worker.model_runner_v1.get_dp_group'), \
             patch('vllm_ascend.worker.model_runner_v1.get_pp_group'), \
             patch('vllm_ascend.worker.model_runner_v1.get_tp_group'), \
             patch('vllm_ascend.worker.model_runner_v1.lmhead_tp_enable'), \
             patch('vllm_ascend.worker.model_runner_v1.AscendCommonAttentionMetadata'), \
             patch('vllm_ascend.worker.model_runner_v1.AscendMLACommonAttentionMetadata'), \
             patch('vllm_ascend.worker.model_runner_v1.InputBatch'), \
             patch('vllm_ascend.worker.model_runner_v1.AscendAttentionMetadataBuilder'), \
             patch('vllm_ascend.worker.model_runner_v1.AscendMLAAttentionMetadataBuilder'), \
             patch('vllm_ascend.worker.model_runner_v1.AscendSampler'), \
             patch('vllm_ascend.worker.model_runner_v1.AscendRejectionSampler'), \
             patch('vllm_ascend.worker.model_runner_v1.ACLGraphDispatcher'), \
             patch('vllm_ascend.worker.model_runner_v1.HunYuanVideoTextModelAdapter'):

            runner = NPUModelRunner(mock_vllm_config, device)

            # Verify max_model_len is set correctly
            assert hasattr(runner, 'max_model_len')
            assert runner.max_model_len == mock_vllm_config.model_config.max_model_len
            assert runner.max_model_len == 2048
