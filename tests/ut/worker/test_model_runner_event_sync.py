from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_npu

from tests.ut.base import PytestBase
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerEventSync(PytestBase):
    """Test event synchronization optimization added in commit b6c5ef9"""

    @pytest.fixture
    def mock_vllm_config(self):
        """Create minimal mock VllmConfig for testing"""
        config = MagicMock()
        config.model_config = MagicMock()
        config.model_config.max_model_len = 1024  # Test the new max_model_len attribute
        config.cache_config = MagicMock()
        config.cache_config.block_size = 16
        config.scheduler_config = MagicMock()
        config.scheduler_config.max_num_batched_tokens = 1024
        config.scheduler_config.max_num_seqs = 64
        config.parallel_config = MagicMock()
        config.parallel_config.data_parallel_size = 1
        config.lora_config = None
        config.speculative_config = None
        return config

    @pytest.fixture
    def model_runner(self, mock_vllm_config):
        """Create NPUModelRunner with minimal mocking"""
        device = torch.device("npu:0")
        with patch('vllm_ascend.worker.model_runner_v1.get_ascend_config'), \
             patch('vllm_ascend.worker.model_runner_v1.envs_ascend'), \
             patch('vllm_ascend.worker.model_runner_v1.get_dp_group'), \
             patch('vllm_ascend.worker.model_runner_v1.get_pp_group'), \
             patch('vllm_ascend.worker.model_runner_v1.get_tp_group'), \
             patch('vllm_ascend.worker.model_runner_v1.InputBatch'), \
             patch('vllm_ascend.worker.model_runner_v1.AscendAttentionMetadataBuilder'), \
             patch('vllm_ascend.worker.model_runner_v1.AscendSampler'), \
             patch('vllm_ascend.worker.model_runner_v1.ACLGraphDispatcher'):
            runner = NPUModelRunner(mock_vllm_config, device)
            return runner

    def test_max_model_len_attribute_added(self, model_runner):
        """Test that max_model_len attribute is properly set from config"""
        # This tests the line: self.max_model_len = self.model_config.max_model_len
        assert hasattr(model_runner, 'max_model_len')
        assert model_runner.max_model_len == 1024

    def test_event_sync_attributes_initialized(self, model_runner):
        """Test that event sync attributes are properly initialized"""
        # Test transfer_event is created
        assert hasattr(model_runner, 'transfer_event')
        assert isinstance(model_runner.transfer_event, torch_npu.npu.Event)

        # Test sampled_token_ids_pinned_cpu is created with correct properties
        assert hasattr(model_runner, 'sampled_token_ids_pinned_cpu')
        pinned_tensor = model_runner.sampled_token_ids_pinned_cpu
        assert isinstance(pinned_tensor, torch.Tensor)
        assert pinned_tensor.device.type == 'cpu'
        assert pinned_tensor.is_pinned()
        assert pinned_tensor.shape == (1024, 1)  # (max_model_len, 1)
        assert pinned_tensor.dtype == torch.int64

    def test_to_list_method_functionality(self, model_runner):
        """Test the new _to_list method implementation"""
        # Create test input tensor
        sampled_token_ids = torch.tensor([[1], [2], [3]],
                                         dtype=torch.int64,
                                         device="npu:0")

        # Mock event methods to verify they're called
        model_runner.transfer_event.record = MagicMock()
        model_runner.transfer_event.synchronize = MagicMock()

        # Test the method
        result = model_runner._to_list(sampled_token_ids)

        # Verify correct result
        assert result == [[1], [2], [3]]

        # Verify event synchronization pattern
        model_runner.transfer_event.record.assert_called_once()
        model_runner.transfer_event.synchronize.assert_called_once()

    def test_to_list_uses_pinned_memory_buffer(self, model_runner):
        """Test that _to_list uses the pinned memory buffer correctly"""
        sampled_token_ids = torch.tensor([[5], [10]],
                                         dtype=torch.int64,
                                         device="npu:0")

        # Mock events
        model_runner.transfer_event.record = MagicMock()
        model_runner.transfer_event.synchronize = MagicMock()

        # Mock copy to verify non_blocking=True is used
        original_copy = model_runner.sampled_token_ids_pinned_cpu.copy_
        model_runner.sampled_token_ids_pinned_cpu.copy_ = MagicMock(
            side_effect=original_copy)

        result = model_runner._to_list(sampled_token_ids)

        # Verify copy was called with non_blocking=True
        model_runner.sampled_token_ids_pinned_cpu.copy_.assert_called_once()
        _, kwargs = model_runner.sampled_token_ids_pinned_cpu.copy_.call_args
        assert kwargs.get('non_blocking')

        assert result == [[5], [10]]
