from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import (CacheConfig, CompilationConfig, CUDAGraphMode,
                         ModelConfig, SchedulerConfig, SpeculativeConfig,
                         VllmConfig)
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.mtp_proposer import MtpProposer


class TestMtpProposer:

    @pytest.fixture(autouse=True)
    def patch_supports_multimodal_inputs(self):
        with patch(
                "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs"
        ):
            yield

    @pytest.fixture
    def vllm_config(self):
        config = MagicMock(spec=VllmConfig)
        config.additional_config = None
        config.speculative_config = MagicMock(spec=SpeculativeConfig)
        config.speculative_config.num_speculative_tokens = 2
        config.speculative_config.method = "deepseek_mtp"
        config.speculative_config.draft_model_config = MagicMock()
        config.speculative_config.draft_model_config.get_hidden_size.return_value = 4096
        config.speculative_config.speculative_token_tree = str([
            (i + 1) * (0, ) for i in range(2)
        ])

        config.model_config = MagicMock(spec=ModelConfig)
        config.model_config.dtype = torch.float16
        config.model_config.max_model_len = 2048
        config.model_config.uses_mrope = False
        config.model_config.hf_text_config = None

        config.load_config = None

        config.cache_config = MagicMock(spec=CacheConfig)
        config.cache_config.block_size = 16

        config.scheduler_config = MagicMock(spec=SchedulerConfig)
        config.scheduler_config.max_num_batched_tokens = 4096
        config.scheduler_config.max_num_seqs = 256

        config.compilation_config = MagicMock(spec=CompilationConfig)
        config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]
        config.compilation_config.static_forward_context = dict()

        config.device_config = MagicMock()
        config.device_config.device = torch.device("cpu")
        init_ascend_config(config)
        return config

    @pytest.fixture
    def runner(self):
        runner = MagicMock()
        runner.pcp_size = 1
        runner.dcp_size = 1
        runner.pcp_rank = 0
        runner.max_num_tokens = 4096
        runner.max_num_reqs = 256
        runner._use_aclgraph.return_value = False
        runner.reserved_mc2_mask = None
        return runner

    @patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
    def test_init(self, mock_cpu_gpu_buffer, vllm_config, runner):
        mock_buffer_instance = MagicMock()
        mock_cpu_gpu_buffer.return_value = mock_buffer_instance

        # Test basic initialization
        proposer = MtpProposer(vllm_config, torch.device("cpu"), runner)

        assert proposer.vllm_config == vllm_config
        assert proposer.device == torch.device("cpu")
        assert proposer.dtype == torch.float16
        assert proposer.num_speculative_tokens == 2
        assert proposer.hidden_size == 4096

        # Test with mrope enabled
        assert hasattr(proposer, "positions")
        assert not hasattr(proposer, "mrope_positions")
        assert proposer.use_sparse is False

    @patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
    def test_init_with_aclgraph(self, mock_cpu_gpu_buffer, vllm_config,
                                runner):
        mock_buffer_instance = MagicMock()
        mock_cpu_gpu_buffer.return_value = mock_buffer_instance
        runner._use_aclgraph.return_value = True
        proposer = MtpProposer(vllm_config, torch.device("cpu"), runner)

        assert proposer.use_aclgraph is True

    @patch("vllm_ascend.spec_decode.mtp_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.mtp_proposer.set_ascend_forward_context")
    @patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
    def test_dummy_run(self, mock_cpu_gpu_buffer, mock_set_context,
                       mock_get_forward_context, vllm_config, runner):
        mock_buffer_instance = MagicMock()
        mock_cpu_gpu_buffer.return_value = mock_buffer_instance
        proposer = MtpProposer(vllm_config, torch.device("cpu"), runner)
        proposer.model = MagicMock()
        proposer.enable_shared_expert_dp = False
        runner._sync_metadata_across_dp.return_value = (8, 8, False)

        mock_get_forward_context = MagicMock()
        mock_get_forward_context.cudagraph_runtime_mode = None
        mock_get_forward_context.capturing = True
        # Execute
        proposer.dummy_run(8)

        # Verify
        runner._sync_metadata_across_dp.assert_called_once()
        mock_set_context.assert_called()

        # Check that model was called correct number of times
        assert proposer.model.call_count == vllm_config.speculative_config.num_speculative_tokens

    @patch("vllm_ascend.spec_decode.mtp_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.mtp_proposer.set_ascend_forward_context")
    @patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
    def test_dummy_run_full_graph(self, mock_cpu_gpu_buffer, mock_set_context,
                                  mock_get_forward_context, vllm_config,
                                  runner):
        # Setup
        mock_buffer_instance = MagicMock()
        mock_cpu_gpu_buffer.return_value = mock_buffer_instance
        proposer = MtpProposer(vllm_config, torch.device("cpu"), runner)
        proposer.enable_shared_expert_dp = False
        proposer.model = MagicMock()
        runner._sync_metadata_across_dp.return_value = (8, 8, False)
        runner.attn_groups = []

        mock_get_forward_context = MagicMock()
        mock_get_forward_context.cudagraph_runtime_mode = None
        mock_get_forward_context.capturing = True
        # Execute
        proposer.dummy_run(num_tokens=8,
                           num_reqs=5,
                           aclgraph_runtime_mode=CUDAGraphMode.FULL)

        # Verify
        runner._sync_metadata_across_dp.assert_called_once()
        mock_set_context.assert_called()

        # Check that model was called correct number of times
        assert proposer.model.call_count == vllm_config.speculative_config.num_speculative_tokens

    @patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
    def test_prepare_next_token_ids_cpu(self, mock_cpu_gpu_buffer):
        mock_buffer_instance = MagicMock()
        mock_cpu_gpu_buffer.return_value = mock_buffer_instance
        sampled_token_ids = [[10, 20, 30], [40, 50], [60]]

        mock_gpu_batch = MagicMock()
        mock_gpu_batch.req_ids = ["req1", "req2", "req3"]
        mock_num_scheduled = {"req1": 0, "req2": 0, "req3": 0}

        proposer = MagicMock(spec=MtpProposer)
        proposer.input_ids = MagicMock(device=torch.device("cpu"))
        proposer.prepare_next_token_ids_cpu = MtpProposer.prepare_next_token_ids_cpu.__get__(
            proposer)
        result = proposer.prepare_next_token_ids_cpu(
            sampled_token_ids=sampled_token_ids,
            requests={},
            gpu_input_batch=mock_gpu_batch,
            num_scheduled_tokens=mock_num_scheduled)

        assert torch.all(
            result == torch.tensor([30, 50, 60], dtype=torch.int32))

    @patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
    def test_prepare_next_token_ids_padded(self, mock_cpu_gpu_buffer):
        mock_common_attn_metadata = MagicMock(spec=CommonAttentionMetadata)
        mock_common_attn_metadata.seq_lens_cpu = torch.tensor(
            [10, 8, 5, 12], dtype=torch.int32)
        mock_sampled_token_ids = torch.tensor([
            [101, 102, 103],
            [201, -1, 203],
            [-1, -1, -1],
            [301, 10000, 303],
        ],
                                              dtype=torch.int32,
                                              device=torch.device("cpu"))

        mock_requests = {}  # dict[str, CachedRequestState]
        req0 = MagicMock(spec=CachedRequestState)
        req0.get_token_id = MagicMock(return_value=1000)
        mock_requests["req_0"] = req0

        req1 = MagicMock(spec=CachedRequestState)
        req1.get_token_id = MagicMock(return_value=2000)
        mock_requests["req_1"] = req1

        req2 = MagicMock(spec=CachedRequestState)
        req2.get_token_id = MagicMock(return_value=3000)
        mock_requests["req_2"] = req2

        req3 = MagicMock(spec=CachedRequestState)
        req3.get_token_id = MagicMock(return_value=4000)
        mock_requests["req_3"] = req3

        mock_gpu_input_batch = MagicMock(spec=InputBatch)
        mock_gpu_input_batch.num_reqs = 4
        mock_gpu_input_batch.req_ids = ["req_0", "req_1", "req_2", "req_3"]
        mock_gpu_input_batch.vocab_size = 5000

        mock_backup = MagicMock()
        mock_backup.np = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
        mock_backup.gpu = torch.tensor([1, 2, 3, 4, 5, 6, 7],
                                       dtype=torch.int32)
        mock_backup.copy_to_gpu = MagicMock()
        mock_cpu_gpu_buffer.return_value = mock_backup

        proposer = MagicMock(spec=MtpProposer)
        proposer.backup_next_token_ids = mock_backup
        proposer.input_ids = MagicMock(device=torch.device("cpu"))
        proposer.prepare_next_token_ids_padded = MtpProposer.prepare_next_token_ids_padded.__get__(
            proposer)

        discard_request_indices = torch.tensor([1, 3], dtype=torch.int64)
        num_discarded_requests = 2

        next_token_ids, valid_sampled_tokens_count = proposer.prepare_next_token_ids_padded(
            common_attn_metadata=mock_common_attn_metadata,
            sampled_token_ids=mock_sampled_token_ids,
            requests=mock_requests,
            gpu_input_batch=mock_gpu_input_batch,
            discard_request_indices=discard_request_indices,
            num_discarded_requests=num_discarded_requests)

        mock_backup_output = proposer.backup_next_token_ids

        expected_backup_cpu = np.array(
            [1000, 2000, 3000, 4000, 0, 0, 0, 0, 0, 0])
        assert np.array_equal(mock_backup_output.np[:4],
                              expected_backup_cpu[:4])
        mock_backup_output.copy_to_gpu.assert_called_once_with(4)

        modified_sampled = mock_sampled_token_ids.clone()
        modified_sampled.index_fill_(
            0, discard_request_indices[:num_discarded_requests], -1)
        assert valid_sampled_tokens_count[1].item() == 0
        assert valid_sampled_tokens_count[3].item() == 0

        expected_valid_count = torch.tensor([3, 0, 0, 0], dtype=torch.int32)
        assert torch.equal(valid_sampled_tokens_count, expected_valid_count)

        expected_next_tokens = torch.tensor([103, 2, 3, 4],
                                            dtype=torch.int32,
                                            device=torch.device("cpu"))
        assert torch.equal(next_token_ids, expected_next_tokens)

    @patch("vllm_ascend.spec_decode.eagle_proposer.HAS_TRITON", False)
    @patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
    def test_prepare_inputs_padded(self, mock_cpu_gpu_buffer):
        mock_buffer_instance = MagicMock()
        mock_cpu_gpu_buffer.return_value = mock_buffer_instance

        mock_common_attn_metadata = MagicMock(spec=CommonAttentionMetadata)
        mock_common_attn_metadata.query_start_loc_cpu = torch.tensor(
            [0, 8, 16, 24], dtype=torch.int32)
        mock_common_attn_metadata.seq_lens_cpu = torch.tensor(
            [8, 8, 8], dtype=torch.int32)
        mock_common_attn_metadata.num_input_tokens = 3
        mock_common_attn_metadata.query_start_loc = torch.tensor(
            [0, 8, 16, 24], dtype=torch.int32)
        mock_common_attn_metadata.seq_lens = torch.tensor([8, 8, 8],
                                                          dtype=torch.int32)
        mock_common_attn_metadata.num_actual_tokens = 24
        mock_common_attn_metadata.num_reqs = 3
        mock_common_attn_metadata.num_computed_tokens_cpu = torch.tensor(
            [5, 6, 7], dtype=torch.int32)
        mock_common_attn_metadata.block_table_tensor = MagicMock()
        mock_common_attn_metadata.slot_mapping = MagicMock()
        mock_common_attn_metadata.positions = MagicMock()

        mock_spec_decode_metadata = MagicMock(spec=SpecDecodeMetadata)
        mock_spec_decode_metadata.cu_num_draft_tokens = torch.tensor(
            [3, 5, 7], dtype=torch.int32)

        mock_runner = MagicMock()
        mock_runner.actual_seq_lengths_q = MagicMock()
        mock_runner.attn_state = MagicMock()
        mock_runner.graph_pad_size = 0
        mock_runner.pcp_size = 1
        mock_runner.decode_token_per_req = MagicMock()

        proposer = MagicMock(spec=MtpProposer)
        proposer.runner = mock_runner
        proposer.pcp_size = 1
        proposer.arange = torch.arange(100, dtype=torch.int32)
        proposer.prepare_inputs_padded = MtpProposer.prepare_inputs_padded.__get__(
            proposer)

        mock_valid_sampled_tokens_count = torch.tensor([2, 1, 2],
                                                       dtype=torch.int32)

        (spec_common_attn_metadata, token_indices,
         token_indices_to_sample) = proposer.prepare_inputs_padded(
             common_attn_metadata=mock_common_attn_metadata,
             spec_decode_metadata=mock_spec_decode_metadata,
             valid_sampled_tokens_count=mock_valid_sampled_tokens_count)

        total_num_tokens = mock_common_attn_metadata.query_start_loc_cpu[
            -1].item()
        expected_token_indices = proposer.arange[:total_num_tokens]
        assert torch.equal(token_indices, expected_token_indices)
        assert token_indices.shape == (24, )
        assert token_indices.dtype == torch.int32

        expected_sample_indices = torch.tensor([5, 13, 22], dtype=torch.int32)
        assert torch.equal(token_indices_to_sample, expected_sample_indices)

        assert isinstance(spec_common_attn_metadata,
                          AscendCommonAttentionMetadata)
        assert torch.equal(spec_common_attn_metadata.query_start_loc,
                           mock_common_attn_metadata.query_start_loc)
        assert torch.equal(spec_common_attn_metadata.query_start_loc_cpu,
                           mock_common_attn_metadata.query_start_loc_cpu)
        assert torch.equal(spec_common_attn_metadata.seq_lens_cpu,
                           mock_common_attn_metadata.seq_lens)
        assert spec_common_attn_metadata.num_reqs == mock_common_attn_metadata.num_reqs
        assert spec_common_attn_metadata.num_actual_tokens == total_num_tokens
        assert spec_common_attn_metadata.max_query_len == 8
        assert spec_common_attn_metadata.actual_seq_lengths_q == proposer.runner.actual_seq_lengths_q
