from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import (CacheConfig, CompilationConfig, ModelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.interface import SpecDcodeType
from vllm_ascend.spec_decode.mtp_proposer import MtpProposer


class TestMtpProposer:

    @pytest.fixture
    def vllm_config(self):
        config = MagicMock(spec=VllmConfig)
        config.speculative_config = MagicMock(spec=SpeculativeConfig)
        config.speculative_config.num_speculative_tokens = 2
        config.speculative_config.method = "deepseek_mtp"
        config.speculative_config.draft_model_config = MagicMock()
        config.speculative_config.draft_model_config.get_hidden_size.return_value = 4096

        config.model_config = MagicMock(spec=ModelConfig)
        config.model_config.dtype = torch.float16
        config.model_config.max_model_len = 2048
        config.model_config.uses_mrope = False
        config.model_config.hf_config = None

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
        config.device_config.device = "npu"

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
        runner.in_profile_run = False
        return runner

    def test_init(self, vllm_config, runner):
        # Test basic initialization
        proposer = MtpProposer(vllm_config, "npu", runner)

        assert proposer.name == SpecDcodeType.MTP
        assert proposer.vllm_config == vllm_config
        assert proposer.device == "npu"
        assert proposer.dtype == torch.float16
        assert proposer.num_speculative_tokens == 2
        assert proposer.hidden_size == 4096
        assert proposer.block_size == 16

        # Test with mrope enabled
        vllm_config.model_config.uses_mrope = True
        proposer = MtpProposer(vllm_config, "npu", runner)
        assert hasattr(proposer, "mrope_positions")
        assert not hasattr(proposer, "positions")

    def test_init_with_aclgraph(self, vllm_config, runner):
        runner._use_aclgraph.return_value = True
        proposer = MtpProposer(vllm_config, "npu", runner)

        assert proposer.use_aclgraph is True
        assert proposer.cudagraph_batch_sizes == [8, 4, 2, 1]  # Reversed order

    @patch("vllm.config.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.mtp_proposer.get_model_loader")
    @patch(
        "vllm_ascend.spec_decode.mtp_proposer.process_weights_after_loading")
    @patch("vllm_ascend.spec_decode.mtp_proposer.set_default_torch_dtype")
    @patch("vllm_ascend.spec_decode.mtp_proposer.set_current_vllm_config")
    def test_load_model(self, mock_set_config, mock_set_dtype,
                        mock_process_weights, mock_get_loader, mock_get_layers,
                        vllm_config, runner):
        attn_layers_all = {
            "target_attn_layer": "val0",
            "draft_attn_layer": "val1",
            "draft_attn_exclude_by_indexer": "val2",
        }

        indexer_layers_all = {
            "target_indexer_0": "val3",
            "draft_attn_exclude_by_indexer": "val4"
        }

        def get_layers_side_effect(vllm_config, cache_cls):
            if cache_cls == AttentionLayerBase:
                return attn_layers_all
            elif cache_cls == DeepseekV32IndexerCache:
                return indexer_layers_all
            else:
                return {}

        # Setup
        proposer = MtpProposer(vllm_config, "npu", runner)
        proposer._init_mtp_model = MagicMock()
        mock_model = MagicMock()
        proposer.model = mock_model

        mock_loader = MagicMock()
        mock_get_loader.return_value = mock_loader
        mock_loader.get_all_weights.return_value = {
            "dummy_weight": torch.tensor([1.0])
        }

        mock_get_layers.side_effect = get_layers_side_effect
        with pytest.raises(AssertionError):
            proposer.load_model(mock_model)

    @patch("vllm_ascend.spec_decode.mtp_proposer.set_ascend_forward_context")
    def test_dummy_run(self, mock_set_context, vllm_config, runner):
        # Setup
        proposer = MtpProposer(vllm_config, "npu", runner)
        proposer.model = MagicMock()
        runner._sync_metadata_across_dp.return_value = (8, 8, False)
        runner._select_moe_comm_method.return_value = "alltoall"

        # Execute
        proposer.dummy_run(8)

        # Verify
        runner._sync_metadata_across_dp.assert_called_once()
        runner._select_moe_comm_method.assert_called_once()
        mock_set_context.assert_called()

        # Check that model was called correct number of times
        assert proposer.model.call_count == vllm_config.speculative_config.num_speculative_tokens

    def test_generate_token_ids(self):
        mock_deps = MagicMock()
        mock_deps.scheduler_output = MagicMock(spec=SchedulerOutput)
        mock_deps.scheduler_output.num_scheduled_tokens = 16
        mock_deps.spec_decode_metadata = MagicMock(spec=SpecDecodeMetadata)
        mock_deps.spec_decode_metadata.num_draft_tokens = 2
        mock_deps.runner = MagicMock()
        mock_deps.runner.input_batch = MagicMock(num_reqs=4)
        mock_deps.runner.input_ids = torch.arange(16, dtype=torch.int32)
        mock_deps.runner.spec_decode_common_attn_metadata = MagicMock()
        mock_deps.runner.pcp_size = 2
        mock_deps.runner.input_ids_pcp_full = torch.arange(32,
                                                           dtype=torch.int32)
        mock_deps.runner.query_start_loc_pcp_full_cpu = torch.tensor(
            [0, 8, 16, 24, 32])
        mock_deps.positions = torch.arange(16, dtype=torch.int32)
        mock_deps.hidden_states = torch.zeros(16, 4096, dtype=torch.float16)
        mock_deps.sampled_token_ids = torch.tensor([[100, 101, -1],
                                                    [200, -1, -1],
                                                    [300, 301, 302]])

        proposer = MagicMock(spec=MtpProposer)
        proposer.runner = mock_deps.runner
        proposer.decode_threshold = 1
        proposer.speculative_config = MagicMock(
            disable_padded_drafter_batch=False)
        proposer.pcp_size = mock_deps.runner.pcp_size
        proposer._get_attn_metadata = MagicMock(return_value=MagicMock())
        proposer.prepare_next_token_ids_padded = MagicMock(
            return_value=(torch.tensor([101, 200, 302]), 3))
        proposer.prepare_inputs_padded = MagicMock(
            return_value=(MagicMock(), torch.tensor([0, 2, 4]),
                          torch.tensor([7, 15, 23])))
        proposer._propose = MagicMock(
            return_value=torch.tensor([400, 401, 402]))
        proposer.generate_token_ids = MtpProposer.generate_token_ids.__get__(
            proposer)

        draft_token_ids = proposer.generate_token_ids(
            sampled_token_ids=mock_deps.sampled_token_ids,
            scheduler_output=mock_deps.scheduler_output,
            spec_decode_metadata=mock_deps.spec_decode_metadata,
            positions=mock_deps.positions,
            num_scheduled_tokens=mock_deps.scheduler_output.
            num_scheduled_tokens,
            hidden_states=mock_deps.hidden_states,
        )

        proposer.prepare_next_token_ids_padded.assert_called_once()
        proposer.prepare_inputs_padded.assert_called_once()
        proposer._get_attn_metadata.assert_called_once()
        proposer._propose.assert_called_once()
        assert torch.equal(draft_token_ids, proposer._propose.return_value)

    def test_prepare_next_token_ids_cpu(self):
        mock_requests = {}
        req1 = MagicMock(spec=CachedRequestState)
        req1.num_computed_tokens = 5
        req1.get_token_id = MagicMock(return_value=1000)
        mock_requests["req_001"] = req1
        req2 = MagicMock(spec=CachedRequestState)
        req2.num_computed_tokens = 8
        req2.get_token_id = MagicMock(return_value=2000)
        mock_requests["req_002"] = req2
        req3 = MagicMock(spec=CachedRequestState)
        req3.num_computed_tokens = 3
        req3.get_token_id = MagicMock(return_value=3000)
        mock_requests["req_003"] = req3

        mock_gpu_input_batch = MagicMock(spec=InputBatch)
        mock_gpu_input_batch.req_ids = ["req_001", "req_002", "req_003"]

        mock_num_scheduled_tokens = {"req_001": 2, "req_002": 0, "req_003": 1}

        sampled_token_ids = [
            [101, 102, 103],  # req001: return 103
            [],  # req002: return fallback 2000
            [301]  # req003: return 301
        ]

        proposer = MagicMock(spec=MtpProposer)
        proposer.input_ids = MagicMock(device=torch.device("cpu"))
        proposer.prepare_next_token_ids_cpu = MtpProposer.prepare_next_token_ids_cpu.__get__(
            proposer)

        next_token_ids = proposer.prepare_next_token_ids_cpu(
            sampled_token_ids=sampled_token_ids,
            requests=mock_requests,
            gpu_input_batch=mock_gpu_input_batch,
            num_scheduled_tokens=mock_num_scheduled_tokens)

        expected = torch.tensor([103, 2000, 301],
                                dtype=torch.int32,
                                device="cpu")
        assert torch.equal(next_token_ids, expected)

        mock_requests["req_002"].get_token_id.assert_called_once_with(8)
        mock_requests["req_001"].get_token_id.assert_not_called()
        mock_requests["req_003"].get_token_id.assert_not_called()

    def test_prepare_next_token_ids_padded(self):
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
                                              device="cpu")

        mock_requests = {}
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
                                            device="cpu")
        assert torch.equal(next_token_ids, expected_next_tokens)

    def test_prepare_inputs_padded(self):
        mock_common_attn_metadata = MagicMock(spec=CommonAttentionMetadata)
        mock_common_attn_metadata.query_start_loc_cpu = torch.tensor(
            [0, 8, 16, 24], dtype=torch.int32)
        mock_common_attn_metadata.query_start_loc = torch.tensor(
            [0, 8, 16, 24], dtype=torch.int32)
        mock_common_attn_metadata.seq_lens = torch.tensor([8, 8, 8],
                                                          dtype=torch.int32)
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
        mock_runner.attn_mask = MagicMock()
        mock_runner.spec_attn_mask = MagicMock()
        mock_runner.attn_state = MagicMock()
        mock_runner.graph_pad_size = 0
        mock_runner.decode_token_per_req = MagicMock()

        proposer = MagicMock(spec=MtpProposer)
        proposer.runner = mock_runner
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
        assert spec_common_attn_metadata.attn_mask == proposer.runner.attn_mask
        assert spec_common_attn_metadata.spec_attn_mask == proposer.runner.spec_attn_mask
