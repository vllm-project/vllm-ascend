# ruff: noqa: E501
import inspect
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import CacheConfig, CompilationMode, CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.forward_context import BatchDescriptor
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.spec_decode.draft_model import DraftModelProposer

import vllm_ascend.spec_decode.eagle_proposer as eagle_proposer
from tests.ut.base import TestBase
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer


def test_prepare_inputs_padded_preserves_internal_seq_lens_cpu():
    proposer = AscendEagleProposer.__new__(AscendEagleProposer)
    proposer.pcp_size = 1
    proposer.arange = torch.arange(16, dtype=torch.int32)
    proposer.runner = MagicMock()
    proposer.runner.actual_seq_lengths_q = [3, 3]
    proposer.runner.attn_state = AscendAttentionState.SpecDecoding
    proposer.runner.decode_token_per_req = 4

    internal_seq_lens_cpu = torch.tensor([7, 9], dtype=torch.int32)
    common_attn_metadata = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3, 6], dtype=torch.int32),
        seq_lens=torch.tensor([7, 9], dtype=torch.int32),
        _seq_lens_cpu=internal_seq_lens_cpu,
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=2,
        num_actual_tokens=6,
        num_input_tokens=6,
        max_query_len=3,
        actual_seq_lengths_q=[3, 3],
        block_table_tensor=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.arange(6, dtype=torch.int32),
        positions=torch.arange(6),
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=4,
        max_seq_len=9,
    )
    spec_decode_metadata = MagicMock()
    spec_decode_metadata.cu_num_draft_tokens = torch.tensor([2, 3], dtype=torch.int32)
    valid_sampled_tokens_count = torch.tensor([3, 1], dtype=torch.int32)

    with patch.object(eagle_proposer, "HAS_TRITON", False):
        spec_common_attn_metadata, *_ = proposer.prepare_inputs_padded(
            common_attn_metadata,
            spec_decode_metadata,
            valid_sampled_tokens_count,
        )

    assert spec_common_attn_metadata._seq_lens_cpu is internal_seq_lens_cpu
    assert spec_common_attn_metadata.seq_lens_cpu is None


class TestEagleProposerInitialization(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.cache_config = MagicMock(spec=CacheConfig)
        self.vllm_config.scheduler_config = MagicMock()
        self.vllm_config.model_config = MagicMock()
        self.vllm_config.model_config.hf_text_config = MagicMock(
            spec=[]
        )  # Empty spec to prevent hasattr from returning True
        self.vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        self.vllm_config.compilation_config = MagicMock()
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.enable_expert_parallel = False
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.num_speculative_tokens = 2
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(2)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    def test_initialization_eagle_graph(self):
        self.vllm_config.speculative_config.method = "eagle"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 4096
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        self.vllm_config.model_config.enforce_eager = False
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = False
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 4096)
            self.assertTrue(proposer.use_cuda_graph)

            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.input_ids.shape, (expected_max_num_tokens,))
            self.assertEqual(proposer.positions.shape, (expected_max_num_tokens,))
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 4096))
            self.assertEqual(proposer.arange.shape, (expected_max_num_tokens,))

    def test_initialization_eagle3_enforce_eager(self):
        self.vllm_config.speculative_config.method = "eagle3"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.NONE
        self.vllm_config.compilation_config.pass_config = MagicMock()
        self.vllm_config.compilation_config.pass_config.enable_sp = False
        self.vllm_config.model_config.enforce_eager = True
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 2048)
            self.assertFalse(proposer.use_cuda_graph)
            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 2048))

    def test_initialization_eagle3_full_graph_async(self):
        self.vllm_config.speculative_config.method = "eagle3"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        self.vllm_config.model_config.enforce_eager = False
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = True
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 2048)
            self.assertTrue(proposer.use_cuda_graph)
            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 2048))

    def test_initialization_mtp_full_graph_async(self):
        self.vllm_config.speculative_config.method = "mtp"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        self.vllm_config.model_config.enforce_eager = False
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = True
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 2048)
            self.assertTrue(proposer.use_cuda_graph)
            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 2048))

    def test_initialization_draft_model(self):
        self.vllm_config.speculative_config.method = "draft_model"
        self.vllm_config.speculative_config.parallel_drafting = False
        # TODO(klyzhenko-vadim): remove when target_tp != draft_tp will be supported.
        self.vllm_config.speculative_config.draft_parallel_config.tensor_parallel_size = 1
        self.vllm_config.speculative_config.target_parallel_config.tensor_parallel_size = 1
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendDraftModelProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertTrue(isinstance(proposer, DraftModelProposer))
            self.assertFalse(proposer.pass_hidden_states_to_model)
            self.assertTrue(proposer.needs_extra_input_slots)


@unittest.skip("Skip due to the changes in #7153, fix me later")
class TestEagleProposerLoadModel(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.speculative_config.method = "eagle"
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.enable_expert_parallel = False
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.num_speculative_tokens = 2
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(2)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)
        self.proposer.parallel_drafting = False

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    def test_load_model_pp1(self, mock_pp_group, mock_get_model, mock_get_layers):
        mock_pp_group.return_value.world_size = 1
        mock_target_layer1 = MagicMock()
        mock_target_layer2 = MagicMock()
        mock_draft_layer1 = MagicMock()
        mock_draft_layer3 = MagicMock()
        mock_get_layers.side_effect = [
            {"layer1": mock_target_layer1, "layer2": mock_target_layer2},
            {},
            {},
            {"layer1": mock_draft_layer1, "layer3": mock_draft_layer3},
        ]

        weight = torch.zeros(0)

        mock_model = MagicMock()
        mock_model.supports_multimodal = False
        mock_model.lm_head = MagicMock()
        mock_model.multimodal_cpu_fields = None
        mock_model.merge_by_field_config = None
        mock_model.model.embed_tokens = MagicMock()
        mock_model.model.embed_tokens.weight = weight

        mock_get_model.return_value = MagicMock()
        mock_get_model.return_value.model.embed_tokens.weight = weight

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)
            mock_get_model.assert_called_once()
            self.assertEqual(self.proposer.attn_layer_names, ["layer3"])
            self.assertIs(self.proposer.model.model.embed_tokens, mock_model.model.embed_tokens)

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    def test_load_model_pp_gt1(self, mock_pp_group, mock_get_model, mock_get_layers):
        mock_pp_group.return_value.world_size = 2
        mock_target_layer1 = MagicMock()
        mock_draft_layer2 = MagicMock()

        mock_get_layers.side_effect = [{"layer1": mock_target_layer1}, {}, {}, {"layer2": mock_draft_layer2}]

        mock_model = MagicMock()
        original_embed = MagicMock()
        mock_model.multimodal_cpu_fields = None
        mock_model.merge_by_field_config = None
        mock_get_model.return_value = MagicMock(model=MagicMock(embed_tokens=original_embed))

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)

            self.assertIsNot(self.proposer.model.model.embed_tokens, mock_model.model.embed_tokens)
            self.assertEqual(self.proposer.attn_layer_names, ["layer2"])

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    @patch("vllm_ascend.spec_decode.eagle_proposer.supports_multimodal")
    def test_load_model_multimodal(self, mock_supports_multi, mock_pp_group, mock_get_model, mock_get_layers):
        mock_model = MagicMock()
        mock_model.get_language_model.return_value.lm_head = MagicMock()
        mock_supports_multi.return_value = True
        original_embed = MagicMock()
        mock_get_model.return_value = MagicMock(model=MagicMock(embed_tokens=original_embed))

        mock_target_layer1 = MagicMock()
        mock_draft_layer2 = MagicMock()

        mock_get_layers.side_effect = [{"layer1": mock_target_layer1}, {}, {}, {"layer2": mock_draft_layer2}]
        mock_pp_group.return_value.world_size = 2

        self.proposer.model = MagicMock()

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)
            self.assertEqual(mock_model.get_language_model.call_count, 2)
            self.assertIs(self.proposer.model.lm_head, mock_model.get_language_model.return_value.lm_head)


class TestEagleProposerDummyRun(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.speculative_config.num_speculative_tokens = 4
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1
        self.runner.pin_memory = False
        self.runner._sync_metadata_across_dp.return_value = (8, torch.tensor([8]), CUDAGraphMode.NONE)

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.model_config.use_mla = False
        self.vllm_config.model_config.hf_text_config = MagicMock(
            spec=[]
        )  # Empty spec to prevent hasattr from returning True
        self.vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(4)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Mock parallel state functions
        self.mock_tp_world_size = patch(
            "vllm_ascend.ascend_forward_context.get_tensor_model_parallel_world_size", return_value=1
        )
        self.mock_tp_world_size.start()

        mock_dp_group = MagicMock()
        mock_dp_group.world_size = 1
        self.mock_dp_group = patch("vllm_ascend.ascend_forward_context.get_dp_group", return_value=mock_dp_group)
        self.mock_dp_group.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)
        self.proposer.model = MagicMock()
        self.proposer._runnable = MagicMock()
        self.proposer.update_stream = MagicMock()

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        self.mock_tp_world_size.stop()
        self.mock_dp_group.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    # cpu does not support parallel-group, let alone `sp`
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch(
        "vllm_ascend.spec_decode.eagle_proposer.get_forward_context", **{"return_value.flash_comm_v1_enabled": False}
    )
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_basic(self, mock_context, mock_get_context, mock_get_context_2):
        num_tokens = 32
        with_prefill = False

        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=num_tokens, with_prefill=with_prefill)

            self.assertTrue(self.proposer._runnable.call_count == 1)

    # cpu does not support parallel-group, let alone `sp`
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch(
        "vllm_ascend.spec_decode.eagle_proposer.get_forward_context", **{"return_value.flash_comm_v1_enabled": False}
    )
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_with_prefill(self, mock_context, mock_get_context, mock_get_context_2):
        mock_context.return_value.__enter__.return_value = None
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64, with_prefill=True, num_reqs=4)
            self.assertTrue(self.proposer._runnable.call_count == 1)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.update_full_graph_params")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_in_graph_capture(
        self, mock_context, mock_get_context, mock_update_full_graph_params, mock_get_context_2
    ):
        last_use_cuda_graph = self.proposer.use_cuda_graph
        mock_return_context = MagicMock()
        mock_return_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        mock_return_context.capturing = True
        # cpu does not support parallel-group, let alone `sp`
        mock_return_context.flash_comm_v1_enabled = False
        mock_get_context.return_value = mock_return_context
        mock_get_context_2.return_value = mock_return_context
        self.proposer.use_cuda_graph = True
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64, in_graph_capturing=True, aclgraph_runtime_mode=CUDAGraphMode.FULL)
            self.assertTrue(self.proposer._runnable.call_count == 1)
            mock_update_full_graph_params.assert_not_called()
            self.proposer.use_cuda_graph = last_use_cuda_graph

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.update_full_graph_params")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_in_graph_run(
        self, mock_context, mock_get_context, mock_update_full_graph_params, mock_get_context_2
    ):
        last_use_cuda_graph = self.proposer.use_cuda_graph
        mock_return_context = MagicMock()
        mock_return_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        mock_return_context.capturing = False
        # cpu does not support parallel-group, let alone `sp`
        mock_return_context.flash_comm_v1_enabled = False
        mock_get_context.return_value = mock_return_context
        mock_get_context_2.return_value = mock_return_context
        self.proposer.use_cuda_graph = True
        self.proposer.draft_attn_groups = [MagicMock()]
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64, in_graph_capturing=False, aclgraph_runtime_mode=CUDAGraphMode.FULL)
            self.assertTrue(self.proposer._runnable.call_count == 1)
            self.assertTrue(mock_update_full_graph_params.call_count == 1)
            self.proposer.use_cuda_graph = last_use_cuda_graph


class TestEagleProposerHelperMethods(TestBase):
    # TODO: Can add some tests about prepare_next_token_ids in future.

    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.scheduler_config = MagicMock(max_num_seqs=3)
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.input_batch = MagicMock()
        self.runner.input_batch.req_ids = [0, 1, 2]
        self.runner.arange_np = np.arange(10)
        self.runner.input_batch.num_reqs = 3
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.enable_expert_parallel = False
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.num_speculative_tokens = 2
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(2)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    # TODO: This is equivalent to disable_padded_drafter_batch=True.
    # We need to add a test_prepare_inputs_padded in future.
    def test_prepare_inputs(self):
        self.proposer.token_arange_np = np.arange(10)
        mock_attn = MagicMock()
        mock_attn.slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5])
        num_rejected = torch.tensor([1, 0, 1], device=self.device)
        mock_return_attn = MagicMock()

        with (
            set_current_vllm_config(self.vllm_config),
            patch.object(self.proposer, "prepare_inputs", return_value=(mock_return_attn, torch.tensor([1, 2, 4]))),
        ):
            return_attn, indices = self.proposer.prepare_inputs(mock_attn, num_rejected)
            self.assertEqual(indices.tolist(), [1, 2, 4])


# fmt: off
class TestEagleProposerPropose:
    @pytest.fixture(autouse=True)
    def setUp_and_tearDown(self):

        # before mock and patch, add assertions to ensure
        # that the mocked functions and parameters exist
        self.check_mock()

        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.speculative_config.num_speculative_tokens = 3
        self.vllm_config.speculative_config.method = "eagle3"
        self.vllm_config.speculative_config.parallel_drafting = False
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1
        self.runner.max_num_tokens = 8192
        self.runner.max_num_reqs = 256
        self.runner.pin_memory = False

        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 32768
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.model_config.use_mla = False
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(4)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Mock parallel state functions
        self.mock_tp_world_size = patch(
            "vllm_ascend.ascend_forward_context.get_tensor_model_parallel_world_size", return_value=1
        )
        self.mock_tp_world_size.start()

        mock_dp_group = MagicMock()
        mock_dp_group.world_size = 1
        self.mock_dp_group = patch("vllm_ascend.ascend_forward_context.get_dp_group", return_value=mock_dp_group)
        self.mock_dp_group.start()

        # Mock sp
        self.mock_enable_sp = patch(
            "vllm_ascend.utils.enable_sp", return_value=False
        )
        self.mock_enable_sp.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

        yield

        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        self.mock_tp_world_size.stop()
        self.mock_dp_group.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    # config: prefill and decode, Qwen3-8B, tp1, enforce_eager, no_async_scheduling, eagle3, k=3, "disable_padded_drafter_batch": False
    @pytest.mark.parametrize(
        'flag_prefill_decode, query_start_loc, query_start_loc_cpu, seq_lens, num_reqs,' \
        'num_actual_tokens, max_query_len, max_seq_len, block_table_tensor,' \
        'slot_mapping, causal, logits_indices_padded, num_logits_indices,' \
        'encoder_seq_lens, encoder_seq_lens_cpu, dcp_local_seq_lens,' \
        'dcp_local_seq_lens_cpu, _seq_lens_cpu, _num_computed_tokens_cpu,' \
        '_num_computed_tokens_cache, seq_lens_cpu, num_computed_tokens_cpu,' \
        'decode_token_per_req, actual_seq_lengths_q, positions, attn_state,' \
        'graph_pad_size, num_input_tokens, prefill_context_parallel_metadata',
        [
            (
                "prefill", torch.tensor([ 0, 13], device=torch.device("cpu"), dtype=torch.int32), torch.tensor([ 0, 13], dtype=torch.int32), 
                torch.tensor([13], device=torch.device("cpu"), dtype=torch.int32), 1, 13, 13, 13,
                torch.eye(256, device=torch.device("cpu"), dtype=torch.int32)[0].unsqueeze(0),
                torch.tensor([128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140], device=torch.device("cpu"), dtype=torch.int32),
                True, None, None, None, None, None, None, None, None, None, torch.tensor([13], dtype=torch.int32), torch.tensor([0], dtype=torch.int32), 4, [],
                torch.cat([torch.arange(13), torch.zeros(8704 - 13)]),
                AscendAttentionState.PrefillNoCache, -1, 13, None
            ),
            (
                "decode", torch.tensor([ 0, 4, 8, 12], device=torch.device("cpu"), dtype=torch.int32), torch.tensor([ 0, 4, 8, 12], dtype=torch.int32), 
                torch.tensor([21, 17, 17], device=torch.device("cpu"), dtype=torch.int32), 3, 12, 4, 0,
                torch.cat([torch.eye(256, device="cpu", dtype=torch.int32)[0].unsqueeze(0)*i for i in [1,2,3]], dim=0),
                torch.tensor([145, 146, 147, 148, 269, 270, 271, 272, 397, 398, 399, 400], device=torch.device("cpu"), dtype=torch.int32),
                True, None, None, None, None, None, None, None, None, None, torch.tensor([21, 17, 17], dtype=torch.int32), torch.tensor([17, 13, 13], dtype=torch.int32), 4, [],
                torch.cat([torch.tensor([17, 18, 19, 20, 13, 14, 15, 16, 13, 14, 15, 16, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), torch.zeros(8704 - 30)]),
                AscendAttentionState.ChunkedPrefill, -1, 12, None
            ),
        ]
    )
    # config: prefill and decode, Qwen3-30B, tp2, ep_enable, enforce_eager, no_async_scheduling, eagle3, k=3, "disable_padded_drafter_batch": False
    @pytest.mark.parametrize('model_type', ['qwen_dense','qwen_moe', 'deepseek'])
    @pytest.mark.parametrize('graphmode', ['eager','full'])
    @patch('vllm_ascend.spec_decode.eagle_proposer.AscendEagleProposer.get_model')
    def test_propose(self, mock_get_model, graphmode, model_type, flag_prefill_decode,
                     query_start_loc, query_start_loc_cpu, seq_lens, num_reqs,
                     num_actual_tokens, max_query_len, max_seq_len, block_table_tensor,
                     slot_mapping, causal, logits_indices_padded, num_logits_indices,
                     encoder_seq_lens, encoder_seq_lens_cpu, dcp_local_seq_lens,
                     dcp_local_seq_lens_cpu, _seq_lens_cpu, _num_computed_tokens_cpu,
                     _num_computed_tokens_cache, seq_lens_cpu, num_computed_tokens_cpu,
                     decode_token_per_req, actual_seq_lengths_q, positions, attn_state,
                     graph_pad_size, num_input_tokens, prefill_context_parallel_metadata
                    ):
        # adjust for fullgraph mode
        if graphmode == 'full':
            if model_type == "qwen_dense" and self.is_decode(flag_prefill_decode):
                slot_mapping = torch.tensor([145, 146, 147, 148, 269, 270, 271, 272, 397, 398, 399, 400, -1, -1], device=torch.device("cpu"), dtype=torch.int32)
                positions = torch.cat([torch.tensor([17, 18, 19, 20, 13, 14, 15, 16, 13, 14, 15, 16, 0, 0, 0, 0, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), torch.zeros(8704 - 30)])
                num_input_tokens = 16
                target_model_batch_desc = BatchDescriptor(num_tokens=num_input_tokens, num_reqs=4, uniform=True, has_lora=False, num_active_loras=0)
                self.proposer.use_cuda_graph = True
            else:
                pytest.skip("For the entire graph test, only one model needs to be tested to avoid repeated tests.")

        # mock and adjust functions and var in propose
        if model_type == 'deepseek':
            self.proposer.method = 'mtp'
            if not self.is_decode(flag_prefill_decode):
                num_actual_tokens = 9
        self.runner._sync_metadata_across_dp.return_value = (num_actual_tokens, None, False)
        self.proposer.model = MagicMock(spec=Eagle3LlamaForCausalLM)
        custom_combined_hidden_states = torch.zeros(num_actual_tokens, 4096, device=self.device, dtype=torch.bfloat16)
        self.proposer.model.combine_hidden_states.return_value = custom_combined_hidden_states
        mock_get_model.return_value = self.proposer.model
        self.proposer.hidden_size = 4096
        if model_type == 'deepseek':
            self.proposer.hidden_states = torch.zeros(8192, 7168, device=self.device, dtype=torch.bfloat16)
        else:
            self.proposer.hidden_states = torch.zeros(8192, 4096, device=self.device, dtype=torch.bfloat16)
        mock_attn_group = MagicMock()
        mock_builder = MagicMock()
        mock_attn_metadata = MagicMock()
        mock_builder.build.return_value = mock_attn_metadata
        mock_attn_group.get_metadata_builder.return_value = mock_builder
        self.proposer.draft_attn_groups = [mock_attn_group]
        self.proposer.attn_layer_names = ['model.layers.36.self_attn.attn']
        self.proposer.kernel_block_size = 128
        self.proposer._runnable = MagicMock()
        self.proposer._runnable.return_value = [0, 0, 0]
        captured_common_attn_metadata = None
        original_method = self.proposer.attn_update_stack_num_spec_norm
        mock_bd = MagicMock()
        mock_bd.num_tokens = 16
        self.runner.cudagraph_dispatcher.dispatch.return_value = (CUDAGraphMode.FULL, mock_bd)
        self.runner._pad_query_start_loc_for_fia.return_value = 4
        self.runner.query_start_loc.gpu = torch.tensor([0, 4, 8, 12, 16], device=torch.device("cpu"), dtype=torch.int32)
        self.runner.query_start_loc.cpu = torch.tensor([0, 4, 8, 12, 16], device=torch.device("cpu"), dtype=torch.int32)
        self.runner.seq_lens = seq_lens
        self.runner.optimistic_seq_lens_cpu = seq_lens_cpu
        self.proposer._update_full_graph_params = MagicMock()

        def side_effect(*args, **kwargs):
            nonlocal captured_common_attn_metadata
            res_common, res_attn = original_method(*args, **kwargs)
            captured_common_attn_metadata = res_common
            return res_common, res_attn

        # create common_attn_metadata
        mock_common_attn_metadata= MagicMock()
        if not self.is_decode(flag_prefill_decode):
            mock_common_attn_metadata.batch_size.return_value = 1
            if model_type == 'qwen_moe':
                _seq_lens_cpu = torch.tensor([13], dtype=torch.int32)
            if model_type == 'deepseek':
                query_start_loc = torch.tensor([0, 9], device=torch.device("cpu"), dtype=torch.int32)
                query_start_loc_cpu = torch.tensor([0, 9], device=torch.device("cpu"), dtype=torch.int32)
                seq_lens = torch.tensor([9], device=torch.device("cpu"), dtype=torch.int32)
                max_query_len = 9
                max_seq_len = 9
                slot_mapping = torch.tensor([128, 129, 130, 131, 132, 133, 134, 135, 136], device=torch.device("cpu"), dtype=torch.int32)
                _seq_lens_cpu = torch.tensor([9], dtype=torch.int32)
                seq_lens_cpu = torch.tensor([9], dtype=torch.int32)
                positions = torch.cat([torch.arange(9), torch.zeros(8704 - 9)])
                num_input_tokens = 9
        if self.is_decode(flag_prefill_decode):
            mock_common_attn_metadata.batch_size.return_value = 3
            if model_type == 'qwen_moe':
                seq_lens = torch.tensor([19, 17, 17], device=torch.device("cpu"), dtype=torch.int32)
                slot_mapping = torch.tensor([143, 144, 145, 146, 269, 270, 271, 272, 397, 398, 399, 400], device=torch.device("cpu"), dtype=torch.int32)
                seq_lens_cpu = torch.tensor([19, 17, 17], dtype=torch.int32)
                num_computed_tokens_cpu = torch.tensor([15, 13, 13], dtype=torch.int32)
                positions = torch.cat([torch.tensor([15, 16, 17, 18, 13, 14, 15, 16, 13, 14, 15, 16, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), torch.zeros(8704 - 30)])
            if model_type == 'deepseek':
                seq_lens = torch.tensor([14, 13, 14], device=torch.device("cpu"), dtype=torch.int32)
                slot_mapping = torch.tensor([138, 139, 140, 141, 265, 266, 267, 268, 394, 395, 396, 397], device=torch.device("cpu"), dtype=torch.int32)
                seq_lens_cpu = torch.tensor([14, 13, 14], dtype=torch.int32)
                num_computed_tokens_cpu = torch.tensor([10, 9, 10], dtype=torch.int32)
                positions = torch.cat([torch.tensor([10, 11, 12, 13, 9, 10, 11, 12, 10, 11, 12, 13, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), torch.zeros(8704 - 23)])
                attn_state = AscendAttentionState.SpecDecoding
        self.value_mock_common_attn_metadata(mock_common_attn_metadata, query_start_loc, query_start_loc_cpu, seq_lens, num_reqs,
                                        num_actual_tokens, max_query_len, max_seq_len, block_table_tensor,
                                        slot_mapping, causal, logits_indices_padded, num_logits_indices,
                                        encoder_seq_lens, encoder_seq_lens_cpu, dcp_local_seq_lens,
                                        dcp_local_seq_lens_cpu, _seq_lens_cpu, _num_computed_tokens_cpu,
                                        _num_computed_tokens_cache, seq_lens_cpu, num_computed_tokens_cpu,
                                        decode_token_per_req, actual_seq_lengths_q, positions, attn_state,
                                        graph_pad_size, num_input_tokens, prefill_context_parallel_metadata
                                        )
        
        # create other parameters
        if not self.is_decode(flag_prefill_decode):
            if model_type == 'qwen_dense' or model_type == 'qwen_moe':
                target_token_ids = torch.tensor([151644, 872, 198, 5501, 7512, 14678, 51765, 30, 151645, 198, 151644, 77091, 198], device=self.device, dtype=torch.int32)
                target_positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], device=self.device)
                next_token_ids = torch.tensor([151667], device=self.device, dtype=torch.int32)
                req_scheduled_tokens = {'0-8222703c': 13}
            if model_type == 'deepseek':
                target_token_ids = torch.tensor([ 0, 0, 128803, 12473, 9734, 19991, 50096, 33, 128804], device=self.device, dtype=torch.int32)
                target_positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], device=self.device)
                target_hidden_states = torch.zeros(num_actual_tokens, 7168, device=self.device, dtype=torch.bfloat16)
                next_token_ids = torch.tensor([128798], device=self.device, dtype=torch.int32)
                req_scheduled_tokens = {'0-b4ed8210': 9}
            if model_type == 'qwen_dense':
                target_hidden_states = torch.zeros(num_actual_tokens, 12288, device=self.device, dtype=torch.bfloat16)
            if model_type == 'qwen_moe':
                target_hidden_states = torch.zeros(num_actual_tokens, 6144, device=self.device, dtype=torch.bfloat16)
            token_indices_to_sample = None
            target_model_batch_desc = BatchDescriptor(num_tokens=num_actual_tokens, num_reqs=None, uniform=False, has_lora=False, num_active_loras=0)
            mock_sampling_metadata = MagicMock()
            mm_embed_inputs = None
            long_seq_metadata = None
            num_prefill_reqs = 0
            num_decode_reqs = 0
            scheduler_output = MagicMock()
            num_scheduled_tokens = num_actual_tokens
            num_rejected_tokens_gpu = None

        if self.is_decode(flag_prefill_decode):
            if model_type == 'qwen_dense':
                target_token_ids = torch.tensor([279, 1196, 374, 8014, 151667, 198, 32313, 11, 151667, 198, 32313, 11], device=self.device, dtype=torch.int32)
                target_positions = torch.tensor([17, 18, 19, 20, 13, 14, 15, 16, 13, 14, 15, 16], device=self.device)
                target_hidden_states = torch.zeros(num_actual_tokens, 12288, device=self.device, dtype=torch.bfloat16)
                next_token_ids = torch.tensor([4588, 279, 279], device=self.device, dtype=torch.int32)
                token_indices_to_sample = torch.tensor([1, 7, 11], device=self.device, dtype=torch.int32)
                num_rejected_tokens_gpu = torch.tensor([2, 0, 0], device=self.device, dtype=torch.int32)
            if model_type == 'qwen_moe':
                target_token_ids = torch.tensor([32313, 2776, 198, 198, 151667, 198, 198, 198, 151667, 198, 198, 198], device=self.device, dtype=torch.int32)
                target_positions = torch.tensor([15, 16, 17, 18, 13, 14, 15, 16, 13, 14, 15, 16], device=self.device)
                target_hidden_states = torch.zeros(num_actual_tokens, 6144, device=self.device, dtype=torch.bfloat16)
                next_token_ids = torch.tensor([11, 32313, 32313], device=self.device, dtype=torch.int32)
                token_indices_to_sample = torch.tensor([0, 5, 9], device=self.device, dtype=torch.int32)
                num_rejected_tokens_gpu = torch.tensor([3, 2, 2], device=self.device, dtype=torch.int32)
            if model_type == 'deepseek':
                target_token_ids = torch.tensor([201, 33001, 14, 832, 128798, 271, 5, 128798, 128798, 271, 5, 128798], device=self.device, dtype=torch.int32)
                target_positions = torch.tensor([10, 11, 12, 13, 9, 10, 11, 12, 10, 11, 12, 13], device=self.device)
                target_hidden_states = torch.zeros(num_actual_tokens, 7168, device=self.device, dtype=torch.bfloat16)
                next_token_ids = torch.tensor([270, 128799, 201], device=self.device, dtype=torch.int32)
                token_indices_to_sample = torch.tensor([2, 5, 8], device=self.device, dtype=torch.int32)
                num_rejected_tokens_gpu = torch.tensor([1, 2, 3], device=self.device, dtype=torch.int32)
            target_model_batch_desc = BatchDescriptor(num_tokens=num_actual_tokens, num_reqs=None, uniform=False, has_lora=False, num_active_loras=0)
            mock_sampling_metadata = MagicMock()
            mm_embed_inputs = None
            req_scheduled_tokens = {'0-b69afbe5': 4, '1-b60368b9': 4, '2-82281e95': 4}
            long_seq_metadata = None
            num_prefill_reqs = 0
            num_decode_reqs = 0
            scheduler_output = MagicMock()
            num_scheduled_tokens = num_actual_tokens

        #run
        with (
            patch.object(self.proposer, 'attn_update_stack_num_spec_norm', side_effect=side_effect),
            set_current_vllm_config(self.vllm_config),
        ):
            self.proposer._propose(target_token_ids, target_positions, target_hidden_states, next_token_ids,
                                token_indices_to_sample, mock_common_attn_metadata, target_model_batch_desc, mock_sampling_metadata,
                                mm_embed_inputs, req_scheduled_tokens, long_seq_metadata, num_prefill_reqs, num_decode_reqs,
                                scheduler_output, num_scheduled_tokens, num_rejected_tokens_gpu,
                                )
            self.assert_value_common_attn_metadata(captured_common_attn_metadata, flag_prefill_decode, model_type, graphmode)

    # give common_attn_metadata value
    def value_mock_common_attn_metadata(self, mock_common_attn_metadata, query_start_loc, query_start_loc_cpu, seq_lens, num_reqs,
                                        num_actual_tokens, max_query_len, max_seq_len, block_table_tensor,
                                        slot_mapping, causal, logits_indices_padded, num_logits_indices,
                                        encoder_seq_lens, encoder_seq_lens_cpu, dcp_local_seq_lens,
                                        dcp_local_seq_lens_cpu, _seq_lens_cpu, _num_computed_tokens_cpu,
                                        _num_computed_tokens_cache, seq_lens_cpu, num_computed_tokens_cpu,
                                        decode_token_per_req, actual_seq_lengths_q, positions, attn_state,
                                        graph_pad_size, num_input_tokens, prefill_context_parallel_metadata
                                        ):
        mock_common_attn_metadata.query_start_loc = query_start_loc
        mock_common_attn_metadata.query_start_loc_cpu = query_start_loc_cpu
        mock_common_attn_metadata.seq_lens = seq_lens
        mock_common_attn_metadata.num_reqs = num_reqs
        mock_common_attn_metadata.num_actual_tokens = num_actual_tokens
        mock_common_attn_metadata.max_query_len = max_query_len
        mock_common_attn_metadata.max_seq_len = max_seq_len
        mock_common_attn_metadata.block_table_tensor = block_table_tensor
        mock_common_attn_metadata.slot_mapping = slot_mapping
        mock_common_attn_metadata.causal = causal
        mock_common_attn_metadata.logits_indices_padded = logits_indices_padded
        mock_common_attn_metadata.num_logits_indices = num_logits_indices
        mock_common_attn_metadata.encoder_seq_lens = encoder_seq_lens
        mock_common_attn_metadata.encoder_seq_lens_cpu = encoder_seq_lens_cpu
        mock_common_attn_metadata.dcp_local_seq_lens = dcp_local_seq_lens
        mock_common_attn_metadata.dcp_local_seq_lens_cpu = dcp_local_seq_lens_cpu
        mock_common_attn_metadata._seq_lens_cpu = _seq_lens_cpu
        mock_common_attn_metadata._num_computed_tokens_cpu = _num_computed_tokens_cpu
        mock_common_attn_metadata._num_computed_tokens_cache = _num_computed_tokens_cache
        mock_common_attn_metadata.seq_lens_cpu = seq_lens_cpu
        mock_common_attn_metadata.num_computed_tokens_cpu = num_computed_tokens_cpu
        mock_common_attn_metadata.decode_token_per_req = decode_token_per_req
        mock_common_attn_metadata.actual_seq_lengths_q = actual_seq_lengths_q
        mock_common_attn_metadata.positions = positions
        mock_common_attn_metadata.attn_state = attn_state
        mock_common_attn_metadata.graph_pad_size = graph_pad_size
        mock_common_attn_metadata.num_input_tokens = num_input_tokens
        mock_common_attn_metadata.prefill_context_parallel_metadata = prefill_context_parallel_metadata

    # assert the value common_attn_metadata
    def assert_value_common_attn_metadata(self, captured_common_attn_metadata, flag_prefill_decode, model_type, graphmode):
        if not self.is_decode(flag_prefill_decode):
            assert torch.equal(captured_common_attn_metadata.query_start_loc, torch.tensor([0, 1]))
            assert torch.equal(captured_common_attn_metadata.query_start_loc_cpu, torch.tensor([0, 1]))
            assert captured_common_attn_metadata.num_reqs == 1
            assert captured_common_attn_metadata.num_actual_tokens == 1
            assert captured_common_attn_metadata.max_query_len == 1
            assert torch.equal(captured_common_attn_metadata.block_table_tensor, torch.eye(256, dtype=torch.int32)[0].unsqueeze(0))
            assert torch.equal(captured_common_attn_metadata.num_computed_tokens_cpu, torch.tensor([2]))
            if model_type == 'qwen_moe':
                assert captured_common_attn_metadata._seq_lens_cpu == torch.tensor([15])
            if model_type == 'qwen_dense':
                assert captured_common_attn_metadata._seq_lens_cpu is None
            if model_type == 'deepseek':
                assert torch.equal(captured_common_attn_metadata.seq_lens, torch.tensor([11]))
                assert captured_common_attn_metadata.max_seq_len == 9
                assert torch.equal(captured_common_attn_metadata.slot_mapping, torch.cat([torch.tensor([138]), torch.full((8703,), -1)]))
                assert torch.equal(captured_common_attn_metadata.seq_lens_cpu, torch.tensor([11]))
                assert captured_common_attn_metadata._seq_lens_cpu == torch.tensor([11])
                assert torch.equal(captured_common_attn_metadata.positions, torch.tensor([10, 1, 2, 3, 4, 5, 6, 7, 8] + [0]*(8704-9), dtype=torch.int64))
                assert captured_common_attn_metadata.num_input_tokens == 9
            if model_type == 'qwen_dense' or model_type == 'qwen_moe':
                assert torch.equal(captured_common_attn_metadata.seq_lens, torch.tensor([15]))
                assert captured_common_attn_metadata.max_seq_len == 13
                assert torch.equal(captured_common_attn_metadata.slot_mapping, torch.cat([torch.tensor([142]), torch.full((8703,), -1)]))
                assert torch.equal(captured_common_attn_metadata.seq_lens_cpu, torch.tensor([15]))
                assert torch.equal(captured_common_attn_metadata.positions, torch.tensor([14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] + [0]*(8704-13), dtype=torch.int64))
                assert captured_common_attn_metadata.num_input_tokens == 13

        if self.is_decode(flag_prefill_decode):
            if graphmode == 'full':
                assert torch.equal(captured_common_attn_metadata.query_start_loc, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))
                assert torch.equal(captured_common_attn_metadata.query_start_loc_cpu, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))
                assert captured_common_attn_metadata.num_reqs == 16
                assert torch.equal(captured_common_attn_metadata.block_table_tensor, torch.cat([torch.eye(256, device="cpu", dtype=torch.int32)[0].unsqueeze(0)*i for i in [1,2,3]]
                                                                                                + [torch.zeros(13, 256, device="cpu", dtype=torch.int32)], dim=0))
                assert captured_common_attn_metadata.num_input_tokens == 16
            else:
                assert torch.equal(captured_common_attn_metadata.query_start_loc, torch.tensor([0, 1, 2, 3]))
                assert torch.equal(captured_common_attn_metadata.query_start_loc_cpu, torch.tensor([0, 1, 2, 3]))
                assert captured_common_attn_metadata.num_reqs == 3
                assert torch.equal(captured_common_attn_metadata.block_table_tensor, torch.cat([torch.eye(256, device="cpu", dtype=torch.int32)[0].unsqueeze(0)*i for i in [1,2,3]], dim=0))
                assert captured_common_attn_metadata.num_input_tokens == 12
            assert captured_common_attn_metadata.num_actual_tokens == 3
            assert captured_common_attn_metadata.max_query_len == 1
            assert captured_common_attn_metadata.max_seq_len == 0
            assert captured_common_attn_metadata._seq_lens_cpu is None
            if model_type == 'qwen_dense':
                if graphmode == 'full':
                    assert torch.equal(captured_common_attn_metadata.seq_lens, torch.tensor([23, 19, 19] + [0]*13))
                    assert torch.equal(captured_common_attn_metadata.seq_lens_cpu, torch.tensor([23, 19, 19] + [0]*13))
                    assert torch.equal(captured_common_attn_metadata.num_computed_tokens_cpu, torch.tensor([19, 15, 15] + [0]*13))
                    assert torch.equal(captured_common_attn_metadata.positions, torch.tensor([20, 18, 18, 20, 13, 14, 15, 16, 13, 14, 15, 16, 0, 0, 0, 0, 12, 0, 1,
                                                                                            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] + [0]*(8704-30), dtype=torch.int64))
                else:
                    assert torch.equal(captured_common_attn_metadata.seq_lens, torch.tensor([23, 19, 19]))
                    assert torch.equal(captured_common_attn_metadata.seq_lens_cpu, torch.tensor([23, 19, 19]))
                    assert torch.equal(captured_common_attn_metadata.num_computed_tokens_cpu, torch.tensor([19, 15, 15]))
                    assert torch.equal(captured_common_attn_metadata.positions, torch.tensor([20, 18, 18, 20, 13, 14, 15, 16, 13, 14, 15, 16, 8, 9, 10, 11, 12, 0, 1,
                                                                                            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] + [0]*(8704-30), dtype=torch.int64))
                assert torch.equal(captured_common_attn_metadata.slot_mapping, torch.cat([torch.tensor([148, 274, 402]), torch.full((8701,), -1)]))
            if model_type == 'qwen_moe':
                assert torch.equal(captured_common_attn_metadata.seq_lens, torch.tensor([21, 19, 19]))
                assert torch.equal(captured_common_attn_metadata.slot_mapping, torch.cat([torch.tensor([145, 272, 400]), torch.full((8701,), -1)]))
                assert torch.equal(captured_common_attn_metadata.seq_lens_cpu, torch.tensor([21, 19, 19]))
                assert torch.equal(captured_common_attn_metadata.num_computed_tokens_cpu, torch.tensor([17, 15, 15]))
                assert torch.equal(captured_common_attn_metadata.positions, torch.tensor([17, 16, 16, 18, 13, 14, 15, 16, 13, 14, 15, 16, 8, 9, 10, 11, 12, 0, 1,
                                                                                          2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] + [0]*(8704-30), dtype=torch.int64))
            if model_type == 'deepseek':
                assert torch.equal(captured_common_attn_metadata.seq_lens, torch.tensor([16, 15, 16]))
                assert torch.equal(captured_common_attn_metadata.slot_mapping, torch.cat([torch.tensor([142, 268, 396]), torch.full((8701,), -1)]))
                assert torch.equal(captured_common_attn_metadata.seq_lens_cpu, torch.tensor([16, 15, 16]))
                assert torch.equal(captured_common_attn_metadata.num_computed_tokens_cpu, torch.tensor([12, 11, 12]))
                assert torch.equal(captured_common_attn_metadata.positions, torch.tensor([14, 12, 12, 13, 9, 10, 11, 12, 10, 11, 12, 13, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + [0]*(8704-23), dtype=torch.int64))
        assert captured_common_attn_metadata.causal
        assert captured_common_attn_metadata.logits_indices_padded is None
        assert captured_common_attn_metadata.num_logits_indices is None
        assert captured_common_attn_metadata.encoder_seq_lens is None
        assert captured_common_attn_metadata.encoder_seq_lens_cpu is None
        assert captured_common_attn_metadata.dcp_local_seq_lens is None
        assert captured_common_attn_metadata.dcp_local_seq_lens_cpu is None
        assert captured_common_attn_metadata._num_computed_tokens_cpu is None
        assert captured_common_attn_metadata._num_computed_tokens_cache is None
        assert captured_common_attn_metadata.decode_token_per_req == 1
        assert captured_common_attn_metadata.actual_seq_lengths_q == []
        if model_type == 'deepseek':
            assert captured_common_attn_metadata.attn_state == AscendAttentionState.SpecDecoding
        else:
            assert captured_common_attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill
        assert captured_common_attn_metadata.graph_pad_size == -1
        assert captured_common_attn_metadata.prefill_context_parallel_metadata is None

    # prefill or decode
    def is_decode(self, flag_prefill_decode):
        if flag_prefill_decode == "decode":
            return True
        if flag_prefill_decode == "prefill":
            return False

    # Add assertions to ensure that the mocked functions and parameters exist
    def check_mock(self):
        import vllm.config
        assert hasattr(vllm.config, "VllmConfig"), "VllmConfig not found"

        fields = {
            "speculative_config",
            "scheduler_config",
            "model_config",
            "parallel_config",
            "additional_config",
        }

        actual = set(vllm.config.VllmConfig.__dataclass_fields__)
        missing = fields - actual

        assert not missing, f"Missing dataclass fields: {missing}"


        assert hasattr(vllm.config, "SpeculativeConfig"), "SpeculativeConfig not found"
        fields = {
            "num_speculative_tokens",
            "method",
            "parallel_drafting",
            "draft_tensor_parallel_size",
            "speculative_token_tree",
            "draft_model_config",
            "disable_padded_drafter_batch",
        }

        actual = set(vllm.config.SpeculativeConfig.__dataclass_fields__)
        missing = fields - actual

        assert not missing, f"Missing dataclass fields: {missing}"


        assert hasattr(vllm.config, "SchedulerConfig")
        assert "max_num_batched_tokens" in vllm.config.SchedulerConfig.__dataclass_fields__
        assert "max_num_seqs" in vllm.config.SchedulerConfig.__dataclass_fields__

        assert hasattr(vllm.config, "ModelConfig")
        assert "dtype" in vllm.config.ModelConfig.__dataclass_fields__
        assert "max_model_len" in vllm.config.ModelConfig.__dataclass_fields__

        assert isinstance(
            inspect.getattr_static(vllm.config.ModelConfig, "uses_mrope"),
            property
        )
        assert isinstance(
            inspect.getattr_static(vllm.config.ModelConfig, "uses_xdrope_dim"),
            property
        )
        assert isinstance(
            inspect.getattr_static(vllm.config.ModelConfig, "use_mla"),
            property
        )

        assert hasattr(vllm.config, "ParallelConfig"), "ParallelConfig not found"
        fields = {
            "tensor_parallel_size",
            "data_parallel_rank",
            "data_parallel_size",
            "prefill_context_parallel_size",
        }

        actual = set(vllm.config.ParallelConfig.__dataclass_fields__)
        missing = fields - actual

        assert not missing, f"Missing dataclass fields: {missing}"


        import vllm_ascend.worker.model_runner_v1
        assert hasattr(vllm_ascend.worker.model_runner_v1, "NPUModelRunner")
        RunnerCls = vllm_ascend.worker.model_runner_v1.NPUModelRunner
        src = inspect.getsource(RunnerCls.__init__)
        fields = {
            "pcp_size",
            "dcp_size",
            "max_num_tokens",
            "max_num_reqs",
            "pin_memory",
            "query_start_loc",
        }

        for f in fields:
            assert f"self.{f}" in src, f"missing self.{f} in __init__"

        assert hasattr(RunnerCls, "_sync_metadata_across_dp")
        sig = inspect.signature(RunnerCls._sync_metadata_across_dp)
        sig_name = self.get_param_names(sig)
        assert sig_name == ['self', 'num_tokens', 'is_draft_model', 'cudagraph_mode', 'allow_dp_padding']

        assert hasattr(RunnerCls, "_pad_query_start_loc_for_fia")
        sig = inspect.signature(RunnerCls._pad_query_start_loc_for_fia)
        sig_name = self.get_param_names(sig)
        assert sig_name == ['self', 'num_tokens_padded', 'num_reqs_padded', 'num_reqs', 'cudagraph_runtime_mode', 'batch_desc_num_reqs']


        import vllm_ascend.spec_decode.eagle_proposer
        assert hasattr(vllm_ascend.spec_decode.eagle_proposer, "AscendSpecDecodeBaseProposer")
        RunnerCls = vllm_ascend.spec_decode.eagle_proposer.AscendSpecDecodeBaseProposer
        assert hasattr(RunnerCls, "_get_model")
        assert hasattr(RunnerCls, "_update_full_graph_params")
        assert hasattr(RunnerCls, "_propose")
        sig = inspect.signature(RunnerCls._get_model)
        sig_name = self.get_param_names(sig)
        assert sig_name == ['self']
        sig = inspect.signature(RunnerCls._update_full_graph_params)
        sig_name = self.get_param_names(sig)
        assert sig_name == ['self', 'forward_context', 'num_tokens', 'draft_attn_metadatas']
        src = inspect.getsource(RunnerCls.load_model)
        assert 'self.attn_layer_names' in src
        assert 'self.kernel_block_size' in src
        sig = inspect.signature(RunnerCls._propose)
        sig_name = self.get_param_names(sig)
        assert sig_name == ['self', 'target_token_ids', 'target_positions', 'target_hidden_states', 'next_token_ids',
                            'token_indices_to_sample', 'common_attn_metadata', 'target_model_batch_desc',
                            'sampling_metadata', 'mm_embed_inputs', 'req_scheduled_tokens', 'long_seq_metadata',
                            'num_prefill_reqs', 'num_decode_reqs', 'scheduler_output', 'num_scheduled_tokens',
                            'num_rejected_tokens_gpu'
                        ]


        import vllm.model_executor.models.llama_eagle3
        assert hasattr(vllm.model_executor.models.llama_eagle3, "Eagle3LlamaForCausalLM")
        RunnerCls = vllm.model_executor.models.llama_eagle3.Eagle3LlamaForCausalLM
        assert hasattr(RunnerCls, "combine_hidden_states")
        sig = inspect.signature(RunnerCls.combine_hidden_states)
        sig_name = self.get_param_names(sig)
        assert sig_name == ['self', 'hidden_states']


        import vllm.v1.spec_decode.eagle
        assert hasattr(vllm.v1.spec_decode.eagle, 'SpecDecodeBaseProposer')
        RunnerCls = vllm.v1.spec_decode.eagle.SpecDecodeBaseProposer
        src = inspect.getsource(RunnerCls.__init__)
        assert 'self.hidden_size' in src
        assert 'self.draft_attn_groups' in src


        import vllm.v1.worker.gpu_model_runner
        assert hasattr(vllm.v1.worker.gpu_model_runner, 'GPUModelRunner')
        RunnerCls = vllm.v1.worker.gpu_model_runner.GPUModelRunner
        src = inspect.getsource(RunnerCls.__init__)
        assert 'self.cudagraph_dispatcher' in src
        assert 'self.seq_lens' in src
        assert 'self.optimistic_seq_lens_cpu' in src


        import vllm.v1.cudagraph_dispatcher
        assert hasattr(vllm.v1.cudagraph_dispatcher, 'CudagraphDispatcher')
        assert hasattr(vllm.v1.cudagraph_dispatcher.CudagraphDispatcher, 'dispatch')
        RunnerCls = vllm.v1.cudagraph_dispatcher.CudagraphDispatcher
        sig = inspect.signature(RunnerCls.dispatch)
        sig_name = self.get_param_names(sig)
        assert sig_name == ['self', 'num_tokens', 'uniform_decode', 'has_lora', 'num_active_loras', 'valid_modes', 'invalid_modes']


        import vllm.v1.attention.backend
        assert hasattr(vllm.v1.attention.backend, 'CommonAttentionMetadata')
        fields = {
            'query_start_loc', 'query_start_loc_cpu', 'seq_lens', 'num_reqs', \
            'num_actual_tokens', 'max_query_len', 'max_seq_len', 'block_table_tensor', \
            'slot_mapping', 'causal', 'logits_indices_padded', 'num_logits_indices', \
            'encoder_seq_lens', 'encoder_seq_lens_cpu', 'dcp_local_seq_lens', \
            'dcp_local_seq_lens_cpu', '_seq_lens_cpu', '_num_computed_tokens_cpu', \
            '_num_computed_tokens_cache'
        }

        actual = set(vllm.v1.attention.backend.CommonAttentionMetadata.__dataclass_fields__)
        missing = fields - actual

        assert not missing, f"Missing dataclass fields: {missing}"


        import vllm_ascend.attention.utils
        assert hasattr(vllm_ascend.attention.utils, 'AscendCommonAttentionMetadata')
        fields = {
            'positions', 'seq_lens_cpu', 'decode_token_per_req', \
            'prefill_context_parallel_metadata', 'actual_seq_lengths_q', \
            'attn_state', 'num_computed_tokens_cpu', 'num_input_tokens', \
            'graph_pad_size'
        }

        actual = set(vllm_ascend.attention.utils.AscendCommonAttentionMetadata.__dataclass_fields__)
        missing = fields - actual

        assert not missing, f"Missing dataclass fields: {missing}"


        import vllm_ascend.spec_decode.eagle_proposer
        assert hasattr(vllm_ascend.spec_decode.eagle_proposer, "AscendSpecDecodeBaseProposer")
        RunnerCls = vllm_ascend.spec_decode.eagle_proposer.AscendSpecDecodeBaseProposer
        assert hasattr(RunnerCls, "_run_merged_draft")
        sig = inspect.signature(RunnerCls._run_merged_draft)
        sig_name = self.get_param_names(sig)
        assert sig_name == ['self', 'num_input_tokens', 'batch_size', 'token_indices_to_sample',
                            'target_positions', 'inputs_embeds', 'multi_steps_attn_metadata',
                            'num_tokens', 'is_prefill'
                        ]


        import vllm.v1.worker.utils
        assert hasattr(vllm.v1.worker.utils, "AttentionGroup")
        assert hasattr(vllm.v1.worker.utils.AttentionGroup, "get_metadata_builder")
        fields = {
            'backend', 'layer_names', 'kv_cache_spec', \
            'kv_cache_group_id'
        }

        actual = set(vllm.v1.worker.utils.AttentionGroup.__dataclass_fields__)
        missing = fields - actual

        assert not missing, f"Missing dataclass fields: {missing}"


        import vllm.v1.attention.backend
        assert hasattr(vllm.v1.attention.backend, "AttentionMetadataBuilder")
        assert hasattr(vllm.v1.attention.backend.AttentionMetadataBuilder, "build")
        assert hasattr(vllm.v1.attention.backend.AttentionMetadataBuilder, "build_for_drafting")
        RunnerCls = vllm.v1.attention.backend.AttentionMetadataBuilder
        sig = inspect.signature(RunnerCls.build)
        sig_name = self.get_param_names(sig)
        assert sig_name == ['self', 'common_prefix_len', 'common_attn_metadata', 'fast_build']


    # get the param in inspect sig, for check_mock()
    def get_param_names(self, sig):
        return [p.name for p in sig.parameters.values()]
# fmt: on


class MockCpuGpuBuffer:
    """Mock CpuGpuBuffer for testing"""

    def __init__(self, max_size, dtype, device="cpu", **kwargs):
        self.max_size = max_size
        self.dtype = dtype
        self.device = device
        self.cpu = torch.zeros(max_size, dtype=dtype, device="cpu")
        self.np = self.cpu.numpy()
        self.gpu = torch.zeros(max_size, dtype=dtype, device=device)

    def copy_to_gpu(self, size=None):
        if size is None:
            size = self.max_size
        self.gpu[:size].copy_(self.cpu[:size])


class MockCachedRequestState:
    """Mock CachedRequestState for testing"""

    def __init__(self, req_id, token_ids):
        self.req_id = req_id
        self.token_ids = token_ids

    def get_token_id(self, position):
        if position < len(self.token_ids):
            return self.token_ids[position]
        return 0


class MockInputBatch:
    """Mock InputBatch for testing"""

    def __init__(self, num_reqs, req_ids, vocab_size, num_tokens_no_spec=None):
        self.num_reqs = num_reqs
        self.req_ids = req_ids
        self.vocab_size = vocab_size
        # num_tokens_no_spec represents the sequence length (excluding speculative tokens)
        # for each request. Default to seq_len + 1 for each request.
        if num_tokens_no_spec is None:
            self.num_tokens_no_spec = np.array([i + 11 for i in range(num_reqs)], dtype=np.int64)
        else:
            self.num_tokens_no_spec = np.array(num_tokens_no_spec, dtype=np.int64)


class TestPrepareNextTokenIdsPadded(TestBase):
    """Test prepare_next_token_ids_padded method with precision validation"""

    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.cache_config = MagicMock(spec=CacheConfig)
        self.vllm_config.scheduler_config = MagicMock()
        self.vllm_config.model_config = MagicMock()
        self.vllm_config.model_config.hf_text_config = MagicMock(spec=[])
        self.vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        self.vllm_config.compilation_config = MagicMock()
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.enable_expert_parallel = False
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.num_speculative_tokens = 4
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(4)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer", MockCpuGpuBuffer)
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        set_current_vllm_config(None)

    def test_all_valid_tokens(self):
        """Test case where all requests have valid sampled tokens"""
        num_reqs = 3
        vocab_size = 1000

        sampled_token_ids = torch.tensor(
            [
                [100, 101, 102, 103, 104],
                [200, 201, 202, 203, 204],
                [300, 301, 302, 303, 304],
            ],
            dtype=torch.int64,
        )

        requests = {
            "req_0": MockCachedRequestState("req_0", list(range(10))),
            "req_1": MockCachedRequestState("req_1", list(range(15))),
            "req_2": MockCachedRequestState("req_2", list(range(20))),
        }

        gpu_input_batch = MockInputBatch(
            num_reqs=num_reqs,
            req_ids=["req_0", "req_1", "req_2"],
            vocab_size=vocab_size,
            num_tokens_no_spec=[11, 16, 21],  # seq_len = num_tokens_no_spec - 1 = [10, 15, 20]
        )

        discard_request_indices = torch.tensor([], dtype=torch.int64)
        num_discarded_requests = 0

        next_token_ids, valid_sampled_tokens_count = self.proposer.prepare_next_token_ids_padded(
            sampled_token_ids=sampled_token_ids,
            requests=requests,
            gpu_input_batch=gpu_input_batch,
            discard_request_indices=discard_request_indices,
            num_discarded_requests=num_discarded_requests,
        )

        self.assertEqual(next_token_ids.shape[0], num_reqs)
        self.assertEqual(valid_sampled_tokens_count.shape[0], num_reqs)

        expected_valid_counts = torch.tensor([5, 5, 5], dtype=torch.int64)
        self.assertTrue(torch.equal(valid_sampled_tokens_count, expected_valid_counts))

        expected_next_tokens = torch.tensor([104, 204, 304], dtype=torch.int64)
        self.assertTrue(torch.equal(next_token_ids, expected_next_tokens))

    def test_partial_rejected_tokens(self):
        """Test case where some tokens are rejected (marked as -1)"""
        num_reqs = 3
        vocab_size = 1000

        sampled_token_ids = torch.tensor(
            [
                [100, 101, -1, -1, -1],
                [200, 201, 202, 203, -1],
                [300, 301, 302, 303, 304],
            ],
            dtype=torch.int64,
        )

        requests = {
            "req_0": MockCachedRequestState("req_0", list(range(10))),
            "req_1": MockCachedRequestState("req_1", list(range(15))),
            "req_2": MockCachedRequestState("req_2", list(range(20))),
        }

        gpu_input_batch = MockInputBatch(
            num_reqs=num_reqs,
            req_ids=["req_0", "req_1", "req_2"],
            vocab_size=vocab_size,
            num_tokens_no_spec=[11, 16, 21],  # seq_len = num_tokens_no_spec - 1 = [10, 15, 20]
        )

        discard_request_indices = torch.tensor([], dtype=torch.int64)
        num_discarded_requests = 0

        next_token_ids, valid_sampled_tokens_count = self.proposer.prepare_next_token_ids_padded(
            sampled_token_ids=sampled_token_ids,
            requests=requests,
            gpu_input_batch=gpu_input_batch,
            discard_request_indices=discard_request_indices,
            num_discarded_requests=num_discarded_requests,
        )

        expected_valid_counts = torch.tensor([2, 4, 5], dtype=torch.int64)
        self.assertTrue(torch.equal(valid_sampled_tokens_count, expected_valid_counts))

        expected_next_tokens = torch.tensor([101, 203, 304], dtype=torch.int64)
        self.assertTrue(torch.equal(next_token_ids, expected_next_tokens))

    def test_all_rejected_tokens_with_backup(self):
        """Test case where all tokens are rejected, should use backup token"""
        num_reqs = 3
        vocab_size = 1000

        sampled_token_ids = torch.tensor(
            [
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [300, 301, 302, 303, 304],
            ],
            dtype=torch.int64,
        )

        requests = {
            "req_0": MockCachedRequestState("req_0", list(range(15))),
            "req_1": MockCachedRequestState("req_1", list(range(20))),
            "req_2": MockCachedRequestState("req_2", list(range(25))),
        }

        gpu_input_batch = MockInputBatch(
            num_reqs=num_reqs,
            req_ids=["req_0", "req_1", "req_2"],
            vocab_size=vocab_size,
            num_tokens_no_spec=[11, 16, 26],  # seq_len = num_tokens_no_spec - 1 = [10, 15, 25]
        )

        discard_request_indices = torch.tensor([], dtype=torch.int64)
        num_discarded_requests = 0

        next_token_ids, valid_sampled_tokens_count = self.proposer.prepare_next_token_ids_padded(
            sampled_token_ids=sampled_token_ids,
            requests=requests,
            gpu_input_batch=gpu_input_batch,
            discard_request_indices=discard_request_indices,
            num_discarded_requests=num_discarded_requests,
        )

        expected_valid_counts = torch.tensor([0, 0, 5], dtype=torch.int64)
        self.assertTrue(torch.equal(valid_sampled_tokens_count, expected_valid_counts))

        expected_backup_token_0 = requests["req_0"].get_token_id(10)
        expected_backup_token_1 = requests["req_1"].get_token_id(15)
        expected_next_tokens = torch.tensor([expected_backup_token_0, expected_backup_token_1, 304], dtype=torch.int64)
        self.assertTrue(torch.equal(next_token_ids, expected_next_tokens))

    def test_discarded_requests(self):
        """Test case with discarded requests"""
        num_reqs = 3
        vocab_size = 1000

        sampled_token_ids = torch.tensor(
            [
                [100, 101, 102, 103, 104],
                [200, 201, 202, 203, 204],
                [300, 301, 302, 303, 304],
            ],
            dtype=torch.int64,
        )

        requests = {
            "req_0": MockCachedRequestState("req_0", list(range(15))),
            "req_1": MockCachedRequestState("req_1", list(range(20))),
            "req_2": MockCachedRequestState("req_2", list(range(25))),
        }

        gpu_input_batch = MockInputBatch(
            num_reqs=num_reqs,
            req_ids=["req_0", "req_1", "req_2"],
            vocab_size=vocab_size,
            num_tokens_no_spec=[11, 21, 21],  # seq_len = num_tokens_no_spec - 1 = [10, 20, 20]
        )

        discard_request_indices = torch.tensor([0, 2], dtype=torch.int64)
        num_discarded_requests = 2

        next_token_ids, valid_sampled_tokens_count = self.proposer.prepare_next_token_ids_padded(
            sampled_token_ids=sampled_token_ids,
            requests=requests,
            gpu_input_batch=gpu_input_batch,
            discard_request_indices=discard_request_indices,
            num_discarded_requests=num_discarded_requests,
        )

        expected_valid_counts = torch.tensor([0, 5, 0], dtype=torch.int64)
        self.assertTrue(torch.equal(valid_sampled_tokens_count, expected_valid_counts))

        expected_backup_token_0 = requests["req_0"].get_token_id(10)
        expected_backup_token_2 = requests["req_2"].get_token_id(20)
        expected_next_tokens = torch.tensor([expected_backup_token_0, 204, expected_backup_token_2], dtype=torch.int64)
        self.assertTrue(torch.equal(next_token_ids, expected_next_tokens))

    def test_mixed_scenario(self):
        """Test mixed scenario: some rejected tokens, some discarded requests, some all-rejected"""
        num_reqs = 4
        vocab_size = 1000

        sampled_token_ids = torch.tensor(
            [
                [100, 101, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [300, 301, 302, 303, 304],
                [400, 401, 402, -1, -1],
            ],
            dtype=torch.int64,
        )

        requests = {
            "req_0": MockCachedRequestState("req_0", list(range(15))),
            "req_1": MockCachedRequestState("req_1", list(range(20))),
            "req_2": MockCachedRequestState("req_2", list(range(25))),
            "req_3": MockCachedRequestState("req_3", list(range(30))),
        }

        gpu_input_batch = MockInputBatch(
            num_reqs=num_reqs,
            req_ids=["req_0", "req_1", "req_2", "req_3"],
            vocab_size=vocab_size,
            num_tokens_no_spec=[11, 16, 26, 31],  # seq_len = num_tokens_no_spec - 1 = [10, 15, 25, 30]
        )

        discard_request_indices = torch.tensor([1], dtype=torch.int64)
        num_discarded_requests = 1

        next_token_ids, valid_sampled_tokens_count = self.proposer.prepare_next_token_ids_padded(
            sampled_token_ids=sampled_token_ids,
            requests=requests,
            gpu_input_batch=gpu_input_batch,
            discard_request_indices=discard_request_indices,
            num_discarded_requests=num_discarded_requests,
        )

        expected_valid_counts = torch.tensor([2, 0, 5, 3], dtype=torch.int64)
        self.assertTrue(torch.equal(valid_sampled_tokens_count, expected_valid_counts))

        expected_backup_token_1 = requests["req_1"].get_token_id(15)
        expected_next_tokens = torch.tensor([101, expected_backup_token_1, 304, 402], dtype=torch.int64)
        self.assertTrue(torch.equal(next_token_ids, expected_next_tokens))

    def test_single_request(self):
        """Test with single request"""
        num_reqs = 1
        vocab_size = 1000

        sampled_token_ids = torch.tensor([[100, 101, 102, 103, 104]], dtype=torch.int64)

        requests = {
            "req_0": MockCachedRequestState("req_0", list(range(15))),
        }

        gpu_input_batch = MockInputBatch(
            num_reqs=num_reqs,
            req_ids=["req_0"],
            vocab_size=vocab_size,
            num_tokens_no_spec=[16],  # seq_len = num_tokens_no_spec - 1 = [15]
        )

        discard_request_indices = torch.tensor([], dtype=torch.int64)
        num_discarded_requests = 0

        next_token_ids, valid_sampled_tokens_count = self.proposer.prepare_next_token_ids_padded(
            sampled_token_ids=sampled_token_ids,
            requests=requests,
            gpu_input_batch=gpu_input_batch,
            discard_request_indices=discard_request_indices,
            num_discarded_requests=num_discarded_requests,
        )

        expected_valid_counts = torch.tensor([5], dtype=torch.int64)
        self.assertTrue(torch.equal(valid_sampled_tokens_count, expected_valid_counts))

        expected_next_tokens = torch.tensor([104], dtype=torch.int64)
        self.assertTrue(torch.equal(next_token_ids, expected_next_tokens))

    def test_vocab_size_boundary(self):
        """Test with tokens at vocab size boundary"""
        num_reqs = 2
        vocab_size = 100

        sampled_token_ids = torch.tensor(
            [
                [99, 100, 101, -1, -1],
                [50, 51, 52, 53, 54],
            ],
            dtype=torch.int64,
        )

        requests = {
            "req_0": MockCachedRequestState("req_0", list(range(15))),
            "req_1": MockCachedRequestState("req_1", list(range(20))),
        }

        gpu_input_batch = MockInputBatch(
            num_reqs=num_reqs,
            req_ids=["req_0", "req_1"],
            vocab_size=vocab_size,
            num_tokens_no_spec=[16, 21],  # seq_len = num_tokens_no_spec - 1 = [15, 20]
        )

        discard_request_indices = torch.tensor([], dtype=torch.int64)
        num_discarded_requests = 0

        next_token_ids, valid_sampled_tokens_count = self.proposer.prepare_next_token_ids_padded(
            sampled_token_ids=sampled_token_ids,
            requests=requests,
            gpu_input_batch=gpu_input_batch,
            discard_request_indices=discard_request_indices,
            num_discarded_requests=num_discarded_requests,
        )

        # Token 100 and 101 are >= vocab_size (100), so they are invalid
        # Only token 99 is valid for the first request
        expected_valid_counts = torch.tensor([1, 5], dtype=torch.int64)
        self.assertTrue(torch.equal(valid_sampled_tokens_count, expected_valid_counts))

        # Next token should be 99 (last valid token) for first request
        # and 54 for second request
        expected_next_tokens = torch.tensor([99, 54], dtype=torch.int64)
        self.assertTrue(torch.equal(next_token_ids, expected_next_tokens))

    def test_intermediate_variables_precision(self):
        """Test to verify key variables that affect downstream computation"""
        num_reqs = 2
        vocab_size = 1000

        sampled_token_ids = torch.tensor(
            [
                [100, 101, -1, -1, -1],
                [200, 201, 202, -1, -1],
            ],
            dtype=torch.int64,
        )

        requests = {
            "req_0": MockCachedRequestState("req_0", list(range(15))),
            "req_1": MockCachedRequestState("req_1", list(range(20))),
        }

        gpu_input_batch = MockInputBatch(
            num_reqs=num_reqs,
            req_ids=["req_0", "req_1"],
            vocab_size=vocab_size,
            num_tokens_no_spec=[11, 16],  # seq_len = num_tokens_no_spec - 1 = [10, 15]
        )

        discard_request_indices = torch.tensor([], dtype=torch.int64)
        num_discarded_requests = 0

        next_token_ids, valid_sampled_tokens_count = self.proposer.prepare_next_token_ids_padded(
            sampled_token_ids=sampled_token_ids,
            requests=requests,
            gpu_input_batch=gpu_input_batch,
            discard_request_indices=discard_request_indices,
            num_discarded_requests=num_discarded_requests,
        )

        # Verify return values
        self.assertEqual(next_token_ids[0].item(), 101, "next_token_ids[0] should be 101")
        self.assertEqual(next_token_ids[1].item(), 202, "next_token_ids[1] should be 202")
        self.assertEqual(valid_sampled_tokens_count[0].item(), 2, "valid_sampled_tokens_count[0] should be 2")
        self.assertEqual(valid_sampled_tokens_count[1].item(), 3, "valid_sampled_tokens_count[1] should be 3")

        # Verify public member that affects downstream computation
        expected_backup_0 = requests["req_0"].get_token_id(10)
        expected_backup_1 = requests["req_1"].get_token_id(15)
        self.assertEqual(
            self.proposer.backup_next_token_ids.np[0],
            expected_backup_0,
            f"backup_next_token_ids[0] should be {expected_backup_0}",
        )
        self.assertEqual(
            self.proposer.backup_next_token_ids.np[1],
            expected_backup_1,
            f"backup_next_token_ids[1] should be {expected_backup_1}",
        )

        # Verify data types
        self.assertEqual(next_token_ids.dtype, torch.int64, "next_token_ids dtype should be torch.int64")
        self.assertEqual(
            valid_sampled_tokens_count.dtype, torch.int64, "valid_sampled_tokens_count dtype should be torch.int64"
        )


# fmt: off
class MockDraftModel:
    """Draft model that records prepared forward inputs."""

    def __init__(self, returns_tuple=True, vocab_size=200000):
        self.returns_tuple = returns_tuple
        self.vocab_size = vocab_size
        self.calls = []
        self.logit_inputs = []
        self.returned_hidden_states = []

    def __call__(self, **kwargs):
        self.calls.append({key: value.clone() if torch.is_tensor(value) else value for key, value in kwargs.items()})
        input_ids = kwargs["input_ids"].to(torch.long)
        call_idx = len(self.returned_hidden_states)

        last_hidden_states = torch.zeros(input_ids.shape[0], 4, dtype=torch.float32)
        last_hidden_states[:, 0] = input_ids + call_idx
        last_hidden_states[:, 1] = 100 + input_ids + call_idx

        hidden_states = torch.zeros_like(last_hidden_states)
        hidden_states[:, 0] = 1000 + input_ids + call_idx
        hidden_states[:, 1] = 2000 + input_ids + call_idx

        self.returned_hidden_states.append((last_hidden_states.clone(), hidden_states.clone()))
        if self.returns_tuple:
            return last_hidden_states, hidden_states
        return last_hidden_states

    def compute_logits(self, sample_hidden_states):
        self.logit_inputs.append(sample_hidden_states.clone())
        token_ids = sample_hidden_states[:, 0].to(torch.long)
        logits = torch.full((sample_hidden_states.shape[0], self.vocab_size), -1000.0)
        logits[torch.arange(sample_hidden_states.shape[0]), token_ids] = 1000.0
        return logits

    def embed_input_ids(self, input_ids):
        return torch.stack((input_ids.float() + 5000, input_ids.float() + 6000), dim=1).repeat(1, 2)


class TestRunMergedDraft(TestBase):

    def setUp(self):
        self.check_mock()

        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.cache_config = MagicMock(spec=CacheConfig)
        self.vllm_config.scheduler_config = MagicMock()
        self.vllm_config.model_config = MagicMock()
        self.vllm_config.model_config.hf_text_config = MagicMock(spec=[])
        self.vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        self.vllm_config.compilation_config = MagicMock()
        self.vllm_config.compilation_config.mode = CompilationMode.NONE
        self.vllm_config.compilation_config.pass_config = MagicMock()
        self.vllm_config.compilation_config.pass_config.enable_sp = False

        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1
        self.runner.pcp_rank = 0
        self.runner.dcp_rank = 0
        self.runner.max_num_tokens = 64
        self.runner.max_num_reqs = 8
        self.runner.uniform_decode_query_len = 2
        self.runner.enable_enpu = False
        self.runner.use_eagle = True
        self.runner._use_aclgraph.return_value = False
        self.runner._make_buffer.side_effect = lambda size, dtype: torch.zeros(size, dtype=dtype, device=self.device)

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.async_scheduling = False
        self.vllm_config.scheduler_config.max_num_batched_tokens = 64
        self.vllm_config.scheduler_config.max_num_seqs = 8
        self.vllm_config.model_config.dtype = torch.float32
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.model_config.use_mla = False
        self.vllm_config.model_config.is_multimodal_model = False
        self.vllm_config.model_config.enforce_eager = True
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.enable_expert_parallel = False
        self.vllm_config.speculative_config.method = "eagle3"
        self.vllm_config.speculative_config.num_speculative_tokens = 3
        self.vllm_config.speculative_config.parallel_drafting = False
        self.vllm_config.speculative_config.enforce_eager = True
        self.vllm_config.speculative_config.use_local_argmax_reduction = False
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(3)])
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 4
        self.vllm_config.speculative_config.draft_model_config.get_inputs_embeds_size.return_value = 4
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer", MockCpuGpuBuffer)
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()
        self.mock_enable_sp = patch("vllm_ascend.spec_decode.eagle_proposer.enable_sp", return_value=False)
        self.mock_enable_sp.start()
        self.mock_shared_expert_dp = patch(
            "vllm_ascend.spec_decode.eagle_proposer.shared_expert_dp_enabled", return_value=False
        )
        self.mock_shared_expert_dp.start()
        self.mock_extra_ctx = patch("vllm_ascend.spec_decode.eagle_proposer._EXTRA_CTX", new=MagicMock())
        self.mock_extra_ctx.start()
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)
        self.proposer.maybe_pad_and_reduce = MagicMock(
            side_effect=lambda hidden_states, positions: (hidden_states, positions)
        )
        self.proposer.maybe_all_gather_and_unpad = MagicMock(
            side_effect=lambda last_hidden_states, positions, hidden_states: (
                last_hidden_states,
                positions,
                hidden_states,
            )
        )

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        self.mock_enable_sp.stop()
        self.mock_shared_expert_dp.stop()
        self.mock_extra_ctx.stop()
        set_current_vllm_config(None)

    def check_mock(self):
        import vllm.config

        assert hasattr(vllm.config, "VllmConfig"), "VllmConfig not found"
        fields = {
            "speculative_config",
            "cache_config",
            "scheduler_config",
            "model_config",
            "parallel_config",
            "compilation_config",
            "additional_config",
        }
        actual = set(vllm.config.VllmConfig.__dataclass_fields__)
        missing = fields - actual
        assert not missing, f"Missing dataclass fields: {missing}"

        assert hasattr(vllm.config, "CacheConfig"), "CacheConfig not found"
        assert "block_size" in vllm.config.CacheConfig.__dataclass_fields__

        assert hasattr(vllm.config, "SchedulerConfig"), "SchedulerConfig not found"
        fields = {"async_scheduling", "max_num_batched_tokens", "max_num_seqs"}
        actual = set(vllm.config.SchedulerConfig.__dataclass_fields__)
        missing = fields - actual
        assert not missing, f"Missing dataclass fields: {missing}"

        assert hasattr(vllm.config, "ModelConfig"), "ModelConfig not found"
        fields = {"dtype", "max_model_len", "enforce_eager", "hf_text_config"}
        actual = set(vllm.config.ModelConfig.__dataclass_fields__)
        missing = fields - actual
        assert not missing, f"Missing dataclass fields: {missing}"
        for field in ("uses_mrope", "uses_xdrope_dim", "use_mla", "is_multimodal_model"):
            assert isinstance(inspect.getattr_static(vllm.config.ModelConfig, field), property)
        for method in ("get_hidden_size", "get_inputs_embeds_size"):
            assert hasattr(vllm.config.ModelConfig, method)

        assert hasattr(vllm.config, "ParallelConfig"), "ParallelConfig not found"
        fields = {
            "tensor_parallel_size",
            "data_parallel_rank",
            "data_parallel_size",
            "prefill_context_parallel_size",
            "enable_expert_parallel",
        }
        actual = set(vllm.config.ParallelConfig.__dataclass_fields__)
        missing = fields - actual
        assert not missing, f"Missing dataclass fields: {missing}"

        assert hasattr(vllm.config, "SpeculativeConfig"), "SpeculativeConfig not found"
        fields = {
            "method",
            "num_speculative_tokens",
            "parallel_drafting",
            "enforce_eager",
            "use_local_argmax_reduction",
            "draft_tensor_parallel_size",
            "speculative_token_tree",
            "draft_model_config",
            "disable_padded_drafter_batch",
        }
        actual = set(vllm.config.SpeculativeConfig.__dataclass_fields__)
        missing = fields - actual
        assert not missing, f"Missing dataclass fields: {missing}"

        assert hasattr(vllm.config, "CompilationConfig"), "CompilationConfig not found"
        fields = {"mode", "pass_config"}
        actual = set(vllm.config.CompilationConfig.__dataclass_fields__)
        missing = fields - actual
        assert not missing, f"Missing dataclass fields: {missing}"
        assert hasattr(vllm.config, "PassConfig"), "PassConfig not found"
        assert "enable_sp" in vllm.config.PassConfig.__dataclass_fields__

        import vllm.forward_context

        assert hasattr(vllm.forward_context, "get_forward_context")

        import vllm.multimodal.registry

        assert hasattr(vllm.multimodal.registry, "MultiModalRegistry")
        assert hasattr(vllm.multimodal.registry.MultiModalRegistry, "supports_multimodal_inputs")
        sig = inspect.signature(vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs)
        sig_name = self.get_param_names(sig)
        assert sig_name == ["self", "model_config"]

        import vllm.v1.spec_decode.eagle

        assert hasattr(vllm.v1.spec_decode.eagle, "CpuGpuBuffer")
        RunnerCls = vllm.v1.spec_decode.eagle.SpecDecodeBaseProposer
        for attr in ("_get_positions", "_set_positions"):
            assert hasattr(RunnerCls, attr), f"SpecDecodeBaseProposer.{attr} not found"
        sig = inspect.signature(RunnerCls._get_positions)
        sig_name = self.get_param_names(sig)
        assert sig_name == ["self", "num_tokens"]
        sig = inspect.signature(RunnerCls._set_positions)
        sig_name = self.get_param_names(sig)
        assert sig_name == ["self", "num_tokens", "positions"]

        import vllm.model_executor.models.llama_eagle3

        assert hasattr(vllm.model_executor.models.llama_eagle3, "Eagle3LlamaForCausalLM")
        RunnerCls = vllm.model_executor.models.llama_eagle3.Eagle3LlamaForCausalLM
        for attr in ("forward", "compute_logits", "embed_input_ids"):
            assert hasattr(RunnerCls, attr), f"Eagle3LlamaForCausalLM.{attr} not found"
        sig = inspect.signature(RunnerCls.forward)
        sig_name = self.get_param_names(sig)
        assert sig_name == ["self", "input_ids", "positions", "hidden_states", "inputs_embeds"]
        sig = inspect.signature(RunnerCls.compute_logits)
        sig_name = self.get_param_names(sig)
        assert sig_name == ["self", "hidden_states"]

        import vllm_ascend.ascend_forward_context

        assert hasattr(vllm_ascend.ascend_forward_context, "_EXTRA_CTX")
        extra_attrs = set(vllm_ascend.ascend_forward_context._ExtraForwardContextProxy.extra_attrs)
        fields = {"num_tokens", "num_accept_tokens", "flash_comm_v1_enabled"}
        missing = fields - extra_attrs
        assert not missing, f"Missing extra forward context attrs: {missing}"

        import vllm_ascend.spec_decode.eagle_proposer

        for attr in (
            "AscendEagleProposer",
            "AscendSpecDecodeBaseProposer",
            "enable_sp",
            "shared_expert_dp_enabled",
            "lmhead_tp_enable",
            "get_forward_context",
            "_EXTRA_CTX",
        ):
            assert hasattr(vllm_ascend.spec_decode.eagle_proposer, attr), (
                f"vllm_ascend.spec_decode.eagle_proposer.{attr} not found"
            )
        RunnerCls = vllm_ascend.spec_decode.eagle_proposer.AscendSpecDecodeBaseProposer
        for attr in (
            "_run_merged_draft",
            "maybe_pad_and_reduce",
            "maybe_all_gather_and_unpad",
            "model_returns_tuple",
        ):
            assert hasattr(RunnerCls, attr), f"AscendSpecDecodeBaseProposer.{attr} not found"

        sig = inspect.signature(RunnerCls._run_merged_draft)
        sig_name = self.get_param_names(sig)
        assert sig_name == [
            "self",
            "num_input_tokens",
            "batch_size",
            "token_indices_to_sample",
            "target_positions",
            "inputs_embeds",
            "multi_steps_attn_metadata",
            "num_tokens",
            "is_prefill",
        ]
        sig = inspect.signature(RunnerCls.maybe_pad_and_reduce)
        sig_name = self.get_param_names(sig)
        assert sig_name == ["self", "hidden_states", "positions"]
        sig = inspect.signature(RunnerCls.maybe_all_gather_and_unpad)
        sig_name = self.get_param_names(sig)
        assert sig_name == ["self", "last_hidden_states", "positions", "hidden_states"]
        sig = inspect.signature(RunnerCls.model_returns_tuple)
        sig_name = self.get_param_names(sig)
        assert sig_name == ["self"]

        import vllm_ascend.spec_decode.dflash_proposer

        assert hasattr(vllm_ascend.spec_decode.dflash_proposer, "AscendDflashProposer")
        RunnerCls = vllm_ascend.spec_decode.dflash_proposer.AscendDflashProposer
        assert hasattr(RunnerCls, "build_model_inputs_first_pass")
        sig = inspect.signature(RunnerCls.build_model_inputs_first_pass)
        sig_name = self.get_param_names(sig)
        assert sig_name == ["self", "num_input_tokens"]

        import vllm_ascend.worker.model_runner_v1

        assert hasattr(vllm_ascend.worker.model_runner_v1, "NPUModelRunner")
        RunnerCls = vllm_ascend.worker.model_runner_v1.NPUModelRunner
        src = inspect.getsource(RunnerCls.__init__)
        fields = {
            "pcp_size",
            "dcp_size",
            "pcp_rank",
            "dcp_rank",
            "max_num_tokens",
            "max_num_reqs",
            "uniform_decode_query_len",
            "enable_enpu",
            "use_eagle",
            "pin_memory",
        }
        for f in fields:
            assert f"self.{f}" in src, f"missing self.{f} in __init__"
        assert hasattr(RunnerCls, "_use_aclgraph")
        sig = inspect.signature(RunnerCls._use_aclgraph)
        sig_name = self.get_param_names(sig)
        assert sig_name == ["self"]
        assert hasattr(RunnerCls, "_make_buffer")
        sig = inspect.signature(RunnerCls._make_buffer)
        sig_name = self.get_param_names(sig)
        assert sig_name == ["self", "size", "dtype", "numpy"]

    def get_param_names(self, sig):
        return [p.name for p in sig.parameters.values()]

    def test_run_merged_draft_eagle3_decode_prepares_each_forward_input(self):
        self.proposer.model = MockDraftModel(returns_tuple=True)
        self.proposer.supports_mm_inputs = True
        initial_input_ids = torch.tensor(
            [279, 1196, 374, 8014, 151667, 198, 32313, 11, 151667, 198, 32313, 11],
            dtype=torch.int32,
        )
        initial_positions = torch.tensor(
            [17, 18, 19, 20, 13, 14, 15, 16, 13, 14, 15, 16],
            dtype=torch.int32,
        )
        initial_hidden_states = torch.arange(48, dtype=torch.float32).view(12, 4)
        self.proposer.input_ids[:12] = initial_input_ids
        self.proposer.positions[:12] = initial_positions
        self.proposer.hidden_states[:12] = initial_hidden_states

        token_indices_to_sample = torch.tensor([1, 7, 11], dtype=torch.int64)
        forward_context = MagicMock()
        forward_context.moe_layer_index = 5
        forward_context.attn_metadata = None
        multi_steps_attn_metadata = [MagicMock(), MagicMock(), MagicMock()]

        with (
            patch.object(eagle_proposer, "lmhead_tp_enable", return_value=False),
            patch.object(eagle_proposer, "get_forward_context", return_value=forward_context),
        ):
            draft_token_ids = self.proposer._run_merged_draft(
                num_input_tokens=12,
                batch_size=3,
                token_indices_to_sample=token_indices_to_sample,
                target_positions=self.proposer.positions[:12],
                inputs_embeds=None,
                multi_steps_attn_metadata=multi_steps_attn_metadata,
                num_tokens=12,
                is_prefill=False,
            )

        model = self.proposer.model
        self.assertEqual(draft_token_ids.tolist(), [[1196, 1197, 1199], [11, 12, 14], [11, 12, 14]])
        self.assertEqual(len(model.calls), 3)

        first_call = model.calls[0]
        self.assertTrue(torch.equal(first_call["input_ids"], initial_input_ids))
        self.assertTrue(torch.equal(first_call["positions"], initial_positions))
        self.assertTrue(torch.equal(first_call["hidden_states"], initial_hidden_states))
        self.assertIsNone(first_call["inputs_embeds"])

        second_call = model.calls[1]
        self.assertTrue(torch.equal(second_call["input_ids"], torch.tensor([1196, 11, 11], dtype=torch.int32)))
        self.assertTrue(torch.equal(second_call["positions"], torch.tensor([19, 17, 17], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(
                second_call["hidden_states"],
                model.returned_hidden_states[0][1][token_indices_to_sample],
            )
        )
        self.assertTrue(
            torch.equal(
                second_call["inputs_embeds"],
                model.embed_input_ids(torch.tensor([1196, 11, 11], dtype=torch.int32)),
            )
        )

        third_call = model.calls[2]
        self.assertTrue(torch.equal(third_call["input_ids"], torch.tensor([1197, 12, 12], dtype=torch.int32)))
        self.assertTrue(torch.equal(third_call["positions"], torch.tensor([20, 18, 18], dtype=torch.int32)))
        self.assertTrue(torch.equal(model.logit_inputs[0], model.returned_hidden_states[0][0][token_indices_to_sample]))
        self.assertEqual(forward_context.moe_layer_index, 0)
        self.assertIs(forward_context.attn_metadata, multi_steps_attn_metadata[2])
        self.assertEqual(eagle_proposer._EXTRA_CTX.num_tokens, 3)
        self.assertEqual(eagle_proposer._EXTRA_CTX.num_accept_tokens, 3)

    def test_run_merged_draft_dflash_uses_first_pass_inputs_and_returns_early(self):
        self.proposer.method = "dflash"
        self.proposer.num_speculative_tokens = 1
        self.proposer.pass_hidden_states_to_model = False
        self.proposer.model = MockDraftModel(returns_tuple=False)
        self.proposer.build_model_inputs_first_pass = MagicMock(
            return_value={
                "input_ids": torch.tensor([151667, 32313], dtype=torch.int32),
                "positions": torch.tensor([20, 16], dtype=torch.int64),
                "inputs_embeds": torch.ones(2, 4, dtype=torch.float32),
            }
        )

        with patch.object(eagle_proposer, "lmhead_tp_enable", return_value=False):
            draft_token_ids = self.proposer._run_merged_draft(
                num_input_tokens=12,
                batch_size=2,
                token_indices_to_sample=torch.tensor([0, 1], dtype=torch.int64),
                target_positions=torch.tensor([20, 16], dtype=torch.int64),
                inputs_embeds=None,
                multi_steps_attn_metadata=None,
                num_tokens=12,
                is_prefill=False,
            )

        self.proposer.build_model_inputs_first_pass.assert_called_once_with(12)
        self.proposer.maybe_all_gather_and_unpad.assert_not_called()
        self.assertNotIn("hidden_states", self.proposer.model.calls[0])
        self.assertTrue(
            torch.equal(
                self.proposer.model.calls[0]["input_ids"],
                torch.tensor([151667, 32313], dtype=torch.int32),
            )
        )
        self.assertEqual(draft_token_ids.tolist(), [[151667], [32313]])

    def test_run_merged_draft_mtp_mrope_graph_and_lmhead_tp_preparation(self):
        self.proposer.method = "mtp"
        self.proposer.uses_mrope = True
        self.proposer.use_cuda_graph = True
        self.proposer.vllm_config.model_config.max_model_len = 4
        self.proposer.vllm_config.scheduler_config.max_num_seqs = 3
        self.proposer.runner.uniform_decode_query_len = 2
        self.proposer.mrope_positions = torch.zeros((3, self.proposer.max_num_tokens + 1), dtype=torch.int64)
        self.proposer.model = MockDraftModel(returns_tuple=False)
        self.proposer.input_ids[:6] = torch.tensor([201, 33001, 14, 832, 128798, 271], dtype=torch.int32)
        initial_mrope_positions = torch.tensor(
            [
                [0, 1, 3, 0, 1, 3],
                [10, 11, 13, 10, 11, 13],
                [20, 21, 23, 20, 21, 23],
            ],
            dtype=torch.int64,
        )
        initial_hidden_states = torch.arange(24, dtype=torch.float32).view(6, 4)
        self.proposer.mrope_positions[:, :6] = initial_mrope_positions
        self.proposer.hidden_states[:6] = initial_hidden_states
        token_indices_to_sample = torch.tensor([2, 5], dtype=torch.int64)
        forward_context = MagicMock()
        forward_context.moe_layer_index = 9
        forward_context.attn_metadata = None
        multi_steps_attn_metadata = [MagicMock(), MagicMock(), MagicMock()]

        with (
            patch.object(eagle_proposer, "lmhead_tp_enable", return_value=True),
            patch.object(eagle_proposer, "get_forward_context", return_value=forward_context),
        ):
            draft_token_ids = self.proposer._run_merged_draft(
                num_input_tokens=6,
                batch_size=2,
                token_indices_to_sample=token_indices_to_sample,
                target_positions=self.proposer.mrope_positions[:, :6],
                inputs_embeds=None,
                multi_steps_attn_metadata=multi_steps_attn_metadata,
                num_tokens=6,
                is_prefill=False,
            )

        model = self.proposer.model
        self.assertEqual(draft_token_ids.tolist(), [[14, 15, 17], [271, 272, 274]])
        self.assertTrue(all(logit_input.shape[0] == 6 for logit_input in model.logit_inputs))

        first_call = model.calls[0]
        self.assertTrue(torch.equal(first_call["positions"], initial_mrope_positions))
        self.assertTrue(torch.equal(first_call["hidden_states"], initial_hidden_states))

        second_call = model.calls[1]
        self.assertTrue(
            torch.equal(
                second_call["input_ids"],
                torch.tensor([14, 271, 14, 832, 128798, 271], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                second_call["positions"],
                torch.tensor(
                    [
                        [0, 0, 3, 0, 1, 3],
                        [0, 0, 13, 10, 11, 13],
                        [0, 0, 23, 20, 21, 23],
                    ],
                    dtype=torch.int64,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                second_call["hidden_states"][:2],
                model.returned_hidden_states[0][0][token_indices_to_sample],
            )
        )
        self.assertIs(forward_context.attn_metadata, multi_steps_attn_metadata[2])

    def test_run_merged_draft_early_return_conditions(self):
        test_cases = [
            (1, False, torch.tensor([1, 3], dtype=torch.int64), (2, 1)),
            (2, True, torch.tensor([0, 1, 2, 3], dtype=torch.int64), (2, 2)),
        ]
        for num_speculative_tokens, parallel_drafting, token_indices_to_sample, expected_shape in test_cases:
            with self.subTest(num_speculative_tokens=num_speculative_tokens, parallel_drafting=parallel_drafting):
                self.proposer.method = "eagle3"
                self.proposer.num_speculative_tokens = num_speculative_tokens
                self.proposer.parallel_drafting = parallel_drafting
                self.proposer.pass_hidden_states_to_model = False
                self.proposer.model = MockDraftModel(returns_tuple=True)
                self.proposer.input_ids[:4] = torch.tensor([279, 1196, 374, 8014], dtype=torch.int32)
                self.proposer.positions[:4] = torch.tensor([17, 18, 19, 20], dtype=torch.int64)

                with patch.object(eagle_proposer, "lmhead_tp_enable", return_value=False):
                    draft_token_ids = self.proposer._run_merged_draft(
                        num_input_tokens=4,
                        batch_size=2,
                        token_indices_to_sample=token_indices_to_sample,
                        target_positions=self.proposer.positions[:4],
                        inputs_embeds=None,
                        multi_steps_attn_metadata=None,
                        num_tokens=4,
                        is_prefill=False,
                    )

                self.assertEqual(tuple(draft_token_ids.shape), expected_shape)
                self.assertEqual(len(self.proposer.model.calls), 1)
# fmt: on
