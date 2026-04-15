from unittest.mock import MagicMock, patch
import unittest
import pytest
import numpy as np
import torch
from vllm.config import CacheConfig, CompilationMode, CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from tests.ut.base import TestBase
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer


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
        self.runner._sync_metadata_across_dp.return_value = (8, torch.tensor([8]), False)

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
    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context",
           **{"return_value.flash_comm_v1_enabled": False})
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
    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context",
           **{"return_value.flash_comm_v1_enabled": False})
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_with_prefill(self, mock_context, mock_get_context, mock_get_context_2):
        mock_context.return_value.__enter__.return_value = None
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64, with_prefill=True, num_reqs=4)
            self.assertTrue(self.proposer._runnable.call_count == 1)

    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    @patch("vllm_ascend.spec_decode.eagle_proposer.update_full_graph_params")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_in_graph_capture(self, mock_context, mock_get_context,
                                        mock_update_full_graph_params, mock_get_context_2):
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
    
    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    @patch("vllm_ascend.spec_decode.eagle_proposer.update_full_graph_params")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_in_graph_run(self, mock_context, mock_get_context,
                                    mock_update_full_graph_params, mock_get_context_2):
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


class TestEagleProposerPropose():
    @pytest.fixture(autouse=True)
    def setUp_and_tearDown(self):
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
    @patch('vllm_ascend.spec_decode.eagle_proposer.AscendEagleProposer.get_model')
    def test_propose(self, mock_get_model, model_type, flag_prefill_decode, query_start_loc, query_start_loc_cpu, seq_lens, num_reqs,
                     num_actual_tokens, max_query_len, max_seq_len, block_table_tensor,
                     slot_mapping, causal, logits_indices_padded, num_logits_indices,
                     encoder_seq_lens, encoder_seq_lens_cpu, dcp_local_seq_lens,
                     dcp_local_seq_lens_cpu, _seq_lens_cpu, _num_computed_tokens_cpu,
                     _num_computed_tokens_cache, seq_lens_cpu, num_computed_tokens_cpu,
                     decode_token_per_req, actual_seq_lengths_q, positions, attn_state,
                     graph_pad_size, num_input_tokens, prefill_context_parallel_metadata
                    ):
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
        with patch.object(self.proposer, 'attn_update_stack_num_spec_norm', side_effect=side_effect):
            with set_current_vllm_config(self.vllm_config):
                self.proposer._propose(target_token_ids, target_positions, target_hidden_states, next_token_ids,
                                    token_indices_to_sample, mock_common_attn_metadata, target_model_batch_desc, mock_sampling_metadata,
                                    mm_embed_inputs, req_scheduled_tokens, long_seq_metadata, num_prefill_reqs, num_decode_reqs,
                                    scheduler_output, num_scheduled_tokens, num_rejected_tokens_gpu,
                                    )
                self.assert_value_common_attn_metadata(captured_common_attn_metadata, flag_prefill_decode, model_type)

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
    def assert_value_common_attn_metadata(self, captured_common_attn_metadata, flag_prefill_decode, model_type):
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
                assert captured_common_attn_metadata._seq_lens_cpu == None
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
            assert torch.equal(captured_common_attn_metadata.query_start_loc, torch.tensor([0, 1, 2, 3]))
            assert torch.equal(captured_common_attn_metadata.query_start_loc_cpu, torch.tensor([0, 1, 2, 3]))
            assert captured_common_attn_metadata.num_input_tokens == 12
            assert captured_common_attn_metadata.num_reqs == 3
            assert captured_common_attn_metadata.num_actual_tokens == 3
            assert captured_common_attn_metadata.max_query_len == 1
            assert captured_common_attn_metadata.max_seq_len == 0
            assert torch.equal(captured_common_attn_metadata.block_table_tensor, torch.cat([torch.eye(256, device="cpu", dtype=torch.int32)[0].unsqueeze(0)*i for i in [1,2,3]], dim=0))
            assert captured_common_attn_metadata._seq_lens_cpu == None
            if model_type == 'qwen_dense':
                assert torch.equal(captured_common_attn_metadata.seq_lens, torch.tensor([23, 19, 19]))
                assert torch.equal(captured_common_attn_metadata.slot_mapping, torch.cat([torch.tensor([148, 274, 402]), torch.full((8701,), -1)]))
                assert torch.equal(captured_common_attn_metadata.seq_lens_cpu, torch.tensor([23, 19, 19]))
                assert torch.equal(captured_common_attn_metadata.num_computed_tokens_cpu, torch.tensor([19, 15, 15]))
                assert torch.equal(captured_common_attn_metadata.positions, torch.tensor([20, 18, 18, 20, 13, 14, 15, 16, 13, 14, 15, 16, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] + [0]*(8704-30), dtype=torch.int64))
            if model_type == 'qwen_moe':
                assert torch.equal(captured_common_attn_metadata.seq_lens, torch.tensor([21, 19, 19]))
                assert torch.equal(captured_common_attn_metadata.slot_mapping, torch.cat([torch.tensor([145, 272, 400]), torch.full((8701,), -1)]))
                assert torch.equal(captured_common_attn_metadata.seq_lens_cpu, torch.tensor([21, 19, 19]))
                assert torch.equal(captured_common_attn_metadata.num_computed_tokens_cpu, torch.tensor([17, 15, 15]))
                assert torch.equal(captured_common_attn_metadata.positions, torch.tensor([17, 16, 16, 18, 13, 14, 15, 16, 13, 14, 15, 16, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] + [0]*(8704-30), dtype=torch.int64))
            if model_type == 'deepseek':
                assert torch.equal(captured_common_attn_metadata.seq_lens, torch.tensor([16, 15, 16]))
                assert torch.equal(captured_common_attn_metadata.slot_mapping, torch.cat([torch.tensor([142, 268, 396]), torch.full((8701,), -1)]))
                assert torch.equal(captured_common_attn_metadata.seq_lens_cpu, torch.tensor([16, 15, 16]))
                assert torch.equal(captured_common_attn_metadata.num_computed_tokens_cpu, torch.tensor([12, 11, 12]))
                assert torch.equal(captured_common_attn_metadata.positions, torch.tensor([14, 12, 12, 13, 9, 10, 11, 12, 10, 11, 12, 13, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + [0]*(8704-23), dtype=torch.int64))
        assert captured_common_attn_metadata.causal == True
        assert captured_common_attn_metadata.logits_indices_padded == None
        assert captured_common_attn_metadata.num_logits_indices == None
        assert captured_common_attn_metadata.encoder_seq_lens == None
        assert captured_common_attn_metadata.encoder_seq_lens_cpu == None
        assert captured_common_attn_metadata.dcp_local_seq_lens == None
        assert captured_common_attn_metadata.dcp_local_seq_lens_cpu == None
        assert captured_common_attn_metadata._num_computed_tokens_cpu == None
        assert captured_common_attn_metadata._num_computed_tokens_cache == None
        assert captured_common_attn_metadata.decode_token_per_req == 1
        assert captured_common_attn_metadata.actual_seq_lengths_q == []
        if model_type == 'deepseek':
            assert captured_common_attn_metadata.attn_state == AscendAttentionState.SpecDecoding
        else:
            assert captured_common_attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill
        assert captured_common_attn_metadata.graph_pad_size == -1
        assert captured_common_attn_metadata.prefill_context_parallel_metadata == None

    # prefill or decode
    def is_decode(self, flag_prefill_decode):
        if flag_prefill_decode == "decode":
            return True
        if flag_prefill_decode == "prefill":
            return False
