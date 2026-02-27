from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.config import CacheConfig, CompilationMode, CUDAGraphMode, VllmConfig, set_current_vllm_config

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.spec_decode.eagle_proposer import EagleProposer
from vllm_ascend.spec_decode.interface import SpecDcodeType


class TestEagleProposerInitialization(TestBase):

    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.cache_config = MagicMock(spec=CacheConfig)
        self.vllm_config.scheduler_config = MagicMock()
        self.vllm_config.model_config = MagicMock()
        self.vllm_config.model_config.hf_text_config = MagicMock(spec=[])  # Empty spec to prevent hasattr from returning True
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
        self.vllm_config.speculative_config.speculative_token_tree = str([
            (i + 1) * (0, ) for i in range(2)
        ])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None

        self.mock_cpugpubuffer = patch(
            "vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs",
            return_value=False
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
            proposer = EagleProposer(vllm_config=self.vllm_config,
                                     device=self.device,
                                     runner=self.runner)

            self.assertEqual(proposer.hidden_size, 4096)
            self.assertTrue(proposer.use_cuda_graph)

            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.input_ids.shape, (expected_max_num_tokens, ))
            self.assertEqual(proposer.positions.shape, (expected_max_num_tokens, ))
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 4096))
            self.assertEqual(proposer.arange.shape, (expected_max_num_tokens, ))

    def test_initialization_eagle3_enforce_eager(self):
        self.vllm_config.speculative_config.method = "eagle3"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.NONE
        self.vllm_config.compilation_config.pass_config = MagicMock()
        self.vllm_config.compilation_config.pass_config.enable_sp = False
        self.vllm_config.model_config.enforce_eager = True
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = EagleProposer(vllm_config=self.vllm_config,
                                     device=self.device,
                                     runner=self.runner)

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
            proposer = EagleProposer(vllm_config=self.vllm_config,
                                     device=self.device,
                                     runner=self.runner)

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
            proposer = EagleProposer(vllm_config=self.vllm_config,
                                     device=self.device,
                                     runner=self.runner)

            self.assertEqual(proposer.hidden_size, 2048)
            self.assertFalse(proposer.use_cuda_graph)
            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 2048))


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
        self.vllm_config.speculative_config.speculative_token_tree = str([
            (i + 1) * (0, ) for i in range(2)
        ])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch(
            "vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs",
            return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = EagleProposer(vllm_config=self.vllm_config,
                                      device=self.device,
                                      runner=self.runner)

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    @patch(
        "vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    def test_load_model_pp1(self, mock_pp_group, mock_get_model,
                            mock_get_layers):
        mock_pp_group.return_value.world_size = 1
        mock_target_layer1 = MagicMock()
        mock_target_layer2 = MagicMock()
        mock_draft_layer1 = MagicMock()
        mock_draft_layer3 = MagicMock()
        mock_get_layers.side_effect = [{
            "layer1": mock_target_layer1,
            "layer2": mock_target_layer2
        }, {}, {}, {
            "layer1": mock_draft_layer1,
            "layer3": mock_draft_layer3
        }]

        weight = torch.zeros(0)

        mock_model = MagicMock()
        mock_model.supports_multimodal = False
        mock_model.lm_head = MagicMock()
        mock_model.multimodal_cpu_fields = None
        mock_model.merge_by_field_config = None
        mock_model.model.embed_tokens = MagicMock()
        mock_model.model.embed_tokens.weight = weight

        self.proposer.name = SpecDcodeType.EAGLE
        mock_get_model.return_value = MagicMock()
        mock_get_model.return_value.model.embed_tokens.weight = weight

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)
            mock_get_model.assert_called_once()
            self.assertEqual(self.proposer.attn_layer_names, ["layer3"])
            self.assertIs(self.proposer.model.model.embed_tokens,
                          mock_model.model.embed_tokens)

    @patch(
        "vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    def test_load_model_pp_gt1(self, mock_pp_group, mock_get_model,
                               mock_get_layers):
        mock_pp_group.return_value.world_size = 2
        mock_target_layer1 = MagicMock()
        mock_draft_layer2 = MagicMock()

        mock_get_layers.side_effect = [{
            "layer1": mock_target_layer1
        }, {}, {}, {
            "layer2": mock_draft_layer2
        }]

        mock_model = MagicMock()
        original_embed = MagicMock()
        mock_model.multimodal_cpu_fields = None
        mock_model.merge_by_field_config = None
        mock_get_model.return_value = MagicMock(model=MagicMock(
            embed_tokens=original_embed))

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)

            self.assertIsNot(self.proposer.model.model.embed_tokens,
                             mock_model.model.embed_tokens)
            self.assertEqual(self.proposer.attn_layer_names, ["layer2"])

    @patch(
        "vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    @patch("vllm_ascend.spec_decode.eagle_proposer.supports_multimodal")
    def test_load_model_multimodal(self, mock_supports_multi, mock_pp_group,
                                   mock_get_model, mock_get_layers):
        mock_model = MagicMock()
        mock_model.get_language_model.return_value.lm_head = MagicMock()
        mock_supports_multi.return_value = True
        original_embed = MagicMock()
        mock_get_model.return_value = MagicMock(model=MagicMock(
            embed_tokens=original_embed))

        mock_target_layer1 = MagicMock()
        mock_draft_layer2 = MagicMock()

        mock_get_layers.side_effect = [{
            "layer1": mock_target_layer1
        }, {}, {}, {
            "layer2": mock_draft_layer2
        }]
        mock_pp_group.return_value.world_size = 2

        self.proposer.model = MagicMock()
        self.proposer.name = SpecDcodeType.EAGLE

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)
            self.assertEqual(mock_model.get_language_model.call_count, 2)
            self.assertIs(self.proposer.model.lm_head,
                          mock_model.get_language_model.return_value.lm_head)


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
        self.vllm_config.model_config.hf_text_config = MagicMock(spec=[])  # Empty spec to prevent hasattr from returning True
        self.vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.speculative_token_tree = str([
            (i + 1) * (0, ) for i in range(4)
        ])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch(
            "vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs",
            return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Mock parallel state functions
        self.mock_tp_world_size = patch(
            "vllm_ascend.ascend_forward_context.get_tensor_model_parallel_world_size",
            return_value=1
        )
        self.mock_tp_world_size.start()

        mock_dp_group = MagicMock()
        mock_dp_group.world_size = 1
        self.mock_dp_group = patch(
            "vllm_ascend.ascend_forward_context.get_dp_group",
            return_value=mock_dp_group
        )
        self.mock_dp_group.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = EagleProposer(vllm_config=self.vllm_config,
                                      device=self.device,
                                      runner=self.runner)
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
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context",
           **{"return_value.flash_comm_v1_enabled": False})
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_basic(self, mock_context, mock_get_context):
        num_tokens = 32
        with_prefill = False

        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=num_tokens,
                                    with_prefill=with_prefill)

            self.assertTrue(self.proposer._runnable.call_count == 1)

    # cpu does not support parallel-group, let alone `sp`
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context",
           **{"return_value.flash_comm_v1_enabled": False})
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_with_prefill(self, mock_context, mock_get_context):
        mock_context.return_value.__enter__.return_value = None
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64, with_prefill=True, num_reqs=4)
            self.assertTrue(self.proposer._runnable.call_count == 1)

    @patch("vllm_ascend.spec_decode.eagle_proposer.update_full_graph_params")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_in_graph_capture(self, mock_context, mock_get_context,
                                        mock_update_full_graph_params):
        last_use_cuda_graph = self.proposer.use_cuda_graph
        mock_return_context = MagicMock()
        mock_return_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        mock_return_context.capturing = True
        # cpu does not support parallel-group, let alone `sp`
        mock_return_context.flash_comm_v1_enabled = False
        mock_get_context.return_value = mock_return_context
        self.proposer.use_cuda_graph = True
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64,
                                    in_graph_capturing=True,
                                    aclgraph_runtime_mode=CUDAGraphMode.FULL)
            self.assertTrue(self.proposer._runnable.call_count == 1)
            mock_update_full_graph_params.assert_not_called()
            self.proposer.use_cuda_graph = last_use_cuda_graph

    @patch("vllm_ascend.spec_decode.eagle_proposer.update_full_graph_params")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_in_graph_run(self, mock_context, mock_get_context,
                                    mock_update_full_graph_params):
        last_use_cuda_graph = self.proposer.use_cuda_graph
        mock_return_context = MagicMock()
        mock_return_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        mock_return_context.capturing = False
        # cpu does not support parallel-group, let alone `sp`
        mock_return_context.flash_comm_v1_enabled = False
        mock_get_context.return_value = mock_return_context
        self.proposer.use_cuda_graph = True
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64,
                                    in_graph_capturing=False,
                                    aclgraph_runtime_mode=CUDAGraphMode.FULL)
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
        self.vllm_config.speculative_config.speculative_token_tree = str([
            (i + 1) * (0, ) for i in range(2)
        ])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch(
            "vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs",
            return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = EagleProposer(vllm_config=self.vllm_config,
                                      device=self.device,
                                      runner=self.runner)

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

        with set_current_vllm_config(self.vllm_config):
            with patch.object(self.proposer,
                              'prepare_inputs',
                              return_value=(mock_return_attn,
                                            torch.tensor([1, 2, 4]))):
                return_attn, indices = self.proposer.prepare_inputs(
                    mock_attn, num_rejected)
                self.assertEqual(indices.tolist(), [1, 2, 4])
