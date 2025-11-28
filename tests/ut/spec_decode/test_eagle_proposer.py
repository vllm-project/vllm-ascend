from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.config import CacheConfig, CompilationMode, VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.spec_decode.eagle_proposer import EagleProposer
from vllm_ascend.spec_decode.interface import SpecDcodeType


class TestEagleProposerInitialization(TestBase):

    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.cache_config = MagicMock(spec=CacheConfig)
        self.vllm_config.scheduler_config = MagicMock()
        self.vllm_config.model_config = MagicMock()
        self.device = torch.device("cpu")
        self.runner = MagicMock()

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048

    def test_initialization_eagle(self):
        self.vllm_config.speculative_config.method = "eagle"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 4096
        self.vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        self.vllm_config.model_config.enforce_eager = False

        proposer = EagleProposer(vllm_config=self.vllm_config,
                                 device=self.device,
                                 runner=self.runner)

        self.assertEqual(proposer.name, SpecDcodeType.EAGLE)
        self.assertEqual(proposer.block_size, 16)
        self.assertEqual(proposer.hidden_size, 4096)
        self.assertTrue(proposer.use_cuda_graph)

        self.assertEqual(proposer.input_ids.shape, (1024, ))
        self.assertEqual(proposer.positions.shape, (1024, ))
        self.assertEqual(proposer.hidden_states.shape, (1024, 4096))
        self.assertEqual(proposer.arange.shape, (33, ))

    def test_initialization_eagle3(self):
        self.vllm_config.speculative_config.method = "eagle3"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.NONE
        self.vllm_config.model_config.enforce_eager = True

        proposer = EagleProposer(vllm_config=self.vllm_config,
                                 device=self.device,
                                 runner=self.runner)

        self.assertEqual(proposer.name, SpecDcodeType.EAGLE3)
        self.assertEqual(proposer.hidden_size, 2048)
        self.assertFalse(proposer.use_cuda_graph)
        self.assertEqual(proposer.hidden_states.shape, (1024, 2048))


class TestEagleProposerLoadModel(TestBase):

    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.speculative_config.method = "eagle"
        self.device = torch.device("cpu")
        self.runner = MagicMock()

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048

        self.proposer = EagleProposer(vllm_config=self.vllm_config,
                                      device=self.device,
                                      runner=self.runner)

    @patch(
        "vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    def test_load_model_pp1(self, mock_pp_group, mock_get_model,
                            mock_get_layers):
        mock_pp_group.return_value.world_size = 1
        mock_target_layers = {"layer1": MagicMock(), "layer2": MagicMock()}
        mock_draft_layers = {"layer1": MagicMock(), "layer3": MagicMock()}
        mock_get_layers.side_effect = [mock_target_layers, mock_draft_layers]

        mock_model = MagicMock()
        mock_model.model.embed_tokens = MagicMock()
        mock_model.lm_head = MagicMock()
        mock_get_model.return_value = MagicMock()
        self.proposer.name = SpecDcodeType.EAGLE

        self.proposer.load_model(mock_model)
        mock_get_model.assert_called_once()
        self.assertEqual(self.proposer.attn_layer_name, "layer3")
        self.assertIs(self.proposer.model.model.embed_tokens,
                      mock_model.model.embed_tokens)

    @patch(
        "vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    def test_load_model_pp_gt1(self, mock_pp_group, mock_get_model,
                               mock_get_layers):
        mock_pp_group.return_value.world_size = 2
        mock_target_layers = {"layer1": MagicMock()}
        mock_draft_layers = {"layer2": MagicMock()}
        mock_get_layers.side_effect = [mock_target_layers, mock_draft_layers]

        mock_model = MagicMock()
        original_embed = MagicMock()
        mock_get_model.return_value = MagicMock(model=MagicMock(
            embed_tokens=original_embed))

        self.proposer.load_model(mock_model)

        self.assertIsNot(self.proposer.model.model.embed_tokens,
                         mock_model.model.embed_tokens)
        self.assertEqual(self.proposer.attn_layer_name, "layer2")

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

        mock_target_layers = {"layer1": MagicMock()}
        mock_draft_layers = {"layer2": MagicMock()}
        mock_get_layers.side_effect = [mock_target_layers, mock_draft_layers]
        mock_pp_group.return_value.world_size = 2

        self.proposer.model = MagicMock()
        self.proposer.name = SpecDcodeType.EAGLE

        self.proposer.load_model(mock_model)
        mock_model.get_language_model.assert_called_once()
        self.assertIs(self.proposer.model.lm_head,
                      mock_model.get_language_model.return_value.lm_head)


class TestEagleProposerDummyRun(TestBase):

    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner._select_moe_comm_method.return_value = "alltoall"

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048

        self.proposer = EagleProposer(vllm_config=self.vllm_config,
                                      device=self.device,
                                      runner=self.runner)
        self.proposer.model = MagicMock()

    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_basic(self, mock_context):
        num_tokens = 32
        with_prefill = False

        self.proposer.dummy_run(num_tokens=num_tokens,
                                with_prefill=with_prefill)

        mock_context.assert_called_once()

    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_with_prefill(self, mock_context):
        mock_context.return_value.__enter__.return_value = None
        self.proposer.dummy_run(num_tokens=64, with_prefill=True, num_reqs=4)

        self.runner._select_moe_comm_method.assert_called_with(64)
        self.proposer.model.assert_called_once()


class TestEagleProposerGenerateTokenIds(TestBase):

    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.speculative_config.method = "eagle"
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.input_batch = MagicMock()
        self.runner.input_batch.req_ids = [0, 1, 2]
        self.runner.requests = {
            0: MagicMock(get_token_id=lambda x: 100),
            1: MagicMock(get_token_id=lambda x: 101),
            2: MagicMock(get_token_id=lambda x: 102),
        }

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048

        self.proposer = EagleProposer(vllm_config=self.vllm_config,
                                      device=self.device,
                                      runner=self.runner)
        self.proposer.attn_layer_name = "layer_0"
        self.proposer._propose = MagicMock(
            return_value=torch.tensor([[1, 2], [3, 4], [5, 6]]))

    def test_generate_token_ids_without_metadata(self):
        valid_sampled = [[20, 30, 40]]
        valid_sampled = [np.array(sublist) for sublist in valid_sampled]
        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = [2, 1, 3]
        positions = torch.tensor([0, 1, 2, 3, 4, 5])
        hidden_states = torch.randn(6, 4096)
        num_scheduled = 6

        mock_attn_metadata = MagicMock()
        mock_attn_metadata.slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5])
        mock_attn_metadata.query_start_loc = torch.tensor([0, 2, 3, 6])
        mock_attn_metadata.block_tables = MagicMock()
        self.proposer._get_eagle_atten_dict = MagicMock(
            return_value={"layer_0": mock_attn_metadata})

        result = self.proposer.generate_token_ids(
            valid_sampled_token_ids=valid_sampled,
            scheduler_output=scheduler_output,
            positions=positions,
            num_scheduled_tokens=num_scheduled,
            hidden_states=hidden_states,
        )

        self.proposer._propose.assert_called_once()
        self.assertEqual(result, [[1, 2], [3, 4], [5, 6]])

    def test_generate_token_ids_with_metadata(self):
        valid_sampled = [[5], [6, 7], [8, 9, 10]]
        valid_sampled = [np.array(sublist) for sublist in valid_sampled]
        spec_metadata = MagicMock()
        spec_metadata.num_draft_tokens = [2, 3, 4]

        mock_attn_metadata = MagicMock()
        mock_attn_metadata.slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5])
        mock_attn_metadata.query_start_loc = torch.tensor([0, 1, 3, 6])
        mock_attn_metadata.block_tables = MagicMock()
        self.proposer._get_eagle_atten_dict = MagicMock(
            return_value={"layer_0": mock_attn_metadata})
        self.proposer._prepare_inputs = MagicMock(
            return_value=(torch.tensor([0, 2, 5]), torch.tensor([1, 3, 5])))

        result = self.proposer.generate_token_ids(
            valid_sampled_token_ids=valid_sampled,
            spec_decode_metadata=spec_metadata,
            positions=torch.randn(6, 1),
            hidden_states=torch.randn(6, 4096),
        )

        self.proposer._prepare_inputs.assert_called_once()
        self.assertEqual(self.proposer._propose.call_count, 1)
        self.assertEqual(len(result), 3)


class TestEagleProposerHelperMethods(TestBase):

    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.scheduler_config = MagicMock(max_num_seqs=3)
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.input_batch = MagicMock()
        self.runner.input_batch.req_ids = [0, 1, 2]
        self.runner.arange_np = np.arange(10)
        self.runner.input_batch.num_reqs = 3

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048

        self.proposer = EagleProposer(vllm_config=self.vllm_config,
                                      device=self.device,
                                      runner=self.runner)

    def test_prepare_inputs(self):
        self.proposer.token_arange_np = np.arange(10)
        mock_attn = MagicMock()
        mock_attn.slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5])
        num_rejected = torch.tensor([1, 0, 1], device=self.device)

        with patch.object(self.proposer,
                          '_prepare_inputs',
                          return_value=(torch.tensor([0, 2, 5]),
                                        torch.tensor([1, 2, 4]))):
            cu_num_tokens, indices = self.proposer._prepare_inputs(
                mock_attn, num_rejected)
            self.assertEqual(cu_num_tokens.tolist(), [0, 2, 5])
            self.assertEqual(indices.tolist(), [1, 2, 4])
