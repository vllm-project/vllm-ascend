import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from vllm_ascend.models.qwen3_moe import AscendQwen3MoeSparseMoeBlock


class TestAscendQwen3MoeSparseMoeBlock(unittest.TestCase):

    def setUp(self):
        # Create a mock config
        self.mock_config = MagicMock()
        self.mock_config.hidden_size = 512
        self.mock_config.num_experts = 8
        self.mock_config.num_experts_per_tok = 2
        self.mock_config.moe_intermediate_size = 1024
        self.mock_config.norm_topk_prob = True

        # Mock all the distributed and environment dependencies
        self.patchers = [
            patch('vllm.distributed.get_tensor_model_parallel_world_size',
                  return_value=1),
            patch('vllm_ascend.ascend_config.get_ascend_config',
                  return_value=MagicMock(torchair_graph_config=MagicMock(
                      enabled=True, enable_multistream_moe=True))),
            patch('vllm.distributed.parallel_state.get_dp_group',
                  return_value=MagicMock(world_size=1)),
            patch('vllm.distributed.get_tp_group',
                  return_value=MagicMock(device_group=None, rank_in_group=0)),
            patch('vllm_ascend.distributed.parallel_state.get_ep_group',
                  return_value=None),
            patch('vllm.forward_context.get_forward_context',
                  return_value=MagicMock(attn_metadata=None)),
            patch('torch.get_default_dtype', return_value=torch.float32)
        ]

        for patcher in self.patchers:
            patcher.start()

        # Mock the ReplicatedLinear and AscendFusedMoE classes
        self.mock_replicated_linear = MagicMock(spec=nn.Linear)
        self.mock_fused_moe = MagicMock()

        with patch('vllm.model_executor.layers.linear.ReplicatedLinear', return_value=self.mock_replicated_linear), \
                patch('vllm_ascend.ops.fused_moe.AscendFusedMoE', return_value=self.mock_fused_moe):

            self.block = AscendQwen3MoeSparseMoeBlock(config=self.mock_config,
                                                      quant_config=None,
                                                      prefix="moe")

    def tearDown(self):
        for patcher in self.patchers:
            patcher.stop()

    def test_initialization(self):
        # Test initialization values
        self.assertEqual(self.block.top_k,
                         self.mock_config.num_experts_per_tok)
        self.assertEqual(self.block.params_dtype, torch.float32)
        self.assertTrue(self.block.torchair_graph_enabled)
        self.assertTrue(self.block.enable_multistream_moe)

        # Check if submodules were created
        self.mock_replicated_linear.assert_called_once()
        self.mock_fused_moe.assert_called_once()

    def test_forward_with_attn_metadata(self):
        # Setup mock return values
        mock_router_logits = torch.randn(10, self.mock_config.num_experts)
        self.mock_replicated_linear.return_value = (mock_router_logits, None)

        mock_hidden_states = torch.randn(10, self.mock_config.hidden_size)
        mock_output = torch.randn(10, self.mock_config.hidden_size)
        self.mock_fused_moe.return_value = mock_output

        # Mock attention metadata
        mock_attn_metadata = MagicMock()
        mock_attn_metadata.with_prefill_across_dp = False

        # Test forward pass
        output = self.block(mock_hidden_states, mock_attn_metadata)

        # Verify calls
        self.mock_replicated_linear.assert_called_once_with(mock_hidden_states)
        self.mock_fused_moe.assert_called_once_with(
            hidden_states=mock_hidden_states,
            router_logits=mock_router_logits,
            is_prefill=False,
            top_k=self.mock_config.num_experts_per_tok,
            enable_force_load_balance=False,
            shared_experts=None)
        self.assertTrue(torch.equal(output, mock_output))

    def test_forward_without_attn_metadata(self):
        # Setup mock return values
        mock_router_logits = torch.randn(10, self.mock_config.num_experts)
        self.mock_replicated_linear.return_value = (mock_router_logits, None)

        mock_hidden_states = torch.randn(10, self.mock_config.hidden_size)
        mock_output = torch.randn(10, self.mock_config.hidden_size)
        self.mock_fused_moe.return_value = mock_output

        # Test forward pass without attention metadata
        output = self.block(mock_hidden_states)

        # Verify calls - should use default values when no metadata
        self.mock_replicated_linear.assert_called_once_with(mock_hidden_states)
        self.mock_fused_moe.assert_called_once_with(
            hidden_states=mock_hidden_states,
            router_logits=mock_router_logits,
            is_prefill=True,
            top_k=self.mock_config.num_experts_per_tok,
            enable_force_load_balance=True,
            shared_experts=None)
        self.assertTrue(torch.equal(output, mock_output))

    def test_tp_size_greater_than_experts(self):
        # Test the validation for TP size vs number of experts
        with patch('vllm.distributed.get_tensor_model_parallel_world_size',
                   return_value=10):
            with self.assertRaises(ValueError) as context:
                self.block = AscendQwen3MoeSparseMoeBlock(
                    config=self.mock_config, quant_config=None, prefix="moe")
            self.assertIn("Tensor parallel size 10 is greater than",
                          str(context.exception))
