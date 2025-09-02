import unittest
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ops.moe.fused_moe_prepare_and_finalize import (
    FusedMoEPrepareAndFinalizeWithAll2All,
    FusedMoEPrepareAndFinalizeWithAllGather, FusedMoEPrepareAndFinalizeWithMC2)


class TestFusedMoEPrepareAndFinalize(unittest.TestCase):

    def setUp(self):
        # Mock FusedMoEConfig
        self.moe_config = MagicMock(spec=FusedMoEConfig)
        self.moe_config.tp_group = MagicMock()
        self.moe_config.tp_group.device_group = MagicMock()
        self.moe_config.dp_size = 1
        self.moe_config.tp_size = 1
        self.moe_config.ep_size = 1
        self.moe_config.dp_group = MagicMock()

    @patch(
        "vllm_ascend.ops.moe.fused_moe_prepare_and_finalize.get_tensor_model_parallel_world_size",
        return_value=1)
    @patch(
        "vllm_ascend.ops.moe.fused_moe_prepare_and_finalize.get_tensor_model_parallel_rank",
        return_value=0)
    @patch(
        "vllm_ascend.ops.moe.fused_moe_prepare_and_finalize.get_forward_context"
    )
    def test_mc2_prepare_finalize(self, mock_get_forward_context, mock_tp_rank,
                                  mock_tp_size):
        mock_context = MagicMock()
        mock_context.mc2_mask = torch.tensor([1, 0, 1])
        mock_context.padded_num_tokens = 4
        mock_get_forward_context.return_value = mock_context

        layer = FusedMoEPrepareAndFinalizeWithMC2(self.moe_config)

        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)

        h_out, r_out, mask = layer.prepare(hidden_states, router_logits)

        # Check padding and split
        self.assertEqual(h_out.shape[0], 4)
        self.assertEqual(r_out.shape[0], 4)
        self.assertEqual(mask.tolist(), [1, 0, 1])

        # Finalize
        result = layer.finalize(h_out, reduce_results=False)
        self.assertEqual(result.shape[0], 3)

    @patch(
        "vllm_ascend.ops.moe.fused_moe_prepare_and_finalize.get_tensor_model_parallel_world_size",
        return_value=2)
    @patch(
        "vllm_ascend.ops.moe.fused_moe_prepare_and_finalize.get_tensor_model_parallel_rank",
        return_value=0)
    @patch(
        "vllm_ascend.ops.moe.fused_moe_prepare_and_finalize.get_forward_context"
    )
    @patch("torch.distributed.all_gather")
    def test_mc2_tp_split_allgather(self, mock_all_gather,
                                    mock_get_forward_context, mock_tp_rank,
                                    mock_tp_size):
        mock_context = MagicMock()
        mock_context.mc2_mask = torch.tensor([1, 0, 1, 0])
        mock_context.padded_num_tokens = 4
        mock_get_forward_context.return_value = mock_context

        layer = FusedMoEPrepareAndFinalizeWithMC2(self.moe_config)
        hidden_states = torch.randn(4, 8)
        router_logits = torch.randn(4, 2)

        h_out, r_out, mask = layer.prepare(hidden_states,
                                           router_logits,
                                           enable_shared_expert_dp=False,
                                           replace_allreduce=False)

        # With TP=2, should split into 2 parts
        self.assertEqual(h_out.shape[0], 2)

        # Mock all_gather behavior
        def mock_all_gather_func(tensor_list, tensor, group=None):
            tensor_list[0] = tensor
            tensor_list[1] = tensor.clone()

        mock_all_gather.side_effect = mock_all_gather_func

        layer.split_hidden_states = [
            torch.zeros_like(h_out),
            torch.zeros_like(h_out)
        ]
        final_result = layer.finalize(h_out, reduce_results=False)

        # Should concat back to original size
        self.assertEqual(final_result.shape[0], 4)

    @patch(
        "vllm_ascend.ops.moe.fused_moe_prepare_and_finalize.get_tensor_model_parallel_world_size",
        return_value=1)
    @patch(
        "vllm_ascend.ops.moe.fused_moe_prepare_and_finalize.get_tensor_model_parallel_rank",
        return_value=0)
    def test_all2all_prepare_finalize(self, mock_tp_rank, mock_tp_size):
        layer = FusedMoEPrepareAndFinalizeWithAll2All(self.moe_config)
        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)

        h_out, r_out, _ = layer.prepare(hidden_states, router_logits)

        # Pad to tp_size=1, so no change
        self.assertEqual(h_out.shape[0], 3)

        result = layer.finalize(h_out, reduce_results=False)
        self.assertEqual(result.shape[0], 3)

    @patch(
        "vllm_ascend.ops.moe.fused_moe_prepare_and_finalize.get_tensor_model_parallel_world_size",
        return_value=2)
    @patch(
        "vllm_ascend.ops.moe.fused_moe_prepare_and_finalize.get_tensor_model_parallel_rank",
        return_value=0)
    @patch("torch.distributed.all_gather")
    def test_all2all_tp_split_allgather(self, mock_all_gather, mock_tp_rank,
                                        mock_tp_size):
        layer = FusedMoEPrepareAndFinalizeWithAll2All(self.moe_config)
        hidden_states = torch.randn(2, 8)
        router_logits = torch.randn(2, 2)

        h_out, r_out, _ = layer.prepare(hidden_states,
                                        router_logits,
                                        enable_shared_expert_dp=False,
                                        replace_allreduce=False)

        # Split due to TP=2
        self.assertEqual(h_out.shape[0], 1)

        # Mock all_gather
        def mock_all_gather_func(tensor_list, tensor, group=None):
            tensor_list[0] = tensor
            tensor_list[1] = tensor.clone()

        mock_all_gather.side_effect = mock_all_gather_func

        layer.split_hidden_states = [
            torch.zeros_like(h_out),
            torch.zeros_like(h_out)
        ]
        final_result = layer.finalize(h_out, reduce_results=False)

        # Should concat back
        self.assertEqual(final_result.shape[0], 2)

    @patch("vllm_ascend.ops.moe.fused_moe_prepare_and_finalize.get_dp_group")
    @patch(
        "vllm_ascend.ops.moe.fused_moe_prepare_and_finalize.tensor_model_parallel_all_reduce"
    )
    @patch(
        "vllm_ascend.ops.moe.fused_moe_prepare_and_finalize.get_forward_context"
    )
    def test_allgather_prepare_finalize(self, mock_get_forward_context,
                                        mock_tp_all_reduce, mock_get_dp_group):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.max_tokens_across_dp = 6
        mock_get_forward_context.return_value = mock_context

        # Create a proper mock for DP group with working all_gather
        mock_dp_group = MagicMock()

        def mock_all_gather_func(tensor, dim):
            # Simulate DP=2: repeat the tensor along the specified dimension
            return torch.cat([tensor, tensor], dim=dim)

        mock_dp_group.all_gather = mock_all_gather_func
        mock_get_dp_group.return_value = mock_dp_group

        self.moe_config.dp_size = 2
        self.moe_config.tp_size = 1
        self.moe_config.ep_size = 1
        self.moe_config.dp_group = mock_dp_group

        layer = FusedMoEPrepareAndFinalizeWithAllGather(self.moe_config)

        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)

        # Mock the gate function for rm_router_logits=False case
        mock_gate = MagicMock()
        mock_gate.return_value = (router_logits.repeat(2, 1), None)

        h_out, r_out, _ = layer.prepare(hidden_states,
                                        router_logits,
                                        rm_router_logits=False,
                                        gate=mock_gate)

        # After all-gather with DP=2, should double the batch size
        self.assertEqual(h_out.shape[0], 12)
        self.assertEqual(r_out.shape[0], 12)

        # Finalize with reduce_scatter
        def mock_reduce_scatter_func(tensor, dim):
            # Simulate reduce_scatter: take first half
            return tensor[:3]

        mock_dp_group.reduce_scatter = mock_reduce_scatter_func
        result = layer.finalize(h_out, reduce_results=False)

        self.assertEqual(result.shape[0], 3)

        # Test with TP all-reduce
        mock_tp_all_reduce.return_value = result
        result_with_tp = layer.finalize(h_out, reduce_results=True)
        self.assertEqual(result_with_tp.shape[0], 3)
