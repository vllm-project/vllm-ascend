from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from tests.ut.base import TestBase
from vllm_ascend.ops.moe.moe_comm_method import (AllGatherCommImpl,
                                                 AlltoAllCommImpl, MC2CommImpl)


class TestMoECommMethod(TestBase):

    def setUp(self):
        # Mock FusedMoEConfig
        self.moe_config = MagicMock(spec=FusedMoEConfig)
        self.moe_config.num_experts = 8
        self.moe_config.num_local_experts = 2
        self.moe_config.experts_per_token = 2
        self.moe_config.tp_group = MagicMock()
        self.moe_config.tp_group.device_group = MagicMock()
        self.moe_config.dp_size = 1
        self.moe_config.tp_size = 1
        self.moe_config.ep_size = 1
        self.moe_config.dp_group = MagicMock()
        self.moe_config.num_global_redundant_experts = 0

    @patch("vllm_ascend.ops.moe.moe_comm_method.get_forward_context")
    @patch(
        "vllm_ascend.ops.moe.moe_comm_method.FusedMoEPrepareAndFinalizeWithAllGather"
    )
    @patch("vllm_ascend.ops.moe.moe_comm_method.TokenDispatcherWithAllGather")
    def test_all_gather_comm_impl(self, mock_token_dispatcher,
                                  mock_prepare_finalize,
                                  mock_get_forward_context):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "all_gather"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = (torch.randn(4, 8),
                                                 torch.randn(4, 2), None)
        mock_pf_instance.finalize.return_value = torch.randn(4, 8)
        mock_prepare_finalize.return_value = mock_pf_instance

        # Mock token dispatcher
        mock_td_instance = MagicMock()
        mock_token_dispatcher.return_value = mock_td_instance

        # Create instance
        comm_impl = AllGatherCommImpl(self.moe_config)

        # Test prepare method
        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)
        h_out, r_out = comm_impl.prepare(hidden_states, router_logits)

        # Verify prepare was called with correct arguments
        mock_pf_instance.prepare.assert_called_once_with(
            hidden_states, router_logits, False, False, False, None)

        # Test finalize method
        comm_impl.finalize(h_out, reduce_results=True)
        mock_pf_instance.finalize.assert_called_once_with(h_out, True)

    @patch("vllm_ascend.ops.moe.moe_comm_method.get_forward_context")
    @patch(
        "vllm_ascend.ops.moe.moe_comm_method.FusedMoEPrepareAndFinalizeWithMC2"
    )
    @patch("vllm_ascend.ops.moe.moe_comm_method.TokenDispatcherWithMC2")
    def test_mc2_comm_impl(self, mock_token_dispatcher, mock_prepare_finalize,
                           mock_get_forward_context):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "mc2"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = (torch.randn(4, 8),
                                                 torch.randn(4, 2),
                                                 torch.tensor([1, 0, 1, 0]))
        mock_pf_instance.finalize.return_value = torch.randn(4, 8)
        mock_prepare_finalize.return_value = mock_pf_instance

        # Mock token dispatcher
        mock_td_instance = MagicMock()
        mock_token_dispatcher.return_value = mock_td_instance

        # Create instance
        comm_impl = MC2CommImpl(self.moe_config)

        # Test prepare method
        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)
        h_out, r_out = comm_impl.prepare(hidden_states, router_logits)

        # Verify prepare was called with correct arguments
        mock_pf_instance.prepare.assert_called_once_with(
            hidden_states, router_logits, False, False, False, None)

        # Test finalize method
        comm_impl.finalize(h_out, reduce_results=True)
        mock_pf_instance.finalize.assert_called_once_with(h_out, True)

    @patch("vllm_ascend.ops.moe.moe_comm_method.get_forward_context")
    @patch(
        "vllm_ascend.ops.moe.moe_comm_method.FusedMoEPrepareAndFinalizeWithAll2All"
    )
    @patch("vllm_ascend.ops.moe.moe_comm_method.TokenDispatcherWithAll2AllV")
    def test_alltoall_comm_impl(self, mock_token_dispatcher,
                                mock_prepare_finalize,
                                mock_get_forward_context):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "alltoall"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = (torch.randn(4, 8),
                                                 torch.randn(4, 2), None)
        mock_pf_instance.finalize.return_value = torch.randn(4, 8)
        mock_prepare_finalize.return_value = mock_pf_instance

        # Mock token dispatcher
        mock_td_instance = MagicMock()
        mock_token_dispatcher.return_value = mock_td_instance

        # Create instance
        comm_impl = AlltoAllCommImpl(self.moe_config)

        # Test prepare method
        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)
        h_out, r_out = comm_impl.prepare(hidden_states, router_logits)

        # Verify prepare was called with correct arguments
        mock_pf_instance.prepare.assert_called_once_with(
            hidden_states, router_logits, False, False, False, None)

    @patch("vllm_ascend.ops.moe.moe_comm_method.get_forward_context")
    @patch(
        "vllm_ascend.ops.moe.moe_comm_method.FusedMoEPrepareAndFinalizeWithAllGather"
    )
    @patch("vllm_ascend.ops.moe.moe_comm_method.TokenDispatcherWithAllGather")
    @patch("vllm_ascend.ops.moe.moe_comm_method.unified_apply_mlp")
    def test_fused_experts_method(self, mock_unified_apply_mlp,
                                  mock_token_dispatcher, mock_prepare_finalize,
                                  mock_get_forward_context):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "all_gather"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = (torch.randn(4, 8),
                                                 torch.randn(4, 2), None)
        mock_pf_instance.finalize.return_value = torch.randn(4, 8)
        mock_prepare_finalize.return_value = mock_pf_instance

        # Mock token dispatcher
        mock_td_instance = MagicMock()
        mock_td_instance.token_dispatch.return_value = {
            "hidden_states": torch.randn(6, 8),
            "group_list": torch.tensor([2, 2, 2]),
            "group_list_type": 1
        }
        mock_td_instance.token_combine.return_value = torch.randn(4, 8)
        mock_token_dispatcher.return_value = mock_td_instance

        # Mock unified_apply_mlp
        mock_unified_apply_mlp.return_value = torch.randn(6, 8)

        # Create instance
        comm_impl = AllGatherCommImpl(self.moe_config)

        # Test fused_experts method
        hidden_states = torch.randn(4, 8).contiguous()
        w1 = torch.randn(16, 8).contiguous()
        w2 = torch.randn(16, 8).contiguous()
        topk_weights = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.8, 0.2],
                                     [0.6, 0.4]])
        topk_ids = torch.tensor([[0, 1], [1, 2], [2, 0], [1, 1]])
        row_idx = torch.arange(4)

        # Make sure tensors are contiguous and have correct strides
        hidden_states = hidden_states.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()

        result = comm_impl.fused_experts(hidden_states=hidden_states,
                                         w1=w1,
                                         w2=w2,
                                         topk_weights=topk_weights,
                                         topk_ids=topk_ids,
                                         row_idx=row_idx,
                                         activation="silu")

        # Verify result shape
        self.assertEqual(result.shape, (4, 8))

        # Verify token_dispatch was called
        mock_td_instance.token_dispatch.assert_called_once()

        # Verify unified_apply_mlp was called
        mock_unified_apply_mlp.assert_called_once()

        # Verify token_combine was called
        mock_td_instance.token_combine.assert_called_once()
