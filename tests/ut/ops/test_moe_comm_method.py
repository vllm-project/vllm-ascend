from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from tests.ut.base import TestBase
from vllm_ascend.ops.activation import SituActivationConfig
from vllm_ascend.ops.fused_moe import comm_utils
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    AllGatherCommImpl,
    AlltoAllCommImpl,
    FusedMC2CommImpl,
    MC2CommImpl,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEAllGatherCombineMetadata,
    MoEFusedExpertsInput,
    MoEPrepareOutput,
    MoEQuantParams,
    MoERoutingParams,
    MoEWeights,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import MoETokenDispatchOutput, TokenDispatcherWithMC2
from vllm_ascend.quantization.methods.base import QuantType


class TestMoECommMethod(TestBase):
    def setUp(self):
        self.mock_ascend_config = MagicMock()
        self.mock_ascend_config.ascend_fusion_config.fusion_ops_gmmswigluquant = False
        self.mock_ascend_config.enable_fused_mc2 = False
        self.mock_ascend_config.mega_moe_max_tokens = 6144
        self.mock_ascend_config.scheduler_config.recompute_scheduler_enable = False
        self._patch_get_ascend_config = patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config",
            return_value=self.mock_ascend_config,
        )
        self._patch_get_ascend_config_module = patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            return_value=self.mock_ascend_config,
        )
        self._patch_get_ascend_config_forward_context = patch(
            "vllm_ascend.ascend_forward_context.get_ascend_config",
            return_value=self.mock_ascend_config,
        )
        self._patch_get_ascend_config.start()
        self._patch_get_ascend_config_module.start()
        self._patch_get_ascend_config_forward_context.start()
        # Mock FusedMoEConfig
        self.moe_config = MagicMock(spec=FusedMoEConfig)
        self.moe_config.num_experts = 8
        self.moe_config.num_local_experts = 2
        self.moe_config.experts_per_token = 2
        self.moe_config.tp_group = MagicMock()
        self.moe_config.tp_group.device_group = MagicMock()
        self.moe_config.dp_size = 1
        self.moe_config.tp_size = 1
        self.moe_config.pcp_size = 1
        self.moe_config.ep_size = 1
        self.moe_config.dp_group = MagicMock()
        self.moe_config.global_redundant_expert_num = 0

    def _make_fused_mc2_comm_for_buffer_init(self):
        comm_impl = object.__new__(FusedMC2CommImpl)
        comm_impl.moe_config = self.moe_config
        comm_impl.moe_config.hidden_dim = 16
        comm_impl.moe_config.intermediate_size_per_partition = 32
        comm_impl.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
        comm_impl.token_dispatcher.global_bs = 0
        comm_impl.token_dispatcher.max_num_tokens_per_rank = 128
        comm_impl.token_dispatcher.ep_world_size = 8
        comm_impl.token_dispatcher.ep_rank_id = 0
        comm_impl.get_symm_buffer_for_mega_moe = MagicMock(return_value="symm_buffer")
        comm_impl.mega_moe_symm_buffer = None
        return comm_impl

    def tearDown(self):
        self._patch_get_ascend_config.stop()
        self._patch_get_ascend_config_module.stop()
        self._patch_get_ascend_config_forward_context.stop()

    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.logger.warning_once")
    def test_mega_moe_symm_buffer_uses_mega_moe_max_tokens(self, mock_warning_once, mock_get_mc2_group):
        self.mock_ascend_config.mega_moe_max_tokens = 32768
        mock_mc2_group = MagicMock()
        mock_mc2_group.device_group = "mc2_group"
        mock_get_mc2_group.return_value = mock_mc2_group
        comm_impl = self._make_fused_mc2_comm_for_buffer_init()

        comm_impl._init_mega_moe_symm_buffer(is_decode_only_node=False)

        comm_impl.get_symm_buffer_for_mega_moe.assert_called_once()
        call_args = comm_impl.get_symm_buffer_for_mega_moe.call_args
        self.assertEqual(call_args.args[:4], ("mc2_group", 8, 128, 2))
        self.assertEqual(call_args.kwargs["max_recv_token_num"], 32768)
        mock_warning_once.assert_called_once()
        self.assertIn("mega_moe_max_tokens", mock_warning_once.call_args.args[0])

    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.logger.warning_once")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group")
    def test_mega_moe_symm_buffer_uses_safe_capacity_for_d_node(self, mock_get_mc2_group, mock_warning_once):
        self.mock_ascend_config.mega_moe_max_tokens = 32768
        mock_mc2_group = MagicMock()
        mock_mc2_group.device_group = "mc2_group"
        mock_get_mc2_group.return_value = mock_mc2_group
        comm_impl = self._make_fused_mc2_comm_for_buffer_init()

        comm_impl._init_mega_moe_symm_buffer(is_decode_only_node=True)

        call_args = comm_impl.get_symm_buffer_for_mega_moe.call_args
        self.assertEqual(call_args.kwargs["max_recv_token_num"], 1024)
        mock_warning_once.assert_not_called()

    def test_cann_mega_moe_maps_kimi_situ_activation(self):
        activation, params = comm_utils.get_cann_mega_moe_activation_settings(
            SituActivationConfig(beta=4.0, linear_beta=25.0)
        )

        self.assertEqual(activation, "situglu")
        self.assertEqual(params, {"beta": 4.0, "linear_beta": 25.0})

    def test_cann_mega_moe_forwards_kimi_situ_to_operator(self):
        comm_impl = object.__new__(FusedMC2CommImpl)
        comm_impl.token_dispatcher = MagicMock(spec=TokenDispatcherWithMC2)
        comm_impl.token_dispatcher.global_bs = 1
        comm_impl.token_dispatcher.max_num_tokens_per_rank = 2
        comm_impl.mega_moe_symm_buffer = SimpleNamespace(
            dispatch_quant_mode=0,
            dispatch_quant_out_dtype=None,
        )
        comm_impl._mega_moe_supports_situ = True
        expected = torch.randn(2, 4)
        expert_tokens = torch.ones(2, dtype=torch.int32)
        comm_impl.mega_moe = MagicMock(return_value=(expected, expert_tokens))
        fused_input = MoEFusedExpertsInput(
            hidden_states=torch.randn(2, 4),
            topk_weights=torch.ones(2, 1),
            topk_ids=torch.zeros(2, 1, dtype=torch.int32),
            weights=MoEWeights(
                w1=[torch.randn(1, 4, 4)],
                w2=[torch.randn(1, 2, 4)],
            ),
            routing=MoERoutingParams(
                expert_map=None,
                global_redundant_expert_num=0,
                mc2_mask=None,
                apply_router_weight_on_input=False,
            ),
            quant=MoEQuantParams(),
            activation=SituActivationConfig(beta=4.0, linear_beta=25.0),
        )

        result, result_expert_tokens = comm_impl._apply_cann_mega_moe(
            fused_input,
            fused_input.topk_ids,
            is_decode_only_node=False,
        )

        self.assertIs(result, expected)
        self.assertIs(result_expert_tokens, expert_tokens)
        call = comm_impl.mega_moe.call_args
        self.assertEqual(call.kwargs["activation"], "situglu")
        self.assertEqual(call.kwargs["activation_params"], {"beta": 4.0, "linear_beta": 25.0})
        self.assertIsNone(call.kwargs["l1_weights_sf"])
        self.assertIsNone(call.kwargs["l2_weights_sf"])

    def test_cann_mega_moe_maps_default_silu_to_swiglu(self):
        self.assertEqual(comm_utils.get_cann_mega_moe_activation_settings("silu"), ("swiglu", None))

    def test_cann_mega_moe_preserves_swigluoai_default(self):
        self.assertEqual(
            comm_utils.get_cann_mega_moe_activation_settings("swigluoai_uninterleave"),
            ("swiglu", None),
        )

    def test_cann_mega_moe_accepts_unquantized_weights(self):
        self.assertEqual(comm_utils._get_cann_mega_moe_quant_settings(QuantType.NONE), (0, None, None))

    def test_cann_mega_moe_detects_situ_api(self):
        def mega_moe(*args, activation="swiglu", activation_params=None):
            return activation, activation_params

        def legacy_mega_moe(*args, activation_clamp=None):
            return activation_clamp

        self.assertTrue(comm_utils.cann_mega_moe_supports_situ(mega_moe))
        self.assertFalse(comm_utils.cann_mega_moe_supports_situ(legacy_mega_moe))

    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group")
    def test_cann_mega_moe_caps_max_recv_tokens(self, mock_get_mc2_group):
        comm_impl = object.__new__(FusedMC2CommImpl)
        comm_impl.moe_config = self.moe_config
        comm_impl.moe_config.num_experts = 384
        comm_impl.moe_config.experts_per_token = 8
        comm_impl.moe_config.hidden_dim = 4096
        comm_impl.moe_config.intermediate_size_per_partition = 1024
        comm_impl.token_dispatcher = MagicMock(spec=TokenDispatcherWithMC2)
        comm_impl.token_dispatcher.global_bs = 32768
        comm_impl.token_dispatcher.ep_world_size = 64
        comm_impl.token_dispatcher.ep_rank_id = 0
        comm_impl.get_symm_buffer_for_mega_moe = MagicMock(return_value=object())
        mock_get_mc2_group.return_value.device_group = MagicMock()

        comm_impl._init_mega_moe_symm_buffer(2, torch.int8, is_decode_only_node=False)

        call = comm_impl.get_symm_buffer_for_mega_moe.call_args
        self.assertEqual(call.args[2], 512)
        self.assertEqual(call.kwargs["max_recv_token_num"], 6144)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAllGather")
    def test_all_gather_comm_impl(self, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "all_gather"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = MoEPrepareOutput(
            hidden_states=torch.randn(4, 8),
            router_logits=torch.randn(4, 2),
            mc2_mask=None,
            padded_hidden_states_shape=None,
        )
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
        prepare_output = comm_impl.prepare(hidden_states, router_logits)
        h_out = prepare_output.hidden_states
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape

        # Verify prepare was called with correct arguments
        mock_pf_instance.prepare.assert_called_once_with(hidden_states, router_logits, False, False, QuantType.NONE)

        # Test finalize method
        comm_impl.finalize(h_out, reduce_results=True, padded_hidden_states_shape=padded_hidden_states_shape)
        mock_pf_instance.finalize.assert_called_once_with(h_out, True, None)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithMC2")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithMC2")
    def test_mc2_comm_impl(self, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "mc2"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = MoEPrepareOutput(
            hidden_states=torch.randn(4, 8),
            router_logits=torch.randn(4, 2),
            mc2_mask=torch.tensor([1, 0, 1, 0]),
            padded_hidden_states_shape=None,
        )
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
        prepare_output = comm_impl.prepare(hidden_states, router_logits)
        h_out = prepare_output.hidden_states
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape

        # Verify prepare was called with correct arguments
        mock_pf_instance.prepare.assert_called_once_with(hidden_states, router_logits, False, False, QuantType.NONE)

        # Test finalize method
        comm_impl.finalize(h_out, reduce_results=True, padded_hidden_states_shape=padded_hidden_states_shape)
        mock_pf_instance.finalize.assert_called_once_with(h_out, True, None)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAll2All")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAll2AllV")
    def test_alltoall_comm_impl(self, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "alltoall"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = MoEPrepareOutput(
            hidden_states=torch.randn(4, 8),
            router_logits=torch.randn(4, 2),
            mc2_mask=None,
            padded_hidden_states_shape=None,
        )
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
        _ = comm_impl.prepare(hidden_states, router_logits)

        # Verify prepare was called with correct arguments
        mock_pf_instance.prepare.assert_called_once_with(hidden_states, router_logits, False, False, QuantType.NONE)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.unified_apply_mlp")
    @patch("torch.npu.current_stream", MagicMock())
    def test_fused_experts_method(
        self, mock_unified_apply_mlp, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context
    ):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "all_gather"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = MoEPrepareOutput(
            hidden_states=torch.randn(4, 8),
            router_logits=torch.randn(4, 2),
            mc2_mask=None,
            padded_hidden_states_shape=None,
        )
        mock_pf_instance.finalize.return_value = torch.randn(4, 8)
        mock_prepare_finalize.return_value = mock_pf_instance

        # Mock token dispatcher
        mock_td_instance = MagicMock()
        dispatch_topk_weights = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.8, 0.2], [0.6, 0.4]])
        mock_td_instance.token_dispatch.return_value = MoETokenDispatchOutput(
            hidden_states=torch.randn(6, 8),
            group_list=torch.tensor([2, 2, 2]),
            group_list_type=1,
            combine_metadata=MoEAllGatherCombineMetadata(
                topk_weights=dispatch_topk_weights,
                expanded_row_idx=torch.arange(8, dtype=torch.int32),
                restore_shape=torch.Size([4, 8]),
            ),
        )
        mock_td_instance.token_combine.return_value = torch.randn(4, 8)
        mock_token_dispatcher.return_value = mock_td_instance

        # Mock unified_apply_mlp returns (tensor, event) tuple
        mock_unified_apply_mlp.return_value = (torch.randn(6, 8), MagicMock())

        # Create instance
        comm_impl = AllGatherCommImpl(self.moe_config)

        # Test fused_experts method
        hidden_states = torch.randn(4, 8).contiguous()
        w1 = torch.randn(16, 8).contiguous()
        w2 = torch.randn(16, 8).contiguous()
        topk_weights = dispatch_topk_weights
        topk_ids = torch.tensor([[0, 1], [1, 2], [2, 0], [1, 1]])

        # Make sure tensors are contiguous and have correct strides
        hidden_states = hidden_states.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()

        result = comm_impl.fused_experts(
            fused_experts_input=MoEFusedExpertsInput(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                weights=MoEWeights(
                    w1=[w1],
                    w2=[w2],
                ),
                routing=MoERoutingParams(
                    expert_map=None,
                    global_redundant_expert_num=0,
                    mc2_mask=None,
                    apply_router_weight_on_input=False,
                ),
                activation="silu",
                need_trans=False,
                dynamic_eplb=False,
                quant=MoEQuantParams(),
            )
        )

        # Verify result shape
        self.assertEqual(result.routed_out.shape, (4, 8))

        # Verify token_dispatch was called
        mock_td_instance.token_dispatch.assert_called_once()

        # Verify unified_apply_mlp was called
        mock_unified_apply_mlp.assert_called_once()
        mlp_compute_input = mock_unified_apply_mlp.call_args.kwargs["mlp_compute_input"]
        self.assertFalse(mlp_compute_input.fusion)
        self.assertFalse(mlp_compute_input.quant.is_mxfp)

        # Verify token_combine was called
        mock_td_instance.token_combine.assert_called_once_with(
            hidden_states=mock_unified_apply_mlp.return_value[0],
            combine_metadata=mock_td_instance.token_dispatch.return_value.combine_metadata,
        )
