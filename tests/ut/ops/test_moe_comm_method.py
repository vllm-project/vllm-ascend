from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch_npu
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from tests.ut.base import TestBase
from vllm_ascend.ops.activation import SituActivationConfig
from vllm_ascend.ops.fused_moe import comm_utils
from vllm_ascend.ops.fused_moe.mega_moe_adapter import CannMegaMoeLayerCapability
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    AllGatherCommImpl,
    AlltoAllCommImpl,
    FusedMC2CommImpl,
    MC2CommImpl,
    MoECommMethod,
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
    @patch("vllm_ascend.ops.fused_moe.comm_utils.import_module")
    def test_load_cann_megamoe_ops_preloads_comm_context(self, mock_import_module):
        comm_context_module = MagicMock()
        ops_module = MagicMock()
        mock_import_module.side_effect = [comm_context_module, ops_module]

        get_symm_buffer, mega_moe = comm_utils.load_cann_mega_moe_ops(preload_comm_context=True)

        comm_context_module.comm_context_op_builder.load.assert_called_once_with()
        self.assertIs(get_symm_buffer, ops_module.get_symm_buffer_for_mega_moe)
        self.assertIs(mega_moe, ops_module.mega_moe)

    @patch("vllm_ascend.ops.fused_moe.comm_utils.import_module")
    def test_load_cann_megamoe_ops_keeps_a2_a3_lazy_behavior(self, mock_import_module):
        ops_module = MagicMock()
        mock_import_module.return_value = ops_module

        get_symm_buffer, mega_moe = comm_utils.load_cann_mega_moe_ops()

        mock_import_module.assert_called_once_with("cann_ops_transformer.ops")
        self.assertIs(get_symm_buffer, ops_module.get_symm_buffer_for_mega_moe)
        self.assertIs(mega_moe, ops_module.mega_moe)

    def setUp(self):
        comm_utils.load_cann_mega_moe_ops.cache_clear()
        self.mock_ascend_config = MagicMock()
        self.mock_ascend_config.ascend_fusion_config.fusion_ops_gmmswigluquant = False
        self.mock_ascend_config.enable_fused_mc2 = False
        self._patch_get_ascend_config = patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config",
            return_value=self.mock_ascend_config,
        )
        self._patch_get_ascend_config_module = patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            return_value=self.mock_ascend_config,
        )
        self._patch_get_ascend_config.start()
        self._patch_get_ascend_config_module.start()
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

    def tearDown(self):
        comm_utils.load_cann_mega_moe_ops.cache_clear()
        self._patch_get_ascend_config.stop()
        self._patch_get_ascend_config_module.stop()

    def test_a3_w4a8_mxfp_situ_falls_back_to_decomposed_mc2_pipeline(self):
        comm_impl = object.__new__(FusedMC2CommImpl)
        comm_impl.cann_mega_moe_capability = CannMegaMoeLayerCapability(
            False,
            "A2/A3 MegaMoe does not support W4A8MXFP",
            QuantType.W4A8MXFP,
        )
        comm_impl.mega_moe = None
        comm_impl.token_dispatcher = SimpleNamespace(a5_need_extra_args=False)
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
            quant=MoEQuantParams(quant_type=QuantType.W4A8MXFP),
            activation=SituActivationConfig(beta=4.0, linear_beta=25.0),
        )
        expected = object()

        with (
            patch("vllm_ascend.ops.fused_moe.moe_comm_method._MEGA_MOE_SUPPORTED", True),
            patch.object(MoECommMethod, "fused_experts", return_value=expected) as mock_decomposed,
        ):
            result = comm_impl.fused_experts(fused_input)

        self.assertIs(result, expected)
        mock_decomposed.assert_called_once_with(fused_input)

    def test_fused_mc2_unquantized_layer_falls_back_to_decomposed_pipeline(self):
        comm_impl = object.__new__(FusedMC2CommImpl)
        comm_impl.cann_mega_moe_capability = CannMegaMoeLayerCapability(
            False,
            "unsupported quantization",
            QuantType.NONE,
        )
        comm_impl.mega_moe = None
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
            activation=None,
        )

        expected = object()

        with (
            patch("vllm_ascend.ops.fused_moe.moe_comm_method._MEGA_MOE_SUPPORTED", True),
            patch.object(MoECommMethod, "fused_experts", return_value=expected) as mock_decomposed,
        ):
            result = comm_impl.fused_experts(fused_input)

        self.assertIs(result, expected)
        mock_decomposed.assert_called_once_with(fused_input)

    def test_cann_megamoe_w4a8_mxfp_quant_settings(self):
        self.assertEqual(
            comm_utils._get_cann_mega_moe_quant_settings(QuantType.W4A8MXFP),
            (4, torch.float8_e4m3fn, torch_npu.float4_e2m1fn_x2),
        )

    def test_init_a5_megamoe_buffer_covers_prefill_token_shard(self):
        comm_impl = object.__new__(FusedMC2CommImpl)
        comm_impl.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
        comm_impl.token_dispatcher.global_bs = 0
        comm_impl.token_dispatcher.max_num_tokens_per_rank = 512
        comm_impl.token_dispatcher.ep_rank_id = 0
        comm_impl.token_dispatcher.ep_world_size = 32
        comm_impl.token_dispatcher.a5_need_extra_args = True
        comm_impl.moe_config = SimpleNamespace(
            num_experts=896,
            experts_per_token=16,
            hidden_dim=7168,
            intermediate_size_per_partition=1024,
        )
        comm_impl.get_symm_buffer_for_mega_moe = MagicMock(return_value=object())
        fused_input = SimpleNamespace(
            quant=SimpleNamespace(quant_type=QuantType.W4A8MXFP),
        )

        with (
            patch(
                "vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group",
                return_value=SimpleNamespace(device_group=object()),
            ),
        ):
            comm_impl._init_mega_moe_symm_buffer(fused_input)

        call = comm_impl.get_symm_buffer_for_mega_moe.call_args
        self.assertEqual(call.args[1:4], (896, 512, 16))
        self.assertEqual(call.kwargs["max_recv_token_num"], 0)

    def test_init_a3_megamoe_buffer_preserves_receive_token_capacity(self):
        comm_impl = object.__new__(FusedMC2CommImpl)
        comm_impl.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
        comm_impl.token_dispatcher.global_bs = 0
        comm_impl.token_dispatcher.max_num_tokens_per_rank = 512
        comm_impl.token_dispatcher.ep_rank_id = 0
        comm_impl.token_dispatcher.ep_world_size = 32
        comm_impl.token_dispatcher.a5_need_extra_args = False
        comm_impl.moe_config = SimpleNamespace(
            num_experts=896,
            experts_per_token=16,
            hidden_dim=7168,
            intermediate_size_per_partition=1024,
        )
        comm_impl.get_symm_buffer_for_mega_moe = MagicMock(return_value=object())
        fused_input = SimpleNamespace(
            quant=SimpleNamespace(quant_type=QuantType.W4A8),
        )

        with patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group",
            return_value=SimpleNamespace(device_group=object()),
        ):
            comm_impl._init_mega_moe_symm_buffer(fused_input)

        call = comm_impl.get_symm_buffer_for_mega_moe.call_args
        self.assertEqual(call.kwargs["max_recv_token_num"], 512 * 32 * 16)

    def test_apply_cann_megamoe_w4a8_mxfp_situ(self):
        comm_impl = object.__new__(FusedMC2CommImpl)
        comm_impl._mega_moe_symm_buffer = object()
        comm_impl._mega_moe_weight_type = 296
        comm_impl.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
        comm_impl.token_dispatcher.global_bs = 0
        comm_impl.token_dispatcher.max_num_tokens_per_rank = 4096
        comm_impl.mega_moe = MagicMock(
            return_value=(torch.randn(2, 4), torch.zeros(2, dtype=torch.int32))
        )

        w1 = torch.empty(2, 2, 6, dtype=torch.uint8)
        w2 = torch.empty(2, 3, 4, dtype=torch.uint8)
        w1_scale = torch.empty(2, 1, 6, 2, dtype=torch.uint8)
        w2_scale = torch.empty(2, 1, 4, 2, dtype=torch.uint8)
        fused_input = MoEFusedExpertsInput(
            hidden_states=torch.randn(2, 4),
            topk_weights=torch.ones(2, 1),
            topk_ids=torch.zeros(2, 1, dtype=torch.int32),
            weights=MoEWeights(
                w1=w1,
                w2=w2,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            ),
            routing=MoERoutingParams(
                expert_map=None,
                global_redundant_expert_num=0,
                mc2_mask=torch.ones(2, dtype=torch.bool),
                apply_router_weight_on_input=False,
            ),
            quant=MoEQuantParams(quant_type=QuantType.W4A8MXFP),
            activation=SituActivationConfig(beta=4.0, linear_beta=25.0),
        )

        comm_impl._apply_cann_mega_moe(fused_input, fused_input.topk_ids)

        call = comm_impl.mega_moe.call_args
        self.assertEqual(call.args[3][0].shape, (2, 6, 2))
        self.assertEqual(call.args[4][0].shape, (2, 4, 3))
        self.assertEqual(call.kwargs["l1_weights_sf"][0].shape, (2, 6, 1, 2))
        self.assertEqual(call.kwargs["l2_weights_sf"][0].shape, (2, 4, 1, 2))
        self.assertEqual(call.args[3][0].data_ptr(), w1.data_ptr())
        self.assertEqual(call.args[4][0].data_ptr(), w2.data_ptr())
        self.assertEqual(call.kwargs["activation"], "situglu")
        self.assertEqual(
            call.kwargs["activation_params"], {"beta": 4.0, "linear_beta": 25.0}
        )
        self.assertIsNone(call.kwargs["activation_clamp"])
        self.assertIsNone(call.kwargs["x_active_mask"])
        self.assertEqual(call.kwargs["weight1_type"], 296)
        self.assertEqual(call.kwargs["weight2_type"], 296)

        comm_impl.mega_moe.reset_mock()
        list_weights_input = replace(
            fused_input,
            weights=MoEWeights(
                w1=list(w1.unbind(0)),
                w2=list(w2.unbind(0)),
                w1_scale=list(w1_scale.unbind(0)),
                w2_scale=list(w2_scale.unbind(0)),
            ),
        )
        comm_impl._apply_cann_mega_moe(list_weights_input, list_weights_input.topk_ids)

        call = comm_impl.mega_moe.call_args
        self.assertEqual(len(call.args[3]), 2)
        self.assertEqual(len(call.args[4]), 2)
        self.assertEqual(call.args[3][0].shape, (6, 2))
        self.assertEqual(call.args[4][0].shape, (4, 3))
        self.assertEqual(call.kwargs["l1_weights_sf"][0].shape, (6, 1, 2))
        self.assertEqual(call.kwargs["l2_weights_sf"][0].shape, (4, 1, 2))
        self.assertEqual(call.args[3][0].data_ptr(), w1[0].data_ptr())
        self.assertEqual(call.args[4][0].data_ptr(), w2[0].data_ptr())

        comm_impl.mega_moe.reset_mock()
        swiglu_input = replace(fused_input, activation="silu")
        comm_impl._apply_cann_mega_moe(swiglu_input, swiglu_input.topk_ids)

        call = comm_impl.mega_moe.call_args
        self.assertNotIn("activation", call.kwargs)
        self.assertNotIn("activation_alpha", call.kwargs)
        self.assertNotIn("activation_beta", call.kwargs)

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
