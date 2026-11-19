from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from tests.ut.base import TestBase
from vllm_ascend._310p.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod310
from vllm_ascend._310p.fused_moe.moe_comm_method import AllGatherCommImpl310
from vllm_ascend._310p.quantization.methods.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod310
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.quantization.quant_type import QuantType


def _make_fake_layer(lora_context):
    layer = torch.nn.Module()
    layer.w13_weight = torch.randn(2, 16, 32, dtype=torch.float16)
    layer.w2_weight = torch.randn(2, 16, 16, dtype=torch.float16)
    layer.ascend_expert_map = None
    layer.global_redundant_expert_num = 0
    layer.ascend_mc2_mask = None
    layer.apply_router_weight_on_input = False
    layer.ascend_pertoken_scale = None
    layer.activation = "silu"
    layer._ascend_moe_lora_context = lora_context
    return layer


class TestMoELoraContext310P(TestBase):
    @patch("vllm_ascend._310p.fused_moe.fused_moe._EXTRA_CTX")
    def test_unquantized_310p_apply_forwards_lora_context(self, mock_extra_ctx):
        lora_ctx = SimpleNamespace(tag="moe_lora_ctx")
        layer = _make_fake_layer(lora_ctx)
        mock_comm = Mock()
        mock_comm.fused_experts.return_value = torch.zeros(2, 16, dtype=torch.float16)
        mock_extra_ctx.moe_comm_method = mock_comm

        method = AscendUnquantizedFusedMoEMethod310.__new__(AscendUnquantizedFusedMoEMethod310)
        method.lora_context = None
        method._lora_routing = None

        x = torch.randn(2, 16, dtype=torch.float16)
        topk_weights = torch.ones(2, 1, dtype=torch.float32)
        topk_ids = torch.zeros(2, 1, dtype=torch.int32)

        method.apply(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )

        fused_experts_input = mock_comm.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertIs(fused_experts_input.lora_context, lora_ctx)

    @patch("vllm_ascend._310p.quantization.methods.w8a8_dynamic._EXTRA_CTX")
    def test_w8a8_dynamic_310p_apply_forwards_lora_context(self, mock_extra_ctx):
        lora_ctx = SimpleNamespace(tag="quant_moe_lora_ctx")
        layer = _make_fake_layer(lora_ctx)
        mock_comm = Mock()
        mock_comm.fused_experts.return_value = torch.zeros(2, 16, dtype=torch.float16)
        mock_extra_ctx.moe_comm_method = mock_comm

        method = AscendW8A8DynamicFusedMoEMethod310.__new__(AscendW8A8DynamicFusedMoEMethod310)
        method.in_dtype = torch.float16

        x = torch.randn(2, 16, dtype=torch.float16)
        topk_weights = torch.ones(2, 1, dtype=torch.float32)
        topk_ids = torch.zeros(2, 1, dtype=torch.int32)

        method.apply(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )

        fused_experts_input = mock_comm.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertIs(fused_experts_input.lora_context, lora_ctx)

    def test_unquantized_310p_gmm_hooks_apply_lora_w13_and_w2(self):
        lora_ctx = SimpleNamespace(exchanged_lora_indices=torch.tensor([0, 0]))
        layer = _make_fake_layer(lora_ctx)

        method = AscendUnquantizedFusedMoEMethod310.__new__(AscendUnquantizedFusedMoEMethod310)
        method.lora_context = None
        method._lora_routing = None
        method.set_lora_context(lora_ctx)
        self.assertIs(method.lora_context, lora_ctx)

        hidden = torch.randn(2, 16, dtype=torch.float16)
        gate_up = torch.randn(2, 32, dtype=torch.float16)
        silu_out = torch.randn(2, 16, dtype=torch.float16)
        down_out = torch.randn(2, 16, dtype=torch.float16)
        group_list = torch.tensor([1, 1], dtype=torch.int64)
        expanded_row_idx = torch.tensor([0, 1], dtype=torch.int32)
        topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)

        mlp_input = MoEMlpComputeInput(
            hidden_states=hidden,
            group_list=group_list,
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MagicMock(),
            quant=MagicMock(),
            fusion=True,
            layer=layer,
            activation=MoEActivation.SILU,
            lora_context=lora_ctx,
            expanded_row_idx=expanded_row_idx,
            topk_ids=topk_ids,
        )

        fake_routing = MagicMock(name="lora_routing")
        with (
            patch(
                "vllm_ascend._310p.fused_moe.fused_moe.torch_npu.npu_grouped_matmul",
                side_effect=[[gate_up], [down_out]],
            ),
            patch(
                "vllm_ascend._310p.fused_moe.fused_moe.torch_npu.npu_swiglu",
                return_value=silu_out,
            ),
            patch(
                "vllm_ascend.lora.fused_moe.has_lora",
                return_value=True,
            ),
            patch(
                "vllm_ascend.lora.fused_moe._recover_moe_lora_routing_allgather",
                return_value=fake_routing,
            ) as mock_recover,
            patch("vllm_ascend.lora.fused_moe.moe_lora_apply_w13") as mock_w13,
            patch("vllm_ascend.lora.fused_moe.moe_lora_apply_w2") as mock_w2,
        ):
            act_out, scale = method.apply_gmm1_act_quant(mlp_input)
            self.assertIs(act_out, silu_out)
            self.assertIsNone(scale)
            mock_recover.assert_called_once_with(lora_ctx, expanded_row_idx, topk_ids)
            mock_w13.assert_called_once_with(
                lora_ctx,
                gate_up_out=gate_up,
                hidden_states=hidden,
                lora_routing=fake_routing,
            )

            final_out = method.apply_gmm2(mlp_input, act_out, scale)
            self.assertIs(final_out, down_out)
            mock_w2.assert_called_once_with(
                lora_ctx,
                down_out=down_out,
                silu_out=silu_out,
                lora_routing=fake_routing,
            )
            self.assertIsNone(method._lora_routing)

    def test_w8a8_dynamic_310p_gmm_hooks_apply_lora_w13_and_w2(self):
        lora_ctx = SimpleNamespace(exchanged_lora_indices=torch.tensor([0, 0]))
        layer = _make_fake_layer(lora_ctx)
        layer.w13_weight_scale = torch.ones(2, 32, dtype=torch.float32)
        layer.w2_weight_scale = torch.ones(2, 16, dtype=torch.float32)

        method = AscendW8A8DynamicFusedMoEMethod310.__new__(AscendW8A8DynamicFusedMoEMethod310)
        method.lora_context = None
        method._lora_routing = None
        method.set_lora_context(lora_ctx)
        self.assertIs(method.lora_context, lora_ctx)

        hidden = torch.randn(2, 16, dtype=torch.float16)
        gate_up = torch.randn(2, 32, dtype=torch.float16)
        silu_out = torch.randn(2, 16, dtype=torch.float16)
        down_out = torch.randn(2, 16, dtype=torch.float16)
        group_list = torch.tensor([1, 1], dtype=torch.int64)
        expanded_row_idx = torch.tensor([0, 1], dtype=torch.int32)
        topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)

        mlp_input = MoEMlpComputeInput(
            hidden_states=hidden,
            group_list=group_list,
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MagicMock(),
            quant=MagicMock(),
            fusion=True,
            layer=layer,
            activation=MoEActivation.SILU,
            lora_context=lora_ctx,
            expanded_row_idx=expanded_row_idx,
            topk_ids=topk_ids,
        )

        fake_routing = MagicMock(name="lora_routing")
        with (
            patch(
                "vllm_ascend._310p.quantization.methods.w8a8_dynamic.torch_npu.npu_quant_grouped_matmul_dequant",
                side_effect=[gate_up, down_out],
                create=True,
            ),
            patch(
                "vllm_ascend._310p.quantization.methods.w8a8_dynamic.torch_npu.npu_swiglu",
                return_value=silu_out,
            ),
            patch("vllm_ascend.lora.fused_moe.has_lora", return_value=True),
            patch(
                "vllm_ascend.lora.fused_moe._recover_moe_lora_routing_allgather",
                return_value=fake_routing,
            ) as mock_recover,
            patch("vllm_ascend.lora.fused_moe.moe_lora_apply_w13") as mock_w13,
            patch("vllm_ascend.lora.fused_moe.moe_lora_apply_w2") as mock_w2,
        ):
            act_out, scale = method.apply_gmm1_act_quant(mlp_input)
            self.assertIs(act_out, silu_out)
            self.assertIsNone(scale)
            mock_recover.assert_called_once_with(lora_ctx, expanded_row_idx, topk_ids)
            mock_w13.assert_called_once_with(
                lora_ctx,
                gate_up_out=gate_up,
                hidden_states=hidden,
                lora_routing=fake_routing,
            )

            final_out = method.apply_gmm2(mlp_input, act_out, scale)
            self.assertIs(final_out, down_out)
            mock_w2.assert_called_once_with(
                lora_ctx,
                down_out=down_out,
                silu_out=silu_out,
                lora_routing=fake_routing,
            )
            self.assertIsNone(method._lora_routing)

    def test_w8a8_dynamic_310p_fused_experts_with_active_lora_dispatches_310p_hooks(self):
        lora_ctx = SimpleNamespace(exchanged_lora_indices=torch.tensor([0, 0]))
        layer = _make_fake_layer(lora_ctx)
        layer.w13_weight = torch.randint(-8, 8, (2, 32, 16), dtype=torch.int8)
        layer.w2_weight = torch.randint(-8, 8, (2, 16, 16), dtype=torch.int8)
        layer.w13_weight_scale = torch.ones(2, 32, dtype=torch.float32)
        layer.w2_weight_scale = torch.ones(2, 16, dtype=torch.float32)

        method = AscendW8A8DynamicFusedMoEMethod310.__new__(AscendW8A8DynamicFusedMoEMethod310)
        method.in_dtype = torch.float16
        method.lora_context = None
        method._lora_routing = None
        method.set_lora_context(lora_ctx)

        comm = AllGatherCommImpl310.__new__(AllGatherCommImpl310)
        comm.moe_config = SimpleNamespace(
            activation=MoEActivation.SILU,
            swiglu_limit=0.0,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            activation_situ_beta=None,
            activation_situ_linear_beta=None,
        )
        comm.token_dispatcher = MagicMock()

        hidden = torch.randn(2, 16, dtype=torch.float16)
        sorted_hidden = torch.randn(2, 16, dtype=torch.float16)
        topk_weights = torch.ones(2, 1, dtype=torch.float16)
        topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)
        expanded_row_idx = torch.tensor([0, 1], dtype=torch.int32)
        group_list = torch.tensor([1, 2], dtype=torch.int64)
        gate_up = torch.randn(2, 32, dtype=torch.float16)
        silu_out = torch.randn(2, 16, dtype=torch.float16)
        down_out = torch.randn(2, 16, dtype=torch.float16)
        combined_out = torch.randn(2, 16, dtype=torch.float16)

        combine_metadata = SimpleNamespace(
            topk_weights=topk_weights,
            expanded_row_idx=expanded_row_idx,
            restore_shape=hidden.shape,
        )
        comm.token_dispatcher.token_dispatch.return_value = SimpleNamespace(
            hidden_states=sorted_hidden,
            group_list=group_list,
            group_list_type=0,
            dynamic_scale=None,
            topk_scales=None,
            combine_metadata=combine_metadata,
        )
        comm.token_dispatcher.token_combine.return_value = combined_out

        fused_experts_input = build_fused_experts_input(
            hidden_states=hidden,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            layer=layer,
            quant_type=QuantType.W8A8,
            dynamic_eplb=False,
            activation=MoEActivation.SILU,
            lora_context=lora_ctx,
        )

        fake_routing = MagicMock(name="lora_routing")
        with (
            patch("vllm_ascend.ops.fused_moe.moe_comm_method._EXTRA_CTX") as mock_ctx,
            patch("vllm_ascend.ops.fused_moe.moe_comm_method.maybe_record_event", return_value=None),
            patch("vllm_ascend.ops.fused_moe.moe_mlp.maybe_record_event", return_value=None),
            patch(
                "vllm_ascend.ops.fused_moe.dataclass.moe_mlp.enable_fusion_gmmswigluquant",
                return_value=True,
            ),
            patch(
                "vllm_ascend._310p.quantization.methods.w8a8_dynamic.torch_npu.npu_quant_grouped_matmul_dequant",
                side_effect=[gate_up, down_out],
                create=True,
            ) as mock_quant_gmm,
            patch(
                "vllm_ascend._310p.quantization.methods.w8a8_dynamic.torch_npu.npu_swiglu",
                return_value=silu_out,
            ) as mock_swiglu,
            patch("vllm_ascend.lora.fused_moe.has_lora", return_value=True),
            patch(
                "vllm_ascend.lora.fused_moe._recover_moe_lora_routing_allgather",
                return_value=fake_routing,
            ) as mock_recover,
            patch("vllm_ascend.lora.fused_moe.moe_lora_apply_w13") as mock_w13,
            patch("vllm_ascend.lora.fused_moe.moe_lora_apply_w2") as mock_w2,
            patch("vllm_ascend.lora.quant_moe.quant_apply_mlp_with_moe_lora") as mock_quant_lora,
            patch("vllm_ascend.device.device_op.DeviceOperator.npu_dynamic_quant") as mock_dyn_quant,
            patch("torch_npu.npu_grouped_matmul", create=True) as mock_gmm,
            patch("vllm_ascend.device.device_op.DeviceOperator.npu_grouped_matmul_gmm2") as mock_gmm2,
        ):
            mock_ctx.moe_comm_method = comm
            result = comm.fused_experts(fused_experts_input=fused_experts_input, quant_method=method)

        self.assertIs(result.routed_out, combined_out)
        self.assertEqual(mock_quant_gmm.call_count, 2)
        mock_quant_gmm.assert_has_calls(
            [
                call(
                    x=sorted_hidden,
                    quantized_weight=layer.w13_weight,
                    weight_scale=layer.w13_weight_scale,
                    group_list=group_list,
                    quant_mode="pertoken",
                ),
                call(
                    x=silu_out,
                    quantized_weight=layer.w2_weight,
                    weight_scale=layer.w2_weight_scale,
                    group_list=group_list,
                    quant_mode="pertoken",
                ),
            ]
        )
        mock_swiglu.assert_called_once_with(gate_up)
        mock_recover.assert_called_once_with(lora_ctx, expanded_row_idx, topk_ids)
        mock_w13.assert_called_once_with(
            lora_ctx,
            gate_up_out=gate_up,
            hidden_states=sorted_hidden,
            lora_routing=fake_routing,
        )
        mock_w2.assert_called_once_with(
            lora_ctx,
            down_out=down_out,
            silu_out=silu_out,
            lora_routing=fake_routing,
        )
        mock_quant_lora.assert_not_called()
        mock_dyn_quant.assert_not_called()
        mock_gmm.assert_not_called()
        mock_gmm2.assert_not_called()
        comm.token_dispatcher.token_combine.assert_called_once_with(
            hidden_states=down_out,
            combine_metadata=combine_metadata,
        )
        self.assertIsNone(method._lora_routing)

    def test_w8a8_dynamic_310p_fused_experts_skips_lora_hooks_when_inactive(self):
        lora_ctx = SimpleNamespace(exchanged_lora_indices=torch.tensor([0, 0]))
        layer = _make_fake_layer(lora_ctx)
        layer.w13_weight = torch.randint(-8, 8, (2, 32, 16), dtype=torch.int8)
        layer.w2_weight = torch.randint(-8, 8, (2, 16, 16), dtype=torch.int8)
        layer.w13_weight_scale = torch.ones(2, 32, dtype=torch.float32)
        layer.w2_weight_scale = torch.ones(2, 16, dtype=torch.float32)

        method = AscendW8A8DynamicFusedMoEMethod310.__new__(AscendW8A8DynamicFusedMoEMethod310)
        method.in_dtype = torch.float16
        method.lora_context = None
        method._lora_routing = None

        comm = AllGatherCommImpl310.__new__(AllGatherCommImpl310)
        comm.moe_config = SimpleNamespace(
            activation=MoEActivation.SILU,
            swiglu_limit=0.0,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            activation_situ_beta=None,
            activation_situ_linear_beta=None,
        )
        comm.token_dispatcher = MagicMock()

        hidden = torch.randn(2, 16, dtype=torch.float16)
        topk_weights = torch.ones(2, 1, dtype=torch.float16)
        topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)
        group_list = torch.tensor([1, 2], dtype=torch.int64)
        gate_up = torch.randn(2, 32, dtype=torch.float16)
        silu_out = torch.randn(2, 16, dtype=torch.float16)
        down_out = torch.randn(2, 16, dtype=torch.float16)
        combine_metadata = SimpleNamespace(
            topk_weights=topk_weights,
            expanded_row_idx=torch.tensor([0, 1], dtype=torch.int32),
            restore_shape=hidden.shape,
        )
        comm.token_dispatcher.token_dispatch.return_value = SimpleNamespace(
            hidden_states=hidden,
            group_list=group_list,
            group_list_type=0,
            dynamic_scale=None,
            topk_scales=None,
            combine_metadata=combine_metadata,
        )
        comm.token_dispatcher.token_combine.return_value = down_out

        fused_experts_input = build_fused_experts_input(
            hidden_states=hidden,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            layer=layer,
            quant_type=QuantType.W8A8,
            dynamic_eplb=False,
            activation=MoEActivation.SILU,
            lora_context=lora_ctx,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_comm_method._EXTRA_CTX") as mock_ctx,
            patch("vllm_ascend.ops.fused_moe.moe_comm_method.maybe_record_event", return_value=None),
            patch("vllm_ascend.ops.fused_moe.moe_mlp.maybe_record_event", return_value=None),
            patch(
                "vllm_ascend.ops.fused_moe.dataclass.moe_mlp.enable_fusion_gmmswigluquant",
                return_value=True,
            ),
            patch(
                "vllm_ascend._310p.quantization.methods.w8a8_dynamic.torch_npu.npu_quant_grouped_matmul_dequant",
                side_effect=[gate_up, down_out],
                create=True,
            ) as mock_quant_gmm,
            patch(
                "vllm_ascend._310p.quantization.methods.w8a8_dynamic.torch_npu.npu_swiglu",
                return_value=silu_out,
            ),
            patch("vllm_ascend.lora.fused_moe.has_lora", return_value=False),
            patch("vllm_ascend.lora.fused_moe.moe_lora_apply_w13") as mock_w13,
            patch("vllm_ascend.lora.fused_moe.moe_lora_apply_w2") as mock_w2,
            patch("vllm_ascend.lora.quant_moe.quant_apply_mlp_with_moe_lora") as mock_quant_lora,
        ):
            mock_ctx.moe_comm_method = comm
            result = comm.fused_experts(fused_experts_input=fused_experts_input, quant_method=method)

        self.assertIs(result.routed_out, down_out)
        self.assertEqual(mock_quant_gmm.call_count, 2)
        mock_w13.assert_not_called()
        mock_w2.assert_not_called()
        mock_quant_lora.assert_not_called()
