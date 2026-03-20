from unittest.mock import Mock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod


class TestAscendW8A8FusedMoEMethod(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 128

    @patch("torch.distributed.get_rank")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_mc2_group")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_ascend_config")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_ep_group")
    def setUp(self, mock_get_ep_group, mock_get_ascend_config,
              mock_get_mc2_group, mock_get_rank):
        with patch(
                'vllm_ascend.quantization.methods.w8a8_dynamic.get_current_vllm_config'
        ) as mock_get_current_vllm_config:
            mock_vllm_config = Mock()
            mock_vllm_config.quant_config = Mock(
                quant_description={"group_size": 256})
            mock_vllm_config.scheduler_config = Mock(
                max_num_batched_tokens=2048,
                max_model_len=2048,
                enable_chunked_prefill=False)
            mock_get_current_vllm_config.return_value = mock_vllm_config
            mock_ep_group = Mock()
            mock_get_ep_group.return_value = mock_ep_group
            mock_ascend_config = Mock()
            mock_ascend_config.enable_chunked_prefill = False
            mock_ascend_config.multistream_overlap_gate = False
            mock_ascend_config.eplb_config = Mock(dynamic_eplb=False)
            mock_get_ascend_config.return_value = mock_ascend_config
            mock_mc2_group = Mock(device_group=0)
            mock_get_mc2_group.return_value = mock_mc2_group
            mock_rank = Mock()
            mock_get_rank.return_value = mock_rank

            self.quant_method = AscendW8A8DynamicFusedMoEMethod()

    def test_get_weight(self):
        param_dict = self.quant_method.get_weight(self.num_experts,
                                                  self.intermediate_size,
                                                  self.hidden_size,
                                                  torch.bfloat16)
        self.assertEqual(param_dict["w13_weight"].dtype, torch.int8)
        self.assertEqual(
            param_dict["w13_weight"].shape,
            (self.num_experts, 2 * self.intermediate_size, self.hidden_size))

    def test_get_dynamic_quant_param(self):
        param_dict = self.quant_method.get_dynamic_quant_param(
            self.num_experts, self.intermediate_size, self.hidden_size,
            torch.bfloat16)
        self.assertEqual(param_dict["w13_weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(param_dict["w13_weight_scale"].shape,
                         (self.num_experts, 2 * self.intermediate_size, 1))

    def build_layer(self):
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(torch.empty(
            self.num_experts,
            2 * self.intermediate_size,
            self.hidden_size,
            dtype=torch.int8),
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.empty(
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
            dtype=torch.int8),
                                             requires_grad=False)
        w13_weight_scale = torch.zeros(
            (self.num_experts, 2 * self.intermediate_size, 1),
            dtype=torch.float32)
        layer.w13_weight_scale = torch.nn.Parameter(w13_weight_scale,
                                                    requires_grad=False)
        w13_weight_offset = torch.zeros(
            (self.num_experts, 2 * self.intermediate_size, 1),
            dtype=torch.float32)
        layer.w13_weight_offset = torch.nn.Parameter(w13_weight_offset,
                                                     requires_grad=False)
        w2_weight_scale = torch.zeros((self.num_experts, self.hidden_size, 1),
                                      dtype=torch.float32)
        layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale,
                                                   requires_grad=False)
        w2_weight_offset = torch.zeros((self.num_experts, self.hidden_size, 1),
                                       dtype=torch.float32)
        layer.w2_weight_offset = torch.nn.Parameter(w2_weight_offset,
                                                    requires_grad=False)
        return layer

    @patch('torch_npu.npu_format_cast')
    def test_process_weights_after_loading(self, mock_npu_format_cast):

        def func_by_args(weight, num_format):
            return weight

        mock_npu_format_cast.side_effect = func_by_args
        new_layer = self.build_layer()
        self.quant_method.process_weights_after_loading(new_layer)
        mock_npu_format_cast.assert_called()

    @patch("vllm_ascend.quantization.methods.w8a8_dynamic._EXTRA_CTX")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.select_experts")
    def test_apply_uses_explicit_dispatch_and_mlp_args(self, mock_select_experts, mock_extra_ctx):
        tokens = 4
        hidden_size = self.hidden_size
        layer = torch.nn.Module()
        layer.w13_weight = torch.randint(
            -8,
            8,
            (self.num_experts, 2 * self.intermediate_size, hidden_size),
            dtype=torch.int8,
        )
        layer.w2_weight = torch.randint(
            -8,
            8,
            (self.num_experts, hidden_size, self.intermediate_size),
            dtype=torch.int8,
        )
        layer.w13_weight_scale_fp32 = torch.ones(self.num_experts, 2 * self.intermediate_size, dtype=torch.float32)
        layer.w2_weight_scale = torch.ones(self.num_experts, hidden_size, dtype=torch.float32)

        x = torch.randn(tokens, hidden_size, dtype=torch.float32)
        router_logits = torch.randn(tokens, self.num_experts, dtype=torch.float32)
        topk_weights = torch.randn(tokens, 2, dtype=torch.float32)
        topk_ids = torch.randint(0, self.num_experts, (tokens, 2), dtype=torch.int64)
        mc2_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        pertoken_scale = torch.randn(tokens, dtype=torch.float32)

        mock_select_experts.return_value = (topk_weights, topk_ids)
        mock_comm = Mock()
        mock_comm.fused_experts.return_value = torch.randn(tokens, hidden_size, dtype=torch.float32)
        mock_extra_ctx.moe_comm_method = mock_comm
        mock_extra_ctx.moe_comm_type = MoECommType.ALLGATHER
        self.quant_method.multistream_overlap_gate = False
        self.quant_method.in_dtype = torch.float32

        self.quant_method.apply(
            layer=layer,
            x=x,
            router_logits=router_logits,
            top_k=2,
            renormalize=True,
            global_num_experts=self.num_experts,
            activation="gelu",
            apply_router_weight_on_input=True,
            mc2_mask=mc2_mask,
            pertoken_scale=pertoken_scale,
        )

        fused_experts_input = mock_comm.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertEqual(fused_experts_input.activation, "gelu")
        self.assertTrue(fused_experts_input.routing.apply_router_weight_on_input)
        self.assertIs(fused_experts_input.routing.mc2_mask, mc2_mask)
        self.assertIs(fused_experts_input.routing.pertoken_scale, pertoken_scale)
        self.assertIs(fused_experts_input.topk_weights, topk_weights)
        self.assertIs(fused_experts_input.topk_ids, topk_ids)

    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_flash_common3_context")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic._EXTRA_CTX")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.select_experts")
    def test_apply_overlap_gate_uses_fc3_context(
        self,
        mock_select_experts,
        mock_extra_ctx,
        mock_get_flash_common3_context,
    ):
        tokens = 4
        hidden_size = self.hidden_size
        layer = torch.nn.Module()
        layer.w13_weight = torch.randint(
            -8,
            8,
            (self.num_experts, 2 * self.intermediate_size, hidden_size),
            dtype=torch.int8,
        )
        layer.w2_weight = torch.randint(
            -8,
            8,
            (self.num_experts, hidden_size, self.intermediate_size),
            dtype=torch.int8,
        )
        layer.w13_weight_scale_fp32 = torch.ones(self.num_experts, 2 * self.intermediate_size, dtype=torch.float32)
        layer.w2_weight_scale = torch.ones(self.num_experts, hidden_size, dtype=torch.float32)

        x = torch.randn(tokens, hidden_size, dtype=torch.float32)
        router_logits = torch.randn(tokens, self.num_experts, dtype=torch.float32)
        topk_weights = torch.randn(tokens, 2, dtype=torch.float32)
        topk_ids = torch.randint(0, self.num_experts, (tokens, 2), dtype=torch.int64)
        mc2_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        pertoken_scale = torch.randn(tokens, dtype=torch.float32)

        self.quant_method.multistream_overlap_gate = True
        self.quant_method.in_dtype = torch.float32
        mock_get_flash_common3_context.return_value = Mock(topk_weights=topk_weights, topk_ids=topk_ids)

        mock_comm = Mock()
        mock_comm.fused_experts.return_value = torch.randn(tokens, hidden_size, dtype=torch.float32)
        mock_extra_ctx.moe_comm_method = mock_comm
        mock_extra_ctx.moe_comm_type = MoECommType.ALLGATHER

        self.quant_method.apply(
            layer=layer,
            x=x,
            router_logits=router_logits,
            top_k=2,
            renormalize=True,
            global_num_experts=self.num_experts,
            activation="gelu",
            apply_router_weight_on_input=True,
            mc2_mask=mc2_mask,
            pertoken_scale=pertoken_scale,
        )

        mock_select_experts.assert_not_called()
        fused_experts_input = mock_comm.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertEqual(fused_experts_input.activation, "gelu")
        self.assertTrue(fused_experts_input.routing.apply_router_weight_on_input)
        self.assertIs(fused_experts_input.routing.mc2_mask, mc2_mask)
        self.assertIs(fused_experts_input.routing.pertoken_scale, pertoken_scale)
        self.assertIs(fused_experts_input.topk_weights, topk_weights)
        self.assertIs(fused_experts_input.topk_ids, topk_ids)
