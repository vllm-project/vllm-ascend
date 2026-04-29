from unittest.mock import MagicMock, Mock, patch

import torch

from tests.ut.base import TestBase
from tests.ut.conftest import npu_test
from tests.ut.quantization.conftest_quantization import (
    create_linear_layer,
    create_mock_ascend_config,
    create_mock_vllm_config,
    create_moe_layer,
)
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
    AscendW8A8DynamicLinearMethod,
)


class TestAscendW8A8DynamicLinearMethod(TestBase):
    def setUp(self):
        self.method = AscendW8A8DynamicLinearMethod()

    def test_get_weight_various_sizes(self):
        sizes = [(64, 128), (256, 512), (1024, 2048)]
        for input_size, output_size in sizes:
            weight = self.method.get_weight(input_size, output_size, torch.bfloat16)
            self.assertEqual(weight["weight"].dtype, torch.int8)
            self.assertEqual(weight["weight"].shape, (output_size, input_size))

    def test_get_perchannel_param_dtype_variations(self):
        dtypes = [torch.bfloat16, torch.float16]
        for dtype in dtypes:
            params = self.method.get_perchannel_param(128, dtype)
            self.assertEqual(params["weight_scale"].dtype, dtype)
            self.assertEqual(params["weight_offset"].dtype, dtype)
            self.assertEqual(params["weight_scale"].shape, (128, 1))
            self.assertEqual(params["weight_offset"].shape, (128, 1))

    @patch("torch_npu.npu_quant_matmul")
    @patch("torch_npu.npu_dynamic_quant")
    def test_apply_3d_input_with_squeeze(self, mock_dyn_quant, mock_matmul):
        mock_dyn_quant.return_value = (
            torch.randint(-128, 127, (32, 1, 128), dtype=torch.int8),
            torch.randn(32, 1, dtype=torch.float32),
        )
        mock_matmul.return_value = torch.randn(32, 1, 256)
        layer = MagicMock()
        layer.weight = torch.randint(-128, 127, (128, 256), dtype=torch.int8)
        layer.weight_scale = torch.randn(256, dtype=torch.float32)
        x = torch.randn(32, 1, 128, dtype=torch.bfloat16)
        output = self.method.apply(layer, x)
        mock_dyn_quant.assert_called_once()
        mock_matmul.assert_called_once()
        self.assertEqual(output.shape, (32, 1, 1, 256))

    def test_process_weights_after_loading(self):
        layer = MagicMock()
        layer.weight.data = torch.randint(-128, 127, (128, 256), dtype=torch.int8)
        layer.weight_scale.data = torch.randn(256, 1, dtype=torch.bfloat16)
        layer.weight_offset.data = torch.randn(256, 1, dtype=torch.bfloat16)
        with patch("vllm_ascend.quantization.methods.w8a8_dynamic.maybe_trans_nz", side_effect=lambda x: x):
            self.method.process_weights_after_loading(layer)
        self.assertEqual(layer.weight_scale_fp32.dtype, torch.float32)
        self.assertEqual(layer.weight_scale.data.shape, (256,))
        self.assertEqual(layer.weight_offset.data.shape, (256,))
        self.assertEqual(layer.weight.data.shape, (256, 128))

    @npu_test(num_npus=1, npu_type="a3")
    def test_apply_with_npu(self):
        input_size, output_size = 128, 256
        params_dtype = torch.bfloat16
        layer = torch.nn.Module()
        layer.weight = create_linear_layer(self.method, input_size, output_size, params_dtype)
        self.method.process_weights_after_loading(layer)

        x = torch.randn(32, input_size, dtype=params_dtype).npu()
        bias = torch.randn(output_size, dtype=torch.float32).npu()

        output = self.method.apply(layer, x, bias)
        self.assertEqual(output.shape, (32, output_size))


class TestAscendW8A8FusedMoEMethod(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 128

    @patch("torch.distributed.get_rank")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_mc2_group")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_ascend_config")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_ep_group")
    def setUp(self, mock_ep, mock_ascend, mock_mc2, mock_rank):
        with patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_current_vllm_config") as mock_vllm:
            mock_vllm.return_value = create_mock_vllm_config()
            mock_ep.return_value = Mock()
            mock_ascend.return_value = create_mock_ascend_config()
            mock_mc2.return_value = MagicMock(
                device_group=Mock(
                    _get_backend=Mock(return_value=Mock(get_hccl_comm_name=Mock(return_value="test_comm")))
                )
            )
            mock_rank.return_value = 0
            self.quant_method = AscendW8A8DynamicFusedMoEMethod()

    def test_get_weight_various_expert_counts(self):
        expert_counts = [4, 8, 16, 32]
        for num_experts in expert_counts:
            param_dict = self.quant_method.get_weight(
                num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16
            )
            self.assertEqual(param_dict["w13_weight"].shape[0], num_experts)
            self.assertEqual(param_dict["w2_weight"].shape[0], num_experts)

    def test_get_dynamic_quant_param_various_sizes(self):
        param_dict = self.quant_method.get_dynamic_quant_param(
            self.num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16
        )
        self.assertEqual(param_dict["w13_weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(param_dict["w13_weight_offset"].shape, (self.num_experts, 2 * self.intermediate_size, 1))
        self.assertEqual(param_dict["w2_weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(param_dict["w2_weight_offset"].shape, (self.num_experts, self.hidden_size, 1))

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

    @patch("torch_npu.npu_format_cast")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.envs_ascend")
    def test_process_weights_after_loading(self, mock_envs, mock_format_cast):
        mock_envs.VLLM_ASCEND_ENABLE_FUSED_MC2 = 1
        self.quant_method.dynamic_eplb = True
        mock_format_cast.return_value = torch.randint(
            -8, 8, (self.num_experts, self.hidden_size, 2 * self.intermediate_size), dtype=torch.int8
        )
        layer = create_moe_layer(
            num_experts=self.num_experts, hidden_size=self.hidden_size, intermediate_size=self.intermediate_size
        )
        self.quant_method.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "w13_weight_list"))
        self.assertFalse(hasattr(layer, "w13_weight_scale_fp32"))
