from unittest.mock import MagicMock, Mock, patch

import torch

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import (
    create_mock_ascend_config,
    create_mock_vllm_config,
)
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.quantization.methods.w8a8.w8a8fp8_dynamic import (
    AscendW8A8FP8DynamicFusedMoEMethod,
    AscendW8A8FP8DynamicLinearMethod,
)


class TestAscendW8A8FP8DynamicLinearMethod(TestBase):
    def setUp(self):
        self.method = AscendW8A8FP8DynamicLinearMethod()

    def test_act_quant_type(self):
        self.assertEqual(self.method.act_quant_type, torch.float8_e4m3fn)

    def test_get_weight_various_sizes(self):
        sizes = [(64, 128), (256, 512), (1024, 2048)]
        for input_size, output_size in sizes:
            weight = self.method.get_weight(input_size, output_size, torch.bfloat16)
            self.assertEqual(weight["weight"].dtype, torch.float8_e4m3fn)
            self.assertEqual(weight["weight"].shape, (output_size, input_size))

    def test_get_perchannel_param_dtype_variations(self):
        dtypes = [torch.bfloat16, torch.float16]
        for dtype in dtypes:
            params = self.method.get_perchannel_param(128, dtype)
            self.assertEqual(params["weight_scale"].dtype, torch.float32)
            self.assertEqual(params["weight_offset"].dtype, dtype)
            self.assertEqual(params["weight_scale"].shape, (128, 1))
            self.assertEqual(params["weight_offset"].shape, (128, 1))


class TestAscendW8A8FP8FusedMoEMethod(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 128

    @patch("torch.distributed.get_rank")
    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_dynamic.get_mc2_group")
    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_dynamic.get_ascend_config")
    def setUp(self, mock_ascend, mock_mc2, mock_rank):
        with patch("vllm_ascend.quantization.methods.w8a8.w8a8_dynamic.get_current_vllm_config") as mock_vllm:
            mock_vllm.return_value = create_mock_vllm_config()
            mock_ascend.return_value = create_mock_ascend_config()
            mock_mc2.return_value = MagicMock(
                device_group=Mock(
                    _get_backend=Mock(return_value=Mock(get_hccl_comm_name=Mock(return_value="test_comm")))
                )
            )
            mock_rank.return_value = 0
            self.quant_method = AscendW8A8FP8DynamicFusedMoEMethod()

    def test_quant_type_is_w8a8fp8(self):
        from vllm_ascend.quantization.quant_type import QuantType

        self.assertEqual(self.quant_method.quant_type, QuantType.W8A8FP)

    def test_get_weight_dtype_is_float8_e4m3fn(self):
        param_dict = self.quant_method.get_weight(
            self.num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16
        )
        self.assertEqual(param_dict["w13_weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(param_dict["w2_weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(
            param_dict["w13_weight"].shape, (self.num_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.assertEqual(param_dict["w2_weight"].shape, (self.num_experts, self.hidden_size, self.intermediate_size))

    def test_get_weight_various_expert_counts(self):
        expert_counts = [4, 8, 16, 32]
        for num_experts in expert_counts:
            param_dict = self.quant_method.get_weight(
                num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16
            )
            self.assertEqual(param_dict["w13_weight"].shape[0], num_experts)
            self.assertEqual(param_dict["w2_weight"].shape[0], num_experts)

    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_dynamic._EXTRA_CTX")
    def test_apply_uses_explicit_dispatch_and_mlp_args(self, mock_extra_ctx):
        tokens = 4
        hidden_size = self.hidden_size
        layer = torch.nn.Module()
        layer.w13_weight = torch.randn(
            self.num_experts, 2 * self.intermediate_size, hidden_size, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        layer.w2_weight = torch.randn(self.num_experts, hidden_size, self.intermediate_size, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        layer.w13_weight_scale_fp32 = torch.ones(self.num_experts, 2 * self.intermediate_size, dtype=torch.float32)
        layer.w2_weight_scale = torch.ones(self.num_experts, hidden_size, dtype=torch.float32)
        layer.swiglu_limit = 1000000

        x = torch.randn(tokens, hidden_size, dtype=torch.float32)
        topk_weights = torch.randn(tokens, 2, dtype=torch.float32)
        topk_ids = torch.randint(0, self.num_experts, (tokens, 2), dtype=torch.int64)
        mc2_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        pertoken_scale = torch.randn(tokens, dtype=torch.float32)
        layer.activation = "gelu"
        layer.apply_router_weight_on_input = True
        layer.ascend_expert_map = None
        layer.global_redundant_expert_num = 0
        layer.log2phy = None
        layer.ascend_mc2_mask = mc2_mask
        layer.ascend_pertoken_scale = pertoken_scale
        layer.swiglu_alpha = 1.0
        layer.swiglu_beta = 0.0

        mock_comm = Mock()
        mock_comm.fused_experts.return_value = torch.randn(tokens, hidden_size, dtype=torch.float32)
        mock_extra_ctx.moe_comm_method = mock_comm
        mock_extra_ctx.moe_comm_type = MoECommType.ALLGATHER
        self.quant_method.in_dtype = torch.float32

        self.quant_method.apply(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )

        fused_experts_input = mock_comm.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertEqual(fused_experts_input.activation, "gelu")
        self.assertTrue(fused_experts_input.routing.apply_router_weight_on_input)
        self.assertIs(fused_experts_input.routing.mc2_mask, mc2_mask)
        self.assertIs(fused_experts_input.routing.pertoken_scale, pertoken_scale)
        self.assertIs(fused_experts_input.topk_weights, topk_weights)
        self.assertIs(fused_experts_input.topk_ids, topk_ids)

    def test_apply_gmm1_act_quant_with_eplb_uses_fp8_v2_kernel_not_int8_nz(self):
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput

        layer = torch.nn.Module()
        layer.w13_weight_list = [
            torch.randn(2 * self.intermediate_size, self.hidden_size, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            for _ in range(self.num_experts)
        ]
        layer.w13_weight_scale_fp32_list = [
            torch.ones(2 * self.intermediate_size, dtype=torch.float32) for _ in range(self.num_experts)
        ]
        layer.w2_weight_list = [
            torch.randn(self.hidden_size, self.intermediate_size, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            for _ in range(self.num_experts)
        ]
        layer.w2_weight_scale_list = [torch.ones(self.hidden_size, dtype=torch.float32) for _ in range(self.num_experts)]

        self.quant_method.use_expert_weight_list = True

        hidden_states = torch.randn(4, self.hidden_size, dtype=torch.bfloat16)
        quant_hidden = hidden_states.to(torch.float8_e4m3fn)
        pertoken_scale = torch.ones(4, dtype=torch.float32)
        group_list = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.int64)
        mlp_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=group_list,
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MagicMock(),
            quant=MagicMock(),
            fusion=True,
            layer=layer,
            activation=MoEActivation.SILU,
            dynamic_eplb=True,
        )

        expected_out = torch.zeros(4, self.intermediate_size, dtype=torch.float8_e4m3fn)
        expected_scale = torch.ones(4, dtype=torch.float32)
        nz_kernel_mock = MagicMock()

        with (
            patch("vllm_ascend.ops.fused_moe.moe_utils.enable_custom_op", return_value=True),
            patch.object(
                torch.ops._C_ascend,
                "grouped_matmul_swiglu_quant_weight_nz_tensor_list",
                nz_kernel_mock,
                create=True,
            ),
            patch.object(self.quant_method, "_quant_hidden_states", return_value=(quant_hidden, pertoken_scale)),
            patch(
                "vllm_ascend.quantization.methods.w8a8.w8a8fp8_dynamic.torch_npu.npu_grouped_matmul_swiglu_quant_v2",
                return_value=(expected_out, expected_scale),
            ) as mock_v2,
            patch("vllm_ascend.quantization.methods.w8a8.w8a8fp8_dynamic.dispose_tensor"),
        ):
            out, out_scale = self.quant_method.apply_gmm1_act_quant(mlp_input)

        nz_kernel_mock.assert_not_called()
        mock_v2.assert_called_once()
        v2_kwargs = mock_v2.call_args.kwargs
        self.assertIs(v2_kwargs["x"], quant_hidden)
        self.assertIs(v2_kwargs["weight"], layer.w13_weight_list)
        self.assertIs(v2_kwargs["weight_scale"], layer.w13_weight_scale_fp32_list)
        self.assertIs(v2_kwargs["x_scale"], pertoken_scale)
        self.assertEqual(v2_kwargs["quant_dtype"], torch.float8_e4m3fn)
        self.assertEqual(v2_kwargs["dequant_dtype"], torch.float32)
        self.assertIs(out, expected_out)
        self.assertIs(out_scale, expected_scale)
