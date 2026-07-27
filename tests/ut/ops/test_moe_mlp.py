import unittest
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ops.activation import SituActivationConfig
from vllm_ascend.ops.fused_moe import moe_mlp as moe_mlp_module
from vllm_ascend.ops.fused_moe.moe_mlp import (
    cumsum_group_list,
    quant_apply_mlp,
    unified_apply_mlp,
    unquant_apply_mlp,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEQuantParams,
    MoEWeights,
)
from vllm_ascend.ops.fused_moe.moe_stage_params import MoEMxfpParams
from vllm_ascend.quantization.quant_type import QuantType

MXFP4_TEST_DTYPE = getattr(torch, "float4_e2m1fn_x2", torch.float16)
MX_SCALE_TEST_DTYPE = getattr(torch, "float8_e8m0fnu", torch.float16)


class TestCumsumGroupList(unittest.TestCase):
    glist_dict: ClassVar[dict[int, torch.Tensor]]

    @classmethod
    def setUpClass(cls):
        cls.glist_dict = {
            0: torch.tensor([0, 2, 3, 3]),
            1: torch.tensor([0, 2, 1, 0]),
            2: torch.tensor([[1, 2], [2, 1], [0, 0], [0, 0]]),
        }

    support_combine = [(0, 0), (1, 0), (0, 1)]
    unsupported_combine = [(0, 2), (2, 1), (1, 2)]

    def test_cumsum_group_list_supported_conversion(self):
        for src_list_type, dst_list_type in self.support_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                result = cumsum_group_list(self.glist_dict[src_list_type], src_list_type, dst_list_type, expert_num=4)
                self.assertTrue(torch.equal(result, self.glist_dict[dst_list_type]))

    def test_cumsum_group_list_invalid_type_valueerror(self):
        with self.assertRaises(ValueError) as excinfo:
            cumsum_group_list(self.glist_dict[0], 4, 0)
        self.assertIn("group_list_type should be in [0, 1, 2], but received", str(excinfo.exception))

    def test_cumsum_group_list_unsupported_conversion_notimplementederror(self):
        for src_list_type, dst_list_type in self.unsupported_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                with self.assertRaises(NotImplementedError) as excinfo:
                    cumsum_group_list(self.glist_dict[0], src_list_type, dst_list_type)
                self.assertIn("This feature is under development.", str(excinfo.exception))


class TestW4A8RuntimeFlags(unittest.TestCase):
    def test_w4a8_per_channel_gmm_swiglu_flag(self):
        self.assertTrue(
            MoEQuantParams(quant_type=QuantType.W4A8, is_per_channel_weight=True).use_w4a8_per_channel_gmm_swiglu
        )
        self.assertFalse(
            MoEQuantParams(quant_type=QuantType.W4A8, is_per_channel_weight=False).use_w4a8_per_channel_gmm_swiglu
        )
        self.assertFalse(
            MoEQuantParams(quant_type=QuantType.W8A8, is_per_channel_weight=True).use_w4a8_per_channel_gmm_swiglu
        )


class TestUnifiedApplyMlpRequest(unittest.TestCase):
    def test_unquant_apply_mlp_wraps_tensor_weights_for_grouped_matmul(self):
        hidden_states = torch.randn(2, 8)
        gate_up_out = torch.randn(2, 16)
        expected = torch.randn(2, 8)
        w1 = torch.randn(2, 8, 16)
        w2 = torch.randn(2, 8, 8)

        with (
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_grouped_matmul",
                side_effect=[[gate_up_out], [expected]],
                create=True,
            ) as mock_grouped_matmul,
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_swiglu",
                return_value=gate_up_out,
                create=True,
            ),
        ):
            output, _ = unquant_apply_mlp(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                group_list=torch.tensor([1, 1]),
                need_trans=True,
            )

        self.assertTrue(output is expected)
        first_call, second_call = mock_grouped_matmul.call_args_list
        self.assertEqual(len(first_call.kwargs["weight"]), 1)
        self.assertEqual(len(second_call.kwargs["weight"]), 1)
        self.assertEqual(first_call.kwargs["weight"][0].shape, torch.Size([2, 16, 8]))
        self.assertEqual(second_call.kwargs["weight"][0].shape, torch.Size([2, 8, 8]))


    def test_w4a8_situ_preserves_packed_weight_scale_view_and_bias(self):
        hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
        quantized_input = torch.ones(2, 4, dtype=torch.int8)
        input_scale = torch.ones(2, 1, dtype=torch.float32)
        gate_up_out = torch.tensor(
            [[-8.0, 1.0, -30.0, 40.0], [0.5, 7.0, -2.0, 3.0]],
            dtype=torch.bfloat16,
        )
        quantized_situ = torch.full((2, 2), 3, dtype=torch.int8)
        situ_scale = torch.full((2, 1), 0.25, dtype=torch.float32)
        down_out = torch.randn(2, 4, dtype=torch.bfloat16)
        activation = SituActivationConfig(beta=4.0, linear_beta=25.0)
        w1_scale_bias = [torch.randn(1, 4)]
        w2_scale_bias = [torch.randn(1, 4)]
        event = object()
        custom_ops = SimpleNamespace(
            grouped_matmul_swiglu_quant_v2=MagicMock(),
            grouped_matmul_swiglu_quant_weight_nz_tensor_list=MagicMock(),
            npu_dequant_swiglu_quant=MagicMock(),
            dequant_situ_quant=MagicMock(return_value=(quantized_situ, situ_scale)),
        )
        packed_w1 = torch.ones(1, 4, 1, dtype=torch.int32)
        w1_scale = torch.ones(1, 4, dtype=torch.int64)

        with (
            patch.object(moe_mlp_module.torch.ops, "_C_ascend", custom_ops),
            patch.object(
                moe_mlp_module,
                "_EXTRA_CTX",
                SimpleNamespace(moe_comm_type=moe_mlp_module.MoECommType.MC2),
            ),
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.DeviceOperator.npu_dynamic_quant",
                return_value=(quantized_input, input_scale),
            ) as mock_input_quant,
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_grouped_matmul",
                return_value=[gate_up_out],
                create=True,
            ) as mock_gmm1,
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.DeviceOperator.npu_grouped_matmul_gmm2",
                return_value=down_out,
            ) as mock_gmm2,
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.DeviceOperator.npu_grouped_matmul_swiglu_quant",
                create=True,
            ) as mock_fused_gmm,
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp._custom_gmm_swiglu_enabled",
                return_value=True,
            ) as mock_custom_enabled,
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_swiglu",
                create=True,
            ) as mock_swiglu,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.dispose_tensor"),
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch.npu.current_stream",
                return_value=MagicMock(record_event=MagicMock(return_value=event)),
            ),
        ):
            output, before_gmm2_evt = quant_apply_mlp(
                hidden_states=hidden_states,
                w1=[packed_w1],
                w1_scale=[w1_scale],
                w2=[torch.ones(1, 2, 1, dtype=torch.int32)],
                w2_scale=[torch.ones(1, 4, dtype=torch.int64)],
                group_list=torch.tensor([2]),
                activation=activation,
                fusion=True,
                mxfp_quant_dtype=QuantType.W4A8,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                use_w4a8_per_channel_gmm_swiglu=True,
            )

        self.assertTrue(output is down_out)
        self.assertIs(before_gmm2_evt, event)
        mock_input_quant.assert_called_once()
        first_call = mock_gmm1.call_args
        self.assertIs(first_call.kwargs["bias"], w1_scale_bias)
        self.assertEqual(first_call.kwargs["output_dtype"], torch.bfloat16)
        self.assertIs(first_call.kwargs["weight"][0], packed_w1)
        self.assertEqual(first_call.kwargs["scale"][0].shape, torch.Size([1, 1, 4]))
        self.assertEqual(w1_scale.shape, torch.Size([1, 4]))
        situ_call = custom_ops.dequant_situ_quant.call_args.kwargs
        self.assertIs(situ_call["x"], gate_up_out)
        self.assertIsNone(situ_call["weight_scale"])
        self.assertIsNone(situ_call["activation_scale"])
        self.assertEqual(situ_call["beta"], 4.0)
        self.assertEqual(situ_call["linear_beta"], 25.0)
        self.assertIs(mock_gmm2.call_args.kwargs["bias"], w2_scale_bias)
        self.assertIs(mock_gmm2.call_args.kwargs["hidden_states"], quantized_situ)
        self.assertIs(mock_gmm2.call_args.kwargs["per_token_scale"], situ_scale)
        self.assertEqual(mock_gmm2.call_args.kwargs["mxfp_quant_dtype"], QuantType.W4A8)
        mock_custom_enabled.assert_not_called()
        mock_fused_gmm.assert_not_called()
        mock_swiglu.assert_not_called()
        custom_ops.grouped_matmul_swiglu_quant_v2.assert_not_called()
        custom_ops.grouped_matmul_swiglu_quant_weight_nz_tensor_list.assert_not_called()
        custom_ops.npu_dequant_swiglu_quant.assert_not_called()

    def test_w4a8_swiglu_stays_on_existing_fused_path(self):
        hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
        quantized_input = torch.ones(2, 4, dtype=torch.int8)
        input_scale = torch.ones(2, 1, dtype=torch.float32)
        quantized_activation = torch.full((2, 2), 3, dtype=torch.int8)
        activation_scale = torch.full((2, 1), 0.25, dtype=torch.float32)
        down_out = torch.randn(2, 4, dtype=torch.bfloat16)
        packed_w1 = torch.ones(1, 4, 1, dtype=torch.int32)
        w1_scale = torch.ones(1, 4, dtype=torch.int64)
        event = object()

        with (
            patch.object(
                moe_mlp_module,
                "_EXTRA_CTX",
                SimpleNamespace(moe_comm_type=moe_mlp_module.MoECommType.MC2),
            ),
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.DeviceOperator.npu_dynamic_quant",
                return_value=(quantized_input, input_scale),
            ),
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.DeviceOperator.npu_grouped_matmul_swiglu_quant",
                return_value=(quantized_activation, activation_scale, None),
            ) as mock_fused_gmm,
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.DeviceOperator.npu_grouped_matmul_gmm2",
                return_value=down_out,
            ),
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp._w4a8_situ_apply_mlp",
            ) as mock_situ,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.get_weight_prefetch_method", return_value=None),
            patch("vllm_ascend.ops.fused_moe.moe_mlp.dispose_tensor"),
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch.npu.current_stream",
                return_value=MagicMock(record_event=MagicMock(return_value=event)),
            ),
        ):
            output, before_gmm2_evt = quant_apply_mlp(
                hidden_states=hidden_states,
                w1=[packed_w1],
                w1_scale=[w1_scale],
                w2=[torch.ones(1, 2, 1, dtype=torch.int32)],
                w2_scale=[torch.ones(1, 4, dtype=torch.int64)],
                group_list=torch.tensor([2]),
                activation="silu",
                fusion=True,
                mxfp_quant_dtype=QuantType.W4A8,
                use_w4a8_per_channel_gmm_swiglu=True,
            )

        self.assertIs(output, down_out)
        self.assertIs(before_gmm2_evt, event)
        mock_situ.assert_not_called()
        self.assertIs(mock_fused_gmm.call_args.kwargs["weight"], packed_w1)
        self.assertIs(mock_fused_gmm.call_args.kwargs["weight_scale"], w1_scale)
        self.assertEqual(w1_scale.shape, torch.Size([1, 4]))

    def test_w4a8_mxfp_situ_uses_situ_mx_quant(self):
        hidden_states = torch.ones(2, 4, dtype=torch.float8_e4m3fn)
        input_scale = torch.ones(2, 1, 2, dtype=MX_SCALE_TEST_DTYPE)
        gate_up_out = torch.randn(2, 4, dtype=torch.bfloat16)
        quantized_situ = torch.ones(2, 2, dtype=torch.float8_e4m3fn)
        situ_scale = torch.ones(2, 1, 2, dtype=MX_SCALE_TEST_DTYPE)
        down_out = torch.randn(2, 4, dtype=torch.bfloat16)
        activation = SituActivationConfig(beta=4.0, linear_beta=25.0)
        event = object()
        custom_ops = SimpleNamespace(
            situ_mx_quant=MagicMock(return_value=(quantized_situ, situ_scale)),
        )

        with (
            patch.object(moe_mlp_module.torch.ops, "_C_ascend", custom_ops),
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.DeviceOperator.maybe_normalize_mxfp_scale_layout",
                return_value=input_scale,
            ),
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_grouped_matmul",
                return_value=[gate_up_out],
                create=True,
            ),
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.DeviceOperator.npu_grouped_matmul_gmm2",
                return_value=down_out,
            ) as mock_gmm2,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.dispose_tensor"),
            patch("vllm_ascend.ops.fused_moe.moe_mlp.get_weight_prefetch_method", return_value=None),
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch.npu.current_stream",
                return_value=MagicMock(record_event=MagicMock(return_value=event)),
            ),
        ):
            output, before_gmm2_evt = quant_apply_mlp(
                hidden_states=hidden_states,
                w1=[torch.ones(1, 4, 1, dtype=torch.int32)],
                w1_scale=[torch.ones(1, 1, 4)],
                w2=[torch.ones(1, 2, 1, dtype=torch.int32)],
                w2_scale=[torch.ones(1, 1, 4)],
                group_list=torch.tensor([2]),
                dynamic_scale=input_scale,
                activation=activation,
                use_mxfp_quant=True,
                mxfp_quant_dtype=QuantType.W4A8MXFP,
                act_quant_type=torch.float8_e4m3fn,
            )

        self.assertIs(output, down_out)
        self.assertIs(before_gmm2_evt, event)
        situ_call = custom_ops.situ_mx_quant.call_args.kwargs
        self.assertIs(situ_call["x"], gate_up_out)
        self.assertEqual(situ_call["beta"], 4.0)
        self.assertEqual(situ_call["linear_beta"], 25.0)
        self.assertEqual(situ_call["dst_type"], moe_mlp_module.SITU_MX_DST_TYPE_E4M3FN)
        self.assertIs(mock_gmm2.call_args.kwargs["hidden_states"], quantized_situ)
        self.assertIs(mock_gmm2.call_args.kwargs["per_token_scale"], situ_scale)

    def test_request_unquant_path(self):
        hidden_states = torch.randn(2, 8)
        expected = torch.randn(2, 8)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=torch.tensor([2, 2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(
                w1=torch.randn(1, 16, 8),
                w2=torch.randn(1, 8, 8),
                w1_bias=torch.randn(1, 16),
                w2_bias=torch.randn(1, 8),
            ),
            quant=MoEQuantParams(quant_type=QuantType.NONE),
            fusion=False,
            activation="silu",
            need_trans=False,
            dynamic_eplb=False,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp", return_value=expected) as mock_unquant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp") as mock_quant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output is expected)
        mock_unquant.assert_called_once()
        self.assertEqual(mock_unquant.call_args.kwargs["activation"], "silu")
        self.assertFalse(mock_unquant.call_args.kwargs["need_trans"])
        mock_quant.assert_not_called()

    def test_request_quant_path(self):
        for quant_type, mxfp_dtype in (
            (QuantType.MXFP8, torch.float8_e4m3fn),
            (QuantType.MXFP4, MXFP4_TEST_DTYPE),
        ):
            with self.subTest(quant_type=quant_type):
                hidden_states = torch.randn(2, 8)
                expected = torch.randn(2, 8)
                mlp_compute_input = MoEMlpComputeInput(
                    hidden_states=hidden_states,
                    group_list=torch.tensor([2, 2], dtype=torch.int64),
                    group_list_type=1,
                    dynamic_scale=torch.randn(2, 1),
                    topk_scales=None,
                    weights=MoEWeights(
                        w1=torch.randn(1, 16, 8),
                        w2=torch.randn(1, 8, 8),
                        w1_scale=[torch.randn(1)],
                        w2_scale=[torch.randn(1)],
                    ),
                    quant=MoEQuantParams(
                        quant_type=quant_type,
                        mxfp=MoEMxfpParams(
                            act_quant_type=mxfp_dtype,
                            weight_quant_type=mxfp_dtype,
                            use_bf16=False,
                        ),
                    ),
                    fusion=True,
                    activation="silu",
                    need_trans=False,
                    dynamic_eplb=True,
                )

                with (
                    patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp", return_value=expected) as mock_quant,
                    patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp") as mock_unquant,
                ):
                    output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

                self.assertTrue(output is expected)
                mock_quant.assert_called_once()
                quant_kwargs = mock_quant.call_args.kwargs
                self.assertTrue(quant_kwargs["use_mxfp_quant"])
                self.assertTrue(quant_kwargs["fusion"])
                self.assertTrue(quant_kwargs["dynamic_eplb"])
                self.assertEqual(quant_kwargs["act_quant_type"], mxfp_dtype)
                self.assertEqual(quant_kwargs["weight_quant_type"], mxfp_dtype)
                self.assertFalse(quant_kwargs["use_bf16"])
                mock_unquant.assert_not_called()

    def test_request_quant_path_passes_w4a8_per_channel_flag(self):
        hidden_states = torch.randn(2, 8)
        expected = torch.randn(2, 8)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=torch.tensor([2, 2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=torch.randn(2, 1),
            topk_scales=None,
            weights=MoEWeights(
                w1=torch.randn(1, 16, 8),
                w2=torch.randn(1, 8, 8),
                w1_scale=[torch.randn(1, 16)],
                w2_scale=[torch.randn(1, 8)],
            ),
            quant=MoEQuantParams(quant_type=QuantType.W4A8, is_per_channel_weight=True),
            fusion=False,
            activation="silu",
            need_trans=False,
            dynamic_eplb=False,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp", return_value=expected) as mock_quant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp") as mock_unquant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output is expected)
        quant_kwargs = mock_quant.call_args.kwargs
        self.assertTrue(quant_kwargs["use_w4a8_per_channel_gmm_swiglu"])
        mock_unquant.assert_not_called()

    def test_request_quant_path_passes_swiglustep_activation(self):
        expected = torch.randn(1, 2)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=torch.ones((1, 2), dtype=torch.float32),
            group_list=torch.tensor([1], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(
                w1=[torch.ones((1, 2, 4), dtype=torch.float32)],
                w2=[torch.ones((1, 2, 2), dtype=torch.float32)],
                w1_scale=[torch.ones((1,), dtype=torch.float32)],
                w2_scale=[torch.ones((1,), dtype=torch.float32)],
            ),
            quant=MoEQuantParams(quant_type=QuantType.W8A8),
            fusion=True,
            activation=MoEActivation.SWIGLUSTEP,
            swiglu_limit=5.0,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp", return_value=expected) as mock_quant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp") as mock_unquant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output is expected)
        quant_kwargs = mock_quant.call_args.kwargs
        self.assertEqual(quant_kwargs["activation"], MoEActivation.SWIGLUSTEP)
        self.assertEqual(quant_kwargs["swiglu_limit"], 5.0)
        mock_unquant.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
