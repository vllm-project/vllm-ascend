import unittest
from typing import ClassVar
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.moe_mlp import cumsum_group_list, quant_apply_mlp, unified_apply_mlp
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEQuantParams,
    MoEWeights,
)
from vllm_ascend.ops.fused_moe.moe_stage_params import MoEMxfpParams
from vllm_ascend.quantization.quant_type import QuantType

MXFP4_TEST_DTYPE = getattr(torch, "float4_e2m1fn_x2", torch.float16)


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


class TestQuantApplyMlpW4A8PerChannel(unittest.TestCase):
    def test_effective_swiglu_limit_keeps_fused_op_path(self):
        hidden_states = torch.arange(12, dtype=torch.int8).view(3, 4)
        dynamic_scale = torch.ones(3, dtype=torch.float32)
        group_list = torch.tensor([2, 1], dtype=torch.int64)
        w1 = [torch.ones(1, 8, 4, dtype=torch.int32)]
        w2 = [torch.ones(1, 4, 4, dtype=torch.int32)]
        w1_scale = [torch.ones(1, 8, dtype=torch.float32)]
        w2_scale = [torch.ones(1, 4, dtype=torch.float32)]
        w1_scale_bias = [torch.ones(1, 8, dtype=torch.float32)]
        w2_scale_bias = [torch.ones(1, 4, dtype=torch.float32)]
        quantized = torch.ones(3, 2, dtype=torch.int8)
        swiglu_out_scale = torch.ones(3, dtype=torch.float32)
        expected = torch.randn(3, 4)
        event = object()
        stream = MagicMock()
        stream.record_event.return_value = event
        extra_ctx = MagicMock()
        extra_ctx.moe_comm_type = MoECommType.ALLGATHER

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp._EXTRA_CTX", extra_ctx),
            patch("vllm_ascend.ops.fused_moe.moe_mlp.enable_custom_op", return_value=True),
            patch("vllm_ascend.ops.fused_moe.moe_mlp.torch.npu.current_stream", return_value=stream),
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_grouped_matmul",
            ) as mock_gmm,
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_dynamic_quant",
                return_value=(quantized, swiglu_out_scale),
            ) as mock_dynamic_quant,
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2",
                return_value=(quantized, swiglu_out_scale),
                create=True,
            ) as mock_gmm_swiglu_v2,
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.DeviceOperator.npu_grouped_matmul_gmm2",
                return_value=expected,
            ),
        ):
            output, before_gmm2_evt = quant_apply_mlp(
                hidden_states=hidden_states,
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                group_list=group_list,
                group_list_type=1,
                dynamic_scale=dynamic_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                use_w4a8_per_channel_gmm_swiglu=True,
                swiglu_limit=10.0,
            )

        mock_gmm_swiglu_v2.assert_called_once()
        self.assertEqual(mock_gmm_swiglu_v2.call_args.kwargs["swiglu_limit"], 10.0)
        mock_dynamic_quant.assert_not_called()
        mock_gmm.assert_not_called()
        self.assertIs(output, expected)
        self.assertIs(before_gmm2_evt, event)

    def test_unlimited_swiglu_limit_keeps_fused_op_path(self):
        hidden_states = torch.arange(12, dtype=torch.int8).view(3, 4)
        dynamic_scale = torch.ones(3, dtype=torch.float32)
        group_list = torch.tensor([2, 1], dtype=torch.int64)
        w1 = [torch.ones(1, 8, 4, dtype=torch.int32)]
        w2 = [torch.ones(1, 4, 4, dtype=torch.int32)]
        w1_scale = [torch.ones(1, 8, dtype=torch.float32)]
        w2_scale = [torch.ones(1, 4, dtype=torch.float32)]
        swiglu_out = torch.ones(3, 2, dtype=torch.int8)
        swiglu_out_scale = torch.ones(3, dtype=torch.float32)
        expected = torch.randn(3, 4)
        event = object()
        stream = MagicMock()
        stream.record_event.return_value = event
        extra_ctx = MagicMock()
        extra_ctx.moe_comm_type = MoECommType.ALLGATHER

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp._EXTRA_CTX", extra_ctx),
            patch("vllm_ascend.ops.fused_moe.moe_mlp.enable_custom_op", return_value=True),
            patch("vllm_ascend.ops.fused_moe.moe_mlp.torch.npu.current_stream", return_value=stream),
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2",
                return_value=(swiglu_out, swiglu_out_scale),
                create=True,
            ) as mock_gmm_swiglu_v2,
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_grouped_matmul",
            ) as mock_gmm,
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.DeviceOperator.npu_grouped_matmul_gmm2",
                return_value=expected,
            ),
        ):
            output, before_gmm2_evt = quant_apply_mlp(
                hidden_states=hidden_states,
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                group_list=group_list,
                group_list_type=1,
                dynamic_scale=dynamic_scale,
                use_w4a8_per_channel_gmm_swiglu=True,
                swiglu_limit=1_000_000,
            )

        mock_gmm_swiglu_v2.assert_called_once()
        mock_gmm.assert_not_called()
        self.assertIs(output, expected)
        self.assertIs(before_gmm2_evt, event)


class TestUnifiedApplyMlpRequest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
