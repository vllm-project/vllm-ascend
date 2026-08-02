import unittest
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe.moe_mlp import (
    cumsum_group_list,
    unified_apply_mlp,
    unquant_apply_mlp,
    w8a8_dynamic_lora_apply_mlp,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEQuantParams,
    MoEWeights,
)
from vllm_ascend.ops.fused_moe.moe_stage_params import MoEMxfpParams
from vllm_ascend.quantization.quant_type import QuantType

MXFP4_TEST_DTYPE = getattr(torch, "float4_e2m1fn_x2", torch.float16)
MOE_MLP = "vllm_ascend.ops.fused_moe.moe_mlp"


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


class TestW8A8DynamicLoraApplyMlp(unittest.TestCase):
    @staticmethod
    def _compute_input(
        *,
        quant_type=QuantType.W8A8,
        no_lora=False,
        dynamic_scale=None,
    ):
        return MoEMlpComputeInput(
            hidden_states=torch.randn(2, 8, dtype=torch.bfloat16),
            group_list=torch.tensor([2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=dynamic_scale,
            topk_scales=None,
            weights=MoEWeights(
                w1=[torch.ones(1, 8, 16, dtype=torch.int8)],
                w2=[torch.ones(1, 8, 8, dtype=torch.int8)],
                w1_scale=[torch.ones(1, 16)],
                w2_scale=[torch.ones(1, 8)],
            ),
            quant=MoEQuantParams(quant_type=quant_type),
            fusion=True,
            activation="silu",
            need_trans=False,
            dynamic_eplb=False,
            expanded_row_idx=torch.tensor([0, 1], dtype=torch.int32),
            topk_ids=torch.tensor([[0], [0]], dtype=torch.int32),
            lora_context=SimpleNamespace(
                punica_wrapper=SimpleNamespace(no_lora=no_lora),
            ),
        )

    def test_active_w8a8_lora_uses_dedicated_path(self):
        expected = torch.randn(2, 8)
        mlp_compute_input = self._compute_input()

        with (
            patch(
                f"{MOE_MLP}.w8a8_dynamic_lora_apply_mlp",
                return_value=expected,
            ) as mock_lora,
            patch(f"{MOE_MLP}.quant_apply_mlp") as mock_quant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertIs(output, expected)
        mock_lora.assert_called_once()
        mock_quant.assert_not_called()

    def test_base_only_w8a8_keeps_existing_path(self):
        expected = torch.randn(2, 8)
        mlp_compute_input = self._compute_input(
            no_lora=True,
            dynamic_scale=torch.randn(2),
        )

        with (
            patch(f"{MOE_MLP}.quant_apply_mlp", return_value=expected) as mock_quant,
            patch(f"{MOE_MLP}.w8a8_dynamic_lora_apply_mlp") as mock_lora,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertIs(output, expected)
        mock_quant.assert_called_once()
        mock_lora.assert_not_called()

    def test_active_non_w8a8_lora_is_rejected(self):
        mlp_compute_input = self._compute_input(quant_type=QuantType.W4A8)

        with self.assertRaisesRegex(NotImplementedError, "only W8A8_DYNAMIC"):
            unified_apply_mlp(mlp_compute_input=mlp_compute_input)

    def test_injects_lora_at_bf16_boundaries(self):
        hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
        quantized_input = torch.ones(2, 4, dtype=torch.int8)
        input_scale = torch.ones(2, dtype=torch.float32)
        gate_up_out = torch.randn(2, 6, dtype=torch.bfloat16)
        activated = torch.randn(2, 3, dtype=torch.bfloat16)
        quantized_activated = torch.ones(2, 3, dtype=torch.int8)
        activated_scale = torch.ones(2, dtype=torch.float32)
        down_out = torch.randn(2, 4, dtype=torch.bfloat16)
        lora_context = SimpleNamespace()
        routing = (torch.tensor([0, 1]), torch.tensor([0, 1]))
        expanded_row_idx = torch.tensor([0, 1], dtype=torch.int32)
        topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)
        event = MagicMock(name="before_gmm2_evt")
        stream = MagicMock(name="npu_stream")
        stream.record_event.return_value = event

        with (
            patch(f"{MOE_MLP}._EXTRA_CTX") as mock_ctx,
            patch("torch.npu.current_stream", return_value=stream),
            patch.object(
                DeviceOperator,
                "npu_dynamic_quant",
                side_effect=[
                    (quantized_input, input_scale),
                    (quantized_activated, activated_scale),
                ],
            ) as mock_quant,
            patch(
                f"{MOE_MLP}.torch_npu.npu_grouped_matmul",
                return_value=[gate_up_out],
                create=True,
            ) as mock_gmm1,
            patch(
                f"{MOE_MLP}.torch_npu.npu_swiglu",
                return_value=activated,
                create=True,
            ),
            patch.object(
                DeviceOperator,
                "npu_grouped_matmul_gmm2",
                return_value=down_out,
            ) as mock_gmm2,
            patch(
                "vllm_ascend.lora.fused_moe.moe_lora_apply_w13",
                return_value=routing,
            ) as mock_w13,
            patch("vllm_ascend.lora.fused_moe.moe_lora_apply_w2") as mock_w2,
        ):
            mock_ctx.moe_comm_type = MoECommType.ALLGATHER
            output, output_event = w8a8_dynamic_lora_apply_mlp(
                hidden_states=hidden_states,
                w1=[torch.ones(1, 4, 6, dtype=torch.int8)],
                w1_scale=[torch.ones(1, 6)],
                w2=[torch.ones(1, 3, 4, dtype=torch.int8)],
                w2_scale=[torch.ones(1, 4, dtype=torch.bfloat16)],
                group_list=torch.tensor([1, 1]),
                group_list_type=1,
                activation="silu",
                swiglu_limit=0.0,
                lora_context=lora_context,
                expanded_row_idx=expanded_row_idx,
                topk_ids=topk_ids,
                dynamic_scale=None,
                dynamic_eplb=False,
            )

        self.assertIs(output, down_out)
        self.assertIs(output_event, event)
        self.assertEqual(mock_quant.call_count, 2)
        self.assertIs(
            mock_quant.call_args_list[0].kwargs["hidden_states"],
            hidden_states,
        )
        self.assertIs(
            mock_quant.call_args_list[1].kwargs["hidden_states"],
            activated,
        )
        self.assertIs(mock_gmm1.call_args.kwargs["x"][0], quantized_input)
        self.assertIs(
            mock_gmm2.call_args.kwargs["hidden_states"],
            quantized_activated,
        )
        mock_w13.assert_called_once()
        self.assertIs(mock_w13.call_args.args[0], lora_context)
        self.assertIs(mock_w13.call_args.kwargs["gate_up_out"], gate_up_out)
        self.assertIs(mock_w13.call_args.kwargs["hidden_states"], hidden_states)
        self.assertIs(
            mock_w13.call_args.kwargs["expanded_row_idx"],
            expanded_row_idx,
        )
        self.assertIs(mock_w13.call_args.kwargs["topk_ids"], topk_ids)
        mock_w2.assert_called_once_with(
            lora_context,
            down_out=down_out,
            silu_out=activated,
            lora_routing=routing,
        )

    def test_rejects_unsupported_execution_modes(self):
        kwargs = {
            "hidden_states": torch.randn(1, 4, dtype=torch.bfloat16),
            "w1": [torch.ones(1, 4, 8, dtype=torch.int8)],
            "w1_scale": [torch.ones(1, 8)],
            "w2": [torch.ones(1, 4, 4, dtype=torch.int8)],
            "w2_scale": [torch.ones(1, 4)],
            "group_list": torch.tensor([1]),
            "group_list_type": 1,
            "activation": "silu",
            "swiglu_limit": 0.0,
            "lora_context": SimpleNamespace(),
            "expanded_row_idx": torch.tensor([0]),
            "topk_ids": torch.tensor([[0]]),
            "dynamic_scale": None,
            "dynamic_eplb": False,
        }

        with patch(f"{MOE_MLP}._EXTRA_CTX") as mock_ctx:
            mock_ctx.moe_comm_type = MoECommType.FUSED_MC2
            with self.assertRaisesRegex(NotImplementedError, "FusedMC2"):
                w8a8_dynamic_lora_apply_mlp(**kwargs)

            mock_ctx.moe_comm_type = MoECommType.ALLGATHER
            with self.assertRaisesRegex(NotImplementedError, "dynamic EPLB"):
                w8a8_dynamic_lora_apply_mlp(**(kwargs | {"dynamic_eplb": True}))

            with self.assertRaisesRegex(
                AssertionError,
                "Dispatch-side quantization",
            ):
                w8a8_dynamic_lora_apply_mlp(
                    **(
                        kwargs
                        | {
                            "hidden_states": torch.ones(1, 4, dtype=torch.int8),
                            "dynamic_scale": torch.ones(1),
                        }
                    )
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
