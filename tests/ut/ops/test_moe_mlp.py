import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import torch
import torch_npu  # noqa: F401  -- registers torch.npu used by the module under test
from torch.nn import functional as F
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe.moe_mlp import (
    cumsum_group_list,
    quant_apply_mlp,
    unified_apply_mlp,
    unquant_apply_mlp,
    weight_only_apply_mlp,
)
from vllm_ascend.ops.fused_moe.moe_mlp_activation import (
    MoEMLPActivationKind,
    apply_unquantized_activation,
    resolve_mlp_activation,
)
from vllm_ascend.ops.fused_moe.moe_mlp_plan import (
    GateUpKernel,
    MoEMLPPlanKey,
    MoEMLPPlanner,
    build_mlp_plan,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEQuantParams,
    MoEWeights,
)
from vllm_ascend.ops.fused_moe.moe_stage_params import MoEMxfpParams
from vllm_ascend.quantization.quant_type import QuantType

MOE_MLP = "vllm_ascend.ops.fused_moe.moe_mlp"
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


class TestMoEMLPPlanSelection(unittest.TestCase):
    def _key(self, quant_type, activation=MoEMLPActivationKind.SWIGLU, **kwargs):
        return MoEMLPPlanKey(
            quant_type=quant_type,
            activation=activation,
            fusion=kwargs.get("fusion", False),
            dynamic_eplb=kwargs.get("dynamic_eplb", False),
            group_list_type=kwargs.get("group_list_type", 1),
            custom_op_enabled=kwargs.get("custom_op_enabled", True),
            moe_comm_type=kwargs.get("moe_comm_type", MoECommType.ALLGATHER),
        )

    def test_w4a8_always_selects_per_channel_path(self):
        plan = build_mlp_plan(self._key(QuantType.W4A8))
        self.assertFalse(hasattr(plan, "family"))
        self.assertEqual(plan.gate_up_kernel, GateUpKernel.W4A8_PER_CHANNEL)

    def test_w4a8_gelu_uses_decomposed_per_channel_path(self):
        plan = build_mlp_plan(self._key(QuantType.W4A8, MoEMLPActivationKind.GELU_TANH))
        self.assertEqual(plan.gate_up_kernel, GateUpKernel.DECOMPOSED)

    def test_comm_type_is_a_plan_dimension(self):
        fields = set(MoEMLPPlanKey.__dataclass_fields__)
        self.assertIn("moe_comm_type", fields)

    def test_dequant_swiglu_is_selected_only_for_mc2(self):
        mc2_plan = build_mlp_plan(self._key(QuantType.W8A8, moe_comm_type=MoECommType.MC2))
        allgather_plan = build_mlp_plan(self._key(QuantType.W8A8, moe_comm_type=MoECommType.ALLGATHER))
        alltoall_plan = build_mlp_plan(self._key(QuantType.W8A8, moe_comm_type=MoECommType.ALLTOALL))

        self.assertEqual(mc2_plan.gate_up_kernel, GateUpKernel.DEQUANT_SWIGLU)
        self.assertEqual(allgather_plan.gate_up_kernel, GateUpKernel.DECOMPOSED)
        self.assertEqual(alltoall_plan.gate_up_kernel, GateUpKernel.DECOMPOSED)

    def test_mxfp_swiglu_selects_fused_kernel(self):
        plan = build_mlp_plan(self._key(QuantType.W4A8MXFP, fusion=False))
        self.assertEqual(plan.gate_up_kernel, GateUpKernel.FUSED)

    def test_w4a16_selects_weight_only_plan(self):
        plan = build_mlp_plan(self._key(QuantType.W4A16, MoEMLPActivationKind.GELU))
        self.assertEqual(plan.gate_up_kernel, GateUpKernel.ANTIQUANT)

    def test_gelu_never_selects_fused_swiglu(self):
        plan = build_mlp_plan(self._key(QuantType.W8A8, MoEMLPActivationKind.GELU_TANH, fusion=True))
        self.assertEqual(plan.gate_up_kernel, GateUpKernel.DECOMPOSED)

    def test_swigluoai_never_selects_standard_fused_swiglu(self):
        for quant_type in (QuantType.W8A8, QuantType.W4A8):
            with self.subTest(quant_type=quant_type):
                plan = build_mlp_plan(self._key(quant_type, MoEMLPActivationKind.SWIGLUOAI, fusion=True))
                self.assertEqual(plan.gate_up_kernel, GateUpKernel.DECOMPOSED)

    def test_swigluoai_uninterleave_selects_dequant_swiglu(self):
        mc2_plan = build_mlp_plan(
            self._key(
                QuantType.W8A8,
                MoEMLPActivationKind.SWIGLUOAI_UNINTERLEAVE,
                fusion=True,
                moe_comm_type=MoECommType.MC2,
            )
        )
        allgather_plan = build_mlp_plan(
            self._key(
                QuantType.W8A8,
                MoEMLPActivationKind.SWIGLUOAI_UNINTERLEAVE,
                fusion=True,
                moe_comm_type=MoECommType.ALLGATHER,
            )
        )

        self.assertEqual(mc2_plan.gate_up_kernel, GateUpKernel.DEQUANT_SWIGLU)
        self.assertEqual(allgather_plan.gate_up_kernel, GateUpKernel.DECOMPOSED)

    def test_planner_caches_static_kernel_decisions(self):
        planner = MoEMLPPlanner(custom_op_enabled=False)
        request = SimpleNamespace(
            quant=MoEQuantParams(quant_type=QuantType.W8A8),
            activation="silu",
            fusion=True,
            dynamic_eplb=False,
            group_list_type=0,
            moe_comm_type=MoECommType.ALLGATHER,
        )
        self.assertIs(planner.get_plan(request), planner.get_plan(request))

    def test_planner_reselects_when_comm_type_changes_between_forwards(self):
        planner = MoEMLPPlanner(custom_op_enabled=False)
        common_request = {
            "quant": MoEQuantParams(quant_type=QuantType.W8A8),
            "activation": "silu",
            "fusion": False,
            "dynamic_eplb": False,
            "group_list_type": 0,
        }
        allgather_request = SimpleNamespace(
            **common_request,
            moe_comm_type=MoECommType.ALLGATHER,
        )
        mc2_request = SimpleNamespace(
            **common_request,
            moe_comm_type=MoECommType.MC2,
        )

        allgather_plan = planner.get_plan(allgather_request)
        mc2_plan = planner.get_plan(mc2_request)

        self.assertEqual(allgather_plan.gate_up_kernel, GateUpKernel.DECOMPOSED)
        self.assertEqual(mc2_plan.gate_up_kernel, GateUpKernel.DEQUANT_SWIGLU)
        self.assertIs(planner.get_plan(allgather_request), allgather_plan)
        self.assertIs(planner.get_plan(mc2_request), mc2_plan)

    def test_planner_resolves_custom_op_capability_lazily(self):
        request = SimpleNamespace(
            quant=MoEQuantParams(quant_type=QuantType.W8A8),
            activation="silu",
            fusion=True,
            dynamic_eplb=True,
            group_list_type=0,
            moe_comm_type=MoECommType.ALLGATHER,
        )
        with patch("vllm_ascend.ops.fused_moe.moe_mlp_plan.enable_custom_op", return_value=False) as mock_enable:
            planner = MoEMLPPlanner()
            mock_enable.assert_not_called()
            planner.get_plan(request)
            planner.get_plan(request)
        mock_enable.assert_called_once_with()


class TestMoEMLPActivation(unittest.TestCase):
    def test_unquantized_swigluoai_forwards_activation_parameters(self):
        hidden_states = torch.zeros(2, 8)
        expected = torch.zeros(2, 4)
        with patch(
            "vllm_ascend.ops.fused_moe.moe_mlp_activation.AscendSwigluOAIAndMul.swiglu_oai_forward",
            return_value=expected,
        ) as mock_oai:
            output = apply_unquantized_activation(
                hidden_states,
                MoEMLPActivationKind.SWIGLUOAI,
                hidden_size=8,
                swiglu_limit=6.0,
                swiglu_alpha=1.5,
                swiglu_beta=0.0,
            )

        self.assertIs(output, expected)
        mock_oai.assert_called_once()
        torch.testing.assert_close(mock_oai.call_args.args[0], hidden_states.view(-1, 8))
        self.assertEqual(mock_oai.call_args.kwargs, {"alpha": 1.5, "limit": 6.0})


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
            output_dtype=hidden_states.dtype,
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
            (QuantType.W8A8MXFP, torch.float8_e4m3fn),
            (QuantType.W4A4MXFP, MXFP4_TEST_DTYPE),
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
                    output_dtype=torch.float16,
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
                self.assertTrue(quant_kwargs["kernel_plan"].key.fusion)
                self.assertTrue(quant_kwargs["kernel_plan"].key.dynamic_eplb)
                self.assertEqual(quant_kwargs["act_quant_type"], mxfp_dtype)
                self.assertEqual(quant_kwargs["weight_quant_type"], mxfp_dtype)
                self.assertFalse(quant_kwargs["use_bf16"])
                mock_unquant.assert_not_called()

    def test_request_weight_only_path_uses_dedicated_executor(self):
        hidden_states = torch.randn(2, 8)
        expected = (torch.randn(2, 8), MagicMock())
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=torch.tensor([2, 2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(
                w1=torch.randn(1, 16, 8),
                w2=torch.randn(1, 8, 8),
                w1_scale=torch.randn(1, 16),
                w2_scale=torch.randn(1, 8),
                w1_offset=torch.randn(1, 16),
                w2_offset=torch.randn(1, 8),
            ),
            quant=MoEQuantParams(quant_type=QuantType.W4A16),
            output_dtype=torch.bfloat16,
            fusion=False,
            activation="silu",
        )

        with (
            patch(f"{MOE_MLP}.weight_only_apply_mlp", return_value=expected) as mock_weight_only,
            patch(f"{MOE_MLP}.quant_apply_mlp") as mock_quant,
            patch(f"{MOE_MLP}.unquant_apply_mlp") as mock_unquant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertIs(output, expected)
        mock_weight_only.assert_called_once()
        mock_quant.assert_not_called()
        mock_unquant.assert_not_called()

    def test_request_quant_path_passes_w4a8_per_channel_plan(self):
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
            quant=MoEQuantParams(quant_type=QuantType.W4A8),
            output_dtype=torch.bfloat16,
            fusion=False,
            activation="silu",
            need_trans=False,
            dynamic_eplb=False,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp_plan.enable_custom_op", return_value=False),
            patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp", return_value=expected) as mock_quant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp") as mock_unquant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output is expected)
        quant_kwargs = mock_quant.call_args.kwargs
        kernel_plan = quant_kwargs["kernel_plan"]
        self.assertEqual(kernel_plan.gate_up_kernel, GateUpKernel.DECOMPOSED)
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
            output_dtype=torch.bfloat16,
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
        self.assertEqual(
            quant_kwargs["kernel_plan"].key.activation,
            MoEMLPActivationKind.SWIGLUSTEP,
        )
        self.assertEqual(quant_kwargs["swiglu_limit"], 5.0)
        mock_unquant.assert_not_called()

    def test_request_quant_path_passes_gelu_activation(self):
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
            output_dtype=torch.bfloat16,
            fusion=True,
            activation=MoEActivation.GELU_TANH,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp", return_value=expected) as mock_quant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp") as mock_unquant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output is expected)
        quant_kwargs = mock_quant.call_args.kwargs
        self.assertEqual(
            quant_kwargs["kernel_plan"].key.activation,
            MoEMLPActivationKind.GELU_TANH,
        )
        mock_unquant.assert_not_called()

    def test_active_quantized_lora_uses_registered_backend(self):
        expected = (torch.randn(2, 8), None)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=torch.randn(2, 8, dtype=torch.bfloat16),
            group_list=torch.tensor([2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(
                w1=[torch.ones(1, 8, 16, dtype=torch.int8)],
                w2=[torch.ones(1, 8, 8, dtype=torch.int8)],
                w1_scale=[torch.ones(1, 16)],
                w2_scale=[torch.ones(1, 8)],
            ),
            quant=MoEQuantParams(quant_type=QuantType.W8A8),
            output_dtype=torch.bfloat16,
            fusion=True,
            expanded_row_idx=torch.tensor([0, 1], dtype=torch.int32),
            topk_ids=torch.tensor([[0], [0]], dtype=torch.int32),
            lora_context=SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=False)),
        )

        with (
            patch(
                "vllm_ascend.lora.quant_moe.quant_apply_mlp_with_moe_lora",
                return_value=expected,
            ) as mock_lora,
            patch(f"{MOE_MLP}.quant_apply_mlp") as mock_quant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertIs(output, expected)
        mock_lora.assert_called_once_with(mlp_compute_input=mlp_compute_input)
        mock_quant.assert_not_called()

    def test_base_only_quantized_lora_context_keeps_existing_path(self):
        expected = (torch.randn(2, 8), None)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=torch.ones(2, 8, dtype=torch.int8),
            group_list=torch.tensor([2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=torch.ones(2),
            topk_scales=None,
            weights=MoEWeights(
                w1=[torch.ones(1, 8, 16, dtype=torch.int8)],
                w2=[torch.ones(1, 8, 8, dtype=torch.int8)],
                w1_scale=[torch.ones(1, 16)],
                w2_scale=[torch.ones(1, 8)],
            ),
            quant=MoEQuantParams(quant_type=QuantType.W8A8),
            output_dtype=torch.bfloat16,
            fusion=True,
            lora_context=SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=True)),
        )

        with (
            patch(f"{MOE_MLP}.quant_apply_mlp", return_value=expected) as mock_quant,
            patch("vllm_ascend.lora.quant_moe.quant_apply_mlp_with_moe_lora") as mock_lora,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertIs(output, expected)
        mock_quant.assert_called_once()
        mock_lora.assert_not_called()


def _patch_npu_stream():
    """Patch ``torch.npu.current_stream`` so ``record_event()`` returns a tag."""
    evt = MagicMock(name="before_gmm2_evt")
    stream = MagicMock(name="npu_stream")
    stream.record_event.return_value = evt
    return patch("torch.npu.current_stream", return_value=stream), evt


@contextmanager
def _mock_w8a8_gelu_compute(gate_up, *, gmm2_out=None, capture_quant=False):
    """Mock the W8A8 GELU-path NPU ops: dequant GMM1 (``npu_grouped_matmul``),
    requant (``npu_dynamic_quant``), GMM2 (``npu_grouped_matmul_gmm2``), plus the
    NPU stream event and ``dispose_tensor``. Yields a namespace with the mocks;
    when ``capture_quant`` is True, ``captured['x']``/``captured['scale']``
    record the requant input and the returned per-token scale."""
    stream_patch, evt = _patch_npu_stream()
    captured = {}

    def _dynamic_quant(x, dst_type=None):
        if capture_quant:
            captured["x"] = x.detach().clone()
            scale = torch.ones(1, dtype=torch.float32)
            captured["scale"] = scale
            return x, scale
        return x, torch.ones(1)

    with (
        stream_patch,
        patch("torch_npu.npu_grouped_matmul", return_value=[gate_up], create=True) as mock_gmm,
        patch("torch_npu.npu_dynamic_quant", side_effect=_dynamic_quant, create=True) as mock_dq,
        patch.object(
            DeviceOperator,
            "npu_grouped_matmul_gmm2",
            return_value=gmm2_out if gmm2_out is not None else torch.zeros(1, 4),
        ) as mock_gmm2,
        patch(f"{MOE_MLP}.dispose_tensor"),
    ):
        yield SimpleNamespace(gmm=mock_gmm, dq=mock_dq, gmm2=mock_gmm2, evt=evt, captured=captured)


class _GeluPathBase(unittest.TestCase):
    """Common helpers for the GELU-path tests."""

    def _common_w8a8_kwargs(
        self,
        *,
        activation,
        w1_scale_dtype=torch.float32,
        w2_scale_dtype=torch.float32,
        w1_scale_bias=None,
        w2_scale_bias=None,
        group_list_type=1,
        group_list=None,
        dynamic_scale=None,
    ):
        kernel_plan = build_mlp_plan(
            MoEMLPPlanKey(
                quant_type=QuantType.W8A8,
                activation=resolve_mlp_activation(activation),
                fusion=False,
                dynamic_eplb=False,
                group_list_type=group_list_type,
                custom_op_enabled=False,
                moe_comm_type=MoECommType.MC2,
            )
        )
        return dict(
            hidden_states=torch.randn(1, 4),
            w1=torch.randn(1, 8, 4),
            w1_scale=[torch.randn(1, 8, dtype=w1_scale_dtype)],
            w2=torch.randn(1, 4, 1),
            w2_scale=[torch.randn(1, 4, dtype=w2_scale_dtype)],
            output_dtype=torch.bfloat16,
            group_list=group_list if group_list is not None else torch.tensor([1], dtype=torch.int64),
            group_list_type=group_list_type,
            dynamic_scale=dynamic_scale if dynamic_scale is not None else torch.randn(1, 1),
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            use_mxfp_quant=False,
            mxfp_quant_dtype=None,
            act_quant_type=torch.int8,
            weight_quant_type=torch.float8_e4m3fn,
            use_bf16=True,
            swiglu_limit=0.0,
            kernel_plan=kernel_plan,
        )


class TestQuantApplyMlpGeluPath(_GeluPathBase):
    """GELU path: dispatch, math, and layout coverage.

    Kernel selection is resolved by the MLP plan before execution.
    """

    def test_w8a8_gelu_tanh_applies_correct_activation(self):
        """W8A8 + gelu_tanh: GMM1(dequant) -> gelu(tanh)·up -> requant -> GMM2."""
        gate = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
        up = torch.tensor([[0.5, -0.5, 1.0, 2.0]])
        gate_up = torch.cat([gate, up], dim=-1)
        expected = F.gelu(gate, approximate="tanh") * up
        gmm2_out = torch.tensor([[9.0]])
        with _mock_w8a8_gelu_compute(gate_up, gmm2_out=gmm2_out, capture_quant=True) as m:
            out, out_evt = quant_apply_mlp(**self._common_w8a8_kwargs(activation=MoEActivation.GELU_TANH))
        # GELU math applied with tanh approximation before requantization.
        self.assertTrue(torch.allclose(m.captured["x"], expected, atol=1e-6))
        # GMM1 used the dequant form (scale + per_token_scale), not antiquant.
        gmm1_kwargs = m.gmm.call_args.kwargs
        self.assertIn("scale", gmm1_kwargs)
        self.assertIn("per_token_scale", gmm1_kwargs)
        self.assertNotIn("antiquant_scale", gmm1_kwargs)
        self.assertEqual(gmm1_kwargs["split_item"], 2)
        # Requant + GMM2 both invoked; GMM2 received the requant per-token scale.
        m.dq.assert_called_once()
        m.gmm2.assert_called_once()
        self.assertIs(m.gmm2.call_args.kwargs["per_token_scale"], m.captured["scale"])
        # Return contract: (hidden_states, before_gmm2_evt).
        self.assertIs(out, gmm2_out)
        self.assertIs(out_evt, m.evt)

    def test_w8a8_gelu_uses_exact_gelu_approximation(self):
        """W8A8 + gelu (not tanh): approximate='none', matching the float path."""
        gate = torch.tensor([[0.5, -0.5, 2.0]])
        up = torch.tensor([[1.0, 1.0, 0.5]])
        gate_up = torch.cat([gate, up], dim=-1)
        expected = F.gelu(gate, approximate="none") * up
        with _mock_w8a8_gelu_compute(gate_up, gmm2_out=torch.zeros(1, 3), capture_quant=True) as m:
            quant_apply_mlp(**self._common_w8a8_kwargs(activation=MoEActivation.GELU))
        # exact GELU (approximate='none') differs from tanh; ensure 'none' used.
        self.assertFalse(torch.allclose(m.captured["x"], F.gelu(gate, approximate="tanh") * up, atol=1e-6))
        self.assertTrue(torch.allclose(m.captured["x"], expected, atol=1e-6))

    def test_w4a16_gelu_uses_antiquant_path(self):
        """W4A16 + gelu: antiquant GMM1 -> gelu·up -> antiquant GMM2, no requant."""
        gate = torch.tensor([[1.0, -1.0]])
        up = torch.tensor([[0.5, 2.0]])
        gate_up = torch.cat([gate, up], dim=-1)
        expected = F.gelu(gate, approximate="tanh") * up
        gmm2_out = torch.tensor([[3.0]])
        stream_patch, evt = _patch_npu_stream()
        with (
            stream_patch,
            patch("torch_npu.npu_grouped_matmul", side_effect=[[gate_up], [gmm2_out]], create=True) as mock_gmm,
            patch("torch_npu.npu_dynamic_quant", create=True) as mock_dq,
            patch.object(DeviceOperator, "npu_grouped_matmul_gmm2") as mock_gmm2,
            patch(f"{MOE_MLP}.dispose_tensor"),
        ):
            out, out_evt = weight_only_apply_mlp(
                hidden_states=torch.randn(1, 4),
                w1=torch.randn(1, 8, 4),
                w1_scale=torch.randn(1, 8),
                w1_offset=torch.randn(1, 8, 4),
                w2=torch.randn(1, 4, 1),
                w2_scale=torch.randn(1, 4),
                w2_offset=torch.randn(1, 4, 1),
                group_list=torch.tensor([1], dtype=torch.int64),
                output_dtype=torch.bfloat16,
                activation=MoEMLPActivationKind.GELU_TANH,
            )

        self.assertEqual(mock_gmm.call_count, 2)
        # Both GMM calls use antiquant (not scale/per_token_scale).
        for call in mock_gmm.call_args_list:
            self.assertIn("antiquant_scale", call.kwargs)
            self.assertIn("antiquant_offset", call.kwargs)
            self.assertNotIn("scale", call.kwargs)
        # GMM2 (second call) input is the GELU activation output.
        gmm2_input = mock_gmm.call_args_list[1].kwargs["x"][0]
        self.assertTrue(torch.allclose(gmm2_input, expected, atol=1e-6))
        # W4A16 path does NOT requantize.
        mock_dq.assert_not_called()
        mock_gmm2.assert_not_called()
        self.assertIs(out, gmm2_out)
        self.assertIs(out_evt, evt)

    def test_w4a8_gelu_uses_bfloat16_intermediate_and_model_output_dtype(self):
        """W4A8 keeps its BF16 GMM1 contract but restores model dtype in GMM2."""
        w1_sb = [torch.zeros(1)]
        w2_sb = [torch.zeros(1)]
        with (
            _mock_w8a8_gelu_compute(torch.zeros(1, 8), gmm2_out=torch.zeros(1, 2)) as m,
            patch("torch.cat", wraps=torch.cat) as mock_cat,
        ):
            kwargs = self._common_w8a8_kwargs(
                activation=MoEActivation.GELU_TANH,
                w1_scale_bias=w1_sb,
                w2_scale_bias=w2_sb,
                group_list_type=0,
                group_list=torch.tensor([0, 1], dtype=torch.int64),
            )
            kwargs["kernel_plan"] = build_mlp_plan(
                MoEMLPPlanKey(
                    quant_type=QuantType.W4A8,
                    activation=MoEMLPActivationKind.GELU_TANH,
                    fusion=False,
                    dynamic_eplb=False,
                    group_list_type=0,
                    custom_op_enabled=False,
                )
            )
            kwargs["output_dtype"] = torch.float16
            quant_apply_mlp(**kwargs)
        # bias1 propagated to GMM1.
        self.assertIs(m.gmm.call_args.kwargs["bias"], w1_sb)
        self.assertEqual(m.gmm.call_args.kwargs["output_dtype"], torch.bfloat16)
        self.assertIs(m.gmm2.call_args.kwargs["bias"], w2_sb)
        self.assertEqual(m.gmm2.call_args.kwargs["fallback_output_dtype"], torch.float16)
        # group_list_type 0 -> 1 conversion invoked (torch.cat + torch.diff).
        self.assertTrue(mock_cat.called)

    def test_w4a8_per_channel_requires_both_scale_biases(self):
        kernel_plan = build_mlp_plan(
            MoEMLPPlanKey(
                quant_type=QuantType.W4A8,
                activation=MoEMLPActivationKind.GELU_TANH,
                fusion=False,
                dynamic_eplb=False,
                group_list_type=1,
                custom_op_enabled=False,
            )
        )
        scale_bias = [torch.zeros(1)]

        for w1_scale_bias, w2_scale_bias in (
            (None, None),
            (scale_bias, None),
            (None, scale_bias),
        ):
            with self.subTest(w1_scale_bias=w1_scale_bias, w2_scale_bias=w2_scale_bias):
                kwargs = self._common_w8a8_kwargs(
                    activation=MoEActivation.GELU_TANH,
                    w1_scale_bias=w1_scale_bias,
                    w2_scale_bias=w2_scale_bias,
                )
                kwargs["kernel_plan"] = kernel_plan

                with self.assertRaisesRegex(
                    ValueError,
                    "W4A8 per-channel MLP requires both w1_scale_bias and w2_scale_bias",
                ):
                    quant_apply_mlp(**kwargs)

    def test_w8a8_dynamic_eplb_swigluoai_passes_all_scales_to_gmm1(self):
        scale0 = torch.arange(8, dtype=torch.float32)
        scale1 = torch.arange(8, 16, dtype=torch.float32)
        kwargs = self._common_w8a8_kwargs(activation="swigluoai_uninterleave")
        kwargs.update(
            {
                "w1": [torch.randn(8, 4), torch.randn(8, 4)],
                "w1_scale": [scale0, scale1],
                "w2": [torch.randn(4, 4), torch.randn(4, 4)],
                "w2_scale": [torch.randn(4), torch.randn(4)],
                "group_list": torch.tensor([1, 1], dtype=torch.int64),
                "kernel_plan": build_mlp_plan(
                    MoEMLPPlanKey(
                        quant_type=QuantType.W8A8,
                        activation=MoEMLPActivationKind.SWIGLUOAI_UNINTERLEAVE,
                        fusion=False,
                        dynamic_eplb=True,
                        group_list_type=1,
                        custom_op_enabled=False,
                        moe_comm_type=MoECommType.ALLTOALL,
                    )
                ),
            }
        )

        with (
            _mock_w8a8_gelu_compute(torch.zeros(1, 8)) as mocks,
            patch("torch_npu.npu_clipped_swiglu", return_value=torch.zeros(1, 4), create=True),
        ):
            quant_apply_mlp(**kwargs)

        gmm1_scales = mocks.gmm.call_args.kwargs["scale"]
        self.assertEqual(len(gmm1_scales), 2)
        torch.testing.assert_close(gmm1_scales[0], scale0.bfloat16())
        torch.testing.assert_close(gmm1_scales[1], scale1.bfloat16())

    def test_w8a8_gelu_uses_explicit_output_dtype_when_hidden_is_quantized(self):
        """The explicit model dtype, rather than w2_scale dtype, controls GMM output."""
        with _mock_w8a8_gelu_compute(torch.zeros(1, 8)) as m:
            kwargs = self._common_w8a8_kwargs(
                activation=MoEActivation.GELU_TANH,
                w1_scale_dtype=torch.float32,
                w2_scale_dtype=torch.float32,
            )
            kwargs["hidden_states"] = torch.zeros(1, 4, dtype=torch.int8)
            kwargs["output_dtype"] = torch.float16
            quant_apply_mlp(**kwargs)
        self.assertEqual(m.gmm2.call_args.kwargs["input_dtype"], torch.float16)
        self.assertIsNone(m.gmm2.call_args.kwargs["bias"])
        self.assertEqual(m.gmm2.call_args.kwargs["fallback_output_dtype"], torch.float16)

    def test_mxfp_gelu_decomposed_gmm1_outputs_bfloat16(self):
        """MXFP decomposed activations require a BF16 GMM1 output."""
        kwargs = self._common_w8a8_kwargs(activation=MoEActivation.GELU_TANH)
        kwargs.update(
            use_mxfp_quant=True,
            mxfp_quant_dtype=QuantType.W8A8MXFP,
            kernel_plan=build_mlp_plan(
                MoEMLPPlanKey(
                    quant_type=QuantType.W8A8MXFP,
                    activation=MoEMLPActivationKind.GELU_TANH,
                    fusion=False,
                    dynamic_eplb=False,
                    group_list_type=1,
                    custom_op_enabled=False,
                )
            ),
        )

        with _mock_w8a8_gelu_compute(torch.zeros(1, 8)) as m:
            quant_apply_mlp(**kwargs)

        gmm1_kwargs = m.gmm.call_args.kwargs
        self.assertEqual(gmm1_kwargs["output_dtype"], torch.bfloat16)
        self.assertEqual(gmm1_kwargs["scale_dtype"], torch_npu.float8_e8m0fnu)
        self.assertEqual(gmm1_kwargs["per_token_scale_dtype"], torch_npu.float8_e8m0fnu)

    def test_gelu_path_does_not_call_swiglu_op(self):
        """GELU path must use torch.gelu, never the SwiGLU NPU op."""
        with _mock_w8a8_gelu_compute(torch.zeros(1, 8)), patch("torch_npu.npu_swiglu", create=True) as mock_swiglu:
            quant_apply_mlp(**self._common_w8a8_kwargs(activation=MoEActivation.GELU_TANH))
        mock_swiglu.assert_not_called()

    def test_fusion_on_gelu_skips_fused_swiglu_quant(self):
        """A GELU plan must not select the fused SwiGLU quant kernel."""
        kwargs = self._common_w8a8_kwargs(activation=MoEActivation.GELU_TANH)
        kwargs["kernel_plan"] = build_mlp_plan(
            MoEMLPPlanKey(
                quant_type=QuantType.W8A8,
                activation=MoEMLPActivationKind.GELU_TANH,
                fusion=True,
                dynamic_eplb=False,
                group_list_type=1,
                custom_op_enabled=True,
            )
        )
        with (
            _mock_w8a8_gelu_compute(torch.zeros(1, 8)) as m,
            patch.object(DeviceOperator, "npu_grouped_matmul_swiglu_quant") as mock_fused,
        ):
            quant_apply_mlp(**kwargs)
        # Fused SwiGLU+quant op must NOT be called for GELU.
        mock_fused.assert_not_called()
        # Non-fused dequant GMM1 (scale + per_token_scale) IS used.
        self.assertIn("scale", m.gmm.call_args.kwargs)
        self.assertIn("per_token_scale", m.gmm.call_args.kwargs)

    def test_gelu_plan_skips_fused_swiglu_branch(self):
        """GELU uses the decomposed plan independently of communication."""
        with (
            _mock_w8a8_gelu_compute(torch.zeros(1, 8)) as m,
            patch("torch.ops._C_ascend.npu_dequant_swiglu_quant", create=True) as mock_dequant_fused,
            patch.object(DeviceOperator, "npu_grouped_matmul_swiglu_quant") as mock_grouped_fused,
        ):
            quant_apply_mlp(**self._common_w8a8_kwargs(activation=MoEActivation.GELU_TANH))
        # The fused SwiGLU op must not be called for GELU.
        mock_dequant_fused.assert_not_called()
        mock_grouped_fused.assert_not_called()
        # Non-fused dequant GMM1 IS used instead.
        self.assertIn("scale", m.gmm.call_args.kwargs)

    def test_dynamic_eplb_swigluoai_stacks_w1_scale(self):
        gate_up_out = torch.zeros(1, 8, dtype=torch.int32)
        expected = torch.zeros(1, 4, dtype=torch.float32)
        scale0 = torch.arange(8, dtype=torch.bfloat16)
        scale1 = torch.arange(8, 16, dtype=torch.bfloat16)

        kwargs = self._common_w8a8_kwargs(activation="swigluoai_uninterleave")
        kwargs.update(
            {
                "w1": [torch.randn(8, 4), torch.randn(8, 4)],
                "w1_scale": [scale0, scale1],
                "w2": [torch.randn(4, 4), torch.randn(4, 4)],
                "w2_scale": [torch.randn(4), torch.randn(4)],
                "group_list": torch.tensor([1, 1], dtype=torch.int64),
            }
        )

        with (
            _patch_npu_stream()[0],
            patch("torch_npu.npu_grouped_matmul", return_value=[gate_up_out], create=True),
            patch(
                "torch.ops._C_ascend.npu_dequant_swiglu_quant",
                return_value=(torch.zeros(1, 4, dtype=torch.int8), torch.ones(1)),
                create=True,
            ) as mock_dequant_swiglu,
            patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=expected),
            patch(f"{MOE_MLP}.dispose_tensor"),
        ):
            out, _ = quant_apply_mlp(**kwargs)

        weight_scale = mock_dequant_swiglu.call_args.kwargs["weight_scale"]
        self.assertEqual(weight_scale.shape, torch.Size([2, 8]))
        self.assertEqual(weight_scale.dtype, torch.float32)
        torch.testing.assert_close(weight_scale[0], scale0.float())
        torch.testing.assert_close(weight_scale[1], scale1.float())
        self.assertIs(out, expected)

    def test_swigluoai_preserves_single_list_w1_scale_shape(self):
        gate_up_out = torch.zeros(1, 8, dtype=torch.int32)
        expected = torch.zeros(1, 4, dtype=torch.float32)
        weight_scale = torch.arange(12, dtype=torch.bfloat16).view(3, 4)

        kwargs = self._common_w8a8_kwargs(activation="swigluoai_uninterleave")
        kwargs.update(
            {
                "w1": [torch.randn(8, 4)],
                "w1_scale": [weight_scale],
                "w2": [torch.randn(4, 4)],
                "w2_scale": [torch.randn(4)],
                "group_list": torch.tensor([1], dtype=torch.int64),
            }
        )

        with (
            _patch_npu_stream()[0],
            patch("torch_npu.npu_grouped_matmul", return_value=[gate_up_out], create=True),
            patch(
                "torch.ops._C_ascend.npu_dequant_swiglu_quant",
                return_value=(torch.zeros(1, 4, dtype=torch.int8), torch.ones(1)),
                create=True,
            ) as mock_dequant_swiglu,
            patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=expected),
            patch(f"{MOE_MLP}.dispose_tensor"),
        ):
            out, _ = quant_apply_mlp(**kwargs)

        packed_scale = mock_dequant_swiglu.call_args.kwargs["weight_scale"]
        self.assertEqual(packed_scale.shape, torch.Size([3, 4]))
        self.assertEqual(packed_scale.dtype, torch.float32)
        torch.testing.assert_close(packed_scale, weight_scale.float())
        self.assertIs(out, expected)


class TestQuantApplyMlpNoGeluImpact(_GeluPathBase):
    """Non-GELU activations must NOT enter the GELU path (no regression)."""

    def _run_non_gelu(self, activation):
        with (
            _mock_w8a8_gelu_compute(torch.zeros(1, 8)),
            patch("vllm_ascend.ops.fused_moe.moe_mlp_activation.HAS_TRITON", False),
            patch("torch_npu.npu_swiglu", return_value=torch.zeros(1, 4), create=True) as mock_swiglu,
            patch("torch.nn.functional.gelu") as mock_gelu,
        ):
            kwargs = self._common_w8a8_kwargs(activation=activation)
            key = kwargs["kernel_plan"].key
            kwargs["kernel_plan"] = type(kwargs["kernel_plan"])(
                key=key,
                gate_up_kernel=GateUpKernel.DECOMPOSED,
            )
            quant_apply_mlp(**kwargs)
        return mock_gelu, mock_swiglu

    def test_silu_activation_skips_gelu_path(self):
        mock_gelu, mock_swiglu = self._run_non_gelu("silu")
        mock_gelu.assert_not_called()
        # SwiGLu op IS used by the existing path -> existing logic intact.
        mock_swiglu.assert_called()

    def test_swiglustep_activation_skips_gelu_path(self):
        mock_gelu, _ = self._run_non_gelu(MoEActivation.SWIGLUSTEP)
        mock_gelu.assert_not_called()

    def test_swigluoai_activation_uses_oai_formula(self):
        gate_up_out = torch.zeros(1, 8)
        activated = torch.zeros(1, 4)
        with (
            _mock_w8a8_gelu_compute(gate_up_out),
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp_activation.AscendSwigluOAIAndMul.swiglu_oai_forward",
                return_value=activated,
            ) as mock_oai,
            patch("torch_npu.npu_swiglu", create=True) as mock_swiglu,
            patch("torch.nn.functional.gelu") as mock_gelu,
        ):
            kwargs = self._common_w8a8_kwargs(activation=MoEActivation.SWIGLUOAI)
            kwargs.update(swiglu_alpha=1.5, swiglu_limit=6.0)
            quant_apply_mlp(**kwargs)

        mock_oai.assert_called_once_with(gate_up_out, alpha=1.5, limit=6.0)
        mock_swiglu.assert_not_called()
        mock_gelu.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
