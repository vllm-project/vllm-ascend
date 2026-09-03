import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import torch
import torch_npu  # noqa: F401  -- registers torch.npu used by the module under test
from torch.nn import functional as F
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SituAndMul
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEMxfpParams, MoEQuantParams
from vllm_ascend.ops.fused_moe.moe_mlp import (
    _swiglu_oai_dynamic_mx_quant,
    cumsum_group_list,
    quant_apply_mlp,
    unified_apply_mlp,
    unquant_apply_mlp,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType

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


class TestSwigluOaiDynamicMxQuant(unittest.TestCase):
    def test_uses_small_op_activation_and_dynamic_mx_quant(self):
        hidden_states = torch.randn(2, 8)
        activated = torch.randn(2, 4)
        quantized = torch.randn(2, 4)
        output_scale = torch.randn(2, 1)

        with (
            patch(f"{MOE_MLP}.ASCEND_DEVICE_TYPE", AscendDeviceType.A5),
            patch(
                f"{MOE_MLP}._apply_clipped_swiglu",
                return_value=activated,
            ) as mock_activation,
            patch.object(
                DeviceOperator,
                "npu_dynamic_quant",
                return_value=(quantized, output_scale),
            ) as mock_dynamic_quant,
        ):
            output, actual_scale = _swiglu_oai_dynamic_mx_quant(
                hidden_states,
                act_quant_type=torch.float8_e4m3fn,
                swiglu_limit=7.0,
                swiglu_alpha=1.5,
                swiglu_beta=0.25,
            )

        self.assertIs(output, quantized)
        self.assertIs(actual_scale, output_scale)
        mock_activation.assert_called_once_with(
            hidden_states,
            swiglu_limit=7.0,
            swiglu_alpha=1.5,
            swiglu_beta=0.25,
        )
        mock_dynamic_quant.assert_called_once_with(
            activated,
            act_quant_type=torch.float8_e4m3fn,
            use_mxfp_quant=True,
        )


class TestUnifiedApplyMlpRequest(unittest.TestCase):
    def test_unquant_situ_matches_upstream_without_config_context(self):
        # The reference CustomOp needs dispatch config, not platform/logging setup.
        reference_config = MagicMock(spec=VllmConfig)
        reference_config.compilation_config = CompilationConfig(custom_ops=["none"])
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for beta, linear_beta in ((None, None), (None, 25.0), (4.0, None), (4.0, 25.0)):
                with self.subTest(dtype=dtype, beta=beta, linear_beta=linear_beta):
                    hidden_states = torch.randn(2, 8, dtype=dtype)
                    gate_up_out = torch.linspace(-40, 40, 32).reshape(2, 16).to(dtype)
                    expected_output = torch.randn(2, 8, dtype=dtype)
                    with set_current_vllm_config(reference_config):
                        expected_activation = SituAndMul(
                            beta=1.0 if beta is None else beta, linear_beta=linear_beta, compile_native=False
                        ).forward_native(gate_up_out)

                    # The worker forward runs outside the model-init config context.
                    with patch(
                        f"{MOE_MLP}.torch_npu.npu_grouped_matmul",
                        side_effect=[[gate_up_out], [expected_output]],
                        create=True,
                    ) as grouped_matmul:
                        output, _ = unquant_apply_mlp(
                            hidden_states=hidden_states,
                            w1=torch.randn(1, 8, 16),
                            w2=torch.randn(1, 8, 8),
                            group_list=torch.tensor([1, 1]),
                            activation=MoEActivation.SITU,
                            activation_situ_beta=beta,
                            activation_situ_linear_beta=linear_beta,
                            need_trans=False,
                        )

                    self.assertIs(output, expected_output)
                    torch.testing.assert_close(grouped_matmul.call_args_list[1].kwargs["x"][0], expected_activation)

    def test_unquant_swigluoai_uninterleave_falls_back_on_a5(self):
        hidden_states = torch.randn(2, 8, dtype=torch.bfloat16)
        gate_up_out = torch.randn(2, 16, dtype=torch.bfloat16)
        expected_output = torch.randn(2, 8, dtype=torch.bfloat16)
        w1 = torch.randn(2, 8, 16, dtype=torch.bfloat16)
        w2 = torch.randn(2, 8, 8, dtype=torch.bfloat16)
        swiglu_limit = 7.0
        swiglu_alpha = 1.702
        swiglu_beta = 1.0

        gate = gate_up_out[..., :8].clamp(max=swiglu_limit)
        up = gate_up_out[..., 8:].clamp(
            min=-swiglu_limit,
            max=swiglu_limit,
        )
        expected_activation = gate * torch.sigmoid(swiglu_alpha * gate) * (up + swiglu_beta)

        with (
            patch(f"{MOE_MLP}.ASCEND_DEVICE_TYPE", AscendDeviceType.A5),
            patch(
                "torch_npu.npu_grouped_matmul",
                side_effect=[[gate_up_out], [expected_output]],
                create=True,
            ) as mock_grouped_matmul,
            patch("torch_npu.npu_clipped_swiglu", create=True) as mock_clipped_swiglu,
        ):
            output, _ = unquant_apply_mlp(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                group_list=torch.tensor([1, 1]),
                activation="swigluoai_uninterleave",
                need_trans=False,
                swiglu_limit=swiglu_limit,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
            )

        self.assertIs(output, expected_output)
        second_call = mock_grouped_matmul.call_args_list[1]
        torch.testing.assert_close(second_call.kwargs["x"][0], expected_activation)
        mock_clipped_swiglu.assert_not_called()

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

    def test_request_quant_path_passes_situ_parameters(self):
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
            fusion=False,
            activation=MoEActivation.SITU,
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
        )

        with (
            patch(f"{MOE_MLP}.quant_apply_mlp", return_value=expected) as mock_quant,
            patch(f"{MOE_MLP}.unquant_apply_mlp") as mock_unquant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertIs(output, expected)
        self.assertEqual(mock_quant.call_args.kwargs["activation"], MoEActivation.SITU)
        self.assertEqual(mock_quant.call_args.kwargs["activation_situ_beta"], 4.0)
        self.assertEqual(mock_quant.call_args.kwargs["activation_situ_linear_beta"], 25.0)
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
        self.assertEqual(quant_kwargs["activation"], MoEActivation.GELU_TANH)
        mock_unquant.assert_not_called()


def _patch_npu_stream():
    """Patch ``torch.npu.current_stream`` so ``record_event()`` returns a tag."""
    evt = MagicMock(name="before_gmm2_evt")
    stream = MagicMock(name="npu_stream")
    stream.record_event.return_value = evt
    return patch(f"{MOE_MLP}.torch.npu.current_stream", return_value=stream), evt


@contextmanager
def _mock_w8a8_gelu_compute(gate_up, *, gmm2_out=None, capture_quant=False):
    """Mock the W8A8 GELU-path NPU ops: dequant GMM1 (``npu_grouped_matmul``),
    requant (``npu_dynamic_quant``), GMM2 (``npu_grouped_matmul_gmm2``), plus the
    NPU stream event and ``dispose_tensor``. Yields a namespace with the mocks;
    when ``capture_quant`` is True, ``captured['x']``/``captured['scale']``
    record the requant input and the returned per-token scale."""
    stream_patch, evt = _patch_npu_stream()
    captured = {}

    def _dynamic_quant(x, dst_type=None, **kwargs):
        if capture_quant:
            captured["x"] = x.detach().clone()
            scale = torch.ones(1, dtype=torch.float32)
            captured["scale"] = scale
            return x, scale
        return x, torch.ones(1)

    with (
        stream_patch,
        patch(f"{MOE_MLP}.torch_npu.npu_grouped_matmul", return_value=[gate_up], create=True) as mock_gmm,
        patch(f"{MOE_MLP}.torch_npu.npu_dynamic_quant", side_effect=_dynamic_quant, create=True) as mock_dq,
        patch.object(DeviceOperator, "npu_dynamic_quant", side_effect=_dynamic_quant),
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
        return dict(
            hidden_states=torch.randn(1, 4),
            w1=torch.randn(1, 8, 4),
            w1_scale=[torch.randn(1, 8, dtype=w1_scale_dtype)],
            w2=torch.randn(1, 4, 1),
            w2_scale=[torch.randn(1, 4, dtype=w2_scale_dtype)],
            group_list=group_list if group_list is not None else torch.tensor([1], dtype=torch.int64),
            group_list_type=group_list_type,
            dynamic_scale=dynamic_scale if dynamic_scale is not None else torch.randn(1, 1),
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            w1_offset=None,
            w2_offset=None,
            fusion=False,
            dynamic_eplb=False,
            use_mxfp_quant=False,
            mxfp_quant_dtype=None,
            act_quant_type=torch.int8,
            weight_quant_type=torch.float8_e4m3fn,
            use_bf16=True,
            activation=activation,
            swiglu_limit=0.0,
            use_w4a8_per_channel_gmm_swiglu=False,
        )


class TestQuantApplyMlpSituEplb(_GeluPathBase):
    def test_dynamic_eplb_tensor_lists_reach_both_grouped_matmuls(self):
        hidden_states = torch.ones(2, 4, dtype=torch.int8)
        w1 = [torch.randn(8, 4), torch.randn(8, 4)]
        w2 = [torch.randn(4, 4), torch.randn(4, 4)]
        w1_scale = [torch.randn(8), torch.randn(8)]
        w2_scale = [torch.randn(4), torch.randn(4)]
        w1_scale_bias = [torch.randn(8), torch.randn(8)]
        w2_scale_bias = [torch.randn(4), torch.randn(4)]
        gate_up_out = torch.randn(2, 8, dtype=torch.bfloat16)
        quantized_situ_out = torch.ones(2, 4, dtype=torch.int8)
        situ_out_scale = torch.ones(2, 1)
        expected = torch.randn(2, 4, dtype=torch.bfloat16)
        stream_patch, evt = _patch_npu_stream()

        with (
            patch(f"{MOE_MLP}._EXTRA_CTX", MagicMock(moe_comm_type=-1)),
            stream_patch,
            patch(f"{MOE_MLP}.envs.VLLM_ASCEND_ENABLE_GMSQ_SITU", False),
            patch("torch_npu.npu_grouped_matmul", return_value=[gate_up_out], create=True) as mock_gmm1,
            patch(
                "torch.ops._C_ascend.dequant_situ_quant",
                return_value=(quantized_situ_out, situ_out_scale),
                create=True,
            ),
            patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=expected) as mock_gmm2,
            patch(f"{MOE_MLP}.dispose_tensor"),
        ):
            output, before_gmm2_evt = quant_apply_mlp(
                hidden_states=hidden_states,
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                group_list=torch.tensor([1, 1], dtype=torch.int64),
                dynamic_scale=torch.ones(2, 1),
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                dynamic_eplb=True,
                act_quant_type=torch.int8,
                activation=MoEActivation.SITU,
                activation_situ_beta=4.0,
                activation_situ_linear_beta=25.0,
                use_w4a8_per_channel_gmm_swiglu=True,
            )

        self.assertIs(output, expected)
        self.assertIs(before_gmm2_evt, evt)
        self.assertIs(mock_gmm1.call_args.kwargs["weight"], w1)
        self.assertEqual(len(mock_gmm1.call_args.kwargs["scale"]), 2)
        self.assertIs(mock_gmm1.call_args.kwargs["bias"], w1_scale_bias)
        self.assertIs(mock_gmm2.call_args.kwargs["weight"], w2)
        self.assertIs(mock_gmm2.call_args.kwargs["weight_scale"], w2_scale)
        self.assertIs(mock_gmm2.call_args.kwargs["bias"], w2_scale_bias)

    def test_a3_w4a8_situ_uses_gmsq_fusion(self):
        from vllm_ascend.ops.fused_moe import moe_mlp as moe_mlp_module
        from vllm_ascend.utils import AscendDeviceType

        hidden_states = torch.ones(2, 4, dtype=torch.int8)
        w1 = torch.ones(2, 4, 1, dtype=torch.int32)
        w2 = [torch.randn(4, 4), torch.randn(4, 4)]
        w1_scale = torch.ones(2, 4, dtype=torch.int64)
        w2_scale = [torch.randn(4), torch.randn(4)]
        w1_scale_bias = [torch.randn(4), torch.randn(4)]
        w2_scale_bias = [torch.randn(4), torch.randn(4)]
        quantized_situ_out = torch.ones(2, 4, dtype=torch.int8)
        situ_out_scale = torch.ones(2, dtype=torch.float32)
        expected = torch.randn(2, 4, dtype=torch.bfloat16)
        mock_gmsq = MagicMock(return_value=(quantized_situ_out, situ_out_scale))
        stream_patch, evt = _patch_npu_stream()

        with (
            patch(f"{MOE_MLP}._EXTRA_CTX", MagicMock(moe_comm_type=-1)),
            stream_patch,
            patch.object(moe_mlp_module, "ASCEND_DEVICE_TYPE", AscendDeviceType.A3),
            patch.object(moe_mlp_module.envs, "VLLM_ASCEND_ENABLE_GMSQ_SITU", True),
            patch.object(moe_mlp_module, "_GMSQ_SITU_OP", mock_gmsq),
            patch.object(moe_mlp_module, "_GMSQ_SITU_OP_RESOLVED", True),
            patch("torch_npu.npu_grouped_matmul", create=True) as mock_gmm1,
            patch(
                "torch.ops._C_ascend.dequant_situ_quant",
                create=True,
            ) as mock_dequant_situ,
            patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=expected) as mock_gmm2,
            patch(f"{MOE_MLP}.dispose_tensor"),
        ):
            output, before_gmm2_evt = quant_apply_mlp(
                hidden_states=hidden_states,
                w1=[w1],
                w1_scale=[w1_scale],
                w2=w2,
                w2_scale=w2_scale,
                group_list=torch.tensor([2], dtype=torch.int64),
                dynamic_scale=torch.ones(2, 1),
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                act_quant_type=torch.int8,
                activation=MoEActivation.SITU,
                activation_situ_beta=4.0,
                activation_situ_linear_beta=25.0,
                use_w4a8_per_channel_gmm_swiglu=True,
            )

        self.assertIs(output, expected)
        self.assertIs(before_gmm2_evt, evt)
        mock_gmm1.assert_not_called()
        mock_dequant_situ.assert_not_called()
        mock_gmsq.assert_called_once()
        gmsq_kwargs = mock_gmsq.call_args.kwargs
        self.assertEqual(len(gmsq_kwargs["weight"]), 2)
        self.assertEqual(len(gmsq_kwargs["weight_scale"]), 2)
        self.assertEqual(gmsq_kwargs["weight_scale"][0].shape, torch.Size([4]))
        self.assertEqual(gmsq_kwargs["beta"], 4.0)
        self.assertEqual(gmsq_kwargs["linear_beta"], 25.0)
        self.assertEqual(gmsq_kwargs["x_scale"].shape, torch.Size([2]))
        self.assertIs(mock_gmm2.call_args.kwargs["hidden_states"], quantized_situ_out)
        self.assertEqual(mock_gmm2.call_args.kwargs["per_token_scale"].shape, torch.Size([2, 1]))
        self.assertIs(mock_gmm2.call_args.kwargs["bias"], w2_scale_bias)

    def test_a3_w8a8_situ_stays_in_legacy_flow(self):
        from vllm_ascend.ops.fused_moe import moe_mlp as moe_mlp_module
        from vllm_ascend.utils import AscendDeviceType

        gate_up_out = torch.randn(1, 8, dtype=torch.bfloat16)
        quantized_situ_out = torch.ones(1, 4, dtype=torch.int8)
        situ_out_scale = torch.ones(1, 1)
        expected = torch.randn(1, 4, dtype=torch.bfloat16)
        mock_gmsq = MagicMock(return_value=(quantized_situ_out, situ_out_scale))
        stream_patch, evt = _patch_npu_stream()

        with (
            patch(f"{MOE_MLP}._EXTRA_CTX", MagicMock(moe_comm_type=-1)),
            stream_patch,
            patch.object(moe_mlp_module, "ASCEND_DEVICE_TYPE", AscendDeviceType.A3),
            patch.object(moe_mlp_module.envs, "VLLM_ASCEND_ENABLE_GMSQ_SITU", True),
            patch.object(moe_mlp_module, "_GMSQ_SITU_OP", mock_gmsq),
            patch.object(moe_mlp_module, "_GMSQ_SITU_OP_RESOLVED", True),
            patch("torch_npu.npu_grouped_matmul", return_value=[gate_up_out], create=True) as mock_gmm1,
            patch(
                "torch.ops._C_ascend.dequant_situ_quant",
                return_value=(quantized_situ_out, situ_out_scale),
                create=True,
            ) as mock_dequant_situ,
            patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=expected),
            patch(f"{MOE_MLP}.dispose_tensor"),
        ):
            output, before_gmm2_evt = quant_apply_mlp(**self._common_w8a8_kwargs(activation=MoEActivation.SITU))

        self.assertIs(output, expected)
        self.assertIs(before_gmm2_evt, evt)
        mock_gmsq.assert_not_called()
        mock_gmm1.assert_called_once()
        mock_dequant_situ.assert_called_once()

    def test_w4a8_mxfp_situ_stays_in_common_grouped_matmul_flow(self):
        gate_up_out = torch.randn(2, 8, dtype=torch.bfloat16)
        quantized_situ_out = torch.ones(2, 4, dtype=torch.float8_e4m3fn)
        situ_out_scale = torch.ones(2, 1)
        expected = torch.randn(2, 4, dtype=torch.bfloat16)
        stream_patch, evt = _patch_npu_stream()

        with (
            stream_patch,
            patch(f"{MOE_MLP}._EXTRA_CTX", MagicMock(moe_comm_type=-1)),
            patch("torch_npu.npu_grouped_matmul", return_value=[gate_up_out], create=True) as mock_gmm1,
            patch(
                "torch.ops._C_ascend.situ_mx_quant",
                return_value=(quantized_situ_out, situ_out_scale),
                create=True,
            ) as situ_mx_quant,
            patch.object(DeviceOperator, "maybe_normalize_mxfp_scale_layout", side_effect=lambda scale: scale),
            patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=expected) as mock_gmm2,
            patch(f"{MOE_MLP}.dispose_tensor"),
        ):
            kwargs = self._common_w8a8_kwargs(activation=MoEActivation.SITU)
            kwargs.update(
                activation_situ_beta=4.0,
                activation_situ_linear_beta=25.0,
                use_mxfp_quant=True,
                mxfp_quant_dtype=QuantType.W4A8MXFP,
            )
            output, before_gmm2_evt = quant_apply_mlp(**kwargs)

        self.assertIs(output, expected)
        self.assertIs(before_gmm2_evt, evt)
        self.assertIsNone(mock_gmm1.call_args.kwargs["scale"])
        self.assertIsNotNone(mock_gmm1.call_args.kwargs["antiquant_scale"])
        self.assertEqual(situ_mx_quant.call_args.kwargs["beta"], 4.0)
        self.assertEqual(situ_mx_quant.call_args.kwargs["linear_beta"], 25.0)
        self.assertIs(mock_gmm2.call_args.kwargs["per_token_scale"], situ_out_scale)

    def test_antiquant_weights_use_native_situ_between_grouped_matmuls(self):
        gate_up_out = torch.tensor([[1.0, -1.0, 0.5, 2.0]])
        gate, up = gate_up_out.chunk(2, dim=-1)
        expected_activation = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
        expected_activation *= 25.0 * torch.tanh(up / 25.0)
        expected = torch.tensor([[3.0]])
        stream_patch, evt = _patch_npu_stream()

        with (
            stream_patch,
            patch(f"{MOE_MLP}._EXTRA_CTX", MagicMock(moe_comm_type=-1)),
            patch(
                "torch_npu.npu_grouped_matmul",
                side_effect=[[gate_up_out], [expected]],
                create=True,
            ) as grouped_matmul,
            patch("torch_npu.npu_dynamic_quant", create=True) as dynamic_quant,
            patch.object(DeviceOperator, "npu_grouped_matmul_gmm2") as gmm2,
            patch(f"{MOE_MLP}.dispose_tensor"),
        ):
            kwargs = self._common_w8a8_kwargs(activation=MoEActivation.SITU)
            kwargs.update(
                activation_situ_beta=4.0,
                activation_situ_linear_beta=25.0,
                w1_offset=torch.randn(1, 8, 4),
                w2_offset=torch.randn(1, 4, 1),
            )
            output, before_gmm2_evt = quant_apply_mlp(**kwargs)

        self.assertIs(output, expected)
        self.assertIs(before_gmm2_evt, evt)
        torch.testing.assert_close(grouped_matmul.call_args_list[1].kwargs["x"][0], expected_activation)
        for call in grouped_matmul.call_args_list:
            self.assertIn("antiquant_scale", call.kwargs)
            self.assertIn("antiquant_offset", call.kwargs)
        dynamic_quant.assert_not_called()
        gmm2.assert_not_called()

    def test_w4a16_mxfp_uses_native_situ_without_activation_requant(self):
        gate_up_out = torch.tensor([[1.0, -1.0, 0.5, 2.0]])
        gate, up = gate_up_out.chunk(2, dim=-1)
        expected_activation = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
        expected_activation *= 25.0 * torch.tanh(up / 25.0)
        expected = torch.tensor([[3.0]])
        stream_patch, evt = _patch_npu_stream()

        with (
            stream_patch,
            patch(f"{MOE_MLP}._EXTRA_CTX", MagicMock(moe_comm_type=-1)),
            patch("torch_npu.npu_grouped_matmul", return_value=[gate_up_out], create=True),
            patch("torch_npu.npu_dynamic_quant", create=True) as dynamic_quant,
            patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=expected) as gmm2,
            patch(f"{MOE_MLP}.dispose_tensor"),
        ):
            kwargs = self._common_w8a8_kwargs(activation=MoEActivation.SITU)
            kwargs.update(
                activation_situ_beta=4.0,
                activation_situ_linear_beta=25.0,
                use_mxfp_quant=True,
                mxfp_quant_dtype=QuantType.W4A16MXFP,
            )
            output, before_gmm2_evt = quant_apply_mlp(**kwargs)

        self.assertIs(output, expected)
        self.assertIs(before_gmm2_evt, evt)
        torch.testing.assert_close(gmm2.call_args.kwargs["hidden_states"], expected_activation)
        self.assertIsNone(gmm2.call_args.kwargs["per_token_scale"])
        dynamic_quant.assert_not_called()


class TestQuantApplyMlpMxfpSwigluOAI(_GeluPathBase):
    def setUp(self):
        self._ctx_mock = MagicMock()
        self._ctx_patch = patch(f"{MOE_MLP}._EXTRA_CTX", self._ctx_mock)
        self._ctx_patch.start()
        self.addCleanup(self._ctx_patch.stop)

    def test_uses_small_op_swiglu_oai_and_dynamic_mx_quant_for_mc2_and_alltoall(self):
        for comm_type in (MoECommType.MC2, MoECommType.ALLTOALL):
            with self.subTest(comm_type=comm_type):
                self._ctx_mock.moe_comm_type = comm_type
                gate_up_out = torch.randn(1, 8, dtype=torch.bfloat16)
                quantized_swiglu_out = torch.randn(1, 4)
                swiglu_out_scale = torch.randn(1, 1)
                expected = torch.randn(1, 4)
                stream_patch, evt = _patch_npu_stream()

                kwargs = self._common_w8a8_kwargs(activation="swigluoai_uninterleave")
                kwargs.update(
                    {
                        "use_mxfp_quant": True,
                        "mxfp_quant_dtype": QuantType.W8A8MXFP,
                        "act_quant_type": torch.float8_e4m3fn,
                    }
                )

                with (
                    stream_patch,
                    patch(
                        f"{MOE_MLP}.torch_npu.npu_grouped_matmul", return_value=[gate_up_out], create=True
                    ) as mock_gmm1,
                    patch(
                        f"{MOE_MLP}._swiglu_oai_dynamic_mx_quant",
                        return_value=(quantized_swiglu_out, swiglu_out_scale),
                    ) as mock_small_ops_quant,
                    patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=expected) as mock_gmm2,
                    patch.object(DeviceOperator, "npu_dynamic_quant") as mock_dynamic_quant,
                    patch(f"{MOE_MLP}.torch_npu.npu_clipped_swiglu", create=True) as mock_clipped_swiglu,
                    patch(f"{MOE_MLP}.dispose_tensor"),
                ):
                    output, output_evt = quant_apply_mlp(**kwargs)

                self.assertIs(output, expected)
                self.assertIs(output_evt, evt)
                gmm1_kwargs = mock_gmm1.call_args.kwargs
                self.assertEqual(gmm1_kwargs["output_dtype"], torch.bfloat16)
                self.assertEqual(gmm1_kwargs["scale_dtype"], torch_npu.float8_e8m0fnu)
                self.assertEqual(gmm1_kwargs["per_token_scale_dtype"], torch_npu.float8_e8m0fnu)
                mock_small_ops_quant.assert_called_once_with(
                    gate_up_out,
                    act_quant_type=torch.float8_e4m3fn,
                    swiglu_limit=0.0,
                    swiglu_alpha=1.0,
                    swiglu_beta=0.0,
                )
                self.assertIs(mock_gmm2.call_args.kwargs["hidden_states"], quantized_swiglu_out)
                self.assertIs(mock_gmm2.call_args.kwargs["per_token_scale"], swiglu_out_scale)
                mock_dynamic_quant.assert_not_called()
                mock_clipped_swiglu.assert_not_called()


class TestQuantApplyMlpGeluPath(_GeluPathBase):
    """GELU path: dispatch, math, and layout coverage.

    In the in-branch/guard variant the GELU path runs through the existing
    branch preamble. Stub `_EXTRA_CTX` in setUp so each test can focus on the
    GELU dispatch/math.
    """

    def setUp(self):
        # Configurable forward-context mock; default moe_comm_type is not MC2.
        self._ctx_mock = MagicMock()
        self._ctx_mock.moe_comm_type = -1
        self._patches = [
            patch(f"{MOE_MLP}._EXTRA_CTX", self._ctx_mock),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(self._stop_patches)

    def _stop_patches(self):
        for p in self._patches:
            p.stop()

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
            patch(
                f"{MOE_MLP}.torch_npu.npu_grouped_matmul", side_effect=[[gate_up], [gmm2_out]], create=True
            ) as mock_gmm,
            patch(f"{MOE_MLP}.torch_npu.npu_dynamic_quant", create=True) as mock_dq,
            patch.object(DeviceOperator, "npu_grouped_matmul_gmm2") as mock_gmm2,
            patch(f"{MOE_MLP}.dispose_tensor"),
        ):
            kwargs = self._common_w8a8_kwargs(activation=MoEActivation.GELU_TANH)
            # Switch to the W4A16 antiquant layout.
            kwargs["w1_offset"] = torch.randn(1, 8, 4)
            kwargs["w2_offset"] = torch.randn(1, 4, 1)
            out, out_evt = quant_apply_mlp(**kwargs)

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

    def test_w8a8_gelu_with_scale_bias_sets_bias_and_bfloat16(self):
        """W8A8 + gelu + scale_bias: bias1/bias2 passed, output dtype bfloat16,
        and group_list_type 0 -> 1 conversion applied."""
        w1_sb = [torch.zeros(1)]
        w2_sb = [torch.zeros(1)]
        with (
            _mock_w8a8_gelu_compute(torch.zeros(1, 8), gmm2_out=torch.zeros(1, 2)) as m,
            patch("torch.cat", wraps=torch.cat) as mock_cat,
        ):
            quant_apply_mlp(
                **self._common_w8a8_kwargs(
                    activation=MoEActivation.GELU_TANH,
                    w1_scale_bias=w1_sb,
                    w2_scale_bias=w2_sb,
                    group_list_type=0,
                    group_list=torch.tensor([0, 1], dtype=torch.int64),
                )
            )
        # bias1 propagated to GMM1.
        self.assertIs(m.gmm.call_args.kwargs["bias"], w1_sb)
        # group_list_type 0 -> 1 conversion invoked (torch.cat + torch.diff).
        self.assertTrue(mock_cat.called)

    def test_w8a8_gelu_converts_w1_scale_dtype_to_output_dtype(self):
        """When w1_scale dtype != _output_dtype, it is cast before GMM1."""
        # w1_scale fp32, w2_scale bf16 -> _output_dtype = bfloat16, so the GELU
        # path must cast w1_scale to bfloat16 before GMM1.
        with _mock_w8a8_gelu_compute(torch.zeros(1, 8)) as m:
            quant_apply_mlp(
                **self._common_w8a8_kwargs(
                    activation=MoEActivation.GELU_TANH,
                    w1_scale_dtype=torch.float32,
                    w2_scale_dtype=torch.bfloat16,
                )
            )
        self.assertEqual(m.gmm.call_args.kwargs["scale"][0].dtype, torch.bfloat16)

    def test_alltoall_dynamic_eplb_swigluoai_passes_all_w1_scales_to_gmm1(self):
        self._ctx_mock.moe_comm_type = MoECommType.ALLTOALL
        scale0 = torch.arange(8, dtype=torch.float32)
        scale1 = torch.arange(8, 16, dtype=torch.float32)
        with (
            _mock_w8a8_gelu_compute(torch.zeros(1, 8)) as m,
            patch("torch_npu.npu_clipped_swiglu", return_value=torch.zeros(1, 4), create=True),
            patch.object(DeviceOperator, "npu_dynamic_quant", return_value=(torch.zeros(1, 4), torch.ones(1))),
        ):
            kwargs = self._common_w8a8_kwargs(
                activation="swigluoai_uninterleave",
                w2_scale_dtype=torch.bfloat16,
            )
            kwargs.update(
                {
                    "w1": [torch.randn(8, 4), torch.randn(8, 4)],
                    "w1_scale": [scale0, scale1],
                    "w2": [torch.randn(4, 4), torch.randn(4, 4)],
                    "w2_scale": [
                        torch.randn(4, dtype=torch.bfloat16),
                        torch.randn(4, dtype=torch.bfloat16),
                    ],
                    "group_list": torch.tensor([1, 1], dtype=torch.int64),
                    "dynamic_eplb": True,
                }
            )
            quant_apply_mlp(**kwargs)

        gmm1_kwargs = m.gmm.call_args.kwargs
        self.assertEqual(len(gmm1_kwargs["weight"]), 2)
        self.assertEqual(len(gmm1_kwargs["scale"]), 2)
        self.assertEqual(gmm1_kwargs["scale"][0].dtype, torch.bfloat16)
        self.assertEqual(gmm1_kwargs["scale"][1].dtype, torch.bfloat16)
        torch.testing.assert_close(gmm1_kwargs["scale"][0], scale0.bfloat16())
        torch.testing.assert_close(gmm1_kwargs["scale"][1], scale1.bfloat16())

    def test_gelu_path_does_not_call_swiglu_op(self):
        """GELU path must use torch.gelu, never the SwiGLU NPU op."""
        with _mock_w8a8_gelu_compute(torch.zeros(1, 8)), patch("torch_npu.npu_swiglu", create=True) as mock_swiglu:
            quant_apply_mlp(**self._common_w8a8_kwargs(activation=MoEActivation.GELU_TANH))
        mock_swiglu.assert_not_called()

    def test_fusion_on_gelu_skips_fused_swiglu_quant(self):
        """Guard: with fusion ON (default), GELU must still skip the fused
        npu_grouped_matmul_swiglu_quant op and use the non-fused GELU path.
        This is the case that breaks without the ``and not is_gelu_activation``
        guard on use_gmm_swiglu_quant_fusion."""
        kwargs = self._common_w8a8_kwargs(activation=MoEActivation.GELU_TANH)
        kwargs["fusion"] = True  # -> use_gmm_swiglu_quant_fusion = True
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

    def test_mc2_gelu_skips_mc2_fused_branch(self):
        """Guard: under MC2 comm, GELU must skip the all-fused MC2 branch and
        use the non-fused GELU path. Without the ``and not is_gelu_activation``
        guard on the MC2 entry, GELU+MC2 would hit npu_dequant_swiglu_quant."""
        self._ctx_mock.moe_comm_type = MoECommType.MC2  # force is_mc2 True
        with (
            _mock_w8a8_gelu_compute(torch.zeros(1, 8)) as m,
            patch("torch.ops._C_ascend.npu_dequant_swiglu_quant", create=True) as mock_mc2_fused,
            patch.object(DeviceOperator, "npu_grouped_matmul_swiglu_quant") as mock_fused,
        ):
            quant_apply_mlp(**self._common_w8a8_kwargs(activation=MoEActivation.GELU_TANH))
        # MC2 fused SwiGLU op must NOT be called for GELU.
        mock_mc2_fused.assert_not_called()
        mock_fused.assert_not_called()
        # Non-fused dequant GMM1 IS used instead.
        self.assertIn("scale", m.gmm.call_args.kwargs)

    def test_mc2_dynamic_eplb_swigluoai_stacks_w1_scale(self):
        self._ctx_mock.moe_comm_type = MoECommType.MC2
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
                "dynamic_eplb": True,
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

    def test_mc2_swigluoai_preserves_single_list_w1_scale_shape(self):
        self._ctx_mock.moe_comm_type = MoECommType.MC2
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
                "dynamic_eplb": False,
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
        gate_up = torch.zeros(1, 16 if activation == MoEActivation.SWIGLUSTEP else 8)
        with (
            _mock_w8a8_gelu_compute(gate_up),
            patch(f"{MOE_MLP}._EXTRA_CTX") as mock_ctx,
            patch(f"{MOE_MLP}.HAS_TRITON", False),
            patch("vllm.triton_utils.HAS_TRITON", False),
            patch("torch_npu.npu_swiglu", return_value=torch.zeros(1, 4), create=True) as mock_swiglu,
            patch("torch.nn.functional.gelu") as mock_gelu,
        ):
            mock_ctx.moe_comm_type = -1  # not MoECommType.MC2
            quant_apply_mlp(**self._common_w8a8_kwargs(activation=activation))
        return mock_gelu, mock_swiglu

    def test_silu_activation_skips_gelu_path(self):
        mock_gelu, mock_swiglu = self._run_non_gelu("silu")
        mock_gelu.assert_not_called()
        # SwiGLu op IS used by the existing path -> existing logic intact.
        mock_swiglu.assert_called()

    def test_swiglustep_activation_skips_gelu_path(self):
        mock_gelu, _ = self._run_non_gelu(MoEActivation.SWIGLUSTEP)
        mock_gelu.assert_not_called()

    def test_swigluoai_activation_skips_gelu_path(self):
        mock_gelu, _ = self._run_non_gelu(MoEActivation.SWIGLUOAI)
        mock_gelu.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
