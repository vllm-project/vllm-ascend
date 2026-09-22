import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch_npu  # noqa: F401 -- registers torch.npu used by the module under test
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.ops.fused_moe.moe_mlp import _unified_apply_activation, apply_moe_mlp
from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicFusedMoEMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod
from vllm_ascend.quantization.methods.wna16.w4a16 import AscendW4A16FusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType

MXFP4_TEST_DTYPE = getattr(torch, "float4_e2m1fn_x2", torch.float16)
MOE_MLP = "vllm_ascend.ops.fused_moe.moe_mlp"


def _mlp_compute_input(**kwargs):
    defaults = dict(
        hidden_states=torch.randn(4, 8),
        group_list=torch.tensor([2, 2]),
        group_list_type=1,
        dynamic_scale=None,
        topk_scales=None,
        weights=MoEWeights(w1=None, w2=None),
        quant=MoEQuantParams(),
        fusion=False,
        activation="silu",
        layer=SimpleNamespace(w13_weight=torch.randn(2, 8, 16)),
    )
    defaults.update(kwargs)
    return MoEMlpComputeInput(**defaults)


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


def _w8a8_layer():
    return SimpleNamespace(
        w13_weight=torch.randn(1, 8, 16),
        w13_weight_scale_fp32=torch.randn(1, 8),
        w2_weight=torch.randn(1, 16, 8),
        w2_weight_scale=torch.randn(1, 16),
        activation="silu",
    )


class TestW8A8FusedMoEMethod(unittest.TestCase):
    def _make_method(self, use_expert_weight_list=False):
        method = AscendW8A8DynamicFusedMoEMethod.__new__(AscendW8A8DynamicFusedMoEMethod)
        method.use_expert_weight_list = use_expert_weight_list
        return method

    def test_apply_gmm1_act_quant_custom_op_path(self):
        method = self._make_method()
        layer = _w8a8_layer()
        mlp_compute_input = _mlp_compute_input(layer=layer, fusion=True, dynamic_eplb=True)
        with (
            patch(
                "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic._EXTRA_CTX",
                SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER),
            ),
            patch("vllm_ascend.ops.fused_moe.moe_utils.enable_custom_op", return_value=True),
            patch(
                "torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list",
                return_value=("out", "scale", None),
                create=True,
            ) as mock_op,
            patch("torch_npu.npu_dynamic_quant", return_value=("qx", "pscale"), create=True),
        ):
            out, scale = method.apply_gmm1_act_quant(mlp_compute_input)
        self.assertEqual((out, scale), ("out", "scale"))
        mock_op.assert_called_once()
        self.assertIs(mock_op.call_args.kwargs["weight"][0], layer.w13_weight)

    def test_apply_gmm1_act_quant_fused_op_path(self):
        method = self._make_method()
        layer = _w8a8_layer()
        mlp_compute_input = _mlp_compute_input(layer=layer, fusion=True, dynamic_eplb=False)
        with (
            patch(
                "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic._EXTRA_CTX",
                SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER),
            ),
            patch(
                "torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz",
                return_value=("out", "scale", None),
                create=True,
            ) as mock_fused,
            patch("torch_npu.npu_dynamic_quant", return_value=("qx", "pscale"), create=True),
        ):
            out, scale = method.apply_gmm1_act_quant(mlp_compute_input)
        self.assertEqual((out, scale), ("out", "scale"))
        mock_fused.assert_called_once()
        self.assertIs(mock_fused.call_args.kwargs["weight"], layer.w13_weight)
        self.assertIs(mock_fused.call_args.kwargs["weight_scale"], layer.w13_weight_scale_fp32)
        self.assertIsNone(mock_fused.call_args.kwargs["bias"])

    def test_apply_gmm1_act_quant_swigluoai_dequant_path(self):
        method = self._make_method()
        layer = _w8a8_layer()
        mlp_compute_input = _mlp_compute_input(
            layer=layer,
            fusion=False,
            activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            swiglu_limit=3.0,
            swiglu_alpha=1.5,
            swiglu_beta=0.25,
        )
        with (
            patch(
                "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic._EXTRA_CTX",
                SimpleNamespace(moe_comm_type=MoECommType.MC2),
            ),
            patch("torch_npu.npu_grouped_matmul", return_value=["int32_out"], create=True) as mock_gmm,
            patch(
                "torch.ops._C_ascend.npu_dequant_swiglu_quant",
                return_value=("out", "scale"),
                create=True,
            ) as mock_dequant,
            patch("torch_npu.npu_dynamic_quant", return_value=("qx", "pscale"), create=True),
        ):
            out, scale = method.apply_gmm1_act_quant(mlp_compute_input)
        self.assertEqual((out, scale), ("out", "scale"))
        mock_gmm.assert_called_once()
        self.assertEqual(mock_gmm.call_args.kwargs["output_dtype"], torch.int32)
        self.assertEqual(mock_dequant.call_args.kwargs["swiglu_mode"], 1)
        self.assertEqual(mock_dequant.call_args.kwargs["clamp_limit"], 3.0)

    def test_apply_gmm1_act_quant_soft_fallback_path(self):
        method = self._make_method()
        layer = _w8a8_layer()
        mlp_compute_input = _mlp_compute_input(layer=layer, fusion=False)
        with (
            patch(
                "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic._EXTRA_CTX",
                SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER),
            ),
            patch("torch_npu.npu_grouped_matmul", return_value=["gmm1_out"], create=True) as mock_gmm,
            patch("vllm_ascend.quantization.methods.w8a8.w8a8_dynamic.HAS_TRITON", False),
            patch("torch_npu.npu_swiglu", return_value="silu_out", create=True),
            patch("torch_npu.npu_dynamic_quant", return_value=("quant_out", "qscale"), create=True),
        ):
            out, scale = method.apply_gmm1_act_quant(mlp_compute_input)
        self.assertEqual((out, scale), ("quant_out", "qscale"))
        mock_gmm.assert_called_once()
        self.assertIs(mock_gmm.call_args.kwargs["weight"][0], layer.w13_weight)

    def test_apply_gmm1_uses_soft_quant_matmul(self):
        method = self._make_method()
        layer = _w8a8_layer()
        mlp_compute_input = _mlp_compute_input(layer=layer)
        with (
            patch("torch_npu.npu_grouped_matmul", return_value=["out"], create=True) as mock_gmm,
            patch("torch_npu.npu_dynamic_quant", return_value=("qx", "pscale"), create=True),
        ):
            out = method.apply_gmm1(mlp_compute_input)
        self.assertEqual(out, "out")
        self.assertEqual(mock_gmm.call_args.kwargs["split_item"], 2)
        self.assertEqual(mock_gmm.call_args.kwargs["per_token_scale"], ["pscale"])

    def test_apply_act_quant_and_gmm2(self):
        method = self._make_method()
        layer = _w8a8_layer()
        mlp_compute_input = _mlp_compute_input(layer=layer)
        with (
            patch("torch_npu.npu_dynamic_quant", return_value=("quant_out", "qscale"), create=True),
            patch("torch_npu.npu_grouped_matmul", return_value=["final_out"], create=True) as mock_gmm2,
        ):
            out, scale = method.apply_act_quant(mlp_compute_input, "x")
            final = method.apply_gmm2(mlp_compute_input, "quant_out", scale)
        self.assertEqual((out, scale), ("quant_out", "qscale"))
        self.assertEqual(final, "final_out")
        mock_gmm2.assert_called_once()
        self.assertIs(mock_gmm2.call_args.kwargs["weight"][0], layer.w2_weight)
        self.assertEqual(mock_gmm2.call_args.kwargs["output_dtype"], torch.float32)

    def test_get_fused_mc2_weights_single_tensor_form(self):
        method = self._make_method()
        layer = _w8a8_layer()
        with patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic._EXTRA_CTX",
            SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER),
        ):
            weights = method.get_fused_mc2_weights(layer)
        self.assertIs(weights.w1[0], layer.w13_weight)
        self.assertIs(weights.w1_scale[0], layer.w13_weight_scale_fp32)
        self.assertIs(weights.w2_scale[0], layer.w2_weight_scale)
        self.assertIsNone(weights.w1_scale_bias)

    def test_get_fused_mc2_weights_fused_mc2_scale_flag(self):
        method = self._make_method()
        layer = SimpleNamespace(
            w13_weight=torch.randn(1, 8, 16),
            fused_w1_scale=torch.randn(1, 8),
            fused_w1_scale_bias=torch.tensor([], dtype=torch.float32),
            w2_weight=torch.randn(1, 16, 8),
            fused_w2_scale=torch.randn(1, 16),
            fused_w2_scale_bias=torch.tensor([], dtype=torch.float32),
            activation="silu",
        )
        with (
            patch(
                "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic._EXTRA_CTX",
                SimpleNamespace(moe_comm_type=MoECommType.FUSED_MC2, use_mega_moe=False),
            ),
            patch("vllm_ascend.quantization.methods.w8a8.w8a8_dynamic.get_ascend_config") as mock_config,
        ):
            mock_config.return_value.enable_fused_mc2 = 1
            weights = method.get_fused_mc2_weights(layer)
        self.assertIs(weights.w1_scale[0], layer.fused_w1_scale)
        self.assertIs(weights.w1_scale_bias, layer.fused_w1_scale_bias)
        self.assertIs(weights.w2_scale_bias, layer.fused_w2_scale_bias)

    def test_get_fused_mc2_weights_expert_list_form(self):
        method = self._make_method(use_expert_weight_list=True)
        layer = SimpleNamespace(
            w13_weight_list=[torch.randn(8, 16)],
            w13_weight_scale_fp32_list=[torch.randn(8)],
            w2_weight_list=[torch.randn(16, 8)],
            w2_weight_scale_list=[torch.randn(16)],
            activation="silu",
        )
        with patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic._EXTRA_CTX",
            SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER),
        ):
            weights = method.get_fused_mc2_weights(layer)
        self.assertIs(weights.w1, layer.w13_weight_list)
        self.assertIs(weights.w1_scale, layer.w13_weight_scale_fp32_list)


class TestW4A8SituPath(unittest.TestCase):
    def test_situ_gmm1_uses_per_channel_scale_layout(self):
        method = AscendW4A8DynamicFusedMoEMethod.__new__(AscendW4A8DynamicFusedMoEMethod)
        method.use_expert_weight_list = False
        layer = SimpleNamespace(
            w13_weight=torch.randn(1, 8, 16),
            w13_weight_scale=torch.randn(1, 8),
            w13_scale_bias=torch.randn(1, 8),
            w2_weight=torch.randn(1, 16, 8),
            w2_weight_scale=torch.randn(1, 16),
            activation="situ",
        )
        mlp_compute_input = _mlp_compute_input(
            layer=layer,
            fusion=False,
            activation=MoEActivation.SITU,
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
            quant=MoEQuantParams(quant_type=QuantType.W4A8, is_per_channel_weight=True),
        )
        with (
            patch("torch_npu.npu_dynamic_quant", return_value=("qx", torch.ones(2)), create=True),
            patch("torch_npu.npu_grouped_matmul", return_value=["bf16_out"], create=True) as mock_gmm,
            patch(
                "torch.ops._C_ascend.dequant_situ_quant",
                return_value=("qout", "oscale"),
                create=True,
            ) as mock_situ,
        ):
            out, scale = method.apply_gmm1_act_quant(mlp_compute_input)

        self.assertEqual((out, scale), ("qout", "oscale"))
        # Per-channel W4A8 must pass the scale in [E, 1, N] layout, matching
        # upstream quant_apply_mlp's SITU branch (unsqueeze(-2)); otherwise the
        # A8W4 tiling reads the flattened scale as quantGroupNum and fails.
        gmm_scale = mock_gmm.call_args.kwargs["scale"][0]
        self.assertEqual(gmm_scale.ndim, 3)
        self.assertEqual(gmm_scale.shape, (1, 1, 8))
        self.assertIs(mock_gmm.call_args.kwargs["bias"][0], layer.w13_scale_bias)
        self.assertEqual(mock_gmm.call_args.kwargs["output_dtype"], torch.bfloat16)
        mock_situ.assert_called_once()
        self.assertEqual(mock_situ.call_args.kwargs["beta"], 4.0)
        self.assertEqual(mock_situ.call_args.kwargs["linear_beta"], 25.0)


class TestW4A16FusedMoEMethod(unittest.TestCase):
    def _make_method(self):
        return AscendW4A16FusedMoEMethod.__new__(AscendW4A16FusedMoEMethod)

    def test_apply_gmm1_and_gmm2_use_antiquant_offsets(self):
        method = self._make_method()
        layer = SimpleNamespace(
            w13_weight_packed=torch.zeros(2, 8, 4, dtype=torch.int32),
            w13_weight_scale=torch.randn(2, 8, 4),
            w13_weight_offset=torch.randn(2, 8, 4),
            w2_weight_packed=torch.zeros(2, 4, 8, dtype=torch.int32),
            w2_weight_scale=torch.randn(2, 4, 8),
            w2_weight_offset=torch.randn(2, 4, 8),
        )
        mlp_compute_input = _mlp_compute_input(layer=layer)
        with patch("torch_npu.npu_grouped_matmul", side_effect=[["gmm1_out"], ["final_out"]], create=True) as mock_gmm:
            gmm1_out = method.apply_gmm1(mlp_compute_input)
            final_out = method.apply_gmm2(mlp_compute_input, gmm1_out, None)
        self.assertEqual((gmm1_out, final_out), ("gmm1_out", "final_out"))
        gmm1_kwargs = mock_gmm.call_args_list[0].kwargs
        gmm2_kwargs = mock_gmm.call_args_list[1].kwargs
        self.assertIs(gmm1_kwargs["weight"][0], layer.w13_weight_packed)
        self.assertIs(gmm1_kwargs["antiquant_offset"][0], layer.w13_weight_offset)
        self.assertIs(gmm2_kwargs["weight"][0], layer.w2_weight_packed)
        self.assertIs(gmm2_kwargs["antiquant_offset"][0], layer.w2_weight_offset)

    def test_apply_act_quant_keeps_activation_unquantized(self):
        method = self._make_method()
        mlp_compute_input = _mlp_compute_input()
        out, scale = method.apply_act_quant(mlp_compute_input, "x")
        self.assertEqual((out, scale), ("x", None))

    def test_no_fused_activation(self):
        method = self._make_method()
        self.assertFalse(method.supports_fused_activation("silu"))


class TestUnquantizedFusedMoEMethod(unittest.TestCase):
    def _make_method(self, has_bias=False):
        method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        method.moe = SimpleNamespace(has_bias=has_bias)
        method.lora_context = None
        method._lora_routing = None
        return method

    def test_apply_gmm1_transposes_and_runs_grouped_matmul(self):
        method = self._make_method()
        layer = SimpleNamespace(
            w13_weight=torch.randn(2, 8, 16),
            w2_weight=torch.randn(2, 16, 8),
            w13_bias=None,
            w2_bias=None,
        )
        mlp_compute_input = _mlp_compute_input(layer=layer, need_trans=True)
        with (
            patch(
                "vllm_ascend.ops.fused_moe.routed_experts._EXTRA_CTX",
                SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER),
            ),
            patch("torch_npu.npu_grouped_matmul", return_value=["gate_up_out"], create=True) as mock_gmm,
        ):
            out = method.apply_gmm1(mlp_compute_input)
        self.assertEqual(out, "gate_up_out")
        self.assertEqual(mock_gmm.call_args.kwargs["weight"][0].shape, torch.Size([2, 16, 8]))

    def test_apply_act_quant_applies_topk_scales(self):
        method = self._make_method()
        mlp_compute_input = _mlp_compute_input(topk_scales=torch.tensor([0.5]))
        x = torch.tensor([[2.0, 4.0]])
        out, scale = method.apply_act_quant(mlp_compute_input, x)
        self.assertTrue(torch.equal(out, torch.tensor([[1.0, 2.0]])))
        self.assertIsNone(scale)

    def test_apply_gmm2_runs_down_proj(self):
        method = self._make_method()
        layer = SimpleNamespace(
            w13_weight=torch.randn(2, 8, 16),
            w2_weight=torch.randn(2, 16, 8),
            w13_bias=None,
            w2_bias=None,
        )
        mlp_compute_input = _mlp_compute_input(layer=layer)
        with (
            patch(
                "vllm_ascend.ops.fused_moe.routed_experts._EXTRA_CTX",
                SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER),
            ),
            patch("torch_npu.npu_grouped_matmul", return_value=["final_out"], create=True) as mock_gmm,
        ):
            out = method.apply_gmm2(mlp_compute_input, "act_out", None)
        self.assertEqual(out, "final_out")
        self.assertEqual(mock_gmm.call_args.kwargs["weight"][0].shape, torch.Size([2, 16, 8]))


class TestApplyMoeMlp(unittest.TestCase):
    def _quant_method(self, fused: bool):
        quant_method = MagicMock()
        quant_method.supports_fused_activation.return_value = fused
        quant_method.apply_gmm1_act_quant.return_value = ("fused_out", "fused_scale")
        quant_method.apply_gmm1.return_value = "gmm1_out"
        quant_method.apply_act_quant.return_value = ("act_quant_out", "act_scale")
        quant_method.apply_gmm2.return_value = "final_out"
        return quant_method

    def test_fused_path_skips_separate_activation(self):
        quant_method = self._quant_method(fused=True)
        mlp_compute_input = _mlp_compute_input()
        stream = MagicMock()
        stream.record_event.return_value = "evt"
        with patch(f"{MOE_MLP}.torch.npu.current_stream", return_value=stream):
            out, evt = apply_moe_mlp(mlp_compute_input, quant_method)

        self.assertEqual((out, evt), ("final_out", "evt"))
        quant_method.supports_fused_activation.assert_called_once_with(mlp_compute_input.activation)
        quant_method.apply_gmm1_act_quant.assert_called_once_with(mlp_compute_input)
        quant_method.apply_gmm1.assert_not_called()
        quant_method.apply_act_quant.assert_not_called()
        quant_method.apply_gmm2.assert_called_once_with(mlp_compute_input, "fused_out", "fused_scale")

    def test_non_fused_path_runs_gmm1_activation_quant_gmm2(self):
        quant_method = self._quant_method(fused=False)
        mlp_compute_input = _mlp_compute_input()
        with (
            patch(f"{MOE_MLP}.torch.npu.current_stream", MagicMock()),
            patch(f"{MOE_MLP}.torch_npu.npu_swiglu", return_value="silu_out", create=True),
        ):
            out, _ = apply_moe_mlp(mlp_compute_input, quant_method)

        self.assertEqual(out, "final_out")
        quant_method.apply_gmm1.assert_called_once_with(mlp_compute_input)
        quant_method.apply_act_quant.assert_called_once_with(mlp_compute_input, "silu_out")
        quant_method.apply_gmm2.assert_called_once_with(mlp_compute_input, "act_quant_out", "act_scale")

    def test_before_gmm2_event_recorded_between_act_quant_and_gmm2(self):
        quant_method = self._quant_method(fused=False)
        mlp_compute_input = _mlp_compute_input()
        calls = []

        def _act_quant(*args, **kwargs):
            calls.append("act_quant")
            return "x", None

        def _gmm2(*args, **kwargs):
            calls.append("gmm2")
            return "y"

        def _record_event():
            calls.append("record_event")
            return "evt"

        quant_method.apply_act_quant.side_effect = _act_quant
        quant_method.apply_gmm2.side_effect = _gmm2
        stream = MagicMock()
        stream.record_event.side_effect = _record_event
        with (
            patch(f"{MOE_MLP}.torch.npu.current_stream", return_value=stream),
            patch(f"{MOE_MLP}.torch_npu.npu_swiglu", return_value="x", create=True),
        ):
            apply_moe_mlp(mlp_compute_input, quant_method)

        self.assertEqual(calls, ["act_quant", "record_event", "gmm2"])

    def test_lora_branch_dispatches_quantized_lora_backend(self):
        quant_method = self._quant_method(fused=False)
        mlp_compute_input = _mlp_compute_input(
            quant=MoEQuantParams(quant_type=QuantType.W8A8),
            lora_context=SimpleNamespace(),
        )
        with (
            patch("vllm_ascend.lora.fused_moe.has_lora", return_value=True),
            patch(
                "vllm_ascend.lora.quant_moe.quant_apply_mlp_with_moe_lora",
                return_value=("lora_out", "lora_evt"),
            ) as mock_lora,
        ):
            out, evt = apply_moe_mlp(mlp_compute_input, quant_method)

        self.assertEqual((out, evt), ("lora_out", "lora_evt"))
        mock_lora.assert_called_once_with(mlp_compute_input=mlp_compute_input, quant_method=quant_method)
        quant_method.apply_gmm1_act_quant.assert_not_called()
        quant_method.apply_gmm1.assert_not_called()

    def test_lora_branch_skips_quantized_mlp_without_active_lora(self):
        quant_method = self._quant_method(fused=True)
        mlp_compute_input = _mlp_compute_input(
            quant=MoEQuantParams(quant_type=QuantType.W8A8),
            lora_context=SimpleNamespace(),
        )
        with (
            patch("vllm_ascend.lora.fused_moe.has_lora", return_value=False),
            patch("vllm_ascend.lora.quant_moe.quant_apply_mlp_with_moe_lora") as mock_lora,
            patch(f"{MOE_MLP}.torch.npu.current_stream", MagicMock()),
        ):
            out, _ = apply_moe_mlp(mlp_compute_input, quant_method)

        self.assertEqual(out, "final_out")
        mock_lora.assert_not_called()
        quant_method.apply_gmm1_act_quant.assert_called_once_with(mlp_compute_input)

    def test_lora_branch_skips_unquant_mlp(self):
        quant_method = self._quant_method(fused=False)
        mlp_compute_input = _mlp_compute_input(lora_context=SimpleNamespace())
        with (
            patch("vllm_ascend.lora.fused_moe.has_lora", return_value=True),
            patch("vllm_ascend.lora.quant_moe.quant_apply_mlp_with_moe_lora") as mock_lora,
            patch(f"{MOE_MLP}.torch.npu.current_stream", MagicMock()),
            patch(f"{MOE_MLP}.torch_npu.npu_swiglu", return_value="silu_out", create=True),
        ):
            out, _ = apply_moe_mlp(mlp_compute_input, quant_method)

        self.assertEqual(out, "final_out")
        mock_lora.assert_not_called()


class TestUnifiedApplyActivation(unittest.TestCase):
    def _quant_method(self):
        quant_method = MagicMock()
        quant_method.get_mlp_weights.return_value = (torch.randn(2, 8, 16), torch.randn(2, 16, 8))
        return quant_method

    def test_gelu_matches_torch_reference(self):
        hidden_states = torch.randn(4, 8)
        out = _unified_apply_activation(
            _mlp_compute_input(activation=MoEActivation.GELU), hidden_states.clone(), self._quant_method()
        )
        gate, up = hidden_states.chunk(2, dim=-1)
        self.assertTrue(torch.allclose(out, torch.nn.functional.gelu(gate) * up))

    def test_gelu_tanh_matches_torch_reference(self):
        hidden_states = torch.randn(4, 8)
        out = _unified_apply_activation(
            _mlp_compute_input(activation=MoEActivation.GELU_TANH), hidden_states.clone(), self._quant_method()
        )
        gate, up = hidden_states.chunk(2, dim=-1)
        self.assertTrue(torch.allclose(out, torch.nn.functional.gelu(gate, approximate="tanh") * up))

    def test_situ_matches_reference(self):
        hidden_states = torch.randn(4, 8)
        beta, linear_beta = 4.0, 25.0
        out = _unified_apply_activation(
            _mlp_compute_input(
                activation=MoEActivation.SITU,
                activation_situ_beta=beta,
                activation_situ_linear_beta=linear_beta,
            ),
            hidden_states.clone(),
            self._quant_method(),
        )
        gate, up = hidden_states.chunk(2, dim=-1)
        gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        up = linear_beta * torch.tanh(up / linear_beta)
        self.assertTrue(torch.allclose(out, gate * up))

    def test_swiglustep_uses_hardcoded_limit(self):
        with patch(f"{MOE_MLP}.AscendSwigluStepAndMul.swiglustep_forward", return_value="out") as mock_act:
            out = _unified_apply_activation(
                _mlp_compute_input(activation=MoEActivation.SWIGLUSTEP, swiglu_limit=5.0),
                "x",
                self._quant_method(),
            )
        self.assertEqual(out, "out")
        mock_act.assert_called_once_with("x", limit=7.0)

    def test_swigluoai_uninterleave_uses_clipped_swiglu(self):
        with patch(f"{MOE_MLP}.DeviceOperator.clipped_swiglu", return_value="out") as mock_act:
            out = _unified_apply_activation(
                _mlp_compute_input(
                    activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
                    swiglu_limit=3.0,
                    swiglu_alpha=1.5,
                    swiglu_beta=0.25,
                ),
                "x",
                self._quant_method(),
            )
        self.assertEqual(out, "out")
        mock_act.assert_called_once_with("x", swiglu_limit=3.0, swiglu_alpha=1.5, swiglu_beta=0.25)

    def test_silu_default_uses_npu_swiglu(self):
        with patch(f"{MOE_MLP}.torch_npu.npu_swiglu", return_value="out", create=True) as mock_act:
            out = _unified_apply_activation(_mlp_compute_input(), "x", self._quant_method())
        self.assertEqual(out, "out")
        mock_act.assert_called_once_with("x")

    def test_silu_clamped_when_limit_set(self):
        x = torch.randn(4, 8)
        with patch(f"{MOE_MLP}.torch_npu.npu_swiglu", return_value="out", create=True) as mock_act:
            _unified_apply_activation(_mlp_compute_input(swiglu_limit=2.0), x.clone(), self._quant_method())
        gate, up = mock_act.call_args.args[0].chunk(2, dim=-1)
        self.assertLessEqual(gate.max().item(), 2.0)
        self.assertLessEqual(up.abs().max().item(), 2.0)

    def test_swigluoai_uses_oai_forward(self):
        with patch(f"{MOE_MLP}.AscendSwigluOAIAndMul.swiglu_oai_forward", return_value="out") as mock_act:
            out = _unified_apply_activation(
                _mlp_compute_input(activation=MoEActivation.SWIGLUOAI),
                torch.randn(2, 32),
                self._quant_method(),
            )
        self.assertEqual(out, "out")
        mock_act.assert_called_once()


if __name__ == "__main__":
    unittest.main()
