import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401 -- registers torch.npu used by the module under test

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput, MoEQuantParams, MoEWeights
from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
from vllm_ascend.quantization.methods.w4a16 import AscendW4A16FusedMoEMethod
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType

MXFP4_TEST_DTYPE = getattr(torch, "float4_e2m1fn_x2", torch.float16)


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
            patch("vllm_ascend.ops.fused_moe.moe_utils.enable_custom_op", return_value=True),
            patch(
                "torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list",
                return_value=("out", "scale", None),
            ) as mock_op,
            patch.object(DeviceOperator, "npu_dynamic_quant", return_value=("qx", "pscale")),
        ):
            out, scale = method.apply_gmm1_act_quant(mlp_compute_input)
        self.assertEqual((out, scale), ("out", "scale"))
        mock_op.assert_called_once()
        self.assertEqual(mock_op.call_args.kwargs["weight"], [layer.w13_weight])

    def test_apply_gmm1_act_quant_fused_op_path(self):
        method = self._make_method()
        layer = _w8a8_layer()
        mlp_compute_input = _mlp_compute_input(layer=layer, fusion=True, dynamic_eplb=False)
        with (
            patch.object(
                DeviceOperator, "npu_grouped_matmul_swiglu_quant", return_value=("out", "scale", None)
            ) as mock_fused,
            patch.object(DeviceOperator, "npu_dynamic_quant", return_value=("qx", "pscale")),
        ):
            out, scale = method.apply_gmm1_act_quant(mlp_compute_input)
        self.assertEqual((out, scale), ("out", "scale"))
        mock_fused.assert_called_once()
        self.assertEqual(mock_fused.call_args.kwargs["act_quant_type"], torch.int8)

    def test_apply_gmm1_act_quant_swigluoai_dequant_path(self):
        method = self._make_method()
        layer = _w8a8_layer()
        mlp_compute_input = _mlp_compute_input(
            layer=layer,
            fusion=False,
            activation="swigluoai_uninterleave",
            swiglu_limit=3.0,
            swiglu_alpha=1.5,
            swiglu_beta=0.25,
        )
        with (
            patch("torch_npu.npu_grouped_matmul", return_value=["int32_out"], create=True) as mock_gmm,
            patch("torch.ops._C_ascend.npu_dequant_swiglu_quant", return_value=("out", "scale")) as mock_dequant,
            patch.object(DeviceOperator, "npu_dynamic_quant", return_value=("qx", "pscale")),
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
            patch("torch_npu.npu_grouped_matmul", return_value=["gmm1_out"], create=True) as mock_gmm,
            patch("vllm_ascend.quantization.methods.w8a8_dynamic.HAS_TRITON", False),
            patch("torch_npu.npu_swiglu", return_value="silu_out", create=True),
            patch("torch_npu.npu_dynamic_quant", return_value=("quant_out", "qscale"), create=True),
            patch.object(DeviceOperator, "npu_dynamic_quant", return_value=("qx", "pscale")),
        ):
            out, scale = method.apply_gmm1_act_quant(mlp_compute_input)
        self.assertEqual((out, scale), ("quant_out", "qscale"))
        mock_gmm.assert_called_once()
        self.assertEqual(mock_gmm.call_args.kwargs["weight"], [layer.w13_weight])

    def test_apply_gmm1_uses_soft_quant_matmul(self):
        method = self._make_method()
        layer = _w8a8_layer()
        mlp_compute_input = _mlp_compute_input(layer=layer)
        with (
            patch("torch_npu.npu_grouped_matmul", return_value=["out"], create=True) as mock_gmm,
            patch.object(DeviceOperator, "npu_dynamic_quant", return_value=("qx", "pscale")),
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
            patch.object(DeviceOperator, "npu_dynamic_quant", return_value=("quant_out", "qscale")),
            patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value="final_out") as mock_gmm2,
        ):
            out, scale = method.apply_act_quant(mlp_compute_input, "x")
            final = method.apply_gmm2(mlp_compute_input, "quant_out", scale)
        self.assertEqual((out, scale), ("quant_out", "qscale"))
        self.assertEqual(final, "final_out")
        mock_gmm2.assert_called_once()
        self.assertEqual(mock_gmm2.call_args.kwargs["weight"], [layer.w2_weight])
        self.assertEqual(mock_gmm2.call_args.kwargs["act_quant_type"], torch.int8)

    def test_get_moe_weights_single_tensor_form(self):
        method = self._make_method()
        layer = _w8a8_layer()
        with patch(
            "vllm_ascend.quantization.methods.w8a8_dynamic._EXTRA_CTX",
            SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER),
        ):
            weights = method.get_moe_weights(layer)
        self.assertEqual(weights.w1, [layer.w13_weight])
        self.assertEqual(weights.w1_scale, [layer.w13_weight_scale_fp32])
        self.assertEqual(weights.w2_scale, [layer.w2_weight_scale])
        self.assertIsNone(weights.w1_scale_bias)

    def test_get_moe_weights_fused_mc2_scale_flag(self):
        method = self._make_method()
        layer = SimpleNamespace(
            w13_weight=torch.randn(1, 8, 16),
            fused_w1_scale=torch.randn(1, 8),
            w2_weight=torch.randn(1, 16, 8),
            fused_w2_scale=torch.randn(1, 16),
            activation="silu",
        )
        with (
            patch(
                "vllm_ascend.quantization.methods.w8a8_dynamic._EXTRA_CTX",
                SimpleNamespace(moe_comm_type=MoECommType.FUSED_MC2),
            ),
            patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_ascend_config") as mock_config,
        ):
            mock_config.return_value.enable_fused_mc2 = 1
            weights = method.get_moe_weights(layer)
        self.assertEqual(weights.w1_scale, [layer.fused_w1_scale])
        self.assertEqual(weights.w1_scale_bias, [torch.tensor([], dtype=torch.float32)])

    def test_get_moe_weights_expert_list_form(self):
        method = self._make_method(use_expert_weight_list=True)
        layer = SimpleNamespace(
            w13_weight_list=[torch.randn(8, 16)],
            w13_weight_scale_fp32_list=[torch.randn(8)],
            w2_weight_list=[torch.randn(16, 8)],
            w2_weight_scale_list=[torch.randn(16)],
            activation="silu",
        )
        with patch(
            "vllm_ascend.quantization.methods.w8a8_dynamic._EXTRA_CTX",
            SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER),
        ):
            weights = method.get_moe_weights(layer)
        self.assertEqual(weights.w1, layer.w13_weight_list)
        self.assertEqual(weights.w1_scale, layer.w13_weight_scale_fp32_list)


class TestW4A16FusedMoEMethod(unittest.TestCase):
    def _make_method(self):
        return AscendW4A16FusedMoEMethod.__new__(AscendW4A16FusedMoEMethod)

    def test_apply_gmm1_and_gmm2_use_antiquant_offsets(self):
        method = self._make_method()
        layer = SimpleNamespace(
            w13_weight_packed=torch.randn(2, 8, 4, dtype=torch.int32),
            w13_weight_scale=torch.randn(2, 8, 4),
            w13_weight_offset=torch.randn(2, 8, 4),
            w2_weight_packed=torch.randn(2, 4, 8, dtype=torch.int32),
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
        self.assertEqual(gmm1_kwargs["weight"], [layer.w13_weight_packed])
        self.assertEqual(gmm1_kwargs["antiquant_offset"], [layer.w13_weight_offset])
        self.assertEqual(gmm2_kwargs["weight"], [layer.w2_weight_packed])
        self.assertEqual(gmm2_kwargs["antiquant_offset"], [layer.w2_weight_offset])

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


if __name__ == "__main__":
    unittest.main()
