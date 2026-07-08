"""Unit tests for GELU activation support in quantized MoE MLP.

These tests cover the GELU early-return path added to ``quant_apply_mlp``
(PR #11609). ``quant_apply_mlp`` previously hardcoded SwiGLU, so quantized MoE
experts of GELU models (e.g. Gemma4, ``activation='gelu_tanh'``) ran the wrong
activation. The new path runs dequant GMM -> GELU -> (re)quant -> GMM2 and
returns before any SwiGLU branch.

All NPU operators are mocked, so the tests run on CPU without Ascend hardware.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch_npu  # noqa: F401  -- registers torch.npu used by the module under test
from torch.nn import functional as F
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe import moe_mlp as moe_mlp_mod
from vllm_ascend.ops.fused_moe.moe_mlp import quant_apply_mlp
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEQuantParams,
    MoEWeights,
)
from vllm_ascend.quantization.quant_type import QuantType

MOE_MLP = "vllm_ascend.ops.fused_moe.moe_mlp"


def _patch_npu_stream():
    """Patch ``torch.npu.current_stream`` so ``record_event()`` returns a tag."""
    evt = MagicMock(name="before_gmm2_evt")
    stream = MagicMock(name="npu_stream")
    stream.record_event.return_value = evt
    return patch("torch.npu.current_stream", return_value=stream), evt


class _GeluPathBase(unittest.TestCase):
    """Common helpers for the GELU-path tests."""

    def _common_w8a8_kwargs(self, *, activation, w1_scale_dtype=torch.float32,
                            w2_scale_dtype=torch.float32,
                            w1_scale_bias=None, w2_scale_bias=None,
                            group_list_type=1, group_list=None,
                            dynamic_scale=None):
        return dict(
            hidden_states=torch.randn(1, 4),
            w1=torch.randn(1, 8, 4),
            w1_scale=[torch.randn(1, 8, dtype=w1_scale_dtype)],
            w2=torch.randn(1, 4, 1),
            w2_scale=[torch.randn(1, 4, dtype=w2_scale_dtype)],
            group_list=group_list if group_list is not None
            else torch.tensor([1], dtype=torch.int64),
            group_list_type=group_list_type,
            dynamic_scale=dynamic_scale if dynamic_scale is not None
            else torch.randn(1, 1),
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


class TestQuantApplyMlpGeluPath(_GeluPathBase):
    """GELU early-return path: dispatch, math, and layout coverage."""

    def test_w8a8_gelu_tanh_applies_correct_activation(self):
        """W8A8 + gelu_tanh: GMM1(dequant) -> gelu(tanh)·up -> requant -> GMM2."""
        gate = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
        up = torch.tensor([[0.5, -0.5, 1.0, 2.0]])
        gate_up = torch.cat([gate, up], dim=-1)
        expected = F.gelu(gate, approximate="tanh") * up

        captured = {}

        def fake_dynamic_quant(x):
            captured["x"] = x.detach().clone()
            scale = torch.ones(1, dtype=torch.float32)
            captured["scale"] = scale
            return x, scale

        gmm2_out = torch.tensor([[9.0]])
        stream_patch, evt = _patch_npu_stream()

        with stream_patch, \
                patch("torch_npu.npu_grouped_matmul", return_value=[gate_up]) as mock_gmm1, \
                patch("torch_npu.npu_dynamic_quant", side_effect=fake_dynamic_quant) as mock_dq, \
                patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=gmm2_out) as mock_gmm2, \
                patch(f"{MOE_MLP}.dispose_tensor") as mock_dispose:
            out, out_evt = quant_apply_mlp(**self._common_w8a8_kwargs(
                activation=MoEActivation.GELU_TANH))

        # GELU math applied with tanh approximation before requantization.
        self.assertTrue(torch.allclose(captured["x"], expected, atol=1e-6))
        # GMM1 used the dequant form (scale + per_token_scale), not antiquant.
        gmm1_kwargs = mock_gmm1.call_args.kwargs
        self.assertIn("scale", gmm1_kwargs)
        self.assertIn("per_token_scale", gmm1_kwargs)
        self.assertNotIn("antiquant_scale", gmm1_kwargs)
        self.assertEqual(gmm1_kwargs["split_item"], 2)
        # Requant + GMM2 both invoked.
        mock_dq.assert_called_once()
        mock_gmm2.assert_called_once()
        # GMM2 received the requant per-token scale returned by dynamic_quant.
        self.assertIs(mock_gmm2.call_args.kwargs["per_token_scale"],
                      captured["scale"])
        # Return contract: (hidden_states, before_gmm2_evt).
        self.assertIs(out, gmm2_out)
        self.assertIs(out_evt, evt)

    def test_w8a8_gelu_uses_exact_gelu_approximation(self):
        """W8A8 + gelu (not tanh): approximate='none', matching the float path."""
        gate = torch.tensor([[0.5, -0.5, 2.0]])
        up = torch.tensor([[1.0, 1.0, 0.5]])
        gate_up = torch.cat([gate, up], dim=-1)
        expected = F.gelu(gate, approximate="none") * up

        captured = {}

        def fake_dynamic_quant(x):
            captured["x"] = x.detach().clone()
            return x, torch.ones(1, dtype=torch.float32)

        stream_patch, _ = _patch_npu_stream()
        with stream_patch, \
                patch("torch_npu.npu_grouped_matmul", return_value=[gate_up]), \
                patch("torch_npu.npu_dynamic_quant", side_effect=fake_dynamic_quant), \
                patch.object(DeviceOperator, "npu_grouped_matmul_gmm2",
                             return_value=torch.zeros(1, 3)), \
                patch(f"{MOE_MLP}.dispose_tensor"):
            quant_apply_mlp(**self._common_w8a8_kwargs(
                activation=MoEActivation.GELU))

        # exact GELU (approximate='none') differs from tanh; ensure 'none' used.
        self.assertFalse(torch.allclose(
            captured["x"], F.gelu(gate, approximate="tanh") * up, atol=1e-6))
        self.assertTrue(torch.allclose(captured["x"], expected, atol=1e-6))

    def test_w4a16_gelu_uses_antiquant_path(self):
        """W4A16 + gelu: antiquant GMM1 -> gelu·up -> antiquant GMM2, no requant."""
        gate = torch.tensor([[1.0, -1.0]])
        up = torch.tensor([[0.5, 2.0]])
        gate_up = torch.cat([gate, up], dim=-1)
        expected = F.gelu(gate, approximate="tanh") * up
        gmm2_out = torch.tensor([[3.0]])

        stream_patch, evt = _patch_npu_stream()
        with stream_patch, \
                patch("torch_npu.npu_grouped_matmul",
                      side_effect=[[gate_up], [gmm2_out]]) as mock_gmm, \
                patch("torch_npu.npu_dynamic_quant") as mock_dq, \
                patch.object(DeviceOperator, "npu_grouped_matmul_gmm2") as mock_gmm2, \
                patch(f"{MOE_MLP}.dispose_tensor"):
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
        gate = torch.tensor([[0.5, -0.5]])
        up = torch.tensor([[1.0, 1.0]])
        gate_up = torch.cat([gate, up], dim=-1)
        w1_sb = [torch.zeros(1)]
        w2_sb = [torch.zeros(1)]

        stream_patch, _ = _patch_npu_stream()
        with stream_patch, \
                patch("torch_npu.npu_grouped_matmul", return_value=[gate_up]) as mock_gmm1, \
                patch("torch_npu.npu_dynamic_quant",
                      side_effect=lambda x: (x, torch.ones(1))), \
                patch.object(DeviceOperator, "npu_grouped_matmul_gmm2",
                             return_value=torch.zeros(1, 2)), \
                patch(f"{MOE_MLP}.dispose_tensor"), \
                patch("torch.cat", wraps=torch.cat) as mock_cat:
            quant_apply_mlp(**self._common_w8a8_kwargs(
                activation=MoEActivation.GELU_TANH,
                w1_scale_bias=w1_sb, w2_scale_bias=w2_sb,
                group_list_type=0,
                group_list=torch.tensor([0, 1], dtype=torch.int64)))

        gmm1_kwargs = mock_gmm1.call_args.kwargs
        # bias1 propagated to GMM1.
        self.assertIs(gmm1_kwargs["bias"], w1_sb)
        # group_list_type 0 -> 1 conversion invoked (torch.cat + torch.diff).
        self.assertTrue(mock_cat.called)

    def test_w8a8_gelu_converts_w1_scale_dtype_to_output_dtype(self):
        """When w1_scale dtype != _output_dtype, it is cast before GMM1."""
        gate_up = torch.zeros(1, 8)
        stream_patch, _ = _patch_npu_stream()
        with stream_patch, \
                patch("torch_npu.npu_grouped_matmul", return_value=[gate_up]) as mock_gmm1, \
                patch("torch_npu.npu_dynamic_quant",
                      side_effect=lambda x: (x, torch.ones(1))), \
                patch.object(DeviceOperator, "npu_grouped_matmul_gmm2",
                             return_value=torch.zeros(1, 4)), \
                patch(f"{MOE_MLP}.dispose_tensor"):
            # w1_scale fp32, w2_scale bf16 -> _output_dtype = bfloat16, so the
            # GELU path must cast w1_scale to bfloat16 before GMM1.
            quant_apply_mlp(**self._common_w8a8_kwargs(
                activation=MoEActivation.GELU_TANH,
                w1_scale_dtype=torch.float32,
                w2_scale_dtype=torch.bfloat16))

        scale_arg = mock_gmm1.call_args.kwargs["scale"]
        self.assertEqual(scale_arg[0].dtype, torch.bfloat16)

    def test_gelu_path_does_not_call_swiglu_op(self):
        """GELU path must use torch.gelu, never the SwiGLU NPU op."""
        gate_up = torch.zeros(1, 8)
        stream_patch, _ = _patch_npu_stream()
        with stream_patch, \
                patch("torch_npu.npu_grouped_matmul", return_value=[gate_up]), \
                patch("torch_npu.npu_dynamic_quant",
                      side_effect=lambda x: (x, torch.ones(1))), \
                patch.object(DeviceOperator, "npu_grouped_matmul_gmm2",
                             return_value=torch.zeros(1, 4)), \
                patch(f"{MOE_MLP}.dispose_tensor"), \
                patch("torch_npu.npu_swiglu") as mock_swiglu:
            quant_apply_mlp(**self._common_w8a8_kwargs(
                activation=MoEActivation.GELU_TANH))
        mock_swiglu.assert_not_called()


class TestQuantApplyMlpNoGeluImpact(unittest.TestCase):
    """Non-GELU activations must NOT enter the GELU path (no regression)."""

    def _run_non_gelu(self, activation):
        gate_up = torch.zeros(1, 8)
        stream_patch, _ = _patch_npu_stream()
        with stream_patch, \
                patch(f"{MOE_MLP}._EXTRA_CTX") as mock_ctx, \
                patch(f"{MOE_MLP}.get_weight_prefetch_method", return_value=None), \
                patch(f"{MOE_MLP}.HAS_TRITON", False), \
                patch("torch_npu.npu_grouped_matmul", return_value=[gate_up]) as mock_gmm, \
                patch("torch_npu.npu_swiglu", return_value=torch.zeros(1, 4)) as mock_swiglu, \
                patch("torch_npu.npu_dynamic_quant",
                      side_effect=lambda x: (x, torch.ones(1))) as mock_dq, \
                patch.object(DeviceOperator, "npu_grouped_matmul_gmm2",
                             return_value=torch.zeros(1, 4)) as mock_gmm2, \
                patch(f"{MOE_MLP}.dispose_tensor"), \
                patch("torch.nn.functional.gelu") as mock_gelu:
            mock_ctx.moe_comm_type = -1  # not MoECommType.MC2
            quant_apply_mlp(
                hidden_states=torch.randn(1, 4),
                w1=[torch.randn(1, 8, 4)],
                w1_scale=[torch.randn(1, 8, dtype=torch.float32)],
                w2=[torch.randn(1, 4, 1)],
                w2_scale=[torch.randn(1, 4, dtype=torch.float32)],
                group_list=torch.tensor([1], dtype=torch.int64),
                group_list_type=1,
                dynamic_scale=torch.randn(1, 1),
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


class TestUnifiedApplyMlpThreadsGeluActivation(unittest.TestCase):
    """unified_apply_mlp must forward the GELU activation to quant_apply_mlp."""

    def test_gelu_tanh_is_forwarded_to_quant_apply_mlp(self):
        hidden_states = torch.randn(2, 8)
        expected = torch.randn(2, 8)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=torch.tensor([2, 2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=torch.randn(2, 1),
            topk_scales=None,
            weights=MoEWeights(
                w1=[torch.randn(1, 16, 8)],
                w2=[torch.randn(1, 8, 8)],
                w1_scale=[torch.randn(1, 16)],
                w2_scale=[torch.randn(1, 8)],
            ),
            quant=MoEQuantParams(quant_type=QuantType.W8A8),
            fusion=False,
            activation=MoEActivation.GELU_TANH,
            need_trans=False,
            dynamic_eplb=False,
        )

        with patch(f"{MOE_MLP}.quant_apply_mlp", return_value=expected) as mock_quant, \
                patch(f"{MOE_MLP}.unquant_apply_mlp") as mock_unquant:
            out = moe_mlp_mod.unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertIs(out, expected)
        mock_unquant.assert_not_called()
        mock_quant.assert_called_once()
        self.assertEqual(mock_quant.call_args.kwargs["activation"],
                         MoEActivation.GELU_TANH)


if __name__ == "__main__":
    unittest.main()
