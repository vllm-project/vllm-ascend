import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch_npu  # noqa: F401 -- registers torch.npu used by the module under test
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ops.fused_moe.moe_activation import (
    GeluMoeActionMethod,
    GeluTanhMoeActionMethod,
    SiluMoeActionMethod,
    SwigluOaiUninterleaveMoeActionMethod,
    SwigluStepMoeActionMethod,
    get_moe_activation_method,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput, MoEQuantParams, MoEWeights

MOE_ACTIVATION = "vllm_ascend.ops.fused_moe.moe_activation"


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


class TestGetMoeActivationMethod(unittest.TestCase):
    def test_dispatch_by_enum(self):
        self.assertIsInstance(get_moe_activation_method(MoEActivation.GELU), GeluMoeActionMethod)
        self.assertIsInstance(get_moe_activation_method(MoEActivation.GELU_TANH), GeluTanhMoeActionMethod)
        self.assertIsInstance(get_moe_activation_method(MoEActivation.SWIGLUSTEP), SwigluStepMoeActionMethod)
        self.assertIsInstance(
            get_moe_activation_method(MoEActivation.SWIGLUOAI_UNINTERLEAVE),
            SwigluOaiUninterleaveMoeActionMethod,
        )
        self.assertIsInstance(get_moe_activation_method(MoEActivation.SILU), SiluMoeActionMethod)

    def test_dispatch_by_string(self):
        self.assertIsInstance(get_moe_activation_method("gelu"), GeluMoeActionMethod)
        self.assertIsInstance(get_moe_activation_method("swiglustep"), SwigluStepMoeActionMethod)
        self.assertIsInstance(get_moe_activation_method("silu"), SiluMoeActionMethod)

    def test_unknown_activation_falls_back_to_default(self):
        self.assertIsInstance(get_moe_activation_method("unknown_act"), SiluMoeActionMethod)


class TestMoeActionMethodOrchestration(unittest.TestCase):
    def _quant_method(self, fused: bool):
        quant_method = MagicMock()
        quant_method.supports_fused_activation.return_value = fused
        quant_method.apply_gmm1_act_quant.return_value = ("fused_out", "fused_scale")
        quant_method.apply_gmm1.return_value = "gmm1_out"
        quant_method.apply_act_quant.return_value = ("act_quant_out", "act_scale")
        quant_method.apply_gmm2.return_value = "final_out"
        return quant_method

    @patch("torch.npu.current_stream", MagicMock())
    def test_fused_path_skips_separate_activation(self):
        quant_method = self._quant_method(fused=True)
        mlp_compute_input = _mlp_compute_input()
        stream = MagicMock()
        stream.record_event.return_value = "evt"
        with patch("torch.npu.current_stream", return_value=stream):
            out, evt = SiluMoeActionMethod().apply_mlp(mlp_compute_input, quant_method)

        self.assertEqual(out, "final_out")
        self.assertEqual(evt, "evt")
        quant_method.apply_gmm1_act_quant.assert_called_once_with(mlp_compute_input)
        quant_method.apply_gmm1.assert_not_called()
        quant_method.apply_act_quant.assert_not_called()
        quant_method.apply_gmm2.assert_called_once_with(mlp_compute_input, "fused_out", "fused_scale")

    @patch("torch.npu.current_stream", MagicMock())
    def test_non_fused_path_runs_gmm1_activation_quant_gmm2(self):
        quant_method = self._quant_method(fused=False)
        mlp_compute_input = _mlp_compute_input()
        stream = MagicMock()
        stream.record_event.return_value = "evt"
        with (
            patch("torch.npu.current_stream", return_value=stream),
            patch(f"{MOE_ACTIVATION}.torch_npu.npu_swiglu", return_value="silu_out", create=True),
        ):
            out, evt = SiluMoeActionMethod().apply_mlp(mlp_compute_input, quant_method)

        self.assertEqual(out, "final_out")
        quant_method.apply_gmm1.assert_called_once_with(mlp_compute_input)
        quant_method.apply_act_quant.assert_called_once_with(mlp_compute_input, "silu_out")
        quant_method.apply_gmm2.assert_called_once_with(mlp_compute_input, "act_quant_out", "act_scale")

    @patch("torch.npu.current_stream", MagicMock())
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
            patch("torch.npu.current_stream", return_value=stream),
            patch(f"{MOE_ACTIVATION}.torch_npu.npu_swiglu", return_value="x", create=True),
        ):
            SiluMoeActionMethod().apply_mlp(mlp_compute_input, quant_method)

        self.assertEqual(calls, ["act_quant", "record_event", "gmm2"])


class TestActivationMath(unittest.TestCase):
    def test_gelu_matches_torch_reference(self):
        hidden_states = torch.randn(4, 8)
        out = GeluMoeActionMethod().apply_activation(_mlp_compute_input(), hidden_states.clone())
        gate, up = hidden_states.chunk(2, dim=-1)
        self.assertTrue(torch.allclose(out, torch.nn.functional.gelu(gate) * up))

    def test_gelu_tanh_matches_torch_reference(self):
        hidden_states = torch.randn(4, 8)
        out = GeluTanhMoeActionMethod().apply_activation(_mlp_compute_input(), hidden_states.clone())
        gate, up = hidden_states.chunk(2, dim=-1)
        self.assertTrue(torch.allclose(out, torch.nn.functional.gelu(gate, approximate="tanh") * up))

    def test_swiglustep_passes_limit(self):
        with patch(f"{MOE_ACTIVATION}.AscendSwigluStepAndMul.swiglustep_forward", return_value="out") as mock_act:
            out = SwigluStepMoeActionMethod().apply_activation(_mlp_compute_input(swiglu_limit=5.0), "x")
        self.assertEqual(out, "out")
        mock_act.assert_called_once_with("x", limit=5.0)

    def test_swiglustep_default_limit(self):
        with patch(f"{MOE_ACTIVATION}.AscendSwigluStepAndMul.swiglustep_forward", return_value="out") as mock_act:
            SwigluStepMoeActionMethod().apply_activation(_mlp_compute_input(swiglu_limit=0.0), "x")
        mock_act.assert_called_once_with("x", limit=7.0)

    def test_swigluoai_uninterleave_uses_clipped_swiglu(self):
        with patch(f"{MOE_ACTIVATION}.torch_npu.npu_clipped_swiglu", return_value="out", create=True) as mock_act:
            out = SwigluOaiUninterleaveMoeActionMethod().apply_activation(
                _mlp_compute_input(swiglu_limit=3.0, swiglu_alpha=1.5, swiglu_beta=0.25), "x"
            )
        self.assertEqual(out, "out")
        mock_act.assert_called_once_with("x", interleaved=False, alpha=1.5, limit=3.0, bias=0.25)

    def test_silu_default_uses_npu_swiglu(self):
        with patch(f"{MOE_ACTIVATION}.torch_npu.npu_swiglu", return_value="out", create=True) as mock_act:
            out = SiluMoeActionMethod().apply_activation(_mlp_compute_input(), "x")
        self.assertEqual(out, "out")
        mock_act.assert_called_once_with("x")

    def test_silu_clamped_when_limit_set(self):
        x = torch.randn(4, 8)
        with patch(f"{MOE_ACTIVATION}.torch_npu.npu_swiglu", return_value="out", create=True) as mock_act:
            SiluMoeActionMethod().apply_activation(_mlp_compute_input(swiglu_limit=2.0), x.clone())
        gate, up = mock_act.call_args.args[0].chunk(2, dim=-1)
        self.assertLessEqual(gate.max().item(), 2.0)
        self.assertLessEqual(up.abs().max().item(), 2.0)

    def test_silu_swigluoai_uses_oai_forward(self):
        layer = SimpleNamespace(w13_weight=torch.randn(2, 8, 16))
        mlp_compute_input = _mlp_compute_input(activation="swigluoai", layer=layer)
        x = torch.randn(2, 32)
        with patch(f"{MOE_ACTIVATION}.AscendSwigluOAIAndMul.swiglu_oai_forward", return_value="out") as mock_act:
            out = SiluMoeActionMethod().apply_activation(mlp_compute_input, x)
        self.assertEqual(out, "out")
        mock_act.assert_called_once()


if __name__ == "__main__":
    unittest.main()
