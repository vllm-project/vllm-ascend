# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU numerical and ordering regressions for MoonEP-only MLP adaptations."""

import ast
import dataclasses
import unittest
from pathlib import Path
from types import SimpleNamespace

import test_moonep_v030_contract as contracts
import torch

MOE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/fused_moe"


@dataclasses.dataclass(frozen=True)
class Input:
    layer: object
    weights: object
    hidden_states: torch.Tensor
    activation: str = "step"
    swiglu_limit: float | None = 0.0
    lora_context: object = None
    topk_scales: torch.Tensor | None = None


def step_reference(x, limit):
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate).clamp(max=limit) * up.clamp(-limit, limit)


class SlotMLPSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.inputs = []
        self.limits = []
        self.activation = SimpleNamespace(
            SILU="silu",
            SITU="situ",
            SWIGLUOAI="oai",
            SWIGLUOAI_UNINTERLEAVE="oai-uninterleave",
            SWIGLUSTEP="step",
            GELU="gelu",
            GELU_TANH="gelu-tanh",
        )

        def step(x, limit):
            self.events.append("step")
            self.limits.append(limit)
            return step_reference(x, limit)

        self.step = SimpleNamespace(swiglustep_forward=step)
        native_node = next(
            n
            for n in contracts.nodes(MOE / "moe_mlp.py")
            if isinstance(n, ast.FunctionDef) and n.name == "_unified_apply_activation"
        )
        native_ns = contracts.execute(
            [native_node],
            {
                "torch": torch,
                "MoEActivation": self.activation,
                "AscendSwigluStepAndMul": self.step,
                "torch_npu": SimpleNamespace(
                    npu_swiglu=lambda x: torch.nn.functional.silu(x.chunk(2, dim=-1)[0]) * x.chunk(2, dim=-1)[1]
                ),
            },
        )
        self.native_activation = native_ns["_unified_apply_activation"]
        selected = [
            n
            for n in contracts.nodes(MOE / "moonep_mlp.py")
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.Assign))
        ]

        def event():
            self.events.append("event")
            return "before-gmm2"

        self.ns = contracts.execute(
            selected,
            {
                "__name__": __name__,
                "dataclass": dataclasses.dataclass,
                "replace": dataclasses.replace,
                "MappingProxyType": contracts.MappingProxyType,
                "MoEActivation": self.activation,
                "AscendSwigluStepAndMul": self.step,
                "_unified_apply_activation": self.native_activation,
                "torch": SimpleNamespace(
                    npu=SimpleNamespace(current_stream=lambda: SimpleNamespace(record_event=event))
                ),
            },
        )
        self.original = SimpleNamespace(w13_weight=torch.full((2, 4), -100.0), w2_weight=torch.full((2, 2), -100.0))
        self.weights = SimpleNamespace(
            w1=torch.tensor([[4.0, -2.0, 3.0, -4.0], [1.0, 2.0, -1.0, 3.0]]),
            w2=torch.tensor([[1.0, 2.0], [-1.0, 3.0]]),
        )
        self.hidden = torch.tensor([[5.0, -2.0], [-3.0, 4.0]])

        def gmm1(mci):
            self.events.append("gmm1")
            self.inputs.append(mci)
            return mci.hidden_states @ mci.layer.w13_weight

        def act_quant(mci, hidden):
            self.events.append("act_quant")
            if mci.topk_scales is not None:
                hidden = hidden * mci.topk_scales
            return hidden, "scale"

        def gmm2(mci, hidden, scale):
            self.events.append("gmm2")
            self.inputs.append(mci)
            self.assertEqual(scale, "scale")
            return hidden @ mci.layer.w2_weight

        self.quant = SimpleNamespace(
            use_expert_weight_list=False,
            supports_fused_activation=lambda activation: False,
            apply_gmm1=gmm1,
            apply_act_quant=act_quant,
            apply_gmm2=gmm2,
        )

    def test_custom_step_limit_and_fallback_preserve_numeric_result(self):
        original_weight = self.original.w13_weight
        for limit in (None, 0.0, 2.5):
            with self.subTest(limit=limit):
                mci = Input(self.original, self.weights, self.hidden, swiglu_limit=limit)
                actual, event = self.ns["apply_slot_mlp"](mci, self.quant)
                expected = step_reference(self.hidden @ self.weights.w1, limit or 7.0) @ self.weights.w2
                torch.testing.assert_close(actual, expected)
                self.assertEqual(self.limits[-1], limit or 7.0)
                self.assertIsNone(event)
                self.assertIs(mci.layer, self.original)
                self.assertIs(self.original.w13_weight, original_weight)
                self.assertIs(self.inputs[-1].layer.weights, self.weights)

    def test_event_is_after_activation_quantization_and_before_gmm2(self):
        mci = Input(self.original, self.weights, self.hidden, swiglu_limit=2.5)
        _, event = self.ns["apply_slot_mlp"](mci, self.quant, record_before_gmm2=True)
        self.assertEqual(event, "before-gmm2")
        self.assertEqual(self.events, ["gmm1", "step", "act_quant", "event", "gmm2"])

    def test_unused_event_is_not_recorded(self):
        mci = Input(self.original, self.weights, self.hidden)
        _, event = self.ns["apply_slot_mlp"](mci, self.quant)
        self.assertIsNone(event)
        self.assertNotIn("event", self.events)

    def test_native_non_moonep_step_behavior_remains_fixed_default(self):
        mci = Input(self.original, self.weights, self.hidden, swiglu_limit=2.5)
        gate_up = self.hidden @ self.weights.w1
        native = self.native_activation(mci, gate_up, self.quant)
        torch.testing.assert_close(native, step_reference(gate_up, 7.0))
        self.assertEqual(self.limits, [7.0])
        moon, _ = self.ns["apply_slot_mlp"](mci, self.quant)
        self.assertFalse(torch.allclose(moon, native @ self.weights.w2))

    def test_silu_uses_native_activation_and_applies_topk_scale_once(self):
        scale = torch.tensor([[0.25], [0.75]])
        mci = Input(self.original, self.weights, self.hidden, activation="silu", swiglu_limit=2.5, topk_scales=scale)
        actual, _ = self.ns["apply_slot_mlp"](mci, self.quant)
        up = self.hidden @ self.weights.w1
        gate, linear = up.chunk(2, dim=-1)
        expected = torch.nn.functional.silu(gate.clamp(max=2.5)) * linear.clamp(-2.5, 2.5)
        torch.testing.assert_close(actual, (expected * scale) @ self.weights.w2)
        self.assertEqual(self.limits, [])

    def test_fused_hook_preserves_scale_and_does_not_repeat_activation(self):
        self.quant.supports_fused_activation = lambda activation: True

        def fused(mci):
            self.events.append("fused")
            self.inputs.append(mci)
            return torch.ones(2, 2), "scale"

        self.quant.apply_gmm1_act_quant = fused
        mci = Input(self.original, self.weights, self.hidden, activation="silu")
        actual, event = self.ns["apply_slot_mlp"](mci, self.quant, record_before_gmm2=True)
        torch.testing.assert_close(actual, torch.ones(2, 2) @ self.weights.w2)
        self.assertEqual(self.events, ["fused", "event", "gmm2"])
        self.assertEqual(event, "before-gmm2")
        self.assertEqual(self.limits, [])

    def test_lora_is_rejected_before_any_phase_or_event(self):
        mci = Input(self.original, self.weights, self.hidden, lora_context=object())
        with self.assertRaisesRegex(NotImplementedError, "LoRA"):
            self.ns["apply_slot_mlp"](mci, self.quant, record_before_gmm2=True)
        self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main()
