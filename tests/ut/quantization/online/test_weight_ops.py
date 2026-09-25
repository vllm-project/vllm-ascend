#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Unit tests for the online quantization weight kernels and cosine tool.

Everything here runs on CPU: the kernels under test are the pure-torch paths
(exactly what NPU-less environments and the numerical oracle consume), and
the NPU-only ``npu_dynamic_mx_quant`` wrapper carries an explicit skip.
"""

import math
import unittest

import torch
import torch.nn as nn

from tests.ut.base import TestBase
from vllm_ascend.quantization.online import accuracy
from vllm_ascend.quantization.online import weight_ops as wo

# Accuracy gate from the quantization acceptance spec: a weight-preserving
# 8-bit recipe must reconstruct with cosine >= 0.999.
COSINE_GATE = 0.999

INT8_MAX = 127
MX_GROUP_SIZE = 32


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Float64 cosine so the gate measures quantization error, not f32 dot noise."""
    a_flat = a.to(torch.float64).reshape(-1)
    b_flat = b.to(torch.float64).reshape(-1)
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm())).item()


def make_weight(out_features=512, in_features=512, seed=0, scale=0.02):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(out_features, in_features, generator=generator) * scale


def make_outlier_weight(out_features=512, in_features=512, seed=1):
    weight = make_weight(out_features, in_features, seed, scale=0.01)
    # Channel-spaced outliers: every per-channel amax is outlier-dominated,
    # which is the adversarial case for symmetric int8.
    weight[:, ::64] = 8.0
    return weight


class TestQuantizeWeightInt8PerChannel(TestBase):
    def test_layout_matches_w8a8_dynamic_convention(self):
        # W8A8_DYNAMIC loads int8 [out, in] plus one scale per output channel
        # (methods/w8a8/w8a8_dynamic.py get_perchannel_param).
        weight = make_weight()
        quantized, scale = wo.quantize_weight_int8_per_channel(weight)
        self.assertEqual(quantized.shape, (512, 512))
        self.assertEqual(quantized.dtype, torch.int8)
        self.assertEqual(scale.shape, (512,))
        self.assertEqual(scale.dtype, torch.float32)

    def test_symmetric_scale_is_amax_over_127(self):
        weight = make_weight(seed=2)
        _, scale = wo.quantize_weight_int8_per_channel(weight)
        expected = weight.to(torch.float32).abs().amax(dim=1) / INT8_MAX
        self.assertTrue(torch.allclose(scale, expected, rtol=0, atol=0))

    def test_reconstruction_cosine_random_matrix(self):
        weight = make_weight()
        quantized, scale = wo.quantize_weight_int8_per_channel(weight)
        reconstructed = wo.dequantize_weight_int8_per_channel(quantized, scale)
        self.assertGreaterEqual(cosine(reconstructed, weight), COSINE_GATE)

    def test_reconstruction_cosine_with_outliers(self):
        weight = make_outlier_weight()
        quantized, scale = wo.quantize_weight_int8_per_channel(weight)
        reconstructed = wo.dequantize_weight_int8_per_channel(quantized, scale)
        self.assertGreaterEqual(cosine(reconstructed, weight), COSINE_GATE)

    def test_quantized_values_are_in_range(self):
        weight = make_weight(seed=3) * 50.0
        quantized, _ = wo.quantize_weight_int8_per_channel(weight)
        self.assertLessEqual(quantized.abs().max().item(), INT8_MAX)

    def test_full_range_is_used(self):
        # At least one element per row must reach +/-127 (round of amax/scale)
        # or the scale derivation is losing a level.
        weight = make_weight(seed=4)
        quantized, _ = wo.quantize_weight_int8_per_channel(weight)
        per_row_max = quantized.abs().amax(dim=1)
        self.assertTrue(bool((per_row_max == INT8_MAX).all()))

    def test_zero_row_quantizes_to_exact_zeros(self):
        weight = make_weight(out_features=8, in_features=64, seed=5)
        weight[3] = 0.0
        quantized, scale = wo.quantize_weight_int8_per_channel(weight)
        self.assertEqual(quantized[3].abs().max().item(), 0)
        self.assertEqual(scale[3].item(), 0.0)
        self.assertFalse(bool(torch.isnan(quantized.to(torch.float32)).any()))

    def test_zero_row_matches_reference_without_nan(self):
        weight = torch.zeros(2, 8)
        quantized, scale = wo.quantize_weight_int8_per_channel(weight)
        ref_quantized, ref_scale = wo.quantize_weight_int8_per_channel_reference(weight)
        self.assertTrue(torch.equal(quantized, ref_quantized))
        self.assertTrue(torch.equal(scale, ref_scale))

    def test_round_before_clamp_keeps_amax_exact(self):
        # round(w / (amax/127)) at w == amax is exactly +/-127 before any
        # clamp; the clamp exists only for fp edge cases, not this tensor.
        weight = torch.tensor([[1.0, -0.5, 0.25], [-2.0, 1.0, -1.0]])
        quantized, _ = wo.quantize_weight_int8_per_channel(weight)
        self.assertEqual(quantized.tolist(), [[127, -64, 32], [-127, 64, -64]])

    def test_bfloat16_input_agrees_with_float32(self):
        weight = make_weight(seed=6)
        quantized_fp32, scale_fp32 = wo.quantize_weight_int8_per_channel(weight)
        quantized_bf16, scale_bf16 = wo.quantize_weight_int8_per_channel(weight.to(torch.bfloat16))
        # The staging is float32 in both paths, so only values a bf16 input
        # cannot represent may differ; the cosine gate still holds.
        reconstructed = wo.dequantize_weight_int8_per_channel(quantized_bf16, scale_bf16)
        self.assertGreaterEqual(cosine(reconstructed, weight), COSINE_GATE)
        self.assertTrue(torch.allclose(scale_fp32, scale_bf16, rtol=1e-2, atol=0))

    def test_chunked_path_is_bit_identical_to_full_path(self):
        weight = make_weight(out_features=300, in_features=256, seed=7)
        full_q, full_s = wo.quantize_weight_int8_per_channel(weight, rows_per_step=10**9)
        chunked_q, chunked_s = wo.quantize_weight_int8_per_channel(weight, rows_per_step=64)
        self.assertTrue(torch.equal(full_q, chunked_q))
        self.assertTrue(torch.equal(full_s, chunked_s))

    def test_matches_per_element_reference(self):
        weight = make_weight(out_features=128, in_features=256, seed=8)
        quantized, scale = wo.quantize_weight_int8_per_channel(weight, rows_per_step=17)
        ref_q, ref_s = wo.quantize_weight_int8_per_channel_reference(weight)
        self.assertTrue(torch.equal(quantized, ref_q))
        self.assertTrue(torch.equal(scale, ref_s))

    def test_non_2d_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "2D"):
            wo.quantize_weight_int8_per_channel(torch.randn(2, 8, 8))

    def test_invalid_rows_per_step_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rows_per_step"):
            wo.quantize_weight_int8_per_channel(make_weight(8, 8), rows_per_step=0)


class TestQuantizeWeightMxFp8(TestBase):
    def test_layout_matches_mxfp8_convention(self):
        # W8A8_MXFP8 loads float8_e4m3fn [out, in] plus one uint8 E8M0 scale
        # per 32-element group along the input dim
        # (methods/w8a8/w8a8_mxfp8.py get_pergroup_param).
        weight = make_weight(out_features=64, in_features=128, seed=9)
        quantized, scale = wo.quantize_weight_mx_fp8(weight)
        self.assertEqual(quantized.shape, (64, 128))
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.shape, (64, 128 // MX_GROUP_SIZE))
        self.assertEqual(scale.dtype, torch.uint8)

    def test_scales_are_powers_of_two(self):
        weight = make_weight(seed=10)
        _, scale = wo.quantize_weight_mx_fp8(weight)
        decoded = wo.e8m0_scale_to_float(scale)
        log2_values = torch.log2(decoded)
        # Every decoded scale is exactly 2**integer: log2 must round-trip.
        self.assertTrue(torch.allclose(log2_values, log2_values.round(), rtol=0, atol=0))

    def test_scale_puts_amax_in_top_binade_without_saturation(self):
        # ceil semantics: scale = 2**ceil(log2(amax/448)) keeps the group's
        # amax at or below 448 (no saturation) and above 224 (full mantissa)
        # whenever the E8M0 range allows it.
        weight = make_weight(out_features=64, in_features=512, seed=11) * 3.0
        _, scale = wo.quantize_weight_mx_fp8(weight)
        decoded = wo.e8m0_scale_to_float(scale)
        amax = weight.abs().reshape(64, 16, 32).amax(dim=2)
        self.assertTrue(bool((amax <= decoded * wo.FP8_E4M3_MAX + 1e-12).all()))
        self.assertTrue(bool((amax > decoded * (wo.FP8_E4M3_MAX / 2) - 1e-12).all()))

    def test_quantization_error_respects_half_ulp_bound(self):
        # E4M3 has a 3-bit mantissa; RNE deviates by at most half an ulp,
        # i.e. scale * 2**-4 at the top binade the scale selects.
        weight = make_weight(out_features=64, in_features=512, seed=12)
        quantized, scale = wo.quantize_weight_mx_fp8(weight)
        decoded = wo.dequantize_weight_mx_fp8(quantized, scale)
        scale_value = wo.e8m0_scale_to_float(scale)
        error = (decoded - weight).abs().reshape(64, 16, 32)
        bound = scale_value.unsqueeze(2) * wo.FP8_E4M3_HALF_ULP_AT_TOP_BINADE
        self.assertTrue(bool((error <= bound + 1e-12).all()))

    def test_relative_error_bound(self):
        # Per element the relative error of MXFP8 quantization stays under
        # the 2**-4 half-ulp bound (scale-relative for the top binade).
        weight = make_weight(out_features=64, in_features=512, seed=13)
        quantized, scale = wo.quantize_weight_mx_fp8(weight)
        decoded = wo.dequantize_weight_mx_fp8(quantized, scale)
        scale_value = wo.e8m0_scale_to_float(scale)
        error = (decoded - weight).abs().reshape(64, 16, 32)
        bound = scale_value.unsqueeze(2) * wo.FP8_E4M3_HALF_ULP_AT_TOP_BINADE
        relative = (error / bound).max().item()
        self.assertLessEqual(relative, 1.0)

    def test_reconstruction_cosine(self):
        weight = make_weight(out_features=64, in_features=512, seed=14)
        quantized, scale = wo.quantize_weight_mx_fp8(weight)
        decoded = wo.dequantize_weight_mx_fp8(quantized, scale)
        self.assertGreaterEqual(cosine(decoded, weight), COSINE_GATE)

    def test_outlier_groups_do_not_destroy_other_groups(self):
        # Outliers live in their own group; other groups keep their own
        # scales, so global cosine still clears the gate.
        weight = make_weight(out_features=64, in_features=512, seed=15)
        weight[:, :32] *= 500.0
        quantized, scale = wo.quantize_weight_mx_fp8(weight)
        decoded = wo.dequantize_weight_mx_fp8(quantized, scale)
        self.assertGreaterEqual(cosine(decoded, weight), COSINE_GATE)

    def test_zero_group_stays_exactly_zero(self):
        weight = make_weight(out_features=8, in_features=64, seed=16)
        weight[:, :32] = 0.0
        quantized, scale = wo.quantize_weight_mx_fp8(weight)
        decoded = wo.dequantize_weight_mx_fp8(quantized, scale)
        self.assertEqual(decoded[:, :32].abs().max().item(), 0.0)

    def test_chunked_path_is_bit_identical_to_full_path(self):
        weight = make_weight(out_features=300, in_features=128, seed=17)
        full_q, full_s = wo.quantize_weight_mx_fp8(weight, rows_per_step=10**9)
        chunked_q, chunked_s = wo.quantize_weight_mx_fp8(weight, rows_per_step=100)
        self.assertTrue(torch.equal(full_q, chunked_q))
        self.assertTrue(torch.equal(full_s, chunked_s))

    def test_matches_reference_implementation(self):
        weight = make_weight(out_features=64, in_features=96, seed=18)
        quantized, scale = wo.quantize_weight_mx_fp8(weight, rows_per_step=7)
        ref_q, ref_s = wo.quantize_weight_mx_fp8_reference(weight)
        self.assertTrue(torch.equal(quantized, ref_q))
        self.assertTrue(torch.equal(scale, ref_s))

    def test_e8m0_encoding_round_trips(self):
        for exponent in (-127, -1, 0, 1, 63, 127):
            encoded = torch.tensor([exponent + wo.E8M0_EXPONENT_BIAS], dtype=torch.uint8)
            decoded = wo.e8m0_scale_to_float(encoded).item()
            self.assertEqual(decoded, math.pow(2, exponent))

    def test_unaligned_reduction_dim_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "multiple of the MX group size"):
            wo.quantize_weight_mx_fp8(make_weight(8, 33))

    def test_npu_wrapper_dispatches_by_device(self):
        # The NPU branch is unreachable on CPU (device check), so assert the
        # routing guard itself rather than the operator.
        cpu_weight = make_weight(8, 32, seed=19)
        self.assertFalse(wo._is_npu_tensor(cpu_weight))

    def test_npu_wrapper_skipped_without_hardware(self):
        if wo.torch_npu is None or not torch.npu.is_available():
            self.skipTest("requires NPU hardware")
        weight = make_weight(64, 64, seed=20).npu()
        quantized, scale = wo.quantize_weight_mx_fp8(weight)
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.dtype, torch.uint8)

    def test_fp4_stub_is_explicit(self):
        with self.assertRaisesRegex(NotImplementedError, "MXFP4"):
            wo.quantize_weight_mx_fp4(make_weight(8, 32))


class TestCompareLayersCosine(TestBase):
    def _make_model(self, seed=0, noise=0.0):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        with torch.no_grad():
            for layer in model:
                if isinstance(layer, nn.Linear):
                    layer.weight += noise * torch.randn_like(layer.weight)
        return model

    def test_identical_models_report_cosine_one(self):
        model = self._make_model(seed=21)
        twin = self._make_model(seed=21)
        x = torch.randn(4, 16)
        result = accuracy.compare_layers_cosine(model, twin, x)
        self.assertEqual(set(result), {"0", "2"})
        for name, value in result.items():
            # The dot product and the two norms accumulate in different
            # orders, so even bit-identical outputs land within one float32
            # epsilon of 1.0 rather than exactly on it.
            self.assertAlmostEqual(value, 1.0, places=6, msg=f"layer {name} should be identical")

    def test_noisy_model_reports_cosine_below_one(self):
        model = self._make_model(seed=22)
        noisy = self._make_model(seed=22, noise=0.05)
        x = torch.randn(4, 16)
        result = accuracy.compare_layers_cosine(model, noisy, x)
        for name, value in result.items():
            self.assertLess(value, 1.0)
            self.assertGreater(value, 0.5)

    def test_keyword_inputs_are_supported(self):
        model = self._make_model(seed=23)

        class KwModel(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, features):
                return self.inner(features)

        wrapped = KwModel(model)
        result = accuracy.compare_layers_cosine(wrapped, wrapped, {"features": torch.randn(4, 16)})
        self.assertEqual(set(result), {"inner.0", "inner.2"})

    def test_module_filter_restriction(self):
        model = self._make_model(seed=24)
        result = accuracy.compare_layers_cosine(
            model,
            model,
            torch.randn(4, 16),
            module_filter=lambda name, module: name.endswith("2"),
        )
        self.assertEqual(set(result), {"2"})

    def test_hooks_are_removed_after_run(self):
        model = self._make_model(seed=25)
        x = torch.randn(4, 16)
        accuracy.compare_layers_cosine(model, model, x)
        for _, module in model.named_modules():
            self.assertFalse(hasattr(module, "_cosine_capture_name"))
        # A plain forward still works afterwards.
        with torch.inference_mode():
            model(x)

    def test_layers_only_in_one_model_are_ignored(self):
        # Two models that share a 'shared' submodule name but otherwise have
        # different trees: only the name they both define is compared.
        torch.manual_seed(26)

        class ModelA(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = nn.Linear(8, 8)
                self.tail = nn.Linear(8, 4)

            def forward(self, x):
                return self.tail(self.shared(x))

        class ModelB(nn.Module):
            def __init__(self):
                super().__init__()
                self.extra = nn.Linear(8, 8)
                self.shared = nn.Linear(8, 8)

            def forward(self, x):
                return self.shared(self.extra(x))

        model_a, model_b = ModelA(), ModelB()
        with torch.no_grad():
            model_b.shared.weight.copy_(model_a.shared.weight)
            model_b.shared.bias.copy_(model_a.shared.bias)
        result = accuracy.compare_layers_cosine(
            model_a, model_b, torch.randn(4, 8), module_filter=lambda n, m: isinstance(m, nn.Linear)
        )
        # 'tail'/'extra' exist in only one model each and are dropped; the
        # shared name keeps its own comparison. Its cosine is not 1.0 — the
        # two models feed that layer different inputs — but it is a real
        # number in [-1, 1], proving the pair was actually compared.
        self.assertEqual(set(result), {"shared"})
        self.assertLessEqual(abs(result["shared"]), 1.0)

    def test_zero_outputs_report_cosine_one(self):
        zero_model = nn.Linear(8, 8)
        with torch.no_grad():
            zero_model.weight.zero_()
            zero_model.bias.zero_()
        twin = nn.Linear(8, 8)
        with torch.no_grad():
            twin.weight.zero_()
            twin.bias.zero_()
        result = accuracy.compare_layers_cosine(zero_model, twin, torch.randn(4, 8))
        self.assertEqual(result[""], 1.0)

    def test_quantized_weight_end_to_end_cosine(self):
        # The actual use case: replace a Linear's weight with the int8
        # quantize-dequantize reconstruction and compare layer outputs.
        torch.manual_seed(27)
        baseline = nn.Linear(64, 64)
        quantized_model = nn.Linear(64, 64)
        with torch.no_grad():
            quantized_model.weight.copy_(baseline.weight)
            quantized_model.bias.copy_(baseline.bias)
            q, s = wo.quantize_weight_int8_per_channel(baseline.weight.detach())
            quantized_model.weight.copy_(wo.dequantize_weight_int8_per_channel(q, s))
        x = torch.randn(8, 64)
        result = accuracy.compare_layers_cosine(baseline, quantized_model, x)
        self.assertGreaterEqual(result[""], COSINE_GATE)


class TestRenderCosineReport(TestBase):
    def test_renders_markdown_table(self):
        report = accuracy.render_cosine_report({"a": 0.99, "b": 0.5})
        self.assertIn("| layer | cosine |", report)
        self.assertIn("| a |", report)
        self.assertIn("| b |", report)
        self.assertIn("min / mean", report)

    def test_worst_first_ordering(self):
        report = accuracy.render_cosine_report({"good": 0.999, "bad": 0.1})
        self.assertLess(report.index("| bad |"), report.index("| good |"))

    def test_name_ordering_when_not_worst_first(self):
        report = accuracy.render_cosine_report({"b": 0.1, "a": 0.9}, worst_first=False)
        self.assertLess(report.index("| a |"), report.index("| b |"))

    def test_empty_report_is_explained(self):
        report = accuracy.render_cosine_report({})
        self.assertIn("No shared captured layers", report)


if __name__ == "__main__":
    unittest.main()
