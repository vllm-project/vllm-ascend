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

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import create_mock_vllm_config
from vllm_ascend.quantization.methods.w8a8.fp8_block import (
    AscendFp8BlockFusedMoEMethod,
    AscendFp8BlockLinearMethod,
    resolve_block_scales,
)

MODULE = "vllm_ascend.quantization.methods.w8a8.fp8_block"

# Elements per shared exponent produced by npu_dynamic_mx_quant.
MX_GROUP_SIZE = 32


def reference_resolve(weight, scale_inv, block_n, block_k, out_dtype):
    """Straightforward per-element expansion used as the numerical oracle."""
    out_features, in_features = weight.shape
    weight_fp32 = weight.to(torch.float32)
    resolved = torch.empty((out_features, in_features), dtype=out_dtype)
    for row in range(out_features):
        for col in range(in_features):
            resolved[row, col] = weight_fp32[row, col] * scale_inv[row // block_n, col // block_k]
    return resolved


def make_block_weight(out_features, in_features, block_n, block_k, seed=0):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(out_features, in_features, generator=generator).to(torch.float8_e4m3fn)
    scale_inv = torch.rand((out_features + block_n - 1) // block_n, (in_features + block_k - 1) // block_k) + 0.5
    return weight, scale_inv


class TestResolveBlockScales(TestBase):
    def test_matches_per_element_expansion(self):
        weight, scale_inv = make_block_weight(16, 12, 4, 3)
        resolved = resolve_block_scales(weight, scale_inv, 4, 3, torch.float32)
        expected = reference_resolve(weight, scale_inv, 4, 3, torch.float32)
        self.assertTrue(torch.equal(resolved, expected))

    def test_handles_blocks_that_do_not_divide_the_shape(self):
        weight, scale_inv = make_block_weight(10, 7, 4, 3, seed=1)
        resolved = resolve_block_scales(weight, scale_inv, 4, 3, torch.float32)
        expected = reference_resolve(weight, scale_inv, 4, 3, torch.float32)
        self.assertEqual(resolved.shape, (10, 7))
        self.assertTrue(torch.equal(resolved, expected))

    def test_row_chunking_does_not_change_the_result(self):
        weight, scale_inv = make_block_weight(10, 7, 2, 3, seed=2)
        expected = reference_resolve(weight, scale_inv, 2, 3, torch.float32)
        with patch(f"{MODULE}._ROWS_PER_DEQUANT_STEP", 4):
            resolved = resolve_block_scales(weight, scale_inv, 2, 3, torch.float32)
        self.assertTrue(torch.equal(resolved, expected))

    def test_scale_is_applied_in_float32(self):
        # A block's shared scale is applied before the result is rounded to the
        # model dtype, so a scale bfloat16 cannot represent is not itself lossy.
        weight = torch.ones(1, 1).to(torch.float8_e4m3fn)
        scale_inv = torch.tensor([[1.001953125]])
        resolved = resolve_block_scales(weight, scale_inv, 128, 128, torch.float32)
        self.assertEqual(resolved.item(), 1.001953125)

    def test_unpaired_scale_shape_is_rejected(self):
        weight = torch.randn(256, 256).to(torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "unpaired"):
            resolve_block_scales(weight, torch.ones(2, 3), 128, 128, torch.bfloat16)

    def test_non_2d_weight_is_rejected(self):
        weight = torch.randn(2, 256, 256).to(torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "2D"):
            resolve_block_scales(weight, torch.ones(2, 2, 2), 128, 128, torch.bfloat16)


class TestAscendFp8BlockLinearMethod(TestBase):
    def build_scheme(self, is_950=False, block_size=(128, 128)):
        with (
            patch(f"{MODULE}.get_current_vllm_config", return_value=create_mock_vllm_config()),
            patch(f"{MODULE}.is_950", return_value=is_950),
            patch(f"{MODULE}.AscendW8A8MXFP8DynamicLinearMethod") as mock_mxfp8,
        ):
            mock_mxfp8.return_value.dynamic_mx_quant_scale_alg = 0
            mock_mxfp8.return_value.group_size = MX_GROUP_SIZE
            scheme = AscendFp8BlockLinearMethod(block_size)
        return scheme

    def test_get_weight_is_fp8(self):
        scheme = self.build_scheme()
        weight = scheme.get_weight(512, 256, torch.bfloat16)["weight"]
        self.assertEqual(weight.shape, (256, 512))
        self.assertEqual(weight.dtype, torch.float8_e4m3fn)

    def test_pergroup_param_declares_block_scale_and_loader_hints(self):
        scheme = self.build_scheme(block_size=(128, 64))
        params = scheme.get_pergroup_param(512, 256, torch.bfloat16)
        self.assertEqual(params["weight_scale_inv"].shape, (2, 8))
        self.assertEqual(params["weight_scale_inv"].dtype, torch.float32)
        # A merged column-parallel loader offsets in weight elements, so it needs
        # the block_n packing factor; a row-parallel loader needs the input dim.
        self.assertEqual(params["_packed_dim"], 0)
        self.assertEqual(params["_packed_factor"], 128)
        self.assertEqual(params["_input_dim"], 1)

    def test_pergroup_param_rounds_up_partial_blocks(self):
        scheme = self.build_scheme()
        params = scheme.get_pergroup_param(300, 200, torch.bfloat16)
        self.assertEqual(params["weight_scale_inv"].shape, (2, 3))

    def _make_layer(self, out_features=8, in_features=64, block=(4, 32)):
        weight, scale_inv = make_block_weight(out_features, in_features, block[0], block[1], seed=3)
        layer = nn.Module()
        layer.prefix = "model.layers.0.mlp.down_proj"
        layer.weight = nn.Parameter(weight, requires_grad=False)
        layer.weight_scale_inv = nn.Parameter(scale_inv, requires_grad=False)
        return layer, weight, scale_inv

    def test_resolves_to_model_dtype_off_950(self):
        scheme = self.build_scheme(is_950=False, block_size=(4, 32))
        layer, weight, scale_inv = self._make_layer()

        with patch(f"{MODULE}.maybe_trans_nz", side_effect=lambda tensor: tensor):
            scheme.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        self.assertFalse(hasattr(layer, "weight_scale_inv"))
        expected = reference_resolve(weight, scale_inv, 4, 32, torch.bfloat16)
        self.assertTrue(torch.equal(layer.weight.data, expected))

    def test_requantizes_to_mxfp8_on_950(self):
        scheme = self.build_scheme(is_950=True, block_size=(4, 32))
        layer, _, _ = self._make_layer()
        quantized = torch.zeros(8, 64).to(torch.float8_e4m3fn)
        mx_scale = torch.zeros(8, 64 // MX_GROUP_SIZE, dtype=torch.uint8)

        with patch(f"{MODULE}.torch_npu") as mock_npu:
            mock_npu.npu_dynamic_mx_quant.return_value = (quantized, mx_scale)
            scheme.process_weights_after_loading(layer)

        mock_npu.npu_dynamic_mx_quant.assert_called_once()
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.weight_scale.dtype, torch.uint8)
        self.assertFalse(hasattr(layer, "weight_scale_inv"))
        scheme.mxfp8_method.process_weights_after_loading.assert_called_once_with(layer)

    def test_falls_back_when_reduction_dim_is_not_mx_aligned(self):
        scheme = self.build_scheme(is_950=True, block_size=(4, 32))
        layer, weight, scale_inv = self._make_layer(out_features=8, in_features=48, block=(4, 32))

        with patch(f"{MODULE}.maybe_trans_nz", side_effect=lambda tensor: tensor):
            scheme.process_weights_after_loading(layer)

        self.assertIsNone(scheme.mxfp8_method)
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        expected = reference_resolve(weight, scale_inv, 4, 32, torch.bfloat16)
        self.assertTrue(torch.equal(layer.weight.data, expected))

    def test_mx_fallback_does_not_require_layer_prefix(self):
        # The 950 MX-aligned path never logs this warning; this only covers the
        # fallback log so a missing prefix cannot raise AttributeError.
        scheme = self.build_scheme(is_950=True, block_size=(4, 32))
        layer, _, _ = self._make_layer(out_features=8, in_features=48, block=(4, 32))
        delattr(layer, "prefix")

        with patch(f"{MODULE}.maybe_trans_nz", side_effect=lambda tensor: tensor):
            scheme.process_weights_after_loading(layer)

        self.assertIsNone(scheme.mxfp8_method)
        self.assertEqual(layer.weight.dtype, torch.bfloat16)

    def test_apply_uses_unquantized_gemm_when_resolved_to_bf16(self):
        scheme = self.build_scheme(is_950=False)
        scheme.mxfp8_method = None
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randn(4, 8), requires_grad=False)
        x = torch.randn(2, 8)

        with patch("torch.ops.vllm.unquantized_gemm", return_value="gemm") as mock_gemm:
            self.assertEqual(scheme.apply(layer, x), "gemm")
        mock_gemm.assert_called_once_with(x, layer.weight, None)

    def test_apply_delegates_to_mxfp8_when_requantized(self):
        scheme = self.build_scheme(is_950=True)
        scheme.mxfp8_method.apply.return_value = "mxfp8"
        layer = nn.Module()
        x = torch.randn(2, 8)

        self.assertEqual(scheme.apply(layer, x, None, 0), "mxfp8")
        scheme.mxfp8_method.apply.assert_called_once_with(layer, x, None, 0)


class TestAscendFp8BlockFusedMoEMethod(TestBase):
    def build_scheme(self, is_950=False, block_size=(4, 32)):
        with (
            patch(f"{MODULE}.get_current_vllm_config", return_value=create_mock_vllm_config()),
            patch(f"{MODULE}.is_950", return_value=is_950),
            patch(f"{MODULE}.AscendW8A8MXFP8DynamicFusedMoEMethod") as mock_mxfp8,
        ):
            mock_mxfp8.return_value.dynamic_mx_quant_scale_alg = 0
            mock_mxfp8.return_value.group_size = MX_GROUP_SIZE
            scheme = AscendFp8BlockFusedMoEMethod(block_size, MagicMock())
        return scheme

    def test_weights_are_fp8_and_scales_are_block_shaped(self):
        scheme = self.build_scheme(block_size=(128, 128))
        weights = scheme.get_weight(4, 256, 512, torch.bfloat16)
        self.assertEqual(weights["w13_weight"].shape, (4, 512, 512))
        self.assertEqual(weights["w2_weight"].shape, (4, 512, 256))
        self.assertEqual(weights["w13_weight"].dtype, torch.float8_e4m3fn)

        scales = scheme.get_dynamic_quant_param(4, 256, 512, torch.bfloat16)
        self.assertEqual(scales["w13_weight_scale_inv"].shape, (4, 4, 4))
        self.assertEqual(scales["w2_weight_scale_inv"].shape, (4, 4, 2))
        self.assertEqual(scales["w13_weight_scale_inv"].dtype, torch.float32)

    def test_group_size_makes_the_loader_shard_the_scales(self):
        # AscendFusedMoEMethod only tags a scale as group-wise, and therefore
        # narrows it per expert shard, when the scheme reports a group size.
        scheme = self.build_scheme(block_size=(128, 64))
        self.assertEqual(scheme.group_size, 64)

    def _make_moe_layer(self, num_experts=2, intermediate=8, hidden=64, block=(4, 32)):
        layer = nn.Module()
        layer.prefix = "model.layers.0.mlp.experts"
        w13, w13_scale = make_block_weight(num_experts * 2 * intermediate, hidden, block[0], block[1], seed=4)
        w2, w2_scale = make_block_weight(num_experts * hidden, intermediate, block[0], block[1], seed=5)
        layer.w13_weight = nn.Parameter(w13.reshape(num_experts, 2 * intermediate, hidden), requires_grad=False)
        layer.w2_weight = nn.Parameter(w2.reshape(num_experts, hidden, intermediate), requires_grad=False)
        layer.w13_weight_scale_inv = nn.Parameter(
            torch.rand(num_experts, 2 * intermediate // block[0], hidden // block[1]) + 0.5, requires_grad=False
        )
        layer.w2_weight_scale_inv = nn.Parameter(
            torch.rand(num_experts, hidden // block[0], intermediate // block[1]) + 0.5, requires_grad=False
        )
        return layer

    def test_resolves_every_expert_off_950(self):
        scheme = self.build_scheme(is_950=False)
        layer = self._make_moe_layer(intermediate=32)
        original_w13 = layer.w13_weight.data.clone()
        original_scale = layer.w13_weight_scale_inv.data.clone()
        scheme._bf16_method = MagicMock()

        scheme.process_weights_after_loading(layer)

        self.assertEqual(layer.w13_weight.dtype, torch.bfloat16)
        self.assertEqual(layer.w2_weight.dtype, torch.bfloat16)
        self.assertFalse(hasattr(layer, "w13_weight_scale_inv"))
        self.assertFalse(hasattr(layer, "w2_weight_scale_inv"))
        scheme._bf16_method.process_weights_after_loading.assert_called_once_with(layer)

        expected = reference_resolve(original_w13[0], original_scale[0], 4, 32, torch.bfloat16)
        self.assertTrue(torch.equal(layer.w13_weight.data[0], expected))

    def test_requantizes_every_expert_on_950(self):
        scheme = self.build_scheme(is_950=True)
        layer = self._make_moe_layer(intermediate=32)
        num_experts = layer.w13_weight.shape[0]

        def fake_mx_quant(tensor, **kwargs):
            rows, cols = tensor.shape
            return (
                torch.zeros(rows, cols).to(torch.float8_e4m3fn),
                torch.zeros(rows, cols // MX_GROUP_SIZE, dtype=torch.uint8),
            )

        with patch(f"{MODULE}.torch_npu") as mock_npu:
            mock_npu.npu_dynamic_mx_quant.side_effect = fake_mx_quant
            scheme.process_weights_after_loading(layer)

        self.assertEqual(mock_npu.npu_dynamic_mx_quant.call_count, 2 * num_experts)
        self.assertEqual(layer.w13_weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.w13_weight_scale.shape, (num_experts, 64, 64 // MX_GROUP_SIZE))
        self.assertEqual(layer.w13_weight_scale.dtype, torch.uint8)
        self.assertFalse(hasattr(layer, "w13_weight_scale_inv"))
        scheme.mxfp8_method.process_weights_after_loading.assert_called_once_with(layer)

    def test_apply_delegates_to_the_active_method(self):
        scheme = self.build_scheme(is_950=False)
        scheme._bf16_method = MagicMock()
        scheme._bf16_method.apply.return_value = "bf16"
        layer = nn.Module()

        result = scheme.apply(layer, torch.randn(2, 8), None, None, None, None)

        self.assertEqual(result, "bf16")
        scheme._bf16_method.apply.assert_called_once()


if __name__ == "__main__":
    unittest.main()
