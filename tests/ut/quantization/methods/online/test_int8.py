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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.linear import LinearBase, RowParallelLinear

from tests.ut.base import TestBase
from vllm_ascend.quantization.methods.online.int8 import (
    AscendInt8PerChannelOnlineLinearMethod,
    AscendInt8PerChannelOnlineMoEMethod,
    _linear_reduce_group,
    _moe_reduce_group,
    _zero_expert_padding,
    quantize_last_dim_to_int8,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
    AscendW8A8DynamicLinearMethod,
)

MODULE = "vllm_ascend.quantization.methods.online.int8"
PROCESSED_FLAG = "_already_called_process_weights_after_loading"


class TestQuantizeLastDimToInt8(TestBase):
    def test_reconstruction_is_within_int8_resolution(self):
        torch.manual_seed(0)
        weight = torch.randn(16, 64, dtype=torch.bfloat16)

        qweight, scale = quantize_last_dim_to_int8(weight, scale_dtype=torch.float32)

        self.assertEqual(qweight.dtype, torch.int8)
        self.assertEqual(qweight.shape, weight.shape)
        self.assertEqual(scale.shape, (16, 1))

        dequant = qweight.float() * scale
        # Symmetric per-output-channel int8: the error is bounded by half a
        # quantization step, i.e. amax / 254 per row.
        row_amax = weight.float().abs().amax(dim=1, keepdim=True)
        max_error = (dequant - weight.float()).abs().amax(dim=1, keepdim=True)
        self.assertTrue(torch.all(max_error <= row_amax / 254 + 1e-6))

    def test_scale_is_amax_over_127(self):
        weight = torch.tensor([[1.0, -0.5, 0.25], [2.0, 0.0, -2.0]], dtype=torch.float32)

        qweight, scale = quantize_last_dim_to_int8(weight)

        self.assertTrue(torch.allclose(scale.flatten(), torch.tensor([1.0 / 127, 2.0 / 127])))
        self.assertTrue(torch.allclose(qweight[0].float(), torch.tensor([127.0, -64.0, 32.0]), atol=1))
        self.assertTrue(torch.allclose(qweight[1].float(), torch.tensor([127.0, 0.0, -127.0]), atol=1))

    def test_zero_row_stays_zero(self):
        weight = torch.zeros(2, 8, dtype=torch.bfloat16)
        weight[1] = 1.0

        qweight, scale = quantize_last_dim_to_int8(weight)

        self.assertTrue(torch.all(qweight[0] == 0))
        self.assertEqual(scale[0].item(), 0.0)
        self.assertGreater(scale[1].item(), 0.0)

    def test_scale_dtype_follows_request(self):
        weight = torch.randn(4, 8, dtype=torch.bfloat16)

        _, bf16_scale = quantize_last_dim_to_int8(weight, scale_dtype=torch.bfloat16)
        _, fp32_scale = quantize_last_dim_to_int8(weight, scale_dtype=torch.float32)

        self.assertEqual(bf16_scale.dtype, torch.bfloat16)
        self.assertEqual(fp32_scale.dtype, torch.float32)

    def test_expert_stack_keeps_leading_dims(self):
        weight = torch.randn(3, 5, 8, dtype=torch.bfloat16)

        qweight, scale = quantize_last_dim_to_int8(weight)

        self.assertEqual(qweight.shape, (3, 5, 8))
        self.assertEqual(scale.shape, (3, 5, 1))

    def test_rejects_already_quantized_weight(self):
        with self.assertRaises(NotImplementedError):
            quantize_last_dim_to_int8(torch.zeros(4, 8, dtype=torch.int8))

    def test_all_reduces_amax_over_the_given_group(self):
        weight = torch.ones(4, 8, dtype=torch.float32)
        group = MagicMock()
        group.world_size = 2

        with patch("torch.distributed.all_reduce") as all_reduce:
            _, scale = quantize_last_dim_to_int8(weight, reduce_group=group)

        all_reduce.assert_called_once()
        self.assertEqual(all_reduce.call_args.kwargs["op"], torch.distributed.ReduceOp.MAX)
        self.assertAlmostEqual(scale[0].item(), 1.0 / 127, places=7)

    def test_chunked_quantization_matches_a_single_chunk(self):
        """The chunk loop must only bound memory, never change the result."""
        torch.manual_seed(0)
        weight = torch.randn(2, 6, 12, dtype=torch.bfloat16)

        qweight, scale = quantize_last_dim_to_int8(weight, scale_dtype=torch.bfloat16)
        with patch(f"{MODULE}._QUANT_CHUNK_ELEMS", 8):
            chunked_qweight, chunked_scale = quantize_last_dim_to_int8(weight, scale_dtype=torch.bfloat16)

        self.assertTrue(torch.equal(qweight, chunked_qweight))
        self.assertTrue(torch.equal(scale, chunked_scale))

    def test_does_not_modify_the_input_weight(self):
        # `.float()` is a no-op view for an f32 input, so an in-place divide
        # would rewrite the caller's weight.
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                weight = torch.randn(8, 16, dtype=dtype)
                original = weight.clone()

                quantize_last_dim_to_int8(weight, scale_dtype=dtype)

                self.assertTrue(torch.equal(weight, original))


class TestInt8PerChannelOnlineLinearMethod(TestBase):
    def _layer(self, weight=None):
        layer = SimpleNamespace(prefix="model.layers.0.mlp.down_proj", params_dtype=torch.bfloat16)
        layer.weight = torch.nn.Parameter(
            torch.randn(32, 64, dtype=torch.bfloat16) if weight is None else weight,
            requires_grad=False,
        )
        return layer

    def _method(self):
        # OnlineLinearBase.__init__ records the model dtype.
        vllm_config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
        with patch(
            "vllm.model_executor.layers.quantization.online.fp8.get_current_vllm_config",
            return_value=vllm_config,
        ):
            return AscendInt8PerChannelOnlineLinearMethod()

    def test_reuses_the_offline_w8a8_dynamic_scheme(self):
        self.assertIs(AscendInt8PerChannelOnlineLinearMethod.scheme_cls, AscendW8A8DynamicLinearMethod)

    def test_process_weights_installs_scale_and_delegates_the_layout(self):
        method = self._method()
        layer = self._layer()

        scale_buffer = torch.empty(32, 1, dtype=torch.bfloat16)
        offset_buffer = torch.empty(32, 1, dtype=torch.bfloat16)
        with (
            patch(
                f"{MODULE}.AscendW8A8DynamicLinearMethod.get_perchannel_param",
                return_value={"weight_scale": scale_buffer, "weight_offset": offset_buffer},
            ),
            patch(f"{MODULE}.AscendW8A8DynamicLinearMethod.process_weights_after_loading") as layout,
            patch(f"{MODULE}._linear_reduce_group", return_value=None),
        ):
            method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertEqual(tuple(layer.weight.shape), (32, 64))
        self.assertEqual(tuple(layer.weight_scale.shape), (32, 1))
        self.assertTrue(torch.all(layer.weight_offset == 0))
        layout.assert_called_once_with(layer)
        self.assertTrue(getattr(layer, PROCESSED_FLAG))

    def test_process_weights_is_idempotent(self):
        layer = self._layer()
        setattr(layer, PROCESSED_FLAG, True)

        with patch(f"{MODULE}.quantize_last_dim_to_int8") as quantize:
            self._method().process_weights_after_loading(layer)

        quantize.assert_not_called()

    def test_process_weights_rejects_non_2d_weight(self):
        layer = self._layer(weight=torch.randn(2, 8, 16, dtype=torch.bfloat16))

        with self.assertRaises(ValueError):
            self._method().process_weights_after_loading(layer)


class TestInt8PerChannelOnlineMoEMethod(TestBase):
    def _method(self):
        # Bypass __init__: it builds the offline scheme, which probes the MC2
        # communication group that does not exist in a unit-test process.
        method = AscendInt8PerChannelOnlineMoEMethod.__new__(AscendInt8PerChannelOnlineMoEMethod)
        method.quant_method = MagicMock()
        return method

    def test_reuses_the_offline_w8a8_dynamic_scheme(self):
        self.assertIs(AscendInt8PerChannelOnlineMoEMethod.scheme_cls, AscendW8A8DynamicFusedMoEMethod)

    def test_process_weights_quantizes_both_projections_and_delegates_the_layout(self):
        method = self._method()
        layer = SimpleNamespace(
            prefix="model.layers.0.mlp.experts",
            moe_config=SimpleNamespace(tp_size=1, dp_size=1, pcp_size=1),
        )
        layer.w13_weight = torch.nn.Parameter(torch.randn(2, 32, 16, dtype=torch.bfloat16), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.randn(2, 16, 16, dtype=torch.bfloat16), requires_grad=False)

        buffers = {
            "w13_weight_scale": torch.empty(2, 32, 1, dtype=torch.bfloat16),
            "w13_weight_offset": torch.empty(2, 32, 1, dtype=torch.bfloat16),
            "w2_weight_scale": torch.empty(2, 16, 1, dtype=torch.bfloat16),
            "w2_weight_offset": torch.empty(2, 16, 1, dtype=torch.bfloat16),
        }
        method.quant_method.get_dynamic_quant_param.return_value = dict(buffers)

        method.process_weights_after_loading(layer)

        method.quant_method.get_dynamic_quant_param.assert_called_once_with(2, 16, 16, torch.bfloat16)
        self.assertEqual(layer.w13_weight.dtype, torch.int8)
        self.assertEqual(layer.w2_weight.dtype, torch.int8)
        self.assertEqual(tuple(layer.w13_weight.shape), (2, 32, 16))
        self.assertTrue(torch.all(layer.w13_weight_offset == 0))
        self.assertTrue(torch.all(layer.w2_weight_offset == 0))
        method.quant_method.process_weights_after_loading.assert_called_once_with(layer)

    def test_tensor_sharded_experts_spanning_dp_are_rejected(self):
        with self.assertRaises(NotImplementedError):
            _moe_reduce_group(SimpleNamespace(tp_size=8, dp_size=2, pcp_size=1))


class TestZeroExpertPadding(TestBase):
    def _layer(self, w2_contraction=8):
        return SimpleNamespace(
            w13_weight=torch.nn.Parameter(torch.ones(2, 8, 6, dtype=torch.bfloat16), requires_grad=False),
            w2_weight=torch.nn.Parameter(torch.ones(2, 6, w2_contraction, dtype=torch.bfloat16), requires_grad=False),
        )

    def test_padded_lanes_are_zeroed_before_quantization(self):
        layer = self._layer()
        moe_config = SimpleNamespace(
            hidden_dim_unpadded=4,
            intermediate_size_per_partition_unpadded=3,
            w13_num_shards=2,
        )

        _zero_expert_padding(moe_config, layer)

        # w13 holds two 4-row shards whose last row (3 and 7) is padding, and
        # columns 4..5 are the padded hidden dim.
        self.assertTrue(torch.all(layer.w13_weight[0, 3] == 0))
        self.assertTrue(torch.all(layer.w13_weight[0, 7] == 0))
        self.assertTrue(torch.all(layer.w13_weight[0, :, 4:] == 0))
        self.assertTrue(torch.all(layer.w13_weight[0, 0:3, 0:4] == 1))
        # w2 is padded on both the hidden rows and the intermediate columns.
        self.assertTrue(torch.all(layer.w2_weight[0, 4:, :] == 0))
        self.assertTrue(torch.all(layer.w2_weight[0, :, 3:] == 0))
        self.assertTrue(torch.all(layer.w2_weight[0, 0:4, 0:3] == 1))

    def test_unpadded_weights_are_left_alone(self):
        layer = self._layer(w2_contraction=4)
        moe_config = SimpleNamespace(
            hidden_dim_unpadded=6,
            intermediate_size_per_partition_unpadded=4,
            w13_num_shards=2,
        )

        _zero_expert_padding(moe_config, layer)

        self.assertTrue(torch.all(layer.w13_weight == 1))
        self.assertTrue(torch.all(layer.w2_weight == 1))

    def test_missing_unpadded_metadata_is_a_noop(self):
        layer = self._layer()

        _zero_expert_padding(SimpleNamespace(), layer)

        self.assertTrue(torch.all(layer.w13_weight == 1))
        self.assertTrue(torch.all(layer.w2_weight == 1))


class TestReduceGroupSelection(TestBase):
    def test_no_reduction_for_replicated_layers(self):
        layer = MagicMock(spec=LinearBase)
        layer.tp_size = 1
        layer.input_size = 64
        layer.input_size_per_partition = 64
        layer.prefix = "qkv_proj"
        self.assertIsNone(_linear_reduce_group(layer))

    def test_no_reduction_for_column_parallel_layers(self):
        layer = MagicMock(spec=LinearBase)
        layer.tp_size = 8
        layer.input_size = 64
        layer.input_size_per_partition = 64
        layer.prefix = "qkv_proj"
        self.assertIsNone(_linear_reduce_group(layer))

    def test_row_parallel_layer_reduces_over_the_tp_group(self):
        layer = MagicMock(spec=RowParallelLinear)
        layer.tp_size = 8
        layer.input_size = 64
        layer.input_size_per_partition = 8
        layer.prefix = "lm_head"
        tp_group = object()
        with (
            patch(f"{MODULE}.oproj_tp_enable", return_value=False),
            patch("vllm.distributed.get_tp_group", return_value=tp_group),
        ):
            self.assertIs(_linear_reduce_group(layer), tp_group)

    def test_reduction_follows_the_fine_grained_o_proj_group(self):
        layer = MagicMock(spec=RowParallelLinear)
        layer.tp_size = 8
        layer.input_size = 64
        layer.input_size_per_partition = 8
        layer.prefix = "model.layers.0.self_attn.o_proj"
        otp_group = object()
        with (
            patch(f"{MODULE}.oproj_tp_enable", return_value=True),
            patch("vllm_ascend.distributed.parallel_state.get_otp_group", return_value=otp_group),
        ):
            self.assertIs(_linear_reduce_group(layer), otp_group)

    def test_reduction_follows_the_fine_grained_down_proj_group(self):
        layer = MagicMock(spec=RowParallelLinear)
        layer.tp_size = 8
        layer.input_size = 64
        layer.input_size_per_partition = 8
        layer.prefix = "model.layers.0.mlp.down_proj"
        mlp_tp_group = object()
        with (
            patch(f"{MODULE}.oproj_tp_enable", return_value=False),
            patch(f"{MODULE}.mlp_tp_enable", return_value=True),
            patch("vllm_ascend.distributed.parallel_state.get_mlp_tp_group", return_value=mlp_tp_group),
        ):
            self.assertIs(_linear_reduce_group(layer), mlp_tp_group)


class TestRequantizationIsRejected(TestBase):
    """vLLM's `resolve_quant_method` opens this path for a quantized checkpoint."""

    def test_linear_method_rejects_a_requantization_source(self):
        vllm_config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
        with patch(
            "vllm.model_executor.layers.quantization.online.fp8.get_current_vllm_config",
            return_value=vllm_config,
        ):
            method = AscendInt8PerChannelOnlineLinearMethod()

        with self.assertRaises(NotImplementedError) as raised:
            method.set_requantization_source(MagicMock())

        self.assertIn("not supported", str(raised.exception))

    def test_moe_method_rejects_a_requantization_source(self):
        method = AscendInt8PerChannelOnlineMoEMethod.__new__(AscendInt8PerChannelOnlineMoEMethod)
        method.quant_method = MagicMock()

        with self.assertRaises(NotImplementedError):
            method.set_requantization_source(MagicMock())


if __name__ == "__main__":
    unittest.main()
