#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.quantization.methods.w8a8_dynamic import (
    _W8A8_DYNAMIC_MATMUL_CHUNK_SIZE,
    AscendW8A8DynamicFusedMoEMethod310,
    AscendW8A8DynamicLinearMethod310,
    enable_dflash_full_decode_graph_safe_w8a8,
    enable_dflash_piecewise_graph_safe_w8a8,
)


class TestAscendW8A8FusedMoEMethod310(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 128

    @patch("vllm_ascend._310p.quantization.methods.w8a8_dynamic.get_ep_group")
    def setUp(self, mock_get_ep_group):
        with patch(
            "vllm_ascend._310p.quantization.methods.w8a8_dynamic.get_current_vllm_config"
        ) as mock_get_current_vllm_config:
            mock_vllm_config = Mock()
            mock_vllm_config.quant_config = Mock(quant_description={"group_size": 0})
            mock_vllm_config.scheduler_config = Mock(
                max_num_batched_tokens=2048, max_model_len=2048, enable_chunked_prefill=False
            )
            mock_get_current_vllm_config.return_value = mock_vllm_config
            mock_ep_group = Mock()
            mock_get_ep_group.return_value = mock_ep_group
            mock_ascend_config = Mock()

            mock_ascend_config.enable_chunked_prefill = False

            self.quant_method = AscendW8A8DynamicFusedMoEMethod310()

    def test_get_weight_310(self):
        param_dict = self.quant_method.get_weight(
            self.num_experts, self.intermediate_size, self.hidden_size, torch.float16
        )
        self.assertEqual(param_dict["w13_weight"].dtype, torch.int8)
        self.assertEqual(
            param_dict["w13_weight"].shape, (self.num_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.assertEqual(param_dict["w2_weight"].dtype, torch.int8)
        self.assertEqual(param_dict["w2_weight"].shape, (self.num_experts, self.hidden_size, self.intermediate_size))

    def test_get_dynamic_quant_param_310(self):
        param_dict = self.quant_method.get_dynamic_quant_param(
            self.num_experts, self.intermediate_size, self.hidden_size, torch.float16
        )
        self.assertEqual(param_dict["w13_weight_scale"].dtype, torch.float32)
        self.assertEqual(param_dict["w13_weight_scale"].shape, (self.num_experts, 2 * self.intermediate_size, 1))
        self.assertEqual(param_dict["w2_weight_scale"].dtype, torch.float32)
        self.assertEqual(param_dict["w2_weight_scale"].shape, (self.num_experts, self.hidden_size, 1))


class TestAscendW8A8DynamicLinearMethod310(TestBase):
    def setUp(self):
        self.method = AscendW8A8DynamicLinearMethod310()

    def test_get_weight_310(self):
        weight = self.method.get_weight(10, 20)
        self.assertEqual(weight["weight"].dtype, torch.int8)
        self.assertEqual(weight["weight"].shape, (20, 10))

    def test_enable_full_decode_does_not_change_piecewise_strategy(self):
        original_piecewise = AscendW8A8DynamicLinearMethod310._dflash_piecewise_graph_safe
        original_full_decode = AscendW8A8DynamicLinearMethod310._dflash_full_decode_graph_safe
        try:
            AscendW8A8DynamicLinearMethod310._dflash_piecewise_graph_safe = False
            AscendW8A8DynamicLinearMethod310._dflash_full_decode_graph_safe = False
            enable_dflash_full_decode_graph_safe_w8a8()
            self.assertFalse(AscendW8A8DynamicLinearMethod310._dflash_piecewise_graph_safe)
            self.assertTrue(AscendW8A8DynamicLinearMethod310._dflash_full_decode_graph_safe)
        finally:
            AscendW8A8DynamicLinearMethod310._dflash_piecewise_graph_safe = original_piecewise
            AscendW8A8DynamicLinearMethod310._dflash_full_decode_graph_safe = original_full_decode

    def test_enable_piecewise_does_not_change_full_decode_strategy(self):
        original_piecewise = AscendW8A8DynamicLinearMethod310._dflash_piecewise_graph_safe
        original_full_decode = AscendW8A8DynamicLinearMethod310._dflash_full_decode_graph_safe
        try:
            AscendW8A8DynamicLinearMethod310._dflash_piecewise_graph_safe = False
            AscendW8A8DynamicLinearMethod310._dflash_full_decode_graph_safe = False
            enable_dflash_piecewise_graph_safe_w8a8()
            self.assertTrue(AscendW8A8DynamicLinearMethod310._dflash_piecewise_graph_safe)
            self.assertFalse(AscendW8A8DynamicLinearMethod310._dflash_full_decode_graph_safe)
        finally:
            AscendW8A8DynamicLinearMethod310._dflash_piecewise_graph_safe = original_piecewise
            AscendW8A8DynamicLinearMethod310._dflash_full_decode_graph_safe = original_full_decode

    def test_get_perchannel_param_310(self):
        params = self.method.get_perchannel_param(10, torch.float32)

        self.assertEqual(params["weight_scale"].dtype, torch.float32)
        self.assertEqual(params["weight_offset"].dtype, torch.float32)

        self.assertEqual(params["weight_scale"].shape, (10, 1))
        self.assertEqual(params["weight_offset"].shape, (10, 1))

    @patch("torch_npu.npu_dynamic_quant", create=True)
    @patch("torch_npu.npu_quant_matmul")
    def test_apply_310(self, mock_npu_quant_matmul, mock_npu_dynamic_quantize):
        layer = MagicMock()
        layer.weight = torch.randn(128, 256, dtype=torch.float16)
        layer.weight_scale = torch.randn(128, dtype=torch.float32)
        layer.params_dtype = torch.float16

        x = torch.randn(32, 128, dtype=torch.float16)
        expect_x_output = torch.randint(-128, 127, x.shape, dtype=torch.int8)
        expect_pertoken_scale_output = torch.randn(x.shape[0], dtype=torch.float32)
        mock_npu_dynamic_quantize.return_value = expect_x_output, expect_pertoken_scale_output

        expected_y_output = torch.randn(32, 256)
        mock_npu_quant_matmul.return_value = expected_y_output

        output = self.method.apply(layer, x, tp_rank=0)

        mock_npu_dynamic_quantize.assert_called_with(x)
        mock_npu_quant_matmul.assert_called_once()
        (args, kwargs) = mock_npu_quant_matmul.call_args

        # positional args
        self.assertTrue(torch.equal(args[0], expect_x_output))
        self.assertTrue(torch.equal(args[1], layer.weight.data))
        self.assertTrue(torch.equal(args[2], layer.weight_scale))

        # kwargs
        self.assertTrue(torch.equal(kwargs["pertoken_scale"], expect_pertoken_scale_output))
        self.assertTrue(kwargs["bias"] is None)
        self.assertEqual(kwargs["output_dtype"], layer.params_dtype)

        self.assertTrue(torch.equal(output, expected_y_output))

    @patch("torch_npu.npu_dynamic_quant", create=True)
    @patch("torch_npu.npu_quant_matmul")
    def test_apply_chunks_large_token_dimension(self, mock_npu_quant_matmul, mock_npu_dynamic_quantize):
        layer = MagicMock()
        layer.weight = torch.randint(-127, 128, (128, 256), dtype=torch.int8)
        layer.weight_scale = torch.randn(256, dtype=torch.float32)

        num_tokens = _W8A8_DYNAMIC_MATMUL_CHUNK_SIZE + 1
        x = torch.randn(num_tokens, 128, dtype=torch.float16)
        quantized_x = torch.randint(-128, 127, x.shape, dtype=torch.int8)
        pertoken_scale = torch.randn(num_tokens, dtype=torch.float32)
        mock_npu_dynamic_quantize.return_value = quantized_x, pertoken_scale

        mock_npu_quant_matmul.side_effect = lambda input_, *_args, **_kwargs: torch.zeros(
            input_.shape[0], 256, dtype=x.dtype
        )

        output = self.method.apply(layer, x)

        self.assertEqual(mock_npu_quant_matmul.call_count, 2)
        first_call, second_call = mock_npu_quant_matmul.call_args_list
        self.assertEqual(first_call.args[0].shape[0], _W8A8_DYNAMIC_MATMUL_CHUNK_SIZE)
        self.assertEqual(second_call.args[0].shape[0], 1)
        self.assertTrue(
            torch.equal(
                first_call.kwargs["pertoken_scale"],
                pertoken_scale[:_W8A8_DYNAMIC_MATMUL_CHUNK_SIZE],
            )
        )
        self.assertTrue(
            torch.equal(
                second_call.kwargs["pertoken_scale"],
                pertoken_scale[_W8A8_DYNAMIC_MATMUL_CHUNK_SIZE:],
            )
        )
        self.assertEqual(output.shape, (num_tokens, 256))

    @patch("torch_npu.npu_dynamic_quant", create=True)
    @patch("torch_npu.npu_quant_matmul")
    def test_apply_piecewise_uses_shape_dynamic_two_way_chunks(
        self,
        mock_npu_quant_matmul,
        mock_npu_dynamic_quantize,
    ):
        self.method._dflash_piecewise_graph_safe = True
        layer = MagicMock()
        layer.weight = torch.randint(-127, 128, (128, 256), dtype=torch.int8)
        layer.weight_scale = torch.randn(256, dtype=torch.float32)

        num_tokens = 1280
        x = torch.randn(num_tokens, 128, dtype=torch.float16)
        quantized_x = torch.randint(-128, 127, x.shape, dtype=torch.int8)
        pertoken_scale = torch.randn(num_tokens, dtype=torch.float32)
        mock_npu_dynamic_quantize.return_value = quantized_x, pertoken_scale
        mock_npu_quant_matmul.side_effect = lambda input_, *_args, **_kwargs: torch.zeros(
            input_.shape[0], 256, dtype=x.dtype
        )

        output = self.method.apply(layer, x)

        self.assertEqual(mock_npu_quant_matmul.call_count, 2)
        first_call, second_call = mock_npu_quant_matmul.call_args_list
        self.assertEqual(first_call.args[0].shape[0], 640)
        self.assertEqual(second_call.args[0].shape[0], 640)
        self.assertTrue(torch.equal(first_call.kwargs["pertoken_scale"], pertoken_scale[:640]))
        self.assertTrue(torch.equal(second_call.kwargs["pertoken_scale"], pertoken_scale[640:]))
        self.assertEqual(output.shape, (num_tokens, 256))

    @patch("vllm_ascend._310p.quantization.methods.w8a8_dynamic.get_forward_context")
    @patch("torch_npu.npu_dynamic_quant", create=True)
    @patch("torch_npu.npu_quant_matmul")
    def test_apply_full_decode_uses_two_way_chunks_only_in_profile_run(
        self,
        mock_npu_quant_matmul,
        mock_npu_dynamic_quantize,
        mock_get_forward_context,
    ):
        self.method._dflash_piecewise_graph_safe = False
        self.method._dflash_full_decode_graph_safe = True
        mock_get_forward_context.return_value = SimpleNamespace(in_profile_run=True)
        layer = MagicMock()
        layer.weight = torch.randint(-127, 128, (128, 256), dtype=torch.int8)
        layer.weight_scale = torch.randn(256, dtype=torch.float32)

        num_tokens = 1280
        x = torch.randn(num_tokens, 128, dtype=torch.float16)
        quantized_x = torch.randint(-128, 127, x.shape, dtype=torch.int8)
        pertoken_scale = torch.randn(num_tokens, dtype=torch.float32)
        mock_npu_dynamic_quantize.return_value = quantized_x, pertoken_scale
        mock_npu_quant_matmul.side_effect = lambda input_, *_args, **_kwargs: torch.zeros(
            input_.shape[0], 256, dtype=x.dtype
        )

        output = self.method.apply(layer, x)

        self.assertEqual(
            [call.args[0].shape[0] for call in mock_npu_quant_matmul.call_args_list],
            [640, 640],
        )
        self.assertEqual(output.shape, (num_tokens, 256))

    @patch("vllm_ascend._310p.quantization.methods.w8a8_dynamic.get_forward_context")
    @patch("torch_npu.npu_dynamic_quant", create=True)
    @patch("torch_npu.npu_quant_matmul")
    def test_apply_full_decode_keeps_normal_forward_numerically_aligned(
        self,
        mock_npu_quant_matmul,
        mock_npu_dynamic_quantize,
        mock_get_forward_context,
    ):
        self.method._dflash_piecewise_graph_safe = False
        self.method._dflash_full_decode_graph_safe = True
        mock_get_forward_context.return_value = SimpleNamespace(in_profile_run=False)
        layer = MagicMock()
        layer.weight = torch.randint(-127, 128, (128, 256), dtype=torch.int8)
        layer.weight_scale = torch.randn(256, dtype=torch.float32)

        num_tokens = 95
        x = torch.randn(num_tokens, 128, dtype=torch.float16)
        quantized_x = torch.randint(-128, 127, x.shape, dtype=torch.int8)
        pertoken_scale = torch.randn(num_tokens, dtype=torch.float32)
        expected_output = torch.randn(num_tokens, 256, dtype=x.dtype)
        mock_npu_dynamic_quantize.return_value = quantized_x, pertoken_scale
        mock_npu_quant_matmul.return_value = expected_output

        output = self.method.apply(layer, x)

        mock_npu_quant_matmul.assert_called_once()
        self.assertTrue(torch.equal(mock_npu_quant_matmul.call_args.args[0], quantized_x))
        self.assertTrue(torch.equal(output, expected_output))

    @patch("vllm_ascend.utils.is_310p", return_value=True)
    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_calls_nz_format_cast_310p(self, mock_npu_format_cast, _mock_is_310p):
        mock_npu_format_cast.side_effect = lambda x, fmt: x

        layer = MagicMock()

        # Attributes used by process_weights_after_loading()
        layer.weight = MagicMock()
        layer.weight_scale = MagicMock()
        layer.weight_offset = MagicMock()

        layer.weight.data = torch.randint(-127, 128, (128, 256), dtype=torch.int8)

        layer.weight_scale.data = torch.randn(128, 1, dtype=torch.bfloat16)
        layer.weight_offset.data = torch.randn(128, 1, dtype=torch.bfloat16)
        # w2_weight_offset is reshaped to (N, -1); any (N, 1) is fine
        layer.w2_weight_offset.data = torch.randn(128, 1, dtype=torch.bfloat16)

        self.method.process_weights_after_loading(layer)

        mock_npu_format_cast.assert_called_once()
