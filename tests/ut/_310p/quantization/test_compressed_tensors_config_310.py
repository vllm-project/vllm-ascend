#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.linear import RowParallelLinear, UnquantizedLinearMethod

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import COMPRESSED_TENSORS_W8A8_CONFIG
from vllm_ascend._310p.quantization.compressed_tensors_config import (
    AscendCompressedTensorsConfig310,
    AscendCompressedTensorsW8A8DynamicLinearMethod310,
)
from vllm_ascend.quantization.configs.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.method_adapters import AscendLinearMethod


class TestAscendCompressedTensorsConfig310(TestBase):
    def setUp(self):
        self.config = AscendCompressedTensorsConfig310.from_config(deepcopy(COMPRESSED_TENSORS_W8A8_CONFIG))

    def test_only_float16_activations_supported(self):
        self.assertEqual(self.config.get_supported_act_dtypes(), [torch.float16])
        self.assertIn(torch.bfloat16, AscendCompressedTensorsConfig.get_supported_act_dtypes())

    @patch("vllm_ascend.quantization.method_adapters.AscendLinearMethod.__init__", return_value=None)
    def test_linear_method_uses_compressed_tensors_310p_scheme(self, _mock_init):
        layer = MagicMock(spec=RowParallelLinear)
        method = self.config.get_quant_method(layer, "model.layers.0.self_attn.q_proj")
        self.assertIsInstance(method, AscendLinearMethod)
        self.assertIsInstance(layer.scheme, AscendCompressedTensorsW8A8DynamicLinearMethod310)

    def test_ignored_lm_head_stays_unquantized(self):
        layer = MagicMock(spec=RowParallelLinear)
        method = self.config.get_quant_method(layer, "lm_head")
        self.assertIsInstance(method, UnquantizedLinearMethod)

    def test_per_group_format_overrides_global_format(self):
        metadata: dict[str, Any] = deepcopy(COMPRESSED_TENSORS_W8A8_CONFIG)
        metadata["format"] = "mixed-precision"
        config = AscendCompressedTensorsConfig310.from_config(metadata)
        layer = MagicMock(spec=RowParallelLinear)
        scheme = config._get_linear_scheme(layer, "model.layers.0.self_attn.q_proj")
        self.assertIsInstance(scheme, AscendCompressedTensorsW8A8DynamicLinearMethod310)

    def test_fused_qkv_targets_select_310p_scheme(self):
        metadata: dict[str, Any] = deepcopy(COMPRESSED_TENSORS_W8A8_CONFIG)
        prefix = "model.layers.0.self_attn."
        metadata["config_groups"]["group_0"]["targets"] = [prefix + name for name in ("q_proj", "k_proj", "v_proj")]
        config = AscendCompressedTensorsConfig310.from_config(metadata)
        config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
        layer = MagicMock(spec=RowParallelLinear)
        scheme = config._get_linear_scheme(layer, prefix + "qkv_proj")
        self.assertIsInstance(scheme, AscendCompressedTensorsW8A8DynamicLinearMethod310)

    def test_partial_fused_ignore_rejected(self):
        metadata: dict[str, Any] = deepcopy(COMPRESSED_TENSORS_W8A8_CONFIG)
        metadata["ignore"] = ["model.layers.0.self_attn.q_proj"]
        config = AscendCompressedTensorsConfig310.from_config(metadata)
        config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
        layer = MagicMock(spec=RowParallelLinear)
        with self.assertRaises(ValueError):
            config._get_linear_scheme(layer, "model.layers.0.self_attn.qkv_proj")

    @patch("vllm_ascend._310p.quantization.compressed_tensors_config.get_scheme_class", return_value=None)
    def test_missing_local_scheme_does_not_fall_back(self, _mock_registry):
        scheme = self.config.target_scheme_map["Linear"]
        with self.assertRaisesRegex(NotImplementedError, "No 310P compressed-tensors"):
            self.config._create_scheme_for_layer_type(scheme["weights"], scheme["input_activations"], None, "linear")

    def test_moe_rejected_before_construction(self):
        scheme = self.config.target_scheme_map["Linear"]
        with self.assertRaisesRegex(NotImplementedError, "layer_type=moe"):
            self.config._create_scheme_for_layer_type(scheme["weights"], scheme["input_activations"], None, "moe")

    def test_unsupported_quantization_combinations(self):
        mutations = (
            {"weights": {"type": "float"}},
            {"input_activations": {"type": "float"}},
            {"weights": {"type": "float"}, "input_activations": {"type": "float"}},
            {"weights": {"num_bits": 4}},
            {"weights": {"symmetric": False}},
            {"input_activations": {"symmetric": False}},
            {"input_activations": {"dynamic": False, "strategy": "tensor"}},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                metadata: dict[str, Any] = deepcopy(COMPRESSED_TENSORS_W8A8_CONFIG)
                group = metadata["config_groups"]["group_0"]
                for target, values in mutation.items():
                    group[target].update(values)
                config = AscendCompressedTensorsConfig310.from_config(metadata)
                scheme = config.target_scheme_map["Linear"]
                with self.assertRaises(NotImplementedError):
                    config._create_scheme_for_layer_type(
                        scheme["weights"], scheme["input_activations"], scheme["format"], "linear"
                    )

    def test_naive_format_not_claimed_supported(self):
        scheme = self.config.target_scheme_map["Linear"]
        with self.assertRaisesRegex(NotImplementedError, "format=naive-quantized"):
            self.config._create_scheme_for_layer_type(
                scheme["weights"], scheme["input_activations"], "naive-quantized", "linear"
            )


class TestCompressedTensorsNDLinear310(TestBase):
    def setUp(self):
        self.method = AscendCompressedTensorsW8A8DynamicLinearMethod310()
        self.layer = SimpleNamespace(
            weight=SimpleNamespace(data=torch.randint(-127, 128, (128, 256), dtype=torch.int8).t()),
            weight_scale=torch.randn(256, 1, dtype=torch.float32),
            weight_offset=torch.zeros(256, 1, dtype=torch.float32),
        )

    def test_loaded_weight_preserves_nk_values_and_layout(self):
        original = self.layer.weight.data.clone()
        self.assertFalse(self.layer.weight.data.is_contiguous())
        self.method.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.weight.data.shape, (256, 128))
        self.assertTrue(self.layer.weight.data.is_contiguous())
        self.assertTrue(torch.equal(self.layer.weight.data, original))
        self.assertEqual(self.layer.weight_scale.shape, (256,))
        self.assertEqual(self.layer.weight_offset.shape, (256,))

    @patch("torch_npu.npu_quant_matmul_dequant", create=True)
    def test_noncontiguous_input_and_bias(self, mock_kernel):
        x = torch.randn(128, 32, dtype=torch.float16).t()
        bias = torch.randn(256, dtype=torch.float16)
        kernel_output = torch.randn(32, 256, dtype=torch.float16)
        mock_kernel.return_value = kernel_output
        output = self.method.apply(self.layer, x, bias=bias)
        args, kwargs = mock_kernel.call_args
        self.assertTrue(args[0].is_contiguous())
        self.assertTrue(torch.equal(args[0], x))
        self.assertNotIn("bias", kwargs)
        self.assertEqual(kwargs["quant_mode"], "pertoken")
        self.assertEqual(output.shape, (32, 256))
        self.assertTrue(torch.equal(output, kernel_output + bias))

    @patch("torch_npu.npu_quant_matmul_dequant", create=True)
    def test_batched_shape_restored(self, mock_kernel):
        x = torch.randn(2, 4, 128, dtype=torch.float16)
        mock_kernel.return_value = torch.randn(8, 256, dtype=torch.float16)
        output = self.method.apply(self.layer, x)
        self.assertEqual(mock_kernel.call_args.args[0].shape, (8, 128))
        self.assertEqual(output.shape, (2, 4, 256))

    @patch("torch_npu.npu_quant_matmul_dequant", create=True)
    def test_bfloat16_rejected_before_kernel(self, mock_kernel):
        with self.assertRaisesRegex(ValueError, "float16 activations"):
            self.method.apply(self.layer, torch.randn(2, 128, dtype=torch.bfloat16))
        mock_kernel.assert_not_called()
