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
# This file is a part of the vllm-ascend project.
#

from unittest.mock import MagicMock, patch

from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy, QuantizationType
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import COMPRESSED_TENSORS_W8A8_CONFIG
from vllm_ascend._310p.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod310
from vllm_ascend._310p.ops.vocab_parallel_embedding import AscendUnquantizedEmbeddingMethod310
from vllm_ascend._310p.quantization.compressed_tensors_config import AscendCompressedTensorsConfig310
from vllm_ascend._310p.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod310,
    AscendW8A8DynamicLinearMethod310,
)
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod, AscendLinearMethod


class TestAscendCompressedTensorsConfig310(TestBase):
    def setUp(self):
        self.config = AscendCompressedTensorsConfig310.from_config(COMPRESSED_TENSORS_W8A8_CONFIG)

    def test_get_quant_method_for_linear(self):
        layer = MagicMock(spec=RowParallelLinear)

        quantized = self.config.get_quant_method(layer, "model.layers.0.self_attn.q_proj")
        self.assertIsInstance(quantized, AscendLinearMethod)
        self.assertIsInstance(quantized.quant_method, AscendW8A8DynamicLinearMethod310)

        ignored = self.config.get_quant_method(layer, "lm_head")
        self.assertIsInstance(ignored, AscendUnquantizedLinearMethod)

    def test_get_quant_method_for_embedding(self):
        layer = MagicMock(spec=VocabParallelEmbedding)

        result = self.config.get_quant_method(layer, "model.embed_tokens")
        self.assertIsInstance(result, AscendUnquantizedEmbeddingMethod310)

        with patch.object(self.config, "get_scheme_dict", return_value={"weights": MagicMock()}):
            with self.assertRaises(NotImplementedError):
                self.config.get_quant_method(layer, "model.embed_tokens")

    def test_get_quant_method_for_fused_moe(self):
        layer = MagicMock(spec=RoutedExperts)
        layer.moe_config = MagicMock(spec=FusedMoEConfig)
        layer.moe_config.moe_parallel_config = MagicMock(spec=FusedMoEParallelConfig)
        layer.moe_config.moe_parallel_config.use_ep = False
        layer.moe_config.moe_parallel_config.dp_size = 1

        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.dtype = "float16"
        mock_vllm_config.compilation_config.custom_ops = ["all"]

        with (
            patch("vllm.config.vllm.get_current_vllm_config", return_value=mock_vllm_config),
            patch.object(self.config, "_get_moe_scheme", return_value=None),
        ):
            unquantized = self.config.get_quant_method(layer, "model.layers.0.mlp.experts")
            self.assertIsInstance(unquantized, AscendUnquantizedFusedMoEMethod310)

        with (
            patch("vllm_ascend._310p.quantization.methods.w8a8_dynamic.get_ep_group", return_value=MagicMock()),
            patch(
                "vllm_ascend._310p.quantization.methods.w8a8_dynamic.get_current_vllm_config",
                return_value=mock_vllm_config,
            ),
        ):
            quantized = self.config.get_quant_method(layer, "model.layers.0.mlp.experts")
            self.assertIsInstance(quantized, AscendFusedMoEMethod)
            self.assertIsInstance(quantized.quant_method, AscendW8A8DynamicFusedMoEMethod310)

    def test_unsupported_schemes_raise_not_implemented(self):
        cases = [
            # Static W8A8 is ModelSlim-only on 310P and must not silently misroute.
            (
                QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.CHANNEL.value,
                    symmetric=True,
                    dynamic=False,
                ),
                QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.TENSOR.value,
                    symmetric=True,
                    dynamic=False,
                ),
                "int-quantized",
            ),
            # W4A16 is not registered on 310P.
            (
                QuantizationArgs(
                    num_bits=4,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.GROUP.value,
                    group_size=128,
                    symmetric=True,
                    dynamic=False,
                ),
                None,
                "pack-quantized",
            ),
        ]
        for weight_quant, input_quant, fmt in cases:
            with self.assertRaises(NotImplementedError):
                self.config._create_scheme_for_layer_type(
                    weight_quant=weight_quant,
                    input_quant=input_quant,
                    format=fmt,
                    layer_type="linear",
                )
