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

from unittest.mock import MagicMock, patch

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig
from vllm.model_executor.layers.linear import LinearBase

from tests.ut.base import TestBase
from vllm_ascend._310p.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod310
from vllm_ascend._310p.ops.linear import AscendUnquantizedLinearMethod310
from vllm_ascend._310p.quantization.modelslim_config import AscendModelSlimConfig310


class TestAscendModelSlimConfig310(TestBase):
    def setUp(self):
        self.sample_config = {
            "weight": "INT8",
            "layer1.weight": "INT8",
            "layer2.weight": "FLOAT",
            "fused_layer.weight": "FLOAT",
            "fused_layer.shard1.weight": "FLOAT",
            "fused_layer.shard2.weight": "FLOAT",
            "shard1.weight": "FLOAT",
            "shard2.weight": "FLOAT",
        }
        self.ascend_config = AscendModelSlimConfig310(self.sample_config)
        self.ascend_config.packed_modules_mapping = None

    def test_get_quant_method_for_linear_310(self):
        mock_config = MagicMock()
        mock_config.model_config.hf_config.model_type = None
        linear_layer = MagicMock(spec=LinearBase)
        # Test skipped layer
        with (
            patch("vllm_ascend._310p.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config),
            patch.object(self.ascend_config, "is_layer_skipped_ascend", return_value=True),
        ):
            method = self.ascend_config.get_quant_method(linear_layer, ".attn")
            self.assertIsInstance(method, AscendUnquantizedLinearMethod310)

        # Test quantized layer
        mock_scheme = MagicMock()
        with (
            patch.object(self.ascend_config, "is_layer_skipped_ascend", return_value=False),
            patch("vllm_ascend._310p.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config),
            patch("vllm_ascend._310p.quantization.modelslim_config.create_scheme_for_layer", return_value=mock_scheme),
            patch(
                "vllm_ascend._310p.quantization.modelslim_config.AscendLinearMethod", return_value=MagicMock()
            ) as mock_ascend_linear,
        ):
            method = self.ascend_config.get_quant_method(linear_layer, ".attn")
            self.assertIs(method, mock_ascend_linear.return_value)
            mock_ascend_linear.assert_called_once_with(mock_scheme)

    def test_get_quant_method_for_fused_moe_310(self):
        fused_moe_layer = MagicMock(spec=FusedMoE)
        fused_moe_layer.moe = MagicMock(spec=FusedMoEConfig)
        fused_moe_layer.moe_config = MagicMock(spec=FusedMoEConfig)
        fused_moe_layer.moe_config.moe_parallel_config = MagicMock(spec=FusedMoEParallelConfig)
        fused_moe_layer.moe_config.moe_parallel_config.use_ep = True
        fused_moe_layer.moe_config.moe_parallel_config.dp_size = 1
        mock_config = MagicMock()
        mock_config.model_config.hf_config.model_type = None
        mock_config.compilation_config.custom_ops = ["all"]
        mock_scheme = MagicMock()
        # Test skipped layer
        with (
            patch("vllm.config.vllm.get_current_vllm_config", return_value=mock_config),
            patch("vllm_ascend._310p.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config),
            patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config),
            patch.object(self.ascend_config, "is_layer_skipped_ascend", return_value=True),
        ):
            method = self.ascend_config.get_quant_method(fused_moe_layer, ".moe")
            self.assertIsInstance(method, AscendUnquantizedFusedMoEMethod310)

        # Test quantized layer
        mock_scheme = MagicMock()
        with (
            patch.object(self.ascend_config, "is_layer_skipped_ascend", return_value=False),
            patch("vllm.config.vllm.get_current_vllm_config", return_value=mock_config),
            patch("vllm_ascend._310p.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config),
            patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config),
            patch("vllm_ascend._310p.quantization.modelslim_config.create_scheme_for_layer", return_value=mock_scheme),
            patch(
                "vllm_ascend._310p.quantization.modelslim_config.AscendFusedMoEMethod", return_value=MagicMock()
            ) as fused_moe_method,
        ):
            method = self.ascend_config.get_quant_method(fused_moe_layer, ".moe")
            self.assertIs(method, fused_moe_method.return_value)
            fused_moe_method.assert_called_once_with(mock_scheme, fused_moe_layer.moe_config)
