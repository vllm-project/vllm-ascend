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

import pytest
import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.utils import WeightsMapper

from tests.ut.base import TestBase
from vllm_ascend._310p.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod310
from vllm_ascend._310p.quantization.methods.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod310
from vllm_ascend._310p.quantization.modelslim_config import AscendModelSlimConfig310
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod


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
        self.ascend_config.packed_modules_mapping = {}

    def test_get_quant_method_for_linear_310(self):
        mock_config = MagicMock()
        mock_config.model_config.hf_config.model_type = None
        linear_layer = MagicMock(spec=LinearBase)
        # Test skipped layer
        with (
            patch("vllm_ascend._310p.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config),
            patch("vllm_ascend._310p.quantization.modelslim_config.get_quant_type_for_layer", return_value=None),
        ):
            method = self.ascend_config.get_quant_method(linear_layer, ".attn")
            self.assertIsInstance(method, AscendUnquantizedLinearMethod)

        # Test quantized layer
        mock_scheme = MagicMock()
        with (
            patch("vllm_ascend._310p.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config),
            patch("vllm_ascend._310p.quantization.modelslim_config.get_quant_type_for_layer", return_value="INT8"),
            patch("vllm_ascend._310p.quantization.modelslim_config.create_scheme_for_layer", return_value=mock_scheme),
            patch(
                "vllm_ascend._310p.quantization.modelslim_config.AscendLinearMethod", return_value=MagicMock()
            ) as mock_ascend_linear,
        ):
            method = self.ascend_config.get_quant_method(linear_layer, ".attn")
            self.assertIs(method, mock_ascend_linear.return_value)
            mock_ascend_linear.assert_called_once_with(mock_scheme)

    def test_get_quant_method_maps_lm_head_prefix_310(self):
        config = AscendModelSlimConfig310({"language_model.lm_head.weight": "INT8"})
        linear_layer = MagicMock(spec=LinearBase)
        mock_config = MagicMock()
        mock_config.model_config.hf_config.model_type = "qwen3_5_moe"
        mock_scheme = MagicMock()

        with (
            patch("vllm_ascend._310p.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config),
            patch("vllm_ascend._310p.quantization.modelslim_config.get_quant_type_for_layer", return_value="INT8"),
            patch(
                "vllm_ascend._310p.quantization.modelslim_config.create_scheme_for_layer",
                return_value=mock_scheme,
            ) as mock_create_scheme,
            patch("vllm_ascend._310p.quantization.modelslim_config.AscendLinearMethod", return_value=MagicMock()),
        ):
            config.get_quant_method(linear_layer, "lm_head")

        mock_create_scheme.assert_called_once_with(
            "INT8",
            "language_model.lm_head",
            "linear",
        )

    def test_get_quant_method_for_fused_moe_310(self):
        fused_moe_layer = MagicMock(spec=RoutedExperts)
        fused_moe_layer.moe = MagicMock(spec=FusedMoEConfig)
        fused_moe_layer.moe_config = MagicMock(spec=FusedMoEConfig)
        fused_moe_layer.moe_config.moe_backend = "auto"
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
            patch(
                "vllm_ascend.quantization.configs.modelslim_config.get_current_vllm_config",
                return_value=mock_config,
            ),
            patch("vllm_ascend._310p.quantization.modelslim_config.get_quant_type_for_layer", return_value=None),
        ):
            method = self.ascend_config.get_quant_method(fused_moe_layer, ".moe")
            self.assertIsInstance(method, AscendUnquantizedFusedMoEMethod310)

        # Test quantized layer
        mock_scheme = MagicMock()
        with (
            patch("vllm.config.vllm.get_current_vllm_config", return_value=mock_config),
            patch("vllm_ascend._310p.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config),
            patch(
                "vllm_ascend.quantization.configs.modelslim_config.get_current_vllm_config",
                return_value=mock_config,
            ),
            patch(
                "vllm_ascend._310p.quantization.modelslim_config.get_quant_type_for_layer", return_value="W8A8_DYNAMIC"
            ),
            patch("vllm_ascend._310p.quantization.modelslim_config.create_scheme_for_layer", return_value=mock_scheme),
            patch(
                "vllm_ascend._310p.quantization.modelslim_config.AscendFusedMoEMethod", return_value=MagicMock()
            ) as fused_moe_method,
        ):
            method = self.ascend_config.get_quant_method(fused_moe_layer, ".moe")
            self.assertIs(method, fused_moe_method.return_value)
            fused_moe_method.assert_called_once_with(mock_scheme, fused_moe_layer.moe_config)


@pytest.mark.parametrize(
    ("model_type", "runtime_prefix", "checkpoint_prefix", "mapped"),
    [
        pytest.param("minimax_m2", "mlp.experts", "block_sparse_moe", False, id="checkpoint"),
        pytest.param("minimax_m2", "mlp.experts", "block_sparse_moe", True, id="transformers-mapped"),
        pytest.param("minimax_m2", "mlp.experts.7", "block_sparse_moe", False, id="numbered-expert"),
        pytest.param("minimax_m2", "block_sparse_moe.experts", "block_sparse_moe", False, id="native-prefix"),
        pytest.param("qwen3_moe", "mlp.experts", "mlp", False, id="qwen"),
        pytest.param(
            "minimax_m2", "aux_mlp.mlp.experts", "aux_mlp.block_sparse_moe", False, id="unrelated-path-segment"
        ),
    ],
)
def test_get_quant_method_resolves_moe_checkpoint_prefix_310(model_type, runtime_prefix, checkpoint_prefix, mapped):
    layer = MagicMock(spec=RoutedExperts)
    layer.moe_config = MagicMock(spec=FusedMoEConfig)
    mock_config = MagicMock()
    mock_config.model_config.hf_config.model_type = model_type
    mock_config.model_config.dtype = torch.float16
    projections = ("gate_proj", "up_proj", "down_proj") if model_type == "qwen3_moe" else ("w1", "w2", "w3")
    config = AscendModelSlimConfig310(
        {
            f"model.layers.0.{checkpoint_prefix}.experts.0.{projection}.weight": "W8A8_DYNAMIC"
            for projection in projections
        }
    )
    if mapped:
        config.apply_vllm_mapper(WeightsMapper(orig_to_new_substr={".block_sparse_moe.": ".mlp."}))

    with (
        patch("vllm_ascend._310p.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config),
        patch("vllm_ascend._310p.quantization.methods.w8a8_dynamic.get_current_vllm_config", return_value=mock_config),
        patch("vllm_ascend._310p.quantization.methods.w8a8_dynamic.get_ep_group"),
    ):
        method = config.get_quant_method(layer, f"model.layers.0.{runtime_prefix}")

    assert isinstance(method, AscendFusedMoEMethod)
    assert isinstance(method.quant_method, AscendW8A8DynamicFusedMoEMethod310)


@pytest.mark.parametrize("case", ["float", "packed-float", "incomplete", "mixed", "non-minimax"])
def test_get_quant_method_preserves_runtime_moe_metadata_310(case):
    runtime_prefix = "model.layers.0.mlp.experts"
    alias_prefix = "model.layers.0.block_sparse_moe.experts"
    quant_description = {f"{alias_prefix}.0.{proj}.weight": "W8A8_DYNAMIC" for proj in ("w1", "w2", "w3")}
    if case in ("float", "mixed"):
        quant_description.update({f"{runtime_prefix}.0.{proj}.weight": "FLOAT" for proj in ("w1", "w2", "w3")})
    if case == "mixed":
        quant_description[f"{runtime_prefix}.0.w1.weight"] = "W8A8_DYNAMIC"
    if case == "incomplete":
        quant_description[f"{runtime_prefix}.0.w1.weight"] = "W8A8_DYNAMIC"
    if case == "packed-float":
        quant_description[f"{runtime_prefix}.weight"] = "FLOAT"

    config = AscendModelSlimConfig310(quant_description)
    mock_config = MagicMock()
    mock_config.model_config.hf_config.model_type = "qwen3_moe" if case == "non-minimax" else "minimax_m2"
    mock_config.compilation_config.custom_ops = ["all"]
    layer = MagicMock(spec=RoutedExperts)
    layer.moe_config = MagicMock(spec=FusedMoEConfig)
    layer.moe_config.moe_backend = "auto"

    with (
        patch("vllm.config.vllm.get_current_vllm_config", return_value=mock_config),
        patch("vllm_ascend._310p.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config),
    ):
        if case in ("float", "packed-float"):
            method = config.get_quant_method(layer, runtime_prefix)
            assert isinstance(method, AscendUnquantizedFusedMoEMethod310)
        elif case == "mixed":
            with pytest.raises(ValueError, match="Not all shards"):
                config.get_quant_method(layer, runtime_prefix)
        else:
            with pytest.raises(KeyError) as exc_info:
                config.get_quant_method(layer, runtime_prefix)
            missing_shard = "w2" if case == "incomplete" else "w1"
            assert exc_info.value.args == (f"{runtime_prefix}.0.{missing_shard}.weight",)
