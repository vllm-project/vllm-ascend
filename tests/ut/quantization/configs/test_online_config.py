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
"""Ascend's online quantization config: validation and unquantized fallback."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt8StaticChannelSym,
)

from tests.ut.base import TestBase
from vllm_ascend.device.hardware_profile import QuantizationBackendFamily
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.configs import online_config
from vllm_ascend.quantization.configs.online_config import AscendOnlineQuantizationConfig
from vllm_ascend.quantization.methods.online.int8 import AscendInt8PerChannelOnlineLinearMethod

MODULE = "vllm_ascend.quantization.configs.online_config"

_STANDARD = SimpleNamespace(quantization_backend_family=QuantizationBackendFamily.STANDARD)
_COMPATIBILITY = SimpleNamespace(quantization_backend_family=QuantizationBackendFamily.COMPATIBILITY)

_VLLM_CONFIG = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))


def _int8_args(**kwargs) -> QuantizationConfigArgs:
    return QuantizationConfigArgs(
        linear=QuantSpec(weight=kInt8StaticChannelSym),
        moe=QuantSpec(weight=kInt8StaticChannelSym),
        **kwargs,
    )


def _profile(profile):
    return patch(f"{MODULE}.get_current_hardware_profile", return_value=profile)


def _patch_vllm_config():
    """`OnlineLinearBase.__init__` records the model dtype from this helper."""
    return patch(
        "vllm.model_executor.layers.quantization.online.fp8.get_current_vllm_config",
        return_value=_VLLM_CONFIG,
    )


class _LinearLayer(LinearBase):
    """Minimal LinearBase subclass: vLLM resolves targets from the layer type."""

    def __init__(self, prefix: str = "model.layers.0.mlp.down_proj") -> None:
        torch.nn.Module.__init__(self)
        self.prefix = prefix
        self.tp_size = 1
        self.input_size = 64
        self.input_size_per_partition = 64


class TestAscendOnlineConfigInit(TestBase):
    def test_missing_config_is_rejected(self):
        with _profile(_STANDARD), self.assertRaises(ValueError) as raised:
            AscendOnlineQuantizationConfig()

        self.assertIn("--quantization-config", str(raised.exception))

    def test_non_args_config_is_rejected(self):
        with _profile(_STANDARD), self.assertRaises(TypeError):
            AscendOnlineQuantizationConfig({"moe": "int8_per_channel_static"})

    def test_supported_int8_spec_is_accepted(self):
        with _profile(_STANDARD):
            config = AscendOnlineQuantizationConfig(_int8_args())

        self.assertIsInstance(config, OnlineQuantizationConfig)

    def test_non_standard_quantization_backend_is_rejected(self):
        with _profile(_COMPATIBILITY), self.assertRaises(NotImplementedError):
            AscendOnlineQuantizationConfig(_int8_args())

    def test_unsupported_linear_weight_key_is_rejected(self):
        args = QuantizationConfigArgs(linear=QuantSpec(weight=kFp8StaticChannelSym))

        with _profile(_STANDARD), self.assertRaises(ValueError) as raised:
            AscendOnlineQuantizationConfig(args)

        self.assertIn("does not support linear weight=", str(raised.exception))
        self.assertIn("int8_per_channel_static", str(raised.exception))

    def test_unsupported_moe_weight_key_is_rejected(self):
        args = QuantizationConfigArgs(moe=QuantSpec(weight=kFp8StaticChannelSym))

        with _profile(_STANDARD), self.assertRaises(ValueError):
            AscendOnlineQuantizationConfig(args)

    def test_deprecated_fp8_alias_spec_is_rejected(self):
        # `--quantization fp8` on a checkpoint without FP8 metadata resolves to
        # the fp8_per_tensor recipe, which Ascend cannot execute.
        args = QuantizationConfigArgs(
            linear=QuantSpec(weight=kFp8StaticTensorSym),
            moe=QuantSpec(weight=kFp8StaticTensorSym),
        )

        with _profile(_STANDARD), self.assertRaises(ValueError):
            AscendOnlineQuantizationConfig(args)

    def test_target_dict_spec_is_rejected_for_an_unsupported_key(self):
        args = QuantizationConfigArgs(targets={"*mlp.experts*": {"weight": "fp8_per_tensor_static"}})

        with _profile(_STANDARD), self.assertRaises(ValueError) as raised:
            AscendOnlineQuantizationConfig(args)

        self.assertIn("targets[*mlp.experts*]", str(raised.exception))

    def test_target_shorthand_name_is_rejected_for_an_unsupported_recipe(self):
        args = QuantizationConfigArgs(targets={"*mlp*": "mxfp8"})

        with _profile(_STANDARD), self.assertRaises(ValueError):
            AscendOnlineQuantizationConfig(args)

    def test_activation_only_target_is_accepted(self):
        # No weight key: this overrides the checkpoint method, not an online
        # recipe, so Ascend leaves it alone.
        args = QuantizationConfigArgs(targets={"*mlp*": {"activation": "int8_per_channel_static"}})

        with _profile(_STANDARD):
            AscendOnlineQuantizationConfig(args)

    def test_explicit_activation_on_an_online_spec_is_rejected(self):
        args = QuantizationConfigArgs(linear=QuantSpec(weight=kInt8StaticChannelSym, activation=kInt8StaticChannelSym))

        with _profile(_STANDARD), self.assertRaises(NotImplementedError):
            AscendOnlineQuantizationConfig(args)

    def test_min_capability_is_zero(self):
        self.assertEqual(AscendOnlineQuantizationConfig.get_min_capability(), 0)


class TestAscendOnlineConfigMethods(TestBase):
    def _config(self) -> AscendOnlineQuantizationConfig:
        with _profile(_STANDARD):
            return AscendOnlineQuantizationConfig(QuantizationConfigArgs(moe=QuantSpec(weight=kInt8StaticChannelSym)))

    def test_a_skipped_linear_layer_uses_the_ascend_unquantized_method(self):
        layer = MagicMock(spec=LinearBase)

        with patch.object(OnlineQuantizationConfig, "resolve_quant_method_cls", return_value=None):
            method = self._config().get_quant_method(layer, "model.layers.0.mlp.down_proj")

        self.assertIsInstance(method, AscendUnquantizedLinearMethod)

    def test_a_skipped_moe_layer_uses_the_ascend_unquantized_method(self):
        layer = MagicMock(spec=RoutedExperts)
        layer.moe_config = sentinel = object()

        with (
            patch.object(OnlineQuantizationConfig, "resolve_quant_method_cls", return_value=None),
            patch.object(online_config, "AscendUnquantizedFusedMoEMethod") as ascend_moe,
        ):
            method = self._config().get_quant_method(layer, "model.layers.0.mlp.experts")

        ascend_moe.assert_called_once_with(moe=sentinel)
        self.assertIs(method, ascend_moe.return_value)

    def test_a_matched_layer_is_left_to_upstream(self):
        with _profile(_STANDARD):
            config = AscendOnlineQuantizationConfig(
                QuantizationConfigArgs(linear=QuantSpec(weight=kInt8StaticChannelSym))
            )
        layer = _LinearLayer("model.layers.0.mlp.down_proj")

        with _patch_vllm_config():
            method = config.get_quant_method(layer, "model.layers.0.mlp.down_proj")

        self.assertIsInstance(method, AscendInt8PerChannelOnlineLinearMethod)
        self.assertIn("model.layers.0.mlp.down_proj", config.quantized_layers)


if __name__ == "__main__":
    unittest.main()
