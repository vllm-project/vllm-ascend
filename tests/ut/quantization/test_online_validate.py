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
"""Online quantization requests are rejected before the model is built."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt8StaticChannelSym,
)

from tests.ut.base import TestBase
from vllm_ascend.device.hardware_profile import QuantizationBackendFamily
from vllm_ascend.quantization.online_validate import (
    is_online_quantization_request,
    validate_online_quantization,
)

MODULE = "vllm_ascend.quantization.online_validate"

_STANDARD = SimpleNamespace(quantization_backend_family=QuantizationBackendFamily.STANDARD)
_COMPATIBILITY = SimpleNamespace(quantization_backend_family=QuantizationBackendFamily.COMPATIBILITY)


def _int8_args(**kwargs) -> QuantizationConfigArgs:
    return QuantizationConfigArgs(
        linear=QuantSpec(weight=kInt8StaticChannelSym),
        moe=QuantSpec(weight=kInt8StaticChannelSym),
        **kwargs,
    )


class _OnlineConfig:
    """Stand-in for vLLM's `OnlineQuantizationConfig`."""

    def __init__(self, args) -> None:
        self.args = args
        self.online_quantization_config = None

    def get_name(self) -> str:
        return "online"


class _CheckpointConfig:
    """Stand-in for a checkpoint quantization format."""

    def __init__(self, name: str, overlay=None) -> None:
        self._name = name
        self.online_quantization_config = overlay

    def get_name(self) -> str:
        return self._name


class TestOnlineRequestDetection(TestBase):
    def test_online_config_is_a_request(self):
        self.assertTrue(is_online_quantization_request(_OnlineConfig(_int8_args()), None))

    def test_overlay_turns_a_checkpoint_into_a_request(self):
        overlay = SimpleNamespace(args=_int8_args())

        self.assertTrue(is_online_quantization_request(_CheckpointConfig("compressed-tensors"), overlay))

    def test_plain_checkpoint_is_not_a_request(self):
        self.assertFalse(is_online_quantization_request(_CheckpointConfig("compressed-tensors"), None))
        self.assertFalse(is_online_quantization_request(None, None))


class TestValidateOnlineQuantization(TestBase):
    def _profile(self, profile):
        return patch(f"{MODULE}.get_current_hardware_profile", return_value=profile)

    @staticmethod
    def _vllm_config(quant_config, model_args=None):
        return SimpleNamespace(
            model_config=SimpleNamespace(quantization="online", quantization_config=model_args),
            quant_config=quant_config,
        )

    def test_supported_int8_spec_is_accepted(self):
        with self._profile(_STANDARD):
            validate_online_quantization(self._vllm_config(_OnlineConfig(_int8_args())))

    def test_unsupported_linear_weight_key_is_rejected(self):
        config = self._vllm_config(
            _OnlineConfig(QuantizationConfigArgs(linear=QuantSpec(weight=kFp8StaticChannelSym))),
        )

        with self._profile(_STANDARD), self.assertRaises(ValueError) as raised:
            validate_online_quantization(config)

        self.assertIn("does not support linear weight=", str(raised.exception))
        self.assertIn("int8_per_channel_static", str(raised.exception))

    def test_unsupported_moe_weight_key_is_rejected(self):
        config = self._vllm_config(
            _OnlineConfig(QuantizationConfigArgs(moe=QuantSpec(weight=kFp8StaticChannelSym))),
        )

        with self._profile(_STANDARD), self.assertRaises(ValueError):
            validate_online_quantization(config)

    def test_deprecated_fp8_alias_is_rejected(self):
        # `--quantization fp8` on a checkpoint without FP8 metadata resolves to
        # the fp8_per_tensor recipe, which Ascend cannot execute.
        args = QuantizationConfigArgs(
            linear=QuantSpec(weight=kFp8StaticTensorSym),
            moe=QuantSpec(weight=kFp8StaticTensorSym),
        )

        with self._profile(_STANDARD), self.assertRaises(ValueError):
            validate_online_quantization(self._vllm_config(_OnlineConfig(args)))

    def test_target_dict_spec_is_rejected_for_an_unsupported_key(self):
        config = self._vllm_config(
            _OnlineConfig(QuantizationConfigArgs(targets={"*mlp.experts*": {"weight": "fp8_per_tensor_static"}})),
        )

        with self._profile(_STANDARD), self.assertRaises(ValueError) as raised:
            validate_online_quantization(config)

        self.assertIn("targets[*mlp.experts*]", str(raised.exception))

    def test_target_shorthand_name_is_rejected_for_an_unsupported_recipe(self):
        config = self._vllm_config(_OnlineConfig(QuantizationConfigArgs(targets={"*mlp*": "mxfp8"})))

        with self._profile(_STANDARD), self.assertRaises(ValueError):
            validate_online_quantization(config)

    def test_activation_only_target_stays_with_the_checkpoint_method(self):
        config = self._vllm_config(
            _OnlineConfig(QuantizationConfigArgs(targets={"*mlp*": {"activation": "int8_per_channel_static"}})),
        )

        with self._profile(_STANDARD):
            validate_online_quantization(config)

    def test_explicit_activation_on_an_online_spec_is_rejected(self):
        config = self._vllm_config(
            _OnlineConfig(
                QuantizationConfigArgs(linear=QuantSpec(weight=kInt8StaticChannelSym, activation=kInt8StaticChannelSym))
            ),
        )

        with self._profile(_STANDARD), self.assertRaises(NotImplementedError):
            validate_online_quantization(config)

    def test_plain_checkpoint_format_is_not_validated(self):
        with self._profile(_STANDARD):
            validate_online_quantization(self._vllm_config(_CheckpointConfig("ascend")))

    def test_composed_online_overlay_is_validated(self):
        overlay = SimpleNamespace(args=QuantizationConfigArgs(linear=QuantSpec(weight=kFp8StaticChannelSym)))
        config = self._vllm_config(_CheckpointConfig("compressed-tensors", overlay))

        with self._profile(_STANDARD), self.assertRaises(ValueError):
            validate_online_quantization(config)

    def test_composed_online_overlay_with_int8_spec_is_accepted(self):
        overlay = SimpleNamespace(args=_int8_args())
        config = self._vllm_config(_CheckpointConfig("ascend", overlay))

        with self._profile(_STANDARD):
            validate_online_quantization(config)

    def test_non_standard_quantization_backend_is_rejected(self):
        config = self._vllm_config(_OnlineConfig(_int8_args()))

        with self._profile(_COMPATIBILITY), self.assertRaises(NotImplementedError):
            validate_online_quantization(config)

    def test_missing_model_config_is_ignored(self):
        validate_online_quantization(SimpleNamespace(model_config=None))


if __name__ == "__main__":
    unittest.main()
