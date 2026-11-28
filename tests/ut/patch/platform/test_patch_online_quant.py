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
"""Online quantization: the patch owns vLLM's tables and resolves Ascend methods."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config import quantization as vllm_quantization
from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, get_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    resolve_quant_method,
)
from vllm.model_executor.layers.quantization.online import base as online_base
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticChannelSym,
    kInt8StaticChannelSym,
)

from tests.ut.base import TestBase
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.patch.platform import patch_online_quant
from vllm_ascend.quantization.methods.online import (
    ONLINE_SHORTHANDS,
    SUPPORTED_ONLINE_WEIGHT_KEYS,
)
from vllm_ascend.quantization.methods.online.int8 import (
    AscendInt8PerChannelOnlineLinearMethod,
    AscendInt8PerChannelOnlineMoEMethod,
)

# vLLM's own online shorthands: Ascend neither removes nor rewrites them, so the
# checkpoint formats and the deprecated `--quantization fp8` alias keep working.
_UPSTREAM_SHORTHANDS = (
    "fp8_per_tensor",
    "fp8_per_block",
    "fp8_per_channel",
    "mxfp4",
    "mxfp8",
    "int8_per_channel_weight_only",
    "nvfp4_per_token",
    "lut_b_moe",
    "mxfp4_hadamard32",
)

_VLLM_CONFIG = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))


def _patch_vllm_config():
    """`OnlineLinearBase.__init__` records the model dtype from this helper."""
    return patch(
        "vllm.model_executor.layers.quantization.online.fp8.get_current_vllm_config",
        return_value=_VLLM_CONFIG,
    )


class _LinearLayer(LinearBase):
    """Minimal LinearBase subclass.

    vLLM resolves online targets from the layer *type*, so the tests need a real
    LinearBase subclass instead of a ``MagicMock(spec=LinearBase)``.
    """

    def __init__(self, prefix: str = "model.layers.0.mlp.down_proj") -> None:
        torch.nn.Module.__init__(self)
        self.prefix = prefix
        self.tp_size = 1
        self.input_size = 64
        self.input_size_per_partition = 64


class _CheckpointConfig(QuantizationConfig):
    """Checkpoint format stand-in: only what `resolve_quant_method` reads."""

    def get_name(self) -> str:
        return "compressed-tensors"

    @classmethod
    def get_supported_act_dtypes(cls) -> list:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list:
        return []

    @classmethod
    def from_config(cls, config) -> "QuantizationConfig":
        raise NotImplementedError

    def get_quant_method(self, layer, prefix):
        return UnquantizedLinearMethod()


class TestOnlineMethodTables(TestBase):
    def test_linear_table_maps_int8_to_the_ascend_method(self):
        # vLLM has no INT8 entry in its linear table at all.
        self.assertIs(
            online_base._ONLINE_LINEAR_METHODS[kInt8StaticChannelSym],
            AscendInt8PerChannelOnlineLinearMethod,
        )

    def test_moe_table_replaces_the_cuda_int8_method(self):
        self.assertIs(
            online_base._ONLINE_MOE_METHODS[kInt8StaticChannelSym],
            AscendInt8PerChannelOnlineMoEMethod,
        )

    def test_upstream_tables_keep_their_own_entries(self):
        # Only Ascend's keys are added/replaced; vLLM's FP8 mapping is not ours.
        self.assertTrue(
            online_base._ONLINE_LINEAR_METHODS[kFp8StaticChannelSym].__module__.startswith("vllm.model_executor")
        )


class TestOnlineNames(TestBase):
    def test_vllm_shorthand_table_is_not_replaced(self):
        # Ascend does not remove vLLM's recipes: checkpoint formats and vLLM's
        # own aliases must keep resolving.
        for name in _UPSTREAM_SHORTHANDS:
            with self.subTest(name=name):
                self.assertIn(name, vllm_quantization._ONLINE_SHORTHANDS)
                self.assertIn(name, vllm_quantization.ONLINE_QUANT_SHORTHAND_NAMES)
                self.assertIn(name, QUANTIZATION_METHODS)

    def test_declared_recipes_are_applied_on_top_of_vllm(self):
        # Ascend's recipe list is applied over vLLM's table: the name resolves
        # and the effective entry is the config Ascend declared.
        for name, args in ONLINE_SHORTHANDS.items():
            with self.subTest(name=name):
                self.assertIn(name, vllm_quantization._ONLINE_SHORTHANDS)
                self.assertIn(name, vllm_quantization.ONLINE_QUANT_SHORTHAND_NAMES)
                self.assertIs(vllm_quantization._ONLINE_SHORTHANDS[name], args)

    def test_declared_recipes_only_use_supported_weight_keys(self):
        # Everything Ascend lists as a recipe must have a method for every spec
        # it selects, otherwise the table would advertise a recipe that
        # `online_validate` then rejects.
        for name, args in ONLINE_SHORTHANDS.items():
            for spec in (args.linear, args.moe):
                if spec is None or spec.weight is None:
                    continue
                with self.subTest(name=name, weight=spec.weight):
                    self.assertIn(spec.weight, SUPPORTED_ONLINE_WEIGHT_KEYS)

    def test_the_deprecated_fp8_alias_still_resolves(self):
        # `get_quant_config` calls `resolve_quantization_config("fp8_per_tensor")`
        # for `--quantization fp8`; the name still has to resolve from vLLM's
        # table, and the recipe is rejected by `online_validate` instead.
        from vllm.config.quantization import resolve_quantization_config

        self.assertIsNotNone(resolve_quantization_config("fp8_per_tensor", None))
        self.assertIs(get_quantization_config("fp8_per_tensor"), OnlineQuantizationConfig)

    def test_shorthand_only_names_still_resolve(self):
        # `_verify_quantization` resolves every name in QUANTIZATION_METHODS when
        # the model carries quantization metadata; a name left without a table
        # entry would raise KeyError there.
        self.assertIs(get_quantization_config("fp8_per_block"), OnlineQuantizationConfig)

    def test_kept_names_still_resolve_to_the_upstream_online_config(self):
        # No subclass is registered: upstream dispatches the same class the
        # composition path and the v0.34.0 lane instantiate.
        for name in ("online", "int8_per_channel_weight_only"):
            with self.subTest(name=name):
                self.assertIs(get_quantization_config(name), OnlineQuantizationConfig)

    def test_reapplying_the_patch_does_not_change_the_tables(self):
        before = dict(vllm_quantization._ONLINE_SHORTHANDS)
        names_before = vllm_quantization.ONLINE_QUANT_SHORTHAND_NAMES

        patch_online_quant.apply_online_quantization_patch()

        self.assertEqual(dict(vllm_quantization._ONLINE_SHORTHANDS), before)
        self.assertEqual(vllm_quantization.ONLINE_QUANT_SHORTHAND_NAMES, names_before)


class TestAscendMethodResolution(TestBase):
    def test_int8_linear_spec_resolves_to_the_ascend_method(self):
        config = OnlineQuantizationConfig(QuantizationConfigArgs(linear=QuantSpec(weight=kInt8StaticChannelSym)))

        resolved = config.resolve_quant_method_cls(LinearBase, "model.layers.0.mlp.down_proj")

        self.assertIsNotNone(resolved)
        self.assertIs(resolved[4], AscendInt8PerChannelOnlineLinearMethod)

    def test_int8_moe_spec_resolves_to_the_ascend_method(self):
        config = OnlineQuantizationConfig(QuantizationConfigArgs(moe=QuantSpec(weight=kInt8StaticChannelSym)))

        resolved = config.resolve_quant_method_cls(RoutedExperts, "model.layers.0.mlp.experts")

        self.assertIsNotNone(resolved)
        self.assertIs(resolved[4], AscendInt8PerChannelOnlineMoEMethod)

    def test_unsupported_weight_key_still_raises(self):
        config = OnlineQuantizationConfig(QuantizationConfigArgs(linear=QuantSpec(weight=kFp8StaticChannelSym)))

        with self.assertRaises(ValueError) as raised:
            config.resolve_quant_method_cls(LinearBase, "model.layers.0.mlp.down_proj")

        self.assertIn("not supported", str(raised.exception))

    def test_activation_override_is_rejected_by_the_method(self):
        # Upstream's shared mixin owns this check; the patch does not replace it.
        with self.assertRaises(NotImplementedError):
            AscendInt8PerChannelOnlineLinearMethod.resolve_activation_quant_key(kInt8StaticChannelSym, explicit=True)


class TestComposedOnlineOverlay(TestBase):
    """The "checkpoint format + online overlay" path.

    `weight_utils.maybe_compose_online_quantization` builds the upstream
    `OnlineQuantizationConfig` directly, so the overlay only reaches Ascend's
    methods through the patched tables.
    """

    def test_overlay_quantizes_an_unquantized_layer_with_the_ascend_method(self):
        checkpoint = _CheckpointConfig()
        checkpoint.online_quantization_config = OnlineQuantizationConfig(
            QuantizationConfigArgs(linear=QuantSpec(weight=kInt8StaticChannelSym))
        )
        layer = _LinearLayer("model.layers.0.mlp.down_proj")

        with _patch_vllm_config():
            method = resolve_quant_method(checkpoint, layer, "model.layers.0.mlp.down_proj")

        self.assertIsInstance(method, AscendInt8PerChannelOnlineLinearMethod)

    def test_layer_without_an_online_target_keeps_the_checkpoint_method(self):
        checkpoint = _CheckpointConfig()
        checkpoint.online_quantization_config = OnlineQuantizationConfig(
            QuantizationConfigArgs(
                linear=QuantSpec(weight=kInt8StaticChannelSym),
                ignore=["model.layers.0.mlp.gate_proj"],
            )
        )
        layer = _LinearLayer("model.layers.0.mlp.gate_proj")

        method = resolve_quant_method(checkpoint, layer, "model.layers.0.mlp.gate_proj")

        self.assertIsInstance(method, UnquantizedLinearMethod)


class TestUnquantizedFallback(TestBase):
    def _config(self) -> OnlineQuantizationConfig:
        return OnlineQuantizationConfig(QuantizationConfigArgs(moe=QuantSpec(weight=kInt8StaticChannelSym)))

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
            patch.object(patch_online_quant, "AscendUnquantizedFusedMoEMethod") as ascend_moe,
        ):
            method = self._config().get_quant_method(layer, "model.layers.0.mlp.experts")

        ascend_moe.assert_called_once_with(moe=sentinel)
        self.assertIs(method, ascend_moe.return_value)

    def test_a_matched_layer_is_left_to_upstream(self):
        config = OnlineQuantizationConfig(QuantizationConfigArgs(linear=QuantSpec(weight=kInt8StaticChannelSym)))
        layer = _LinearLayer("model.layers.0.mlp.down_proj")

        with _patch_vllm_config():
            method = config.get_quant_method(layer, "model.layers.0.mlp.down_proj")

        self.assertIsInstance(method, AscendInt8PerChannelOnlineLinearMethod)
        self.assertIn("model.layers.0.mlp.down_proj", config.quantized_layers)


if __name__ == "__main__":
    unittest.main()
