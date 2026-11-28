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
"""Online quantization: the patch owns vLLM's tables and config class."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

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
from vllm_ascend.device.hardware_profile import QuantizationBackendFamily
from vllm_ascend.patch.platform import patch_online_quant
from vllm_ascend.quantization.configs.online_config import AscendOnlineQuantizationConfig
from vllm_ascend.quantization.methods.online import (
    ONLINE_SHORTHANDS,
    SUPPORTED_ONLINE_WEIGHT_KEYS,
)
from vllm_ascend.quantization.methods.online.int8 import (
    AscendInt8PerChannelOnlineLinearMethod,
    AscendInt8PerChannelOnlineMoEMethod,
)
from vllm_ascend.utils import ASCEND_ONLINE_QUANT_METHODS

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

_ONLINE_CONFIG_MODULE = "vllm_ascend.quantization.configs.online_config"

_VLLM_CONFIG = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))


def _patch_vllm_config():
    """`OnlineLinearBase.__init__` records the model dtype from this helper."""
    return patch(
        "vllm.model_executor.layers.quantization.online.fp8.get_current_vllm_config",
        return_value=_VLLM_CONFIG,
    )


def _patch_standard_profile():
    """The Ascend config rejects a non-standard quantization backend."""
    return patch(
        f"{_ONLINE_CONFIG_MODULE}.get_current_hardware_profile",
        return_value=SimpleNamespace(quantization_backend_family=QuantizationBackendFamily.STANDARD),
    )


def _int8_config(**kwargs) -> OnlineQuantizationConfig:
    """Build the config the way vLLM does: through the rebound symbol."""
    with _patch_standard_profile():
        return online_base.OnlineQuantizationConfig(
            QuantizationConfigArgs(**{"moe": QuantSpec(weight=kInt8StaticChannelSym), **kwargs})
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


class TestClassRebind(TestBase):
    def test_the_config_symbol_is_rebound(self):
        # Every hardcoded construction site imports the class from this module
        # inside its function, so the rebind reaches all of them.
        self.assertIs(online_base.OnlineQuantizationConfig, AscendOnlineQuantizationConfig)

    def test_construction_through_the_upstream_symbol_gets_the_ascend_config(self):
        config = _int8_config()

        self.assertIs(type(config), AscendOnlineQuantizationConfig)
        self.assertIsInstance(config, OnlineQuantizationConfig)

    def test_registry_resolves_the_ascend_config(self):
        # vLLM's `get_quantization_config` imports the class inside the
        # function, so `"online"` and every online shorthand resolve to it.
        for name in ("online", "int8_per_channel", "fp8_per_block"):
            with self.subTest(name=name):
                self.assertIs(get_quantization_config(name), AscendOnlineQuantizationConfig)


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
        # `AscendOnlineQuantizationConfig` then rejects.
        for name, args in ONLINE_SHORTHANDS.items():
            for spec in (args.linear, args.moe):
                if spec is None or spec.weight is None:
                    continue
                with self.subTest(name=name, weight=spec.weight):
                    self.assertIn(spec.weight, SUPPORTED_ONLINE_WEIGHT_KEYS)

    def test_the_deprecated_fp8_alias_still_resolves(self):
        # `get_quant_config` calls `resolve_quantization_config("fp8_per_tensor")`
        # for `--quantization fp8`; the name still has to resolve from vLLM's
        # table, and the recipe is rejected by the Ascend config instead.
        from vllm.config.quantization import resolve_quantization_config

        self.assertIsNotNone(resolve_quantization_config("fp8_per_tensor", None))

    def test_shorthand_only_names_still_resolve(self):
        # `_verify_quantization` resolves every name in QUANTIZATION_METHODS when
        # the model carries quantization metadata; a name left without a table
        # entry would raise KeyError there.
        self.assertIs(get_quantization_config("fp8_per_block"), AscendOnlineQuantizationConfig)

    def test_int8_recipe_covers_dense_and_experts(self):
        args = ONLINE_SHORTHANDS["int8_per_channel"]

        self.assertEqual(args.linear, QuantSpec(weight=kInt8StaticChannelSym))
        self.assertEqual(args.moe, QuantSpec(weight=kInt8StaticChannelSym))

    def test_declared_recipes_are_in_the_platform_whitelist(self):
        # verify_quantization() rejects a --quantization name missing from
        # NPUPlatform.supported_quantization, which is built from this tuple.
        for name in ONLINE_SHORTHANDS:
            with self.subTest(name=name):
                self.assertIn(name, ASCEND_ONLINE_QUANT_METHODS)

    def test_the_upstream_int8_shorthand_is_not_an_ascend_name(self):
        # `int8_per_channel_weight_only` is vLLM's experts-only shorthand;
        # Ascend exposes `int8_per_channel` instead, so the old name must not be
        # on the platform whitelist.
        self.assertNotIn("int8_per_channel_weight_only", ASCEND_ONLINE_QUANT_METHODS)

    def test_reapplying_the_patch_is_idempotent(self):
        before = dict(vllm_quantization._ONLINE_SHORTHANDS)
        names_before = vllm_quantization.ONLINE_QUANT_SHORTHAND_NAMES

        patch_online_quant.apply_online_quantization_patch()

        self.assertEqual(dict(vllm_quantization._ONLINE_SHORTHANDS), before)
        self.assertEqual(vllm_quantization.ONLINE_QUANT_SHORTHAND_NAMES, names_before)
        self.assertIs(online_base.OnlineQuantizationConfig, AscendOnlineQuantizationConfig)


class TestAscendMethodResolution(TestBase):
    def test_int8_linear_spec_resolves_to_the_ascend_method(self):
        config = _int8_config(linear=QuantSpec(weight=kInt8StaticChannelSym))

        resolved = config.resolve_quant_method_cls(LinearBase, "model.layers.0.mlp.down_proj")

        self.assertIsNotNone(resolved)
        self.assertIs(resolved[4], AscendInt8PerChannelOnlineLinearMethod)

    def test_int8_moe_spec_resolves_to_the_ascend_method(self):
        config = _int8_config()

        resolved = config.resolve_quant_method_cls(RoutedExperts, "model.layers.0.mlp.experts")

        self.assertIsNotNone(resolved)
        self.assertIs(resolved[4], AscendInt8PerChannelOnlineMoEMethod)

    def test_activation_override_is_rejected_by_the_method(self):
        # Upstream's shared mixin owns this check; the patch does not replace it.
        with self.assertRaises(NotImplementedError):
            AscendInt8PerChannelOnlineLinearMethod.resolve_activation_quant_key(kInt8StaticChannelSym, explicit=True)


class TestComposedOnlineOverlay(TestBase):
    """The "checkpoint format + online overlay" path.

    `weight_utils.maybe_compose_online_quantization` constructs the rebound
    class, so the overlay resolves Ascend's methods and rejects unsupported
    recipes like the plain online path does.
    """

    def test_overlay_quantizes_an_unquantized_layer_with_the_ascend_method(self):
        checkpoint = _CheckpointConfig()
        checkpoint.online_quantization_config = _int8_config(linear=QuantSpec(weight=kInt8StaticChannelSym))
        layer = _LinearLayer("model.layers.0.mlp.down_proj")

        with _patch_vllm_config():
            method = resolve_quant_method(checkpoint, layer, "model.layers.0.mlp.down_proj")

        self.assertIsInstance(method, AscendInt8PerChannelOnlineLinearMethod)

    def test_layer_without_an_online_target_keeps_the_checkpoint_method(self):
        checkpoint = _CheckpointConfig()
        checkpoint.online_quantization_config = _int8_config(
            linear=QuantSpec(weight=kInt8StaticChannelSym),
            ignore=["model.layers.0.mlp.gate_proj"],
        )
        layer = _LinearLayer("model.layers.0.mlp.gate_proj")

        method = resolve_quant_method(checkpoint, layer, "model.layers.0.mlp.gate_proj")

        self.assertIsInstance(method, UnquantizedLinearMethod)


if __name__ == "__main__":
    unittest.main()
