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

"""Every MXFP scheme must fail at construction on hardware without the MX matmul.

The gate mirrors the block-wise FP8 rejection: schemes whose quant types always
quantize with the A5 MX kernel raise NotImplementedError at __init__, before any
weight is allocated, instead of failing deep inside weight loading with an
obscure aclnn error. The hardware profile is mocked so both branches run on any
machine.
"""

from unittest.mock import Mock, patch

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import create_mock_ascend_config, create_mock_vllm_config
from vllm_ascend.device.hardware_profile import WeightFormat
from vllm_ascend.quantization.methods.kv_cache.mxfp_c8 import AscendC8MXFPKVCacheAttentionMethod
from vllm_ascend.quantization.methods.w4a4.w4a4_mxfp4 import (
    AscendW4A4MXFP4DynamicFusedMoEMethod,
    AscendW4A4MXFP4DynamicLinearMethod,
)
from vllm_ascend.quantization.methods.w4a4.w4a4_mxfp4_flatquant import (
    AscendW4A4MXFP4FlatQuantDynamicLinearMethod,
)
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import AscendW4A8MXFPDynamicLinearMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DSDynamicLinearMethod,
    AscendW8A8MXFP8DynamicFusedMoEMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)
from vllm_ascend.quantization.methods.wna16.w4a16_mxfp4 import AscendW4A16MXFP4FusedMoEMethod

BASE = "vllm_ascend.quantization.methods.base"
C8_MXFP = "vllm_ascend.quantization.methods.kv_cache.mxfp_c8"


def mx_profile_patch(supports_mx: bool):
    """Return an unstarted patch pinning the MX gate's profile to a stub."""
    profile = Mock(supports_weight_format=Mock(return_value=supports_mx))
    return patch(f"{BASE}.get_current_hardware_profile", new=Mock(return_value=profile))


def build_w8a8_mxfp8_linear():
    with (
        patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config",
            return_value=create_mock_vllm_config(),
        ),
        patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_dynamic_mx_quant_scale_alg",
            return_value=0,
        ),
    ):
        return AscendW8A8MXFP8DynamicLinearMethod()


def build_w8a8_mxfp8_moe():
    with (
        patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config",
            return_value=create_mock_vllm_config(),
        ),
        patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_ascend_config",
            return_value=create_mock_ascend_config(),
        ),
        patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_dynamic_mx_quant_scale_alg",
            return_value=0,
        ),
    ):
        return AscendW8A8MXFP8DynamicFusedMoEMethod()


def build_w8a8_mxfp8_ds_linear():
    with patch(
        "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config",
        return_value=create_mock_vllm_config(),
    ):
        return AscendW8A8MXFP8DSDynamicLinearMethod((128, 128))


def build_w4a4_mxfp4_linear():
    with patch(
        "vllm_ascend.quantization.methods.w4a4.w4a4_mxfp4.get_current_vllm_config",
        return_value=create_mock_vllm_config(),
    ):
        return AscendW4A4MXFP4DynamicLinearMethod()


def build_w4a4_mxfp4_moe():
    with (
        patch(
            "vllm_ascend.quantization.methods.w4a4.w4a4_mxfp4.get_current_vllm_config",
            return_value=create_mock_vllm_config(),
        ),
        patch(
            "vllm_ascend.quantization.methods.w4a4.w4a4_mxfp4.get_ascend_config",
            return_value=create_mock_ascend_config(),
        ),
    ):
        return AscendW4A4MXFP4DynamicFusedMoEMethod()


def build_w4a8_mxfp_linear():
    with patch(
        "vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4.get_current_vllm_config",
        return_value=create_mock_vllm_config(),
    ):
        return AscendW4A8MXFPDynamicLinearMethod()


def build_w4a16_mxfp4_moe():
    with (
        patch(
            "vllm_ascend.quantization.methods.wna16.w4a16_mxfp4.get_current_vllm_config",
            return_value=create_mock_vllm_config(),
        ),
        patch(
            "vllm_ascend.quantization.methods.wna16.w4a16_mxfp4.get_ascend_config",
            return_value=create_mock_ascend_config(),
        ),
    ):
        return AscendW4A16MXFP4FusedMoEMethod()


def build_w4a4_mxfp4_flatquant():
    with (
        patch(
            "vllm_ascend.quantization.methods.w4a4.w4a4_mxfp4_flatquant.get_current_vllm_config",
            return_value=create_mock_vllm_config(),
        ),
        patch(
            "vllm_ascend.quantization.methods.w4a4.w4a4_mxfp4_flatquant.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
        return AscendW4A4MXFP4FlatQuantDynamicLinearMethod()


class TestRequireMxWeightFormat(TestBase):
    def _require(self, weight_format, supports_mx):
        profile_patch = mx_profile_patch(supports_mx)
        profile_patch.start()
        self.addCleanup(profile_patch.stop)
        from vllm_ascend.quantization.methods.base import require_mx_weight_format

        return require_mx_weight_format(weight_format)

    def test_rejects_mxfp8_without_mx_matmul(self):
        with self.assertRaisesRegex(NotImplementedError, "Ascend 950 only") as ctx:
            self._require(WeightFormat.MXFP8, supports_mx=False)
        self.assertIn("MXFP8", str(ctx.exception))

    def test_rejects_mxfp4_without_mx_matmul(self):
        with self.assertRaisesRegex(NotImplementedError, "Ascend 950 only") as ctx:
            self._require(WeightFormat.MXFP4, supports_mx=False)
        self.assertIn("MXFP4", str(ctx.exception))

    def test_passes_on_mxfp8_hardware(self):
        self.assertIsNone(self._require(WeightFormat.MXFP8, supports_mx=True))

    def test_passes_on_mxfp4_hardware(self):
        self.assertIsNone(self._require(WeightFormat.MXFP4, supports_mx=True))


class TestMxSchemesRejectedWithoutMxMatmul(TestBase):
    """One reject case per scheme family; the gate fires before anything else."""

    def _assert_rejected(self, build):
        profile_patch = mx_profile_patch(supports_mx=False)
        profile_patch.start()
        self.addCleanup(profile_patch.stop)
        with self.assertRaisesRegex(NotImplementedError, "Ascend 950 only"):
            build()

    def test_w8a8_mxfp8_linear(self):
        self._assert_rejected(build_w8a8_mxfp8_linear)

    def test_w8a8_mxfp8_moe(self):
        self._assert_rejected(build_w8a8_mxfp8_moe)

    def test_w8a8_mxfp8_ds_linear(self):
        self._assert_rejected(build_w8a8_mxfp8_ds_linear)

    def test_w4a4_mxfp4_linear(self):
        self._assert_rejected(build_w4a4_mxfp4_linear)

    def test_w4a4_mxfp4_moe(self):
        self._assert_rejected(build_w4a4_mxfp4_moe)

    def test_w4a8_mxfp_linear(self):
        self._assert_rejected(build_w4a8_mxfp_linear)

    def test_w4a16_mxfp4_moe(self):
        self._assert_rejected(build_w4a16_mxfp4_moe)

    def test_w4a4_mxfp4_flatquant_linear(self):
        self._assert_rejected(build_w4a4_mxfp4_flatquant)


class TestMxSchemesAcceptedOnMxHardware(TestBase):
    """The same builders on an A5-style profile must construct untouched."""

    def _accept(self, build):
        profile_patch = mx_profile_patch(supports_mx=True)
        profile_patch.start()
        self.addCleanup(profile_patch.stop)
        scheme = build()
        self.assertGreaterEqual(scheme.group_size, 1)
        return scheme

    def test_w8a8_mxfp8_linear(self):
        scheme = self._accept(build_w8a8_mxfp8_linear)
        self.assertEqual(scheme.group_size, 32)

    def test_w8a8_mxfp8_moe(self):
        self._accept(build_w8a8_mxfp8_moe)

    def test_w8a8_mxfp8_ds_linear(self):
        scheme = self._accept(build_w8a8_mxfp8_ds_linear)
        self.assertEqual(scheme.block_size, 128)

    def test_w4a4_mxfp4_linear(self):
        self._accept(build_w4a4_mxfp4_linear)

    def test_w4a4_mxfp4_moe(self):
        self._accept(build_w4a4_mxfp4_moe)

    def test_w4a8_mxfp_linear(self):
        self._accept(build_w4a8_mxfp_linear)

    def test_w4a16_mxfp4_moe(self):
        self._accept(build_w4a16_mxfp4_moe)

    def test_w4a4_mxfp4_flatquant_linear(self):
        self._accept(build_w4a4_mxfp4_flatquant)


class TestNonMxSchemeNotGated(TestBase):
    def test_int8_dynamic_scheme_constructs_without_mx_support(self):
        # No get_current_vllm_config patch is needed here on purpose: the int8
        # dynamic scheme's __init__ is a no-op, so construction cannot depend
        # on engine config. If this ever starts failing on a fresh environment,
        # suspect a scheme change, not this gate.
        profile_patch = mx_profile_patch(supports_mx=False)
        profile_patch.start()
        self.addCleanup(profile_patch.stop)
        self.assertIsInstance(AscendW8A8DynamicLinearMethod(), AscendW8A8DynamicLinearMethod)


class TestC8MxFpKvCacheGate(TestBase):
    def _patch_fp8_attention(self, supports: bool):
        profile = Mock(supports=Mock(return_value=supports))
        profile_patch = patch(f"{C8_MXFP}.get_current_hardware_profile", new=Mock(return_value=profile))
        profile_patch.start()
        self.addCleanup(profile_patch.stop)

    def test_rejected_without_fp8_attention(self):
        self._patch_fp8_attention(supports=False)
        with self.assertRaisesRegex(NotImplementedError, "Ascend 950 only"):
            AscendC8MXFPKVCacheAttentionMethod({}, "layer")

    def test_accepted_with_fp8_attention(self):
        self._patch_fp8_attention(supports=True)
        method = AscendC8MXFPKVCacheAttentionMethod({}, "layer")
        self.assertEqual(method.prefix, "layer")
