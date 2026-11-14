# SPDX-License-Identifier: Apache-2.0

import sys
from importlib import metadata
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from vllm_ascend.device.fla import _get_fla_a5_version, is_fla_gdn_chunk32_supported
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention, _get_fla_gdn_prefill_op


@pytest.fixture(autouse=True)
def clear_prefill_probe():
    _get_fla_a5_version.cache_clear()
    _get_fla_gdn_prefill_op.cache_clear()
    yield
    _get_fla_a5_version.cache_clear()
    _get_fla_gdn_prefill_op.cache_clear()


@pytest.mark.parametrize(
    "version_result,expected",
    [
        ("26.9.1+dev153268a", False),
        ("26.9.1", False),
        ("26.9.1+dev5cc6990", False),
        ("26.9.1+devffffffff", False),
        ("27.0.0", False),
        (None, False),
        (metadata.PackageNotFoundError("flash-linear-attention-npu-a5"), False),
        (OSError("unreadable package metadata"), False),
    ],
)
@pytest.mark.parametrize("device", [AscendDeviceType.A5, AscendDeviceType.A2, AscendDeviceType.A3])
def test_importable_fla_requires_chunk32_contract_only_on_a5(version_result, expected, device):
    module = ModuleType("fla_npu.ops.ascendc")
    module.chunk_gated_delta_rule_fwd = Mock()
    version_kwargs = (
        {"side_effect": version_result} if isinstance(version_result, Exception) else {"return_value": version_result}
    )
    with (
        patch.dict(sys.modules, {module.__name__: module}),
        patch("vllm_ascend.ops.gdn.get_current_hardware_profile", return_value=get_hardware_profile(device)),
        patch("vllm_ascend.device.fla.get_current_hardware_profile", return_value=get_hardware_profile(device)),
        patch("importlib.metadata.version", **version_kwargs) as version,
    ):
        op = _get_fla_gdn_prefill_op()
        assert _get_fla_gdn_prefill_op() is op
    if device != AscendDeviceType.A5 or expected:
        assert op is module.chunk_gated_delta_rule_fwd
    else:
        assert op is None
    if device == AscendDeviceType.A5:
        version.assert_called_once_with("flash-linear-attention-npu-a5")
    else:
        version.assert_not_called()


def test_missing_fla_entry_point_uses_fallback():
    with patch.dict(sys.modules, {"fla_npu.ops.ascendc": ModuleType("fla_npu.ops.ascendc")}):
        assert _get_fla_gdn_prefill_op() is None


@pytest.mark.parametrize("device", list(AscendDeviceType))
@pytest.mark.parametrize("approved,installed", [(False, "test-only"), (True, "test-only"), (True, "other-build")])
def test_chunk32_requires_device_and_allowlist(device, approved, installed):
    with (
        patch("vllm_ascend.device.fla.get_current_hardware_profile", return_value=get_hardware_profile(device)),
        patch(
            "vllm_ascend.device.fla._FLA_GDN_CHUNK32_VERSIONS", frozenset({"test-only"}) if approved else frozenset()
        ),
        patch("importlib.metadata.version", return_value=installed),
    ):
        assert is_fla_gdn_chunk32_supported() is (
            device == AscendDeviceType.A5 and approved and installed == "test-only"
        )


@pytest.mark.parametrize("fused_available", [False, True])
def test_legacy_fla_does_not_force_host_only_metadata(fused_available):
    module = ModuleType("fla_npu.ops.ascendc")
    module.chunk_gated_delta_rule_fwd = Mock()
    with (
        patch.dict(sys.modules, {module.__name__: module}),
        patch(
            "vllm_ascend.ops.gdn.get_current_hardware_profile", return_value=get_hardware_profile(AscendDeviceType.A5)
        ),
        patch("importlib.metadata.version", return_value="26.9.1+dev5cc6990"),
        patch.object(AscendGatedDeltaNetAttention, "_probe_fused_chunk", return_value=fused_available),
    ):
        assert AscendGatedDeltaNetAttention._supports_host_metadata_prefill() is fused_available
