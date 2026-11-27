# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from vllm_ascend.attention.gdn import AscendGatedDeltaNetAttention, _get_fla_gdn_prefill_op
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


@pytest.fixture(autouse=True)
def clear_prefill_probe():
    _get_fla_gdn_prefill_op.cache_clear()
    yield
    _get_fla_gdn_prefill_op.cache_clear()


@pytest.mark.parametrize("device", list(AscendDeviceType))
def test_fla_prefill_does_not_require_version(device):
    module = ModuleType("fla_npu.ops.ascendc")
    fla_op = Mock()
    module.__dict__["chunk_gated_delta_rule_fwd"] = fla_op
    with (
        patch.dict(sys.modules, {module.__name__: module}),
        patch("vllm_ascend.attention.gdn.get_current_hardware_profile", return_value=get_hardware_profile(device)),
        patch("importlib.metadata.version", side_effect=AssertionError("FLA version must not be queried")),
    ):
        op = _get_fla_gdn_prefill_op()
        assert _get_fla_gdn_prefill_op() is op
    assert op is fla_op


def test_missing_fla_entry_point_uses_fallback():
    with patch.dict(sys.modules, {"fla_npu.ops.ascendc": ModuleType("fla_npu.ops.ascendc")}):
        assert _get_fla_gdn_prefill_op() is None


@pytest.mark.parametrize("fused_available", [False, True])
def test_fla_supports_host_only_metadata(fused_available):
    module = ModuleType("fla_npu.ops.ascendc")
    module.__dict__["chunk_gated_delta_rule_fwd"] = Mock()
    with (
        patch.dict(sys.modules, {module.__name__: module}),
        patch(
            "vllm_ascend.attention.gdn.get_current_hardware_profile",
            return_value=get_hardware_profile(AscendDeviceType.A5),
        ),
        patch("importlib.metadata.version", side_effect=AssertionError("FLA version must not be queried")),
        patch.object(AscendGatedDeltaNetAttention, "_probe_fused_chunk", return_value=fused_available),
    ):
        assert AscendGatedDeltaNetAttention._supports_host_metadata_prefill() is True
