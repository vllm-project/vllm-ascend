# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.engram_config import AscendEngramConfig, default_engram_overlap


@pytest.mark.parametrize("device", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("explicit", [None, False, True])
def test_engram_overlap_hardware_default_and_explicit_override(monkeypatch, device, explicit):
    monkeypatch.setattr(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile", lambda: get_hardware_profile(device)
    )
    config = AscendEngramConfig(overlap=explicit)
    expected = device == AscendDeviceType.A5 if explicit is None else explicit
    assert (default_engram_overlap() if config.overlap is None else config.overlap) is expected
    assert config.enabled is False


@pytest.mark.parametrize("enabled", [False, True])
def test_engram_requires_explicit_master_opt_in(enabled):
    assert AscendEngramConfig(enabled=enabled).enabled is enabled
