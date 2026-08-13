# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import cast

import pytest

from vllm_ascend.device.device_config import DeviceConfig
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import (
    HardwareCapability,
    WeightLayoutPolicy,
    get_current_hardware_profile,
    get_hardware_profile,
)


@pytest.mark.parametrize(
    ("device_type", "weight_layout_policy", "capabilities"),
    [
        (
            AscendDeviceType.A2,
            WeightLayoutPolicy.CONFIGURABLE,
            frozenset({HardwareCapability.AUTO_ENABLE_CUSTOM_OPS}),
        ),
        (
            AscendDeviceType.A3,
            WeightLayoutPolicy.CONFIGURABLE,
            frozenset({HardwareCapability.AUTO_ENABLE_CUSTOM_OPS}),
        ),
        (AscendDeviceType._310P, WeightLayoutPolicy.FORCE_NZ, frozenset()),
        (
            AscendDeviceType.A5,
            WeightLayoutPolicy.CONFIGURABLE,
            frozenset(
                {
                    HardwareCapability.AUTO_ENABLE_CUSTOM_OPS,
                    HardwareCapability.DYNAMIC_MX_QUANT_FUSION,
                }
            ),
        ),
    ],
)
def test_hardware_profile_matrix(
    device_type: AscendDeviceType,
    weight_layout_policy: WeightLayoutPolicy,
    capabilities: frozenset[HardwareCapability],
) -> None:
    profile = get_hardware_profile(device_type)

    assert profile._device_type is device_type
    assert profile.weight_layout_policy is weight_layout_policy
    assert profile.capabilities == capabilities
    for capability in HardwareCapability:
        assert profile.supports(capability) is (capability in capabilities)


def test_current_hardware_profile_uses_device_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_ascend.device.hardware_profile as profile_module

    monkeypatch.setattr(
        profile_module,
        "get_device_config",
        lambda: DeviceConfig(_device_type=AscendDeviceType.A5),
    )

    assert get_current_hardware_profile() is get_hardware_profile(AscendDeviceType.A5)


def test_unknown_device_type_is_rejected() -> None:
    unknown_device_type = cast(AscendDeviceType, object())

    with pytest.raises(RuntimeError, match="No hardware profile is registered"):
        get_hardware_profile(unknown_device_type)


def test_every_device_type_has_a_profile() -> None:
    for device_type in AscendDeviceType:
        assert get_hardware_profile(device_type)._device_type is device_type
