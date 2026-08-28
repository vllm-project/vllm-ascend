# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Capability-oriented profiles for supported Ascend hardware families."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType

from vllm_ascend.device.device_config import get_device_config
from vllm_ascend.device.hardware import AscendDeviceType


class HardwareCapability(Enum):
    """Independent SoC capabilities consumed by shared business logic."""

    AUTO_ENABLE_CUSTOM_OPS = auto()
    DYNAMIC_MX_QUANT_FUSION = auto()


class WeightLayoutPolicy(Enum):
    """Weight layout selection policies for supported hardware families."""

    CONFIGURABLE = auto()
    FORCE_NZ = auto()


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Immutable capabilities and implementation choices for one SoC family."""

    _device_type: AscendDeviceType
    weight_layout_policy: WeightLayoutPolicy
    capabilities: frozenset[HardwareCapability]

    def supports(self, capability: HardwareCapability) -> bool:
        """Return whether this hardware family provides ``capability``."""

        return capability in self.capabilities


_AUTO_ENABLE_CUSTOM_OP_CAPABILITIES = frozenset({HardwareCapability.AUTO_ENABLE_CUSTOM_OPS})
_HARDWARE_PROFILES: Mapping[AscendDeviceType, HardwareProfile] = MappingProxyType(
    {
        AscendDeviceType.A2: HardwareProfile(
            _device_type=AscendDeviceType.A2,
            weight_layout_policy=WeightLayoutPolicy.CONFIGURABLE,
            capabilities=_AUTO_ENABLE_CUSTOM_OP_CAPABILITIES,
        ),
        AscendDeviceType.A3: HardwareProfile(
            _device_type=AscendDeviceType.A3,
            weight_layout_policy=WeightLayoutPolicy.CONFIGURABLE,
            capabilities=_AUTO_ENABLE_CUSTOM_OP_CAPABILITIES,
        ),
        AscendDeviceType._310P: HardwareProfile(
            _device_type=AscendDeviceType._310P,
            weight_layout_policy=WeightLayoutPolicy.FORCE_NZ,
            capabilities=frozenset(),
        ),
        AscendDeviceType.A5: HardwareProfile(
            _device_type=AscendDeviceType.A5,
            weight_layout_policy=WeightLayoutPolicy.CONFIGURABLE,
            capabilities=frozenset(
                {
                    HardwareCapability.AUTO_ENABLE_CUSTOM_OPS,
                    HardwareCapability.DYNAMIC_MX_QUANT_FUSION,
                }
            ),
        ),
    }
)


def get_hardware_profile(device_type: AscendDeviceType) -> HardwareProfile:
    """Return the immutable profile registered for ``device_type``."""

    try:
        return _HARDWARE_PROFILES[device_type]
    except KeyError as exc:
        raise RuntimeError(f"No hardware profile is registered for device type: {device_type}.") from exc


def get_current_hardware_profile() -> HardwareProfile:
    """Return the profile selected by the current device configuration."""

    return get_hardware_profile(get_device_config()._device_type)
