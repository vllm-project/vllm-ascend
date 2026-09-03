# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Pure hardware identification helpers.

This module deliberately depends only on the Python standard library so it can
also be loaded by ``setup.py`` before vllm-ascend is installed. Device identity
must stay inside the device abstraction package; callers outside this package
should consume semantic capabilities from ``device_config`` instead.
"""

from enum import Enum


class AscendDeviceType(Enum):
    """Internal hardware families used to select a device profile."""

    A2 = 0
    A3 = 1
    _310P = 2
    A5 = 3


_SOC_VERSION_TO_DEVICE_TYPE = {
    "910b": AscendDeviceType.A2,
    "910c": AscendDeviceType.A3,
    "310p": AscendDeviceType._310P,
    "ascend910b1": AscendDeviceType.A2,
    "ascend910b2": AscendDeviceType.A2,
    "ascend910b2c": AscendDeviceType.A2,
    "ascend910b3": AscendDeviceType.A2,
    "ascend910b4": AscendDeviceType.A2,
    "ascend910b4-1": AscendDeviceType.A2,
    "ascend910_9391": AscendDeviceType.A3,
    "ascend910_9381": AscendDeviceType.A3,
    "ascend910_9372": AscendDeviceType.A3,
    "ascend910_9392": AscendDeviceType.A3,
    "ascend910_9382": AscendDeviceType.A3,
    "ascend910_9362": AscendDeviceType.A3,
    "ascend310p1": AscendDeviceType._310P,
    "ascend310p3": AscendDeviceType._310P,
    "ascend310p5": AscendDeviceType._310P,
    "ascend310p7": AscendDeviceType._310P,
    "ascend310p3vir01": AscendDeviceType._310P,
    "ascend310p3vir02": AscendDeviceType._310P,
    "ascend310p3vir04": AscendDeviceType._310P,
    "ascend310p3vir08": AscendDeviceType._310P,
}


def device_type_from_soc_version(soc_version: str) -> AscendDeviceType:
    """Resolve a build-time SOC_VERSION value to a hardware family."""

    normalized = soc_version.strip().lower()
    if "ascend950" in normalized:
        return AscendDeviceType.A5
    try:
        return _SOC_VERSION_TO_DEVICE_TYPE[normalized]
    except KeyError as exc:
        raise RuntimeError(f"Undefined soc_version: {soc_version}. Please file an issue to vllm-ascend.") from exc


def device_type_from_runtime_soc(soc_version: int) -> AscendDeviceType:
    """Resolve the numeric SOC version reported by torch-npu."""

    if 220 <= soc_version <= 225:
        return AscendDeviceType.A2
    if 250 <= soc_version <= 255:
        return AscendDeviceType.A3
    if 200 <= soc_version <= 205:
        return AscendDeviceType._310P
    if soc_version == 260:
        return AscendDeviceType.A5
    raise RuntimeError(f"Cannot support runtime soc_version: {soc_version}.")
