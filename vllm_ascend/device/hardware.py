# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Pure hardware identification helpers.

This module deliberately depends only on the Python standard library so it can
also be loaded by ``setup.py`` before vllm-ascend is installed. Device identity
must stay inside the device abstraction package; callers outside this package
should consume semantic capabilities from ``device_config`` instead.
"""

import logging
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
    "ascend910_9363": AscendDeviceType.A3,
    "ascend310p1": AscendDeviceType._310P,
    "ascend310p3": AscendDeviceType._310P,
    "ascend310p5": AscendDeviceType._310P,
    "ascend310p7": AscendDeviceType._310P,
    "ascend310p3vir01": AscendDeviceType._310P,
    "ascend310p3vir02": AscendDeviceType._310P,
    "ascend310p3vir04": AscendDeviceType._310P,
    "ascend310p3vir08": AscendDeviceType._310P,
}


# Recorded in _build_info.py when a build without custom kernels runs on a
# machine with no NPU to detect. Must be a value device_type_from_soc_version
# accepts; it resolves to AscendDeviceType.A2.
CPU_ONLY_FALLBACK_SOC_VERSION = "ascend910b1"

_MISSING_SOC_VERSION_ERROR = (
    "Could not determine chip type automatically via 'npu-smi'. "
    "This can happen in a CPU-only environment. "
    "Please set the 'SOC_VERSION' environment variable to specify the target chip, for example:\n"
    '  - Atlas A2: export SOC_VERSION="ascend910b1"\n'
    '  - Atlas A3: export SOC_VERSION="ascend910_9391"\n'
    '  - Atlas 300I: export SOC_VERSION="ascend310p1"\n'
    '  - Atlas A5: export SOC_VERSION="<value starting with ascend950>"\n'
    "You can also refer to the SOC_VERSION defaults in Dockerfile*."
)


def resolve_build_soc_version(detected_soc_version: str, compile_custom_kernels: bool) -> str:
    """Pick the SOC_VERSION a build records, when the user set none.

    ``detected_soc_version`` is what ``npu-smi`` reported, empty when no NPU
    driver is present. A non-empty ``detected_soc_version`` is returned
    immediately.
    
    Raises an exception if ``detected_soc_version`` is empty when
    ``compile_custom_kernels`` is ``True``.

    When ``compile_custom_kernels`` is ``False`` and ``detected_soc_version``
    is empty, assume we are in a CPU-only scenario and return a dummy
    SOC version.
    """

    if detected_soc_version:
        return detected_soc_version
    if compile_custom_kernels:
        raise RuntimeError(_MISSING_SOC_VERSION_ERROR)
    logging.warning(
        'No NPU detected and SOC_VERSION is unset; defaulting to "%s" because '
        "COMPILE_CUSTOM_KERNELS=0. This package is suitable for running tests/ut only: "
        "it has no custom kernels, and it will fail check_ascend_device_type() on any "
        "non-A2 device. Set SOC_VERSION explicitly to build for a specific target.",
        CPU_ONLY_FALLBACK_SOC_VERSION,
    )
    return CPU_ONLY_FALLBACK_SOC_VERSION


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
    if 250 <= soc_version <= 256:
        return AscendDeviceType.A3
    if 200 <= soc_version <= 205:
        return AscendDeviceType._310P
    if soc_version == 260:
        return AscendDeviceType.A5
    raise RuntimeError(f"Cannot support runtime soc_version: {soc_version}.")
