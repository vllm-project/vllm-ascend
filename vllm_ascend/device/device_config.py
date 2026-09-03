# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Runtime entry point for Ascend hardware configuration.

Hardware discovery is cached, immutable, and import-safe. In particular, this
module never probes torch-npu while it is being imported. Runtime validation is
performed explicitly by the worker after the NPU runtime is ready.
"""

from dataclasses import dataclass
from functools import lru_cache

from vllm_ascend.device.hardware import (
    AscendDeviceType,
    device_type_from_runtime_soc,
    device_type_from_soc_version,
)


@dataclass(frozen=True)
class DeviceConfig:
    """Immutable configuration selected for the current hardware family."""

    _device_type: AscendDeviceType


def _device_type_from_build_info() -> AscendDeviceType:
    from vllm_ascend import _build_info  # type: ignore

    device_type = getattr(_build_info, "__device_type__", None)
    if device_type is not None:
        try:
            return AscendDeviceType[device_type]
        except KeyError as exc:
            raise RuntimeError(f"Unsupported built-in device type: {device_type}.") from exc

    # Compatibility for packages generated before __device_type__ was added.
    soc_version = getattr(_build_info, "__soc_version__", "ASCEND910B1")
    return device_type_from_soc_version(soc_version)


@lru_cache(maxsize=1)
def get_device_config() -> DeviceConfig:
    """Return the immutable hardware configuration for this installation."""

    return DeviceConfig(_device_type=_device_type_from_build_info())


# Compatibility API. New business code must consume HardwareProfile
# capabilities, policies, or families instead of these hardware identities.
def get_ascend_device_type() -> AscendDeviceType:
    return get_device_config()._device_type


def is_310p() -> bool:
    return get_ascend_device_type() == AscendDeviceType._310P


def is_950() -> bool:
    return get_ascend_device_type() == AscendDeviceType.A5


def check_ascend_device_type() -> None:
    """Validate that the installed package matches the runtime NPU."""

    import torch_npu

    built_device_type = get_device_config()._device_type
    runtime_soc_version = torch_npu.npu.get_soc_version()
    runtime_device_type = device_type_from_runtime_soc(runtime_soc_version)
    if built_device_type != runtime_device_type:
        raise RuntimeError(
            f"Current device type: {runtime_device_type} does not match the installed version's device type: "
            f"{built_device_type}, please check your installation package."
        )
