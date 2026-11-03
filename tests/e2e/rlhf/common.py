# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared device allocation helpers for RLHF E2E tests."""

import os


def server_visible_devices(device_index: int, world_size: int) -> str:
    """Map consecutive parent logical devices to child-server physical IDs."""
    if device_index < 0 or world_size < 1:
        raise ValueError("Device index must be non-negative and world size must be positive")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible is None:
        return ",".join(str(index) for index in range(device_index, device_index + world_size))
    devices = [device.strip() for device in visible.split(",")]
    if any(not device.isdecimal() for device in devices) or len(set(devices)) != len(devices):
        raise ValueError(f"Invalid ASCEND_RT_VISIBLE_DEVICES: {visible!r}; expected distinct numeric device IDs")
    if device_index + world_size > len(devices):
        raise ValueError(
            f"Logical devices [{device_index}, {device_index + world_size}) exceed parent visibility {visible!r}"
        )
    return ",".join(devices[device_index : device_index + world_size])
