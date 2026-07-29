# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device mapping helpers for NPU IPC weight transfer."""

import os
import socket
from functools import lru_cache

import torch


def parse_visible_devices(visible_devices: str | None) -> list[int] | None:
    if not visible_devices:
        return None

    devices: list[int] = []
    for raw_device in visible_devices.split(","):
        device = raw_device.strip()
        if not device:
            raise ValueError(f"Invalid ASCEND_RT_VISIBLE_DEVICES value: {visible_devices!r}")
        try:
            devices.append(int(device))
        except ValueError as exc:
            raise ValueError(f"Invalid device id {device!r} in ASCEND_RT_VISIBLE_DEVICES") from exc
    return devices


def logical_to_physical_device_id(logical_device: int, visible_devices: str | None = None) -> int:
    physical_devices = parse_visible_devices(visible_devices)
    if physical_devices is None:
        return logical_device

    if logical_device < 0 or logical_device >= len(physical_devices):
        raise ValueError(
            f"Logical device index {logical_device} is out of bounds for ASCEND_RT_VISIBLE_DEVICES={visible_devices!r}"
        )
    return physical_devices[logical_device]


def get_current_logical_device_index() -> int:
    return torch.accelerator.current_device_index()


@lru_cache(maxsize=1)
def get_host_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:  # noqa: BLE001
        return socket.gethostbyname(socket.gethostname())


@lru_cache(maxsize=1)
def get_npu_ipc_uuid(host_ip: str | None = None, logical_device: int | None = None) -> str:
    if host_ip is None:
        host_ip = get_host_ip()
    if logical_device is None:
        logical_device = get_current_logical_device_index()
    physical_device = logical_to_physical_device_id(logical_device, os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
    return f"{host_ip}-{physical_device}"
