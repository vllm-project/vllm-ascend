# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared server, device and control helpers for RLHF tests across card counts."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from tests.e2e.conftest import RemoteOpenAIServer

CONTROL_TIMEOUT = 60


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


def post_control(server: RemoteOpenAIServer, route: str, *, json=None, params=None, timeout=CONTROL_TIMEOUT):
    response = requests.post(server.url_for(route), json=json, params=params, timeout=timeout)
    response.raise_for_status()
    return response


def wait_for_free_device_memory(
    device_index: int,
    gpu_memory_utilization: float,
    *,
    timeout: float = 300.0,
    margin_bytes: int = 2 * 2**30,
) -> None:
    """Block until a card has enough free HBM for a server at this utilization.

    The reference server and the live-update lane share one card, and a vLLM
    server reserves ``gpu_memory_utilization`` of whatever HBM is free *when it
    starts*. The previous server's process tree needs a moment to hand its HBM
    back after it exits, so a lane started in that window dies in
    ``init_device`` with "Free memory ... is less than desired GPU memory
    utilization" — an environment race, not a transfer bug. Waiting here turns
    that into a short pause. A device that reports no total (stubbed CPU
    environments) is not gated at all.
    """
    import torch

    deadline = time.monotonic() + timeout
    while True:
        free_bytes, total_bytes = torch.npu.mem_get_info(device_index)
        if total_bytes <= 0:
            return
        if free_bytes >= gpu_memory_utilization * total_bytes + margin_bytes:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"device {device_index} did not free enough HBM for "
                f"gpu_memory_utilization={gpu_memory_utilization}: "
                f"{free_bytes / 2**30:.1f}/{total_bytes / 2**30:.1f} GiB free after {timeout:.0f}s"
            )
        time.sleep(5)


_ENGINES_REGISTERED = False


def register_engines_once() -> None:
    """Register the Ascend weight transfer engines exactly once per process.

    ``register_engine()`` is not idempotent: the underlying factories raise
    ``ValueError: Weight transfer trainer engine 'hccl' is already registered``
    when a backend name is registered twice. Every test module that needs the
    engines must therefore share this process-wide guard instead of keeping a
    per-module flag, which would raise as soon as two modules run in the same
    pytest session (e.g. the NPU IPC and HCCL suites).
    """
    global _ENGINES_REGISTERED
    if _ENGINES_REGISTERED:
        return
    from vllm_ascend.distributed.weight_transfer import register_engine

    register_engine()
    _ENGINES_REGISTERED = True


@contextmanager
def rlhf_server(
    model: str,
    *,
    vllm_serve_args: list[str],
    server_port: int,
    env_dict: dict[str, str] | None = None,
    server_host: str = "127.0.0.1",
) -> Iterator[RemoteOpenAIServer]:
    """Start a server on an explicit port and release it when the caller exits."""
    from tests.e2e.conftest import RemoteOpenAIServer

    with RemoteOpenAIServer(
        model,
        vllm_serve_args=vllm_serve_args,
        server_host=server_host,
        server_port=server_port,
        env_dict=env_dict,
        auto_port=False,
    ) as server:
        yield server
