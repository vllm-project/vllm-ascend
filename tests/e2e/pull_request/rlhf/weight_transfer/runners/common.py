# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared server lifecycle and control requests for weight-transfer E2Es."""

import os
from contextlib import contextmanager

import requests
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.rlhf.weight_transfer.models.base import WeightUpdateModelCase
from tests.e2e.pull_request.rlhf.weight_transfer.utils import (
    FixedRandomWeightSource,
    generation_signature,
    live_update_serve_args,
    reference_signature,
    wait_for_free_device_memory,
)

UPDATE_TIMEOUT = 300
CONTROL_TIMEOUT = 60
UPDATE_ROUNDS = 2


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


@contextmanager
def weight_transfer_servers(
    source: FixedRandomWeightSource,
    case: WeightUpdateModelCase,
    *,
    backend: str,
    tensor_parallel_size: int,
    device_index: int,
    gpu_memory_utilization: float,
    env_dict: dict[str, str],
):
    """Load W independently, then start the dummy lane under identical settings.

    The reference server must release every TP device before the live-update
    server starts. Both use the case's environment and the same generated W.
    """
    reference = reference_signature(
        source,
        case,
        port=get_open_port(),
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        device_index=device_index,
        env_dict=env_dict,
    )
    for rank in range(tensor_parallel_size):
        wait_for_free_device_memory(device_index + rank, gpu_memory_utilization)
    port = get_open_port()
    with RemoteOpenAIServer(
        case.model,
        vllm_serve_args=live_update_serve_args(
            case,
            backend=backend,
            port=port,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        ),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
        auto_port=False,
    ) as server:
        dummy_signature = generation_signature(server.get_client(), case.model)
        yield server, reference, dummy_signature
