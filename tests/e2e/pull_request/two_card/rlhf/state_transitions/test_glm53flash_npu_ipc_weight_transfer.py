# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP=2 Flash IPC unpacked reload with one colocated trainer per NPU.

The two trainer ranks exchange IPC handle metadata over a CPU/Gloo group. Each
inference worker imports the handles for its own physical NPU; checkpoint tensors
are loaded through the normal TP weight loaders. Only trainer rank 0 drives HTTP
control requests. Weight transport remains NPU IPC throughout.
"""

import os
from contextlib import ExitStack
from datetime import timedelta

import pytest
import requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401  # registers the NPU backend
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import (
    FixedRandomWeightSource,
    assert_weight_update_matches_reference,
    generation_signature,
    glm53flash_ipc_case,
    live_update_serve_args,
    reference_signature,
    register_engines_once,
    wait_for_free_device_memory,
)

WORLD_SIZE = 2
UPDATE_ROUNDS = 2
GPU_MEMORY_UTILIZATION = 0.45
CONTROL_TIMEOUT = 60
# Rank 1 may wait while rank 0 starts and captures both reference/live servers.
GROUP_TIMEOUT = timedelta(minutes=20)


def _server_devices(first_device: int, visible_devices: str | None) -> str:
    """Map the selected logical trainer devices to physical server device IDs."""
    if first_device < 0:
        raise ValueError("first_device must be nonnegative")
    if visible_devices:
        devices = [int(value.strip()) for value in visible_devices.split(",")]
        selected = devices[first_device : first_device + WORLD_SIZE]
        if len(selected) != WORLD_SIZE or len(set(selected)) != WORLD_SIZE or min(selected) < 0:
            raise ValueError("two distinct visible NPUs are required")
    else:
        selected = list(range(first_device, first_device + WORLD_SIZE))
    return ",".join(map(str, selected))


def _post(server, route, *, params=None):
    response = requests.post(server.url_for(route), params=params, timeout=CONTROL_TIMEOUT)
    response.raise_for_status()


def _wait_for_devices(first_device):
    for device in range(first_device, first_device + WORLD_SIZE):
        wait_for_free_device_memory(device, GPU_MEMORY_UTILIZATION)


def _start_servers(stack, case, source, first_device):
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "ASCEND_RT_VISIBLE_DEVICES": _server_devices(first_device, os.getenv("ASCEND_RT_VISIBLE_DEVICES")),
        **dict(case.server_env),
    }
    _wait_for_devices(first_device)
    reference = reference_signature(
        source,
        case,
        port=get_open_port(),
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=WORLD_SIZE,
        device_index=first_device,
        env_dict=env_dict,
    )
    _wait_for_devices(first_device)
    port = get_open_port()
    server = stack.enter_context(
        RemoteOpenAIServer(
            case.model,
            vllm_serve_args=live_update_serve_args(
                case,
                backend="npu_ipc",
                port=port,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                tensor_parallel_size=WORLD_SIZE,
            ),
            server_host="127.0.0.1",
            server_port=port,
            env_dict=env_dict,
            auto_port=False,
        )
    )
    dummy = generation_signature(server.get_client(), case.model)
    return server, reference, dummy


def _send_updates(rank, engine, server, case):
    signatures = []
    for _ in range(UPDATE_ROUNDS):
        if rank == 0:
            _post(server, "pause")
            _post(server, "sleep", params={"level": 2})
            _post(server, "wake_up", params={"tags": ["weights"]})
        # Both weights pools must be mapped before either trainer materializes
        # its full checkpoint tensors and publishes device-local IPC handles.
        dist.barrier()
        engine.send_weights()
        if rank == 0:
            _post(server, "wake_up", params={"tags": ["kv_cache"]})
            signatures.append(generation_signature(server.get_client(), case.model))
        dist.barrier()
    return signatures


def _run_rank_transaction(rank, first_device):
    # Lazy imports isolate trainer factory registration inside spawned ranks.
    from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
    from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo

    case = glm53flash_ipc_case()
    source = FixedRandomWeightSource(case, torch.device("npu", first_device + rank))
    with ExitStack() as stack:
        server = None
        endpoint = [None]
        if rank == 0:
            server, reference, dummy = _start_servers(stack, case, source, first_device)
            endpoint[0] = server.url_root
        dist.broadcast_object_list(endpoint, src=0)
        register_engines_once()
        engine = WeightTransferTrainerFactory.trainer_init(
            NPUIPCTrainerInitInfo(rank=rank, packed=False),
            client=HTTPVLLMWeightSyncClient(base_url=endpoint[0]),
            source=source,
        )
        try:
            signatures = _send_updates(rank, engine, server, case)
        finally:
            engine.shutdown()
        if rank == 0:
            assert_weight_update_matches_reference(dummy, reference, *signatures, case)


def _trainer_rank(rank, group_port, first_device):
    torch.npu.set_device(first_device + rank)
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{group_port}",
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=GROUP_TIMEOUT,
    )
    try:
        _run_rank_transaction(rank, first_device)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.npu.device_count() < WORLD_SIZE, reason="Flash TP=2 IPC requires two NPUs")
def test_glm53flash_npu_ipc_unpacked_tp2():
    first_device = int(os.environ.get("VLLM_RL_TEST_DEVICE_INDEX", "0"))
    if first_device < 0 or first_device + WORLD_SIZE > torch.npu.device_count():
        pytest.fail("VLLM_RL_TEST_DEVICE_INDEX must select two available logical NPUs")
    _server_devices(first_device, os.getenv("ASCEND_RT_VISIBLE_DEVICES"))
    # spawn surfaces rank failures and terminates peer trainers on an exception.
    mp.spawn(_trainer_rank, args=(get_open_port(), first_device), nprocs=WORLD_SIZE, join=True)
