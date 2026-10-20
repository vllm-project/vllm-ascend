# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TCP-only HCCL rendezvous; CUDA unique-ID initialization is not supported."""

from typing import Any

import torch
from vllm.distributed.utils import StatelessProcessGroup


def stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: int,
) -> Any:
    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

    group = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
    communicator = PyHcclCommunicator(group, device=device)
    if not communicator.available or communicator.disabled:
        raise RuntimeError("HCCL communicator is unavailable or disabled")
    return communicator


def trainer_init(init_info: Any, rank: int = 0) -> Any:
    fields = ("master_address", "master_port", "world_size")
    address, port, size = (
        init_info[name] if isinstance(init_info, dict) else getattr(init_info, name) for name in fields
    )
    device = torch.accelerator.current_device_index()
    if device is None:
        raise ValueError("HCCL trainer requires a current NPU device")
    return stateless_init_process_group(address, port, rank, size, device)
