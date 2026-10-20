# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TCP-rendezvous helpers for the HCCL weight-transfer backend.

Upstream's `nccl_common` builds a `PyNcclCommunicator` inside every helper, so
the HCCL variants cannot delegate to it. Everything else is reused from
upstream: the worker init payload serializer and the rank arithmetic (kept
inline in the engine, mirroring `nccl_common.worker_init_process_group`).
"""

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator


def stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: int,
) -> "PyHcclCommunicator":
    """Create an HCCL communicator without touching torch's default group."""
    from vllm.distributed.utils import StatelessProcessGroup

    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

    pg = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=rank,
        world_size=world_size,
    )
    communicator = PyHcclCommunicator(pg, device=device)
    if not communicator.available or communicator.disabled:
        raise RuntimeError(
            "HCCL weight-transfer communicator is unavailable or disabled; "
            "refusing to continue as if the broadcast succeeded"
        )
    return communicator


def trainer_init(init_info: Any, rank: int = 0) -> "PyHcclCommunicator":
    """Open the trainer-side endpoint as ``rank``, defaulting to the sender."""
    fields = ("master_address", "master_port", "world_size")
    address, port, world_size = (
        init_info[name] if isinstance(init_info, dict) else getattr(init_info, name) for name in fields
    )
    device = torch.accelerator.current_device_index()
    if device is None:
        raise ValueError("HCCL trainer requires an explicit current NPU device")
    return stateless_init_process_group(address, port, rank, world_size, device)
