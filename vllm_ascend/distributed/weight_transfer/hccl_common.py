# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared HCCL process-group initialization helpers."""

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import ParallelConfig

    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator
    from vllm_ascend.distributed.weight_transfer.hccl_engine import (
        HCCLTrainerInitInfo,
        HCCLWeightTransferInitInfo,
    )


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
    pyhccl = PyHcclCommunicator(pg, device=device)
    if not pyhccl.available or pyhccl.disabled:
        raise RuntimeError(
            "HCCL weight-transfer communicator is unavailable or disabled; "
            "refusing to continue as if the broadcast succeeded"
        )
    return pyhccl


def worker_init_process_group(
    init_info: "HCCLWeightTransferInitInfo",
    parallel_config: "ParallelConfig",
) -> "PyHcclCommunicator":
    """Create the worker communicator using its DP/TP/PP rank."""
    dp_rank = parallel_config.data_parallel_index
    world_size_per_dp = parallel_config.world_size
    rank_within_dp = parallel_config.rank
    worker_rank = dp_rank * world_size_per_dp + rank_within_dp
    rank = worker_rank + init_info.rank_offset
    device = torch.accelerator.current_device_index()
    if device is None:
        raise ValueError("HCCL worker requires an explicit current NPU device")
    return stateless_init_process_group(
        init_info.master_address,
        init_info.master_port,
        rank,
        init_info.world_size,
        device,
    )


def trainer_init(
    init_info: "HCCLTrainerInitInfo | HCCLWeightTransferInitInfo | Any",
    rank: int = 0,
) -> "PyHcclCommunicator":
    """Create the trainer-side communicator, defaulting to rank zero."""
    import torch

    device = torch.accelerator.current_device_index()
    if device is None:
        raise ValueError("HCCL trainer requires an explicit current NPU device")
    return stateless_init_process_group(
        init_info.master_address,
        init_info.master_port,
        rank,
        init_info.world_size,
        device,
    )
