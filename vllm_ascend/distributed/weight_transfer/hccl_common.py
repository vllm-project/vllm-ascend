# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared HCCL initialization helpers for weight transfer engines."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.distributed.weight_transfer.base import WeightTransferInitInfo

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig

    from vllm_ascend.distributed.device_communicators.pyhccl import (
        PyHcclCommunicator,
    )


@dataclass
class HCCLWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for HCCL weight transfer backend."""

    master_address: str
    """IP address of the trainer (rank 0) for HCCL process group setup."""
    master_port: int
    """Port on the trainer for HCCL process group setup."""
    rank_offset: int
    """Offset added to each vLLM worker's rank within the HCCL group.
    Typically 1 (trainer is rank 0, workers start at rank 1)."""
    world_size: int
    """Total number of participants in the HCCL group (trainer + all workers)."""


def stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device,
) -> "PyHcclCommunicator":
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (HCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.utils import StatelessProcessGroup

    from vllm_ascend.distributed.device_communicators.pyhccl import (
        PyHcclCommunicator,
    )

    pg = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=rank,
        world_size=world_size,
    )
    return PyHcclCommunicator(pg, device=device)


def worker_init_process_group(
    init_info: HCCLWeightTransferInitInfo,
    parallel_config: "ParallelConfig",
) -> "PyHcclCommunicator":
    """
    Initialize HCCL process group with the trainer.

    Args:
        init_info: HCCL initialization info containing master address, port,
                  rank offset, and world size
        parallel_config: vLLM parallel configuration used to calculate the
                         worker's unique rank across data-parallel groups
    """

    # Calculate the global rank in the trainer-worker process group
    # Must account for data parallel to get unique ranks across all workers
    dp_rank = parallel_config.data_parallel_index
    world_size_per_dp = parallel_config.world_size  # TP * PP
    rank_within_dp = parallel_config.rank

    # Unique rank across all DP groups
    worker_rank = dp_rank * world_size_per_dp + rank_within_dp
    rank = worker_rank + init_info.rank_offset
    # Create stateless process group
    device = torch.accelerator.current_device_index()
    return stateless_init_process_group(
        init_info.master_address,
        init_info.master_port,
        rank,
        init_info.world_size,
        device=device,
    )


def trainer_init(
    init_info: HCCLWeightTransferInitInfo | dict,
) -> "PyHcclCommunicator":
    """
    Initialize HCCL process group for trainer-side weight transfer.

    The trainer is always rank 0 in the process group. Uses the current
    Ascend device (torch.accelerator.current_device_index()).

    Args:
        init_info: Either an HCCLWeightTransferInitInfo object or a dict with keys:
            - master_address: str
            - master_port: int
            - world_size: int

    Returns:
        PyHcclCommunicator for weight transfer.

    Example:
        >>> from vllm.distributed.weight_transfer.hccl_engine import (
        ...     HCCLWeightTransferEngine,
        ... )
        >>> group = HCCLWeightTransferEngine.trainer_init(
        ...     dict(
        ...         master_address=master_address,
        ...         master_port=master_port,
        ...         world_size=world_size,
        ...     ),
        ... )
    """
    if isinstance(init_info, dict):
        master_address = init_info["master_address"]
        master_port = init_info["master_port"]
        world_size = init_info["world_size"]
    else:
        # HCCLWeightTransferInitInfo object
        master_address = init_info.master_address
        master_port = init_info.master_port
        world_size = init_info.world_size

    # Trainer is always rank 0
    device = torch.accelerator.current_device_index()
    return stateless_init_process_group(
        master_address,
        master_port,
        0,
        world_size,
        device,
    )
