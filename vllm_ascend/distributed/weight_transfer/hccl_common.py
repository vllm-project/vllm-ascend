# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared HCCL initialization helpers for weight transfer engines.

The worker (`HCCLWeightTransferEngine`) and the trainer
(`HCCLTrainerWeightTransferEngine`) are independent engines that share *only*
their process-group initialization and their worker-side init info. That common
logic lives here, mirroring ``vllm/distributed/weight_transfer/nccl_common.py``,
so the trainer engine does not have to reach into the inference engine's private
helpers to open its own endpoint.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from vllm.distributed.weight_transfer.base import WeightTransferInitInfo
from vllm.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
)

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig

    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator


@dataclass
class HCCLWeightTransferInitInfo(WeightTransferInitInfo):
    """Worker-side initialization info for HCCL weight transfer.

    The worker joins the HCCL group at `rank_offset + <its rank across all DP
    groups>`; the trainer sender is always rank 0, so the workers start at
    `rank_offset` (typically 1).

    `packed` / buffer sizes are must-agree wire params: the trainer ships them
    here at the init handshake, so the worker decodes with the same setting the
    trainer encoded with. Their defaults only apply when no trainer ever shipped
    a value (the unreachable receive-before-init case).
    """

    master_address: str
    """IP address of the trainer (rank 0) for HCCL process group setup."""
    master_port: int
    """Port on the trainer for HCCL process group setup."""
    rank_offset: int
    """Offset added to each vLLM worker's rank within the HCCL group.
    Typically 1 (trainer is rank 0, workers start at rank 1)."""
    world_size: int
    """Total number of participants in the HCCL group (trainer + all workers)."""
    packed: bool = False
    """Whether the transfer is packed, matching the trainer's setting."""
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    """Size in bytes for each packed tensor buffer (packed mode only)."""
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS
    """Number of buffers for double/triple buffering (packed mode only)."""


class HCCLRendezvous(Protocol):
    """The TCP rendezvous fields `trainer_init` needs.

    Structural so each engine can keep its own trainer init info next to its
    engine (`HCCLTrainerInitInfo` in ``hccl_engine``) without this shared module
    importing it.
    """

    master_address: str
    master_port: int
    world_size: int


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

    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
    return PyHcclCommunicator(pg, device=device)


def worker_init_process_group(
    init_info: HCCLWeightTransferInitInfo,
    parallel_config: "ParallelConfig",
) -> "PyHcclCommunicator":
    """Create the trainer<->worker HCCL group on an inference worker.

    Computes a unique rank for this worker across all data-parallel groups and
    joins the trainer's endpoint from it.
    """
    # Calculate the global rank in the trainer-worker process group
    # Must account for data parallel to get unique ranks across all workers
    dp_rank = parallel_config.data_parallel_index
    world_size_per_dp = parallel_config.world_size  # TP * PP
    rank_within_dp = parallel_config.rank

    # Unique rank across all DP groups
    worker_rank = dp_rank * world_size_per_dp + rank_within_dp
    rank = worker_rank + init_info.rank_offset

    device = torch.accelerator.current_device_index()
    return stateless_init_process_group(
        init_info.master_address,
        init_info.master_port,
        rank,
        init_info.world_size,
        device=device,
    )


def trainer_init(
    init_info: HCCLRendezvous | dict,
) -> "PyHcclCommunicator":
    """
    Open the trainer-side (rank 0) HCCL endpoint for weight transfer.

    The trainer is always rank 0 in the process group. Uses the current
    Ascend device (torch.accelerator.current_device_index()).

    Args:
        init_info: Any object carrying the `HCCLRendezvous` fields (a trainer or
            worker HCCL init info), or a dict with keys:
            - master_address: str
            - master_port: int
            - world_size: int

    Returns:
        PyHcclCommunicator for weight transfer.
    """
    if isinstance(init_info, dict):
        master_address = init_info["master_address"]
        master_port = init_info["master_port"]
        world_size = init_info["world_size"]
    else:
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
