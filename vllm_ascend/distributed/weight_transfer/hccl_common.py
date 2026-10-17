# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HCCL process-group initialization helpers.

Mirrors `vllm.distributed.weight_transfer.nccl_common`: the HCCL backend speaks
upstream's broadcast protocol over a `StatelessProcessGroup`, so only the
communicator constructor, the TCP-only rendezvous and the NPU device lookup
differ from the upstream helpers.
"""

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig

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


def worker_init_payload(init_info: "HCCLWeightTransferInitInfo") -> dict[str, Any]:
    """Serialize a worker init info for `init_weight_transfer_engine`.

    Unset (``None``) wire params are dropped, so the payload carries only what
    the trainer actually fixed and the worker keeps its per-update fallback for
    the ones left out.
    """
    return {key: value for key, value in asdict(init_info).items() if value is not None}


def worker_init_process_group(
    init_info: "HCCLWeightTransferInitInfo",
    parallel_config: "ParallelConfig",
) -> "PyHcclCommunicator":
    """Create the trainer<->worker HCCL group on an inference worker.

    The rank arithmetic mirrors
    `vllm.distributed.weight_transfer.nccl_common.worker_init_process_group`: a
    worker's rank has to be unique across data-parallel groups, and the trainer
    occupies the ranks before `rank_offset`.
    """
    dp_rank = parallel_config.data_parallel_index
    world_size_per_dp = parallel_config.world_size
    rank_within_dp = parallel_config.rank
    rank = dp_rank * world_size_per_dp + rank_within_dp + init_info.rank_offset

    device = torch.accelerator.current_device_index()
    if device is None:
        raise ValueError("HCCL weight transfer requires an explicit current NPU device")
    return stateless_init_process_group(
        init_info.master_address,
        init_info.master_port,
        rank,
        init_info.world_size,
        device,
    )


def trainer_init(
    init_info: "HCCLTrainerInitInfo | HCCLWeightTransferInitInfo",
    rank: int = 0,
) -> "PyHcclCommunicator":
    """Open the trainer-side endpoint as ``rank``, defaulting to the sender."""
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
