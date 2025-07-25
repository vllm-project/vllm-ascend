from typing import Optional

import os

import torch
from vllm.distributed.parallel_state import (GroupCoordinator, get_world_group,
                                             init_model_parallel_group,
                                             logger, get_ep_group)

# vllm-ascend will maintain its own _LOCAL_COMM GroupCoordinator for
# customize parallel solution
_LOCAL_COMM_GROUP: Optional[GroupCoordinator] = None

def ascend_model_parallel_initialized():
    return _LOCAL_COMM_GROUP is not None

def calculate_effective_local_size(local_size: int, world_size: int) -> int:
    """
    Calculate the effective local size based on available devices and world size.

    Args:
        local_size (int): Number of available NPU devices.
        world_size (int): Total number of processes in the distributed setup.

    Returns:
        int: The effective local size (minimum of local_size and world_size).

    Notes:
        - Logs a warning if not all devices are used.
        - Ensures world_size is divisible by the effective local size (raises AssertionError otherwise).
    """
    effective_local_size = min(local_size, world_size)
    if effective_local_size < local_size:
        logger.info(f"Note: Using only {effective_local_size} of {local_size} available NPU devices")

    if world_size % effective_local_size != 0:
        raise AssertionError(
            f"world_size ({world_size}) must be divisible by effective_local_size ({effective_local_size})"
        )
    return effective_local_size

def initialize_local_comm_group(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    visible_devices_count = len(os.getenv("ASCEND_RT_VISIBLE_DEVICES", "").split(","))
    local_size = calculate_effective_local_size(torch.npu.device_count() if visible_devices_count == 0 \
        else visible_devices_count, world_size)

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    num_local_groups: int = world_size // local_size
    global _LOCAL_COMM_GROUP
    if _LOCAL_COMM_GROUP is not None:
        raise RuntimeError("_LOCAL_COMM_GROUP must be None")
    group_ranks = []
    for i in range(num_local_groups):
        ranks = list(range(i * local_size, (i + 1) * local_size))
        group_ranks.append(ranks)
    
    logger.info(f"vllm-ascend: world size {world_size}, visible device count {visible_devices_count}, local size {local_size}, "
            "num local groups {num_local_groups}, group ranks {group_ranks}")

    _LOCAL_COMM_GROUP = init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                use_message_queue_broadcaster=True,
                group_name="world_local",
            )

def get_mlp_tp_world_size():
    return get_mlp_world_group().world_size


def get_mlp_tp_rank():
    return get_mlp_world_group().rank_in_group

def get_mlp_world_group() -> GroupCoordinator:
    # Can be enabled
    # ToDo 配置化
    if True:
        return get_local_comm_group()
    else:
        return get_ep_group()

def mlp_tp_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_mlp_world_group().all_gather(input_, dim)

def mlp_tp_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_mlp_world_group().all_reduce(input_)

def mlp_tp_reduce_scatter(input_: torch.Tensor) -> torch.Tensor:
    """reduce scatter the input tensor across model parallel group."""
    return get_mlp_world_group().reduce_scatter(input_, dim=0)


def get_local_comm_group() -> GroupCoordinator:
    return _LOCAL_COMM_GROUP

def init_ascend_model_parallel(
    backend: Optional[str] = None,
):
    if ascend_model_parallel_initialized():
        return
    initialize_local_comm_group(backend)
    
    logger.info(
    "vllm-ascend: rank %s in world size %s is assigned as "
    "MLP TP rank %s", torch.distributed.get_rank(), torch.distributed.get_world_size(), get_mlp_tp_rank())


def destory_ascend_model_parallel():
    global _LOCAL_COMM_GROUP
    if _LOCAL_COMM_GROUP:
        _LOCAL_COMM_GROUP.destroy()
    _LOCAL_COMM_GROUP = None
