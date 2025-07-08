from typing import Optional

import torch
from vllm.distributed.parallel_state import (GroupCoordinator, get_world_group,
                                             init_model_parallel_group,
                                             logger, get_dp_group, get_pp_group, get_tp_group)

# vllm-ascend will maintain its own EP GroupCoordinator and ETP GroupCoordinator for
# customize parallel solution
_EP: Optional[GroupCoordinator] = None
_ETP: Optional[GroupCoordinator] = None
_MLPTP: Optional[GroupCoordinator] = None


def get_ep_group() -> GroupCoordinator:
    assert _EP is not None, ("expert model parallel group is not initialized")
    return _EP


def get_etp_group() -> GroupCoordinator:
    assert _ETP is not None, (
        "expert tensor parallel group is not initialized")
    return _ETP

def get_mlptp_group() -> GroupCoordinator:
    assert _MLPTP is not None, (
        "mlp tensor parallel group is not initialized")
    return _MLPTP

def get_mlp_tensor_model_parallel_world_size():
    """Return world size for the mlp tensor model parallel group."""
    return get_mlptp_group().world_size

def get_mlp_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return get_mlptp_group().rank_in_group

def model_parallel_initialized():
    return (_ETP is not None and _EP is not None and _MLPTP is not None)


def init_ascend_model_parallel(
    expert_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    world_size: Optional[int] = None,
    backend: Optional[str] = None,
    mlp_tensor_parallel_size: Optional[int] = 4,
):
    if model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    world_size = world_size or torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    num_expert_parallel_groups = expert_tensor_parallel_size
    num_expert_tensor_parallel_groups = expert_parallel_size
    assert (world_size % mlp_tensor_parallel_size == 0), (
                "world_size must be divisible by mlp_tensor_parallel_size")
    num_mlp_tensor_parallel_groups = world_size // mlp_tensor_parallel_size

    global _EP
    group_ranks = []
    for i in range(num_expert_parallel_groups):
        ranks = list(range(i, world_size, num_expert_parallel_groups))
        group_ranks.append(ranks)

    _EP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="ep")

    group_ranks = []
    global _ETP
    for i in range(num_expert_tensor_parallel_groups):
        ranks = list(
            range(i * expert_tensor_parallel_size,
                  (i + 1) * expert_tensor_parallel_size))
        group_ranks.append(ranks)

    _ETP = init_model_parallel_group(group_ranks,
                                     get_world_group().local_rank,
                                     backend,
                                     group_name="etp")

    group_ranks = []
    global _MLPTP
    for i in range(num_mlp_tensor_parallel_groups):
        ranks = list(
            range(i * mlp_tensor_parallel_size,
                  (i + 1) * mlp_tensor_parallel_size))
        group_ranks.append(ranks)
    # Build the mlp tensor model-parallel groups.
    _MLPTP = init_model_parallel_group(group_ranks,
                                     get_world_group().local_rank,
                                     backend,
                                     group_name="mlptp")
    
    logger.info(
    "vllm-ascend: rank %s in world size %s is assigned as "
    "DP rank %s, PP rank %s, TP rank %s, EP rank %s, MLP TP rank %s", torch.distributed.get_rank(), world_size,
    get_dp_group.rank_in_group, get_pp_group.rank_in_group, get_tp_group.rank_in_group,
    _EP.rank_in_group, _MLPTP.rank_in_group)


def destory_ascend_model_parallel():
    global _EP
    if _EP:
        _EP.destroy()
    _EP = None

    global _ETP
    if _ETP:
        _ETP.destroy()
    _ETP = None

    global _MLPTP
    if _MLPTP:
        _MLPTP.destroy()
    _MLPTP = None
