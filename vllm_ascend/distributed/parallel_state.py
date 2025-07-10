from typing import Optional

import torch
from vllm.distributed.parallel_state import (GroupCoordinator, get_world_group,
                                             init_model_parallel_group)

# vllm-ascend will maintain its own EP GroupCoordinator and ETP GroupCoordinator for
# customize parallel solution
_EP: Optional[GroupCoordinator] = None
_ETP: Optional[GroupCoordinator] = None


def get_ep_group() -> GroupCoordinator:
    assert _EP is not None, ("expert model parallel group is not initialized")
    return _EP


def get_etp_group() -> GroupCoordinator:
    assert _ETP is not None, (
        "expert tensor parallel group is not initialized")
    return _ETP


def model_parallel_initialized():
    return (_ETP is not None and _EP is not None)


def init_ascend_model_parallel(
    expert_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    world_size: Optional[int] = None,
    backend: Optional[str] = None,
):
    if model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    world_size = world_size or torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    num_expert_parallel_groups = expert_tensor_parallel_size
    num_expert_tensor_parallel_groups = expert_parallel_size

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


def destory_ascend_model_parallel():
    global _EP
    if _EP:
        _EP.destroy()
    _EP = None

    global _ETP
    if _ETP:
        _ETP.destroy()
    _ETP = None

_DP1: Optional[GroupCoordinator] = None
_DP2: Optional[GroupCoordinator] = None

def get_dp1_group() -> GroupCoordinator:
    assert _DP1 is not None, ("data parallel group is not initialized")
    return _DP1


def get_dp2_group() -> GroupCoordinator:
    assert _DP2 is not None, ("data parallel group is not initialized")
    return _DP2


def initialize_flashcomm_dp(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        data_parallel_size=1,
        backend: Optional[str] = None,
) -> None:
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    all_ranks = torch.arange(world_size).reshape(
        -1, data_parallel_size, pipeline_model_parallel_size,
        tensor_model_parallel_size)  # noqa
    global _DP1
    assert _DP1 is None, ("data parallel group is already initialized")
    group_ranks = all_ranks.transpose(1,
                                      3).reshape(-1,
                                                 data_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    _DP1 = init_model_parallel_group(group_ranks,
                                     get_world_group().local_rank,
                                     backend,
                                     group_name="dp")
    global _DP2
    assert _DP2 is None, ("data parallel group is already initialized")
    group_ranks = all_ranks.transpose(1,
                                      3).reshape(-1,
                                                 data_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    _DP2 = init_model_parallel_group(group_ranks,
                                     get_world_group().local_rank,
                                     backend,
                                     group_name="dp")
