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
    backend: Optional[str] = None,
):
    if model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)

    # The layout of all ranks: ExternalDP * EP * ETP
    # ExternalDP is the data parallel group that is not part of the model,
    # every dp rank can generate independently (in verl integration).
    all_ranks = torch.arange(world_size).reshape(-1, expert_parallel_size,
                                                 expert_tensor_parallel_size)

    global _EP
    group_ranks = all_ranks.transpose(1,
                                      2).view(-1,
                                              expert_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    _EP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="ep")

    global _ETP
    group_ranks = all_ranks.view(-1, expert_tensor_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
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
