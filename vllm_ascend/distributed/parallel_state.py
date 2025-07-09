from typing import Optional

import torch
from vllm.distributed.parallel_state import (GroupCoordinator, get_world_group,
                                             init_model_parallel_group)

# Currently, mc2 op need their own group coordinator.
_MC2: Optional[GroupCoordinator] = None


def get_mc2_group() -> GroupCoordinator:
    assert _MC2 is not None, ("mc2 group is not initialized")
    return _MC2


def model_parallel_initialized():
    return (_MC2 is not None)


def init_ascend_model_parallel(
    expert_parallel_size: int = 1,
    world_size: Optional[int] = None,
    backend: Optional[str] = None,
):
    if model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    world_size = world_size or torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    num_expert_parallel_groups = world_size // expert_parallel_size

    global _MC2
    group_ranks = []
    for i in range(num_expert_parallel_groups):
        ranks = list(range(i, world_size, num_expert_parallel_groups))
        group_ranks.append(ranks)

    _MC2 = init_model_parallel_group(group_ranks,
                                     get_world_group().local_rank,
                                     backend,
                                     group_name="mc2")


def destroy_ascend_model_parallel():
    global _MC2
    if _MC2:
        _MC2.destroy()
    _MC2 = None
