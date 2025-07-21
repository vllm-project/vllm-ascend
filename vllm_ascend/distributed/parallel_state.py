from typing import Optional

import torch
from vllm.distributed.parallel_state import (GroupCoordinator, get_world_group,
                                             init_model_parallel_group)

# Currently, mc2 op need their own group coordinator.
_MC2: Optional[GroupCoordinator] = None
_LM_HEAD_TP: Optional[GroupCoordinator] = None

def get_lm_tp_group() -> GroupCoordinator:
    assert _LM_HEAD_TP is not None, ("lm tensor model parallel group is not initialized")
    return _LM_HEAD_TP

def get_mc2_group() -> GroupCoordinator:
    assert _MC2 is not None, ("mc2 group is not initialized")
    return _MC2


def model_parallel_initialized():
    return (_MC2 is not None)


def init_ascend_model_parallel(
    expert_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
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
    
    global _LM_HEAD_TP
    assert _LM_HEAD_TP is None, ("lm head tensor model parallel group is already initialized")
    lm_tp  = 8
    
    all_ranks_lm_head = torch.arange(world_size).reshape(
        -1, lm_tp, pipeline_parallel_size, 1)  # noqa
    group_ranks = all_ranks_lm_head.view(-1, lm_tp).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    
    # message queue broadcaster is only used in tensor model parallel group
    _LM_HEAD_TP = init_model_parallel_group(group_ranks,
                                            get_world_group().local_rank,
                                            backend,
                                            group_name="lm_head_tp")

def get_lm_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_lm_tp_group().world_size

def get_lm_tensor_model_parallel_rank():
    """Return world size for the tensor model parallel group."""
    return get_lm_tp_group().rank_in_group


def destroy_ascend_model_parallel():
    global _MC2
    if _MC2:
        _MC2.destroy()
    _MC2 = None
