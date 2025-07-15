from typing import Optional

import torch
from vllm.distributed.parallel_state import (GroupCoordinator, get_world_group,
                                             init_model_parallel_group)

from vllm_ascend.ascend_config import get_ascend_config

# Currently, mc2 op need their own group coordinator.
_MC2: Optional[GroupCoordinator] = None
# Local TP Group of size 2, used to perform Tensor Parallel
# for mla o_proj and LLM-Head of deepseek in pure DP/EP scenario.
_OPROJ_TP: Optional[GroupCoordinator] = None


def get_mc2_group() -> GroupCoordinator:
    assert _MC2 is not None, ("mc2 group is not initialized")
    return _MC2


def get_oproj_tp_group() -> GroupCoordinator:
    assert _OPROJ_TP is not None, ("local oproj tp group has not been initialized")
    return _OPROJ_TP


def init_ascend_model_parallel(
    expert_parallel_size: int = 1,
    world_size: Optional[int] = None,
    backend: Optional[str] = None,
):
    assert torch.distributed.is_initialized()
    world_size = world_size or torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    num_expert_parallel_groups = world_size // expert_parallel_size
    global _MC2, _OPROJ_TP
    if _MC2 is None:
        group_ranks = []
        for i in range(num_expert_parallel_groups):
            ranks = list(range(i, world_size, num_expert_parallel_groups))
            group_ranks.append(ranks)

        _MC2 = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="mc2")
    ascend_config = get_ascend_config()
    o_proj_tp = ascend_config.o_proj_tp
    if o_proj_tp > 1 and _OPROJ_TP is None:
        assert world_size % o_proj_tp == 0, "World size must be divisible by o_proj_tp in order to use local o_proj TP." \
            f"Current world_size: {world_size}"
        group_ranks = []
        for i in range(0, world_size, o_proj_tp):
            group_ranks.append(list(range(i, i + o_proj_tp)))
        _OPROJ_TP = init_model_parallel_group(group_ranks,
                                              get_world_group().local_rank,
                                              backend,
                                              group_name="oproj_tp")    


def destroy_ascend_model_parallel():
    global _MC2, _OPROJ_TP
    if _MC2:
        _MC2.destroy()
    if _OPROJ_TP:
        _OPROJ_TP.destroy()
    _MC2 = None
    _OPROJ_TP = None
