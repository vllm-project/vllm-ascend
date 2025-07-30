from typing import Optional

import torch
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import (GroupCoordinator, get_world_group,
                                             init_model_parallel_group)

# Currently, mc2 op need their own group coordinator.
_MC2: Optional[GroupCoordinator] = None
_LMHEAD: Optional[GroupCoordinator] = None


def get_mc2_group() -> GroupCoordinator:
    assert _MC2 is not None, ("mc2 group is not initialized")
    return _MC2


def get_lmhead_group() -> GroupCoordinator:
    assert _LMHEAD is not None, ("mc2 group is not initialized")
    return _LMHEAD


def model_parallel_initialized():
    return (_MC2 is not None and _LMHEAD is not None)


def init_ascend_model_parallel(parallel_config: ParallelConfig, ):
    if model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    backend = torch.distributed.get_backend(get_world_group().device_group)

    # The layout of all ranks: ExternalDP * EP
    # ExternalDP is the data parallel group that is not part of the model,
    # every dp rank can generate independently (in verl integration).
    all_ranks = torch.arange(world_size).reshape(
        -1, parallel_config.data_parallel_size *
        parallel_config.tensor_parallel_size)
    global _MC2
    group_ranks = all_ranks.unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]

    _MC2 = init_model_parallel_group(group_ranks,
                                     get_world_group().local_rank,
                                     backend,
                                     group_name="mc2")

    all_ranks = torch.arange(world_size).reshape(-1, parallel_config.lmhead_tp_size)
    global _LMHEAD
    group_ranks = all_ranks.unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]

    _LMHEAD = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="lmhead")


def destroy_ascend_model_parallel():
    global _MC2
    if _MC2:
        _MC2.destroy()
    _MC2 = None
    global _LMHEAD
    if _LMHEAD:
        _LMHEAD.destroy()
    _LMHEAD = None
