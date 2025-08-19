from typing import Optional

import torch
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import (GroupCoordinator, get_world_group,
                                             init_model_parallel_group)
from vllm_ascend.utils import oproj_tp_enable

# Currently, mc2 op need their own group coordinator.
_MC2: Optional[GroupCoordinator] = None
_OTP: Optional[GroupCoordinator] = None


def get_mc2_group() -> GroupCoordinator:
    assert _MC2 is not None, ("mc2 group is not initialized")
    return _MC2


def get_otp_group() -> GroupCoordinator:
    assert _OTP is not None, ("oproj tp group is not initialized")
    return _OTP


def model_parallel_initialized():
    return (_MC2 is not None)


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
    
    if oproj_tp_enable():
        group_ranks = []
        global _OTP
        num_o_proj_tensor_parallel_groups: int = (world_size //
                                                parallel_config.oproj_tensor_parallel_size)
        for i in range(num_o_proj_tensor_parallel_groups):
            ranks = list(
                range(i * parallel_config.oproj_tensor_parallel_size,
                    (i + 1) * parallel_config.oproj_tensor_parallel_size))
            group_ranks.append(ranks)
        _OTP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="otp")


def destroy_ascend_model_parallel():
    global _MC2
    if _MC2:
        _MC2.destroy()
    _MC2 = None

    global _OTP
    if _OTP:
        _OTP.destroy()
    _OTP = None
