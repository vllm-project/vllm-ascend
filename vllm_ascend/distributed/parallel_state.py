from typing import Optional

import torch
from vllm.distributed.parallel_state import (GroupCoordinator, get_world_group,
                                             init_model_parallel_group)

# Currently, mc2 op need their own group coordinator.
_MC2: Optional[GroupCoordinator] = None
_LMHEAD: Optional[GroupCoordinator] = None
_OTP: Optional[GroupCoordinator] = None
_QKVTP: Optional[GroupCoordinator] = None


def get_mc2_group() -> GroupCoordinator:
    assert _MC2 is not None, ("mc2 group is not initialized")
    return _MC2


def get_lmhead_group() -> GroupCoordinator:
    assert _LMHEAD is not None, ("lmhead group is not initialized")
    return _LMHEAD

def get_otp_group() -> GroupCoordinator:
    assert _OTP is not None, ("otp group is not initialized")
    return _OTP

def get_qkvtp_group() -> GroupCoordinator:
    assert _QKVTP is not None, ("qkvtp group is not initialized")
    return _QKVTP

def model_parallel_initialized():
    return (_MC2 is not None)


def init_ascend_model_parallel(
    expert_parallel_size: int = 1,
    lm_head_tp_size: int = -1,
    oproj_tensor_parallel_size: int = 1,
    qkvproj_tensor_parallel_size: int = 1,
    backend: Optional[str] = None,
):
    if model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)

    # The layout of all ranks: ExternalDP * EP
    # ExternalDP is the data parallel group that is not part of the model,
    # every dp rank can generate independently (in verl integration).
    all_ranks = torch.arange(world_size).reshape(-1, expert_parallel_size)
    global _MC2
    group_ranks = all_ranks.unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]

    _MC2 = init_model_parallel_group(group_ranks,
                                     get_world_group().local_rank,
                                     backend,
                                     group_name="mc2")

    if lm_head_tp_size > 0:
        all_ranks = torch.arange(world_size).reshape(-1, lm_head_tp_size)
        global _LMHEAD
        group_ranks = all_ranks.unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]

        _LMHEAD = init_model_parallel_group(group_ranks,
                                            get_world_group().local_rank,
                                            backend,
                                            group_name="lmhead")
    
    if oproj_tensor_parallel_size > 1:
        group_ranks = []
        global _OTP
        num_o_proj_tensor_parallel_groups: int = (world_size //
                                                oproj_tensor_parallel_size)
        for i in range(num_o_proj_tensor_parallel_groups):
            ranks = list(
                range(i * oproj_tensor_parallel_size,
                    (i + 1) * oproj_tensor_parallel_size))
            group_ranks.append(ranks)
        _OTP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="otp")
    
    if qkvproj_tensor_parallel_size > 1:
        group_ranks = []
        global _QKVTP
        
        num_qkv_proj_tensor_parallel_groups: int = (world_size //
                                                qkvproj_tensor_parallel_size)
        for i in range(num_qkv_proj_tensor_parallel_groups):
            ranks = list(
                range(i * qkvproj_tensor_parallel_size,
                    (i + 1) * qkvproj_tensor_parallel_size))
            group_ranks.append(ranks)
        _QKVTP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank,
                                        backend,
                                        group_name="qkvtp")


def destroy_ascend_model_parallel():
    global _MC2
    if _MC2:
        _MC2.destroy()
    _MC2 = None
    global _LMHEAD
    if _LMHEAD:
        _LMHEAD.destroy()
    _LMHEAD = None
    global _OTP
    if _OTP:
        _OTP.destroy()
    _OTP = None

    global _QKVTP
    if _QKVTP:
        _QKVTP.destroy()
    _QKVTP = None
