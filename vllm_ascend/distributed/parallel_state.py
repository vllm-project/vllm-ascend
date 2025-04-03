from typing import Optional
import torch
from torch.distributed import ProcessGroup, TCPStore, is_hccl_available
from torch.distributed.distributed_c10d import (Backend, PrefixStore,
                                                _get_default_timeout,
                                                is_nccl_available)
from vllm.distributed.parallel_state import GroupCoordinator, get_world_group, init_model_parallel_group
from vllm.config import ParallelConfig
from torch.distributed.rendezvous import rendezvous

# vllm-ascend will maintain its own EP GroupCoordinator and ETP GroupCoordinator for 
# customize parallel solution
_EP: Optional[GroupCoordinator] = None
_ETP: Optional[list[GroupCoordinator]] = None


def get_ep_group() -> GroupCoordinator:
    assert _EP is not None, ("expert model parallel group is not initialized")
    return _EP

def get_etp_group() -> GroupCoordinator:
    assert _ETP is not None, ("expert tensor parallel group is not initialized")
    return _ETP

def init_ascend_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    backend: Optional[str] = None,
):
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    num_expert_parallel_groups: int = (world_size //
                                       (tensor_model_parallel_size // 
                                        expert_tensor_parallel_size))
    num_expert_tensor_parallel_groups: int = (tensor_model_parallel_size // 
                                              expert_tensor_parallel_size)

    global _EP
    assert _EP is None, ("expert parallel group is already initialized")
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
    assert _ETP is None, (
        "expert tensor parallel group is already initialized")
    for i in range(num_expert_tensor_parallel_groups):
        ranks = list(range(i * expert_tensor_parallel_size,
                           (i + 1) * expert_tensor_parallel_size))
        group_ranks.append(ranks)

    _ETP = init_model_parallel_group(group_ranks,
                                     get_world_group().local_rank,
                                     backend,
                                     group_name="etp")





