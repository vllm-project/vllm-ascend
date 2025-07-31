from typing import Optional

import torch
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import (GroupCoordinator, get_world_group,
                                             init_model_parallel_group)

import vllm_ascend.envs as envs_ascend

# Currently, mc2 op need their own group coordinator.
_MC2: Optional[GroupCoordinator] = None
_MLP_TP: Optional[GroupCoordinator] = None


def get_mc2_group() -> GroupCoordinator:
    assert _MC2 is not None, ("mc2 group is not initialized")
    return _MC2


def get_mlp_tp_group() -> GroupCoordinator:
    assert _MLP_TP is not None, ("mlp group is not initialized")
    return _MLP_TP


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
    if envs_ascend.VLLM_ASCEND_ENABLE_MLP_OPTIMIZE:
        global _MLP_TP
        assert _MLP_TP is None, (
            "mlp tensor model parallel group is already initialized")

        mlp_tp = parallel_config.data_parallel_size

        all_ranks_mlp_head = torch.arange(world_size).reshape(
            -1, mlp_tp, parallel_config.pipeline_parallel_size, 1)  # noqa
        group_ranks = all_ranks_mlp_head.view(-1, mlp_tp).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]

        # message queue broadcaster is only used in tensor model parallel group
        _MLP_TP = init_model_parallel_group(group_ranks,
                                            get_world_group().local_rank,
                                            backend,
                                            group_name="mlp_tp")


def get_mlp_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_mlp_tp_group().world_size


def get_mlp_tensor_model_parallel_rank():
    """Return world size for the tensor model parallel group."""
    return get_mlp_tp_group().rank_in_group


def destroy_ascend_model_parallel():
    global _MC2
    if _MC2:
        _MC2.destroy()
    _MC2 = None

    global _MLP_TP
    if _MLP_TP:
        _MLP_TP.destroy()
    _MLP_TP = None
