from typing import Optional

import torch
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import (GroupCoordinator, get_world_group,
                                             init_model_parallel_group)

# Currently, mc2 op need their own group coordinator.
_MC2: Optional[GroupCoordinator] = None


def get_mc2_group() -> GroupCoordinator:
    assert _MC2 is not None, ("mc2 group is not initialized")
    return _MC2


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

    init_ascend_mla_sp_model_parallel()

# vllm-ascend will maintain its own MLA SP world GroupCoordinator and o_proj sharding GroupCoordinator for
# customize parallel solution
_MLA_SP_WORLD: Optional[GroupCoordinator] = None
_O_SHARD: Optional[GroupCoordinator] = None

def get_mla_sp_world_group() -> GroupCoordinator:
    assert _MLA_SP_WORLD is not None, ("MLA sequence parallel world group is not initialized")
    return _MLA_SP_WORLD

def get_o_shard_group() -> GroupCoordinator:
    assert _O_SHARD is not None, ("o_proj sharding group is not initialized")
    return _O_SHARD

def init_ascend_mla_sp_model_parallel():
    from vllm_ascend.ascend_config import get_ascend_config
    ascend_config = get_ascend_config()
    world_size = torch.distributed.get_world_size()
    backend = torch.distributed.get_backend(get_world_group().device_group)

    if ascend_config.enable_mla_sp:
        assert ascend_config.enable_o_shard, "MLA SP must be enabled with o_proj sharding"
        global _MLA_SP_WORLD
        group_ranks = [list(range(torch.distributed.get_world_size()))]
        _MLA_SP_WORLD = init_model_parallel_group(group_ranks,
                                                  get_world_group().local_rank,
                                                  backend,
                                                  group_name="mla_sp_world")

    if ascend_config.enable_o_shard:
        o_shard_parallel_size = ascend_config.o_shard_parallel_size
        assert o_shard_parallel_size >= 2, "o_shard_parallel_size must be >= 2"
        assert world_size % o_shard_parallel_size == 0, "o_shard_parallel_size must be a divisor of world_size"
        global _O_SHARD
        all_ranks = torch.arange(world_size)
        group_ranks = all_ranks.view(-1, o_shard_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        _O_SHARD = init_model_parallel_group(group_ranks,
                                             get_world_group().local_rank,
                                             backend,
                                             group_name="o_shard")

def destroy_ascend_model_parallel():
    global _MC2
    if _MC2:
        _MC2.destroy()
    _MC2 = None

    global _MLA_SP_WORLD
    if _MLA_SP_WORLD:
        _MLA_SP_WORLD.destroy()
    _MLA_SP_WORLD = None

    global _O_SHARD
    if _O_SHARD:
        _O_SHARD.destroy()
    _O_SHARD = None
