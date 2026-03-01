from copy import deepcopy

from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.v1.kv_cache_interface import KVCacheConfig


# Make a fake device group;
# vllm code depends on calling device_group.size() as a method
class FakeDeviceGroup:
    def __init__(
        self,
        size: int,
    ):
        self._size = size

    def size(self):
        """Return the size"""
        return self._size


# Adapted from vllm/distributed/parallel_state.py (GroupCoordinator)
class FakeGroupCoordinator:
    def __init__(
        self,
        rank: int,
        size: int,
        group_ranks: list[list[int]],
        backend="hccl",
        group_name: str | None = None,
        fake_group=True,
        force_world_size=0,
    ):
        import torch

        self.ranks = group_ranks
        self.rank = torch.distributed.get_rank()
        self.device_group = None
        self.group_name = group_name
        assert self.rank == rank

        for ranks in group_ranks:
            if fake_group:
                device_group = FakeDeviceGroup(size=size)
            else:
                print(f"Creating group {self.group_name} with ranks {group_ranks}")
                device_group = torch.distributed.new_group(ranks, backend=backend)

            # a group with `gloo` backend, to allow direct coordination between processes through the CPU.
            # cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            cpu_group = None

            if self.rank in ranks:
                self.ranks = ranks
                if force_world_size > 0:
                    self.world_size = force_world_size
                else:
                    self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group
                self.cpu_group = cpu_group

    @property
    def first_rank(self):
        """Return the global rank of the first process in the group"""
        return self.ranks[0]

    @property
    def last_rank(self):
        """Return the global rank of the last process in the group"""
        return self.ranks[-1]

    @property
    def is_first_rank(self):
        """Return whether the caller is the first process in the group"""
        return self.rank == self.first_rank

    @property
    def is_last_rank(self):
        """Return whether the caller is the last process in the group"""
        return self.rank == self.last_rank

    @property
    def next_rank(self):
        """Return the global rank of the process that follows the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group + 1) % world_size]

    @property
    def prev_rank(self):
        """Return the global rank of the process that precedes the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group - 1) % world_size]


# Adapted from vllm/distributed/parallel_state.py, vllm_ascend/distributed/parallel_state.py
def initialize_fake_model_parallel(
    rank: int,
    world_size=1,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
    backend="hccl",
) -> None:
    import torch
    import vllm.distributed.parallel_state as ps

    import vllm_ascend.distributed.parallel_state as ascend_ps

    # Intentionally forced DP=1 as model is always loaded with DP=1
    data_parallel_size = 1
    model_world_size: int = tensor_model_parallel_size * data_parallel_size
    all_ranks = torch.arange(model_world_size).reshape(
        -1, data_parallel_size, pipeline_model_parallel_size, tensor_model_parallel_size
    )

    group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    ps._TP = FakeGroupCoordinator(
        rank=rank,
        size=tensor_model_parallel_size,
        group_ranks=group_ranks,
        backend=backend,
        group_name="tp",
        fake_group=True,
    )

    group_ranks = all_ranks.transpose(2, 3).reshape(-1, pipeline_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    ps._PP = FakeGroupCoordinator(
        rank=rank,
        size=pipeline_model_parallel_size,
        group_ranks=group_ranks,
        backend=backend,
        group_name="pp",
        fake_group=True,
    )

    group_ranks = all_ranks.transpose(1, 3).reshape(-1, data_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    ps._DP = FakeGroupCoordinator(
        rank=rank, size=data_parallel_size, group_ranks=group_ranks, backend=backend, group_name="dp", fake_group=True
    )

    group_ranks = all_ranks.view(-1, expert_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    group_ranks = [group_ranks[0]]
    ps._EP = FakeGroupCoordinator(
        rank=rank, size=expert_parallel_size, group_ranks=group_ranks, backend=backend, group_name="ep", fake_group=True
    )

    # PCP needs all ranks to have a group. Hence forcing it on all ranks.
    pcp_all_ranks = torch.arange(world_size).reshape(
        -1, data_parallel_size, pipeline_model_parallel_size, tensor_model_parallel_size
    )
    group_ranks = pcp_all_ranks.transpose(2, 3).reshape(-1, pipeline_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    ps._PCP = FakeGroupCoordinator(
        rank=rank,
        size=prefill_context_parallel_size,
        group_ranks=group_ranks,
        backend=backend,
        group_name="pcp",
        fake_group=True,
        force_world_size=1,
    )

    group_ranks = all_ranks.transpose(2, 3).reshape(-1, expert_tensor_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    ps._ETP = FakeGroupCoordinator(
        rank=rank,
        size=expert_tensor_parallel_size,
        group_ranks=group_ranks,
        backend=backend,
        group_name="etp",
        fake_group=True,
    )

    all_ranks = torch.arange(model_world_size).reshape(-1, data_parallel_size * tensor_model_parallel_size)
    group_ranks = all_ranks.unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    ascend_ps._MC2 = FakeGroupCoordinator(
        rank=rank, size=len(group_ranks), group_ranks=group_ranks, backend=backend, group_name="mc2", fake_group=True
    )


# # Adapted from vllm_ascend/worker/model_runner_v1.py
def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
    """
    Initialize KV cache based on `kv_cache_config`.
    Args:
        kv_cache_config: Configuration for the KV cache, including the KV
        cache size of each layer
    """
    kv_cache_config = deepcopy(kv_cache_config)
    self.kv_cache_config = kv_cache_config

    kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)

    if has_kv_transfer_group():
        get_kv_transfer_group().register_kv_caches(kv_caches)

    return kv_caches


# Takes an iterable, tries to mimic as a dict without converting to dict
class IterableDict:
    def __init__(self, iterable_factory):
        self._iterable_factory = iterable_factory

    def __getitem__(self, key):
        for name, param in self._iterable_factory():
            if name == key:
                return param
        raise KeyError(f"Parameter '{key}' not found")

    def __iter__(self):
        return (name for name, _ in self._iterable_factory())

    def items(self):
        return self._iterable_factory()

    def keys(self):
        return (name for name, _ in self._iterable_factory())

    def values(self):
        return (param for _, param in self._iterable_factory())

    def __contains__(self, key):
        return any(name == key for name, _ in self._iterable_factory())

    def __len__(self):
        return sum(1 for _ in self._iterable_factory())
