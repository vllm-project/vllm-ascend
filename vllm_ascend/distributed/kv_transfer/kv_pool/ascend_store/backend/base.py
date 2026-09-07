from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.platforms import current_platform
from vllm.platforms.interface import set_assigned_physical_gpu_ids

# QoS range supported by the pooled KV store backends. A larger value
# means a higher transfer priority.
QOS_VALUE_MIN = 0
QOS_VALUE_MAX = 4


def get_scheduler_device_id(parallel_config: ParallelConfig) -> int:
    """Use the colocated worker's device, or the first device in this DP shard."""
    try:
        get_world_group()
    except AssertionError:
        # With a separate executor, the scheduler has no world group and may
        # not have initialized NPU yet. Resolve the mapping before querying
        # current_device(), which would otherwise initialize the default NPU.
        assigned_ids = parallel_config.assigned_physical_gpu_ids
        local_rank = 0
        if assigned_ids is not None:
            set_assigned_physical_gpu_ids(assigned_ids)
        elif (
            parallel_config.distributed_executor_backend not in ("ray", "external_launcher")
            and parallel_config.data_parallel_backend != "ray"
            and parallel_config.nnodes_within_dp == 1
        ):
            # Match NPUWorker's device selection when no explicit mapping is
            # supplied. Multi-node and Ray executors manage their own ranks.
            dp_local_rank = parallel_config.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = parallel_config.data_parallel_index
            local_rank = dp_local_rank * parallel_config.tensor_parallel_size * parallel_config.pipeline_parallel_size
        return current_platform.logical_device_id_to_visible_device_id(local_rank)

    return torch.npu.current_device()


class Backend(ABC):
    store: Any | None = None
    # Whether the connector must filter existing keys before calling put().
    requires_exists_before_put: bool = True

    @abstractmethod
    def __init__(self, parallel_config: ParallelConfig, lazy_init: bool = False):
        pass

    @classmethod
    def create_scheduler_client(cls, parallel_config: ParallelConfig):
        return cls(parallel_config)

    @abstractmethod
    def set_device(self):
        pass

    @abstractmethod
    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        pass

    @abstractmethod
    def exists(self, keys: list[str]) -> list[int]:
        pass

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return self.exists(keys)

    def batch_get_key_info(self, keys: list[str]):
        raise NotImplementedError(f"{type(self).__name__} does not support batch_get_key_info")

    def batch_alloc(self, keys: list[str], sizes: list[int]) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_alloc")

    def batch_add_lease(self, keys: list[str], lease_ttl_ms: int = 0) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_add_lease")

    def batch_remove_lease(self, keys: list[str]) -> int:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_remove_lease")

    def batch_write_finish(self, keys: list[str], results: list[int]) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_write_finish")

    @abstractmethod
    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass

    @abstractmethod
    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass
