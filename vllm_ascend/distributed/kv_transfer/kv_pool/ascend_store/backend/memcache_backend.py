from enum import Enum

import torch

from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.memcache_utils import (
    set_memcache_client_cpu_affinity,
)
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


class MmcDirect(Enum):
    COPY_L2G = 0
    COPY_G2L = 1
    COPY_G2H = 2
    COPY_H2G = 3


class MemcacheBackend(Backend):
    def __init__(
        self,
        parallel_config: ParallelConfig,
        memcache_client_cpus=None,
        local_rank: int | None = None,
        init_bm: bool = True,
        defer_init: bool = False,
    ):
        try:
            from memcache_hybrid import DistributedObjectStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install memcache by following the instructions at "
                "https://gitee.com/ascend/memfabric_hybrid "  # noqa: E501
                "to run vLLM with MemcacheConnector."
            ) from e
        try:
            self.local_rank = local_rank if local_rank is not None else get_world_group().local_rank
            self.memcache_client_cpus = memcache_client_cpus
            self._distributed_object_store_cls = DistributedObjectStore
            self.store = None
            if not defer_init:
                self.init_store(init_bm=init_bm)
        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

    @classmethod
    def create_scheduler_client(cls, parallel_config: ParallelConfig):
        # The scheduler is a single metadata client. It is initialized before
        # the world group exists and must not initialize memcache storage, so
        # keep the old device_id=0/init_bm=False behavior here.
        return cls(parallel_config, local_rank=0, init_bm=False)

    def init_store(self, init_bm: bool = True):
        if self.store is not None:
            return
        soc_version = get_ascend_device_type()
        if init_bm and soc_version in {AscendDeviceType.A2}:
            tmp_tensor = torch.zeros(1, device="npu")
            output_tensor_list = [torch.empty_like(tmp_tensor) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(output_tensor_list, tmp_tensor, group=get_world_group().device_group)
        self.store = self._distributed_object_store_cls()
        set_memcache_client_cpu_affinity(
            self.store, self.local_rank, self.memcache_client_cpus)
        res = self.store.init(self.local_rank, init_bm=init_bm)
        assert res == 0

    def set_device(self):
        device = torch.device(f"npu:{self.local_rank}")
        torch.npu.set_device(device)

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        soc_version = get_ascend_device_type()
        if soc_version in {AscendDeviceType.A2}:
            for ptr, size in zip(ptrs, sizes):
                self.store.register_buffer(ptr, size)
        else:
            pass

    def exists(self, keys: list[str]) -> list[int]:
        return self.store.batch_is_exist(keys)

    def batch_get_key_info(self, keys: list[str]):
        return self.store.batch_get_key_info(keys)

    def batch_alloc(self, keys: list[str], sizes: list[int]) -> list[int]:
        return self.store.batch_alloc(keys, sizes)

    def get(self, key: list[str], addr: list[list[int]], size: list[list[int]]):
        try:
            res = self.store.batch_get_into_layers(key, addr, size, MmcDirect.COPY_G2L.value)
            for value in res:
                if value != 0:
                    logger.error(f"Failed to get key {key},res:{res}")
        except Exception as e:
            logger.error(f"Failed to get key {key}. {e}")

    def put(self, key: list[str], addr: list[list[int]], size: list[list[int]]):
        try:
            res = self.store.batch_put_from_layers(key, addr, size, MmcDirect.COPY_L2G.value)
            for value in res:
                if value != 0:
                    logger.error(f"Failed to get key {key},res:{res}")
        except Exception as e:
            logger.error(f"Failed to put key {key},error:{e}")
