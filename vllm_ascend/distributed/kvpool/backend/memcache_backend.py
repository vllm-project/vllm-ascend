# Standard
from enum import Enum

import torch
from vllm.config import ParallelConfig
from vllm.utils import logger

from vllm_ascend.distributed.kvpool.backend.backend import Backend


class MmcDirect(Enum):
    COPY_L2G = 0
    COPY_G2L = 1
    COPY_G2H = 2
    COPY_H2G = 3


class MemcacheBackend(Backend):

    def __init__(self, parallel_config: ParallelConfig):
        try:
            from memcache import DistributedObjectStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install memcache by following the instructions at "
                "https://gitee.com/ascend/memfabric_hybrid "  # noqa: E501
                "to run vLLM with MemcacheConnector.") from e
        try:
            self.rank = parallel_config.rank
            self.store = DistributedObjectStore()
            res = self.store.init(self.rank)
            assert res == 0
        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error(
                "An error occurred while loading the configuration: %s", exc)
            raise

    def set_device(self):
        device = torch.device(f"npu:{self.rank}")
        torch.npu.set_device(device)

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        for ptr, size in zip(ptrs, sizes):
            ret_value = self.store.register_buffer(ptr, size)
            if ret_value != 0:
                raise RuntimeError("Memcache memory registration failed.")

    def exists(self, keys: list[str]) -> list[int]:
        return self.store.batch_is_exist(keys)

    def get(self, key: list[str], addr: list[list[int]],
            size: list[list[int]]):
        try:
            res = self.store.batch_get_into_layers(key, addr, size,
                                                   MmcDirect.COPY_G2L.value)
            for value in res:
                if value != 0:
                    logger.error(f"Failed to get key {key},res:{res}")
        except Exception as e:
            logger.error(f"Failed to get key {key}. {e}")

    def put(self, key: list[str], addr: list[list[int]],
            size: list[list[int]]):
        try:
            res = self.store.batch_put_from_layers(key, addr, size,
                                                   MmcDirect.COPY_L2G.value)
            for value in res:
                if value != 0:
                    logger.error(f"Failed to get key {key},res:{res}")
        except Exception as e:
            logger.error(f"Failed to put key {key},error:{e}")
