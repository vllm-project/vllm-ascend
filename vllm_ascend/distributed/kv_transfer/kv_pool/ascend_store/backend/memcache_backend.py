# Standard
import threading
from enum import Enum
from typing import Any

import torch
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


class MmcDirect(Enum):
    COPY_L2G = 0
    COPY_G2L = 1
    COPY_G2H = 2
    COPY_H2G = 3


class MemcacheBackend(Backend):
    def __init__(self, parallel_config: ParallelConfig, lazy_init: bool = False):
        self.local_rank = get_world_group().local_rank
        self.store: Any | None = None
        self._is_a2 = get_ascend_device_type() in {AscendDeviceType.A2}
        self._lazy_init = lazy_init and not self._is_a2
        self._store_initialized = False
        self._store_init_lock = threading.Lock()
        self._registered_buffers: tuple[list[int], list[int]] | None = None
        self._buffers_registered = False

        if not self._lazy_init:
            self.store = self._setup_store()
            self._store_initialized = True

    def _ensure_initialized(self):
        if self._store_initialized:
            return

        with self._store_init_lock:
            if self._store_initialized:
                return

            logger.info("Initializing Memcache store. local_rank=%d", self.local_rank)
            self.store = self._setup_store()
            self._store_initialized = True
            self._register_buffers_if_needed()

    def _setup_store(self):
        try:
            from memcache_hybrid import DistributedObjectStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install memcache by following the instructions at "
                "https://gitee.com/ascend/memfabric_hybrid "  # noqa: E501
                "to run vLLM with MemcacheConnector."
            ) from e
        try:
            if self._is_a2:
                tmp_tensor = torch.zeros(1, device="npu")
                output_tensor_list = [torch.empty_like(tmp_tensor) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(output_tensor_list, tmp_tensor, group=get_world_group().device_group)
            store = DistributedObjectStore()
            res = store.init(self.local_rank)
            assert res == 0
            return store
        except ValueError as e:
            logger.error("Configuration loading failed. error=%s. Check memcache config and environment.", e)
            raise
        except Exception as exc:
            logger.error("Store initialization failed. error=%s. Check memcache setup and dependencies.", exc)
            raise

    def set_device(self):
        device = torch.device(f"npu:{self.local_rank}")
        torch.npu.set_device(device)

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        self._registered_buffers = (list(ptrs), list(sizes))
        self._register_buffers_if_needed()

    def _register_buffers_if_needed(self):
        if not self._is_a2:
            return
        if self._registered_buffers is None or self._buffers_registered:
            return
        if not self._store_initialized:
            return
        assert self.store is not None
        ptrs, sizes = self._registered_buffers
        for ptr, size in zip(ptrs, sizes):
            self.store.register_buffer(ptr, size)
        self._buffers_registered = True

    def exists(self, keys: list[str]) -> list[int]:
        if self._lazy_init and not self._store_initialized:
            logger.debug(
                "MemcacheBackend.exists called before store initialization; treating %d keys as missing.",
                len(keys),
            )
            return [0] * len(keys)
        assert self.store is not None
        return self.store.batch_is_exist(keys)

    def get(self, key: list[str], addr: list[list[int]], size: list[list[int]]):
        if self._lazy_init and not self._store_initialized:
            logger.error(
                "Failed to get %d keys out of %d. Store is not initialized; "
                "call put() first to trigger initialization.",
                len(key),
                len(key),
            )
            logger.debug("Failed to get key details. keys=%s", key)
            return
        assert self.store is not None
        try:
            res = self.store.batch_get_into_layers(key, addr, size, MmcDirect.COPY_G2L.value)
            failed_codes = [int(value) for value in res if value != 0]
            failed_count = len(failed_codes)
            if failed_count:
                error_codes = sorted(set(failed_codes))
                logger.error(
                    "Failed to get %d keys out of %d. error_codes=%s. Check key existence and memory state.",
                    failed_count,
                    len(key),
                    error_codes,
                )
                logger.debug("Failed to get key details. keys=%s, result=%s", key, res)
            return res
        except Exception as e:
            logger.error(
                "Failed to get %d keys out of %d. Check store state and network.",
                len(key),
                len(key),
            )
            logger.debug(
                "Failed to get key details. keys=%s, type=%s, error=%s",
                key,
                type(e).__name__,
                e,
            )
            return None

    def put(self, key: list[str], addr: list[list[int]], size: list[list[int]]):
        try:
            self._ensure_initialized()
            assert self.store is not None
            res = self.store.batch_put_from_layers(key, addr, size, MmcDirect.COPY_L2G.value)
            failed_codes = [int(value) for value in res if value != 0]
            failed_count = len(failed_codes)
            if failed_count:
                error_codes = sorted(set(failed_codes))
                logger.error(
                    "Failed to put %d keys out of %d. error_codes=%s. Check memory and store capacity.",
                    failed_count,
                    len(key),
                    error_codes,
                )
                logger.debug("Failed to put key details. keys=%s, result=%s", key, res)
                if self._lazy_init:
                    logger.warning("First DSV4(compress) request failure is expected. This is normal behavior.")
        except Exception as e:
            logger.error(
                "Failed to put %d keys out of %d. Check store state and memory.",
                len(key),
                len(key),
            )
            logger.debug(
                "Failed to put key details. keys=%s, type=%s, error=%s",
                key,
                type(e).__name__,
                e,
            )
            if self._lazy_init:
                logger.warning("First DSV4(compress) request failure is expected. This is normal behavior.")
