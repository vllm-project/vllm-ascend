# Standard
import os
import threading
import time
from enum import Enum
from typing import Any

import torch
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger
from vllm_ascend.cpu_binding import bind_thread_to_cpus
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.memcache_utils import (
    format_cpu_affinity,
    get_default_memcache_cpu_affinity,
)
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

MEMCACHE_THREAD_NAME_PREFIXES = (
    "client_pool",
    "read_pool",
    "write_pool",
    "net_pool",
    "AccWrk",
    "AccDelayClean",
    "executor",
)
MEMCACHE_THREAD_NAMES = {
    "config_store_hb",
    "grp_listen_evt",
    "ptracer_dump",
}
MEMCACHE_THREAD_BIND_RETRY_TIMES = 10
MEMCACHE_THREAD_BIND_RETRY_INTERVAL_S = 0.05
MEMCACHE_THREAD_START_WAIT_S = 0.1


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
        lazy_init: bool = False,
    ):
        self.local_rank = local_rank if local_rank is not None else get_world_group().local_rank
        self.memcache_client_cpus = memcache_client_cpus
        self._init_bm = init_bm
        self._is_a2 = get_ascend_device_type() in {AscendDeviceType.A2}
        self._lazy_init = lazy_init and not self._is_a2

        self.store: Any | None = None
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

        client_cpu_affinity = self._get_client_cpu_affinity()
        client_cpus = self._parse_cpu_affinity(client_cpu_affinity)

        if self._init_bm and self._is_a2:
            tmp_tensor = torch.zeros(1, device="npu")
            output_tensor_list = [torch.empty_like(tmp_tensor) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(output_tensor_list, tmp_tensor, group=get_world_group().device_group)

        store = DistributedObjectStore()

        original_cpus = os.sched_getaffinity(0)
        expanded_cpus = original_cpus | set(client_cpus)
        affinity_changed = False
        if client_cpus and expanded_cpus != original_cpus:
            affinity_changed = self._set_current_thread_affinity(expanded_cpus)
            if affinity_changed:
                logger.info(
                    "Temporarily expanded current thread CPUs from %s to %s "
                    "while initializing MemCache store.",
                    sorted(original_cpus),
                    sorted(expanded_cpus),
                )
        try:
            before_thread_ids = self._list_thread_ids()
            res = store.init(self.local_rank, init_bm=self._init_bm)
        except ValueError as e:
            logger.error("Configuration loading failed. error=%s. Check memcache config and environment.", e)
            raise
        except Exception as exc:
            logger.error("Store initialization failed. error=%s. Check memcache setup and dependencies.", exc)
            raise
        finally:
            if affinity_changed and self._set_current_thread_affinity(original_cpus):
                logger.info(
                    "Restored current thread CPUs to %s after initializing MemCache store.",
                    sorted(original_cpus),
                )

        assert res == 0
        time.sleep(MEMCACHE_THREAD_START_WAIT_S)
        self._bind_new_memcache_threads(before_thread_ids, client_cpus)
        return store

    @classmethod
    def create_scheduler_client(cls, parallel_config: ParallelConfig):
        # The scheduler is a single metadata client. It is initialized before
        # the world group exists and must not initialize memcache storage, so
        # keep the old device_id=0/init_bm=False behavior here.
        return cls(parallel_config, local_rank=0, init_bm=False)

    @staticmethod
    def _parse_cpu_affinity(cpu_affinity) -> list[int]:
        if cpu_affinity is None:
            return []
        if not isinstance(cpu_affinity, str):
            return [int(cpu) for cpu in cpu_affinity]

        cpus: list[int] = []
        for part in cpu_affinity.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                cpus.extend(range(int(start), int(end) + 1))
            else:
                cpus.append(int(part))
        return cpus

    def _get_client_cpu_affinity(self) -> str:
        client_cpu_affinity = format_cpu_affinity(self.memcache_client_cpus)
        if not client_cpu_affinity:
            client_cpu_affinity = get_default_memcache_cpu_affinity(self.local_rank)
        return client_cpu_affinity

    @staticmethod
    def _list_thread_ids() -> set[int]:
        task_dir = f"/proc/{os.getpid()}/task"
        try:
            return {int(task_id) for task_id in os.listdir(task_dir)}
        except OSError as err:
            logger.warning("Failed to list MemCache threads: %s", err)
            return set()

    @staticmethod
    def _get_thread_name(thread_id: int) -> str:
        comm_path = f"/proc/{os.getpid()}/task/{thread_id}/comm"
        try:
            with open(comm_path) as f:
                return f.read().strip()
        except OSError as err:
            logger.warning(
                "Failed to read MemCache thread name for tid=%d: %s",
                thread_id,
                err,
            )
            return ""

    @staticmethod
    def _is_memcache_thread(thread_name: str) -> bool:
        return (
            thread_name in MEMCACHE_THREAD_NAMES
            or thread_name.startswith(MEMCACHE_THREAD_NAME_PREFIXES)
        )

    def _bind_new_memcache_threads(
        self,
        before_thread_ids: set[int],
        client_cpus: list[int],
    ) -> None:
        if not client_cpus:
            logger.warning("Skip binding MemCache threads because MemCache client CPUs are not configured.")
            return

        handled_thread_ids: set[int] = set()
        current_pid = os.getpid()
        for _ in range(MEMCACHE_THREAD_BIND_RETRY_TIMES):
            new_thread_ids = self._list_thread_ids() - before_thread_ids - handled_thread_ids
            new_thread_ids.discard(current_pid)
            for thread_id in sorted(new_thread_ids):
                thread_name = self._get_thread_name(thread_id)
                if not self._is_memcache_thread(thread_name):
                    logger.debug(
                        "Skip non-MemCache thread name=%s tid=%d while binding MemCache threads.",
                        thread_name,
                        thread_id,
                    )
                    handled_thread_ids.add(thread_id)
                    continue
                try:
                    bind_thread_to_cpus(thread_id, client_cpus)
                    logger.info(
                        "Bound MemCache thread name=%s tid=%d to CPUs %s",
                        thread_name,
                        thread_id,
                        client_cpus,
                    )
                    handled_thread_ids.add(thread_id)
                except Exception as err:
                    logger.warning(
                        "Failed to bind MemCache thread name=%s tid=%d to CPUs %s: %s",
                        thread_name,
                        thread_id,
                        client_cpus,
                        err,
                    )
            time.sleep(MEMCACHE_THREAD_BIND_RETRY_INTERVAL_S)

    @staticmethod
    def _set_current_thread_affinity(cpus: set[int]) -> bool:
        try:
            os.sched_setaffinity(0, cpus)
            return True
        except OSError as err:
            logger.warning(
                "Failed to set current thread CPU affinity to %s: %s",
                sorted(cpus),
                err,
            )
            return False

    def init_store(self, init_bm: bool = True):
        if self.store is not None:
            return
        self._init_bm = init_bm
        self.store = self._setup_store()
        self._store_initialized = True
        self._register_buffers_if_needed()

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

    def batch_get_key_info(self, keys: list[str]):
        return self.store.batch_get_key_info(keys)

    def batch_alloc(self, keys: list[str], sizes: list[int]) -> list[int]:
        return self.store.batch_alloc(keys, sizes)

    def get(self, key: list[str], addr: list[list[int]], size: list[list[int]]):
        if self._lazy_init and not self._store_initialized:
            logger.error("get() called before store init. keys=%s. Call put() first to trigger initialization.", key)
            return
        assert self.store is not None
        try:
            res = self.store.batch_get_into_layers(key, addr, size, MmcDirect.COPY_G2L.value)
            for value in res:
                if value != 0:
                    logger.error(
                        "Failed to get key. keys=%s, result=%s. Check key existence and memory state.", key, res
                    )
            return res
        except Exception as e:
            logger.error(
                "Failed to get key. keys=%s, type=%s, error=%s. Check store state and network.",
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
            for value in res:
                if value != 0:
                    logger.error("Failed to put key. keys=%s, result=%s. Check memory and store capacity.", key, res)
                    if self._lazy_init:
                        logger.warning("First DSV4(compress) request failure is expected. This is normal behavior.")
        except Exception as e:
            logger.error(
                "Failed to put key. keys=%s, type=%s, error=%s. Check store state and memory.", key, type(e).__name__, e
            )
            if self._lazy_init:
                logger.warning("First DSV4(compress) request failure is expected. This is normal behavior.")
