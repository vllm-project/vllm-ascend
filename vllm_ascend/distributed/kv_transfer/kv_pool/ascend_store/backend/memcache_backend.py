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

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import (
    QOS_VALUE_MAX,
    QOS_VALUE_MIN,
    Backend,
)


def _is_device_sdma() -> bool:
    config_path = os.getenv("MMC_LOCAL_CONFIG_PATH")
    if not config_path:
        raise ValueError("The environment variable 'MMC_LOCAL_CONFIG_PATH' is not set.")
    with open(config_path, encoding="utf-8") as config_file:
        for line in config_file:
            line = line.strip()
            if not line or line.startswith(("#", ";")):
                continue
            key, separator, value = line.partition("=")
            if separator and key.strip() == "ock.mmc.local_service.protocol":
                return value.strip() == "device_sdma"
    return False


MEMCACHE_THREAD_START_WAIT_S = 0.1


def _validate_device_ub_qos() -> None:
    """Validate MF_DEVICE_UB_QOS used by the MemCache store transfers.

    Only integers in [QOS_VALUE_MIN, QOS_VALUE_MAX] are supported; MemCache
    silently falls back to its default QoS on invalid values, so fail fast
    with a clear error instead.
    """
    qos_str = os.getenv("MF_DEVICE_UB_QOS")
    if qos_str is None or not qos_str.strip():
        return
    try:
        qos = int(qos_str)
    except ValueError as e:
        raise ValueError(
            f"Invalid MF_DEVICE_UB_QOS value {qos_str!r}: QoS must be an integer in [{QOS_VALUE_MIN}, {QOS_VALUE_MAX}]."
        ) from e
    if not (QOS_VALUE_MIN <= qos <= QOS_VALUE_MAX):
        raise ValueError(
            f"Invalid MF_DEVICE_UB_QOS value {qos}: QoS must be an integer in [{QOS_VALUE_MIN}, {QOS_VALUE_MAX}]."
        )


class MmcDirect(Enum):
    COPY_L2G = 0
    COPY_G2L = 1
    COPY_G2H = 2
    COPY_H2G = 3


# =========================================================================
# Layerwise transfer protocol
# =========================================================================
# The generic layers (worker / scheduler / layout) resolve these functions
# through backend/__init__.py:get_layerwise_protocol -- a module-convention
# lookup, they never import this module by name. The key strings are wire
# formats shared with deployed clusters: a single character of drift turns
# hits into misses after an upgrade.
# tests/ut/distributed/ascend_store/test_backend.py locks the key formats
# with snapshot assertions.


def extract_layout_config(extra_config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the connector's extra config when it opts into the layerwise
    transfer, None otherwise.

    Called by the generic layout layer through the backend registry; the
    protocol itself owns the opt-in check so the layout layer never spells
    out the gate.
    """
    if extra_config.get("use_layerwise", False):
        return extra_config
    return None


def make_full_key(
    model_name: str,
    group_id: int,
    block_hash_hex: str,
    head_or_tp_rank: int,
    num_groups: int,
) -> str:
    """Full-block key for the layerwise transfer.

    Single-group models use the PR #11585 format (model@hash@rank) for
    backward compatibility. Multi-group models include group_id
    (model@group_id@hash@rank) to distinguish groups.
    """
    if num_groups > 1:
        return f"{model_name}@{group_id}@{block_hash_hex}@{head_or_tp_rank}"
    else:
        return f"{model_name}@{block_hash_hex}@{head_or_tp_rank}"


def make_partial_key(
    model_name: str,
    req_id: str,
    group_id: int,
    block_index: int,
    end_token: int,
    head_or_tp_rank: int,
) -> str:
    return f"{model_name}@partial@{req_id}@{group_id}@{block_index}@{end_token}@{head_or_tp_rank}"


def make_hit_check_keys(
    model_name: str,
    group_id: int,
    block_hash_hex: str,
    num_ranks: int,
    num_groups: int,
) -> list[str]:
    """All-rank keys for scheduler-side hit check.

    Returns one key per head_or_tp_rank (ranks in the same put_step
    group share one key for MLA).
    """
    if num_groups > 1:
        return [f"{model_name}@{group_id}@{block_hash_hex}@{h}" for h in range(num_ranks)]
    else:
        return [f"{model_name}@{block_hash_hex}@{h}" for h in range(num_ranks)]


class MemcacheBackend(Backend):
    def __init__(
        self,
        parallel_config: ParallelConfig,
        local_rank: int | None = None,
        init_bm: bool = True,
        lazy_init: bool = False,
    ):
        _validate_device_ub_qos()
        self.local_rank = local_rank if local_rank is not None else get_world_group().local_rank
        self._init_bm = init_bm
        self._lazy_init = lazy_init and _is_device_sdma()

        self.store: Any | None = None
        self._store_initialized = False
        self._store_init_lock = threading.Lock()
        self._pending_buffers: tuple[list[int], list[int]] | None = None

        if not self._lazy_init:
            self.store = self._setup_store()
            self._store_initialized = True

    def ensure_initialized(self):
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

        store = DistributedObjectStore()

        try:
            res = store.init(self.local_rank, init_bm=self._init_bm)
        except ValueError as e:
            logger.error("Configuration loading failed. error=%s. Check memcache config and environment.", e)
            raise
        except Exception as exc:
            logger.error("Store initialization failed. error=%s. Check memcache setup and dependencies.", exc)
            raise

        assert res == 0
        time.sleep(MEMCACHE_THREAD_START_WAIT_S)
        return store

    @classmethod
    def create_scheduler_client(cls, parallel_config: ParallelConfig):
        # The scheduler is a single metadata client. It is initialized before
        # the world group exists and must not initialize memcache storage, so
        # keep the old device_id=0/init_bm=False behavior here.
        return cls(parallel_config, local_rank=0, init_bm=False)

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
        self._pending_buffers = (list(ptrs), list(sizes))
        self._register_buffers_if_needed()

    def _register_buffers_if_needed(self):
        if self._pending_buffers is None or not self._store_initialized:
            return
        assert self.store is not None
        ptrs, sizes = self._pending_buffers
        for ptr, size in zip(ptrs, sizes):
            self.store.register_buffer(ptr, size)
        self._pending_buffers = None

    def exists(self, keys: list[str]) -> list[int]:
        if self._lazy_init and not self._store_initialized:
            logger.debug(
                "MemcacheBackend.exists called before store initialization; treating %d keys as missing.",
                len(keys),
            )
            return [0] * len(keys)
        assert self.store is not None
        return self.store.batch_is_exist(keys)

    def batch_get_key_info(self, keys: list[str]) -> list[Any]:
        if self._lazy_init and not self._store_initialized:
            logger.debug(
                "MemcacheBackend.batch_get_key_info called before store initialization; "
                "returning empty list for %d keys.",
                len(keys),
            )
            return []
        assert self.store is not None
        return self.store.batch_get_key_info(keys)

    def batch_alloc(self, keys: list[str], sizes: list[int]) -> list[int]:
        self.ensure_initialized()
        assert self.store is not None
        return self.store.batch_alloc(keys, sizes)

    def batch_add_lease(self, keys: list[str], lease_ttl_ms: int = 0) -> list[int]:
        assert self.store is not None
        return self.store.batch_add_lease(keys, lease_ttl_ms)

    def batch_remove_lease(self, keys: list[str]) -> int:
        assert self.store is not None
        return self.store.batch_remove_lease(keys)

    def batch_write_finish(self, keys: list[str], results: list[int]) -> list[int]:
        assert self.store is not None
        finish = getattr(self.store, "batch_write_finish", None)
        if finish is None:
            # Older MemCache releases publish writes directly in batch_copy.
            return [0] * len(keys)
        return finish(keys, results)

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
                "Failed to get %d keys out of %d. type=%s, error=%s. Check store state and network.",
                len(key),
                len(key),
                type(e).__name__,
                e,
            )
            logger.debug("Failed to get key details. keys=%s", key)
            return None

    def put(self, key: list[str], addr: list[list[int]], size: list[list[int]]):
        self.ensure_initialized()
        assert self.store is not None
        try:
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
                "Failed to put %d keys out of %d. type=%s, error=%s. Check store state and memory.",
                len(key),
                len(key),
                type(e).__name__,
                e,
            )
            logger.debug("Failed to put key details. keys=%s", key)
            if self._lazy_init:
                logger.warning("First DSV4(compress) request failure is expected. This is normal behavior.")
