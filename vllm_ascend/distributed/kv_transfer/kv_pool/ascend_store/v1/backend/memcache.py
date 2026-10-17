"""Memcache native client adapter owned by AscendStore v1."""

from __future__ import annotations

import threading
import time
from numbers import Integral
from typing import Any

import torch
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_dp_group
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MEMCACHE_THREAD_START_WAIT_S,
    MmcDirect,
    _inject_device_ub_qos,
    _is_device_sdma,
    _validate_device_ub_qos,
)

from .port import GVARegion
from .registration import BufferRegistration, rollback_buffer_registration
from .results import presence_results, result_code, result_codes

READABLE_GVA_QUERY = 1


class MemcacheBackend:
    """Translate v1 operations directly to one owned Memcache client."""

    requires_exists_before_put = True

    def __init__(
        self,
        parallel_config: ParallelConfig,
        device_id: int | None = None,
        init_bm: bool = True,
        lazy_init: bool = False,
        extra_config: dict[str, Any] | None = None,
        dp_init_barrier: bool = True,
    ) -> None:
        if extra_config is not None:
            dp_init_barrier = extra_config.get("memcache_dp_init_barrier", dp_init_barrier)
        if not isinstance(dp_init_barrier, bool):
            raise ValueError("memcache_dp_init_barrier in kv_connector_extra_config must be a boolean")
        _inject_device_ub_qos(extra_config)
        _validate_device_ub_qos()
        self.device_id = torch.npu.current_device() if device_id is None else device_id
        self._init_bm = init_bm
        self._lazy_init = lazy_init and _is_device_sdma()
        self._dp_init_barrier = dp_init_barrier and parallel_config.data_parallel_size > 1 and not self._lazy_init

        self._store: Any | None = None
        self._initialized = False
        self._initialization_lock = threading.Lock()
        self._pending_buffers: tuple[list[int], list[int]] | None = None
        self._pending_registration: BufferRegistration | None = None
        self._registration_error: BaseException | None = None
        self._closed = False

        if not self._lazy_init:
            self._store = self._setup_store()
            self._initialized = True

    def set_device(self) -> None:
        torch.npu.set_device(self.device_id)

    def register_buffer(self, addresses: list[int], sizes: list[int]) -> BufferRegistration:
        self._require_open()
        if len(addresses) != len(sizes):
            raise ValueError(f"addresses and sizes must have the same length: {len(addresses)} != {len(sizes)}")
        if self._pending_registration is not None:
            raise RuntimeError("Memcache v1 Backend already has a pending buffer registration")

        registration: BufferRegistration

        def cancel_pending() -> None:
            self._cancel_pending_registration(registration)

        registration = BufferRegistration(self._unregister_native_region, before_release=cancel_pending)
        self._pending_buffers = (list(addresses), list(sizes))
        self._pending_registration = registration
        self._register_buffers_if_ready()
        return registration

    def close(self) -> None:
        if self._closed:
            return
        if self._pending_registration is not None:
            self._pending_registration.close()
        if self._store is not None:
            self._close_store(self._store)
        self._store = None
        self._initialized = False
        self._closed = True

    def exists(self, keys: list[str]) -> tuple[bool, ...]:
        if self._lazy_init and not self._initialized:
            return (False,) * len(keys)
        assert self._store is not None
        return presence_results("Memcache batch_is_exist", keys, self._store.batch_is_exist(keys))

    def load(
        self,
        keys: list[str],
        addresses: list[list[int]],
        sizes: list[list[int]],
    ) -> tuple[int, ...]:
        if self._lazy_init and not self._initialized:
            raise RuntimeError("Memcache store is not initialized; Store must initialize it before Load")
        assert self._store is not None
        return result_codes(
            "Memcache batch_get_into_layers",
            keys,
            self._store.batch_get_into_layers(keys, addresses, sizes, MmcDirect.COPY_G2L.value),
        )

    def store(
        self,
        keys: list[str],
        addresses: list[list[int]],
        sizes: list[list[int]],
    ) -> tuple[int, ...]:
        self._ensure_initialized()
        assert self._store is not None
        return result_codes(
            "Memcache batch_put_from_layers",
            keys,
            self._store.batch_put_from_layers(keys, addresses, sizes, MmcDirect.COPY_L2G.value),
        )

    def validate_gva_support(self) -> None:
        self._ensure_initialized()
        assert self._store is not None
        required = (
            "batch_get_key_info",
            "batch_add_lease",
            "batch_remove_lease",
            "batch_alloc",
            "batch_copy",
            "batch_write_finish",
        )
        for method_name in required:
            if not callable(getattr(self._store, method_name, None)):
                raise RuntimeError(f"Memcache GVA requires native {method_name}; upgrade the Backend library")

    def query_gva_regions(self, keys: list[str]) -> tuple[GVARegion | None, ...]:
        if not keys:
            return ()
        self._ensure_initialized()
        assert self._store is not None
        infos = tuple(self._store.batch_get_key_info(keys, READABLE_GVA_QUERY))
        if len(infos) != len(keys):
            raise RuntimeError(f"batch_get_key_info returned {len(infos)} results for {len(keys)} keys")
        regions: list[GVARegion | None] = []
        for info in infos:
            try:
                size = info.size()
                addresses = tuple(info.gva_list())
            except (AttributeError, TypeError, ValueError) as error:
                raise RuntimeError("batch_get_key_info returned invalid key metadata") from error
            if isinstance(size, bool) or not isinstance(size, Integral):
                raise RuntimeError("batch_get_key_info returned a non-integer object size")
            if size <= 0 or not addresses:
                regions.append(None)
                continue
            if len(addresses) != 1 or isinstance(addresses[0], bool) or not isinstance(addresses[0], Integral):
                raise RuntimeError("GVA sessions require exactly one integer address per object")
            address = int(addresses[0])
            regions.append(GVARegion(address, int(size)) if address > 0 else None)
        return tuple(regions)

    def add_gva_leases(self, keys: list[str]) -> tuple[int, ...]:
        self._ensure_initialized()
        assert self._store is not None
        return result_codes("Memcache batch_add_lease", keys, self._store.batch_add_lease(keys))

    def remove_gva_leases(self, keys: list[str]) -> None:
        self._ensure_initialized()
        assert self._store is not None
        native_result = result_code("Memcache batch_remove_lease", self._store.batch_remove_lease(keys))
        if native_result != 0:
            raise RuntimeError(f"Memcache batch_remove_lease failed with result {native_result}")

    def allocate_gva(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        self._ensure_initialized()
        assert self._store is not None
        return result_codes("Memcache batch_alloc", keys, self._store.batch_alloc(keys, object_sizes))

    def copy_from_gva(
        self,
        remote_addresses: list[int],
        local_addresses: list[int],
        sizes: list[int],
    ) -> int:
        return self._copy_gva(remote_addresses, local_addresses, sizes, MmcDirect.COPY_G2L.value)

    def copy_to_gva(
        self,
        remote_addresses: list[int],
        local_addresses: list[int],
        sizes: list[int],
    ) -> int:
        return self._copy_gva(remote_addresses, local_addresses, sizes, MmcDirect.COPY_L2G.value)

    def publish_gva(self, keys: list[str]) -> tuple[int, ...]:
        self._ensure_initialized()
        assert self._store is not None
        return result_codes(
            "Memcache batch_write_finish",
            keys,
            self._store.batch_write_finish(keys, [0] * len(keys)),
        )

    def _ensure_initialized(self) -> None:
        self._require_open()
        if self._registration_error is not None:
            raise RuntimeError("Memcache buffer registration previously failed") from self._registration_error
        if self._initialized:
            return
        with self._initialization_lock:
            if self._initialized:
                return
            logger.info("Initializing Memcache v1 store. device_id=%d", self.device_id)
            self._store = self._setup_store()
            self._initialized = True
            self._register_buffers_if_ready()

    def _setup_store(self) -> Any:
        try:
            from memcache_hybrid import DistributedObjectStore  # type: ignore
        except ImportError as error:
            raise ImportError(
                "Please install memcache by following https://gitee.com/ascend/memfabric_hybrid"
            ) from error

        if self._init_bm:
            self.set_device()
        store = DistributedObjectStore()
        try:
            native_result = store.init(self.device_id, init_bm=self._init_bm)
            if result_code("Memcache init", native_result) != 0:
                raise RuntimeError(f"Memcache store initialization failed with result {native_result!r}")
            if self._init_bm and self._dp_init_barrier:
                logger.info("Waiting for all DP Memcache v1 initializations")
                torch.distributed.barrier(group=get_dp_group().cpu_group)
                logger.info("All DP Memcache v1 initializations completed")
            time.sleep(MEMCACHE_THREAD_START_WAIT_S)
        except BaseException:
            self._close_after_setup_failure(store)
            raise
        return store

    def _register_buffers_if_ready(self) -> None:
        if self._pending_buffers is None or not self._initialized:
            return
        assert self._store is not None
        registration = self._pending_registration
        assert registration is not None
        addresses, sizes = self._pending_buffers
        try:
            for address, size in zip(addresses, sizes, strict=True):
                native_result = self._store.register_buffer(address, size)
                if isinstance(native_result, Integral) and native_result != 0:
                    raise RuntimeError(
                        f"Memcache buffer registration failed: address={address}, size={size}, result={native_result}"
                    )
                registration.acquire(address, size)
        except BaseException as error:
            self._registration_error = error
            try:
                rollback_buffer_registration(registration, error)
            except BaseException as rollback_error:
                self._registration_error = rollback_error
                raise
            raise
        self._pending_buffers = None
        self._pending_registration = None

    def _cancel_pending_registration(self, registration: BufferRegistration) -> None:
        if self._pending_registration is registration:
            self._pending_buffers = None
            self._pending_registration = None

    def _unregister_native_region(self, address: int, size: int) -> None:
        if not self._initialized or self._store is None:
            raise RuntimeError("Memcache store is unavailable during buffer deregistration")
        native_result = self._store.unregister_buffer(address, size)
        if isinstance(native_result, Integral) and native_result != 0:
            raise RuntimeError(
                f"Memcache buffer deregistration failed: address={address}, size={size}, result={native_result}"
            )

    def _copy_gva(
        self,
        remote_addresses: list[int],
        local_addresses: list[int],
        sizes: list[int],
        direction: int,
    ) -> int:
        self._ensure_initialized()
        assert self._store is not None
        return result_code(
            "Memcache batch_copy",
            self._store.batch_copy(remote_addresses, local_addresses, sizes, direction),
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Memcache v1 Backend is closed")

    @staticmethod
    def _close_store(store: Any) -> None:
        close_store = getattr(store, "close", None)
        if not callable(close_store):
            return
        native_result = close_store()
        if isinstance(native_result, Integral) and native_result != 0:
            raise RuntimeError(f"Memcache store close failed: result={native_result}")

    @classmethod
    def _close_after_setup_failure(cls, store: Any) -> None:
        try:
            cls._close_store(store)
        except Exception:
            logger.exception("Failed to close Memcache v1 store after setup failed")
