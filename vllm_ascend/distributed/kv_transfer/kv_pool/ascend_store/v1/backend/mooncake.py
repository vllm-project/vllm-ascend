"""Mooncake native client adapter owned by AscendStore v1."""

from __future__ import annotations

import os
import threading
from typing import Any

import torch
from mooncake.store import ReplicateConfig  # type: ignore
from vllm.config import ParallelConfig
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
    DEFAULT_TENANT_ID,
    MOONCAKE_LAYERWISE_CLIENT_METHODS,
    MooncakeStoreConfig,
    _emit_whole_key_debug_event,
    _inject_store_qos,
    _ssd_setup_kwargs,
    _validate_store_qos,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    RegistrationAcquisitionError,
    global_te,
)
from vllm_ascend.distributed.parallel_state import get_global_rank

from .registration import (
    BufferRegistration,
    BufferRegistrationError,
    Registration,
    rollback_buffer_registration,
)
from .results import presence_results, result_code, result_codes


class MooncakeBackend:
    """Translate v1 operations directly to one owned Mooncake store client."""

    requires_exists_before_put = True

    def __init__(
        self,
        parallel_config: ParallelConfig,
        lazy_init: bool = False,
        contribute_memory: bool = True,
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        self.parallel_config = parallel_config
        self.config = MooncakeStoreConfig.load_from_env()
        if self.config.protocol != "ascend":
            raise NotImplementedError(f"MooncakeBackend does not support protocol {self.config.protocol!r}.")
        _inject_store_qos(extra_config)
        _validate_store_qos()
        self.device_id = torch.npu.current_device()

        self._store: Any | None = None
        self._local_segment: str | None = None
        self._use_fabric_mem = os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") == "1"
        self._use_store_independent_te = bool(os.getenv("ASCEND_GLOBAL_RESOURCE_CONFIG")) and not self._use_fabric_mem
        self._lazy_init = lazy_init and self._use_fabric_mem
        self._contribute_memory = contribute_memory
        self._initialized = False
        self._initialization_lock = threading.Lock()
        self._closed = False

        if not self._lazy_init:
            self._store = self._setup_store()
            self._initialized = True

    def set_device(self) -> None:
        torch.npu.set_device(self.device_id)

    def register_buffer(self, addresses: list[int], sizes: list[int]) -> Registration:
        self._require_open()
        if len(addresses) != len(sizes):
            raise ValueError(f"addresses and sizes must have the same length: {len(addresses)} != {len(sizes)}")
        if self._use_store_independent_te:
            self._ensure_initialized()
            registration = BufferRegistration(self._unregister_store_buffer)
            assert self._store is not None
            try:
                for address, size in zip(addresses, sizes, strict=True):
                    native_result = result_code("Mooncake register_buffer", self._store.register_buffer(address, size))
                    if native_result != 0:
                        raise RuntimeError(
                            f"Mooncake store buffer registration failed: address={address}, size={size}, "
                            f"result={native_result}"
                        )
                    registration.acquire(address, size)
            except BaseException as error:
                rollback_buffer_registration(registration, error)
                raise
            return registration
        if self._use_fabric_mem:
            return BufferRegistration()

        global_te.get_transfer_engine(get_ip(), device_name=None)
        try:
            return global_te.acquire_registration(addresses, sizes)
        except RegistrationAcquisitionError as error:
            raise BufferRegistrationError(
                error.operation_error,
                error.rollback_error,
                error.registration,
            ) from error

    def close(self) -> None:
        if self._closed:
            return
        if self._store is not None:
            close_store = getattr(self._store, "close", None)
            if not callable(close_store):
                raise RuntimeError("Installed Mooncake client does not expose close()")
            native_result = close_store()
            if native_result is not None and result_code("Mooncake close", native_result) != 0:
                raise RuntimeError(f"Mooncake store close failed: result={native_result}")
        self._store = None
        self._initialized = False
        self._closed = True

    def exists(self, keys: list[str]) -> tuple[bool, ...]:
        if self._lazy_init and not self._initialized:
            return (False,) * len(keys)
        assert self._store is not None
        return presence_results("Mooncake batch_is_exist", keys, self._store.batch_is_exist(keys))

    def load(
        self,
        keys: list[str],
        addresses: list[list[int]],
        sizes: list[list[int]],
    ) -> tuple[int, ...]:
        if self._lazy_init and not self._initialized:
            raise RuntimeError("Mooncake store is not initialized; Store must initialize it before Load")
        assert self._store is not None
        _emit_whole_key_debug_event("get", len(keys))
        codes = result_codes(
            "Mooncake batch_get_into_multi_buffers",
            keys,
            self._store.batch_get_into_multi_buffers(keys, addresses, sizes),
        )
        return tuple(0 if code > 0 else code for code in codes)

    def store(
        self,
        keys: list[str],
        addresses: list[list[int]],
        sizes: list[list[int]],
    ) -> tuple[int, ...]:
        self._ensure_initialized()
        assert self._store is not None
        _emit_whole_key_debug_event("put", len(keys))
        return result_codes(
            "Mooncake batch_put_from_multi_buffers",
            keys,
            self._store.batch_put_from_multi_buffers(keys, addresses, sizes, self._replicate_config()),
        )

    def validate_key_range_support(self) -> None:
        self._ensure_initialized()
        missing = [
            method_name
            for method_name in MOONCAKE_LAYERWISE_CLIENT_METHODS
            if self._store is None or not callable(getattr(self._store, method_name, None))
        ]
        if missing:
            raise RuntimeError(
                "Mooncake layerwise requires the session/range APIs from Mooncake PR #2881. Missing methods: "
                + ", ".join(missing)
            )

    def start_key_range_load(self, keys: list[str]) -> tuple[int, ...]:
        return self._call_key_range("batch_get_session_start", keys, keys)

    def copy_key_range_load(
        self,
        keys: list[str],
        addresses: list[list[int]],
        sizes: list[list[int]],
        source_offsets: list[list[int]],
    ) -> tuple[int, ...]:
        return self._call_key_range(
            "batch_get_into_multi_buffer_ranges",
            keys,
            keys,
            addresses,
            sizes,
            source_offsets,
        )

    def finish_key_range_load(self, keys: list[str]) -> None:
        self._ensure_initialized()
        if self._store is None:
            raise RuntimeError("Mooncake store is unavailable for batch_get_session_end")
        method = getattr(self._store, "batch_get_session_end", None)
        if not callable(method):
            raise RuntimeError("Mooncake client does not support batch_get_session_end")
        native_result = result_code("Mooncake batch_get_session_end", method(keys))
        if native_result != 0:
            raise RuntimeError(f"Mooncake batch_get_session_end failed with result {native_result}")

    def start_key_range_store(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        return self._call_key_range(
            "batch_put_session_start",
            keys,
            keys,
            object_sizes,
            self._replicate_config(),
        )

    def copy_key_range_store(
        self,
        keys: list[str],
        addresses: list[list[int]],
        sizes: list[list[int]],
        destination_offsets: list[list[int]],
    ) -> tuple[int, ...]:
        return self._call_key_range(
            "batch_put_from_multi_buffer_ranges",
            keys,
            keys,
            addresses,
            sizes,
            destination_offsets,
        )

    def commit_key_range_store(self, keys: list[str]) -> tuple[int, ...]:
        return self._call_key_range("batch_put_session_end", keys, keys)

    def revoke_key_range_store(self, keys: list[str]) -> tuple[int, ...]:
        return self._call_key_range("batch_put_session_revoke", keys, keys)

    def _ensure_initialized(self) -> None:
        self._require_open()
        if self._initialized:
            return
        with self._initialization_lock:
            if self._initialized:
                return
            logger.info("Initializing Mooncake v1 store. metadata_server=%s", self.config.metadata_server)
            self._store = self._setup_store()
            self._initialized = True

    def _setup_store(self) -> Any:
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError as error:
            raise ImportError(
                "Please install mooncake by following https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md"
            ) from error

        self.set_device()
        store = MooncakeDistributedStore()
        try:
            local_hostname = get_ip()
            setup_kwargs = _ssd_setup_kwargs(self.config)
            if setup_kwargs.get("ssd_offload_path") and self._contribute_memory:
                rank_path = os.path.join(
                    str(setup_kwargs["ssd_offload_path"]),
                    f"rank_{get_global_rank(self.parallel_config)}",
                )
                try:
                    os.makedirs(rank_path, exist_ok=True)
                except OSError as error:
                    raise RuntimeError(
                        f"Failed to create per-rank SSD offload directory: {rank_path!r} ({error})"
                    ) from error
                setup_kwargs["ssd_offload_path"] = rank_path
            if self.config.tenant_id != DEFAULT_TENANT_ID:
                setup_kwargs["tenant_id"] = self.config.tenant_id
            native_result = self._setup_native_store(store, local_hostname, setup_kwargs)
            if result_code("Mooncake setup", native_result) != 0:
                raise RuntimeError(
                    f"Mooncake store initialization failed: result={native_result}, "
                    f"metadata_server={self.config.metadata_server}"
                )
        except BaseException:
            self._close_after_setup_failure(store)
            raise
        return store

    def _setup_native_store(self, store: Any, local_hostname: str, setup_kwargs: dict[str, object]) -> Any:
        common = {
            "metadata_server": self.config.metadata_server,
            "global_segment_size": self.config.global_segment_size if self._contribute_memory else 0,
            "protocol": self.config.protocol,
            "rdma_devices": self.config.device_name,
            "master_server_addr": self.config.master_server_address,
            **setup_kwargs,
        }
        if not self._use_fabric_mem and not self._use_store_independent_te:
            transfer_engine = global_te.get_transfer_engine(local_hostname, device_name=None)
            self._local_segment = f"{local_hostname}:{transfer_engine.get_rpc_port()}"
            return store.setup(
                local_hostname=self._local_segment,
                local_buffer_size=self.config.local_buffer_size if self._contribute_memory else 0,
                engine=transfer_engine.get_engine(),
                **common,
            )
        self._local_segment = local_hostname
        return store.setup(local_hostname=local_hostname, local_buffer_size=0, **common)

    def _replicate_config(self) -> ReplicateConfig:
        config = ReplicateConfig()
        if self.config.preferred_segment:
            config.preferred_segment = self._local_segment
        config.prefer_alloc_in_same_node = self.config.prefer_alloc_in_same_node
        return config

    def _call_key_range(self, operation: str, keys: list[str], *arguments: object) -> tuple[int, ...]:
        self._ensure_initialized()
        if self._store is None:
            raise RuntimeError(f"Mooncake store is unavailable for {operation}")
        method = getattr(self._store, operation, None)
        if not callable(method):
            raise RuntimeError(f"Mooncake client does not support {operation}")
        return result_codes(f"Mooncake {operation}", keys, method(*arguments))

    def _unregister_store_buffer(self, address: int, _size: int) -> None:
        if self._store is None:
            raise RuntimeError("Mooncake store is unavailable during buffer deregistration")
        native_result = result_code("Mooncake unregister_buffer", self._store.unregister_buffer(address))
        if native_result != 0:
            raise RuntimeError(
                f"Mooncake store buffer deregistration failed: address={address}, result={native_result}"
            )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Mooncake v1 Backend is closed")

    @staticmethod
    def _close_after_setup_failure(store: Any) -> None:
        close_store = getattr(store, "close", None)
        if callable(close_store):
            try:
                close_store()
            except Exception:
                logger.exception("Failed to close Mooncake v1 store after setup failed")
