"""Backend selection owned by AscendStore v1."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import ParallelConfig


_BACKEND_PACKAGE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend"

BACKEND_IMPORTS = MappingProxyType(
    {
        "mooncake": (f"{_BACKEND_PACKAGE}.mooncake_backend", "MooncakeBackend"),
        "memcache": (f"{_BACKEND_PACKAGE}.memcache_backend", "MemcacheBackend"),
        "yuanrong": (f"{_BACKEND_PACKAGE}.yuanrong_backend", "YuanrongBackend"),
    }
)

BLOCK_KEY_LAYERWISE_BACKENDS = frozenset({"mooncake"})


@dataclass(frozen=True, slots=True)
class BackendStoreEvidence:
    """Normalized Store facts returned by one synchronous Backend call."""

    result_codes: tuple[int, ...] | None
    succeeded: bool
    source_release_confirmed: bool
    error: Exception | None = None


class BackendAdapter:
    """Expose Backend operations without discarding native Store results."""

    def __init__(self, backend_name: str, backend: Any, backend_module: Any) -> None:
        self._backend_name = backend_name
        self._backend = backend
        self._backend_module = backend_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> BackendStoreEvidence:
        source_addresses_handed_off = False
        try:
            if self._backend_name == "mooncake":
                self._backend.ensure_initialized()
                if self._backend.store is None:
                    raise RuntimeError("Mooncake store is unavailable for put")
                replicate_config = self._backend._build_replicate_config()
                put = self._backend.store.batch_put_from_multi_buffers
                source_addresses_handed_off = True
                native_result = put(keys, addrs, sizes, replicate_config)
            elif self._backend_name == "memcache":
                self._backend.ensure_initialized()
                if self._backend.store is None:
                    raise RuntimeError("Memcache store is unavailable for put")
                direction = self._backend_module.MmcDirect.COPY_L2G.value
                put = self._backend.store.batch_put_from_layers
                source_addresses_handed_off = True
                native_result = put(keys, addrs, sizes, direction)
            else:
                if self._backend.store is None:
                    raise RuntimeError("Yuanrong store is unavailable for put")
                put = self._backend.store.mset_d2h_from_multi_buffers
                set_param = self._backend._ds_set_param
                source_addresses_handed_off = True
                native_result = put(keys, addrs, sizes, set_param)
        except Exception as error:
            return BackendStoreEvidence(None, False, not source_addresses_handed_off, error)

        if self._backend_name == "yuanrong":
            return BackendStoreEvidence(None, True, source_release_confirmed=True)
        return self._interpret_result_codes(keys, native_result)

    def validate_layerwise_support(self) -> None:
        self._backend.validate_layerwise_support()

    def start_load_sessions(self, keys: list[str]) -> tuple[int, ...]:
        return self._require_result_codes("batch_get_start", keys, self._backend.batch_get_start(keys))

    def load_session_ranges(
        self,
        keys: list[str],
        addresses: list[list[int]],
        sizes: list[list[int]],
        remote_offsets: list[list[int]],
    ) -> tuple[int, ...]:
        native_result = self._backend.batch_copy_get(keys, addresses, sizes, remote_offsets)
        return self._require_result_codes("batch_copy_get", keys, native_result)

    def finish_load_sessions(self, keys: list[str]) -> int:
        native_result = self._backend.batch_get_end(keys)
        if isinstance(native_result, bool) or not isinstance(native_result, Integral):
            raise RuntimeError("batch_get_end returned a non-integer result")
        return int(native_result)

    def start_store_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        return self._require_result_codes("batch_put_start", keys, self._backend.batch_put_start(keys, object_sizes))

    def store_session_ranges(
        self,
        keys: list[str],
        addresses: list[list[int]],
        sizes: list[list[int]],
        remote_offsets: list[list[int]],
    ) -> BackendStoreEvidence:
        source_addresses_handed_off = False
        try:
            store_ranges = self._backend.batch_copy_put
            source_addresses_handed_off = True
            native_result = store_ranges(keys, addresses, sizes, remote_offsets)
        except Exception as error:
            return BackendStoreEvidence(None, False, not source_addresses_handed_off, error)
        return self._interpret_result_codes(keys, native_result)

    def commit_store_sessions(self, keys: list[str]) -> tuple[int, ...]:
        return self._require_result_codes("batch_commit", keys, self._backend.batch_commit(keys))

    def revoke_store_sessions(self, keys: list[str]) -> tuple[int, ...]:
        return self._require_result_codes("batch_revoke", keys, self._backend.batch_revoke(keys))

    def _interpret_result_codes(self, keys: list[str], native_result: Any) -> BackendStoreEvidence:
        try:
            codes = None if native_result is None else tuple(native_result)
        except Exception as error:
            return BackendStoreEvidence(None, False, source_release_confirmed=True, error=error)
        if codes is None:
            store_error = RuntimeError(f"{self._backend_name} Store returned no per-key results")
            return BackendStoreEvidence(None, False, source_release_confirmed=True, error=store_error)
        if len(codes) != len(keys):
            store_error = RuntimeError(f"{self._backend_name} Store returned {len(codes)} results for {len(keys)} keys")
            return BackendStoreEvidence(codes, False, source_release_confirmed=True, error=store_error)
        return BackendStoreEvidence(codes, all(code == 0 for code in codes), source_release_confirmed=True)

    @staticmethod
    def _require_result_codes(operation: str, keys: list[str], native_result: Any) -> tuple[int, ...]:
        try:
            codes = tuple(native_result)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"{operation} returned non-integer results") from error
        if len(codes) != len(keys):
            raise RuntimeError(f"{operation} returned {len(codes)} results for {len(keys)} keys")
        if any(isinstance(code, bool) or not isinstance(code, Integral) for code in codes):
            raise RuntimeError(f"{operation} returned non-integer results")
        return tuple(int(code) for code in codes)


def create_backend(backend_name: str, parallel_config: ParallelConfig, extra_config: dict[str, Any]) -> BackendAdapter:
    backend_import = BACKEND_IMPORTS.get(backend_name)
    if backend_import is None:
        raise ValueError(f"Unsupported AscendStore v1 backend: {backend_name}")
    module_path, class_name = backend_import
    backend_module = importlib.import_module(module_path)
    backend_type = getattr(backend_module, class_name)
    backend = backend_type(parallel_config, extra_config=extra_config)
    return BackendAdapter(backend_name, backend, backend_module)
