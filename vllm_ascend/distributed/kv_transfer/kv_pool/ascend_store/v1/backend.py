"""Backend selection owned by AscendStore v1."""

from __future__ import annotations

import importlib
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


class BackendAdapter:
    """Expose Backend operations without discarding native Store results."""

    def __init__(self, backend_name: str, backend: Any, backend_module: Any) -> None:
        self._backend_name = backend_name
        self._backend = backend
        self._backend_module = backend_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> list[int] | None:
        if self._backend_name == "mooncake":
            self._backend.ensure_initialized()
            if self._backend.store is None:
                raise RuntimeError("Mooncake store is unavailable for put")
            return self._backend.store.batch_put_from_multi_buffers(
                keys, addrs, sizes, self._backend._build_replicate_config()
            )
        if self._backend_name == "memcache":
            self._backend.ensure_initialized()
            if self._backend.store is None:
                raise RuntimeError("Memcache store is unavailable for put")
            direction = self._backend_module.MmcDirect.COPY_L2G.value
            return self._backend.store.batch_put_from_layers(keys, addrs, sizes, direction)
        if self._backend.store is None:
            raise RuntimeError("Yuanrong store is unavailable for put")
        return self._backend.store.mset_d2h_from_multi_buffers(keys, addrs, sizes, self._backend._ds_set_param)


def create_backend(backend_name: str, parallel_config: ParallelConfig, extra_config: dict[str, Any]) -> BackendAdapter:
    backend_import = BACKEND_IMPORTS.get(backend_name)
    if backend_import is None:
        raise ValueError(f"Unsupported AscendStore v1 backend: {backend_name}")
    module_path, class_name = backend_import
    backend_module = importlib.import_module(module_path)
    backend_type = getattr(backend_module, class_name)
    backend = backend_type(parallel_config, extra_config=extra_config)
    return BackendAdapter(backend_name, backend, backend_module)
