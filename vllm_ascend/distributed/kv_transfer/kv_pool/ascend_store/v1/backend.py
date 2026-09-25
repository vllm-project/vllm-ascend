"""Backend selection owned by AscendStore v1."""

from __future__ import annotations

import importlib
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend

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


def create_backend(backend_name: str, parallel_config: ParallelConfig, extra_config: dict[str, Any]) -> Backend:
    backend_import = BACKEND_IMPORTS.get(backend_name)
    if backend_import is None:
        raise ValueError(f"Unsupported AscendStore v1 backend: {backend_name}")
    module_path, class_name = backend_import
    backend_type = getattr(importlib.import_module(module_path), class_name)
    return backend_type(parallel_config, extra_config=extra_config)
