"""Backend selection owned by AscendStore v1."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
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

BLOCK_KEY_LAYERWISE_BACKENDS = frozenset({"mooncake"})


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """Fix one Backend implementation and its startup-time capabilities."""

    name: str
    backend_type: type[Backend]
    backend_module: ModuleType
    supports_layerwise: bool
    requires_exists_before_put: bool


def resolve_backend_spec(backend_name: str) -> BackendSpec:
    backend_import = BACKEND_IMPORTS.get(backend_name)
    if backend_import is None:
        raise ValueError(f"Unsupported AscendStore v1 backend: {backend_name}")
    module_path, class_name = backend_import
    backend_module = importlib.import_module(module_path)
    backend_type = getattr(backend_module, class_name)
    return BackendSpec(
        backend_name,
        backend_type,
        backend_module,
        backend_name in BLOCK_KEY_LAYERWISE_BACKENDS,
        bool(getattr(backend_type, "requires_exists_before_put", True)),
    )


def create_backend(
    backend_spec: BackendSpec,
    parallel_config: ParallelConfig,
    extra_config: dict[str, Any],
) -> Backend:
    return backend_spec.backend_type(parallel_config, extra_config=extra_config)
