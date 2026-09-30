"""Backend selection owned by AscendStore v1."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..backend.memcache_backend import MemcacheBackend
    from ..backend.mooncake_backend import MooncakeBackend
    from ..backend.yuanrong_backend import YuanrongBackend

_BACKEND_PACKAGE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend"

BACKEND_IMPORTS = MappingProxyType(
    {
        "mooncake": (f"{_BACKEND_PACKAGE}.mooncake_backend", "MooncakeBackend"),
        "memcache": (f"{_BACKEND_PACKAGE}.memcache_backend", "MemcacheBackend"),
        "yuanrong": (f"{_BACKEND_PACKAGE}.yuanrong_backend", "YuanrongBackend"),
    }
)


class LayerwiseAccessKind(Enum):
    KEY_RANGE = "key_range"
    GVA = "gva"


LAYERWISE_ACCESS = MappingProxyType({"mooncake": LayerwiseAccessKind.KEY_RANGE, "memcache": LayerwiseAccessKind.GVA})


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """Fix one Backend implementation and its startup-time capabilities."""

    name: str
    backend_type: type[MooncakeBackend] | type[MemcacheBackend] | type[YuanrongBackend]
    backend_module: ModuleType
    layerwise_access: LayerwiseAccessKind | None
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
        LAYERWISE_ACCESS.get(backend_name),
        bool(getattr(backend_type, "requires_exists_before_put", True)),
    )
