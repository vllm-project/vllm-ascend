"""Select native Backend adapters owned by AscendStore v1."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from vllm.config import ParallelConfig

from .port import GVABackend, GVARegion, KeyRangeBackend, KVStoreBackend
from .registration import (
    BufferRegion,
    BufferRegistration,
    BufferRegistrationError,
    BufferReleaseError,
    Registration,
    rollback_buffer_registration,
)

BackendFactory = Callable[..., KVStoreBackend]

BACKEND_IMPORTS = MappingProxyType(
    {
        "mooncake": (f"{__name__}.mooncake", "MooncakeBackend"),
        "memcache": (f"{__name__}.memcache", "MemcacheBackend"),
    }
)


class LayerwiseAccessKind(Enum):
    KEY_RANGE = "key_range"
    GVA = "gva"


LAYERWISE_ACCESS = MappingProxyType(
    {
        "mooncake": LayerwiseAccessKind.KEY_RANGE,
        "memcache": LayerwiseAccessKind.GVA,
    }
)


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """Fix one v1 Adapter and its startup-time capabilities."""

    name: str
    backend_factory: BackendFactory
    layerwise_access: LayerwiseAccessKind | None
    requires_exists_before_put: bool

    def create(
        self,
        parallel_config: ParallelConfig,
        *,
        extra_config: dict[str, Any] | None = None,
    ) -> KVStoreBackend:
        return self.backend_factory(parallel_config, extra_config=extra_config)


def resolve_backend_spec(backend_name: str) -> BackendSpec:
    backend_import = BACKEND_IMPORTS.get(backend_name)
    if backend_import is None:
        raise ValueError(f"Unsupported AscendStore v1 backend: {backend_name}")
    module_path, class_name = backend_import
    backend_factory = getattr(importlib.import_module(module_path), class_name)
    return BackendSpec(
        backend_name,
        backend_factory,
        LAYERWISE_ACCESS.get(backend_name),
        bool(getattr(backend_factory, "requires_exists_before_put", True)),
    )


__all__ = [
    "BACKEND_IMPORTS",
    "BackendFactory",
    "BackendSpec",
    "BufferRegion",
    "BufferRegistration",
    "BufferRegistrationError",
    "BufferReleaseError",
    "GVABackend",
    "GVARegion",
    "KVStoreBackend",
    "KeyRangeBackend",
    "LayerwiseAccessKind",
    "Registration",
    "resolve_backend_spec",
    "rollback_buffer_registration",
]
