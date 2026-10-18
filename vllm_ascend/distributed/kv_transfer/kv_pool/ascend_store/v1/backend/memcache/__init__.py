"""Memcache adapter; native registration and session mechanics stay in this package."""

from ....backend.memcache_backend import bind_layerwise_keys
from .. import LayerwiseObjectLayoutKind
from .backend import MemcacheBackend

LAYERWISE_OBJECT_LAYOUT = LayerwiseObjectLayoutKind.GLOBAL_GVA

__all__ = ["MemcacheBackend", "bind_layerwise_keys"]
