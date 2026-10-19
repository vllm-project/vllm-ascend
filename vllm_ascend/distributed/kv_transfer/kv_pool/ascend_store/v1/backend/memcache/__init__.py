"""Memcache adapter; native registration and session mechanics stay in this package."""

from ....backend.memcache_backend import _full_key_prefix
from ...infrastructure.configuration.naming import KeyNamingConfig
from ..naming import WireKeyBuilder
from .backend import MemcacheBackend
from .layout import create_object_layout


def create_key_builder(config: KeyNamingConfig) -> WireKeyBuilder:
    group_count = len(config.group_block_sizes)
    return WireKeyBuilder(
        tuple(
            tuple(
                _full_key_prefix(config.model_name, group, group_count, stage, config.pp_size)
                for stage in range(config.pp_size)
            )
            for group in range(group_count)
        ),
        tuple(f"@{head}" for head in range(config.tp_size)),
    )


__all__ = ["MemcacheBackend", "create_key_builder", "create_object_layout"]
