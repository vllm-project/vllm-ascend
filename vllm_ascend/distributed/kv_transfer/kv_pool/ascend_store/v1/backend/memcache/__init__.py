"""Memcache adapter; native registration and session mechanics stay in this package."""

from ...infrastructure.configuration.naming import KeyNamingConfig
from ..naming import WireKeyBuilder
from .backend import MemcacheBackend


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


def _full_key_prefix(model_name: str, group_id: int, num_groups: int, pp_rank: int, pp_size: int) -> str:
    pp_tag = f"@pp{pp_rank}" if pp_size > 1 else ""
    group_tag = f"@{group_id}" if num_groups > 1 else ""
    return f"{model_name}{group_tag}{pp_tag}@"


__all__ = ["MemcacheBackend", "create_key_builder"]
