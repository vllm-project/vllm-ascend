"""Mooncake adapter; native registration and session mechanics stay in this package."""

import hashlib
import json

from ....backend.mooncake_layerwise import _block_key_prefix, _hybrid_key_prefix
from ...infrastructure.configuration.naming import KeyNamingConfig
from ..naming import WireKeyBuilder
from .backend import MooncakeBackend

KEY_REQUIRES_LAYOUT_IDENTITY = True


def create_key_builder(config: KeyNamingConfig) -> WireKeyBuilder:
    """Encode the established wire identity from resolved, immutable facts only."""
    namespace = ""
    if config.pp_size > 1:
        empty = [index for index, names in enumerate(config.group_layer_names) if not names]
        if empty:
            raise ValueError(
                "Mooncake layerwise PP requires each stage to own layers in every KV cache group; "
                f"empty groups: {empty}. Use a partition covering all groups, or disable layerwise pooling. "
                "Draft-only groups are not supported."
            )
        previous_end = 0
        for start, end in config.layer_ranges:
            if start != previous_end or end <= start:
                raise ValueError("Mooncake PP requires contiguous, nonempty model layer partitions")
            previous_end = end
        if len(config.layer_ranges) != config.pp_size or previous_end != config.total_layers:
            raise ValueError("Mooncake PP partitions must cover every model layer")
        identity = (
            config.tp_size,
            config.layer_ranges,
            config.model_hash,
            config.cache_hash,
            config.cache_dtype,
            config.cache_block_size,
            config.speculative_hash,
        )
        namespace = "mooncake_pp_v3:" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if config.use_hybrid:
        groups = [
            (() if namespace else names, sizes)
            for names, sizes in zip(config.group_layer_names, config.group_page_sizes, strict=True)
        ]
        layout_identity = (namespace, config.tp_size, groups) if namespace else (config.tp_size, groups)
        layout = hashlib.sha256(json.dumps(layout_identity, sort_keys=True).encode()).hexdigest()
        prefixes = tuple(
            tuple(
                _hybrid_key_prefix(config.model_name, layout, group, size, stage if config.pp_size > 1 else None)
                for stage in range(config.pp_size)
            )
            for group, size in enumerate(config.group_block_sizes)
        )
    else:
        prefixes = (tuple(_block_key_prefix(config.model_name, namespace, stage) for stage in range(config.pp_size)),)
    return WireKeyBuilder(prefixes, tuple(f"@{head}" for head in range(config.tp_size)))


__all__ = ["MooncakeBackend", "create_key_builder"]
