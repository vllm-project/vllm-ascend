# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Compatibility shim exposing vLLM #61452 KVCacheGroup APIs on vLLM v0.34.0.

Loaded only when ``vllm_version_is("0.34.0")`` is True (and by CPU unit test
conftest on v0.34.0). Delete this file once vLLM v0.34.0 support is dropped.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from vllm.v1 import kv_cache_interface
from vllm.v1.core import single_type_kv_cache_manager as st_mgr
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    RSWASpec,
    SinkFullAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

_FALLBACK_SPEC_MANAGERS: tuple[tuple[type[KVCacheSpec], type[SingleTypeKVCacheManager]], ...] = (
    (HiSparseHotSpec, st_mgr.HiSparseHotManager),
    (HiSparseResidentSpec, st_mgr.HiSparseResidentManager),
    (CircularBufferSpec, st_mgr.CircularBufferManager),
    (MambaSpec, st_mgr.MambaManager),
    (CrossAttentionSpec, st_mgr.CrossAttentionManager),
    (ChunkedLocalAttentionSpec, st_mgr.ChunkedLocalAttentionManager),
    (SlidingWindowSpec, st_mgr.SlidingWindowManager),
    (SinkFullAttentionSpec, st_mgr.SinkFullAttentionManager),
    (RSWASpec, st_mgr.RSWAManager),
    (FullAttentionSpec, st_mgr.FullAttentionManager),
)


def _first_spec(group: KVCacheGroupSpec) -> KVCacheSpec:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.first_spec
    return spec


def _kv_cache_specs(self: KVCacheGroupSpec) -> dict[str, KVCacheSpec]:
    spec = self.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return {name: spec.kv_cache_specs[name] for name in self.layer_names}
    return {name: spec for name in self.layer_names}


def _distinct_specs(self: KVCacheGroupSpec) -> tuple[KVCacheSpec, ...]:
    spec = self.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        # KVCacheSpec.is_draft uses compare=False on v0.34.0; keep draft and
        # target entries distinct so callers inspecting draft ownership see both.
        seen: dict[tuple[KVCacheSpec, bool], KVCacheSpec] = {}
        for layer_spec in spec.kv_cache_specs.values():
            seen.setdefault((layer_spec, getattr(layer_spec, "is_draft", False)), layer_spec)
        return tuple(seen.values())
    return (spec,)


def _block_size(self: KVCacheGroupSpec) -> int:
    return self.kv_cache_spec.block_size


def _dcp_sharded(self: KVCacheGroupSpec) -> bool:
    return bool(getattr(_first_spec(self), "dcp_sharded", False))


def _manager_cls(self: KVCacheGroupSpec) -> type[SingleTypeKVCacheManager]:
    first = _first_spec(self)
    role = getattr(self, "role", None)
    manager = KVCacheSpecRegistry.get_manager_class(first, role=role)
    if manager is not None:
        return manager
    for spec_cls, manager_cls in _FALLBACK_SPEC_MANAGERS:
        if isinstance(first, spec_cls):
            return manager_cls
    return st_mgr.FullAttentionManager


def _manager_config(self: KVCacheGroupSpec) -> Any:
    return SimpleNamespace(manager_cls=self.manager_cls, params=_first_spec(self))


def _prefix_cacheable(self: KVCacheGroupSpec) -> bool:
    from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable

    return is_prefix_cacheable(self.kv_cache_spec)


def _prefix_replay_tokens(self: KVCacheGroupSpec) -> int:
    return int(getattr(_first_spec(self), "prefix_replay_tokens", 0))


def _sliding_window_reach(self: KVCacheGroupSpec) -> int:
    first = _first_spec(self)
    if isinstance(first, SlidingWindowSpec):
        return int(first.num_retained_tokens)
    return 0


def _uses_slot_mapping(self: KVCacheGroupSpec) -> bool:
    return bool(self.kv_cache_spec.uses_slot_mapping)


def _page_size_bytes(self: KVCacheGroupSpec) -> int:
    spec = self.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return sum(layer_spec.page_size_bytes for layer_spec in spec.kv_cache_specs.values())
    return spec.page_size_bytes * len(self.layer_names)


def _max_allocated_blocks_per_request(self: KVCacheGroupSpec, vllm_config: VllmConfig) -> int:
    first = _first_spec(self)
    return first.max_memory_usage_bytes(vllm_config) // first.page_size_bytes


def _max_memory_usage_bytes(self: KVCacheGroupSpec, vllm_config: VllmConfig) -> int:
    return self.kv_cache_spec.max_memory_usage_bytes(vllm_config)


def _max_num_blocks_per_req(self: KVCacheGroupSpec, vllm_config: VllmConfig, max_len: int | None = None) -> int:
    if max_len is None:
        max_len = vllm_config.model_config.max_model_len
    return self.kv_cache_spec.max_num_blocks_per_req(vllm_config, max_len)


def _block_table_width(self: KVCacheGroupSpec, vllm_config: VllmConfig, max_len: int) -> int:
    from vllm.v1.kv_cache_interface import get_block_table_width

    spec = self.kv_cache_spec
    max_num_blocks = spec.max_num_blocks_per_req(vllm_config, max_len)
    return get_block_table_width(
        max_num_blocks,
        spec.block_size,
        token_alignment=spec.block_table_token_alignment,
    )


def apply_v034_kv_cache_compat() -> None:
    """Attach #61452 KVCacheGroup properties/methods onto v0.34.0 KVCacheGroupSpec."""
    if not hasattr(kv_cache_interface, "KVCacheGroup"):
        kv_cache_interface.KVCacheGroup = KVCacheGroupSpec  # type: ignore[attr-defined]
    if getattr(KVCacheGroupSpec, "_ascend_v034_compat_applied", False):
        return
    KVCacheGroupSpec.kv_cache_specs = property(_kv_cache_specs)  # type: ignore[attr-defined]
    KVCacheGroupSpec.distinct_specs = property(_distinct_specs)  # type: ignore[attr-defined]
    KVCacheGroupSpec.block_size = property(_block_size)  # type: ignore[attr-defined]
    KVCacheGroupSpec.dcp_sharded = property(_dcp_sharded)  # type: ignore[attr-defined]
    KVCacheGroupSpec.manager_cls = property(_manager_cls)  # type: ignore[attr-defined]
    KVCacheGroupSpec.manager_config = property(_manager_config)  # type: ignore[attr-defined]
    KVCacheGroupSpec.prefix_cacheable = property(_prefix_cacheable)  # type: ignore[attr-defined]
    KVCacheGroupSpec.prefix_replay_tokens = property(_prefix_replay_tokens)  # type: ignore[attr-defined]
    KVCacheGroupSpec.sliding_window_reach = property(_sliding_window_reach)  # type: ignore[attr-defined]
    KVCacheGroupSpec.uses_slot_mapping = property(_uses_slot_mapping)  # type: ignore[attr-defined]
    KVCacheGroupSpec.page_size_bytes = property(_page_size_bytes)  # type: ignore[attr-defined]
    KVCacheGroupSpec.max_allocated_blocks_per_request = _max_allocated_blocks_per_request  # type: ignore[attr-defined]
    KVCacheGroupSpec.max_memory_usage_bytes = _max_memory_usage_bytes  # type: ignore[attr-defined]
    KVCacheGroupSpec.max_num_blocks_per_req = _max_num_blocks_per_req  # type: ignore[attr-defined]
    KVCacheGroupSpec.block_table_width = _block_table_width  # type: ignore[attr-defined]
    KVCacheGroupSpec._ascend_v034_compat_applied = True  # type: ignore[attr-defined]


apply_v034_kv_cache_compat()
