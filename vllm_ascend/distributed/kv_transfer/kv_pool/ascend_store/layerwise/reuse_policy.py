# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit model buffer assignments and the compiled layerwise layout."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from vllm.v1.kv_cache_interface import KVCacheSpec, KVCacheTensor

# A predicate handles specs whose eligibility also depends on their metadata.
SpecReuseRules = Mapping[type[KVCacheSpec], bool | Callable[[KVCacheSpec], bool]]


@dataclass(frozen=True)
class LayerwiseReusePlan:
    layer_indices: dict[str, int]
    buffer_slots: tuple[tuple[str, ...], ...]
    independent_layers: list[int]
    num_prefetch_layers: int
    draft_layer_names: frozenset[str] = frozenset()


@dataclass(frozen=True)
class LayerwiseReuseLayout:
    layers: tuple[int, ...]
    cache_layer_indices: dict[str, int]
    kv_cache_tensors: tuple[KVCacheTensor, ...]
    # Target layer -> source layers whose saves must finish before buffer reuse.
    # A target may reuse several buffers, e.g. {4: (0, 2)} waits for both saves.
    # Empty saves signal completion independently, so waiting only for the
    # largest source index does not guarantee that earlier saves have finished.
    prefetch_layer_map: dict[int, tuple[int, ...]]
    independent_layers: list[int]
    num_prefetch_layers: int
    reused_cache_names: frozenset[str]

    @property
    def has_layer_reuse(self) -> bool:
        return bool(self.prefetch_layer_map)
