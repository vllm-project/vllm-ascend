# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp grouping policy; storage and block planning belong to vLLM."""

import math
from dataclasses import replace

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

QWEN_CACHE_GROUP_SIZE = 12
CACHE_ALIGNMENT = 16
GBSA_PAGE_TOKENS = 128


def is_qwen4_exp_cache(specs_or_groups: dict[str, KVCacheSpec] | list[KVCacheGroupSpec]) -> bool:
    items = specs_or_groups.values() if isinstance(specs_or_groups, dict) else specs_or_groups
    for item in items:
        spec = getattr(item, "kv_cache_spec", item)
        specs = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        if any(getattr(member, "model_version", None) == "qwen4_exp" for member in specs):
            return True
    return False


def _layer_index(name: str) -> int:
    if ".layers." not in name:
        return -1
    return int(name.rsplit(".layers.", 1)[1].split(".", 1)[0])


def _cache_kind(spec: KVCacheSpec) -> int:
    # Keep PLE separate from GDN, including its replicated TP semantics.
    if isinstance(spec, HiddenStateCacheSpec):
        return 5
    if type(spec) is FullAttentionSpec:
        return 0
    if isinstance(spec, MLAAttentionSpec):
        return 1
    if isinstance(spec, MambaSpec):
        if len(spec.shapes) != len(spec.dtypes) or len(spec.shapes) not in (1, 2):
            raise ValueError("Qwen4Exp requires one PLE state or two GDN states")
        return 2 if len(spec.shapes) == 2 else 3
    if isinstance(spec, CircularBufferSpec):
        return 4
    raise ValueError(f"Unsupported Qwen4Exp cache spec: {type(spec).__name__}")


def get_qwen4_exp_kv_cache_groups(vllm_config: VllmConfig, specs: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec]:
    """Pad physical pages only, then round-robin homogeneous buckets by 12.

    Remove this policy once the generic grouper supports the desired group
    width. Ordinary specs/groups retain all scheduler and transfer semantics;
    no model-specific tensors, allocation budget or block manager is needed.
    """
    if not specs:
        return []
    target_layers = len(vllm_config.model_config.hf_text_config.layer_types)
    unpadded = {name: replace(spec, page_size_padded=None) for name, spec in specs.items()}
    alignment = CACHE_ALIGNMENT
    for spec in unpadded.values():
        _cache_kind(spec)
        if type(spec) is FullAttentionSpec:
            if spec.block_size % GBSA_PAGE_TOKENS:
                raise ValueError("Qwen4Exp attention blocks must contain complete GBSA pages")
            # Both planar K and V must be representable as whole GBSA pages.
            for width in (spec.head_size, spec.head_size_v):
                alignment = math.lcm(alignment, GBSA_PAGE_TOKENS * spec.num_kv_heads * width * spec.dtype.itemsize)
        alignment = math.lcm(alignment, spec.block_stride_alignment or 1)
    page_size = round_up(max(spec.page_size_bytes for spec in unpadded.values()), alignment)
    buckets: dict[tuple[bool, int, KVCacheSpec], list[str]] = {}
    for name in sorted(specs, key=lambda name: (_layer_index(name), name)):
        spec = replace(unpadded[name], page_size_padded=page_size)
        # Mutate the merged spec mapping as the standard grouping contract
        # permits. PP projects these normalized specs, not worker originals.
        specs[name] = spec
        is_draft = ".mtp." in name or _layer_index(name) >= target_layers
        buckets.setdefault((is_draft, _cache_kind(spec), spec), []).append(name)
    groups = []
    for (is_draft, _, spec), names in sorted(buckets.items(), key=lambda item: item[0][:2]):
        num_groups = cdiv(len(names), QWEN_CACHE_GROUP_SIZE)
        for index in range(num_groups):
            members = names[index::num_groups]
            # Equal full specs preserve dtype, shapes, compression, checkpoint
            # policy, TP replication, and any future management attributes.
            merged = type(spec).merge([specs[name] for name in members])
            if merged != spec:
                raise ValueError("Qwen4Exp group merge changed cache semantics")
            groups.append(
                KVCacheGroupSpec(
                    layer_names=members,
                    kv_cache_spec=merged,
                    is_eagle_group=is_draft,
                    enable_kv_transfer=not isinstance(spec, CircularBufferSpec),
                )
            )
    return groups
