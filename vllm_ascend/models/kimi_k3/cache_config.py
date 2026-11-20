# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Shared full-history and recurrent grouping for Kimi K3 and K3.1."""

from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MambaSpec,
)

from vllm_ascend.core.kv_cache_layers import CacheGroup, cache_layouts, make_cache_group
from vllm_ascend.utils import vllm_version_is

_KIMI_K3_TARGET_LAYER_PREFIX = "language_model.model.layers."
_KIMI_K3_DRAFT_LAYER_PREFIX = "model.layers."


if vllm_version_is("0.33.0"):
    from vllm.v1.kv_cache_interface import FullAttentionSpec


def make_kimi_cache_groups(
    full_history_specs: dict[str, KVCacheSpec],
    recurrent_specs: dict[str, MambaSpec],
    *,
    recurrent_group_size: int,
    uniform_recurrent: bool,
    configs=None,
) -> list[CacheGroup] | None:
    """Share full-history ownership and balance recurrent groups by slot budget.

    Callers validate roles and geometry, and supply specs in layer order with
    target attention first for scheduler conversion. The recurrent slot budget
    is explicit: K3 uses target plus draft pages, whereas K3.1 overlays only
    target MLA pages, excluding both indexer and draft storage.

    K3 retains per-layer recurrent specs for its shared-tuple allocation;
    K3.1 keeps native merged Mamba specs for its paired-page allocation.
    """
    full_history = make_cache_group(configs, full_history_specs)
    if full_history is None:
        return None
    groups = [full_history]
    names = list(recurrent_specs)
    count = cdiv(len(names), recurrent_group_size)
    for i in range(count):
        members = {name: recurrent_specs[name] for name in names[i::count]}
        group = make_cache_group(configs, members, uniform=uniform_recurrent)
        if group is None:
            return None
        groups.append(group)
    return groups


def _get_kimi_k3_dspark_mixed_kv_cache_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[CacheGroup] | None:
    """Build topology-independent Kimi K3 DSpark scheduler groups.

    Target and causal draft attention layers require the same full-sequence
    block ownership. Putting them in one UniformType group lets them share one
    scheduler block table while preserving a separate physical page per layer.
    Recurrent layers are split into the fewest balanced groups whose size does
    not exceed the attention group. This minimizes scheduler groups while
    keeping the recurrent groups balanced.

    Block and page sizes are resolved by the runtime and intentionally not
    fixed here: TP8 and TP16 produce different sizes but the same ownership
    relation. An unrecognized or incompatible signature falls back to vLLM's
    generic hybrid grouping.
    """
    configs = kv_cache_spec
    kv_cache_spec = cache_layouts(configs)

    def full_history(name, spec):
        return (
            issubclass(configs[name].manager_cls, FullAttentionManager)
            if not vllm_version_is("0.33.0")
            else isinstance(spec, FullAttentionSpec)
        )

    target_attention_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and full_history(name, spec)
    }
    draft_attention_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_DRAFT_LAYER_PREFIX) and full_history(name, spec)
    }
    mamba_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and isinstance(spec, MambaSpec)
    }

    matched_layer_count = len(target_attention_specs) + len(draft_attention_specs) + len(mamba_specs)
    if (
        not target_attention_specs
        or not draft_attention_specs
        or not mamba_specs
        or matched_layer_count != len(kv_cache_spec)
    ):
        return None

    attention_specs = [*target_attention_specs.values(), *draft_attention_specs.values()]
    all_specs = [*attention_specs, *mamba_specs.values()]
    # Only attention layers share a scheduler block table. Mamba keeps its
    # own groups and may use max_model_len as block_size when cache_mode=none.
    if (
        len({spec.block_size for spec in attention_specs}) != 1
        or len({spec.dcp_sharded for spec in attention_specs}) != 1
        or len({spec.page_size_bytes for spec in all_specs}) != 1
    ):
        return None

    first_mamba_spec = next(iter(mamba_specs.values()))
    if any(spec != first_mamba_spec for spec in mamba_specs.values()):
        return None

    # Insert target attention first. generate_scheduler_kv_cache_config unwraps a
    # UniformType group to its first spec, and this representative is registered
    # with the FullAttentionManager needed by both target and draft attention.
    mixed_attention_specs = {**target_attention_specs, **draft_attention_specs}
    groups = make_kimi_cache_groups(
        mixed_attention_specs,
        mamba_specs,
        recurrent_group_size=len(mixed_attention_specs),
        uniform_recurrent=True,
        configs=configs,
    )
    if groups is None:
        return None

    logger.info(
        "Using Kimi K3 DSpark mixed KV grouping: %d target + %d draft attention layers, followed by Mamba groups %s",
        len(target_attention_specs),
        len(draft_attention_specs),
        [len(group.layer_names) for group in groups[1:]],
    )
    return groups


def _uses_causal_draft_attention(config) -> bool:
    dflash_config = getattr(config, "dflash_config", None)
    if isinstance(dflash_config, dict) and "causal" in dflash_config:
        return bool(dflash_config["causal"])
    return bool(getattr(config, "full_attention_causal", False))
