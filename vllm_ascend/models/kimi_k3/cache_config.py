# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Shared full-history and recurrent grouping for Kimi K3 and K3.1."""

from collections.abc import Mapping
from typing import Any

from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec

from vllm_ascend.core.kv_cache_interface import KVCacheGroup
from vllm_ascend.utils import vllm_version_is
_KIMI_K3_TARGET_LAYER_PREFIX = "language_model.model.layers."
_KIMI_K3_DRAFT_LAYER_PREFIX = "model.layers."


def _make_kimi_cache_groups_v034(
    full_history_specs: Mapping[str, KVCacheSpec],
    recurrent_specs: Mapping[str, MambaSpec],
    recurrent_layers: list[list[str]],
    *,
    uniform_recurrent: bool,
) -> list[KVCacheGroup] | None:
    from vllm.v1.core.kv_cache_planning import create_kv_cache_group_specs
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs

    full_history = UniformTypeKVCacheSpecs.from_specs(dict(full_history_specs))
    if full_history is None:
        return None
    groups = [KVCacheGroupSpec(list(full_history_specs), full_history)]
    if uniform_recurrent:
        for layer_names in recurrent_layers:
            group_specs = {name: recurrent_specs[name] for name in layer_names}
            uniform = UniformTypeKVCacheSpecs.from_specs(group_specs)
            assert uniform is not None
            groups.append(KVCacheGroupSpec(layer_names, uniform))
    elif recurrent_layers:
        groups.extend(create_kv_cache_group_specs(dict(recurrent_specs), recurrent_layers))
    return groups


def _make_kimi_cache_groups_from_layer_configs(
    full_history_configs: Mapping[str, Any],
    recurrent_configs: Mapping[str, Any],
    recurrent_layers: list[list[str]],
) -> list[KVCacheGroup] | None:
    if not full_history_configs:
        return None
    first = next(iter(full_history_configs.values()))
    if any(
        lc.manager_config != first.manager_config
        or lc.spec.block_size != first.spec.block_size
        or lc.spec.dcp_sharded != first.spec.dcp_sharded
        for lc in full_history_configs.values()
    ):
        return None
    groups = [KVCacheGroup.from_layers(full_history_configs)]
    for layer_names in recurrent_layers:
        groups.append(KVCacheGroup.from_layers({name: recurrent_configs[name] for name in layer_names}))
    return groups


def make_kimi_cache_groups(
    full_history_specs: Mapping[str, Any],
    recurrent_specs: Mapping[str, Any],
    *,
    recurrent_group_size: int,
    uniform_recurrent: bool,
) -> list[KVCacheGroup] | None:
    """Share full-history ownership and balance recurrent groups by slot budget.

    Callers validate roles and geometry, and supply specs in layer order with
    target attention first for scheduler conversion. The recurrent slot budget
    is explicit: K3 uses target plus draft pages, whereas K3.1 overlays only
    target MLA pages, excluding both indexer and draft storage.

    K3 retains per-layer recurrent specs for its shared-tuple allocation;
    K3.1 keeps native merged Mamba specs for its paired-page allocation.
    """
    names = list(recurrent_specs)
    count = cdiv(len(names), recurrent_group_size)
    recurrent_layers = [names[i::count] for i in range(count)]
    if vllm_version_is("0.34.0"):
        return _make_kimi_cache_groups_v034(
            full_history_specs,
            recurrent_specs,
            recurrent_layers,
            uniform_recurrent=uniform_recurrent,
        )
    return _make_kimi_cache_groups_from_layer_configs(
        full_history_specs,
        recurrent_specs,
        recurrent_layers,
    )


def _partition_kimi_k3_dspark_entries_v034(
    kv_cache_spec: Mapping[str, KVCacheSpec],
) -> tuple[dict[str, KVCacheSpec], dict[str, KVCacheSpec], dict[str, MambaSpec], dict[str, KVCacheSpec]]:
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    target_attention = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and isinstance(spec, FullAttentionSpec)
    }
    draft_attention = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_DRAFT_LAYER_PREFIX) and isinstance(spec, FullAttentionSpec)
    }
    mamba = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and isinstance(spec, MambaSpec)
    }
    return target_attention, draft_attention, mamba, dict(kv_cache_spec)


def _partition_kimi_k3_dspark_layer_configs(
    layer_configs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, KVCacheSpec]]:
    target_attention = {
        name: lc
        for name, lc in layer_configs.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and issubclass(lc.manager_cls, FullAttentionManager)
    }
    draft_attention = {
        name: lc
        for name, lc in layer_configs.items()
        if name.startswith(_KIMI_K3_DRAFT_LAYER_PREFIX) and issubclass(lc.manager_cls, FullAttentionManager)
    }
    mamba = {
        name: lc
        for name, lc in layer_configs.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and issubclass(lc.manager_cls, MambaManager)
    }
    specs = {name: lc.spec for name, lc in layer_configs.items()}
    return target_attention, draft_attention, mamba, specs


def _get_kimi_k3_dspark_mixed_kv_cache_groups(
    kv_cache_spec: Mapping[str, Any],
) -> list[KVCacheGroup] | None:
    """Build topology-independent Kimi K3 DSpark scheduler groups.

    Target and causal draft attention layers require the same full-sequence
    block ownership. Putting them in one group lets them share one scheduler
    block table while preserving a separate physical page per layer.
    Recurrent layers are split into the fewest balanced groups whose size does
    not exceed the attention group. This minimizes scheduler groups while
    keeping the recurrent groups balanced.

    Block and page sizes are resolved by the runtime and intentionally not
    fixed here: TP8 and TP16 produce different sizes but the same ownership
    relation. An unrecognized or incompatible signature falls back to vLLM's
    generic hybrid grouping.
    """
    if vllm_version_is("0.34.0"):
        target_attention, draft_attention, mamba, specs = _partition_kimi_k3_dspark_entries_v034(kv_cache_spec)
    else:
        target_attention, draft_attention, mamba, specs = _partition_kimi_k3_dspark_layer_configs(kv_cache_spec)

    matched_layer_count = len(target_attention) + len(draft_attention) + len(mamba)
    if (
        not target_attention
        or not draft_attention
        or not mamba
        or matched_layer_count != len(kv_cache_spec)
    ):
        return None

    attention_specs = [specs[name] for name in (*target_attention, *draft_attention)]
    all_specs = [*attention_specs, *(specs[name] for name in mamba)]
    # Only attention layers share a scheduler block table. Mamba keeps its
    # own groups and may use max_model_len as block_size when cache_mode=none.
    if (
        len({spec.block_size for spec in attention_specs}) != 1
        or len({spec.dcp_sharded for spec in attention_specs}) != 1
        or len({spec.page_size_bytes for spec in all_specs}) != 1
    ):
        return None

    first_mamba_entry = next(iter(mamba.values()))
    if any(entry != first_mamba_entry for entry in mamba.values()):
        return None

    # Insert target attention first. generate_scheduler_kv_cache_config unwraps a
    # UniformType group to its first spec, and this representative is registered
    # with the FullAttentionManager needed by both target and draft attention.
    mixed_attention = {**target_attention, **draft_attention}
    groups = make_kimi_cache_groups(
        mixed_attention,
        mamba,
        recurrent_group_size=len(mixed_attention),
        uniform_recurrent=True,
    )
    if groups is None:
        return None

    logger.info(
        "Using Kimi K3 DSpark mixed KV grouping: %d target + %d draft attention layers, followed by Mamba groups %s",
        len(target_attention),
        len(draft_attention),
        [len(group.layer_names) for group in groups[1:]],
    )
    return groups
