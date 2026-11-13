# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Shared full-history and recurrent grouping for Kimi K3 and K3.1."""

from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.utils import vllm_version_is

# vLLM main moved this helper to kv_cache_planning after v0.32.0.
if vllm_version_is("0.32.0"):
    from vllm.v1.core.kv_cache_utils import create_kv_cache_group_specs
else:
    from vllm.v1.core.kv_cache_planning import create_kv_cache_group_specs


_KIMI_K3_TARGET_LAYER_PREFIX = "language_model.model.layers."
_KIMI_K3_DRAFT_LAYER_PREFIX = "model.layers."


def make_kimi_cache_groups(
    full_history_specs: dict[str, KVCacheSpec],
    recurrent_specs: dict[str, MambaSpec],
    *,
    recurrent_group_size: int,
    uniform_recurrent: bool,
) -> list[KVCacheGroupSpec] | None:
    """Share full-history ownership and balance recurrent groups by slot budget.

    Callers validate roles and geometry, and supply specs in layer order with
    target attention first for scheduler conversion. The recurrent slot budget
    is explicit: K3 uses target plus draft pages, whereas K3.1 overlays only
    target MLA pages, excluding both indexer and draft storage.

    K3 retains per-layer recurrent specs for its shared-tuple allocation;
    K3.1 keeps native merged Mamba specs for its paired-page allocation.
    """
    full_history = UniformTypeKVCacheSpecs.from_specs(full_history_specs)
    if full_history is None:
        return None
    groups = [KVCacheGroupSpec(list(full_history_specs), full_history)]
    names = list(recurrent_specs)
    count = cdiv(len(names), recurrent_group_size)
    recurrent_layers = [names[i::count] for i in range(count)]
    if uniform_recurrent:
        for layer_names in recurrent_layers:
            group_specs = {name: recurrent_specs[name] for name in layer_names}
            uniform = UniformTypeKVCacheSpecs.from_specs(group_specs)
            assert uniform is not None
            groups.append(KVCacheGroupSpec(layer_names, uniform))
    elif recurrent_layers:
        groups.extend(create_kv_cache_group_specs(recurrent_specs, recurrent_layers))
    return groups


def _get_kimi_k3_dspark_mixed_kv_cache_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
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
    target_attention_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and isinstance(spec, FullAttentionSpec)
    }
    draft_attention_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_DRAFT_LAYER_PREFIX) and isinstance(spec, FullAttentionSpec)
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
