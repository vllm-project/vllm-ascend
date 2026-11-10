# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import math
from dataclasses import replace
from typing import Any

# Both supported versions use the KV-cache planner API from #61183/#42309.
import vllm.v1.core.kv_cache_planning as _kv_cache_planning
import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheSpecKind,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
)

from vllm_ascend.core.kv_cache_config_builder import _is_deepseek_v4_groups
from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable
from vllm_ascend.models.deepseek_v41.cache_config import (
    is_deepseek_v41_cache,
)
from vllm_ascend.models.glm5next.cache_config import (
    _get_glm5_next_cache_layout,
    get_glm5_next_max_memory_usage,
)
from vllm_ascend.models.kimi_k3.paired_sfa_cache import paired_sfa_max_memory
from vllm_ascend.models.qwen4_exp.cache_config import (
    get_qwen4_exp_max_memory_usage,
)
from vllm_ascend.quantization.methods.kv_cache.turboquant import cache as tq_cache

_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes
_orig_annotate_eagle_groups = getattr(_kv_cache_planning, "_annotate_eagle_groups", None)
_orig_max_memory_usage_bytes_from_groups = _kv_cache_planning._max_memory_usage_bytes_from_groups
if UniformTypeKVCacheSpecs.max_num_blocks_per_req is KVCacheSpec.max_num_blocks_per_req:

    def _uniform_type_max_num_blocks_per_req(
        self: UniformTypeKVCacheSpecs,
        vllm_config: VllmConfig,
        max_len: int,
    ) -> int:
        """Preserve the inner spec's block-table width."""
        widths = {spec.max_num_blocks_per_req(vllm_config, max_len) for spec in self.kv_cache_specs.values()}
        assert len(widths) == 1, (
            "All layers in the same KV cache group must need the same number "
            f"of block table entries, got {sorted(widths)}."
        )
        return next(iter(widths))

    UniformTypeKVCacheSpecs.max_num_blocks_per_req = (  # type: ignore[method-assign]
        _uniform_type_max_num_blocks_per_req
    )


def _page_sizes(spec: UniformTypeKVCacheSpecs) -> set[int]:
    """Distinct page sizes across a group's cache specs.

    vLLM #51718 removed ``UniformTypeKVCacheSpecs.get_page_sizes()`` on main;
    the underlying ``kv_cache_specs``/``page_size_bytes`` are unchanged on both
    lanes, so compute it inline.
    """
    return {s.page_size_bytes for s in spec.kv_cache_specs.values()}


def _ascend_resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Ascend-compatible resolve_kv_cache_block_sizes.

    vLLM PR #40860 added a restriction that hybrid KV cache groups with
    multiple block sizes do not support DCP.
    This restriction is correct for CUDA but not for Ascend, which implements
    context parallelism for MLA and SWA-MLA layers independently.

    For multiple KV cache groups with DCP, compute scheduler_block_size as
    lcm(group_block_sizes) * dcp to maintain alignment.
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    groups = kv_cache_config.kv_cache_groups
    cacheable_groups = [group for group in groups if is_prefix_cacheable(group.kv_cache_spec)]
    filtered_private_groups = bool(cacheable_groups) and len(cacheable_groups) != len(groups)
    if filtered_private_groups and not is_deepseek_v41_cache(groups):
        # A fixed tail block is not a token-page scheduling or hashing unit.
        # Pool alignment is enforced by the GLM planner and prefix coordinator.
        kv_cache_config = replace(kv_cache_config, kv_cache_groups=cacheable_groups)
        groups = cacheable_groups

    cacheable_groups = [g for g in groups if is_prefix_cacheable(g.kv_cache_spec)]
    if len(cacheable_groups) != len(groups):
        scheduler_block_size = math.lcm(*(g.kv_cache_spec.block_size for g in groups)) * dcp
        if not cache_config.enable_prefix_caching or not cacheable_groups:
            return scheduler_block_size, scheduler_block_size
        filtered = replace(kv_cache_config, kv_cache_groups=cacheable_groups)
        if dcp == 1:
            _, hash_block_size = _orig_resolve_kv_cache_block_sizes(filtered, vllm_config)
        else:
            hash_block_size = math.gcd(*(g.kv_cache_spec.block_size for g in cacheable_groups))
        return scheduler_block_size, hash_block_size

    if len(groups) <= 1:
        # EngineCore's global size is the minimum across all original groups.
        # After filtering a private tail, it can therefore be smaller than the
        # sole cacheable group's token-page size.
        block_size = groups[0].kv_cache_spec.block_size if filtered_private_groups else cache_config.block_size
        bs = block_size * dcp
        return bs, bs

    group_block_sizes = [group.kv_cache_spec.block_size for group in groups]
    if dcp != 1:
        # Ascend supports CP with multiple KV cache groups; compute
        # scheduler_block_size using the LCM of all group block sizes
        # multiplied by DCP for proper alignment.
        scheduler_block_size = math.lcm(*group_block_sizes) * dcp
        if not cache_config.enable_prefix_caching:
            return scheduler_block_size, scheduler_block_size
        hash_block_size = math.gcd(*group_block_sizes)
        return scheduler_block_size, hash_block_size

    return _orig_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)


def _kv_cache_config_has_mamba_layers(self: KVCacheConfig) -> bool:
    """Recognize Mamba layers nested in UniformType cache groups."""
    return any(get_kv_cache_spec_kind(group.kv_cache_spec) == KVCacheSpecKind.MAMBA for group in self.kv_cache_groups)


def _get_max_layers_per_page_size(spec: UniformTypeKVCacheSpecs) -> int:
    """Bridge the UniformTypeKVCacheSpecs helper renamed by vLLM #53896."""
    return spec.get_max_layers_per_page_size()


def _uses_qwen4_exp_mtp(vllm_config: VllmConfig) -> bool:
    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        return False
    draft_model_config = getattr(
        speculative_config,
        "draft_model_config",
        None,
    )
    architectures = tuple(getattr(draft_model_config, "architectures", ()) or ())
    if "Qwen4ExpMTP" in architectures:
        return True
    text_config = getattr(draft_model_config, "hf_text_config", None)
    text_architectures = tuple(getattr(text_config, "architectures", ()) or ())
    return getattr(text_config, "model_type", None) == "qwen4_exp_mtp" or ("Qwen4ExpMTP" in text_architectures)


def _ascend_annotate_eagle_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    kv_cache_groups: list[KVCacheGroupSpec],
    use_trailing_layer_fallback: bool = False,
) -> None:
    """Annotate Qwen4Exp's actual MTP cache group, not every group.

    Qwen4Exp's MTP attention spec is otherwise indistinguishable from the
    target's spec. Its predictor is registered at the target layer count, so
    use that model-scoped position to find only the group containing the real
    draft layer. Leaving every group unannotated triggers vLLM's flag-all
    fallback, which incorrectly treats target Mamba groups as draft state and
    disables cross-request prefix reuse.
    """
    if _orig_annotate_eagle_groups is not None:
        _orig_annotate_eagle_groups(
            vllm_config,
            kv_cache_spec,
            kv_cache_groups,
            use_trailing_layer_fallback=use_trailing_layer_fallback,
        )
    speculative_config = vllm_config.speculative_config
    if (
        speculative_config is None
        or not speculative_config.use_eagle_block_drop()
        or not _uses_qwen4_exp_mtp(vllm_config)
    ):
        return

    text_config = vllm_config.model_config.hf_text_config
    first_draft_layer = int(text_config.num_hidden_layers)
    num_draft_layers = int(
        getattr(
            getattr(
                speculative_config.draft_model_config,
                "hf_text_config",
                None,
            ),
            "mtp_num_hidden_layers",
            1,
        )
        or 1
    )
    draft_markers = tuple(
        f".layers.{layer_idx}."
        for layer_idx in range(
            first_draft_layer,
            first_draft_layer + num_draft_layers,
        )
    )
    for group in kv_cache_groups:
        if any(marker in f".{layer_name}." for marker in draft_markers for layer_name in group.layer_names):
            group.is_eagle_group = True

    if not any(group.is_eagle_group for group in kv_cache_groups):
        raise ValueError(f"Qwen4Exp MTP is enabled, but no KV cache group contains its draft layer(s) {draft_markers}.")


def _ascend_max_memory_usage_bytes_from_groups(
    builder: Any,
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Keep Ascend admission accounting with the upstream builder API."""
    if vllm_config.model_config.hf_config.model_type == "kimi_k31":
        return paired_sfa_max_memory(vllm_config, kv_cache_groups)
    if tq_cache.uses_turboquant_groups(kv_cache_groups):
        return tq_cache.max_memory_usage(vllm_config, kv_cache_groups)
    if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        return get_glm5_next_max_memory_usage(vllm_config, kv_cache_groups)
    qwen_bytes = get_qwen4_exp_max_memory_usage(vllm_config, kv_cache_groups)
    if qwen_bytes is not None:
        return qwen_bytes
    if not _is_deepseek_v4_groups(kv_cache_groups):
        return _orig_max_memory_usage_bytes_from_groups(builder, vllm_config, kv_cache_groups)

    assert all(isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in kv_cache_groups)
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    layer_tuple_bytes = sum(_page_sizes(full_mla_spec))
    num_layer_tuples = max(
        _get_max_layers_per_page_size(group.kv_cache_spec)
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    )
    return sum(
        num_layer_tuples * group.kv_cache_spec.max_memory_usage_pages(vllm_config) * layer_tuple_bytes
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    )


vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
if _orig_annotate_eagle_groups is not None:
    _kv_cache_planning._annotate_eagle_groups = _ascend_annotate_eagle_groups
KVCacheConfig.has_mamba_layers = property(  # type: ignore[assignment]
    _kv_cache_config_has_mamba_layers
)
_kv_cache_planning._max_memory_usage_bytes_from_groups = _ascend_max_memory_usage_bytes_from_groups

# Also patch the reference used by engine/core.py which imports the function directly.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
