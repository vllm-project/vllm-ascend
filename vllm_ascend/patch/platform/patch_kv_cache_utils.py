# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from typing import Any

# Both supported versions use the KV-cache planner and builder API from #61183/#42309.
import vllm.v1.core.kv_cache_planning as _kv_cache_planning
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_planning import may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheSpecKind,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
)

from vllm_ascend.models.deepseek_v4.cache_config import (
    _get_kv_cache_groups_uniform_groups,
    _page_sizes,
    group_and_unify_kv_cache_specs,
)
from vllm_ascend.models.deepseek_v4.cache_config import (
    deepseek_v4_pool_bytes_per_block as _deepseek_v4_pool_bytes_per_block,
)
from vllm_ascend.models.deepseek_v4.cache_config import (
    get_deepseek_v4_cache_layout as _get_deepseek_v4_cache_layout,
)
from vllm_ascend.models.deepseek_v4.cache_config import (
    is_deepseek_v4_groups as _is_deepseek_v4_groups,
)
from vllm_ascend.models.deepseek_v41.cache_config import (
    get_deepseek_v41_kv_cache_config,
    get_deepseek_v41_pool_bytes_per_block,
    group_cache_specs,
    is_deepseek_v41_cache,
    make_cache_groups,
)
from vllm_ascend.models.glm5next.cache_config import (
    _get_glm5_next_cache_layout,
    get_glm5_next_kv_cache_config,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_max_memory_usage,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.models.glm5next.kv_cache import is_glm5_next_cache_spec
from vllm_ascend.models.kimi_k3.cache_config import _get_kimi_k3_dspark_mixed_kv_cache_groups
from vllm_ascend.models.kimi_k3.paired_sfa_cache import (
    paired_sfa_cache_config,
    paired_sfa_cache_groups,
    paired_sfa_max_memory,
    paired_sfa_pool_bytes,
)
from vllm_ascend.models.qwen4_exp.cache_config import (
    get_qwen4_exp_kv_cache_config,
    get_qwen4_exp_kv_cache_groups,
    get_qwen4_exp_max_memory_usage,
    get_qwen4_exp_pool_bytes_per_block,
)
from vllm_ascend.quantization.methods.kv_cache.turboquant import cache as tq_cache
from vllm_ascend.utils import vllm_version_is

_orig_get_kv_cache_groups_uniform_page_size = _kv_cache_planning._get_kv_cache_groups_uniform_page_size
_orig_get_kv_cache_groups = _kv_cache_planning.get_kv_cache_groups
_orig_get_packed_kv_cache_groups = _kv_cache_planning._get_packed_kv_cache_groups
_orig_annotate_eagle_groups = getattr(_kv_cache_planning, "_annotate_eagle_groups", None)
_orig_get_kv_cache_config_from_groups = _kv_cache_planning.get_kv_cache_config_from_groups
_orig_max_memory_usage_bytes_from_groups = _kv_cache_planning._max_memory_usage_bytes_from_groups
_orig_pool_bytes_per_block = _kv_cache_planning._pool_bytes_per_block
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


def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    kimi_k3_groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(kv_cache_spec)
    if kimi_k3_groups is not None:
        return kimi_k3_groups
    return _orig_get_kv_cache_groups_uniform_page_size(kv_cache_spec)


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


def _ascend_get_packed_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Preserve Ascend's DSV4 grouping on the live packed-group hook."""
    if is_deepseek_v41_cache(kv_cache_spec):
        grouped_specs = group_cache_specs(kv_cache_spec)
        groups = make_cache_groups(grouped_specs)
    else:
        grouped_specs = group_and_unify_kv_cache_specs(kv_cache_spec)
        if grouped_specs is None:
            assert _orig_get_packed_kv_cache_groups is not None
            return _orig_get_packed_kv_cache_groups(vllm_config, kv_cache_spec)
        groups = _get_kv_cache_groups_uniform_groups(grouped_specs)
    return groups


def _get_kv_cache_config_deepseek_v4_main(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    if tq_cache.uses_turboquant_groups(kv_cache_groups):
        return tq_cache.cache_config(vllm_config, kv_cache_groups, available_memory)
    layout = _get_deepseek_v4_cache_layout(kv_cache_groups)

    bytes_per_tuple = layout.bytes_per_tuple
    num_blocks = available_memory // (bytes_per_tuple * layout.num_layer_tuples)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    tuple_stride = bytes_per_tuple * num_blocks
    backing_size = tuple_stride * layout.num_layer_tuples

    # Within every tuple slot, page-size buckets are placed consecutively.
    page_offsets: dict[int, int] = {}
    page_prefix = 0
    for page_size in layout.page_sizes:
        page_offsets[page_size] = page_prefix * num_blocks
        page_prefix += page_size

    tensors: list[KVCacheTensor] = []

    # Keep each descriptor inside one cache group. Descriptors from different
    # groups alias corresponding tuple slots by using identical geometry.
    for group_buckets in layout.bucketed:
        for page_size in layout.page_sizes:
            layer_names = group_buckets.get(page_size)
            if not layer_names:
                continue

            tensors.append(
                KVCacheTensor(
                    size=backing_size,
                    layers=list(layer_names),
                    offset=page_offsets[page_size],
                    layer_stride=tuple_stride,
                    block_stride=page_size,
                )
            )

    # MTP layers receive trailing tuple slots. Unused page buckets remain
    # padding so num_blocks and memory accounting retain the existing contract.
    normal_tuple_slots = layout.num_layer_tuples - len(layout.mtp_layer_names)
    for index, layer_name in enumerate(layout.mtp_layer_names):
        slot = normal_tuple_slots + index
        tensors.append(
            KVCacheTensor(
                size=backing_size,
                layers=[layer_name],
                offset=slot * tuple_stride + page_offsets[layout.mtp_page_size],
                layer_stride=0,
                block_stride=layout.mtp_page_size,
            )
        )

    return num_blocks, tensors


def _ascend_pool_bytes_per_block(  # type: ignore[misc]
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Use the same DSV4 divisor as Ascend's shared-tuple planner.

    vLLM #51718 re-plans ranks with more KV memory using
    ``min_num_blocks * _pool_bytes_per_block(groups)``. Its standardized
    per-group layout has a different divisor from Ascend's DSV4 shared-tuple
    layout, so using the upstream value changes ``num_blocks`` during the
    re-plan and leaves ranks inconsistent.
    """
    if vllm_config.model_config.hf_config.model_type == "kimi_k31":
        return paired_sfa_pool_bytes(kv_cache_groups)
    if tq_cache.uses_turboquant_groups(kv_cache_groups):
        return tq_cache.pool_bytes_per_block(kv_cache_groups)
    if is_deepseek_v41_cache(kv_cache_groups):
        return get_deepseek_v41_pool_bytes_per_block(kv_cache_groups)
    if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        return get_glm5_next_pool_bytes_per_block(kv_cache_groups)
    qwen_bytes = get_qwen4_exp_pool_bytes_per_block(kv_cache_groups)
    if qwen_bytes is not None:
        return qwen_bytes
    if not _is_deepseek_v4_groups(kv_cache_groups):
        return _orig_pool_bytes_per_block(vllm_config, kv_cache_groups)

    return _deepseek_v4_pool_bytes_per_block(kv_cache_groups)


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


def _get_ascend_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    if vllm_config.model_config.hf_config.model_type == "kimi_k31":
        return paired_sfa_cache_groups(vllm_config, kv_cache_spec)
    if any(is_glm5_next_cache_spec(spec) for spec in kv_cache_spec.values()):
        return get_glm5_next_kv_cache_groups(vllm_config, kv_cache_spec)
    groups = get_qwen4_exp_kv_cache_groups(vllm_config, kv_cache_spec, _orig_get_kv_cache_groups)
    if groups is not None:
        return groups
    return _orig_get_kv_cache_groups(vllm_config, kv_cache_spec)


def _ascend_get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Restore Ascend-specific cache planners while retaining transfer metadata.

    Attach ``kv_transfer_config`` onto every built config: ``KVCacheConfig``
    has no native field, and the PD prefill-producer role read back by the
    kv-cache coordinator (see ``patch_kv_cache_coordinator.py``) must ride on
    the config object - the coordinator factory receives ``KVCacheConfig``
    but not ``VllmConfig``. The attribute survives the scheduler-side
    ``copy.deepcopy`` in ``generate_scheduler_kv_cache_config`` and is
    dropped by worker pickle IPC, which never reads it.
    """
    qwen_config = (
        None
        if vllm_version_is("0.28.0")
        else get_qwen4_exp_kv_cache_config(vllm_config, kv_cache_groups, available_memory)
    )
    if vllm_config.model_config.hf_config.model_type == "kimi_k31":
        kv_cache_config = paired_sfa_cache_config(vllm_config, kv_cache_groups, available_memory)
    elif qwen_config is not None:
        kv_cache_config = qwen_config
    elif is_deepseek_v41_cache(kv_cache_groups):
        kv_cache_config = get_deepseek_v41_kv_cache_config(vllm_config, kv_cache_groups, available_memory)
    elif _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        kv_cache_config = get_glm5_next_kv_cache_config(vllm_config, kv_cache_groups, available_memory)
    elif not _is_deepseek_v4_groups(kv_cache_groups):
        kv_cache_config = _orig_get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory)
    else:
        num_blocks, kv_cache_tensors = _get_kv_cache_config_deepseek_v4_main(
            vllm_config,
            kv_cache_groups,
            available_memory,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
        )
    kv_cache_config.kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is not None and speculative_config.method == "dspark":
        from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import resident_mla_context_group_ids
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_layout import draft_only_group_ids

        # Upstream scheduler conversion keeps only the first spec in a uniform
        # group, losing both residency and per-layer is_draft ownership. Keep
        # the PD context and external prefix-store classifications separate:
        # a mixed group needs draft block IDs AND ordinary target hit keys.
        # Remove these snapshots once upstream preserves per-layer ownership.
        kv_cache_config.dspark_context_group_ids = resident_mla_context_group_ids(kv_cache_config.kv_cache_groups)
        kv_cache_config.dspark_draft_only_group_ids = draft_only_group_ids(kv_cache_config.kv_cache_groups)
    return kv_cache_config


assert _orig_get_packed_kv_cache_groups is not None
if _orig_annotate_eagle_groups is not None:
    _kv_cache_planning._annotate_eagle_groups = _ascend_annotate_eagle_groups
_kv_cache_planning._get_packed_kv_cache_groups = _ascend_get_packed_kv_cache_groups
_kv_cache_planning._get_kv_cache_groups_uniform_page_size = _get_kv_cache_groups_uniform_page_size
_kv_cache_planning.get_kv_cache_groups = _get_ascend_kv_cache_groups
KVCacheConfig.has_mamba_layers = property(  # type: ignore[assignment]
    _kv_cache_config_has_mamba_layers
)
_kv_cache_planning.get_kv_cache_config_from_groups = _ascend_get_kv_cache_config_from_groups
_kv_cache_planning._max_memory_usage_bytes_from_groups = _ascend_max_memory_usage_bytes_from_groups
_kv_cache_planning._pool_bytes_per_block = _ascend_pool_bytes_per_block
