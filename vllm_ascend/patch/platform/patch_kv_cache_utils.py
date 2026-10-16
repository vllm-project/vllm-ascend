# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import math
from collections import defaultdict
from dataclasses import replace

import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_utils import _approximate_gcd, may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheSpecKind,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
)

from vllm_ascend.attention.cache_layout import LAYERWISE_PREFILL_WORKSPACE_ARENA_COUNT
from vllm_ascend.attention.dsa_offload_runtime import build_dsa_group_specs
from vllm_ascend.attention.offload_capability import (
    get_dsa_offload_backends,
    is_speculative_cache_layer,
)
from vllm_ascend.ops.dsa_offload import (
    DsaOffloadConfig,
    get_dsa_config_value,
    get_dsa_raw_seq,
)
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.attention.cache_layout import (
    CacheComponentRole,
    build_mla_layer_cache_descriptor,
)

from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable
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
from vllm_ascend.quantization.methods.kv_cache.turboquant import TURBOQUANT_CACHE_DTYPE
from vllm_ascend.quantization.methods.kv_cache.turboquant import cache as tq_cache

_KIMI_K3_TARGET_LAYER_PREFIX = "language_model.model.layers."
_KIMI_K3_DRAFT_LAYER_PREFIX = "model.layers."
_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes
_orig_get_kv_cache_groups_uniform_page_size = vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_page_size
_orig_get_kv_cache_groups = vllm.v1.core.kv_cache_utils.get_kv_cache_groups
_orig_get_packed_kv_cache_groups = vllm.v1.core.kv_cache_utils._get_packed_kv_cache_groups
_orig_get_kv_cache_configs = vllm.v1.core.kv_cache_utils.get_kv_cache_configs
_orig_get_kv_cache_config_from_groups = vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups
_orig_max_memory_usage_bytes_from_groups = vllm.v1.core.kv_cache_utils._max_memory_usage_bytes_from_groups
_orig_pool_bytes_per_block = vllm.v1.core.kv_cache_utils._pool_bytes_per_block
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


def is_layerwise_host_offload_prefill(vllm_config: VllmConfig) -> bool:
    """Return whether this worker uses the Prefill shared-workspace plan."""

    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None or kv_transfer_config.kv_role != "kv_producer":
        return False
    extra_config = kv_transfer_config.kv_connector_extra_config or {}
    return bool(extra_config.get("layerwise_host_kv_offload", False))


def _is_mtp_cache_layer(
    vllm_config: VllmConfig,
    group: KVCacheGroupSpec,
    layer_name: str,
) -> bool:
    """Identify speculative layers using shared offload role resolution."""

    model_config = vllm_config.model_config
    num_hidden_layers = get_dsa_config_value(model_config, "num_hidden_layers")
    return is_speculative_cache_layer(
        layer_name,
        num_hidden_layers=num_hidden_layers,
        is_eagle_group=bool(group.is_eagle_group),
        group_layer_count=len(group.layer_names),
    )


def _get_group_layer_kv_cache_specs(
    group: KVCacheGroupSpec,
) -> dict[str, KVCacheSpec]:
    """Expand vLLM group specs into a per-layer mapping.

    vLLM keeps per-layer specs in ``UniformTypeKVCacheSpecs`` when layer
    sizes differ, but uses one merged attention spec when all layers are
    equivalent.  Both representations are valid group contracts.
    """

    group_spec = group.kv_cache_spec
    if isinstance(group_spec, UniformTypeKVCacheSpecs):
        missing = set(group.layer_names) - set(group_spec.kv_cache_specs)
        if missing:
            raise ValueError(
                f"KV cache group is missing specs for layers: {sorted(missing)!r}"
            )
        return {
            layer_name: group_spec.kv_cache_specs[layer_name]
            for layer_name in group.layer_names
        }
    if isinstance(group_spec, MLAAttentionSpec):
        return {layer_name: group_spec for layer_name in group.layer_names}
    raise NotImplementedError(
        "Layerwise Host offload requires UniformTypeKVCacheSpecs or a merged "
        f"MLAAttentionSpec, got {type(group_spec).__name__}"
    )


def _get_layerwise_prefill_cache_layout(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> tuple[list[str], int, dict[str, int], dict[str, KVCacheSpec]]:
    """Resolve the physical Prefill cache layout shared by all planners."""

    if len(kv_cache_groups) != 1:
        raise NotImplementedError(
            "Layerwise Host Offload Prefill currently requires one uniform "
            "KV cache group"
        )
    group = kv_cache_groups[0]
    per_layer_specs = _get_group_layer_kv_cache_specs(group)
    workspace_layers: list[str] = []
    mtp_layers: list[str] = []
    for layer_name in group.layer_names:
        if _is_mtp_cache_layer(vllm_config, group, layer_name):
            mtp_layers.append(layer_name)
        else:
            workspace_layers.append(layer_name)
    if not workspace_layers:
        raise ValueError("Layerwise Host Offload Prefill found no base-model layers")

    workspace_page_sizes = {
        per_layer_specs[layer_name].page_size_bytes
        for layer_name in workspace_layers
    }
    if len(workspace_page_sizes) != 1:
        raise NotImplementedError(
            "Layerwise Host Offload Prefill requires compatible base-layer "
            "cache page sizes; heterogeneous layers need separate arenas"
        )
    workspace_page_size = next(iter(workspace_page_sizes))
    mtp_page_sizes = {
        layer_name: per_layer_specs[layer_name].page_size_bytes
        for layer_name in mtp_layers
    }
    return (
        workspace_layers,
        workspace_page_size,
        mtp_page_sizes,
        per_layer_specs,
    )


def _get_layerwise_prefill_max_memory_usage_bytes(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Return HBM needed for one maximum-length layerwise request."""

    (
        _workspace_layers,
        workspace_page_size,
        mtp_page_sizes,
        per_layer_specs,
    ) = _get_layerwise_prefill_cache_layout(vllm_config, kv_cache_groups)
    num_blocks = max(
        cdiv(
            spec.max_memory_usage_bytes(vllm_config),
            spec.page_size_bytes,
        )
        for spec in per_layer_specs.values()
    )
    bytes_per_block = (
        LAYERWISE_PREFILL_WORKSPACE_ARENA_COUNT * workspace_page_size
        + sum(mtp_page_sizes.values())
    )
    return num_blocks * bytes_per_block


def _ascend_max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    # PD-separated producer and consumer workers use distinct planners.
    if is_layerwise_host_offload_decode(vllm_config):
        return _get_layerwise_decode_max_memory_usage_bytes(
            vllm_config, kv_cache_groups
        )
    if is_layerwise_host_offload_prefill(vllm_config):
        return _get_layerwise_prefill_max_memory_usage_bytes(
            vllm_config, kv_cache_groups
        )
    return _ascend_model_max_memory_usage_bytes(vllm_config, kv_cache_groups)


def _get_layerwise_prefill_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Plan two grouped-prefetch Main/Indexer banks plus ordinary MTP caches.

    The scheduler still owns the full logical block table.  ``shared_by`` only
    aliases base-model layers to a bounded set of physical tensors, so
    every arena has the complete logical block capacity.  Draft layers remain
    ordinary per-layer PA caches and are included in the per-block denominator.
    """

    (
        workspace_layers,
        workspace_page_size,
        mtp_page_sizes,
        _per_layer_specs,
    ) = _get_layerwise_prefill_cache_layout(vllm_config, kv_cache_groups)
    if len(workspace_layers) < LAYERWISE_PREFILL_WORKSPACE_ARENA_COUNT:
        raise ValueError(
            "Layerwise Host Offload Prefill grouped prefetch requires at least "
            f"{LAYERWISE_PREFILL_WORKSPACE_ARENA_COUNT} base layers"
        )
    bytes_per_block = (
        LAYERWISE_PREFILL_WORKSPACE_ARENA_COUNT * workspace_page_size
        + sum(mtp_page_sizes.values())
    )
    if bytes_per_block <= 0:
        raise ValueError("Layerwise Host Offload Prefill has an invalid page budget")

    num_blocks = available_memory // bytes_per_block
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    if num_blocks <= 0:
        raise ValueError(
            "No KV cache block fits the Layerwise Host Offload Prefill budget"
        )

    # v0.30 KVCacheTensor replaces `shared_by` with `layers` plus stride
    # metadata. A workspace arena is one physical buffer whose member layers
    # all alias it (layer_stride=0 mirrors the upstream reuse-slot contract),
    # and each MTP layer keeps its own private buffer.
    kv_cache_tensors = [
        KVCacheTensor(
            size=workspace_page_size * num_blocks,
            layers=workspace_layers[
                arena_id::LAYERWISE_PREFILL_WORKSPACE_ARENA_COUNT
            ],
            offset=0,
            layer_stride=0,
            block_stride=workspace_page_size,
        )
        for arena_id in range(LAYERWISE_PREFILL_WORKSPACE_ARENA_COUNT)
    ]
    kv_cache_tensors.extend(
        KVCacheTensor(
            size=mtp_page_sizes[layer_name] * num_blocks,
            layers=[layer_name],
            offset=0,
            layer_stride=mtp_page_sizes[layer_name],
            block_stride=mtp_page_sizes[layer_name],
        )
        for layer_name in mtp_page_sizes
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def is_layerwise_host_offload_decode(vllm_config: VllmConfig) -> bool:
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None or kv_transfer_config.kv_role != "kv_consumer":
        return False
    extra_config = kv_transfer_config.kv_connector_extra_config or {}
    return bool(extra_config.get("layerwise_host_kv_offload", False))


def _dsa_group_count(vllm_config: VllmConfig, num_hidden_layers: int) -> int:
    """Return the number of DSA metadata sets allocated by Model Runner."""
    indexer_types = get_dsa_config_value(
        vllm_config.model_config, "indexer_types"
    )
    return len(build_dsa_group_specs(indexer_types, num_hidden_layers))


def _dsa_workspace_bytes(
    vllm_config: VllmConfig,
    group: KVCacheGroupSpec,
    specs: dict[str, KVCacheSpec],
) -> int:
    """Return the fixed Decode DSA workspace reservation."""
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
    default_pool_size = (
        16384
        if bool(
            get_dsa_config_value(
                vllm_config.model_config, "enlarge_pool_size", False
            )
        )
        else 8192
    )
    pool_size = int(extra_config.get("dsa_pool_size", default_pool_size))
    id_range = max(
        int(vllm_config.model_config.max_model_len),
        int(extra_config.get("dsa_id_range", 131072)),
    )
    batch_capacity = int(vllm_config.scheduler_config.max_num_seqs)
    num_hidden_layers = int(
        get_dsa_config_value(vllm_config.model_config, "num_hidden_layers", 0)
    )
    if num_hidden_layers <= 0:
        raise ValueError("DSA workspace accounting requires num_hidden_layers")
    metadata_groups = _dsa_group_count(vllm_config, num_hidden_layers)
    speculative_config = getattr(vllm_config, "speculative_config", None)
    raw_seq = get_dsa_raw_seq(speculative_config)
    topk_value = get_dsa_config_value(vllm_config.model_config, "index_topk")
    if topk_value is None:
        raise ValueError("DSA workspace accounting requires index_topk")
    topk = int(topk_value)
    block_size = int(group.kv_cache_spec.block_size)
    if pool_size <= 0 or pool_size > 16384 or pool_size % 16 != 0:
        raise ValueError("dsa_pool_size must be in (0, 16384] and divisible by 16")
    dsa_config = DsaOffloadConfig(
        raw_seq=raw_seq,
        topk=topk,
        selection_block_size=block_size,
    )
    selection_rows = dsa_config.selection_rows(batch_capacity)
    selection_blocks = dsa_config.selection_block_count(batch_capacity)
    total = 0
    dsa_layers = [
        name
        for name in group.layer_names
        if not _is_mtp_cache_layer(vllm_config, group, name)
    ]
    if not dsa_layers:
        raise ValueError("Decode Host offload found no base DSA layers")
    backends = get_dsa_offload_backends(vllm_config, dsa_layers)
    for layer_name in dsa_layers:
        spec = specs[layer_name]
        memory = backends[layer_name].get_dsa_offload_cache_memory(
            layer_name=layer_name,
            kv_cache_spec=spec,
        )
        main_bytes_per_token = memory.main_bytes_per_token
        total += batch_capacity * pool_size * main_bytes_per_token
        total += selection_blocks * block_size * main_bytes_per_token
        total += selection_rows * 4
        # selection_default_indices is an expanded view of one [1, 1, topk]
        # arange allocation in Model Runner; count backing storage, not its
        # logical expanded numel.
        total += topk * 4
        total += selection_rows * dsa_config.selection_blocks_per_row * 4
    # DsaGroupMetadata is allocated once per full/shared indexer group.
    total += metadata_groups * (
        batch_capacity * pool_size * 4
        + batch_capacity * id_range * 4
        + batch_capacity * (pool_size // 16) * 4
    )
    return total


def _get_layerwise_decode_hbm_layout(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> tuple[KVCacheGroupSpec, dict[str, KVCacheSpec], int]:
    """Resolve the Decode device-resident bytes charged per scheduler block."""

    if len(kv_cache_groups) != 1:
        raise NotImplementedError(
            "Decode Layerwise Host offload requires exactly one KV cache group; "
            "hybrid groups are not supported in the first release"
        )
    group = kv_cache_groups[0]
    specs = _get_group_layer_kv_cache_specs(group)
    hbm_bytes_per_block = 0
    dsa_layers = [
        name
        for name in group.layer_names
        if not _is_mtp_cache_layer(vllm_config, group, name)
    ]
    if not dsa_layers:
        raise ValueError("Decode Layerwise Host offload has no base DSA layers")
    backends = get_dsa_offload_backends(vllm_config, dsa_layers)
    for layer_name in group.layer_names:
        spec = specs[layer_name]
        if _is_mtp_cache_layer(vllm_config, group, layer_name):
            hbm_bytes_per_block += spec.page_size_bytes
            continue
        memory = backends[layer_name].get_dsa_offload_cache_memory(
            layer_name=layer_name,
            kv_cache_spec=spec,
        )
        hbm_bytes_per_block += memory.device_bytes_per_block
    if hbm_bytes_per_block <= 0:
        raise ValueError("Decode Layerwise Host offload has no device Indexer cache")
    return group, specs, hbm_bytes_per_block


def _get_layerwise_decode_required_blocks(
    vllm_config: VllmConfig,
    specs: dict[str, KVCacheSpec],
) -> int:
    return max(
        cdiv(
            spec.max_memory_usage_bytes(vllm_config),
            spec.page_size_bytes,
        )
        for spec in specs.values()
    )


def _get_layerwise_decode_max_memory_usage_bytes(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Return Indexer/MTP cache plus fixed DSA HBM for max_model_len."""

    group, specs, hbm_bytes_per_block = _get_layerwise_decode_hbm_layout(
        vllm_config, kv_cache_groups
    )
    return _dsa_workspace_bytes(
        vllm_config, group, specs
    ) + _get_layerwise_decode_required_blocks(
        vllm_config, specs
    ) * hbm_bytes_per_block


def _get_layerwise_decode_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    group, specs, hbm_bytes_per_block = _get_layerwise_decode_hbm_layout(
        vllm_config, kv_cache_groups
    )
    dsa_bytes = _dsa_workspace_bytes(vllm_config, group, specs)
    # vLLM deliberately passes available_memory=0 while creating the minimal
    # KV cache used by CUDA/ACL graph profiling. In that call path
    # num_gpu_blocks_override is temporarily set to min_blocks, so zero is a
    # profiling sentinel rather than the real HBM budget. Do not subtract the
    # fixed DSA reservation from it or reject the request as an actual OOM.
    is_minimal_profile = (
        available_memory == 0
        and vllm_config.cache_config.num_gpu_blocks_override is not None
    )
    if is_minimal_profile:
        effective_memory = 0
        num_blocks = may_override_num_blocks(vllm_config, 0)
        if num_blocks <= 0:
            raise ValueError(
                "Minimal KV cache profiling requires a positive "
                "num_gpu_blocks_override"
            )
    else:
        effective_memory = available_memory - dsa_bytes
        if effective_memory <= 0:
            raise MemoryError(
                "Decode Layerwise Host offload cannot reserve fixed DSA workspace "
                f"({dsa_bytes} bytes) from {available_memory} bytes"
            )
        num_blocks = may_override_num_blocks(
            vllm_config, effective_memory // hbm_bytes_per_block
        )
    if num_blocks <= 0:
        raise ValueError("No KV cache block fits Decode Indexer-only HBM budget")
    if not is_minimal_profile and num_blocks * hbm_bytes_per_block > effective_memory:
        raise MemoryError(
            "Configured KV cache block override exceeds Decode HBM budget after "
            f"DSA workspace reservation ({dsa_bytes} bytes)"
        )
    base = _orig_get_kv_cache_config_from_groups(
        vllm_config,
        kv_cache_groups,
        max(effective_memory, group.kv_cache_spec.page_size_bytes),
    )
    old_blocks = base.num_blocks
    tensors = [
        replace(
            tensor,
            size=(tensor.size // old_blocks) * num_blocks,
        )
        for tensor in base.kv_cache_tensors
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=kv_cache_groups,
    )


def _ascend_get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    if is_layerwise_host_offload_decode(vllm_config):
        return _get_layerwise_decode_kv_cache_config(
            vllm_config, kv_cache_groups, available_memory
        )
    if is_layerwise_host_offload_prefill(vllm_config):
        return _get_layerwise_prefill_kv_cache_config(
            vllm_config,
            kv_cache_groups,
            available_memory,
        )
    return _ascend_model_get_kv_cache_config_from_groups(
        vllm_config,
        kv_cache_groups,
        available_memory,
    )


def _apply_decode_num_blocks_override(
    vllm_config: VllmConfig,
    kv_cache_configs: list[KVCacheConfig],
    override: int,
) -> None:
    """Validate an override against profiled HBM, then shrink allocations."""

    if override <= 0:
        raise ValueError("num_gpu_blocks_override must be positive")
    for config in kv_cache_configs:
        if override > config.num_blocks:
            raise MemoryError(
                "Configured KV cache block override exceeds the real Decode "
                f"Indexer/MTP HBM budget: override={override}, "
                f"capacity={config.num_blocks}"
            )
        if config.kv_cache_groups:
            _, specs, _ = _get_layerwise_decode_hbm_layout(
                vllm_config, config.kv_cache_groups
            )
            required_blocks = _get_layerwise_decode_required_blocks(
                vllm_config, specs
            )
            if override < required_blocks:
                raise ValueError(
                    "Configured KV cache block override cannot serve "
                    f"max_model_len: override={override}, "
                    f"required={required_blocks}"
                )

    for config in kv_cache_configs:
        old_blocks = config.num_blocks
        for tensor in config.kv_cache_tensors:
            if tensor.size % old_blocks != 0:
                raise ValueError(
                    "KV cache tensor size is not divisible by num_blocks"
                )
            tensor.size = tensor.size // old_blocks * override
        config.num_blocks = override


def _ascend_get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """Preserve real profiled HBM when Decode uses a block override.

    Upstream rewrites available_memory to override times full-KV bytes before
    calling the per-group planner. That virtual capacity is invalid for the
    Indexer-only Decode layout. Suppress the upstream rewrite, build against the
    real profiled budgets, then validate and apply the requested block count.
    """

    override = vllm_config.cache_config.num_gpu_blocks_override
    if not is_layerwise_host_offload_decode(vllm_config) or override is None:
        return _orig_get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_memory
        )

    vllm_config.cache_config.num_gpu_blocks_override = None
    try:
        configs = _orig_get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_memory
        )
    finally:
        vllm_config.cache_config.num_gpu_blocks_override = override
    _apply_decode_num_blocks_override(vllm_config, configs, override)
    return configs
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
    mixed_attention_spec = UniformTypeKVCacheSpecs.from_specs(mixed_attention_specs)
    if mixed_attention_spec is None:
        return None

    groups = [
        KVCacheGroupSpec(
            layer_names=list(mixed_attention_specs),
            kv_cache_spec=mixed_attention_spec,
        )
    ]
    mamba_layer_names = list(mamba_specs)
    mamba_group_count = cdiv(len(mamba_layer_names), len(mixed_attention_specs))
    for group_idx in range(mamba_group_count):
        layer_names = mamba_layer_names[group_idx::mamba_group_count]
        group_specs = {name: mamba_specs[name] for name in layer_names}
        uniform_mamba_spec = UniformTypeKVCacheSpecs.from_specs(group_specs)
        assert uniform_mamba_spec is not None
        groups.append(
            KVCacheGroupSpec(
                layer_names=layer_names,
                kv_cache_spec=uniform_mamba_spec,
            )
        )

    logger.info(
        "Using Kimi K3 DSpark mixed KV grouping: %d target + %d draft attention layers, followed by Mamba groups %s",
        len(target_attention_specs),
        len(draft_attention_specs),
        [len(group.layer_names) for group in groups[1:]],
    )
    return groups


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


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    if not any(isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()):
        return None

    logical_block_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    grouped_swa_mla_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[spec.block_size][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            logical_block_specs[spec.block_size][name] = spec

    mla_uniform_specs = []
    for block_size in sorted(logical_block_specs):
        spec_dict = logical_block_specs[block_size]
        assert len(spec_dict) > 0
        mla_uniform_specs.append(UniformTypeKVCacheSpecs.from_specs(spec_dict))
    assert mla_uniform_specs is not None

    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    return [*mla_uniform_specs, *swa_uniform_specs]


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """
    Generate the KV cache groups from the grouped specs.
    """
    assert len(grouped_specs) > 0 and all(isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs)
    if any(
        getattr(spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE
        for group in grouped_specs
        for spec in group.kv_cache_specs.values()
    ):
        return tq_cache.group_specs(grouped_specs)
    # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
    # containing only MLAAttentionSpec.
    full_mla_spec = grouped_specs[0]
    full_mla_c128_spec = grouped_specs[1]

    assert all(isinstance(spec, MLAAttentionSpec) for spec in full_mla_spec.kv_cache_specs.values())
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_spec,
    )
    full_mla_c128_group = KVCacheGroupSpec(
        layer_names=list(full_mla_c128_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_c128_spec,
    )

    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other uniform KV cache specs will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
    num_layer_tuples_per_group: list[int] = [_get_max_layers_per_page_size(g_spec) for g_spec in grouped_specs]
    # Choose `num_layer_tuples` to minimize total padding across groups.
    num_layer_tuples = _approximate_gcd(num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0])
    # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
    num_layer_tuples_per_group = [round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group]

    # TODO(cmq): this is not general enough
    swa_mla_specs = grouped_specs[2:]

    assert all(
        isinstance(spec, SlidingWindowMLASpec) for group in swa_mla_specs for spec in group.kv_cache_specs.values()
    )

    # Split each SWA UniformKV group into smaller groups to align their #(layer tuples)
    # Possibly padding layer tuples for this.
    # Additionally, we also pad KV blocks in each SWA layer, to align the page size
    # with the corresponding layer in the full-MLA group.
    all_page_sizes = _page_sizes(full_mla_spec)
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = _page_sizes(sm_spec)
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        assert max(sm_page_sizes) <= max(all_page_sizes)

        # Unify page size by padding layers' page_size to the nearest larger page_size.
        # Compute candidate (nearest larger page_size) for each unique page size.
        size_to_candidate: dict[int, int] = {}
        for ps in sm_page_sizes:
            size_to_candidate[ps] = min(x for x in all_page_sizes if x >= ps)
        # Pad and collect layer names per page size.
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            current_size = layer_spec.page_size_bytes
            candidate = size_to_candidate[current_size]
            if current_size < candidate:
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)
        # NOTE(yifan): for now, inside a UniformKV group, each page_size should
        # have the same number of layers. This also means we don't need to pad layers
        # inside a partial-full layer tuple.
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        # Split layers inside each UniformKV group for aligned #(layers).
        # See `_get_kv_cache_groups_uniform_page_size` for more details.
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            # Flatten tuples and build dict for from_specs
            group_layer_names = [name for layer_tuple in group_layer_tuples for name in layer_tuple]
            group_layer_specs = {name: sm_spec.kv_cache_specs[name] for name in group_layer_names}
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_sm_spec is not None
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )

    return [full_mla_group, full_mla_c128_group, *swa_mla_groups]


def _get_max_layers_per_page_size(spec: UniformTypeKVCacheSpecs) -> int:
    """Bridge the UniformTypeKVCacheSpecs helper renamed by vLLM #53896."""
    return spec.get_max_layers_per_page_size()


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
    vllm.v1.core.kv_cache_utils._annotate_eagle_groups(
        vllm_config,
        kv_cache_spec,
        groups,
        use_deepseek_v4_fallback=True,
    )
    vllm.v1.core.kv_cache_utils._warn_if_unannotated_eagle_mamba(
        vllm_config,
        groups,
    )
    return groups


def _get_deepseek_v4_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> tuple[list[int], list[dict[int, list[str]]], list[str], int, int]:
    """Return the geometry shared by DSV4 planning and rank normalization.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream (see _get_kv_cache_groups_uniform_groups) so
    every layer's page_size matches one of the full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each layer
    at tuple_idx = position-within-bucket.
    """
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    page_sizes = sorted(_page_sizes(full_mla_spec))

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    mtp_layer_names = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs = group.kv_cache_spec.kv_cache_specs
        b: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            if "mtp" not in name:
                b[specs[name].page_size_bytes].append(name)
            else:
                mtp_layer_names.append(name)
                mtp_page_size = specs[name].page_size_bytes
        bucketed.append(b)

    # num_layer_tuples = longest bucket list across all groups. For the
    # full-MLA group this equals the count of layers in the largest
    # per-page-size bucket (= get_num_layer_tuples()); for SWA sub-groups
    # this equals the sub-group size (each has a single page_size).
    num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values()) + len(mtp_layer_names)

    return page_sizes, bucketed, mtp_layer_names, mtp_page_size, num_layer_tuples


def _get_kv_cache_config_deepseek_v4_main(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    if tq_cache.uses_turboquant_groups(kv_cache_groups):
        return tq_cache.cache_config(vllm_config, kv_cache_groups, available_memory)
    (
        page_sizes,
        bucketed,
        mtp_layer_names,
        mtp_page_size,
        num_tuple_slots,
    ) = _get_deepseek_v4_cache_layout(kv_cache_groups)

    bytes_per_tuple = sum(page_sizes)
    num_blocks = available_memory // (bytes_per_tuple * num_tuple_slots)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    tuple_stride = bytes_per_tuple * num_blocks
    backing_size = tuple_stride * num_tuple_slots

    # Within every tuple slot, page-size buckets are placed consecutively.
    page_offsets: dict[int, int] = {}
    page_prefix = 0
    for page_size in page_sizes:
        page_offsets[page_size] = page_prefix * num_blocks
        page_prefix += page_size

    tensors: list[KVCacheTensor] = []

    # Keep each descriptor inside one cache group. Descriptors from different
    # groups alias corresponding tuple slots by using identical geometry.
    for group_buckets in bucketed:
        for page_size in page_sizes:
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
    normal_tuple_slots = num_tuple_slots - len(mtp_layer_names)
    for index, layer_name in enumerate(mtp_layer_names):
        slot = normal_tuple_slots + index
        tensors.append(
            KVCacheTensor(
                size=backing_size,
                layers=[layer_name],
                offset=slot * tuple_stride + page_offsets[mtp_page_size],
                layer_stride=0,
                block_stride=mtp_page_size,
            )
        )

    return num_blocks, tensors


def _is_deepseek_v4_groups(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
    if (
        is_deepseek_v41_cache(kv_cache_groups)
        or not kv_cache_groups
        or not all(isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in kv_cache_groups)
    ):
        return False
    for group in kv_cache_groups:
        group_spec = group.kv_cache_spec
        assert isinstance(group_spec, UniformTypeKVCacheSpecs)
        specs = group_spec.kv_cache_specs.values()
        if any(getattr(spec, "model_version", None) == "deepseek_v4" for spec in specs):
            return True
    return False


def _ascend_pool_bytes_per_block(kv_cache_groups: list[KVCacheGroupSpec]) -> int:
    """Use the same DSV4 divisor as Ascend's shared-tuple planner.

    vLLM #51718 re-plans ranks with more KV memory using
    ``min_num_blocks * _pool_bytes_per_block(groups)``. Its standardized
    per-group layout has a different divisor from Ascend's DSV4 shared-tuple
    layout, so using the upstream value changes ``num_blocks`` during the
    re-plan and leaves ranks inconsistent.
    """
    if tq_cache.uses_turboquant_groups(kv_cache_groups):
        return tq_cache.pool_bytes_per_block(kv_cache_groups)
    if is_deepseek_v41_cache(kv_cache_groups):
        return get_deepseek_v41_pool_bytes_per_block(kv_cache_groups)
    if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        return get_glm5_next_pool_bytes_per_block(kv_cache_groups)
    if not _is_deepseek_v4_groups(kv_cache_groups):
        return _orig_pool_bytes_per_block(kv_cache_groups)

    page_sizes, _, _, _, num_layer_tuples = _get_deepseek_v4_cache_layout(kv_cache_groups)
    return sum(page_sizes) * num_layer_tuples


def _ascend_model_max_memory_usage_bytes(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Keep the pre-#51718 DSV4 admission formula for its shared tuples."""
    if tq_cache.uses_turboquant_groups(kv_cache_groups):
        return tq_cache.max_memory_usage(vllm_config, kv_cache_groups)
    if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        return get_glm5_next_max_memory_usage(vllm_config, kv_cache_groups)
    if not _is_deepseek_v4_groups(kv_cache_groups):
        return _orig_max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)

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


def _get_glm5_next_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    if any(is_glm5_next_cache_spec(spec) for spec in kv_cache_spec.values()):
        return get_glm5_next_kv_cache_groups(vllm_config, kv_cache_spec)
    return _orig_get_kv_cache_groups(vllm_config, kv_cache_spec)


def _ascend_model_get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Restore Ascend's DSV4 shared-tuple planner removed by vLLM #51718.

    Attach ``kv_transfer_config`` onto every built config: ``KVCacheConfig``
    has no native field, and the PD prefill-producer role read back by the
    kv-cache coordinator (see ``patch_kv_cache_coordinator.py``) must ride on
    the config object - the coordinator factory receives ``KVCacheConfig``
    but not ``VllmConfig``. The attribute survives the scheduler-side
    ``copy.deepcopy`` in ``generate_scheduler_kv_cache_config`` and is
    dropped by worker pickle IPC, which never reads it.
    """
    if is_deepseek_v41_cache(kv_cache_groups):
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

        # Upstream scheduler conversion keeps only the first spec in a uniform
        # group, losing draft residency when target and draft share block IDs.
        # Preserve the worker-derived ownership through that deepcopy. Remove
        # this bridge once upstream KVCacheConfig carries per-layer ownership.
        kv_cache_config.dspark_context_group_ids = resident_mla_context_group_ids(kv_cache_config.kv_cache_groups)
        draft_only_group_ids = []
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            specs = (
                tuple(group_spec.kv_cache_specs.values())
                if isinstance(group_spec, UniformTypeKVCacheSpecs)
                else (group_spec,)
            )
            # Keep mixed target/draft groups in the ordinary target lookup.
            # Empty PP-local layer names still carry the global group spec.
            if specs and all(
                isinstance(spec, MLAAttentionSpec) and spec.non_causal_multi_token_decode for spec in specs
            ):
                draft_only_group_ids.append(group_id)
        kv_cache_config.dspark_draft_only_group_ids = tuple(draft_only_group_ids)
    return kv_cache_config


vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.core.kv_cache_utils.get_kv_cache_configs = _ascend_get_kv_cache_configs
vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups = (
    _ascend_get_kv_cache_config_from_groups
)
vllm.v1.core.kv_cache_utils._max_memory_usage_bytes_from_groups = (
    _ascend_max_memory_usage_bytes_from_groups
)
vllm.v1.core.kv_cache_utils.group_and_unify_kv_cache_specs = group_and_unify_kv_cache_specs
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_groups = _get_kv_cache_groups_uniform_groups
# vllm v0.24.0 renamed _get_kv_cache_config_deepseek_v4 to _get_kv_cache_config_packed and
# get_kv_cache_config_from_groups now calls _get_kv_cache_config_packed directly, bypassing
# the alias patch above. Patch the canonical name so Ascend's non-packed layout is used.
# The v0.30.0 base carries this planner as `_get_kv_cache_config_deepseek_v4_main`
# (with TurboQuant group support); alias it for the canonical upstream name.
_get_kv_cache_config_deepseek_v4 = _get_kv_cache_config_deepseek_v4_main
if vllm_version_is("0.23.0"):
    vllm.v1.core.kv_cache_utils._get_kv_cache_config_deepseek_v4 = _get_kv_cache_config_deepseek_v4
else:
    vllm.v1.core.kv_cache_utils._get_kv_cache_config_packed = _get_kv_cache_config_deepseek_v4
assert _orig_get_packed_kv_cache_groups is not None
vllm.v1.core.kv_cache_utils._get_packed_kv_cache_groups = _ascend_get_packed_kv_cache_groups
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_page_size = _get_kv_cache_groups_uniform_page_size
vllm.v1.core.kv_cache_utils.get_kv_cache_groups = _get_glm5_next_kv_cache_groups
KVCacheConfig.has_mamba_layers = property(  # type: ignore[assignment]
    _kv_cache_config_has_mamba_layers
)
vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups = _ascend_get_kv_cache_config_from_groups
vllm.v1.core.kv_cache_utils._max_memory_usage_bytes_from_groups = _ascend_max_memory_usage_bytes_from_groups
vllm.v1.core.kv_cache_utils._pool_bytes_per_block = _ascend_pool_bytes_per_block

# Also patch the reference used by engine/core.py which imports the function directly.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.engine.core.get_kv_cache_configs = _ascend_get_kv_cache_configs
