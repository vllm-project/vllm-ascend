# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect actual draft consumers before selecting a read-view provider.

Resolve every real consumer after ``set_attn`` without owning tensors.
Unknown or incompatible consumers fail before proposal work; successful
selection describes an addressing contract, not device qualification.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import torch


class DraftWindowProviderStatus(str, Enum):
    DIRECT_DENSE = "DIRECT_DENSE"
    NATIVE_SWA = "NATIVE_SWA"
    DIRECT_MLA = "DIRECT_MLA"


@dataclass(frozen=True)
class DraftWindowGroupRoute:
    gid: int
    layers: tuple[str, ...]
    table_column_span: int
    status: DraftWindowProviderStatus
    native_window: int | None = None
    dense_read: bool = False


def _select_read_provider(
    backend: type,
    builder: Any,
    specs: Sequence[Any],
    *,
    uncompressed: bool,
    native_window: int | None,
    uniform_windows: bool,
    full_retention_dflash2: bool,
) -> DraftWindowProviderStatus:
    """Match one actual consumer, then apply its layout-specific restrictions."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionMetadataBuilder
    from vllm_ascend.attention.mla_v1 import AscendMLABackend, AscendMLAMetadataBuilder
    from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

    if not uncompressed:
        raise NotImplementedError("Compressed or packed cache geometry needs a separate read provider")

    def matches(backend_type: type, builder_type: type, spec_type: type) -> bool:
        return (
            backend is backend_type and type(builder) is builder_type and all(type(item) is spec_type for item in specs)
        )

    if full_retention_dflash2 and any(
        type(item) is not FullAttentionSpec or getattr(item, "attention_chunk_size", None) is not None for item in specs
    ):
        raise NotImplementedError("DFlash2 requires Full-retention FullAttentionSpec without chunked attention")

    if matches(AscendAttentionBackend, AscendAttentionMetadataBuilder, FullAttentionSpec):
        if any(getattr(item, "attention_chunk_size", None) is not None for item in specs) or not uniform_windows:
            raise NotImplementedError("Dense chunked or nonuniform native attention needs a separate provider")
        if not full_retention_dflash2 or native_window is None:
            return DraftWindowProviderStatus.DIRECT_DENSE
    elif matches(AscendMLABackend, AscendMLAMetadataBuilder, AscendMLAAttentionSpec):
        if builder.flashmla_state is not None:
            raise NotImplementedError("FlashMLA persistent derived state needs a separate provider")
        if any(
            item.model_version is not None
            or item.cache_sparse_sfa_c8
            or item.cache_sparse_sfa_turboquant
            or item.store_on_host
            or item.indexes_kv_by_block_stride
            for item in specs
        ):
            raise NotImplementedError("Packed, offloaded, or strided MLA caches need a separate provider")
        return DraftWindowProviderStatus.DIRECT_MLA
    elif not matches(AscendAttentionBackend, AscendAttentionMetadataBuilder, SlidingWindowSpec):
        raise NotImplementedError(f"No read-view provider for {backend.__name__}/{type(builder).__name__}")
    if not uniform_windows or native_window is None or native_window <= 0:
        raise NotImplementedError("Native SWA group has inconsistent windows")
    return DraftWindowProviderStatus.NATIVE_SWA


def inventory_draft_window_groups(speculator: Any) -> tuple[DraftWindowGroupRoute, ...]:
    """Inventory every actual consumer without inspecting tensor contents.

    AscendBlockTables currently expands manager blocks into kernel columns.
    Thus ``kernel_block_sizes[gid]`` is the logical raw-token span of one
    table column; compressed physical rows are that span divided by ratio.
    No resident-set guarantee is inferred from capacity or valid counts.
    """
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    from vllm_ascend.core.kv_cache_interface import get_kv_cache_compression_ratio, get_storage_block_size

    attn_groups, block_tables = speculator.attn_groups, speculator.block_tables
    kv_cache_config = speculator.kv_cache_config
    config = speculator.attn_vllm_config
    parallel_config = config.parallel_config
    context = config.compilation_config.static_forward_context
    full_retention_dflash2 = getattr(speculator, "_dflash2_full_kv_reference", False)
    if parallel_config.prefill_context_parallel_size != 1 or parallel_config.decode_context_parallel_size != 1:
        raise NotImplementedError("Context-parallel read coordinates need a separate provider")
    cache_groups = kv_cache_config.kv_cache_groups
    if not (
        len(attn_groups) == len(cache_groups) == len(block_tables.block_sizes) == len(block_tables.kernel_block_sizes)
    ):
        raise ValueError("Draft attention groups and block-table geometry do not match")
    routes = []
    for gid, (groups, cache_group) in enumerate(zip(attn_groups, cache_groups)):
        if not groups:
            # init_attn_backend preserves indices for target-only cache groups
            # after filtering by active_layer_names. They have no draft reader.
            continue
        spec = cache_group.kv_cache_spec
        if block_tables.cp_size > 1 and spec.dcp_sharded is not False:
            raise NotImplementedError("Draft Window requires replicated Draft KV tables")
        specs = tuple(spec.kv_cache_specs.values()) if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        if not specs:
            raise ValueError(f"Draft KV group {gid} has no cache spec")
        geometries = {
            (int(item.block_size), int(get_kv_cache_compression_ratio(item)), int(get_storage_block_size(item)))
            for item in specs
        }
        if len(geometries) != 1:
            raise ValueError(f"Draft KV group {gid} has inconsistent cache geometry")
        scheduler_span, ratio, storage_size = geometries.pop()
        if int(block_tables.block_sizes[gid]) != scheduler_span:
            raise ValueError(f"Draft KV group {gid} scheduler and canonical table spans disagree")
        column_span = int(block_tables.kernel_block_sizes[gid])
        if (
            min(scheduler_span, column_span, ratio, storage_size) <= 0
            or scheduler_span % column_span
            or column_span % ratio
        ):
            raise ValueError(f"Draft KV group {gid} has invalid block geometry")
        for group in groups:
            if group.kv_cache_group_id != gid or int(group.kv_cache_spec.block_size) != scheduler_span:
                raise ValueError(f"Draft KV group {gid} consumer and canonical source geometry disagree")
            if isinstance(spec, UniformTypeKVCacheSpecs):
                try:
                    consumer_specs = tuple(spec.kv_cache_specs[name] for name in group.layer_names)
                except KeyError as error:
                    raise ValueError(f"Draft KV group {gid} omits an active consumer spec") from error
            else:
                consumer_specs = specs
            builder = group.get_metadata_builder(0)
            backend = group.backend
            windows = {getattr(item, "sliding_window", None) for item in consumer_specs}
            native_window = next(iter(windows)) if len(windows) == 1 else None
            try:
                status = _select_read_provider(
                    backend,
                    builder,
                    consumer_specs,
                    uncompressed=ratio == 1 and storage_size == scheduler_span,
                    native_window=native_window,
                    uniform_windows=len(windows) == 1,
                    full_retention_dflash2=full_retention_dflash2,
                )
            except NotImplementedError as error:
                raise NotImplementedError(f"Draft Window gid={gid}: {error}") from error
            implementations = [getattr(context.get(name), "impl", None) for name in group.layer_names]
            unpacked_dense = all(
                type(impl) is AscendAttentionBackendImpl and not impl.use_bnsd_kv_cache and impl.sinks is None
                for impl in implementations
            )
            if status is DraftWindowProviderStatus.NATIVE_SWA:
                if not unpacked_dense or any(
                    cast(AscendAttentionBackendImpl, impl).sliding_window != native_window for impl in implementations
                ):
                    raise NotImplementedError(f"Draft Window native SWA gid={gid} has no coverage certificate")
            elif full_retention_dflash2 and any(
                impl is None or impl.sliding_window != native_window for impl in implementations
            ):
                raise NotImplementedError("DFlash2 inventory native band differs from its actual Draft consumer")
            dense_read = (
                status is DraftWindowProviderStatus.DIRECT_DENSE
                and all(item.dtype in (torch.float16, torch.bfloat16) for item in specs)
                and unpacked_dense
                and all(cast(AscendAttentionBackendImpl, impl).sliding_window is None for impl in implementations)
            )
            routes.append(
                DraftWindowGroupRoute(
                    gid=gid,
                    layers=tuple(group.layer_names),
                    table_column_span=column_span,
                    status=status,
                    native_window=native_window,
                    dense_read=dense_read,
                )
            )
    if not routes:
        raise ValueError("Draft Window requires at least one attention consumer")
    if {name for route in routes for name in route.layers} != set(speculator.draft_attn_layer_names):
        raise ValueError("Draft Window providers do not cover the actual draft attention layers")
    return tuple(routes)


def require_direct_prefix_routes(
    speculator: Any, *, window_size: int | None = None
) -> tuple[DraftWindowGroupRoute, ...]:
    """Admit actual consumers once, retaining only groups that need projection.

    Inventory tags describe layouts, not read qualification. Native consumers
    keep their original band and require the method's real causal/residency
    contract here. Each admitted route is the original inventory object.
    """
    config = speculator.vllm_config
    routes = inventory_draft_window_groups(speculator)
    native = tuple(route for route in routes if route.status is DraftWindowProviderStatus.NATIVE_SWA)
    if native:
        if window_size is None or speculator.speculative_config.method != "dflash":
            raise NotImplementedError("Native SWA coverage is only certified for the DFlash extra-Window route")
        scheduler = config.scheduler_config
        query_bound = max(
            scheduler.max_num_batched_tokens,
            scheduler.max_num_scheduled_tokens or 0,
            speculator.num_query_per_req,
        )
        future = speculator.num_speculative_steps
        if query_bound < 1 or future < 0:
            raise ValueError("Native SWA coverage requires a positive query bound")
        native_only = len(native) == len(routes)
        declared = getattr(speculator, "_group_causal", True)
        omitted: set[DraftWindowGroupRoute] = set()
        for route in native:
            # Inventory already established a positive, homogeneous native P.
            native_window = cast(int, route.native_window)
            causal = declared.get(route.gid) if isinstance(declared, dict) else declared
            if type(causal) is not bool:
                raise ValueError("Native SWA requires an explicit per-group causal declaration")
            if not causal:
                if not (
                    native_only and getattr(speculator.draft_model_config, "architectures", ()) == ["DFlash2DraftModel"]
                ):
                    raise NotImplementedError(
                        "Draft Window non-causal native groups require the DFlash2 resident provider"
                    )
                # sparse_mode=0 reads the complete rebased interval, including
                # future queries: it cannot inherit the causal-band proof.
                width = speculator.num_query_per_req
                if width < 1 or window_size + route.table_column_span - 1 + width > native_window:
                    raise NotImplementedError(
                        "DFlash2 non-causal draft_window_size plus column alignment and query width "
                        "must fit its resident native window"
                    )
            # S <= max(E+f-W, 0); the first native query reads from
            # max(E-Q-P, 0). W >= P+Q+f contains that complete read set.
            # It then needs no projection beside Full groups. Native-only
            # batches still need their own builder for the proposal scope.
            if not native_only and window_size >= native_window + query_bound + future:
                omitted.add(route)
        routes = tuple(route for route in routes if route not in omitted)
    prefix = tuple(route for route in routes if route.status is not DraftWindowProviderStatus.NATIVE_SWA)
    if prefix:
        # Every group must represent any start aligned to the first span;
        # checking only the current request would miss later block boundaries.
        anchor = prefix[0].table_column_span
        for route in prefix:
            if anchor % route.table_column_span:
                raise NotImplementedError(
                    "Draft Window provider missing: "
                    f"gid={route.gid} span {route.table_column_span} cannot represent every start "
                    f"aligned to the first group span {anchor}"
                )
    return routes
