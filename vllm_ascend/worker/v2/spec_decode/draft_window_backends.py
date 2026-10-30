# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect actual draft consumers before selecting a read-view provider.

This inventory is created after ``set_attn`` and owns no tensors. A candidate
means the dense provider's addressing contract matches, not device validation.
Unknown consumers remain missing providers instead of being silently skipped.
"""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any


class DraftWindowProviderStatus(str, Enum):
    DIRECT_DENSE = "DIRECT_DENSE"
    NATIVE_SWA_CANDIDATE = "NATIVE_SWA_CANDIDATE"
    NATIVE_SWA_INTERSECTION = "NATIVE_SWA_INTERSECTION"
    DIRECT_MLA = "DIRECT_MLA"
    DIRECT_DSA = "DIRECT_DSA"
    DIRECT_SFA = "DIRECT_SFA"
    DIRECT_SFA_INDEXER = "DIRECT_SFA_INDEXER"
    PROVIDER_MISSING = "PROVIDER_MISSING"


@dataclass(frozen=True)
class DraftWindowTopology:
    tensor_parallel_size: int
    data_parallel_size: int
    pipeline_parallel_size: int
    prefill_context_parallel_size: int
    decode_context_parallel_size: int
    cp_rank: int
    cp_interleave: int
    sequence_parallel: bool
    expert_parallel: bool
    flashcomm_requested: bool | None
    dsa_cp_requested: bool | None
    dsa_cp_effective: bool
    tensor_parallel_rank: int | None = None
    dsa_cp_config_enabled: bool | None = None

    @property
    def flashcomm_effective(self) -> bool:
        return self.sequence_parallel

    @classmethod
    def from_config(
        cls,
        config: Any,
        block_tables: Any,
        *,
        dsa_cp_effective: bool,
        flashcomm_requested: bool | None = None,
        dsa_cp_requested: bool | None = None,
    ) -> "DraftWindowTopology":
        """Use the reconstructed draft config, not the target model's config.

        ``use_sequence_parallel_moe`` is the effective FlashComm/SP authority.
        Requested flags are diagnostics only and may be unknown (e.g. an
        environment request already consumed by config validation).
        """
        parallel = config.parallel_config
        return cls(
            tensor_parallel_size=parallel.tensor_parallel_size,
            data_parallel_size=parallel.data_parallel_size,
            pipeline_parallel_size=parallel.pipeline_parallel_size,
            prefill_context_parallel_size=parallel.prefill_context_parallel_size,
            decode_context_parallel_size=parallel.decode_context_parallel_size,
            cp_rank=block_tables.cp_rank,
            cp_interleave=block_tables.cp_interleave,
            sequence_parallel=parallel.use_sequence_parallel_moe,
            expert_parallel=parallel.enable_expert_parallel,
            flashcomm_requested=flashcomm_requested,
            dsa_cp_requested=dsa_cp_requested,
            dsa_cp_effective=dsa_cp_effective,
        )


@dataclass(frozen=True)
class DraftWindowBackendTypes:
    """Concrete runtime classes; injected explicitly in CPU contract tests."""

    dense_backend: type
    dense_builder: type
    full_attention_spec: type
    uniform_spec: type[Any]
    builder_families: tuple[tuple[type, str], ...]
    storage_block_size: Callable[[Any], int]
    compression_ratio: Callable[[Any], int]
    mla_backend: type | None = None
    mla_builder: type | None = None
    mla_spec: type | None = None
    sfa_backend: type | None = None
    sfa_builder: type | None = None
    indexer_backend: type | None = None
    indexer_builder: type | None = None
    indexer_spec: type | None = None
    dsa_backend: type | None = None
    dsa_builder: type | None = None
    dsa_cp_builder: type | None = None
    dsa_spec: type | None = None
    sliding_window_spec: type | None = None

    @classmethod
    def load(cls) -> "DraftWindowBackendTypes":
        # Keep device/backend imports out of CONFIG_OFF and policy evaluation.
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs

        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionMetadataBuilder
        from vllm_ascend.attention.context_parallel.dsa_cp import (
            AscendDSACPMetadataBuilder,
            AscendDSAPCPMetadataBuilder,
        )
        from vllm_ascend.attention.context_parallel.sfa_cp import (
            AscendSFADSACPMetadataBuilder,
            AscendSFADSADCPMetadataBuilder,
        )
        from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder, AscendDSASWABackend
        from vllm_ascend.attention.dsa_v41 import AscendDSAV41MetadataBuilder
        from vllm_ascend.attention.indexer import AscendSFAIndexerBackend, AscendSFAIndexerMetadataBuilder
        from vllm_ascend.attention.mla_v1 import AscendMLABackend, AscendMLAMetadataBuilder
        from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAMetadataBuilder
        from vllm_ascend.core.kv_cache_interface import (
            AscendMLAAttentionSpec,
            AscendSFAIndexerCacheSpec,
            AscendSlidingWindowMLASpec,
            get_kv_cache_compression_ratio,
            get_storage_block_size,
        )

        return cls(
            AscendAttentionBackend,
            AscendAttentionMetadataBuilder,
            FullAttentionSpec,
            UniformTypeKVCacheSpecs,
            (
                (AscendDSACPMetadataBuilder, "DSA_CP"),
                (AscendDSAPCPMetadataBuilder, "DSA_PCP"),
                (AscendSFADSADCPMetadataBuilder, "SFA_DSA_DCP"),
                (AscendSFADSACPMetadataBuilder, "SFA_DSA_CP"),
                (AscendDSAV41MetadataBuilder, "DSA_V41"),
                (AscendDSAMetadataBuilder, "DSA"),
                (AscendSFAIndexerMetadataBuilder, "SFA_INDEXER"),
                (AscendSFAMetadataBuilder, "SFA"),
                (AscendMLAMetadataBuilder, "MLA"),
                (AscendAttentionMetadataBuilder, "GQA"),
            ),
            get_storage_block_size,
            get_kv_cache_compression_ratio,
            mla_backend=AscendMLABackend,
            mla_builder=AscendMLAMetadataBuilder,
            mla_spec=AscendMLAAttentionSpec,
            sfa_backend=AscendSFABackend,
            sfa_builder=AscendSFAMetadataBuilder,
            indexer_backend=AscendSFAIndexerBackend,
            indexer_builder=AscendSFAIndexerMetadataBuilder,
            indexer_spec=AscendSFAIndexerCacheSpec,
            dsa_backend=AscendDSASWABackend,
            dsa_builder=AscendDSAMetadataBuilder,
            dsa_cp_builder=AscendDSACPMetadataBuilder,
            dsa_spec=AscendSlidingWindowMLASpec,
            sliding_window_spec=SlidingWindowSpec,
        )


@dataclass(frozen=True)
class DraftWindowGroupRoute:
    gid: int
    layers: tuple[str, ...]
    backend: str
    builder: str
    family: str
    kv_specs: tuple[str, ...]
    scheduler_block_size: int
    table_column_span: int
    physical_block_size: int
    compression_ratio: int
    topology: DraftWindowTopology
    status: DraftWindowProviderStatus
    reason: str
    native_window: int | None = None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["topology"]["flashcomm_effective"] = self.topology.flashcomm_effective
        return result


def _type_name(value: type) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def inventory_draft_window_groups(
    *,
    attn_groups: Sequence[Sequence[Any]],
    kv_cache_config: Any,
    block_tables: Any,
    topology: DraftWindowTopology,
    types: DraftWindowBackendTypes | None = None,
    full_retention_dflash2: bool = False,
) -> tuple[DraftWindowGroupRoute, ...]:
    """Inventory every actual consumer without inspecting tensor contents.

    AscendBlockTables currently expands manager blocks into kernel columns.
    Thus ``kernel_block_sizes[gid]`` is the logical raw-token span of one
    table column; compressed physical rows are that span divided by ratio.
    No resident-set guarantee is inferred from capacity or valid counts.
    """
    if types is None:
        types = DraftWindowBackendTypes.load()
    cache_groups = kv_cache_config.kv_cache_groups
    if len(attn_groups) != len(cache_groups) or len(block_tables.kernel_block_sizes) != len(cache_groups):
        raise ValueError("Draft attention groups and block-table geometry do not match")
    routes = []
    for gid, (groups, cache_group) in enumerate(zip(attn_groups, cache_groups)):
        if not groups:
            # init_attn_backend preserves indices for target-only cache groups
            # after filtering by active_layer_names. They have no draft reader.
            continue
        spec = cache_group.kv_cache_spec
        specs = tuple(spec.kv_cache_specs.values()) if isinstance(spec, types.uniform_spec) else (spec,)
        if not specs:
            raise ValueError(f"Draft KV group {gid} has no cache spec")
        spans = {int(item.block_size) for item in specs}
        ratios = {int(types.compression_ratio(item)) for item in specs}
        storage_sizes = {int(types.storage_block_size(item)) for item in specs}
        if len(spans) != 1 or len(ratios) != 1 or len(storage_sizes) != 1:
            raise ValueError(f"Draft KV group {gid} has inconsistent cache geometry")
        scheduler_span, ratio, storage_size = spans.pop(), ratios.pop(), storage_sizes.pop()
        column_span = int(block_tables.kernel_block_sizes[gid])
        if (
            min(scheduler_span, column_span, ratio, storage_size) <= 0
            or scheduler_span % column_span
            or column_span % ratio
        ):
            raise ValueError(f"Draft KV group {gid} has invalid block geometry")
        for group in groups:
            if isinstance(spec, types.uniform_spec):
                try:
                    consumer_specs = tuple(spec.kv_cache_specs[name] for name in group.layer_names)
                except KeyError as error:
                    raise ValueError(f"Draft KV group {gid} omits an active consumer spec") from error
            else:
                consumer_specs = specs
            builder = group.get_metadata_builder(0)
            backend = group.backend
            family = next((name for kind, name in types.builder_families if isinstance(builder, kind)), "UNKNOWN")
            # The target can enable DSA-CP while the draft is dense GQA.
            # Selection of the actual builder, not the global flag, decides
            # whether this reader uses legacy DSA-CP coordinates.
            route_topology = replace(
                topology,
                dsa_cp_effective=(
                    family in ("DSA_CP", "SFA_DSA_CP", "SFA_DSA_DCP")
                    or (family == "SFA_INDEXER" and builder.use_dsa_cp)
                ),
            )
            reason = "No read-view provider is registered for this consumer"
            status = DraftWindowProviderStatus.PROVIDER_MISSING
            native_window = None
            if (
                backend is types.dense_backend
                and type(builder) is types.dense_builder
                and all(type(item) is types.full_attention_spec for item in consumer_specs)
                and not (
                    full_retention_dflash2 and all(getattr(item, "sliding_window", None) for item in consumer_specs)
                )
                and ratio == 1
                and storage_size == scheduler_span
            ):
                if topology.prefill_context_parallel_size != 1 or topology.decode_context_parallel_size != 1:
                    reason = "Context-parallel read coordinates need a separate provider"
                else:
                    status = DraftWindowProviderStatus.DIRECT_DENSE
                    reason = "Dense kernel-column read view; Full write/query coordinates stay unchanged"
            elif (
                types.sliding_window_spec is not None
                and backend is types.dense_backend
                and type(builder) is types.dense_builder
                and (
                    all(type(item) is types.sliding_window_spec for item in consumer_specs)
                    or (
                        full_retention_dflash2
                        and all(type(item) is types.full_attention_spec for item in consumer_specs)
                    )
                )
                and ratio == 1
                and storage_size == scheduler_span
            ):
                windows = {item.sliding_window for item in consumer_specs}
                if topology.prefill_context_parallel_size != 1 or topology.decode_context_parallel_size != 1:
                    reason = "Native SWA context-parallel coordinates need a separate provider"
                elif len(windows) != 1 or next(iter(windows)) <= 0:
                    reason = "Native SWA group has inconsistent windows"
                else:
                    native_window = int(next(iter(windows)))
                    status = DraftWindowProviderStatus.NATIVE_SWA_CANDIDATE
                    reason = "Native SWA may already be a subset; requires a static query/window coverage certificate"
            elif (
                types.mla_backend is not None
                and backend is types.mla_backend
                and type(builder) is types.mla_builder
                and all(type(item) is types.mla_spec for item in consumer_specs)
                and ratio == 1
                and storage_size == scheduler_span
            ):
                if topology.prefill_context_parallel_size != 1 or topology.decode_context_parallel_size != 1:
                    reason = "MLA context-parallel read coordinates need a separate provider"
                elif builder.flashmla_state is not None:
                    reason = "FlashMLA persistent derived state needs a separate provider"
                elif any(
                    item.model_version is not None
                    or item.cache_sparse_sfa_c8
                    or item.cache_sparse_sfa_turboquant
                    or item.store_on_host
                    or item.indexes_kv_by_block_stride
                    for item in consumer_specs
                ):
                    reason = "Packed, offloaded, or strided MLA caches need a separate provider"
                else:
                    status = DraftWindowProviderStatus.DIRECT_MLA
                    reason = "Uncompressed MLA read projection with proposal-owned decode RoPE"
            elif (
                types.dsa_backend is not None
                and backend is types.dsa_backend
                and type(builder) in (types.dsa_builder, types.dsa_cp_builder)
                and all(type(item) is types.dsa_spec for item in consumer_specs)
                and ratio == 1
                and storage_size == scheduler_span
                and builder.compressor_ratio == 1
            ):
                hf = builder.model_config.hf_config
                if type(builder) is types.dsa_cp_builder:
                    reason = "Legacy DSA-CP is rejected by the MRV2 startup contract; no Window provider is enabled"
                elif topology.prefill_context_parallel_size != 1 or topology.decode_context_parallel_size != 1:
                    reason = "DSA PCP/DCP read coordinates need a separate provider"
                elif builder._device_metadata_enabled:
                    reason = "Deferred DSA metadata needs an executor completion contract"
                elif getattr(hf, "vision_n_layers", 0) > 0:
                    reason = "DSA vision ranges require absolute-to-window coordinate projection"
                elif any(
                    item.model_version is not None or item.indexes_kv_by_block_stride or item.bounded_replay
                    for item in consumer_specs
                ):
                    reason = "Packed, strided, or replay DSA needs a separate read provider"
                elif hf.sliding_window <= 0 or any(item.sliding_window != hf.sliding_window for item in consumer_specs):
                    reason = "DSA native-window kernel and resident cache geometry disagree"
                else:
                    status = DraftWindowProviderStatus.DIRECT_DSA
                    reason = "Native-window intersection with proposal-owned DSA derived state"
            elif (
                types.sfa_backend is not None
                and backend is types.sfa_backend
                and type(builder) is types.sfa_builder
                and all(type(item) is types.mla_spec for item in consumer_specs)
                and ratio == 1
                and storage_size == scheduler_span
            ):
                if topology.prefill_context_parallel_size != 1 or topology.decode_context_parallel_size != 1:
                    reason = "SFA context-parallel read coordinates need a separate provider"
                elif builder.nope:
                    reason = "NoPE SFA has persistent sparse-kernel state requiring its own owner"
                elif any(
                    item.model_version is not None
                    or item.cache_sparse_sfa_c8
                    or item.cache_sparse_sfa_turboquant
                    or item.store_on_host
                    or item.indexes_kv_by_block_stride
                    for item in consumer_specs
                ):
                    reason = "Packed, offloaded, or strided SFA caches need a separate provider"
                elif any(item.non_causal_multi_token_decode for item in consumer_specs):
                    reason = "Ordinary SFA kernels use causal sparse_mode=3 for multi-token queries"
                else:
                    status = DraftWindowProviderStatus.DIRECT_SFA
                    reason = "SFA and its indexer select relative indices from the same logical read window"
            elif (
                types.indexer_backend is not None
                and backend is types.indexer_backend
                and type(builder) is types.indexer_builder
                and all(type(item) is types.indexer_spec for item in consumer_specs)
                and ratio == 1
                and storage_size == scheduler_span
            ):
                if builder.use_pcp or builder.use_dcp or builder.use_dsa_cp:
                    reason = "SFA indexer rank-local read coordinates need a separate provider"
                elif any(
                    item.cache_sparse_li_c4 or item.sfa_dcp_replicated_indexer_size != 1 for item in consumer_specs
                ):
                    reason = "Packed C4 or replicated indexer pages need a separate provider"
                else:
                    status = DraftWindowProviderStatus.DIRECT_SFA_INDEXER
                    reason = "Full indexer K/scale pages use the same table-relative Window coordinates"
            routes.append(
                DraftWindowGroupRoute(
                    gid=gid,
                    layers=tuple(group.layer_names),
                    backend=_type_name(backend),
                    builder=_type_name(type(builder)),
                    family=family,
                    kv_specs=tuple(_type_name(type(item)) for item in consumer_specs),
                    scheduler_block_size=scheduler_span,
                    table_column_span=column_span,
                    physical_block_size=column_span // ratio,
                    compression_ratio=ratio,
                    topology=route_topology,
                    status=status,
                    reason=reason,
                    native_window=native_window,
                )
            )
    return tuple(routes)


def require_draft_window_providers(routes: Sequence[DraftWindowGroupRoute]) -> None:
    """Fail before materializing any group if any required provider is missing."""
    if not routes:
        raise ValueError("Draft Window requires at least one attention consumer")
    missing = [
        route
        for route in routes
        if route.status in (DraftWindowProviderStatus.PROVIDER_MISSING, DraftWindowProviderStatus.NATIVE_SWA_CANDIDATE)
    ]
    if missing:
        details = "; ".join(f"gid={route.gid}, {route.builder}: {route.reason}" for route in missing)
        raise NotImplementedError(f"Draft Window provider missing: {details}")


def describe_draft_window_routes(speculator: Any) -> tuple[DraftWindowGroupRoute, ...]:
    """Inspect initialized draft attention; never invoke from the OFF path."""
    from vllm.distributed import get_tensor_model_parallel_rank

    from vllm_ascend.utils import enable_dsa_cp

    config = speculator.attn_vllm_config
    requested = config.additional_config or {}
    flashcomm = requested.get("enable_flashcomm1")
    dsa_cp = requested.get("enable_dsa_cp")
    topology = DraftWindowTopology.from_config(
        config,
        speculator.block_tables,
        dsa_cp_effective=enable_dsa_cp(),
        flashcomm_requested=flashcomm if isinstance(flashcomm, bool) else None,
        dsa_cp_requested=dsa_cp if isinstance(dsa_cp, bool) else None,
    )
    topology = replace(
        topology,
        tensor_parallel_rank=get_tensor_model_parallel_rank(),
        dsa_cp_config_enabled=topology.dsa_cp_effective,
    )
    routes = inventory_draft_window_groups(
        attn_groups=speculator.attn_groups,
        kv_cache_config=speculator.kv_cache_config,
        block_tables=speculator.block_tables,
        topology=topology,
        full_retention_dflash2=getattr(speculator, "_dflash2_full_kv_reference", False),
    )
    actual_layers = {name for route in routes for name in route.layers}
    if actual_layers != set(speculator.draft_attn_layer_names):
        raise ValueError("Draft Window inventory does not cover the actual draft attention layers")
    return routes


def require_compatible_direct_prefix_routes(routes: Sequence[DraftWindowGroupRoute]) -> None:
    """Keep the first group's window alignment and project it into all groups.

    Requiring each group span to divide the first span ensures every future
    aligned start is expressible, without changing the window to an LCM or
    failing after an ordinary request crosses a block boundary.
    """
    require_draft_window_providers(routes)
    prefix_routes = tuple(
        route for route in routes if route.status is not DraftWindowProviderStatus.NATIVE_SWA_INTERSECTION
    )
    if not prefix_routes:
        # Native-band builders validate and rebase their own coordinates;
        # unlike a direct-prefix reader, they need no companion Full group.
        if routes:
            return
        raise NotImplementedError("Draft Window requires at least one certified read group")
    anchor_span = prefix_routes[0].table_column_span
    for route in prefix_routes:
        if anchor_span % route.table_column_span:
            raise NotImplementedError(
                "Draft Window provider missing: "
                f"gid={route.gid} span {route.table_column_span} cannot represent every start "
                f"aligned to the first group span {anchor_span}"
            )


def select_native_swa_subset_routes(
    routes: Sequence[DraftWindowGroupRoute],
    *,
    window_size: int,
    max_query_tokens: int,
    future_offset_bound: int,
    context: Any,
    dense_impl_type: type[Any],
    group_causal: dict[int, bool] | bool = True,
    noncausal_dflash2: bool = False,
    query_width: int | None = None,
) -> tuple[DraftWindowGroupRoute, ...]:
    """Keep a certified native SWA reader unchanged, never claim its residency.

    S <= max(E+f-W, 0). A native P-window's earliest query reads no
    earlier than max(E-Q-P, 0). W >= P+Q+f proves that its complete
    existing read set lies inside the requested extra Window. Smaller W
    selects a typed native-band intersection, whose original builder exact
    pair is reused after build. Neither route claims a resident Full prefix.
    """
    if max_query_tokens < 1 or future_offset_bound < 0:
        raise ValueError("Native SWA coverage requires a positive query bound")
    native_only = bool(routes) and all(
        route.status is DraftWindowProviderStatus.NATIVE_SWA_CANDIDATE for route in routes
    )
    projected = []
    for route in routes:
        if route.status is not DraftWindowProviderStatus.NATIVE_SWA_CANDIDATE:
            projected.append(route)
            continue
        native_window = route.native_window
        if native_window is None or native_window <= 0:
            raise NotImplementedError(f"Draft Window gid={route.gid} has no native-SWA geometry")
        causal = group_causal.get(route.gid) if isinstance(group_causal, dict) else group_causal
        if type(causal) is not bool:
            raise ValueError("Native SWA requires an explicit per-group causal declaration")
        if not causal:
            if not (noncausal_dflash2 and native_only):
                raise NotImplementedError("Draft Window non-causal native groups require the DFlash2 resident provider")
            # sparse_mode=0 reads the entire projected interval, including
            # future query positions. It must not inherit a causal-band
            # certificate or assume an evicted Full prefix is resident.
            if (
                query_width is None
                or query_width < 1
                or window_size + route.table_column_span - 1 + query_width > native_window
            ):
                raise NotImplementedError(
                    "DFlash2 non-causal draft_window_size plus column alignment and query width "
                    "must fit its resident native window"
                )
        for name in route.layers:
            impl = getattr(context.get(name), "impl", None)
            if (
                type(impl) is not dense_impl_type
                or impl.sliding_window != native_window
                or impl.sinks is not None
                or impl.use_bnsd_kv_cache
            ):
                raise NotImplementedError(f"Draft Window native SWA consumer {name} has no coverage certificate")
        if native_only or window_size < native_window + max_query_tokens + future_offset_bound:
            projected.append(
                replace(
                    route,
                    status=DraftWindowProviderStatus.NATIVE_SWA_INTERSECTION,
                    reason=(
                        "Typed causal native band intersected using its builder's current exact pair"
                        if causal
                        else "Typed DFlash2 unmasked resident interval using its builder's current exact pair"
                    ),
                )
            )
    if not projected:
        raise NotImplementedError("Draft Window requires at least one certified read group")
    return tuple(projected)


def require_direct_prefix_routes(
    speculator: Any, *, window_size: int | None = None
) -> tuple[DraftWindowGroupRoute, ...]:
    """Select all direct-prefix consumers before any proposal work is submitted."""
    routes = describe_draft_window_routes(speculator)
    if any(route.status is DraftWindowProviderStatus.NATIVE_SWA_CANDIDATE for route in routes):
        if window_size is None or speculator.speculative_config.method != "dflash":
            raise NotImplementedError("Native SWA coverage is only certified for the DFlash extra-Window route")
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

        config = speculator.attn_vllm_config
        scheduler = config.scheduler_config
        query_bound = max(
            scheduler.max_num_batched_tokens,
            scheduler.max_num_scheduled_tokens or 0,
            speculator.num_query_per_req,
        )
        routes = select_native_swa_subset_routes(
            routes,
            window_size=window_size,
            max_query_tokens=query_bound,
            future_offset_bound=speculator.num_speculative_steps,
            context=config.compilation_config.static_forward_context,
            dense_impl_type=AscendAttentionBackendImpl,
            group_causal=getattr(speculator, "_group_causal", True),
            noncausal_dflash2=getattr(speculator.draft_model_config, "architectures", ()) == ["DFlash2DraftModel"],
            query_width=speculator.num_query_per_req,
        )
    if any(route.status is DraftWindowProviderStatus.DIRECT_SFA for route in routes):
        from vllm_ascend.worker.v2.spec_decode.draft_window_sfa import validate_sfa_read_consumers

        validate_sfa_read_consumers(speculator.attn_vllm_config.compilation_config.static_forward_context, routes)
    require_compatible_direct_prefix_routes(routes)
    return routes


def project_draft_window_start(start_tokens: int, route: DraftWindowGroupRoute) -> int:
    """Project one logical start exactly; never round away a read offset."""
    if start_tokens < 0 or start_tokens % route.table_column_span:
        raise ValueError(f"Draft Window start {start_tokens} is not representable by KV group {route.gid}")
    return start_tokens // route.table_column_span
