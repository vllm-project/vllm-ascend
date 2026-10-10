"""Translate vLLM-owned state into AscendStore v1 domain facts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import vllm.envs as vllm_envs
import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.distributed import get_dcp_group, get_pcp_group, get_pp_group, get_tp_group
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_cache_spec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    KeyMetadata,
    infer_cacheable_group_ids,
    infer_dcp_mismatch_info,
    infer_group_cache_families,
    infer_tp_mismatch_info,
    uses_hybrid_kv_cache,
)

from ..backend import get_layerwise_data_plane, get_layerwise_protocol, validate_layerwise_topology
from ..layerwise_cache_layout import (
    build_layerwise_reuse_layout,
    get_layerwise_kv_cache_specs,
    get_layerwise_reuse_config,
)
from .backend import LayerwiseAccessKind, resolve_backend_spec
from .projection import (
    BulkProjectionBinder,
    GVALayerwiseProjectionBinder,
    KeyRangeLayerwiseProjectionBinder,
    LayerwiseProjectionBinder,
    compile_bulk_projection_binder,
)
from .route import KVPoolRouteSpec
from .scheduler import (
    AsynchronousBulkScheduler,
    KVPoolScheduler,
    LayerwiseScheduler,
    SchedulerConfig,
    SynchronousBulkScheduler,
)
from .scheduler.lookup import RemoteLookup
from .topology import (
    KVPoolGroupTopology,
    KVPoolTopology,
    TPPartitionSpec,
    kv_cache_spec_contains_mamba,
    kv_cache_spec_uses_align_state,
    resolve_group_layers,
)
from .worker.base import KVPoolWorker
from .worker.bulk import AsynchronousBulkWorker, SynchronousBulkWorker
from .worker.layerwise import GVALayerwiseWorker, KeyRangeLayerwiseWorker
from .worker.resources import GVAObjectLayout, KVPoolResources

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


def create_kv_pool_scheduler(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    lookup_address: str,
) -> KVPoolScheduler:
    """Bind one concrete Scheduler route from immutable vLLM configuration."""

    transfer_config = vllm_config.kv_transfer_config
    extra_config = transfer_config.kv_connector_extra_config
    use_layerwise = bool(extra_config.get("use_layerwise", False))
    backend_name = extra_config.get("backend", "mooncake").strip().lower()
    _validate_kv_pool_preflight(vllm_config, backend_name, use_layerwise=use_layerwise)
    cache_transfer_granularity, hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    transfer_group_ids, _ = _resolve_transfer_group_ids(kv_cache_config)
    align_state_group_ids = frozenset(
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if group_id in transfer_group_ids and kv_cache_spec_uses_align_state(group.kv_cache_spec)
    )
    has_private_state = len(transfer_group_ids) != len(kv_cache_config.kv_cache_groups)
    discard_partial_chunks = bool(extra_config.get("discard_partial_chunks", True))
    _validate_partial_object_support(
        vllm_config,
        kv_cache_config,
        use_layerwise=use_layerwise,
        discard_partial_chunks=discard_partial_chunks,
    )
    if use_layerwise and has_private_state:
        raise ValueError("AscendStore private KV state requires non-layerwise transfer")
    _validate_bulk_topology_support(
        vllm_config,
        kv_cache_config,
        use_layerwise=use_layerwise,
        has_private_state=has_private_state,
    )

    config = SchedulerConfig(
        cache_transfer_granularity=cache_transfer_granularity,
        hash_block_size=hash_block_size,
        transfer_group_ids=transfer_group_ids,
        align_state_group_ids=align_state_group_ids,
        load_enabled=transfer_config.kv_role != "kv_consumer" or bool(extra_config.get("consumer_is_to_load", False)),
        store_enabled=_store_enabled(vllm_config, extra_config),
        save_decode_cache=bool(extra_config.get("save_decode_cache", False)),
        discard_partial_chunks=discard_partial_chunks,
        use_eagle_block_drop=_uses_eagle(vllm_config),
        has_private_state=has_private_state,
        expected_worker_count=vllm_config.parallel_config.world_size,
    )
    scheduler_type: type[KVPoolScheduler]
    if use_layerwise:
        scheduler_type = LayerwiseScheduler
    elif extra_config.get("load_async", False):
        scheduler_type = AsynchronousBulkScheduler
    else:
        scheduler_type = SynchronousBulkScheduler
    return scheduler_type(config, RemoteLookup(lookup_address))


def create_kv_pool_worker(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> KVPoolWorker:
    """Create the Worker-side owner of projection, memory, I/O, and timelines."""

    route_spec = resolve_kv_pool_route_spec(vllm_config, kv_cache_config)
    projection_binder = _compile_kv_pool_projection_binder(route_spec, vllm_config, kv_cache_config)
    backend_spec = resolve_backend_spec(route_spec.backend_name)
    backend = backend_spec.create(
        vllm_config.parallel_config,
        extra_config=vllm_config.kv_transfer_config.kv_connector_extra_config,
    )
    resources: KVPoolResources | None = None
    try:
        gva_layout = None
        if route_spec.use_layerwise and backend_spec.layerwise_access is LayerwiseAccessKind.GVA:
            layer_start, _ = vllm_config.model_config.get_layers_start_end_indices(vllm_config.parallel_config)
            topology = route_spec.topology
            gva_layout = GVAObjectLayout(
                layer_start,
                vllm_config.model_config.get_total_num_hidden_layers(),
                topology.transfer_groups[0].key_metadata.dcp_rank,
                topology.dcp_size,
                topology.put_step,
            )
        resources = KVPoolResources(
            backend,
            backend_spec,
            kv_cache_config.num_blocks,
            route_spec.topology.transfer_groups,
            gva_layout=gva_layout,
            align_shared_storage=gva_layout is not None
            and uses_hybrid_kv_cache(vllm_config.scheduler_config, kv_cache_config.kv_cache_groups),
        )
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        store_enabled = _store_enabled(vllm_config, extra_config)
        if route_spec.use_layerwise:
            prefetch_layers = _resolve_layerwise_prefetch_layers(extra_config, True)
            if backend_spec.layerwise_access is LayerwiseAccessKind.GVA:
                if not isinstance(projection_binder, GVALayerwiseProjectionBinder):
                    raise TypeError("GVA route did not compile a GVA Layerwise projection binder")
                return GVALayerwiseWorker(
                    route_spec.topology,
                    projection_binder,
                    resources,
                    store_enabled=store_enabled,
                    layerwise_prefetch_layers=prefetch_layers,
                )
            if backend_spec.layerwise_access is LayerwiseAccessKind.KEY_RANGE:
                if not isinstance(projection_binder, KeyRangeLayerwiseProjectionBinder):
                    raise TypeError("KeyRange route did not compile a KeyRange Layerwise projection binder")
                return KeyRangeLayerwiseWorker(
                    route_spec.topology,
                    projection_binder,
                    resources,
                    store_enabled=store_enabled,
                    layerwise_prefetch_layers=prefetch_layers,
                )
            raise TypeError("Layerwise route requires a concrete Backend data plane")

        if not isinstance(projection_binder, BulkProjectionBinder):
            raise TypeError("Bulk route did not compile a Bulk projection binder")
        worker_type = AsynchronousBulkWorker if extra_config.get("load_async", False) else SynchronousBulkWorker
        return worker_type(
            route_spec.topology,
            projection_binder,
            resources,
            store_enabled=store_enabled,
        )
    except BaseException:
        try:
            if resources is not None:
                resources.close()
            else:
                backend.close()
        except BaseException:
            logger.exception("Failed to close KV Pool Backend after Worker initialization failed")
        raise


def compile_kv_pool_projection_binder(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> LayerwiseProjectionBinder | BulkProjectionBinder:
    """Select static projection from vLLM state and return the memory binder."""

    route_spec = resolve_kv_pool_route_spec(vllm_config, kv_cache_config)
    return _compile_kv_pool_projection_binder(route_spec, vllm_config, kv_cache_config)


def _compile_kv_pool_projection_binder(
    spec: KVPoolRouteSpec,
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> LayerwiseProjectionBinder | BulkProjectionBinder:
    if not spec.use_layerwise:
        return compile_bulk_projection_binder(
            spec.topology,
            spec.max_model_len,
            use_eagle=spec.use_eagle,
            retention_interval=spec.retention_interval,
            kv_cache_layout=vllm_envs.VLLM_KV_CACHE_LAYOUT,
        )

    protocol = get_layerwise_protocol(spec.backend_name)
    if protocol is None:
        raise ValueError(f"Backend {spec.backend_name!r} has no Layerwise key protocol")
    topology = spec.topology
    kv_cache_groups = kv_cache_config.kv_cache_groups
    data_plane = get_layerwise_data_plane(protocol)
    use_hybrid = uses_hybrid_kv_cache(vllm_config.scheduler_config, kv_cache_groups) or (
        data_plane == "block_key" and len(kv_cache_groups) > 1
    )
    model_name = topology.groups[0].key_metadata.model_name
    key_builder = protocol.bind_layerwise_keys(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        model_name=model_name,
        use_hybrid=use_hybrid,
        grouped_block_size=[group.block_size for group in topology.groups],
    )
    backend_spec = resolve_backend_spec(spec.backend_name)
    if backend_spec.layerwise_access is None:
        raise ValueError(f"Backend {spec.backend_name!r} has no Layerwise data plane")
    binder_type: type[GVALayerwiseProjectionBinder] | type[KeyRangeLayerwiseProjectionBinder]
    if backend_spec.layerwise_access is LayerwiseAccessKind.GVA:
        binder_type = GVALayerwiseProjectionBinder
    else:
        binder_type = KeyRangeLayerwiseProjectionBinder
    return binder_type(
        topology,
        spec.max_model_len,
        key_builder.make_full_key,
        spec.use_eagle,
        spec.retention_interval,
    )


def resolve_kv_pool_route_spec(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> KVPoolRouteSpec:
    """Validate configuration and capture the static facts consumed by KV projection compilation."""

    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    use_layerwise = bool(extra_config.get("use_layerwise", False))
    backend_name = extra_config.get("backend", "mooncake").strip().lower()
    _validate_kv_pool_preflight(vllm_config, backend_name, use_layerwise=use_layerwise)
    _validate_partial_object_support(
        vllm_config,
        kv_cache_config,
        use_layerwise=use_layerwise,
        discard_partial_chunks=bool(extra_config.get("discard_partial_chunks", True)),
    )
    _validate_bulk_topology_support(
        vllm_config,
        kv_cache_config,
        use_layerwise=use_layerwise,
    )
    topology = _resolve_kv_pool_topology(vllm_config, kv_cache_config)
    return KVPoolRouteSpec(
        topology=topology,
        backend_name=backend_name,
        max_model_len=vllm_config.model_config.max_model_len,
        use_layerwise=use_layerwise,
        use_eagle=_uses_eagle(vllm_config),
        retention_interval=kv_cache_config.prefix_cache_retention_interval,
    )


def _validate_partial_object_support(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    *,
    use_layerwise: bool,
    discard_partial_chunks: bool,
) -> None:
    if not discard_partial_chunks:
        raise ValueError(
            "AscendStore v1 does not support discard_partial_chunks=False until "
            "native partial-object identity and transfer are implemented"
        )
    if use_layerwise and _uses_layerwise_buffer_reuse(vllm_config, kv_cache_config):
        raise ValueError("AscendStore v1 Layerwise buffer reuse requires native partial-object support")


def _validate_kv_pool_preflight(
    vllm_config: VllmConfig,
    backend_name: str,
    *,
    use_layerwise: bool,
) -> None:
    if backend_name == "yuanrong":
        raise ValueError("AscendStore v1 temporarily does not support the Yuanrong Backend")
    resolve_backend_spec(backend_name)

    if _kvpp_size(vllm_config) > 1:
        raise ValueError("AscendStore v1 does not support active KVPP until KVPP ownership is represented")

    transfer_config = vllm_config.kv_transfer_config
    parallel_config = vllm_config.parallel_config
    dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
    pcp_size = getattr(parallel_config, "prefill_context_parallel_size", 1)
    if (
        use_layerwise
        and transfer_config.kv_role in ("kv_producer", "kv_consumer")
        and infer_dcp_mismatch_info(
            transfer_config.kv_role,
            transfer_config.kv_connector_extra_config,
            dcp_size,
            pcp_size,
        )
    ):
        peer_role = "prefill" if transfer_config.kv_role == "kv_consumer" else "decode"
        raise ValueError(
            "Decode-context-parallel mismatch in PD-disaggregation "
            f"(local dcp_size={dcp_size}, local pcp_size={pcp_size}, peer role={peer_role}) "
            "is not supported with layerwise KV transfer. Both the producer and consumer must use "
            "the same dcp_size/pcp_size so the layerwise GVA shard layout is consistent."
        )

    protocol = get_layerwise_protocol(backend_name)
    validate_layerwise_topology(protocol, parallel_config, use_layerwise)


def _validate_bulk_topology_support(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    *,
    use_layerwise: bool,
    has_private_state: bool | None = None,
) -> None:
    """Reject Bulk compositions whose identity or writer ownership is unproven."""

    cache_groups = kv_cache_config.kv_cache_groups
    transfer_group_ids, _ = _resolve_transfer_group_ids(kv_cache_config)
    if has_private_state is None:
        has_private_state = len(transfer_group_ids) != len(cache_groups)

    if use_layerwise:
        if any(kv_cache_spec_uses_align_state(cache_groups[group_id].kv_cache_spec) for group_id in transfer_group_ids):
            raise ValueError("AscendStore v1 Layerwise transfer does not support Mamba align-state groups")
    else:
        for group_id in transfer_group_ids:
            spec = cache_groups[group_id].kv_cache_spec
            if kv_cache_spec_contains_mamba(spec) and not kv_cache_spec_uses_align_state(spec):
                raise ValueError("AscendStore v1 Bulk supports Mamba state only in mamba_cache_mode='align'")
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None or not callable(getattr(model_config, "get_total_num_kv_heads", None)):
        # Lightweight Scheduler unit fixtures omit Worker-only model geometry.
        # Real VllmConfig instances always carry it, and Worker resolution
        # validates the same boundary independently.
        return
    tp_partition = _resolve_tp_partition(vllm_config)
    if use_layerwise:
        if tp_partition.tp_mismatch:
            raise ValueError("AscendStore v1 Layerwise transfer does not support TP mismatch")
        return

    parallel_config = vllm_config.parallel_config
    pcp_size = int(getattr(parallel_config, "prefill_context_parallel_size", 1))
    dcp_size = int(getattr(parallel_config, "decode_context_parallel_size", 1))
    tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1))
    num_kv_heads = 1 if getattr(model_config, "use_mla", False) else model_config.get_total_num_kv_heads()
    put_step = tp_size // num_kv_heads if num_kv_heads < tp_size else 1
    if dcp_size > 1 and pcp_size > 1:
        raise ValueError("AscendStore v1 Bulk does not support combined DCP and PCP writer partitioning")
    if dcp_size > 1 and put_step > dcp_size:
        raise ValueError("AscendStore v1 Bulk does not support DCP with remaining same-key TP writer replicas")

    if not tp_partition.tp_mismatch:
        return
    hf_text_config = getattr(model_config, "hf_text_config", None)
    if hf_text_config is not None and hasattr(hf_text_config, "index_topk"):
        raise ValueError("AscendStore v1 TP mismatch does not support sparse KV layouts")
    if has_private_state or len(cache_groups) != 1 or transfer_group_ids != (0,):
        raise ValueError("AscendStore v1 TP mismatch requires one dense transferable KV cache group")
    partitions = resolve_consumer_pipeline_partitions(vllm_config)
    if partitions is not None and len(partitions) > 1:
        raise ValueError("AscendStore v1 TP mismatch cannot be composed with Consumer pipeline Store")


def _resolve_transfer_group_ids(kv_cache_config: KVCacheConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Project explicitly enabled groups onto prefix-cacheable external objects."""

    cache_groups = kv_cache_config.kv_cache_groups
    try:
        cacheable_group_ids = tuple(infer_cacheable_group_ids(cache_groups))
    except AssertionError as error:
        raise ValueError("AscendStore requires at least one prefix-cacheable KV cache group") from error
    cacheable = frozenset(cacheable_group_ids)
    transfer_group_ids = tuple(group_id for group_id in kv_cache_config.transfer_group_ids if group_id in cacheable)
    if not transfer_group_ids:
        raise ValueError("AscendStore requires at least one transferable prefix-cacheable KV cache group")
    return transfer_group_ids, cacheable_group_ids


def _kvpp_size(vllm_config: VllmConfig) -> int:
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if not additional_config.get("enable_kvpp", False):
        return 1

    # Ascend configuration imports hardware discovery state, so keep the
    # authoritative parser lazy on the only route that needs it.
    from vllm_ascend.ascend_config import KVPPConfig

    return KVPPConfig.from_vllm_config(vllm_config).size


def _uses_layerwise_buffer_reuse(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> bool:
    layout_config = get_layerwise_reuse_config(vllm_config.kv_transfer_config)
    if layout_config is None:
        return False
    base_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    return build_layerwise_reuse_layout(
        get_layerwise_kv_cache_specs(kv_cache_config),
        base_layers,
        layout_config,
    ).has_layer_reuse


def resolve_consumer_pipeline_partitions(vllm_config: VllmConfig) -> tuple[int, ...] | None:
    """Resolve the producer PP layer ranges required by a consumer that stores KV."""

    transfer_config = vllm_config.kv_transfer_config
    extra_config = transfer_config.kv_connector_extra_config
    if transfer_config.kv_role != "kv_consumer" or not extra_config.get("consumer_is_to_put", False):
        return None

    num_hidden_layers = vllm_config.model_config.hf_text_config.num_hidden_layers
    prefill_pp_size = int(extra_config.get("prefill_pp_size", 1))
    if prefill_pp_size <= 0:
        raise ValueError(f"prefill_pp_size must be positive, received {prefill_pp_size}")

    partition_config = extra_config.get("prefill_pp_layer_partition")
    if partition_config is not None:
        return _parse_pipeline_partitions(partition_config, prefill_pp_size, num_hidden_layers)

    layers_per_partition, remaining_layers = divmod(num_hidden_layers, prefill_pp_size)
    partitions = [layers_per_partition] * prefill_pp_size
    for index in range(2, remaining_layers + 2):
        partitions[-index] += 1
    return tuple(partitions)


def _resolve_kv_pool_topology(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> KVPoolTopology:
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    tp_rank = get_tp_group().rank_in_group
    pp_rank = get_pp_group().rank_in_group
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    pcp_size = getattr(parallel_config, "prefill_context_parallel_size", 1)
    pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
    dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
    dcp_rank = get_dcp_group().rank_in_group if dcp_size > 1 else 0
    num_kv_heads = 1 if getattr(model_config, "use_mla", False) else model_config.get_total_num_kv_heads()
    put_step = tp_size // num_kv_heads if num_kv_heads < tp_size else 1
    head_or_tp_rank = tp_rank // put_step
    cache_transfer_granularity, hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    model_name = model_config.model.rstrip("/").split("/")[-1]
    base_layer_count = model_config.get_total_num_hidden_layers()
    hf_text_config = getattr(model_config, "hf_text_config", None)
    hf_config = getattr(model_config, "hf_config", hf_text_config)
    cache_family_hf_config = hf_text_config or hf_config
    compress_ratios = getattr(hf_text_config, "compress_ratios", None)
    if compress_ratios is None:
        compress_ratios = getattr(hf_config, "compress_ratios", None)
    kv_cache_groups = kv_cache_config.kv_cache_groups
    transfer_group_ids, _ = _resolve_transfer_group_ids(kv_cache_config)
    group_cache_families = infer_group_cache_families(
        kv_cache_groups,
        compress_ratios,
        cache_family_hf_config,
    )
    groups = []
    for group_id, group in enumerate(kv_cache_groups):
        kv_cache_spec = resolve_dcp_kv_cache_spec(group.kv_cache_spec, dcp_size)
        uses_align_state = kv_cache_spec_uses_align_state(kv_cache_spec)
        key_tp_rank = tp_rank if uses_align_state else head_or_tp_rank
        groups.append(
            KVPoolGroupTopology(
                group_id=group_id,
                kv_cache_spec=kv_cache_spec,
                layers=resolve_group_layers(group.layer_names, base_layer_count),
                key_metadata=KeyMetadata(
                    model_name,
                    key_tp_rank,
                    dcp_rank,
                    pp_rank,
                    kv_cache_group_id=group_id,
                    cache_family=group_cache_families[group_id],
                ),
                is_eagle_group=group.is_eagle_group,
            )
        )
    return KVPoolTopology(
        tp_rank,
        tp_size,
        pp_size,
        pcp_rank,
        pcp_size,
        dcp_size,
        put_step,
        cache_transfer_granularity,
        hash_block_size,
        _resolve_tp_partition(vllm_config),
        tuple(groups),
        transfer_group_ids,
        resolve_consumer_pipeline_partitions(vllm_config),
    )


def _resolve_tp_partition(vllm_config: VllmConfig) -> TPPartitionSpec:
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    tp_size = parallel_config.tensor_parallel_size
    use_mla = getattr(model_config, "use_mla", False)
    num_kv_heads = 1 if use_mla else model_config.get_total_num_kv_heads()
    mismatch_info = infer_tp_mismatch_info(
        vllm_config.kv_transfer_config.kv_role,
        vllm_config.kv_transfer_config.kv_connector_extra_config,
        tp_size,
        num_kv_heads,
        use_mla,
    )
    if mismatch_info.peer_tp_size != tp_size and not mismatch_info.enabled:
        raise ValueError(
            "AscendStore v1 cannot represent the configured TP mismatch; it requires a non-MLA "
            "dense KV layout whose head count is divisible by the effective TP size"
        )
    key_rank_count = mismatch_info.effective_tp_size if mismatch_info.enabled else min(tp_size, num_kv_heads)
    return TPPartitionSpec(mismatch_info.enabled, key_rank_count, mismatch_info.num_sub_keys)


def _store_enabled(vllm_config: VllmConfig, extra_config: dict[str, Any]) -> bool:
    transfer_config = vllm_config.kv_transfer_config
    return transfer_config.kv_role in ("kv_producer", "kv_both") or bool(extra_config.get("consumer_is_to_put", False))


def _resolve_layerwise_prefetch_layers(extra_config: dict[str, Any], use_layerwise: bool) -> int:
    default_prefetch_layers = 2
    if not use_layerwise:
        return default_prefetch_layers
    configured_prefetch_layers = extra_config.get("layerwise_prefetch_layers", default_prefetch_layers)
    if isinstance(configured_prefetch_layers, bool):
        raise ValueError("layerwise_prefetch_layers must be a positive integer")
    try:
        prefetch_layers = int(configured_prefetch_layers)
    except (TypeError, ValueError) as error:
        raise ValueError("layerwise_prefetch_layers must be a positive integer") from error
    if prefetch_layers <= 0:
        raise ValueError("layerwise_prefetch_layers must be a positive integer")
    return prefetch_layers


def _parse_pipeline_partitions(
    partition_config: str,
    prefill_pp_size: int,
    num_hidden_layers: int,
) -> tuple[int, ...]:
    try:
        partitions = tuple(int(layer_count) for layer_count in partition_config.split(","))
    except ValueError as error:
        raise ValueError(f"Invalid prefill PP layer partition: {partition_config}") from error
    if len(partitions) != prefill_pp_size:
        raise ValueError(f"Partition count {len(partitions)} does not match prefill_pp_size {prefill_pp_size}")
    if sum(partitions) != num_hidden_layers:
        raise ValueError(f"Partition layer count {sum(partitions)} does not match model layers {num_hidden_layers}")
    return partitions


def _uses_eagle(vllm_config: VllmConfig) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    # Match production's external-cache policy, including when local block drop is disabled.
    use_eagle = getattr(speculative_config, "use_eagle", None)
    return use_eagle() is True if callable(use_eagle) else False
