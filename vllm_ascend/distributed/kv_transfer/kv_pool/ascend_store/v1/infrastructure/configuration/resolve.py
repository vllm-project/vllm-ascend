"""Resolve and admit static configuration before binding ranks or acquiring resources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import regex as re
import torch.distributed as dist
import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.distributed import get_dcp_group, get_pcp_group, get_pp_group, get_tp_group
from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_cache_spec
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    KeyMetadata,
    infer_cacheable_group_ids,
    infer_dcp_mismatch_info,
    infer_group_cache_families,
    infer_tp_mismatch_info,
)

from ....layerwise.layerwise_cache_layout import (
    build_layerwise_reuse_layout,
    get_layerwise_kv_cache_specs,
    get_layerwise_reuse_config,
)
from ...backend import BACKEND_IMPORTS, LAYERWISE_OBJECT_FORMATS
from .topology import (
    ContextRank,
    CPPartitionSpec,
    KVPoolGroupTopology,
    KVPoolLayerTopology,
    KVPoolTopology,
    PPPartitionSpec,
    TPPartitionSpec,
    context_replica_groups,
    kv_cache_spec_uses_align_state,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass(frozen=True, slots=True)
class KVPoolConfig:
    """Admitted execution choices and geometry, shared by both process-side factories.

    Group specs retain original cache-group indexing and are already adjusted for DCP.
    No rank, registered tensor, Backend client, or execution owner is needed here.
    """

    backend_name: str
    use_layerwise: bool
    load_async: bool
    load_enabled: bool
    store_enabled: bool
    save_decode_cache: bool
    use_eagle: bool
    layerwise_prefetch_layers: int
    lookup_rpc_timeout_ms: int
    transfer_group_ids: tuple[int, ...]
    group_specs: tuple[KVCacheSpec, ...]
    align_state_group_ids: frozenset[int]
    cache_transfer_granularity: int
    hash_block_size: int
    tp_partition: TPPartitionSpec
    pp_partition: PPPartitionSpec
    cp_partition: CPPartitionSpec
    put_step: int
    retention_interval: int | None

    @property
    def consumer_pipeline_partitions(self) -> tuple[int, ...] | None:
        return self.pp_partition.target_layer_counts

    @property
    def has_private_state(self) -> bool:
        return len(self.transfer_group_ids) != len(self.group_specs)


# =============================================================================
# Static Configuration Admission
# =============================================================================
#
# Both process-side factories run this same sequence. Native client creation,
# process ranks, and resource acquisition follow these configuration-only checks.


def resolve_kv_pool_config(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> KVPoolConfig:
    transfer_config = vllm_config.kv_transfer_config
    extra_config = transfer_config.kv_connector_extra_config
    backend_name = extra_config.get("backend", "mooncake")
    if not isinstance(backend_name, str):
        raise ValueError("backend must be a string")
    backend_name = backend_name.strip().lower()
    use_layerwise = _boolean_option(extra_config, "use_layerwise")
    _validate_kv_pool_preflight(vllm_config, backend_name, use_layerwise=use_layerwise)

    load_async = _boolean_option(extra_config, "load_async")
    consumer_load = _boolean_option(extra_config, "consumer_is_to_load")
    consumer_store = _boolean_option(extra_config, "consumer_is_to_put")
    save_decode_cache = _boolean_option(extra_config, "save_decode_cache")
    prefetch_layers = _resolve_layerwise_prefetch_layers(extra_config, use_layerwise)
    _validate_partial_object_support(
        vllm_config,
        kv_cache_config,
        use_layerwise=use_layerwise,
        discard_partial_chunks=_boolean_option(extra_config, "discard_partial_chunks", default=True),
    )
    transfer_group_ids = _resolve_transfer_group_ids(kv_cache_config)
    parallel_config = vllm_config.parallel_config
    dcp_size = int(getattr(parallel_config, "decode_context_parallel_size", 1))
    group_specs = tuple(
        resolve_dcp_kv_cache_spec(group.kv_cache_spec, dcp_size) for group in kv_cache_config.kv_cache_groups
    )
    align_state_group_ids = frozenset(
        group_id for group_id, spec in enumerate(group_specs) if kv_cache_spec_uses_align_state(spec)
    )
    model_config = vllm_config.model_config
    use_mla = bool(getattr(model_config, "use_mla", False))
    num_kv_heads = 1 if use_mla else model_config.get_total_num_kv_heads()
    tp_size = parallel_config.tensor_parallel_size
    put_step = tp_size // num_kv_heads if num_kv_heads < tp_size else 1
    store_enabled = transfer_config.kv_role in ("kv_producer", "kv_both") or consumer_store
    tp_partition = _resolve_tp_partition(vllm_config, num_kv_heads, use_mla)
    partitions = resolve_consumer_pipeline_partitions(vllm_config, store_enabled=store_enabled)
    pp_partition, cp_partition = resolve_object_partitions(
        vllm_config, backend_name, use_layerwise=use_layerwise, put_step=put_step, target_layer_counts=partitions
    )
    cache_transfer_granularity, hash_block_size = _resolve_kv_pool_block_sizes(
        vllm_config, kv_cache_config, group_specs, transfer_group_ids
    )
    config = KVPoolConfig(
        backend_name=backend_name,
        use_layerwise=use_layerwise,
        load_async=load_async,
        load_enabled=transfer_config.kv_role != "kv_consumer" or consumer_load,
        store_enabled=store_enabled,
        save_decode_cache=save_decode_cache,
        use_eagle=_uses_eagle(vllm_config),
        layerwise_prefetch_layers=prefetch_layers,
        lookup_rpc_timeout_ms=extra_config.get("lookup_rpc_timeout_ms", -1),
        transfer_group_ids=transfer_group_ids,
        group_specs=group_specs,
        align_state_group_ids=align_state_group_ids,
        cache_transfer_granularity=cache_transfer_granularity,
        hash_block_size=hash_block_size,
        tp_partition=tp_partition,
        pp_partition=pp_partition,
        cp_partition=cp_partition,
        put_step=put_step,
        retention_interval=getattr(kv_cache_config, "prefix_cache_retention_interval", None),
    )
    _validate_kv_pool_topology_support(vllm_config, config)
    return config


def resolve_object_partitions(
    vllm_config: VllmConfig,
    backend_name: str,
    *,
    use_layerwise: bool,
    put_step: int,
    target_layer_counts: tuple[int, ...] | None,
) -> tuple[PPPartitionSpec, CPPartitionSpec]:
    """Translate wire requirements and rank-free parallel sizes into partition contracts."""
    parallel = vllm_config.parallel_config
    pcp_size = parallel.prefill_context_parallel_size
    dcp_size = parallel.decode_context_parallel_size
    if not use_layerwise:
        return PPPartitionSpec(target_layer_counts), CPPartitionSpec(pcp_size, dcp_size)
    object_format = LAYERWISE_OBJECT_FORMATS[backend_name]
    global_layers = vllm_config.model_config.get_total_num_hidden_layers() if object_format.global_layer_span else None
    shared = object_format.shared_context_shards and dcp_size > 1 and (put_step > 1 or pcp_size > 1)
    return (
        PPPartitionSpec(target_layer_counts, global_layers),
        CPPartitionSpec(
            pcp_size,
            dcp_size,
            dcp_size if shared else 1,
            object_format.context_alignment_bytes if shared else 1,
        ),
    )


def _validate_kv_pool_preflight(
    vllm_config: VllmConfig,
    backend_name: str,
    *,
    use_layerwise: bool,
) -> None:
    if backend_name == "yuanrong":
        raise ValueError("AscendStore v1 temporarily does not support the Yuanrong Backend")
    if backend_name not in BACKEND_IMPORTS:
        raise ValueError(f"Unsupported AscendStore v1 backend: {backend_name}")

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

    if use_layerwise and backend_name == "mooncake" and (dcp_size > 1 or pcp_size > 1):
        raise ValueError("Mooncake Layerwise does not support context parallelism")


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


def _validate_kv_pool_topology_support(vllm_config: VllmConfig, config: KVPoolConfig) -> None:
    """Check compositions against the same resolved facts that execution will consume."""
    partitions = config.consumer_pipeline_partitions
    transferable_align_groups = config.align_state_group_ids.intersection(config.transfer_group_ids)
    if config.use_layerwise:
        if config.has_private_state:
            raise ValueError("AscendStore private KV state requires non-layerwise transfer")
        if transferable_align_groups:
            raise ValueError("AscendStore v1 Layerwise transfer does not support Mamba align-state groups")
        if partitions is not None and len(partitions) > 1:
            raise ValueError("AscendStore v1 Layerwise transfer cannot be composed with Consumer pipeline Store")
        if config.tp_partition.tp_mismatch:
            raise ValueError("AscendStore v1 Layerwise transfer does not support TP mismatch")
        return

    for group_id in config.transfer_group_ids:
        if kv_cache_spec_contains_mamba(config.group_specs[group_id]) and group_id not in config.align_state_group_ids:
            raise ValueError("AscendStore v1 Bulk supports Mamba state only in mamba_cache_mode='align'")
    parallel_config = vllm_config.parallel_config
    pcp_size = int(getattr(parallel_config, "prefill_context_parallel_size", 1))
    dcp_size = int(getattr(parallel_config, "decode_context_parallel_size", 1))
    if dcp_size > 1 and pcp_size > 1:
        raise ValueError("AscendStore v1 Bulk does not support combined DCP and PCP writer partitioning")
    if dcp_size > 1 and config.put_step > dcp_size:
        raise ValueError("AscendStore v1 Bulk does not support DCP with remaining same-key TP writer replicas")
    if not config.tp_partition.tp_mismatch:
        return
    hf_text_config = getattr(vllm_config.model_config, "hf_text_config", None)
    if hf_text_config is not None and hasattr(hf_text_config, "index_topk"):
        raise ValueError("AscendStore v1 TP mismatch does not support sparse KV layouts")
    if (
        config.has_private_state
        or len(config.group_specs) != 1
        or config.transfer_group_ids != (0,)
        or transferable_align_groups
    ):
        raise ValueError("AscendStore v1 TP mismatch requires one dense transferable KV cache group")
    if partitions is not None and len(partitions) > 1:
        raise ValueError("AscendStore v1 TP mismatch cannot be composed with Consumer pipeline Store")


def _resolve_kv_pool_block_sizes(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    group_specs: tuple[KVCacheSpec, ...],
    transfer_group_ids: tuple[int, ...],
) -> tuple[int, int]:
    """Consume vLLM scheduling and request-hash geometry without deriving another hash grain."""

    scheduler_block_size, hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    if scheduler_block_size <= 0 or hash_block_size <= 0 or scheduler_block_size % hash_block_size:
        raise ValueError("AscendStore scheduler block size must be a positive multiple of the hash block size")
    for group_id in transfer_group_ids:
        block_size = group_specs[group_id].block_size
        if block_size <= 0 or block_size % hash_block_size or scheduler_block_size % block_size:
            raise ValueError(
                f"AscendStore transfer group {group_id} block size ({block_size}) must be a positive multiple "
                f"of hash block size ({hash_block_size}) and divide scheduler block size ({scheduler_block_size})"
            )
    return scheduler_block_size, hash_block_size


def _resolve_transfer_group_ids(kv_cache_config: KVCacheConfig) -> tuple[int, ...]:
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
    return transfer_group_ids


def _resolve_tp_partition(vllm_config: VllmConfig, num_kv_heads: int, use_mla: bool) -> TPPartitionSpec:
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
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


def resolve_consumer_pipeline_partitions(vllm_config: VllmConfig, *, store_enabled: bool) -> tuple[int, ...] | None:
    """Resolve the producer PP layer ranges required by a consumer that stores KV."""

    transfer_config = vllm_config.kv_transfer_config
    extra_config = transfer_config.kv_connector_extra_config
    if transfer_config.kv_role != "kv_consumer" or not store_enabled:
        return None

    num_hidden_layers = vllm_config.model_config.hf_text_config.num_hidden_layers
    prefill_pp_size = _positive_integer(extra_config.get("prefill_pp_size", 1), "prefill_pp_size")

    partition_config = extra_config.get("prefill_pp_layer_partition")
    if partition_config is not None:
        return _parse_pipeline_partitions(partition_config, prefill_pp_size, num_hidden_layers)

    layers_per_partition, remaining_layers = divmod(num_hidden_layers, prefill_pp_size)
    partitions = [layers_per_partition] * prefill_pp_size
    for index in range(2, remaining_layers + 2):
        partitions[-index] += 1
    return tuple(partitions)


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
    if any(layer_count < 0 for layer_count in partitions):
        raise ValueError("Prefill PP layer counts must be nonnegative")
    if sum(partitions) != num_hidden_layers:
        raise ValueError(f"Partition layer count {sum(partitions)} does not match model layers {num_hidden_layers}")
    return partitions


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


def _uses_eagle(vllm_config: VllmConfig) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    # Match production's external-cache policy, including when local block drop is disabled.
    use_eagle = getattr(speculative_config, "use_eagle", None)
    return use_eagle() is True if callable(use_eagle) else False


def kv_cache_spec_contains_mamba(kv_cache_spec: KVCacheSpec) -> bool:
    """Whether a cache group contains any recurrent Mamba state."""

    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        specs = kv_cache_spec.kv_cache_specs.values()
    else:
        specs = (kv_cache_spec,)
    return any(isinstance(spec, MambaSpec) for spec in specs)


def _boolean_option(extra_config: dict[str, Any], name: str, *, default: bool = False) -> bool:
    value = extra_config.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _positive_integer(value: Any, name: str) -> int:
    # Keep decimal-string configuration compatibility without truncating floats or accepting bools.
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _resolve_layerwise_prefetch_layers(extra_config: dict[str, Any], use_layerwise: bool) -> int:
    default_prefetch_layers = 2
    if not use_layerwise:
        return default_prefetch_layers
    return _positive_integer(
        extra_config.get("layerwise_prefetch_layers", default_prefetch_layers), "layerwise_prefetch_layers"
    )


# =============================================================================
# Worker-local Topology Binding
# =============================================================================
#
# Static admission is complete before distributed ranks or layer identities are
# read. Reuse the admitted specs and partitions; do not rederive their geometry.


def resolve_worker_topology(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    config: KVPoolConfig,
) -> KVPoolTopology:
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    tp_group = get_tp_group()
    tp_rank = tp_group.rank_in_group
    pp_rank = get_pp_group().rank_in_group
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    pcp_size = config.cp_partition.pcp_size
    pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
    dcp_size = config.cp_partition.dcp_size
    dcp_rank = get_dcp_group().rank_in_group if dcp_size > 1 else 0
    context_ranks: tuple[ContextRank, ...] = ()
    if config.use_layerwise and config.store_enabled:
        context_ranks = (
            _gather_context_ranks(tp_group, get_pcp_group() if pcp_size > 1 else None, dcp_rank)
            if dcp_size > 1
            else tuple(ContextRank(rank, index, 0) for index, rank in enumerate(tp_group.ranks))
        )
    head_or_tp_rank = tp_rank // config.put_step
    model_name = model_config.model.rstrip("/").split("/")[-1]
    base_layer_count = model_config.get_total_num_hidden_layers()
    hf_text_config = getattr(model_config, "hf_text_config", None)
    hf_config = getattr(model_config, "hf_config", hf_text_config)
    cache_family_hf_config = hf_text_config or hf_config
    compress_ratios = getattr(hf_text_config, "compress_ratios", None)
    if compress_ratios is None:
        compress_ratios = getattr(hf_config, "compress_ratios", None)
    kv_cache_groups = kv_cache_config.kv_cache_groups
    group_cache_families = infer_group_cache_families(
        kv_cache_groups,
        compress_ratios,
        cache_family_hf_config,
    )
    groups = []
    for group_id, group in enumerate(kv_cache_groups):
        kv_cache_spec = config.group_specs[group_id]
        uses_align_state = group_id in config.align_state_group_ids
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
    stage_start = 0
    stage_count = len(
        {layer.physical_layer_id for group_id in config.transfer_group_ids for layer in groups[group_id].layers}
    )
    if config.pp_partition.global_layer_count is not None:
        stage_start, _ = model_config.get_layers_start_end_indices(parallel_config)
        stage_count = max(model_config.get_num_layers(parallel_config), stage_count)
    return KVPoolTopology(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_size=pp_size,
        pcp_rank=pcp_rank,
        put_step=config.put_step,
        cache_transfer_granularity=config.cache_transfer_granularity,
        hash_block_size=config.hash_block_size,
        tp_partition=config.tp_partition,
        pp_partition=config.pp_partition,
        cp_partition=config.cp_partition,
        groups=tuple(groups),
        transfer_group_ids=config.transfer_group_ids,
        stage_layer_start=stage_start,
        stage_layer_count=stage_count,
        object_shard_index=dcp_rank if config.cp_partition.object_shard_count > 1 else 0,
        global_rank=tp_group.ranks[tp_rank],
        context_global_ranks=tuple(rank.global_rank for rank in context_ranks),
        context_replica_groups=context_replica_groups(context_ranks, tp_rank, config.put_step),
    )


def _gather_context_ranks(
    tp_group: GroupCoordinator, pcp_group: GroupCoordinator | None, dcp_rank: int
) -> tuple[ContextRank, ...]:
    """Snapshot TP x PCP membership before any background Store can use collectives.

    DCP may traverse PCP before TP. Exchange the actual coordinates instead of
    inferring them from rank arithmetic. Model CPU groups are used only during
    initialization; Store execution later owns a separate communication channel.
    """
    ranks: tuple[ContextRank, ...] = (
        ContextRank(tp_group.ranks[tp_group.rank_in_group], tp_group.rank_in_group, dcp_rank),
    )
    for group in (pcp_group, tp_group):
        if group is None or len(group.ranks) == 1:
            continue
        gathered: list[tuple[ContextRank, ...] | None] = [None] * len(group.ranks)
        dist.all_gather_object(gathered, ranks, group=group.cpu_group)
        ranks = tuple(rank for peers in gathered for rank in cast(tuple[ContextRank, ...], peers))
    return tuple(sorted(ranks, key=lambda rank: rank.global_rank))


def resolve_group_layers(layer_names: list[str], base_layer_count: int) -> tuple[KVPoolLayerTopology, ...]:
    names_by_physical_layer: dict[int, list[str]] = {}
    for layer_name in layer_names:
        physical_layer_id = _resolve_physical_layer_id(layer_name, base_layer_count)
        names_by_physical_layer.setdefault(physical_layer_id, []).append(layer_name)
    return tuple(
        KVPoolLayerTopology(physical_layer_id, tuple(sorted(names)))
        for physical_layer_id, names in sorted(names_by_physical_layer.items())
    )


def _resolve_physical_layer_id(layer_name: str, base_layer_count: int) -> int:
    mtp_layer = re.search(r"(?:^|\.)mtp(?:\.layers)?\.(\d+)(?:\.|$)", layer_name)
    if mtp_layer is not None:
        return base_layer_count + int(mtp_layer.group(1))
    model_layer = re.search(r"layers\.(\d+)", layer_name)
    if model_layer is not None:
        return int(model_layer.group(1))
    first_number = re.search(r"\d+", layer_name)
    if first_number is not None:
        return int(first_number.group())
    raise ValueError(f"Cannot resolve a physical layer from cache entry {layer_name!r}")
