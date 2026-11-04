"""Parse AscendStore options and reuse vLLM's static cache geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_cache_spec
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    infer_dcp_mismatch_info,
    infer_tp_mismatch_info,
    uses_hybrid_kv_cache,
)

from ...layerwise.layerwise_cache_layout import (
    build_layerwise_reuse_layout,
    get_layerwise_kv_cache_specs,
    get_layerwise_reuse_config,
)
from ..backend import BACKEND_IMPORTS, LAYERWISE_OBJECT_FORMATS
from .topology import CPPartitionSpec, PPPartitionSpec, TPPartitionSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass(frozen=True, slots=True)
class KVPoolConfig:
    """Parsed execution options and static cache facts shared by Scheduler and Worker.

    Group specs retain original cache-group indexing and are already adjusted for DCP.
    """

    # Backend and execution.
    backend_name: str
    use_layerwise: bool
    load_async: bool
    layerwise_prefetch_layers: int

    # Load and Store policy.
    load_enabled: bool
    store_enabled: bool
    save_decode_cache: bool
    discard_partial_chunks: bool

    # Model and cache groups.
    model_name: str
    use_hybrid: bool
    use_eagle: bool
    transfer_group_ids: tuple[int, ...]
    group_specs: tuple[KVCacheSpec, ...]
    align_state_group_ids: frozenset[int]
    eagle_group_ids: frozenset[int]

    # Parallelism and object partitions.
    tp_size: int
    pp_size: int
    expected_worker_count: int
    tp_partition: TPPartitionSpec
    pp_partition: PPPartitionSpec
    cp_partition: CPPartitionSpec
    put_step: int

    # Cache granularity and retention.
    cache_transfer_granularity: int
    hash_block_size: int
    retention_interval: int | None

    # Lookup RPC.
    lookup_rpc_timeout_ms: int
    store_prefill_enabled: bool = True
    enable_kv_events: bool = False

    @property
    def has_private_state(self) -> bool:
        return len(self.transfer_group_ids) != len(self.group_specs)


def resolve_kv_pool_config(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> KVPoolConfig:
    transfer_config = vllm_config.kv_transfer_config
    extra_config = transfer_config.kv_connector_extra_config
    backend_name = extra_config.get("backend", "mooncake")
    if not isinstance(backend_name, str):
        raise ValueError("backend must be a string")
    backend_name = backend_name.strip().lower()
    if backend_name not in BACKEND_IMPORTS:
        raise ValueError(f"Unsupported AscendStore v1 backend: {backend_name}")
    use_layerwise = _boolean_option(extra_config, "use_layerwise")
    load_async = _boolean_option(extra_config, "load_async")
    consumer_load = _boolean_option(extra_config, "consumer_is_to_load")
    consumer_store = _boolean_option(extra_config, "consumer_is_to_put")
    save_decode_cache = _boolean_option(extra_config, "save_decode_cache")
    prefetch_layers = _resolve_layerwise_prefetch_layers(extra_config, use_layerwise)
    discard_partial_chunks = _boolean_option(extra_config, "discard_partial_chunks", default=True)
    load_enabled = transfer_config.kv_role != "kv_consumer" or consumer_load
    store_prefill_enabled = transfer_config.kv_role in ("kv_producer", "kv_both") or consumer_store
    store_enabled = store_prefill_enabled or save_decode_cache
    lookup_rpc_timeout_ms = extra_config.get("lookup_rpc_timeout_ms", -1)

    model_config = vllm_config.model_config
    model_name = model_config.model.rstrip("/").split("/")[-1]
    cache_groups = kv_cache_config.kv_cache_groups
    use_hybrid = uses_hybrid_kv_cache(vllm_config.scheduler_config, cache_groups)
    speculative_config = vllm_config.speculative_config
    use_eagle = speculative_config is not None and speculative_config.use_eagle()
    transfer_group_ids = _resolve_transfer_group_ids(kv_cache_config)
    parallel_config = vllm_config.parallel_config
    dcp_size = parallel_config.decode_context_parallel_size
    group_specs = tuple(resolve_dcp_kv_cache_spec(group.kv_cache_spec, dcp_size) for group in cache_groups)
    align_state_group_ids = frozenset(
        group_id for group_id, spec in enumerate(group_specs) if kv_cache_spec_uses_align_state(spec)
    )
    eagle_group_ids = frozenset(group_id for group_id, group in enumerate(cache_groups) if group.is_eagle_group)

    use_mla = model_config.use_mla
    num_kv_heads = 1 if use_mla else model_config.get_total_num_kv_heads()
    tp_size = parallel_config.tensor_parallel_size
    put_step = tp_size // num_kv_heads if num_kv_heads < tp_size else 1
    tp_partition = resolve_tp_partition(vllm_config, num_kv_heads, use_mla)
    partitions = resolve_consumer_pipeline_partitions(vllm_config, store_enabled=store_enabled)
    pp_partition, cp_partition = resolve_object_partitions(
        vllm_config, backend_name, use_layerwise=use_layerwise, put_step=put_step, target_layer_counts=partitions
    )
    cache_transfer_granularity, hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    return KVPoolConfig(
        backend_name=backend_name,
        use_layerwise=use_layerwise,
        load_async=load_async,
        layerwise_prefetch_layers=prefetch_layers,
        load_enabled=load_enabled,
        store_enabled=store_enabled,
        save_decode_cache=save_decode_cache,
        discard_partial_chunks=discard_partial_chunks,
        model_name=model_name,
        use_hybrid=use_hybrid,
        use_eagle=use_eagle,
        transfer_group_ids=transfer_group_ids,
        group_specs=group_specs,
        align_state_group_ids=align_state_group_ids,
        eagle_group_ids=eagle_group_ids,
        tp_size=tp_size,
        pp_size=parallel_config.pipeline_parallel_size,
        expected_worker_count=parallel_config.world_size,
        tp_partition=tp_partition,
        pp_partition=pp_partition,
        cp_partition=cp_partition,
        put_step=put_step,
        cache_transfer_granularity=cache_transfer_granularity,
        hash_block_size=hash_block_size,
        retention_interval=kv_cache_config.prefix_cache_retention_interval,
        lookup_rpc_timeout_ms=lookup_rpc_timeout_ms,
        store_prefill_enabled=store_prefill_enabled,
        enable_kv_events=bool(
            getattr(vllm_config, "kv_events_config", None) and vllm_config.kv_events_config.enable_kv_cache_events
        ),
    )


def _boolean_option(extra_config: dict[str, Any], name: str, *, default: bool = False) -> bool:
    value = extra_config.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _resolve_layerwise_prefetch_layers(extra_config: dict[str, Any], use_layerwise: bool) -> int:
    default_prefetch_layers = 2
    if not use_layerwise:
        return default_prefetch_layers
    return _positive_integer(
        extra_config.get("layerwise_prefetch_layers", default_prefetch_layers), "layerwise_prefetch_layers"
    )


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


def _resolve_transfer_group_ids(kv_cache_config: KVCacheConfig) -> tuple[int, ...]:
    """Project explicitly enabled groups onto prefix-cacheable external objects."""

    cache_groups = kv_cache_config.kv_cache_groups
    transfer_group_ids = tuple(
        group_id
        for group_id in kv_cache_config.transfer_group_ids
        if is_prefix_cacheable(cache_groups[group_id].kv_cache_spec)
    )
    if not transfer_group_ids:
        raise ValueError("AscendStore requires at least one transferable prefix-cacheable KV cache group")
    return transfer_group_ids


def kv_cache_spec_uses_align_state(kv_cache_spec: KVCacheSpec) -> bool:
    """Whether an upstream cache spec contains mutable Mamba align state."""

    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        specs = kv_cache_spec.kv_cache_specs.values()
    else:
        specs = (kv_cache_spec,)
    return any(isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align" for spec in specs)


def resolve_tp_partition(vllm_config: VllmConfig, num_kv_heads: int, use_mla: bool) -> TPPartitionSpec:
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
    # Match vLLM: spare the last stage (output layers), then fill preceding stages backwards.
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
    total_layers = sum(partitions)
    if total_layers != num_hidden_layers:
        raise ValueError(f"Partition layer count {total_layers} does not match model layers {num_hidden_layers}")
    return partitions


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


def validate_kv_pool_config(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, config: KVPoolConfig) -> None:
    """Check compositions against the same resolved facts that execution will consume."""
    if _kvpp_size(vllm_config) > 1:
        raise ValueError("AscendStore v1 does not support active KVPP until KVPP ownership is represented")

    transfer_config = vllm_config.kv_transfer_config
    if config.use_layerwise and config.store_enabled and not config.store_prefill_enabled:
        raise ValueError("Decode-only Store requires Bulk transfer; enable consumer_is_to_put for Layerwise Store")
    dcp_size = config.cp_partition.dcp_size
    pcp_size = config.cp_partition.pcp_size
    if (
        config.use_layerwise
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

    if config.use_layerwise and config.backend_name == "mooncake" and (dcp_size > 1 or pcp_size > 1):
        raise ValueError("Mooncake Layerwise does not support context parallelism")
    if not config.discard_partial_chunks:
        raise ValueError(
            "AscendStore v1 does not support discard_partial_chunks=False until "
            "native partial-object identity and transfer are implemented"
        )
    if config.use_layerwise and _uses_layerwise_buffer_reuse(vllm_config, kv_cache_config):
        raise ValueError("AscendStore v1 Layerwise buffer reuse requires native partial-object support")

    # Every transferred logical block needs a request hash at its ending
    # boundary. Kernel-block segmentation only changes the memory layout.
    for group_id in config.transfer_group_ids:
        block_size = config.group_specs[group_id].block_size
        if block_size % config.hash_block_size:
            raise ValueError(
                f"AscendStore transfer group {group_id} block size ({block_size}) must be a multiple "
                f"of hash block size ({config.hash_block_size})"
            )

    partitions = config.pp_partition.target_layer_counts
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
        if _kv_cache_spec_contains_mamba(config.group_specs[group_id]) and group_id not in config.align_state_group_ids:
            raise ValueError("AscendStore v1 Bulk supports Mamba state only in mamba_cache_mode='align'")
    if dcp_size > 1 and pcp_size > 1:
        raise ValueError("AscendStore v1 Bulk does not support combined DCP and PCP writer partitioning")
    if config.put_step > dcp_size > 1:
        raise ValueError("AscendStore v1 Bulk does not support DCP with remaining same-key TP writer replicas")
    if not config.tp_partition.tp_mismatch:
        return
    if hasattr(vllm_config.model_config.hf_text_config, "index_topk"):
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


def _kv_cache_spec_contains_mamba(kv_cache_spec: KVCacheSpec) -> bool:
    """Whether a cache group contains any recurrent Mamba state."""

    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        specs = kv_cache_spec.kv_cache_specs.values()
    else:
        specs = (kv_cache_spec,)
    return any(isinstance(spec, MambaSpec) for spec in specs)
