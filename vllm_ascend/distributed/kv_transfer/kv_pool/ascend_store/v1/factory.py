"""Assemble Scheduler and Worker owners from admitted AscendStore configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import vllm.envs as vllm_envs
from vllm.logger import logger
from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import infer_dcp_mismatch_info

from ..layerwise.layerwise_cache_layout import (
    build_layerwise_reuse_layout,
    get_layerwise_kv_cache_specs,
    get_layerwise_reuse_config,
)
from .backend import LayerwiseAccessKind, resolve_backend_spec
from .infrastructure.configuration import KVPoolConfig, resolve_kv_pool_config
from .infrastructure.coordinator import Coordinator
from .infrastructure.resources import Resources
from .infrastructure.topology import resolve_worker_topology
from .protocol.lookup import LookupClient
from .scheduler import (
    AsynchronousBulkScheduler,
    KVPoolScheduler,
    LayerwiseScheduler,
    SynchronousBulkScheduler,
)
from .worker.base import KVPoolWorker
from .worker.bulk import BulkWorker
from .worker.layerwise import GVALayerwiseWorker, KeyRangeLayerwiseWorker
from .worker.layerwise.gva.collective import StoreCollective

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec


def create_kv_pool_scheduler(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, lookup_address: str
) -> KVPoolScheduler:
    config = resolve_kv_pool_config(vllm_config, kv_cache_config)
    _validate_support(vllm_config, kv_cache_config, config)
    scheduler_type: type[KVPoolScheduler]
    if config.use_layerwise:
        scheduler_type = LayerwiseScheduler
    elif config.load_async:
        scheduler_type = AsynchronousBulkScheduler
    else:
        scheduler_type = SynchronousBulkScheduler
    return scheduler_type(config, LookupClient(lookup_address, timeout_ms=config.lookup_rpc_timeout_ms))


def create_kv_pool_worker(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> KVPoolWorker:
    """Acquire resources only after common configuration and local topology are resolved."""
    config = resolve_kv_pool_config(vllm_config, kv_cache_config)
    _validate_support(vllm_config, kv_cache_config, config)
    topology = resolve_worker_topology(vllm_config, kv_cache_config, config)
    coordinator = Coordinator(
        topology.transfer_groups,
        config.cache_transfer_granularity,
        config.hash_block_size,
        vllm_config.model_config.max_model_len,
        use_eagle=config.use_eagle,
        retention_interval=config.retention_interval,
    )
    backend_spec = resolve_backend_spec(config.backend_name)
    if config.use_layerwise and backend_spec.layerwise_access is None:
        raise TypeError("Layerwise route requires a concrete Backend data plane")
    key_builder = None
    if config.use_layerwise and backend_spec.layerwise_key_factory is not None:
        key_builder = backend_spec.layerwise_key_factory(vllm_config, kv_cache_config, config)
    align_shared_storage = (
        config.use_layerwise and backend_spec.layerwise_access is LayerwiseAccessKind.GVA and config.use_hybrid
    )
    # Backend-private options remain owned by the selected Backend, not the common resolver.
    backend = backend_spec.create(
        vllm_config.parallel_config,
        extra_config=vllm_config.kv_transfer_config.kv_connector_extra_config,
    )
    resources: Resources | None = None
    try:
        resources = Resources(
            backend,
            backend_spec,
            topology.transfer_groups,
            align_shared_storage=align_shared_storage,
        )
        if config.use_layerwise:
            if backend_spec.layerwise_access is LayerwiseAccessKind.GVA:
                collective = None
                if config.store_enabled and config.cp_partition.object_shard_count > 1:
                    collective = StoreCollective(topology.context_global_ranks)
                    resources.bind_execution_cleanup(collective.close)
                return GVALayerwiseWorker(
                    topology,
                    resources,
                    store_enabled=config.store_enabled,
                    coordinator=coordinator,
                    kv_cache_layout=vllm_envs.VLLM_KV_CACHE_LAYOUT,
                    layerwise_prefetch_layers=config.layerwise_prefetch_layers,
                    key_builder=key_builder,
                    store_collective=collective,
                )
            return KeyRangeLayerwiseWorker(
                topology,
                resources,
                store_enabled=config.store_enabled,
                coordinator=coordinator,
                kv_cache_layout=vllm_envs.VLLM_KV_CACHE_LAYOUT,
                layerwise_prefetch_layers=config.layerwise_prefetch_layers,
                key_builder=key_builder,
            )
        return BulkWorker(
            topology,
            resources,
            coordinator=coordinator,
            kv_cache_layout=vllm_envs.VLLM_KV_CACHE_LAYOUT,
            store_enabled=config.store_enabled,
            load_async=config.load_async,
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


def _validate_support(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, config: KVPoolConfig) -> None:
    """Check compositions against the same resolved facts that execution will consume."""
    if _kvpp_size(vllm_config) > 1:
        raise ValueError("AscendStore v1 does not support active KVPP until KVPP ownership is represented")

    transfer_config = vllm_config.kv_transfer_config
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
