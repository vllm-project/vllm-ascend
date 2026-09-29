"""Define the static topology and cache geometry of one KV Pool program."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.distributed import get_pcp_group, get_tensor_model_parallel_rank
from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    KeyMetadata,
    infer_tp_mismatch_info,
)
from vllm_ascend.distributed.utils import get_decode_context_model_parallel_rank

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass(frozen=True, slots=True)
class TPPartitionSpec:
    """Effective Backend key partitioning for the configured TP relationship."""

    tp_mismatch: bool
    key_rank_count: int
    key_slices_per_rank: int


@dataclass(frozen=True, slots=True)
class KVPoolLayerTopology:
    """Cache entries owned by one physical model layer inside one group."""

    physical_layer_id: int
    layer_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KVPoolGroupTopology:
    """Instance-lifetime topology for one vLLM KV cache group."""

    group_id: int
    block_size: int
    layers: tuple[KVPoolLayerTopology, ...]
    key_metadata: KeyMetadata
    uses_align_state: bool = False

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(layer_name for layer in self.layers for layer_name in layer.layer_names)


@dataclass(frozen=True, slots=True)
class KVPoolTopology:
    """Instance-lifetime topology and cache geometry compiled into program stages."""

    tp_rank: int
    tp_size: int
    pp_size: int
    pcp_rank: int
    pcp_size: int
    dcp_size: int
    put_step: int
    cache_transfer_granularity: int
    hash_block_size: int
    tp_partition: TPPartitionSpec
    groups: tuple[KVPoolGroupTopology, ...]
    transfer_group_ids: tuple[int, ...]
    consumer_pipeline_partitions: tuple[int, ...] | None


def resolve_kv_pool_topology(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> KVPoolTopology:
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    pp_rank = (parallel_config.rank // tp_size) % pp_size
    pcp_size = getattr(parallel_config, "prefill_context_parallel_size", 1)
    pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
    dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
    dcp_rank = get_decode_context_model_parallel_rank() if dcp_size > 1 else 0
    num_kv_heads = 1 if getattr(model_config, "use_mla", False) else model_config.get_total_num_kv_heads()
    put_step = tp_size // num_kv_heads if num_kv_heads < tp_size else 1
    head_or_tp_rank = tp_rank // put_step
    tp_partition = resolve_tp_partition(vllm_config)
    cache_transfer_granularity, hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    model_name = model_config.model.rstrip("/").split("/")[-1]
    base_layer_count = model_config.get_total_num_hidden_layers()
    groups = []
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        uses_align_state = _uses_align_state(group)
        key_tp_rank = tp_rank if uses_align_state else head_or_tp_rank
        groups.append(
            KVPoolGroupTopology(
                group_id,
                kv_cache_utils.resolve_dcp_kv_block_size(group.kv_cache_spec, dcp_size),
                _resolve_group_layers(group.layer_names, base_layer_count),
                KeyMetadata(model_name, key_tp_rank, dcp_rank, pp_rank, group_id),
                uses_align_state,
            )
        )
    transfer_group_ids = tuple(
        getattr(kv_cache_config, "transfer_group_ids", range(len(kv_cache_config.kv_cache_groups)))
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
        tp_partition,
        tuple(groups),
        transfer_group_ids,
        resolve_consumer_pipeline_partitions(vllm_config),
    )


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
        try:
            parsed_partitions = tuple(int(layer_count) for layer_count in partition_config.split(","))
        except ValueError as error:
            raise ValueError(f"Invalid prefill PP layer partition: {partition_config}") from error
        if len(parsed_partitions) != prefill_pp_size:
            raise ValueError(
                f"Partition count {len(parsed_partitions)} does not match prefill_pp_size {prefill_pp_size}"
            )
        if sum(parsed_partitions) != num_hidden_layers:
            raise ValueError(
                f"Partition layer count {sum(parsed_partitions)} does not match model layers {num_hidden_layers}"
            )
        return parsed_partitions

    layers_per_partition, remaining_layers = divmod(num_hidden_layers, prefill_pp_size)
    partitions = [layers_per_partition] * prefill_pp_size
    for index in range(2, remaining_layers + 2):
        partitions[-index] += 1
    return tuple(partitions)


def resolve_tp_partition(vllm_config: VllmConfig) -> TPPartitionSpec:
    """Resolve the effective TP key namespace and local slicing requirement."""

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
    key_rank_count = mismatch_info.effective_tp_size if mismatch_info.enabled else min(tp_size, num_kv_heads)
    return TPPartitionSpec(mismatch_info.enabled, key_rank_count, mismatch_info.num_sub_keys)


def _uses_align_state(group) -> bool:
    kv_cache_spec = group.kv_cache_spec
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        for layer_name in group.layer_names:
            spec = kv_cache_spec.kv_cache_specs[layer_name]
            if isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align":
                return True
        return False
    return isinstance(kv_cache_spec, MambaSpec) and kv_cache_spec.mamba_cache_mode == "align"


def _resolve_group_layers(layer_names: list[str], base_layer_count: int) -> tuple[KVPoolLayerTopology, ...]:
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
