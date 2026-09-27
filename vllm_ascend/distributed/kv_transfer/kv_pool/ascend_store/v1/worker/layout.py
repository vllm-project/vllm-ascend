"""Resolve the static Worker KV transfer layout."""

from __future__ import annotations

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
class KVCacheGroupLayout:
    """Worker-local layout for one vLLM KV cache group."""

    group_id: int
    block_size: int
    layer_names: tuple[str, ...]
    key_metadata: KeyMetadata
    uses_align_state: bool = False


@dataclass(frozen=True, slots=True)
class WorkerTransferLayout:
    """Static topology and cache-layout facts for one Worker."""

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
    kv_cache_groups: tuple[KVCacheGroupLayout, ...]
    transfer_group_ids: tuple[int, ...]


def resolve_worker_transfer_layout(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> WorkerTransferLayout:
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
    kv_cache_groups = []
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        uses_align_state = _uses_align_state(group)
        key_tp_rank = tp_rank if uses_align_state else head_or_tp_rank
        kv_cache_groups.append(
            KVCacheGroupLayout(
                group_id,
                kv_cache_utils.resolve_dcp_kv_block_size(group.kv_cache_spec, dcp_size),
                tuple(group.layer_names),
                KeyMetadata(model_name, key_tp_rank, dcp_rank, pp_rank, group_id),
                uses_align_state,
            )
        )
    transfer_group_ids = tuple(
        getattr(kv_cache_config, "transfer_group_ids", range(len(kv_cache_config.kv_cache_groups)))
    )
    return WorkerTransferLayout(
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
        tuple(kv_cache_groups),
        transfer_group_ids,
    )


def _uses_align_state(group) -> bool:
    kv_cache_spec = group.kv_cache_spec
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        specs = (kv_cache_spec.kv_cache_specs[layer_name] for layer_name in group.layer_names)
    else:
        specs = (kv_cache_spec,)
    return any(isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align" for spec in specs)


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
