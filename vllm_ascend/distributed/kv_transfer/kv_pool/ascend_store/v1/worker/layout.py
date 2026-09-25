"""Resolve the static Worker KV transfer layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.distributed import get_pcp_group, get_tensor_model_parallel_rank

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata, infer_group_block_sizes
from vllm_ascend.distributed.utils import get_decode_context_model_parallel_rank

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass(frozen=True, slots=True)
class WorkerTransferLayout:
    tp_rank: int
    tp_size: int
    pp_size: int
    pcp_rank: int
    pcp_size: int
    dcp_size: int
    num_kv_heads: int
    put_step: int
    block_size: int
    hash_block_size: int
    key_metadata: KeyMetadata


def resolve_worker_transfer_layout(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> WorkerTransferLayout:
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
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

    group_block_sizes = infer_group_block_sizes(cache_config.block_size, kv_cache_config.kv_cache_groups)
    original_block_size = group_block_sizes[0]
    block_size = original_block_size * dcp_size
    requested_hash_block_size = cache_config.prefix_match_unit
    if isinstance(requested_hash_block_size, int):
        hash_block_size = requested_hash_block_size * dcp_size
    else:
        hash_block_size = block_size
    model_name = model_config.model.rstrip("/").split("/")[-1]
    key_metadata = KeyMetadata(model_name, head_or_tp_rank, dcp_rank, pp_rank)
    return WorkerTransferLayout(
        tp_rank,
        tp_size,
        pp_size,
        pcp_rank,
        pcp_size,
        dcp_size,
        num_kv_heads,
        put_step,
        block_size,
        hash_block_size,
        key_metadata,
    )
