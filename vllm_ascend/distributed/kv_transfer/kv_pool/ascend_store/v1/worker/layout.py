"""Resolve the static Worker KV transfer layout."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.distributed import get_pcp_group, get_tensor_model_parallel_rank

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    infer_group_block_sizes,
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
class WorkerTransferLayout:
    """Static topology and cache-layout facts for one Worker."""

    tp_rank: int
    tp_size: int
    pp_size: int
    pcp_rank: int
    pcp_size: int
    dcp_size: int
    put_step: int
    block_size: int
    hash_block_size: int
    tp_partition: TPPartitionSpec
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

    tp_partition = resolve_tp_partition(vllm_config)

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
        put_step,
        block_size,
        hash_block_size,
        tp_partition,
        key_metadata,
    )


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


class StridedKVPartitioner:
    """Partition local KV blocks into effective-rank keys and strided segments."""

    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        key_slices_per_rank: int,
    ) -> None:
        self._token_database = token_database
        self._block_size = block_size
        self.tp_rank = tp_rank
        self._key_slices_per_rank = key_slices_per_rank

    def partition(
        self,
        base_key: str,
        block_id: int,
        token_count: int,
    ) -> Iterator[tuple[str, tuple[int, ...], tuple[int, ...]]]:
        for slice_index in range(self._key_slices_per_rank):
            addresses, sizes = self._resolve_segments(block_id, token_count, slice_index)
            yield self._replace_key_rank(base_key, slice_index), tuple(addresses), tuple(sizes)

    def _replace_key_rank(self, key: str, slice_index: int) -> str:
        marker = "@head_or_tp_rank:"
        marker_start = key.find(marker)
        if marker_start < 0:
            return key
        value_start = marker_start + len(marker)
        value_end = key.find("@", value_start)
        if value_end < 0:
            value_end = len(key)
        effective_rank = self.tp_rank * self._key_slices_per_rank + slice_index
        return f"{key[:value_start]}{effective_rank}{key[value_end:]}"

    def _resolve_segments(self, block_id: int, token_count: int, slice_index: int) -> tuple[list[int], list[int]]:
        group_addresses = self._token_database.group_kv_caches_base_addr[0]
        group_block_lengths = self._token_database.group_block_len[0]
        group_block_strides = self._token_database.group_block_stride.get(0)
        slice_size = group_block_lengths[0] // self._block_size // self._key_slices_per_rank
        head_offset = slice_index * slice_size
        addresses = []
        sizes = []
        for index, base_address in enumerate(group_addresses):
            block_length = group_block_lengths[index]
            block_stride = group_block_strides[index] if group_block_strides else block_length
            bytes_per_token = block_length // self._block_size
            block_address = base_address + block_id * block_stride
            for token_index in range(token_count):
                addresses.append(block_address + token_index * bytes_per_token + head_offset)
                sizes.append(slice_size)
        return addresses, sizes
