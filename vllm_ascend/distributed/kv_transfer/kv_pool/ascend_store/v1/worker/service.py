"""Worker-side classic Lookup, Load and queued Store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.distributed import get_pcp_group, get_tensor_model_parallel_rank
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata, infer_group_block_sizes
from vllm_ascend.distributed.utils import get_decode_context_model_parallel_rank

from ..metadata import AscendStoreV1Metadata
from .load import LoadService
from .lookup import LookupService
from .resources import WorkerCacheResources
from .store import StoreService

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass(frozen=True, slots=True)
class _WorkerTransferLayout:
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


class WorkerService:
    """Orchestrate the classic Worker Lookup, Load and Store operations."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> None:
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        kv_role = vllm_config.kv_transfer_config.kv_role
        consumer_is_to_put = extra_config.get("consumer_is_to_put", False)
        self.can_store = kv_role in ("kv_producer", "kv_both") or consumer_is_to_put
        layout = _resolve_worker_transfer_layout(vllm_config, kv_cache_config)
        self._cache_resources = WorkerCacheResources.create(
            parallel_config,
            extra_config,
            layout.key_metadata,
            layout.block_size,
            layout.hash_block_size,
            kv_cache_config.num_blocks,
        )
        backend = self._cache_resources.backend
        token_database = self._cache_resources.token_database
        self._lookup_service = LookupService(
            backend,
            token_database,
            layout.tp_size,
            layout.pp_size,
            layout.dcp_size,
            layout.num_kv_heads,
            model_config.max_model_len,
            layout.block_size,
        )
        block_size = layout.block_size
        load_async = extra_config.get("load_async", False)
        self._load_service = LoadService(backend, token_database, block_size, block_size, layout.tp_rank, load_async)
        self._store_service: StoreService | None = None
        if self.can_store:
            self._store_service = StoreService(
                backend,
                token_database,
                layout.block_size,
                layout.tp_rank,
                layout.pcp_rank,
                layout.pcp_size,
                layout.dcp_size,
                layout.put_step,
                kv_role,
            )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        try:
            self._cache_resources.register_kv_caches(kv_caches)
            if self._store_service is not None:
                self._store_service.start()
            self._load_service.start()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        try:
            try:
                if self._store_service is not None:
                    self._store_service.close()
            finally:
                self._load_service.close()
        finally:
            self._cache_resources.close()

    def lookup(self, token_len: int, block_hashes: list[BlockHash] | list[str]) -> int:
        return self._lookup_service.lookup(token_len, block_hashes)

    def load(self, metadata: AscendStoreV1Metadata) -> None:
        self._load_service.load(metadata.load_requests)

    def submit_store(self, metadata: AscendStoreV1Metadata) -> None:
        if self._store_service is not None:
            self._store_service.submit(metadata.store_requests)

    def wait_for_previous_store(self) -> None:
        if self._store_service is not None:
            self._store_service.wait_for_previous_store()

    def clear_store_completion_bookkeeping(self, preempted_req_ids: set[str]) -> None:
        if self._store_service is not None:
            self._store_service.discard_preempted_and_finished_requests(preempted_req_ids)

    def take_finished_load_request_ids(self, loading_ids: set[str], finished_ids: set[str]) -> set[str]:
        return self._load_service.take_finished_request_ids(loading_ids, finished_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        return self._load_service.take_failed_block_ids()


def _resolve_worker_transfer_layout(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> _WorkerTransferLayout:
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

    original_block_size = infer_group_block_sizes(cache_config.block_size, kv_cache_config.kv_cache_groups)[0]
    block_size = original_block_size * dcp_size
    requested_hash_block_size = cache_config.prefix_match_unit
    hash_block_size = (
        requested_hash_block_size if isinstance(requested_hash_block_size, int) else original_block_size
    ) * dcp_size
    key_metadata = KeyMetadata(model_config.model.rstrip("/").split("/")[-1], head_or_tp_rank, dcp_rank, pp_rank)
    return _WorkerTransferLayout(
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
