# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""V2 Pull into manager-owned Mooncake Host KV.

The Decode TP leader receives target and ordinary MTP KV into shared Host
views. Every rank receives its Indexer through the ordinary V2 planner.
"""

import math
from concurrent.futures import Future
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakePullConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeTransferMetadataGroups,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import (
    MooncakePullConnectorWorker,
    MooncakePullRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
    as_kv_cache_tensors,
    collect_configured_register_regions,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    get_sparse_kv_offload_manager,
)
from vllm_ascend.distributed.kv_transfer.utils.utils import validate_register_region_count
from vllm_ascend.utils import get_kv_cache_tensor_layers

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_host_pool import MooncakeHostPool
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import SparseKVOffloadManager


def validate_dsa_config(vllm_config: VllmConfig) -> None:
    """Validate the specialization before the parent starts any threads."""
    transfer = vllm_config.kv_transfer_config
    assert transfer is not None
    extra = transfer.kv_connector_extra_config
    if extra.get("use_layerwise", False):
        raise ValueError("Mooncake DSA offload requires Pull mode (use_layerwise=false)")
    if transfer.is_kv_consumer == transfer.is_kv_producer:
        raise ValueError("Mooncake DSA requires exactly one KV transfer role")
    sparse = init_ascend_config(vllm_config).sparse_kv_offload_config
    if transfer.is_kv_producer:
        if sparse.enabled:
            raise ValueError("Mooncake DSA Prefill requires sparse KV offload disabled")
        return
    if not sparse.enabled or sparse.host_backend != "mooncake" or not sparse.use_fused_overlap:
        raise ValueError("Mooncake DSA Decode requires enabled Mooncake Host offload with fused overlap")
    if transfer.kv_load_failure_policy != "fail":
        raise ValueError("Mooncake DSA Host offload requires kv_load_failure_policy='fail'")
    parallel = vllm_config.parallel_config
    if parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1:
        raise ValueError("Mooncake DSA Decode requires DCP=PCP=1")


def validate_shared_host_views(manager: "SparseKVOffloadManager", pool: "MooncakeHostPool") -> None:
    """Exchange offsets and initialization errors, never peer virtual pointers."""
    descriptor, error = None, None
    try:
        views = [
            (name, component, tensor)
            for name in manager.offload_layer_names
            for component, tensor in enumerate(manager.get_local_host_kv_views(name))
        ]
        descriptor = pool.describe_local_views(views, manager.kv_cache_config.num_blocks)
    except (TypeError, ValueError, RuntimeError) as exc:
        error = str(exc)
    layouts: list[tuple[object, str | None] | None] = [(descriptor, error)]
    if manager.tp_size > 1:
        layouts = [None] * manager.tp_size
        torch.distributed.all_gather_object(layouts, (descriptor, error), group=manager.tp_group.cpu_group)
    for layout in layouts:
        if layout is None:
            raise ValueError("Mooncake shared Host layout was not received from every rank")
        peer_descriptor, peer_error = layout
        if peer_error is not None or peer_descriptor != descriptor:
            raise ValueError(f"Mooncake shared Host layout mismatch: {peer_error}")


class MooncakeDsaRecvingThread(MooncakePullRecvingThread):
    """Reuse Pull planning and execution with logical block-capacity checks."""

    def _submit_transfer_group(
        self,
        remote_engine_id: str,
        remote_metadata: MooncakeTransferMetadataGroups,
        requests: dict[str, ReqMeta],
        selection_index_offset: int = 0,
    ) -> tuple[dict[Future[None], tuple[int, int, int, set[str]]], Exception | None]:
        for request in requests.values():
            local_groups = (*request.local_block_ids, *request.local_full_block_ids)
            if any(block < 0 or block >= self.local_metadata.num_blocks for ids in local_groups for block in ids):
                return {}, ValueError("Mooncake DSA local block exceeds logical capacity")
            for pp in remote_metadata.metadata_by_pp_rank.values():
                if any(block < 0 or block >= pp.num_blocks for ids in request.remote_block_ids for block in ids):
                    return {}, ValueError("Mooncake DSA remote block exceeds logical capacity")
        return super()._submit_transfer_group(remote_engine_id, remote_metadata, requests, selection_index_offset)


class MooncakeDsaWorker(MooncakePullConnectorWorker):
    """Project manager-owned Host KV and device views onto the Pull lifecycle."""

    receiving_thread_cls = MooncakeDsaRecvingThread

    def _get_shared_page_metadata(self, caches):
        # P can pack NoPE and RoPE into one physical slot. Keep each component's
        # identity so the same ordinary address planner can target separate Host K/V.
        return None

    def _prepare_transfer_caches(self, kv_caches):
        self._host_layer_names: set[str] = set()
        caches = dict(kv_caches)
        config = self.kv_cache_config
        excluded = set()
        if self.kv_transfer_config.is_kv_consumer:
            manager = get_sparse_kv_offload_manager()
            pool = manager.get_mooncake_host_pool()
            validate_shared_host_views(manager, pool)
            self._host_layer_names = set(manager.offload_layer_names)
            if self.tp_rank == 0:
                for name in self._host_layer_names:
                    caches[name] = manager.get_local_host_kv_views(name)
            else:
                excluded = self._host_layer_names
                caches = {name: cache for name, cache in caches.items() if name not in excluded}
        # Group IDs and allocator state remain unchanged. Only the connector's
        # physical view list omits Main on followers, as in the HiSparse adapter.
        tensors = []
        for tensor in config.kv_cache_tensors:
            names = [name for name in get_kv_cache_tensor_layers(tensor) if name not in excluded]
            if names:
                tensors.append(replace(tensor, layers=names))
        transfer_config = replace(config, kv_cache_tensors=tensors)
        for name, value in caches.items():
            for cache in as_kv_cache_tensors(value):
                if cache.numel() == 0 or cache.shape[0] % config.num_blocks:
                    raise ValueError(f"Mooncake DSA invalid physical block capacity for {name}")
                block_bytes = math.prod(cache.shape[1:]) * cache.element_size()
                if not cache[0].is_contiguous() or cache.stride(0) * cache.element_size() < block_bytes:
                    raise ValueError(f"Mooncake DSA requires contiguous, non-overlapping component blocks: {name}")
        return caches, transfer_config

    def register_kv_caches(self, kv_caches) -> None:
        caches, transfer_config = self._prepare_transfer_caches(kv_caches)
        original_config = self.kv_cache_config
        self.kv_cache_config = transfer_config
        try:
            super().register_kv_caches(caches)
        finally:
            self.kv_cache_config = original_config

    def _register_transfer_buffers(self, ptrs: list[int], lengths: list[int]) -> None:
        # Shared VMM Host KV is directly addressable. Register device
        # destinations through the ordinary buffer path.
        if self.kv_transfer_config.is_kv_consumer:
            device_caches = {
                name: (() if name in self._host_layer_names else value) for name, value in self.kv_caches.items()
            }
            regions = collect_configured_register_regions(self.kv_cache_config, device_caches)
            validate_register_region_count(regions)
            ptrs, lengths = regions.ptrs, regions.lengths
        super()._register_transfer_buffers(ptrs, lengths)


class MooncakeDsaConnector(MooncakePullConnector):
    """Select a DSA Worker while inheriting the ordinary V2 scheduler."""

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig) -> None:
        validate_dsa_config(vllm_config)
        super().__init__(vllm_config, role, kv_cache_config)

    def _create_worker(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> MooncakeDsaWorker:
        return MooncakeDsaWorker(vllm_config, str(self.engine_id), kv_cache_config)
