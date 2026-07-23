from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    ChunkedTokenDatabase,
    GroupBatchPlan,
    GroupTransferData,
    LayerBlockRange,
    LayerTransferArrays,
    LayerTransferTask,
    LayerwisePreparation,
    TransferCompletion,
    block_hash_to_str,
    get_block_hashes,
)

LAYERWISE_READ_LEASE_TTL_MS = 5 * 60 * 1000


class LayerTransferArrayBuilder:
    """Build vectorized local/GVA transfer arrays for one KV cache group."""

    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        num_layers: int,
        group_id: int = 0,
    ) -> None:
        self._block_len_np = np.asarray(token_database.group_block_len[group_id], dtype=np.int64)
        self.page_size_bytes = int(self._block_len_np.sum()) // num_layers
        self._kv_caches_base_addr_np = np.asarray(
            token_database.group_kv_caches_base_addr[group_id],
            dtype=np.int64,
        )
        group_block_stride = token_database.group_block_stride.get(group_id, token_database.group_block_len[group_id])
        self._block_stride_np = np.asarray(group_block_stride, dtype=np.int64)
        self._caches_per_layer = max(1, self._block_len_np.shape[0] // max(1, num_layers))
        layer_shape = (-1, self._caches_per_layer)
        self._layer_block_len_np = self._block_len_np.reshape(layer_shape)
        self._layer_base_addr_np = self._kv_caches_base_addr_np.reshape(layer_shape)
        self._layer_block_stride_np = self._block_stride_np.reshape(layer_shape)
        self._layer_inner_offsets_np = np.zeros_like(self._layer_block_len_np)
        if self._caches_per_layer > 1:
            self._layer_inner_offsets_np[:, 1:] = np.cumsum(
                self._layer_block_len_np[:, :-1],
                axis=1,
                dtype=np.int64,
            )
        self._rank_layer_offsets_np = (
            np.arange(
                self._layer_block_len_np.shape[0],
                dtype=np.int64,
            )
            * self.page_size_bytes
        )

    def _build_transfer_arrays(
        self,
        block_ids_arr: np.ndarray,
        base_gvas_arr: np.ndarray,
        layer_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        layer_base_addrs = self._layer_base_addr_np[layer_id]
        layer_block_len = self._layer_block_len_np[layer_id]
        layer_block_stride = self._layer_block_stride_np[layer_id]
        layer_inner_offsets = self._layer_inner_offsets_np[layer_id]
        rank_layer_offset = self._rank_layer_offsets_np[layer_id]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[KVPOOL] build_transfer layer=%d page_size=%d caches_per_layer=%d "
                "rank_layer_offset=%d layer_block_len=%s layer_inner_offsets=%s base_gvas=%s",
                layer_id,
                self.page_size_bytes,
                self._caches_per_layer,
                rank_layer_offset,
                layer_block_len.tolist(),
                layer_inner_offsets.tolist(),
                base_gvas_arr.tolist(),
            )

        addr_arr = layer_base_addrs[None, :] + block_ids_arr[:, None] * layer_block_stride[None, :]
        size_arr = np.broadcast_to(layer_block_len, addr_arr.shape)
        gvas_arr = base_gvas_arr[:, None] + rank_layer_offset + layer_inner_offsets[None, :]
        return addr_arr.ravel(), size_arr.ravel(), gvas_arr.ravel()

    def build_addrs(self, data: GroupTransferData, layer_id: int) -> LayerTransferArrays:
        """Compute per-layer arrays from block IDs and base GVAs."""
        addr_array, size_array, gvas_array = self._build_transfer_arrays(
            data.block_ids_arr,
            data.base_gvas_arr,
            layer_id,
        )
        return LayerTransferArrays(
            addr_array=addr_array,
            size_array=size_array,
            gvas_array=gvas_array,
        )


class LayerwiseTransferPreparer:
    """Prepare backend GVAs and shared metadata once per layerwise batch."""

    def __init__(
        self,
        m_store: Backend,
        model_name: str,
        head_or_tp_rank: int,
        hash_block_size: int,
        *,
        enabled: bool,
        can_allocate: bool,
        num_groups: int,
    ) -> None:
        self.m_store = m_store
        self.model_name = model_name
        self.head_or_tp_rank = head_or_tp_rank
        self.hash_block_size = hash_block_size
        self.enabled = enabled
        self.can_allocate = can_allocate
        self.num_groups = num_groups
        self.group_block_len: dict[int, list[int]] = {}
        self.load_lease_keys_by_request: dict[str, set[str]] = {}
        self.load_lease_refcounts: dict[str, int] = {}
        self._lease_lock = threading.Lock()

    def configure_layout(
        self,
        group_block_len: dict[int, list[int]],
    ) -> None:
        self.group_block_len = group_block_len

    def make_gva_key(self, group_id: int, block_hash_hex: str) -> str:
        if self.num_groups > 1:
            return f"{self.model_name}@{group_id}@{block_hash_hex}@{self.head_or_tp_rank}"
        return f"{self.model_name}@{block_hash_hex}@{self.head_or_tp_rank}"

    def resolve_save_groups(
        self,
        plans: list[GroupBatchPlan],
    ) -> dict[int, tuple[GroupTransferData, TransferCompletion]]:
        """Resolve per-group save keys to base GVAs in one backend call."""
        if not self.enabled or not self.can_allocate:
            return {}

        alloc_keys: list[str] = []
        alloc_sizes: list[int] = []
        key_indices: dict[str, int] = {}
        key_slices_by_group: dict[int, slice] = {}
        block_ids_by_group: dict[int, list[int]] = {}
        completions_by_group: dict[int, TransferCompletion] = {}

        for plan in plans:
            if not plan.save_ranges:
                continue
            group_id = plan.group_id
            block_lengths = self.group_block_len.get(group_id)
            if not block_lengths:
                raise RuntimeError(f"Block lengths are not initialized for KV cache group {group_id}")

            alloc_size = sum(block_lengths)
            key_start = len(alloc_keys)
            block_ids: list[int] = []
            completion = TransferCompletion([], [])

            for block_range in plan.save_ranges:
                request = block_range.request
                if group_id >= len(request.block_ids_by_group_np):
                    raise RuntimeError(f"Block IDs are not initialized for request {request.req_id}, group {group_id}")
                group_block_ids = request.block_ids_by_group_np[group_id]
                group_block_hashes = get_block_hashes(
                    request.block_hashes,
                    plan.block_size,
                    self.hash_block_size,
                )
                completion.req_ids.append(request.req_id)
                completion.is_last_chunks.append(request.is_last_chunk)

                for block_index in range(block_range.start_block, block_range.end_block):
                    key = self.make_gva_key(group_id, block_hash_to_str(group_block_hashes[block_index]))
                    key_index = key_indices.get(key)
                    if key_index is not None:
                        if alloc_sizes[key_index] != alloc_size:
                            raise RuntimeError(f"Conflicting allocation sizes for layerwise key {key}")
                        continue
                    key_index = len(alloc_keys)
                    key_indices[key] = key_index
                    alloc_keys.append(key)
                    alloc_sizes.append(alloc_size)
                    block_ids.append(int(group_block_ids[block_index]))

            key_slices_by_group[group_id] = slice(key_start, len(alloc_keys))
            block_ids_by_group[group_id] = block_ids
            completions_by_group[group_id] = completion

        allocated_gvas_np = np.empty(len(alloc_keys), dtype=np.int64)
        if alloc_keys:
            allocated_gvas = self.m_store.batch_alloc(alloc_keys, alloc_sizes)
            if len(allocated_gvas) != len(alloc_keys):
                raise RuntimeError(
                    "batch_alloc returned an unexpected number of GVAs: "
                    f"expected={len(alloc_keys)}, actual={len(allocated_gvas)}"
                )
            allocated_gvas_np = np.asarray(allocated_gvas, dtype=np.int64)
            invalid_count = int(np.count_nonzero(allocated_gvas_np <= 0))
            if invalid_count:
                raise RuntimeError(
                    "batch_alloc returned invalid GVAs: "
                    f"keys={len(alloc_keys)}, invalid={invalid_count}, sample={allocated_gvas[:5]}"
                )

        resolved: dict[int, tuple[GroupTransferData, TransferCompletion]] = {}
        for group_id, block_ids in block_ids_by_group.items():
            resolved[group_id] = (
                GroupTransferData(
                    block_ids_arr=np.asarray(block_ids, dtype=np.int64),
                    base_gvas_arr=allocated_gvas_np[key_slices_by_group[group_id]],
                ),
                completions_by_group[group_id],
            )
        return resolved

    @staticmethod
    def _slice_load_group(
        group_id: int,
        block_ranges: list[LayerBlockRange],
        full: tuple[GroupTransferData, TransferCompletion],
        request_slices: dict[str, tuple[int, int, int]],
    ) -> tuple[GroupTransferData, TransferCompletion]:
        full_data, _ = full
        total_blocks = sum(block_range.end_block - block_range.start_block for block_range in block_ranges)
        block_ids = np.empty(total_blocks, dtype=np.int64)
        base_gvas = np.empty(total_blocks, dtype=np.int64)
        req_ids: list[str] = []
        is_last_chunks: list[bool | None] = []
        offset = 0
        for block_range in block_ranges:
            request = block_range.request
            prepared_start, array_start, prepared_count = request_slices[request.req_id]
            relative_start = block_range.start_block - prepared_start
            relative_end = block_range.end_block - prepared_start
            if relative_start < 0 or relative_end > prepared_count:
                raise RuntimeError(
                    f"Load range [{block_range.start_block}, {block_range.end_block}) is outside prepared "
                    f"range for request {request.req_id}, group {group_id}"
                )
            count = relative_end - relative_start
            source_start = array_start + relative_start
            source_end = source_start + count
            block_ids[offset : offset + count] = full_data.block_ids_arr[source_start:source_end]
            base_gvas[offset : offset + count] = full_data.base_gvas_arr[source_start:source_end]
            offset += count
            req_ids.append(request.req_id)
            is_last_chunks.append(request.is_last_chunk)
        return (
            GroupTransferData(block_ids, base_gvas),
            TransferCompletion(req_ids, is_last_chunks),
        )

    def resolve_load_groups(
        self,
        plans: list[GroupBatchPlan],
    ) -> dict[tuple[int, bool], tuple[GroupTransferData, TransferCompletion]]:
        """Resolve load keys once and materialize full/tail data per group."""
        if not self.enabled:
            return {}

        keys: list[str] = []
        key_indices: dict[str, int] = {}
        keys_by_request: dict[str, set[str]] = {}
        block_ids_by_group: dict[int, list[int]] = {}
        key_indices_by_group: dict[int, list[int]] = {}
        request_slices_by_group: dict[int, dict[str, tuple[int, int, int]]] = {}
        completions_by_group: dict[int, TransferCompletion] = {}
        tail_ranges_by_group: dict[int, list[LayerBlockRange]] = {}

        for plan in plans:
            if not plan.full_load_ranges:
                continue
            group_id = plan.group_id
            block_ids: list[int] = []
            group_key_indices: list[int] = []
            request_slices: dict[str, tuple[int, int, int]] = {}
            completion = TransferCompletion([], [])

            for block_range in plan.full_load_ranges:
                request = block_range.request
                if group_id >= len(request.block_ids_by_group_np):
                    raise RuntimeError(f"Block IDs are not initialized for request {request.req_id}, group {group_id}")
                group_block_ids = request.block_ids_by_group_np[group_id]
                group_block_hashes = get_block_hashes(
                    request.block_hashes,
                    plan.block_size,
                    self.hash_block_size,
                )
                request_keys = keys_by_request.setdefault(request.req_id, set())
                array_start = len(block_ids)

                for block_index in range(block_range.start_block, block_range.end_block):
                    key = self.make_gva_key(group_id, block_hash_to_str(group_block_hashes[block_index]))
                    key_index = key_indices.get(key)
                    if key_index is None:
                        key_index = len(keys)
                        key_indices[key] = key_index
                        keys.append(key)
                    block_ids.append(int(group_block_ids[block_index]))
                    group_key_indices.append(key_index)
                    request_keys.add(key)

                block_count = block_range.end_block - block_range.start_block
                request_slices[request.req_id] = (block_range.start_block, array_start, block_count)
                completion.req_ids.append(request.req_id)
                completion.is_last_chunks.append(request.is_last_chunk)

            block_ids_by_group[group_id] = block_ids
            key_indices_by_group[group_id] = group_key_indices
            request_slices_by_group[group_id] = request_slices
            completions_by_group[group_id] = completion
            tail_ranges_by_group[group_id] = plan.hbm_tail_load_ranges

        base_gvas = np.empty(len(keys), dtype=np.int64)
        if keys:
            key_infos = self.m_store.batch_get_key_info(keys, flag=1)
            lease_results = self.m_store.batch_add_lease(keys, LAYERWISE_READ_LEASE_TTL_MS)
            if len(key_infos) != len(keys) or len(lease_results) != len(keys):
                raise RuntimeError(
                    "Layerwise load preparation returned unexpected result counts: "
                    f"keys={len(keys)}, key_infos={len(key_infos)}, leases={len(lease_results)}"
                )
            invalid_gvas = 0
            for position, key_info in enumerate(key_infos):
                size = key_info.size()
                base_gvas[position] = key_info.gva_list()[0] if size and size > 0 else 0
                invalid_gvas += base_gvas[position] == 0
            lease_failure_count = int(np.count_nonzero(np.asarray(lease_results, dtype=np.int32)))
            if invalid_gvas or lease_failure_count:
                logger.error(
                    "Layerwise load preparation has invalid entries: keys=%d invalid_gvas=%d lease_failures=%d",
                    len(keys),
                    invalid_gvas,
                    lease_failure_count,
                )
            self.register_load_leases(keys_by_request)

        prepared: dict[tuple[int, bool], tuple[GroupTransferData, TransferCompletion]] = {}
        for group_id, block_ids in block_ids_by_group.items():
            key_indices_np = np.asarray(key_indices_by_group[group_id], dtype=np.int64)
            full = (
                GroupTransferData(
                    block_ids_arr=np.asarray(block_ids, dtype=np.int64),
                    base_gvas_arr=base_gvas[key_indices_np],
                ),
                completions_by_group[group_id],
            )
            prepared[(group_id, False)] = full
            tail_ranges = tail_ranges_by_group[group_id]
            if tail_ranges:
                prepared[(group_id, True)] = self._slice_load_group(
                    group_id,
                    tail_ranges,
                    full,
                    request_slices_by_group[group_id],
                )
        return prepared

    def register_load_leases(self, keys_by_request: dict[str, set[str]]) -> None:
        with self._lease_lock:
            for req_id, keys in keys_by_request.items():
                registered = self.load_lease_keys_by_request.setdefault(req_id, set())
                for key in keys - registered:
                    self.load_lease_refcounts[key] = self.load_lease_refcounts.get(key, 0) + 1
                registered.update(keys)

    def release_finished_load_leases(self, finished_req_ids: set[str]) -> None:
        if not finished_req_ids or not self.enabled:
            return
        keys_to_release: list[str] = []
        with self._lease_lock:
            for req_id in finished_req_ids:
                for key in self.load_lease_keys_by_request.pop(req_id, ()):
                    refcount = self.load_lease_refcounts[key] - 1
                    if refcount == 0:
                        del self.load_lease_refcounts[key]
                        keys_to_release.append(key)
                    else:
                        self.load_lease_refcounts[key] = refcount
        if keys_to_release:
            result = self.m_store.batch_remove_lease(keys_to_release)
            if result != 0:
                logger.error(
                    "Failed to release %d layerwise load leases for %d finished requests: res=%d",
                    len(keys_to_release),
                    len(finished_req_ids),
                    result,
                )

    def create_save_preparation(
        self,
        plans: list[GroupBatchPlan],
        layer_tasks: list[list[LayerTransferTask]],
        prepare_tasks: Callable[[list[list[LayerTransferTask]]], None] | None,
    ) -> LayerwisePreparation:
        def prepare() -> None:
            if not self.enabled:
                if prepare_tasks is not None:
                    prepare_tasks(layer_tasks)
                return
            resolved_groups = self.resolve_save_groups(plans)
            for tasks in layer_tasks:
                for task in tasks:
                    resolved = resolved_groups.get(task.group_id)
                    if resolved is None:
                        raise RuntimeError(f"Save batch is not initialized for KV cache group {task.group_id}")
                    task.transfer_data, task.completion = resolved
            if prepare_tasks is not None:
                prepare_tasks(layer_tasks)

        preparation = LayerwisePreparation(prepare)
        for tasks in layer_tasks:
            for task in tasks:
                task.preparation = preparation
        return preparation

    def create_load_preparation(
        self,
        plans: list[GroupBatchPlan],
        layer_tasks: list[list[LayerTransferTask]],
        prepare_tasks: Callable[[list[list[LayerTransferTask]]], None] | None = None,
    ) -> LayerwisePreparation:
        def prepare() -> None:
            prepared_groups = self.resolve_load_groups(plans)
            for tasks in layer_tasks:
                for task in tasks:
                    resolved = prepared_groups.get((task.group_id, task.uses_hbm_tail))
                    if self.enabled and resolved is None:
                        raise RuntimeError(f"Load batch is not initialized for KV cache group {task.group_id}")
                    if resolved is not None:
                        task.transfer_data, task.completion = resolved
            if prepare_tasks is not None:
                prepare_tasks(layer_tasks)

        return LayerwisePreparation(prepare)
