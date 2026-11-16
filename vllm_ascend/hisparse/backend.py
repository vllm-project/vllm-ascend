# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Core HiSparse backend: NPU streams, mapped Host memory and DMA."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import logger
from vllm.v1.hisparse.backend import HiSparseBackend

if TYPE_CHECKING:
    from vllm.v1.hisparse.runtime import HiSparseRuntime
    from vllm.v1.kv_cache_interface import KVCacheConfig


def _install_worker_dma_hooks() -> None:
    """Adapt Core until it exposes backend hooks for staging and invalidation."""
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker

    from vllm_ascend.hisparse.recycled_invalidate import RecycledGroupInvalidator

    original_init = HiSparseConnectorWorker._init_dma
    if getattr(original_init, "_ascend_row_dma", False):
        return

    @wraps(original_init)
    def init(worker):
        original_init(worker)
        if isinstance(worker.backend, AscendHiSparseBackend):
            if worker.is_host_writer and worker.shared_host_region is None:
                # Core allocates int64 staging; the bound Ascend slot producer uses int32.
                # Cache binding is complete here and precedes graph capture.
                staging = worker._slot_mapping_staging
                slots = worker.cache_handles[0].slot_mapping
                if staging is not None and not staging.num_tokens and slots.device.type == "npu":
                    if slots.dtype == torch.int32:
                        staging.slots = torch.empty(
                            staging.slots.shape, dtype=slots.dtype, device="cpu", pin_memory=True
                        )
            worker._ascend_recycled_invalidator = RecycledGroupInvalidator.bind(worker)

    init.__dict__["_ascend_row_dma"] = True
    HiSparseConnectorWorker._init_dma = init
    original_invalidate = HiSparseConnectorWorker.invalidate_blocks

    @wraps(original_invalidate)
    def invalidate(worker, block_ids, request_state_indices):
        owner = getattr(worker, "_ascend_recycled_invalidator", None)
        if owner is None or not owner.try_call(worker, block_ids, request_state_indices):
            return original_invalidate(worker, block_ids, request_state_indices)

    HiSparseConnectorWorker.invalidate_blocks = invalidate


def _allocate_resolver_workspace(runtime: HiSparseRuntime):
    from vllm_ascend.hisparse.cache_ops import ResolverWorkspace

    return ResolverWorkspace(
        max_rows=runtime.max_swap_rows,
        hot_size=runtime.region_stride,
        max_union_rows=runtime.max_union_rows,
        device=runtime.device,
        max_topk=runtime.index_group.shared_topk.device_topk_rows.shape[1],
    )


def _allocate_attention_lengths(runtime: HiSparseRuntime) -> tuple[torch.Tensor, torch.Tensor] | None:
    if runtime.kv_dtype != torch.float8_e4m3fn:
        return None
    return (
        torch.arange(runtime.max_swap_rows + 1, dtype=torch.int32, device=runtime.device),
        torch.empty((1,), dtype=torch.int32, device=runtime.device),
    )


@torch.inference_mode(False)
def prepare_hisparse_workspaces(forward_context: dict[str, Any]) -> None:
    """Create real next-binding storage before Worker memory profiling.

    Core fixes group geometry when loading the model. Retain one scratch owner
    per actual group, and fresh C8 lengths for each consumed binding generation.
    Core's total-consumption measurement includes these retained allocations.
    """
    from vllm.v1.worker.workspace import current_workspace_manager

    for layer in forward_context.values():
        cache = getattr(layer, "hisparse_cache", None)
        if cache is None:
            continue
        runtime = cache.runtime
        group = runtime.index_group
        workspace = getattr(group, "_ascend_profiled_resolver", None)
        if workspace is None:
            # Identity, not shape, separates groups. The manager key does not
            # retain the runtime or any Host-pool/Graph owner.
            key = ("ascend_hisparse_resolver", object())
            workspace = current_workspace_manager().get_persistent_resource(
                key, lambda runtime=runtime: _allocate_resolver_workspace(runtime)
            )
            group._ascend_profiled_resolver = workspace
        if not hasattr(runtime, "_ascend_profiled_cache_ops"):
            runtime._ascend_profiled_cache_ops = workspace, _allocate_attention_lengths(runtime)


class AscendHiSparseBackend(HiSparseBackend):
    # Keep private pools until NPU IPC events interoperate with Core's generic streams.
    stats_dtype = torch.int64
    # Verified for row and page D2H on the installed A5 runtime. Use this
    # conservative bound so the driver never receives an oversized batch.
    _D2H_BATCH_LIMIT = 2047

    def __init__(self) -> None:
        self._failed_dma_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def check_available(self) -> None:
        from vllm_ascend.hisparse.cache_ops import _load_resolver
        from vllm_ascend.hisparse.host_pool import install
        from vllm_ascend.utils import bootstrap_custom_op_env

        install()
        if not hasattr(torch.ops._C_ascend, "swap_blocks_batch"):
            # A5 disables the global custom-op selection policy. HiSparse still
            # needs its explicit DMA dependency; load the extension lazily here
            # without changing that policy for unrelated model operators.
            bootstrap_custom_op_env()
            try:
                importlib.import_module("vllm_ascend.vllm_ascend_C")
            except ImportError as error:
                raise RuntimeError("Ascend HiSparse requires the compiled Ascend DMA extension.") from error
            if not hasattr(torch.ops._C_ascend, "swap_blocks_batch"):
                raise RuntimeError("Ascend HiSparse requires the Ascend swap_blocks_batch operator.")
        _load_resolver()
        _install_worker_dma_hooks()

    def pin_tensor(self, tensor: torch.Tensor) -> None:
        from vllm_ascend.hisparse.host_pool import MappedHostTensor

        tensor._hisparse_registration = MappedHostTensor(tensor)

    def unpin_tensor(self, tensor: torch.Tensor) -> None:
        registration = tensor._hisparse_registration
        registration.close()
        del tensor._hisparse_registration

    def create_cache_ops(self, runtime: HiSparseRuntime) -> Any:
        from vllm_ascend.hisparse.cache_ops import HiSparseNPUOps

        group = runtime.index_group
        prepared = getattr(runtime, "_ascend_profiled_cache_ops", None)
        if runtime is group.leader:
            workspace = prepared[0] if prepared is not None else _allocate_resolver_workspace(runtime)
        else:
            leader_ops = group.leader.cache_ops
            if not isinstance(leader_ops, HiSparseNPUOps):
                raise RuntimeError("Bind the HiSparse leader before follower caches.")
            workspace = leader_ops.workspace
        mapping = runtime.registered_host_pool._hisparse_registration
        attention_capacity = 0
        attention_block_stride = None
        attention_lengths = None
        if runtime.kv_dtype == torch.float8_e4m3fn:
            from vllm_ascend.hisparse.c8 import c8_storage_view

            # The resolver consumes virtual page rows. Attention consumes dense
            # addresses relative to its own layer view, including any HMA gaps.
            # Qualify the bound geometry once, without adding replay-time work.
            view, block, attention_block_stride, _ = c8_storage_view(runtime.hot.cache)
            attention_view, _, _, _ = c8_storage_view(runtime.hot.attention_cache)
            attention_capacity = view.shape[0]
            if block != runtime.hot.block_size or attention_capacity > attention_view.shape[0]:
                raise ValueError("HiSparse C8 resolver and Attention views have incompatible geometry")
            if runtime is not group.leader and (
                attention_capacity != group.leader.cache_ops.attention_capacity
                or attention_block_stride != group.leader.cache_ops.attention_block_stride
            ):
                raise ValueError("Shared HiSparse C8 indices require matching per-layer page geometry")
            # Decode query shapes are bounded by the resolver workspace. These
            # values describe immutable geometry, not the current valid rows.
            # Prepare them before capture and retain them with this binding;
            # a rebind gets its own table and current physical KV span.
            attention_lengths = prepared[1] if prepared is not None else _allocate_attention_lengths(runtime)
            assert attention_lengths is not None
            attention_lengths[1].fill_(attention_view.shape[0])
        ops = HiSparseNPUOps(
            mapping,
            workspace,
            attention_capacity=attention_capacity,
            attention_block_stride=attention_block_stride,
            attention_lengths=attention_lengths,
        )
        if prepared is not None:
            del runtime._ascend_profiled_cache_ops
        return ops

    def validate_kv_cache_config(self, kv_cache_config: KVCacheConfig) -> None:
        from .binding import _validate_indexer_spec, get_hisparse_indexer_specs

        for spec in get_hisparse_indexer_specs(kv_cache_config).values():
            _validate_indexer_spec(spec)

    def prepare_kv_cache_views(
        self,
        kv_cache_config: KVCacheConfig,
        kv_caches: dict[str, Any],
        forward_context: dict[str, Any],
    ) -> None:
        from .binding import _bind_indexer_views, get_hisparse_indexer_specs

        _bind_indexer_views(kv_caches, get_hisparse_indexer_specs(kv_cache_config), forward_context)

    def is_current_stream_capturing(self) -> bool:
        return torch.npu.is_current_stream_capturing()

    def gather_resident_rows(self, source: torch.Tensor, destination: torch.Tensor, row_ids: torch.Tensor) -> None:
        from .cache_ops import gather_resident_rows

        gather_resident_rows(source, destination, row_ids)

    def swap_blocks_batch(self, src: torch.Tensor, dst: torch.Tensor, sizes: torch.Tensor) -> None:
        if self._failed_dma_batches:
            raise RuntimeError("HiSparse DMA submission previously failed; restart this worker.")
        limit = self._D2H_BATCH_LIMIT
        count = src.size(0)
        if count > limit and any(t.ndim != 1 or t.size(0) != count or not t.is_contiguous() for t in (src, dst, sizes)):
            raise ValueError("HiSparse D2H split requires equal-length contiguous descriptor arrays.")
        submit = torch.ops._C_ascend.swap_blocks_batch
        try:
            if count <= limit:
                submit(src, dst, sizes, 1, prefer_small_async=True)
                return
            for start in range(0, count, limit):
                end = min(start + limit, count)
                # Keep every chunk on Core's current DMA stream and on the batch API.
                submit(src[start:end], dst[start:end], sizes[start:end], 1, prefer_small_async=False)
            logger.info_once("[hisparse] Oversized D2H descriptor batch submitted in bounded chunks.", scope="process")
        except BaseException:
            # Core publishes its completion only after this method returns. Retain
            # the original storage in the process-cached backend on an uncertain
            # partial submit; Core shutdown drains the stream before freeing payloads.
            self._failed_dma_batches.append((src, dst, sizes))
            raise
