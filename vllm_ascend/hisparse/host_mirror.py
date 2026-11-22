# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Order direct HiSparse Host writeback against drafting and output completion."""

from __future__ import annotations

import inspect
from functools import wraps

import torch

from vllm_ascend.hisparse import FLASH_MTP_QUERY_LEN


def _host_disjoint(caches, draft_layers):
    if any(cache.ndim != 2 or not cache.is_contiguous() or cache.numel() == 0 for cache in caches):
        return False
    spans = [(cache.data_ptr(), cache.data_ptr() + cache.numel() * cache.element_size()) for cache in caches]
    return all(
        left_end <= right_start or right_end <= left_start
        for index, (left_start, left_end) in enumerate(spans)
        if index in draft_layers
        for other, (right_start, right_end) in enumerate(spans)
        if other not in draft_layers
    )


def _draft_groups_disjoint(handles, draft_layers):
    draft_runtimes = {id(handles[index].runtime) for index in draft_layers}
    # A draft-led resolver can refill every follower from Host in one call.
    for index in draft_layers:
        group = handles[index].runtime.index_group
        if id(group.leader) not in draft_runtimes or any(
            id(follower) not in draft_runtimes for follower in group.followers
        ):
            return False
    return True


def _source_layout_valid(worker):
    """Check per-page layer ranges; HMA owns live physical block IDs across groups."""
    backing = worker.cache_handles[0].runtime.hot_backing
    if backing.ndim != 1 or not backing.is_contiguous() or backing.storage_offset() != 0:
        return False
    base = backing.data_ptr()
    total = backing.numel() * backing.element_size()
    stride = None
    ranges = []
    for handle, cache in zip(worker.cache_handles, worker.resident_caches, strict=True):
        if cache.ndim != 3:
            return False
        hot = handle.runtime.hot.cache
        item = cache.element_size()
        row = cache.shape[-1] * item
        page = worker.kernel_block_size * row
        step = cache.stride(0) * item
        offset = cache.data_ptr() - base
        group = handle.runtime.resident_source_index
        if (
            cache.shape[1] != worker.kernel_block_size
            or cache.stride(1) * item != row
            or cache.stride(2) != 1
            or cache.device != backing.device
            or cache.untyped_storage().data_ptr() != base
            or step <= 0
            or offset < 0
            or offset + page > step
            or cache.shape[0] * step > total
            or group < 0
            or (stride is not None and stride != step)
            or cache.data_ptr() != hot.data_ptr()
            or cache.dtype != hot.dtype
            or cache.shape != hot.shape
            or cache.stride() != hot.stride()
        ):
            return False
        stride = step
        ranges.append((group, offset, offset + page))
    # Within a group the same physical page can serve Target and draft, but
    # their layer slices must not overlap. Different groups' live writable
    # pages are distinct allocations in Core's common BlockPool.
    return all(
        left[0] != right[0] or left[2] <= right[1] or right[2] <= left[1]
        for index, left in enumerate(ranges)
        if index in worker._draft_layers
        for other, right in enumerate(ranges)
        if other not in worker._draft_layers
    )


class _DirectMirrorSources:
    """Retain one canonical worker binding and recognize its complete Target plan."""

    def __init__(self, worker, targets):
        self._sources = worker.resident_caches
        self._destinations = worker.host_caches
        self._targets = targets

    @classmethod
    def bind(cls, worker, speculator):
        handles = worker.cache_handles
        layers = worker.vllm_config.compilation_config.static_forward_context
        draft = set(worker._draft_layers)
        expected = {
            id(getattr(layers.get(name), "hisparse_cache", None))
            for name in getattr(speculator, "draft_attn_layer_names", ())
        } & {id(handle) for handle in handles}
        targets = tuple(index for index in range(len(handles)) if index not in draft)
        if (
            not draft
            or not targets
            or {id(handles[index]) for index in draft} != expected
            or not _draft_groups_disjoint(worker.cache_handles, draft)
            or not _host_disjoint(worker.host_caches, draft)
            or not _source_layout_valid(worker)
        ):
            return None
        return cls(worker, targets)

    def can_defer(self, worker):
        return (
            worker.resident_caches is self._sources
            and worker.host_caches is self._destinations
            and not worker._dma_submitted
            and not worker._per_layer_mirrored
            and not worker._submitted_mirror_layers
            and tuple(
                index
                for index, cache in enumerate(worker.cache_handles)
                if cache.num_actual_tokens and index not in worker._draft_layers
            )
            == self._targets
        )


class _LegacyDecodeRowMirror:
    """Defer current rows on Core pins without the native per-step flag."""

    deferred = False

    @classmethod
    def bind(cls, worker):
        state = getattr(worker, "_ascend_legacy_decode_row_mirror", None)
        if state is None:
            state = worker._ascend_legacy_decode_row_mirror = cls()
        worker._stage_row_mirror_mapping = state._guard(worker._stage_row_mirror_mapping)
        worker._enqueue_host_mirror = state._guard(worker._enqueue_host_mirror)
        for handle in worker.cache_handles:
            submit = handle.submit_layer_mirror
            if submit is not None:
                # Core binds this callback before the runner wraps its connector.
                handle.submit_layer_mirror = state._guard(submit)
        return state

    def _guard(self, callback):
        if getattr(callback, "_ascend_legacy_decode_row_mirror", None) is self:
            return callback

        @wraps(callback)
        def guarded(*args, **kwargs):
            if not self.deferred:
                return callback(*args, **kwargs)

        guarded._ascend_legacy_decode_row_mirror = self
        return guarded


class HostMirrorConnector:
    """Join one worker's Host mirror at the existing Core draft finalizer."""

    def __init__(
        self,
        connector,
        worker,
        output_copy_stream,
        sources,
        logger=None,
        rank=None,
        *,
        defer_resident_decode_row_mirror: bool = False,
        legacy_decode_row_mirror: _LegacyDecodeRowMirror | None = None,
    ):
        self._connector = connector
        self._worker = worker
        self._output_copy_stream = output_copy_stream
        self._sources = sources
        self._defer_resident_decode_row_mirror = defer_resident_decode_row_mirror
        self._legacy_decode_row_mirror = legacy_decode_row_mirror
        self._logger = logger
        self._rank = rank
        self._logged_lazy_decode = False
        self._logged_draft_defer = False
        self._logged_finalize = False
        self._overlap_draft = False
        self._pending_event = None
        self._allow_step = False
        self._eligible_forward = False

    def __getattr__(self, name):
        return getattr(self._connector, name)

    def join_pending(self) -> None:
        if self._pending_event is not None:
            torch.accelerator.current_stream().wait_event(self._pending_event)
            self._pending_event = None
        self._overlap_draft = False

    def begin_execution(self, allow_step: bool) -> None:
        # The runner calls this before preemption or any scheduler state writes.
        self.join_pending()
        self._allow_step = allow_step
        self._eligible_forward = False
        if self._legacy_decode_row_mirror is not None:
            self._legacy_decode_row_mirror.deferred = False

    def pre_forward(self, scheduler_output, input_batch=None, attn_metadata=None):
        legacy = self._legacy_decode_row_mirror
        if legacy is not None:
            legacy.deferred = False
        self._eligible_forward = (
            self._allow_step
            and scheduler_output.has_sync_kv_loads
            and input_batch is not None
            and attn_metadata is not None
            and not input_batch.has_prefill
            and 0 < input_batch.num_tokens == input_batch.num_tokens_after_padding
        )
        if (
            self._eligible_forward
            and self._defer_resident_decode_row_mirror
            and input_batch is not None
            and attn_metadata is not None
            and bool((input_batch.num_scheduled_tokens == FLASH_MTP_QUERY_LEN).all())
            and getattr(scheduler_output.kv_connector_metadata, "all_context_pages_resident", False)
        ):
            target_metadata = [
                attn_metadata.get(name) for name, handle in self._worker._group_leaders if not handle.draft_layer
            ]
            if target_metadata and all(
                metadata is not None and 0 < metadata.num_decode_tokens == metadata.num_actual_tokens
                for metadata in target_metadata
            ):
                if self._logger is not None and not self._logged_lazy_decode:
                    self._logger.info(
                        "HiSparse lazy row mirroring active for fully resident target decode: tokens=%d requests=%d",
                        input_batch.num_tokens,
                        input_batch.num_reqs,
                    )
                    self._logged_lazy_decode = True
                if legacy is not None:
                    # The old worker drains prior draft rows through _enqueue_row_dma
                    # and whole-page transfers through _submit_transfers.
                    legacy.deferred = not self._connector._disabled
                    try:
                        return self._connector.pre_forward(scheduler_output, input_batch, attn_metadata=attn_metadata)
                    except BaseException:
                        legacy.deferred = False
                        raise
                return self._connector.pre_forward(
                    scheduler_output, input_batch, attn_metadata=attn_metadata, defer_decode_row_mirror=True
                )
        return self._connector.pre_forward(scheduler_output, input_batch, attn_metadata=attn_metadata)

    def finish_forward(self) -> None:
        from vllm_ascend.hisparse.pcp import PCPHiSparseConnector

        eligible = self._eligible_forward
        self._eligible_forward = False
        worker = self._worker
        if not eligible or self._connector._disabled or not self._sources.can_defer(worker):
            return self._connector.finish_forward()
        if isinstance(self._connector, PCPHiSparseConnector):
            self._connector.prepare_host_mirror()

        compute_stream = torch.accelerator.current_stream()
        worker._forward_ready_event.record(compute_stream)
        pending_before = len(worker._pending_dma_descriptors)
        try:
            worker._finish_mirror_phase(worker._forward_ready_event)
            if worker._dma_submitted:
                self._pending_event = worker.host_write_event
                worker._dma_submitted = False
                self._overlap_draft = len(worker._pending_dma_descriptors) == pending_before + 1
                if self._overlap_draft:
                    # AsyncOutput must not report completion before Host KV is durable.
                    self._output_copy_stream.wait_event(self._pending_event)
                else:
                    self.join_pending()
            else:
                worker.host_write_event.record(compute_stream)
            worker._release_completed_dma_descriptors()
        except BaseException:
            # The mirror phase may fail after recording a submitted copy.
            if worker._dma_submitted:
                self._pending_event = worker.host_write_event
            self.join_pending()
            raise

    def post_forward(self, finished_req_ids, defer_finalize=False):
        if not defer_finalize or not self._overlap_draft:
            self.join_pending()
        elif not self._logged_draft_defer and self._logger is not None:
            self._logger.info(
                "HiSparse direct Target mirror draft defer used: rank=%d; source-safe=HMA layer/page ownership; "
                "Host-durable=DMA host_write_event; join=after draft",
                self._rank,
            )
            self._logged_draft_defer = True
        return self._connector.post_forward(finished_req_ids, defer_finalize=defer_finalize)

    def finalize_step(self):
        deferred = self._pending_event is not None and self._overlap_draft
        self.join_pending()
        if deferred and not self._logged_finalize and self._logger is not None:
            self._logger.info(
                "HiSparse direct Target mirror Host durability wait enqueued after draft: rank=%d", self._rank
            )
            self._logged_finalize = True
        return self._connector.finalize_step()

    def no_forward(self, scheduler_output):
        self.begin_execution(False)
        return self._connector.no_forward(scheduler_output)

    def set_disabled(self, disabled: bool) -> None:
        self.begin_execution(False)
        self._connector.set_disabled(disabled)

    def reset_capture_state(self) -> None:
        self.begin_execution(False)
        self._connector.reset_capture_state()


def overlap_target_host_mirror(runner):
    """Use Core's private Host writer and live HMA binding, or keep its original fence."""
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import HiSparseConnector
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
    from vllm.logger import logger
    from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

    from vllm_ascend.attention.sfa_hisparse import supports_flash_resident_decode_graph
    from vllm_ascend.hisparse.backend import AscendHiSparseBackend
    from vllm_ascend.hisparse.pcp import PCPHiSparseConnector

    connector = runner.kv_connector
    if (
        type(connector) not in (ActiveKVConnector, PCPHiSparseConnector)
        or type(connector.kv_connector) is not HiSparseConnector
    ):
        return connector
    worker = connector.kv_connector.connector_worker
    if (
        type(worker) is not HiSparseConnectorWorker
        or not worker._initialized
        or not isinstance(worker.backend, AscendHiSparseBackend)
        or worker.shared_host_region is not None
        or not worker.is_host_writer
        or worker.cache_handles[0].runtime.device.type != "npu"
    ):
        return connector
    sources = _DirectMirrorSources.bind(worker, runner.speculator)
    if sources is None:
        return connector
    output = runner.output_copy_stream
    output_stream = torch.Stream(
        stream_id=output.stream_id, device_index=output.device_index, device_type=output.device_type
    )
    compute_stream = torch.accelerator.current_stream()
    if (
        output_stream.device != compute_stream.device
        or compute_stream.device != worker.cache_handles[0].runtime.device
        or output_stream == compute_stream
    ):
        return connector
    rank = get_tensor_model_parallel_rank()
    logger.info("HiSparse direct Target Host mirror overlap enabled: rank=%d", rank)
    defer_decode_row_mirror = (
        type(connector) is ActiveKVConnector
        and not worker.cache_handles[0].runtime.eager_host_mirror
        and supports_flash_resident_decode_graph(runner.vllm_config)
        and runner.vllm_config.num_speculative_tokens + 1 == FLASH_MTP_QUERY_LEN
    )
    legacy = None
    if defer_decode_row_mirror and "defer_decode_row_mirror" not in inspect.signature(connector.pre_forward).parameters:
        legacy = _LegacyDecodeRowMirror.bind(worker)
    return HostMirrorConnector(
        connector,
        worker,
        output_stream,
        sources,
        logger,
        rank,
        defer_resident_decode_row_mirror=defer_decode_row_mirror,
        legacy_decode_row_mirror=legacy,
    )
