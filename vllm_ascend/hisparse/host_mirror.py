# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Own HiSparse Host snapshots and sampling/draft mirror completion fences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


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


def _index_plan(addresses, sizes, base, width, source_rows, limit):
    if not len(addresses):
        return None
    first, alignment = np.divmod(addresses - base, width)
    counts, size_remainder = np.divmod(sizes, width)
    if (
        np.any(alignment)
        or np.any(size_remainder)
        or np.any(counts <= 0)
        or np.any(first < 0)
        or np.any(first + counts > source_rows)
    ):
        return None
    ends = np.cumsum(counts)
    total = int(ends[-1])
    if total > limit:
        return None
    starts = ends - counts
    rows = np.repeat(first - starts, counts) + np.arange(total, dtype=np.int64)
    return starts, rows


@dataclass
class _Scratch:
    cpu_indices: torch.Tensor
    device_indices: torch.Tensor
    payload: torch.Tensor
    ready: torch.Event
    completion: torch.Event | None = None
    in_flight: bool = False

    def available(self):
        return not self.in_flight or (self.completion is not None and self.completion.query())


class TargetMirrorSnapshot:
    """Stage exact Target source bytes before draft can overwrite shared HMA."""

    def __init__(self, worker, layers, width, capacity):
        backing = worker.hot_backing
        self._source = backing.view(torch.uint8).reshape(-1, width)
        self._base = backing.data_ptr()
        self._width = width
        self._layers = layers
        self._sources = worker.resident_caches
        self._destinations = worker.host_caches
        self._capacity = capacity
        self._slots = tuple(
            _Scratch(
                torch.empty(capacity, dtype=torch.int64, pin_memory=True),
                torch.empty(capacity, dtype=torch.int64, device=backing.device),
                torch.empty((capacity, width), dtype=torch.uint8, device=backing.device),
                torch.Event(),
            )
            for _ in range(2)
        )
        self.armed = False
        self.submitted_completion = None

    @classmethod
    def bind(cls, worker, speculator):
        geometry = getattr(worker, "_ascend_row_dma_geometry", None)
        draft = set(worker._draft_layers)
        names = worker.cache_layer_names
        expected = set(getattr(speculator, "draft_attn_layer_names", ())) & set(names)
        if geometry is None or not draft or {names[i] for i in draft} != expected:
            return None
        layers = tuple(index for index in range(len(names)) if index not in draft)
        if (
            not layers
            or not _draft_groups_disjoint(worker.cache_handles, draft)
            or not _host_disjoint(worker.host_caches, draft)
        ):
            return None
        widths = {int(geometry.row_bytes[i]) for i in layers}
        if len(widths) != 1:
            return None
        width = widths.pop()
        backing = worker.hot_backing
        base = backing.data_ptr()
        if (
            width <= 0
            or backing.device.type != "npu"
            or backing.dtype not in (torch.int8, torch.uint8)
            or backing.ndim != 1
            or not backing.is_contiguous()
            or backing.storage_offset() != 0
            or backing.numel() % width
            or any(
                cache.untyped_storage().data_ptr() != base
                or (cache.data_ptr() - base) % width
                or geometry.source_page_bytes[index] % width
                for index in layers
                for cache in (worker.resident_caches[index],)
            )
        ):
            return None
        config = worker.vllm_config
        capacity = config.scheduler_config.max_num_seqs * (config.num_lookahead_tokens + 1) * len(layers)
        return cls(worker, layers, width, capacity)

    def try_submit(self, worker, descriptors, count, layer_indices, ready_event):
        if (
            not self.armed
            or self.submitted_completion is not None
            or ready_event is not worker._forward_ready_event
            or tuple(layer_indices) != self._layers
            or worker.resident_caches is not self._sources
            or worker.host_caches is not self._destinations
        ):
            return False
        slot = next((item for item in self._slots if item.available()), None)
        if slot is None:
            return False
        plan = _index_plan(
            descriptors.src_np[:count],
            descriptors.sizes_np[:count],
            self._base,
            self._width,
            self._source.shape[0],
            self._capacity,
        )
        if plan is None:
            return False
        starts, rows = plan
        size = len(rows)
        # Owners remain pinned if any submission fails. Never retry a partial
        # enqueue using the original device addresses.
        slot.in_flight = True
        slot.completion = None
        slot.cpu_indices[:size].numpy()[:] = rows
        slot.device_indices[:size].copy_(slot.cpu_indices[:size], non_blocking=True)
        torch.index_select(self._source, 0, slot.device_indices[:size], out=slot.payload[:size])
        slot.ready.record()
        descriptors.src_np[:count] = slot.payload.data_ptr() + starts * self._width
        worker._submit_dma_descriptors(descriptors, count, ready_event=slot.ready)
        slot.completion = worker._pending_dma_descriptors[-1][0]
        self.submitted_completion = slot.completion
        return True


class SamplingMirrorOverlapConnector:
    """Let Target logits and sampling run before joining the Host mirror."""

    def __init__(self, connector, worker, output_copy_stream, snapshot=None):
        self._connector = connector
        self._worker = worker
        self._output_copy_stream = output_copy_stream
        self._snapshot = snapshot
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

    def pre_forward(self, scheduler_output, input_batch=None, attn_metadata=None):
        self._eligible_forward = (
            self._allow_step
            and scheduler_output.has_sync_kv_loads
            and input_batch is not None
            and attn_metadata is not None
            and not input_batch.has_prefill
            and 0 < input_batch.num_tokens == input_batch.num_tokens_after_padding
        )
        return self._connector.pre_forward(scheduler_output, input_batch, attn_metadata=attn_metadata)

    def finish_forward(self) -> None:
        eligible = self._eligible_forward
        self._eligible_forward = False
        if not eligible or self._connector._disabled:
            return self._connector.finish_forward()

        worker = self._worker
        compute_stream = torch.accelerator.current_stream()
        worker._forward_ready_event.record(compute_stream)
        snapshot = self._snapshot
        pending_before = 0
        if snapshot is not None:
            pending_before = len(worker._pending_dma_descriptors)
            snapshot.submitted_completion = None
            snapshot.armed = (
                not worker._dma_submitted and not worker._per_layer_mirrored and not worker._submitted_mirror_layers
            )
        try:
            worker._finish_mirror_phase(worker._forward_ready_event)
        except BaseException:
            # The runner's failure boundary joins an already-recorded copy.
            if worker._dma_submitted:
                self._pending_event = worker.host_write_event
            raise
        finally:
            if snapshot is not None:
                snapshot.armed = False
        if worker._dma_submitted:
            self._pending_event = worker.host_write_event
            # Preserve the Host durability fence on AsyncOutput's copy stream.
            self._output_copy_stream.wait_event(self._pending_event)
            worker._dma_submitted = False
            if snapshot is not None and snapshot.submitted_completion is not None:
                pending = worker._pending_dma_descriptors
                # Direct fallback or an extra transfer keeps the sampling-only fence.
                self._overlap_draft = (
                    len(pending) == pending_before + 1 and pending[-1][0] is snapshot.submitted_completion
                )
        else:
            worker.host_write_event.record(compute_stream)
        worker._release_completed_dma_descriptors()

    def post_forward(self, finished_req_ids, defer_finalize=False):
        if not defer_finalize or not self._overlap_draft:
            self.join_pending()
        return self._connector.post_forward(finished_req_ids, defer_finalize=defer_finalize)

    def finalize_step(self):
        self.join_pending()
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


def overlap_mtp_host_mirror(runner):
    """Bind only the audited single-stage GLM MTP sampling path."""
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import HiSparseConnector
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
    from vllm.logger import init_logger
    from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

    from vllm_ascend.hisparse.backend import AscendHiSparseBackend

    connector = runner.kv_connector
    config = runner.vllm_config.parallel_config
    if (
        type(connector) is not ActiveKVConnector
        or type(connector.kv_connector) is not HiSparseConnector
        or runner.speculator is None
        or runner.speculative_config.method != "mtp"
        or runner.model_config.architecture != "GlmMoeDsaForCausalLM"
        or config.pipeline_parallel_size != 1
        or config.data_parallel_size != 1
        or config.decode_context_parallel_size != 1
        or runner.pcp_manager is not None
        or runner.is_pooling_model
        or runner.supports_mm_inputs
        or runner.aux_output_connector is not None
        or runner.kvpp.scheduler is not None
        or getattr(runner, "pd_dspark_aux_layer_ids", ())
    ):
        return connector
    worker = connector.kv_connector.connector_worker
    if (
        worker is None
        or not worker._initialized
        or not isinstance(worker.backend, AscendHiSparseBackend)
        or worker.shared_host_region is not None
        or not worker.is_host_writer
    ):
        return connector
    output = runner.output_copy_stream
    output_stream = torch.Stream(
        stream_id=output.stream_id, device_index=output.device_index, device_type=output.device_type
    )
    if output_stream == torch.accelerator.current_stream():
        return connector
    logger = init_logger(__name__)
    rank = get_tensor_model_parallel_rank()
    snapshot = TargetMirrorSnapshot.bind(worker, runner.speculator)
    worker._ascend_target_snapshot = snapshot
    logger.info("HiSparse MTP mirror overlap enabled: rank=%d", rank)
    if snapshot is not None:
        logger.info("HiSparse Target snapshot enabled: rank=%d", rank)
    return SamplingMirrorOverlapConnector(connector, worker, output_stream, snapshot)
