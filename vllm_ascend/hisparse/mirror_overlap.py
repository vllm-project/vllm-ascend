# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private MTP experiment: preserve output durability while overlapping Target D2H.

The common worker's mirror submission, invalidation, descriptor retirement and
next-step/shutdown fences remain unchanged. Only its immediate compute-stream
join is redirected to the existing output-copy stream, then conservatively
joined on compute at the existing post-draft finalize boundary. No new stream,
event, descriptor pool or asynchronous resource owner is introduced.

This candidate targets the reviewed Core 24420b7 worker/AsyncOutput contract.
It is not qualified for publication or enabled in the public foundation PR.
"""

from __future__ import annotations

import torch


def _payload_regions(tensor: torch.Tensor) -> list[tuple[int, int]] | None:
    """Describe actual contiguous rows, retaining gaps between Resident pages."""
    if tensor.ndim not in (2, 3) or tensor.stride(-1) != 1 or not all(tensor.shape):
        return None
    width = tensor.shape[-1]
    if tensor.stride(-2) != width:
        return None
    start = tensor.data_ptr()
    row_bytes = width * tensor.element_size()
    if tensor.ndim == 2:
        return [(start, start + tensor.shape[0] * row_bytes)]
    page_bytes = tensor.shape[1] * row_bytes
    page_stride = tensor.stride(0) * tensor.element_size()
    if page_stride < page_bytes:
        return None
    if page_stride == page_bytes:
        return [(start, start + tensor.shape[0] * page_bytes)]
    return [(start + page * page_stride, start + page * page_stride + page_bytes) for page in range(tensor.shape[0])]


def _separate_layers(caches, draft_layers: set[int]) -> bool:
    """Prove Target DMA cannot race the draft's cache reads/writes at cold bind."""
    regions = []
    for index, cache in enumerate(caches):
        payload = _payload_regions(cache)
        if payload is None:
            return False
        regions.extend((start, end, index in draft_layers) for start, end in payload)
    # A sweep over occupied ranges also handles interleaved per-page layer views.
    furthest_end = {False: -1, True: -1}
    for start, end, draft in sorted(regions):
        if furthest_end[not draft] > start:
            return False
        furthest_end[draft] = max(furthest_end[draft], end)
    return True


class MTPMirrorOverlapConnector:
    """Wrap this runner's direct connector; delegate all unmodified boundaries."""

    def __init__(self, connector, worker, output_copy_stream):
        self._connector = connector
        self._worker = worker
        self._output_copy_stream = output_copy_stream
        self._pending_event = None

    def __getattr__(self, name):
        return getattr(self._connector, name)

    def _join_compute(self) -> None:
        if self._pending_event is not None:
            torch.accelerator.current_stream().wait_event(self._pending_event)
            self._pending_event = None

    def pre_forward(self, *args, **kwargs):
        # Also protect a subsequent call if a proposal exited before finalize.
        self._join_compute()
        return self._connector.pre_forward(*args, **kwargs)

    def finish_forward(self) -> None:
        if self._connector._disabled:
            return self._connector.finish_forward()
        worker = self._worker
        compute_stream = torch.accelerator.current_stream()
        worker._forward_ready_event.record(compute_stream)
        # This is the original submission + written-slot invalidation routine.
        worker._finish_mirror_phase(worker._forward_ready_event)
        if worker.is_host_writer:
            if worker._dma_submitted:
                self._pending_event = worker.host_write_event
                # AsyncOutput keeps its original count-copy-before-draft order.
                # Its existing completion event cannot publish unfinished Host KV.
                self._output_copy_stream.wait_event(self._pending_event)
                worker._dma_submitted = False
            else:
                worker.host_write_event.record(compute_stream)
        worker._release_completed_dma_descriptors()

    def post_forward(self, finished_req_ids, defer_finalize=False):
        if not defer_finalize:
            self._join_compute()
        return self._connector.post_forward(finished_req_ids, defer_finalize=defer_finalize)

    def finalize_step(self) -> None:
        # Core calls this after the last draft, before clearing step metadata.
        self._join_compute()
        self._connector.finalize_step()

    def no_forward(self, scheduler_output):
        self._join_compute()
        return self._connector.no_forward(scheduler_output)

    def set_disabled(self, disabled) -> None:
        self._join_compute()
        self._connector.set_disabled(disabled)

    def reset_capture_state(self) -> None:
        self._join_compute()
        self._connector.reset_capture_state()


def overlap_mtp_host_mirror(connector, speculator, output_copy_stream):
    """Fall back to the common connector unless the independent-cache proof holds."""
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import HiSparseConnector
    from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

    from vllm_ascend.hisparse.backend import AscendHiSparseBackend

    if speculator is None or type(connector) is not ActiveKVConnector:
        return connector
    if type(connector.kv_connector) is not HiSparseConnector:
        return connector
    worker = connector.kv_connector.connector_worker
    if worker is None or not worker._initialized or not isinstance(worker.backend, AscendHiSparseBackend):
        return connector
    if worker.shared_host_region is not None or not worker.is_host_writer:
        return connector
    draft_layers = set(worker._draft_layers)
    expected_names = set(getattr(speculator, "draft_attn_layer_names", ()))
    if not draft_layers or draft_layers == set(range(len(worker.cache_handles))):
        return connector
    if {worker.cache_layer_names[index] for index in draft_layers} != expected_names:
        return connector
    if output_copy_stream == torch.accelerator.current_stream():
        return connector
    if not _separate_layers(worker.resident_caches, draft_layers):
        return connector
    if not _separate_layers(worker.host_caches, draft_layers):
        return connector
    return MTPMirrorOverlapConnector(connector, worker, output_copy_stream)
