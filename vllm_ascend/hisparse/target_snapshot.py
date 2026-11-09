# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Own Target row snapshots until asynchronous Host mirroring completes."""

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
