# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Queue small mapped Host writes behind the common HiSparse DMA fence.

The common connector keeps its writer policy, stream and completion event.
Only small, disjoint writes use AIV; page transfers retain the native DMA path.
"""

from __future__ import annotations

from pathlib import Path

import torch
from vllm.triton_utils import triton

from vllm_ascend.distributed.kv_transfer.hisparse.host_mirror_descriptors import prepare_snapshot
from vllm_ascend.ops.triton.hisparse_host_mirror import store_opaque_extents


class QueuedHostDMA:
    """Use the caller's stream and completion event; allocate no new stream."""

    MAX_DESCRIPTORS = 256
    # On A5, 1--128 extents did not beat DMA; 312 and larger did. Keep
    # small draft writes off the descriptor-upload/kernel-launch path.
    MIN_MAPPED_DESCRIPTORS = 312

    def __init__(self):
        import vllm_ascend

        library = Path(vllm_ascend.__file__).parent / "ops/hisparse_pta_host2511.so"
        if not hasattr(torch.ops.hisparse_pta_host2511, "batch_copy"):
            torch.ops.load_library(str(library))
        self.extension = torch.ops.hisparse_pta_host2511
        self.cookie = id(self)
        self.pending = {}
        self.uncertain = False
        self.binding = None
        self.owners = []
        self.device = None
        self.submitted_batches = self.completed_batches = 0
        self.last_statuses = []
        self._mapped_hosts = []
        self._device_extents = []
        self.aiv_submitted_batches = self.native_submitted_batches = 0

    def _bind(self, owners):
        binding = tuple(
            (t.device, t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype)
            for t in owners
        )
        if self.binding is not None:
            if binding != self.binding:
                raise RuntimeError("Host DMA payload binding changed before shutdown")
            return
        retained = []
        devices = set()
        storages = set()
        for tensor in owners:
            if tensor.device.type == "cpu":
                if not tensor.is_contiguous():
                    raise ValueError("Host mirror payload must be contiguous")
                # Keep the actual registered Host extent, not allocator padding.
                retained.append(tensor.view(torch.uint8).reshape(-1))
            elif tensor.device.type == "npu":
                storage = tensor.untyped_storage()
                key = (tensor.device, storage.data_ptr(), storage.nbytes())
                devices.add(tensor.device.index)
                if key not in storages:
                    storages.add(key)
                    # Packed resident layer views are strided. Retain an alias
                    # of their real storage without copying or changing layout.
                    alias = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                    alias.set_(storage, 0, (storage.nbytes(),), (1,))
                    retained.append(alias)
            else:
                raise ValueError("Unsupported Host mirror payload device")
        if len(devices) != 1 or not any(t.device.type == "cpu" for t in retained):
            raise ValueError("Host DMA needs Host owners and exactly one NPU")
        self.device = next(iter(devices))
        self.owners = retained
        self.binding = binding
        mappings = []
        for tensor in owners:
            if tensor.device.type == "cpu":
                mapping = getattr(tensor, "_hisparse_registration", None)
                if mapping is None or mapping.closed or not mapping._registered or mapping.device.index != self.device:
                    raise RuntimeError("Queued Host mirror requires the existing live mapping owner")
                mapping.address_of(tensor)
                if not any(mapping is existing for existing in mappings):
                    mappings.append(mapping)
        self._mapped_hosts = mappings
        self._device_extents = [
            (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()) for t in retained if t.device.type == "npu"
        ]

    def submit(self, source, destination, sizes, owners, completion_event):
        if self.uncertain or completion_event in self.pending:
            raise RuntimeError("Uncertain DMA or reused completion event")
        if source.ndim != 1 or destination.shape != source.shape or sizes.shape != source.shape:
            raise ValueError("Host DMA descriptor vectors differ")
        if any(
            t.device.type != "cpu" or t.dtype != torch.int64 or not t.is_contiguous()
            for t in (source, destination, sizes)
        ):
            raise ValueError("Host DMA descriptors must be contiguous CPU int64")
        entry = dict(kind="empty", tickets=[], launched=0)
        self.pending[completion_event] = entry
        if source.numel() == 0:
            return
        try:
            self._bind(owners)
            for mapping in self._mapped_hosts:
                if mapping.closed or not mapping._registered:
                    raise RuntimeError("Host mapping closed while queued DMA is live")
            matrix, eligible, reason = prepare_snapshot(
                source.numpy(),
                destination.numpy(),
                sizes.numpy(),
                self._device_extents,
                [(m.host_address, m.nbytes, m.device_ptr) for m in self._mapped_hosts],
                min_aiv_descriptors=self.MIN_MAPPED_DESCRIPTORS,
            )
            entry["selection"] = reason
            if eligible:
                if torch.npu.is_current_stream_capturing():
                    raise RuntimeError("Graph-external Host writeback was called during capture")
                entry["kind"] = "mapped_aiv"
                # One immutable 3xN upload; each row is a contiguous descriptor vector.
                # Strong references precede enqueue and live until the caller's fence.
                # Pageable H2D can block PTA's dequeue thread behind this
                # stream's producer event. A private pinned snapshot keeps
                # the upload asynchronous and remains owned until the fence.
                entry["host_descriptors"] = torch.from_numpy(matrix).pin_memory()
                entry["device_descriptors"] = entry["host_descriptors"].to(
                    torch.device("npu", self.device), non_blocking=True
                )
                vectors = entry["device_descriptors"]
                chunks = triton.cdiv(int(matrix[2].max()), 512)
                # AIV tiles already partition disjoint extents. One launch
                # avoids creating CPU-dispatched tensor views for each batch.
                store_opaque_extents[(source.numel(), chunks)](vectors, source.numel(), BLOCK=512)
                entry["launched"] += 1
                self.submitted_batches += 1
                self.aiv_submitted_batches += 1
            else:
                # Prefill/page extents and overlapping writes keep original ordered DMA.
                entry["kind"] = "native_dma"
                # The installed C++ interface rejects overlap within one
                # batch. Submit each such extent separately on the same
                # existing stream, preserving original descriptor order.
                batch_size = 1 if reason == "ordered_overlapping_destinations" else self.MAX_DESCRIPTORS
                for offset in range(0, source.numel(), batch_size):
                    end = offset + batch_size
                    ticket = self.extension.batch_copy(
                        source[offset:end],
                        destination[offset:end],
                        sizes[offset:end],
                        1,
                        self.owners,
                        self.device,
                        self.cookie,
                    )
                    entry["tickets"].append(ticket)
                    self.submitted_batches += 1
                    self.native_submitted_batches += 1
        except BaseException:
            # Never retry or fall back after an uncertain asynchronous enqueue.
            self.uncertain = True
            raise

    def release(self, completion_event):
        if self.uncertain or not completion_event.query():
            raise RuntimeError("DMA owners cannot retire before their completion event")
        entry = self.pending[completion_event]
        statuses = [self.extension.copy_status(ticket) for ticket in entry["tickets"]]
        if not all(s[4] == 0 and s[6] == 1 for s in statuses):
            self.uncertain = True
            raise RuntimeError("PTA Host DMA submission failed or is incomplete")
        for ticket in entry["tickets"]:
            self.extension.release_copy(ticket, True, False)
        self.last_statuses = statuses
        self.completed_batches += len(entry["tickets"]) + entry["launched"]
        del self.pending[completion_event]

    def drain(self):
        # Called after the common worker has successfully drained its DMA stream.
        # Failed submissions retain mappings and payloads for explicit teardown.
        if self.uncertain:
            raise RuntimeError("Retain uncertain queued DMA until process teardown")
        for event in tuple(self.pending):
            self.release(event)
        if self.extension.pending_copies(self.cookie):
            raise RuntimeError("Unaccounted native DMA tickets remain")
        self.owners.clear()
        self._mapped_hosts.clear()
        self._device_extents.clear()
        self.binding = None
