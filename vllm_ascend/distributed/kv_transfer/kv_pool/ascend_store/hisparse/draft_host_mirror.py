# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Order shared Host bytes before the standard child-connector save hooks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker


class BulkSaveParticipant(Protocol):
    def has_pending_bulk_save(self) -> bool: ...


class HiSparseDraftHostMirrorSaveHook:
    def __init__(self, source: HiSparseConnectorWorker, stores: Sequence[BulkSaveParticipant]) -> None:
        self._source = source
        self._stores = tuple(stores)

    def flush_host_mirror_if_needed(self) -> None:
        """Order pending draft DMA before the child connectors save Host bytes."""
        if any(store.has_pending_bulk_save() for store in self._stores):
            # Queue the mirror once, then let each child record its save event.
            # Sender threads wait for those events before reading Host bytes.
            finish_draft_host_mirror_for_save(self._source)


def finish_draft_host_mirror_for_save(worker: HiSparseConnectorWorker) -> None:
    """Make Core's pending draft rows durable before the external Host PUT."""
    pending = getattr(worker, "_draft_mirror_pending", None)
    if pending is False:
        return
    if pending is not True:
        raise RuntimeError("HiSparse bulk save requires Core's _draft_mirror_pending flag.")

    draft_layers = getattr(worker, "_draft_layers", None)
    enqueue_row_dma = getattr(worker, "_enqueue_row_dma", None)
    if not isinstance(draft_layers, tuple) or not draft_layers or not callable(enqueue_row_dma):
        raise RuntimeError("HiSparse bulk save requires Core's pending draft layers and row DMA hook.")
    if (
        not isinstance(getattr(worker, "_dma_submitted", None), bool)
        or getattr(worker, "host_write_event", None) is None
    ):
        raise RuntimeError("HiSparse bulk save requires Core's DMA submission flag and Host write event.")

    # Called by the parent connector during Core's post-draft finalize_step. Core
    # ordinarily defers the draft Host copy until the next start_step, too
    # late for this step's PUT. Consume only the draft-mirror flag so Core's
    # _finish_previous_step still submits its post-forward page transfers.
    # Replace this adapter when Core exposes a public draft-row flush hook.
    compute_stream = torch.accelerator.current_stream()
    draft_ready = torch.Event()
    draft_ready.record(compute_stream)
    enqueue_row_dma(draft_layers, ready_event=draft_ready)
    worker._draft_mirror_pending = False
    if worker._dma_submitted:
        compute_stream.wait_event(worker.host_write_event)
        worker._dma_submitted = False
