# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM PR #64531.

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass(slots=True)
class _PendingCopy:
    event: torch.npu.Event | None

    def wait(self) -> None:
        if self.event is not None:
            self.event.synchronize()
            self.event = None


class AscendDraftTokensHandler:
    """Retain draft rows with the event for their own batch's D2H copy."""

    def __init__(self, device: torch.device):
        self.device = device
        self.copy_stream = torch.npu.Stream(device=device)
        self._drafts: dict[str, tuple[np.ndarray, _PendingCopy] | int] = {}
        self._requested_ids: Iterable[str] | None = None

    def begin_step(self, scheduler_output: SchedulerOutput) -> None:
        # Older vLLM baselines expose only the deferred grammar flag.
        needs_drafts = getattr(
            scheduler_output,
            "has_structured_output_spec_decode",
            getattr(scheduler_output, "pending_structured_output_tokens", False),
        )
        self._requested_ids = scheduler_output.scheduled_spec_decode_tokens.keys() if needs_drafts else None

    def set_draft_tokens(self, input_batch: InputBatch, draft_tokens: torch.Tensor) -> None:
        if not input_batch.has_structured_output_reqs:
            for req_id in input_batch.req_ids:
                self._drafts[req_id] = draft_tokens.shape[1]
            return

        self.copy_stream.wait_stream(torch.npu.current_stream(self.device))
        event = torch.npu.Event()
        with torch.npu.stream(self.copy_stream):
            rows = async_copy_to_np(draft_tokens)
            draft_tokens.record_stream(self.copy_stream)
            event.record()
        pending = _PendingCopy(event)
        for req_id, row in zip(input_batch.req_ids, rows):
            self._drafts[req_id] = (row, pending)

    def discard(self, req_id: str) -> None:
        self._drafts.pop(req_id, None)

    def get_draft_tokens(self) -> DraftTokenIds:
        selected = self._drafts if self._requested_ids is None else self._requested_ids
        req_ids: list[str] = []
        token_ids: list[list[int]] = []
        for req_id in selected:
            drafts = self._drafts.get(req_id)
            if drafts is None:
                continue
            req_ids.append(req_id)
            if isinstance(drafts, int):
                token_ids.append([-1] * drafts)
            else:
                row, pending = drafts
                pending.wait()
                token_ids.append(row.tolist())
        return DraftTokenIds(req_ids, token_ids)
