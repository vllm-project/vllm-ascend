# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish synchronous tree proposals independently of draft-chain depth."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

    from vllm_ascend.worker.v2.states import AscendRequestState


class TreeDraftTokensHandler(DraftTokensHandler):
    """Publish the full tree budget from device request state."""

    def __init__(self, device: torch.device, req_states: AscendRequestState) -> None:
        super().__init__(device)
        self.req_states = req_states

    def set_draft_tokens(self, input_batch: InputBatch, draft_tokens: torch.Tensor) -> None:
        if draft_tokens.shape[1] == 0:
            # Preserve the upstream skip_drafts contract instead of publishing
            # the previous tree's selected nodes at the output-length boundary.
            super().set_draft_tokens(input_batch, draft_tokens)
            return
        # MRV2 passes only K columns after #57053. The completed tree proposal
        # is already in request state, where storage retains its B-column width.
        tokens = self.req_states.draft_tokens[input_batch.idx_mapping]
        super().set_draft_tokens(input_batch, tokens)
