# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common partial-budget allocation with eager and captured execution."""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import assign_draft_token_budget


@pytest.mark.parametrize("draft_budget", [2, 5, 7])
@pytest.mark.parametrize("full_graph", [False, True])
def test_common_partial_budget_allocator_keeps_request_prefixes(draft_budget, full_graph):
    """Exercise the shared ranking kernel with unequal counts and inactive rows."""
    device = torch.device("npu:0")
    confidences = torch.ones(5, 5, device=device)
    confidences[3].fill_(0.95)
    confidences[1].fill_(0.2)
    request_slots = torch.tensor([3, 1, 4, 0], dtype=torch.int32, device=device)
    scheduled = torch.tensor([5, 5, 0, 0], dtype=torch.int32, device=device)
    admitted = torch.empty_like(scheduled)
    survival = torch.empty(scheduled.numel() * confidences.shape[1], dtype=torch.int32, device=device)

    def forward():
        admitted.copy_(scheduled)
        assign_draft_token_budget(confidences, request_slots, admitted, draft_budget, survival)

    forward()
    if full_graph:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            forward()
        # Replays must recompute the allocation, including overwriting stale counts.
        admitted.fill_(-1)
        graph.replay()
    torch.npu.synchronize()
    assert admitted.cpu().tolist() == [min(draft_budget, 5), max(draft_budget - 5, 0), 0, 0]
