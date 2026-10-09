# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.qwen3_dflash2 import _score_edges
from vllm_ascend.spec_decode.dflash2_proposer import AscendDflash2Proposer


class _ReplaySelector:
    def __init__(self, vocab_size, rank, top_k):
        self.predecessors = torch.randn(vocab_size, rank, device="npu") * 0.01
        self.successors = torch.randn_like(self.predecessors) * 0.01
        self.top_k = top_k
        self.graphs = {}

    def __call__(self, candidates, unary, hidden, anchors):
        inputs = (candidates, unary, hidden, anchors)
        batch_size = candidates.shape[0]
        if batch_size not in self.graphs:
            _score_edges(self.predecessors, self.successors, *inputs, self.top_k)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                scores = _score_edges(self.predecessors, self.successors, *inputs, self.top_k)
            # Retain inputs so the unfixed implementation fails a numerical
            # assertion instead of indexing freed device memory.
            self.graphs[batch_size] = (graph, scores, inputs)
        graph, scores, _ = self.graphs[batch_size]
        graph.replay()
        reference = _score_edges(self.predecessors, self.successors, *inputs, self.top_k)
        torch.testing.assert_close(scores, reference)
        return scores


@pytest.mark.parametrize("num_steps", [1, 2, 8])
@torch.inference_mode()
def test_dflash2_selector_replay_uses_current_inputs(num_steps):
    torch.manual_seed(42)
    max_reqs, top_k, rank, vocab_size = 3, 4, 8, 1024
    proposer = AscendDflash2Proposer.__new__(AscendDflash2Proposer)
    proposer.num_speculative_tokens = num_steps
    proposer.selector_top_k = top_k
    proposer._anchor_indices = torch.arange(max_reqs, device="npu") * (num_steps + 1)
    proposer.input_ids = torch.arange(max_reqs * (num_steps + 1), device="npu", dtype=torch.int32)
    shape = (max_reqs, num_steps, top_k)
    proposer._selector_input_buffers = (
        torch.empty(shape, device="npu", dtype=torch.int64),
        torch.empty(shape, device="npu"),
        torch.empty(max_reqs, num_steps, rank, device="npu"),
        torch.empty(max_reqs, device="npu", dtype=torch.int32),
    )
    selector = _ReplaySelector(vocab_size, rank, top_k)
    proposer.model = SimpleNamespace(model=SimpleNamespace(candidate_selector=selector))
    for step, num_reqs in enumerate([3, 1, 2, 1, 3, 2]):
        candidates = torch.arange(num_reqs * num_steps * top_k, device="npu").view(-1, top_k) + step * 100
        unary = torch.zeros(candidates.shape, device="npu")
        selected = step % top_k
        unary[:, selected] = 100
        hidden = torch.randn(num_reqs * num_steps, rank, device="npu")
        proposer.input_ids.add_(1)
        proposer.model.compute_candidates = lambda _, candidates=candidates, unary=unary: (candidates, unary)
        tokens, probs = proposer.compute_draft_token_ids(hidden)
        torch.testing.assert_close(tokens.cpu(), candidates[:, selected].cpu())
        assert probs is None
