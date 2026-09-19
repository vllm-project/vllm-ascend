# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify graph replay and source validity on a real NPU."""

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.eplb.diagnostics.probe import ExpertLoadProbe


@pytest.mark.parametrize("source_offset", [0, 2])
def test_graph_replay_excludes_padding_and_dummy(source_offset):
    torch.npu.set_device(0)
    probe = ExpertLoadProbe(2, "npu")
    probe.source_token_count = torch.tensor(source_offset + 2, device="npu")
    probe.source_positions = torch.arange(3, device="npu")
    ids = torch.tensor([[0], [1], [1]], device="npu")
    mask = torch.ones(3, dtype=torch.bool, device="npu")
    probe.record_routes(ids, mask, source_offset)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        probe.record_routes(ids, mask, source_offset)
    graph.replay()
    torch.npu.synchronize()
    pointer = probe.totals.data_ptr()
    probe.totals.zero_()
    graph.replay()
    # Dummy graph participation changes only the call-alignment bookkeeping.
    probe.source_token_count.zero_()
    graph.replay()
    probe.source_token_count.fill_(source_offset + 3)
    ids.copy_(torch.tensor([[0], [0], [1]], device="npu"))
    graph.replay()
    assert probe.totals.cpu().tolist() == [3, 2, 3, 0]
    assert probe.totals.data_ptr() == pointer


def test_graph_history_uses_device_slot_and_excludes_dummy_padding():
    torch.npu.set_device(0)
    probe = ExpertLoadProbe(2, "npu")
    probe.source_token_count = torch.tensor(2, device="npu")
    probe.source_positions = torch.arange(3, device="npu")
    probe.history = torch.zeros((3, 4), dtype=torch.int64, device="npu")
    probe.history_slot = torch.zeros(1, dtype=torch.int64, device="npu")
    ids = torch.tensor([[0], [1], [99]], device="npu")
    probe.record_routes(ids)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        probe.record_routes(ids)
    pointer = probe.history.data_ptr()
    probe.history.zero_()
    for step, tokens in enumerate((2, 0, 1)):
        probe.history_slot.fill_(step)
        probe.source_token_count.fill_(tokens)
        graph.replay()
    assert probe.history.cpu().tolist() == [[1, 1, 1, 0], [0, 0, 1, 0], [1, 0, 1, 0]]
    assert not probe.totals.cpu().any()
    assert probe.history.data_ptr() == pointer
