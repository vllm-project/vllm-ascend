# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.eplb_load import collect_moe_load


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("experts,window,kind", [(28, 50, 0), (29, 600, 0), (29, 5, 1)])
def test_eplb_load_graph_gates_and_window_wrap(experts, window, kind):
    counts = torch.arange(experts, device="npu", dtype=torch.int64) % 7
    tokens = counts.cumsum(0) if kind == 0 else counts
    loads = torch.zeros((window, experts), device="npu", dtype=torch.int32)
    counter = torch.tensor(window - 1, device="npu", dtype=torch.int32)
    enabled = torch.zeros((), device="npu", dtype=torch.int32)
    advance = torch.zeros_like(enabled)
    stream = torch.npu.Stream()

    def step():
        main = torch.npu.current_stream()
        stream.wait_stream(main)
        with torch.npu.stream(stream):
            collect_moe_load(tokens, loads, kind, counter, enabled, advance)
        main.wait_stream(stream)

    step()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        step()
    reference = torch.zeros((window, experts), dtype=torch.int32)
    reference_counter = window - 1
    for load_gate, counter_gate in [(0, 0), (1, 1), (0, 1), (1, 1)]:
        enabled.fill_(load_gate)
        advance.fill_(counter_gate)
        graph.replay()
        reference[reference_counter % window] += counts.cpu().to(torch.int32) * load_gate
        reference_counter += counter_gate
        torch.testing.assert_close(loads.cpu(), reference, rtol=0, atol=0)
        assert counter.item() == reference_counter
