# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-event and graph-replay checks; no model weights are needed."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.observability.forward_metrics import ForwardTimeCollector, npu_stream_id, run_timed_forward


@pytest.mark.parametrize("graph_mode", [False, True])
def test_forward_timing_device_events_and_replay(graph_mode):
    torch.npu.set_device(0)
    value = torch.ones(4096, device="npu")
    result = torch.empty_like(value)

    def body():
        torch.add(value, 2, out=result)
        return result

    body()
    torch.npu.synchronize()
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            body()
        forward = graph.replay
    else:
        forward = body
    collector = ForwardTimeCollector(0, window_size=8, stream_getter=npu_stream_id)
    for step in range(3):
        value.fill_(step)
        run_timed_forward(collector, "draft", lambda: "decode", 1, forward)
        # Synchronization is external to the collector, only for test assertions.
        torch.npu.synchronize()
        torch.testing.assert_close(result.cpu(), torch.full((4096,), step + 2.0))
    rows = collector.flush_window(final=True)
    assert sum(row.count for row in rows) == 3
    assert all(row.errors == row.dropped == 0 for row in rows)
    assert all(row.avg_ms >= 0 for row in rows)


def test_forward_timing_is_inside_reference_event_interval():
    torch.npu.set_device(0)
    value = torch.ones((512, 512), device="npu")
    result = torch.empty_like(value)
    torch.mm(value, value, out=result)
    torch.npu.synchronize()
    reference_start = torch.npu.Event(enable_timing=True)
    reference_end = torch.npu.Event(enable_timing=True)
    collector = ForwardTimeCollector(0, stream_getter=npu_stream_id)
    reference_start.record()
    run_timed_forward(collector, "target", lambda: "decode", 1, lambda: torch.mm(value, value, out=result))
    reference_end.record()
    torch.npu.synchronize()
    rows = collector.flush_window(final=True)
    assert len(rows) == 1
    assert rows[0].count == 1
    assert rows[0].errors == rows[0].dropped == 0
    # The outer interval includes the collector's two event markers as well.
    assert 0 <= rows[0].avg_ms <= reference_start.elapsed_time(reference_end) + 0.01
