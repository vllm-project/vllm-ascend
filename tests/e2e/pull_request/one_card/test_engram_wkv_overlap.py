# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay WKV projections with fresh externally produced lookup rows."""

import torch
import torch_npu  # noqa: F401

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.models.deepseek_v41.engram.layer import AscendEngram
from vllm_ascend.models.deepseek_v41.engram.windowed import EngramProjectionSchedule


@torch.inference_mode()
def test_engram_wkv_external_graph_replay():
    # Exercise the real projection/gate methods without loading a checkpoint.
    layer = object.__new__(AscendEngram)
    torch.nn.Module.__init__(layer)
    layer.dim, layer.hc_mult, layer.eps = 4, 2, 1e-6
    layer.wkv = torch.nn.Linear(8, 12, bias=False, dtype=torch.bfloat16, device="npu")
    layer.q_weight = torch.nn.Parameter(torch.ones(2, 4, dtype=torch.bfloat16, device="npu"), requires_grad=False)
    layer.k_weight = torch.nn.Parameter(torch.ones(2, 4, dtype=torch.bfloat16, device="npu"), requires_grad=False)
    rows = {i: torch.zeros(4, 8, dtype=torch.bfloat16, device="npu") for i in (1, 7)}
    projected = {i: torch.empty(2, 12, dtype=torch.bfloat16, device="npu") for i in rows}
    hidden = torch.ones(2, 2, 4, dtype=torch.bfloat16, device="npu")
    mask = torch.ones(2, dtype=torch.bool, device="npu")
    side, producer = torch.npu.Stream(), torch.npu.Stream()
    windows = {i: torch.npu.Event() for i in rows}
    ready = {i: torch.npu.ExternalEvent() for i in rows}
    for event in (*windows.values(), *ready.values()):
        event.record(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    with (
        torch.npu.graph(graph),
        EngramProjectionSchedule(
            {i: layer for i in rows},
            rows,
            projected,
            side,
            windows,
            tokens=2,
            shard=lambda x: x[2:].clone(),
            lookup_ready=ready,
            external=True,
        ) as schedule,
    ):
        schedule.submit_router(0)
        first = layer.apply_gate(hidden, schedule.finish(1), mask, None)
        schedule.submit_router(1)
        last = layer.apply_gate(hidden, schedule.finish(7), mask, None)
    for step in range(4):
        data = (torch.arange(32, dtype=torch.bfloat16).reshape(4, 8) / 16 + step).npu()
        producer.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(producer):
            for i in rows:
                rows[i].copy_(data + i)
                ready[i].record(producer)
        graph.replay()
        for i, actual in ((1, first), (7, last)):
            expected = layer(hidden, (data + i)[2:].clone(), mask, None)
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
