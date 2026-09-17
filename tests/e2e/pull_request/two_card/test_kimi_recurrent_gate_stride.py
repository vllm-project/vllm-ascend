# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.kda import run_recurrent_kda


@torch.inference_mode()
@pytest.mark.parametrize("batch", [8, 16])
def test_recurrent_bfg_gate_stride(batch):
    """A BFG view must match a materialized gate, including state mutation."""
    torch.manual_seed(170917)
    steps, heads, dim = 4, 12, 128
    tokens = batch * steps
    qkv = torch.randn(tokens, 3 * heads * dim, dtype=torch.bfloat16, device="npu") * 0.2
    bfg = torch.randn(tokens, heads + 2 * heads * dim, dtype=torch.bfloat16, device="npu") * 0.2
    q, k, v = [value.view(1, tokens, heads, dim) for value in qkv.chunk(3, -1)]
    beta, gate, _ = bfg.split((heads, heads * dim, heads * dim), -1)
    gate = gate.view(1, tokens, heads, dim)
    beta = beta.float().sigmoid().unsqueeze(0)
    starts = torch.arange(0, tokens + 1, steps, dtype=torch.int32, device="npu")
    indices = (torch.arange(tokens, dtype=torch.int32, device="npu") + 2).view(batch, steps)
    accepted = torch.ones(batch, dtype=torch.int32, device="npu")
    a_log = torch.zeros(heads, dtype=torch.float32, device="npu")
    dt_bias = torch.zeros(heads * dim, dtype=torch.float32, device="npu")
    initial = torch.randn(tokens + 2, 2, heads, dim, dim, dtype=torch.float32, device="npu") * 0.01
    backing = [initial.clone(), initial.clone()]
    states = [value[:, 0] for value in backing]

    def forward(index):
        return run_recurrent_kda(
            q,
            k,
            v,
            gate.contiguous() if index == 0 else gate,
            beta,
            states[index],
            starts,
            indices,
            a_log,
            dt_bias,
            lower_bound=-5.0,
            num_accepted_tokens=accepted,
        )

    outputs = [forward(index) for index in range(2)]
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    torch.testing.assert_close(*backing, rtol=0, atol=0)
    graphs = []
    for index in range(2):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs[index] = forward(index)
        graphs.append(graph)
    for count in range(1, steps + 1):
        qkv.add_(0.03125)
        bfg.add_(-0.03125)
        accepted.fill_(count)
        for graph in graphs:
            graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        torch.testing.assert_close(*backing, rtol=0, atol=0)
