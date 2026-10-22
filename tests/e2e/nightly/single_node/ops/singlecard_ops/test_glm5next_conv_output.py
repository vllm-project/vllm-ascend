# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401
from torch.nn import functional as F

from vllm_ascend.models.glm5next.ops.causal_conv1d import causal_conv1d


@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("use_graph", [False, True])
def test_causal_conv_shared_output_across_layers_and_graph_replay(dim_first, use_graph):
    """One metadata reset must cover every layer and active-to-padding transition."""
    batch, slots, state_len, dim = 8, 12, 3, 384
    generator = torch.Generator().manual_seed(19408)
    storage = (torch.randn(slots * state_len * dim * 2, generator=generator) * 0.2).bfloat16()
    strides = (state_len * dim * 2, 1, state_len) if dim_first else (state_len * dim * 2, dim, 1)
    backing = storage.npu()
    state = backing.as_strided((slots, state_len, dim), strides)
    weight_cpu = (torch.randn(state_len + 1, dim, generator=generator) * 0.2).bfloat16()
    weight = weight_cpu.npu()
    x = torch.zeros(batch, dim, dtype=torch.bfloat16, device="npu")
    output = torch.empty_like(x)
    indices = torch.arange(2, batch + 2, dtype=torch.int32, device="npu")
    starts = torch.arange(batch + 1, dtype=torch.int32, device="npu")

    def run():
        first = causal_conv1d(x, weight, state, starts, indices, run_mode=1, output_buffer=output)
        # Stand in for the first layer's consumer before the next layer reuses
        # the destination. No second clear is allowed between these two calls.
        consumed = first.clone()
        second = causal_conv1d(x, weight, state, starts, indices, run_mode=1, output_buffer=output)
        return consumed, second

    graph = None
    if use_graph:
        output.zero_()
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = run()

    for live in (batch, 5, 0, batch):
        backing.copy_(storage)
        input_cpu = (torch.randn(batch, dim, generator=generator) * 0.2).bfloat16()
        x.copy_(input_cpu)
        indices_cpu = torch.arange(2, batch + 2, dtype=torch.int32)
        indices_cpu[live:] = 0
        indices.copy_(indices_cpu)
        starts.copy_(torch.arange(batch + 1, dtype=torch.int32).clamp(max=live))
        expected_storage = storage.clone()
        expected_state = expected_storage.as_strided(state.shape, strides)
        expected_outputs = []
        for _ in range(2):
            expected = torch.zeros_like(input_cpu)
            for request in range(live):
                slot = int(indices_cpu[request])
                combined = torch.cat((expected_state[slot].float(), input_cpu[request : request + 1].float()))
                expected[request] = (
                    F.silu(F.conv1d(combined.T.unsqueeze(0), weight_cpu.float().T.unsqueeze(1), groups=dim))
                    .reshape(dim)
                    .bfloat16()
                )
                expected_state[slot].copy_(combined[-state_len:].bfloat16())
            expected_outputs.append(expected)
        # Match metadata preparation: reset the persistent buffer once before
        # replay, including rows that were active in the preceding graph run.
        output.zero_()
        if graph is None:
            result = run()
        else:
            graph.replay()
        for actual, expected in zip(result, expected_outputs):
            torch.testing.assert_close(actual.cpu(), expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(backing.cpu(), expected_storage, atol=0, rtol=0)
