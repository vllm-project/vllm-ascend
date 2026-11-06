# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real NPU recurrence/rollback checks, including a changed graph partition."""

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from vllm_ascend.models.glm5next.ops.causal_conv1d import causal_conv1d
from vllm_ascend.ops.triton.kda.fused_recurrent_kda import fused_recurrent_kda_varlen
from vllm_ascend.ops.triton.kda.output_writeback import write_recurrent_output

HEADS, DIM, WIDTH, TOKENS = 2, 128, 6, 8


def _reference(q, k, v, gate, beta, states, starts, indices, accepted, a_log, bias):
    result = torch.zeros_like(q)
    states = states.clone()
    for request, (begin, end) in enumerate(zip(starts[:-1], starts[1:])):
        if begin == end:
            continue
        hidden = states[indices[request, accepted[request] - 1]].float().clone()
        for token in range(begin, end):
            query = F.normalize(q[0, token].float(), dim=-1, eps=1e-12)
            key = F.normalize(k[0, token].float(), dim=-1, eps=1e-12)
            # The kernels normalize with sqrt(sum(x*x) + epsilon).
            query *= torch.sqrt(q[0, token].float().square().sum(-1, keepdim=True)) / torch.sqrt(
                q[0, token].float().square().sum(-1, keepdim=True) + 1e-6
            )
            key *= torch.sqrt(k[0, token].float().square().sum(-1, keepdim=True)) / torch.sqrt(
                k[0, token].float().square().sum(-1, keepdim=True) + 1e-6
            )
            decay = -4.0 * torch.sigmoid(a_log.exp()[:, None] * (gate[0, token].float() + bias.view(HEADS, DIM)))
            hidden *= decay.exp()[:, None, :]
            residual = v[0, token].float() - torch.einsum("hvk,hk->hv", hidden, key)
            residual *= beta[0, token].float().sigmoid()[:, None]
            hidden += residual[:, :, None] * key[:, None, :]
            result[0, token] = torch.einsum("hvk,hk->hv", hidden, query) * DIM**-0.5
            states[indices[request, token - begin]].copy_(hidden)
    return result, states


@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "acl_graph"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("state_row_width", [6, 8])
@torch.inference_mode()
def test_adaptive_kda_accepted_history_can_exceed_current_query(graph_mode, state_dtype, state_row_width):
    generator = torch.Generator().manual_seed(60229)
    merged_cpu = torch.randn(1, TOKENS, 3 * HEADS * DIM, generator=generator).bfloat16()
    merged = merged_cpu.npu()
    q, k, v = (part.view(1, TOKENS, HEADS, DIM) for part in merged.split(HEADS * DIM, dim=-1))
    q_cpu, k_cpu, v_cpu = (part.view(1, TOKENS, HEADS, DIM) for part in merged_cpu.split(HEADS * DIM, dim=-1))
    gate_cpu = torch.randn(q.shape, generator=generator).bfloat16()
    beta_storage = torch.randn(1, TOKENS, HEADS + 4, generator=generator).bfloat16()
    beta_cpu = beta_storage[..., :HEADS]
    gate, beta = gate_cpu.npu(), beta_storage.npu()[..., :HEADS]
    a_cpu, bias_cpu = torch.zeros(HEADS), torch.full((HEADS * DIM,), -1.0)
    a_log, bias = a_cpu.npu(), bias_cpu.npu()
    initial = (torch.randn(32, HEADS, DIM, DIM, generator=generator) * 0.01).to(state_dtype)
    state = initial.npu()
    indices_storage = torch.arange(1, 4 * state_row_width + 1, dtype=torch.int32).view(4, state_row_width)
    indices_cpu = indices_storage[:, :WIDTH]
    indices = indices_storage.npu()[:, :WIDTH]
    starts = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32, device="npu")
    accepted = torch.ones(4, dtype=torch.int32, device="npu")
    output = torch.empty_like(q)

    def run():
        value = fused_recurrent_kda_varlen(q, k, v, gate, beta, state, starts, indices, accepted, a_log, bias, -4.0)
        write_recurrent_output(value, output, starts)

    graph = None
    if graph_mode:
        for _ in range(3):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
    # Capture has four two-token rows; replay admits one six-token row and
    # another one-token row, then shrinks/reuses them and goes fully idle.
    for lengths, previous in [([1, 6, 0, 0], [6, 1, 1, 1]), ([6, 1, 0, 0], [1, 6, 1, 1]), ([0] * 4, [1] * 4)]:
        starts_cpu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
        accepted_cpu = torch.tensor(previous, dtype=torch.int32)
        starts.copy_(starts_cpu)
        accepted.copy_(accepted_cpu)
        state.copy_(initial)
        output.fill_(float("nan"))
        run() if graph is None else graph.replay()
        expected, expected_state = _reference(
            q_cpu,
            k_cpu,
            v_cpu,
            gate_cpu,
            beta_cpu,
            initial,
            starts_cpu.tolist(),
            indices_cpu,
            accepted_cpu,
            a_cpu,
            bias_cpu,
        )
        torch.testing.assert_close(output.cpu(), expected, rtol=2e-2, atol=3e-4)
        torch.testing.assert_close(state.cpu(), expected_state, rtol=2e-2, atol=3e-4)
        # Untouched cache slots and graph-padding outputs must be exact.
        touched = {int(indices_cpu[r, t]) for r, length in enumerate(lengths) for t in range(length)}
        untouched = [i for i in range(initial.shape[0]) if i not in touched]
        torch.testing.assert_close(state.cpu()[untouched], initial[untouched], rtol=0, atol=0)
        assert torch.count_nonzero(output[:, sum(lengths) :]).item() == 0


@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "acl_graph"])
@torch.inference_mode()
def test_adaptive_conv_rollback_and_padding(graph_mode):
    generator = torch.Generator().manual_seed(60229)
    channels, kernel_width = 384, 4
    x_cpu = torch.randn(TOKENS, channels, generator=generator).bfloat16()
    weights_cpu = torch.randn(kernel_width, channels, generator=generator).bfloat16()
    initial = torch.randn(4, kernel_width + WIDTH - 2, channels, generator=generator).bfloat16()
    x, weights, state = x_cpu.npu(), weights_cpu.npu(), initial.npu()
    starts = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32, device="npu")
    accepted = torch.ones(4, dtype=torch.int32, device="npu")
    slots = torch.tensor([1, 2, 0, 0], dtype=torch.int32, device="npu")
    output = torch.empty_like(x)

    def run():
        output.zero_()
        causal_conv1d(
            x,
            weights,
            state,
            starts,
            slots,
            run_mode=1,
            num_accepted_tokens=accepted,
            output_buffer=output,
            max_query_len=WIDTH,
        )

    graph = None
    if graph_mode:
        for _ in range(3):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
    for lengths, previous in [([1, 6, 0, 0], [6, 1, 1, 1]), ([6, 1, 0, 0], [1, 6, 1, 1]), ([0] * 4, [1] * 4)]:
        ends = [0, *torch.tensor(lengths).cumsum(0).tolist()]
        starts.copy_(torch.tensor(ends, dtype=torch.int32))
        accepted.copy_(torch.tensor(previous, dtype=torch.int32))
        state.copy_(initial)
        run() if graph is None else graph.replay()
        expected, expected_state = torch.zeros_like(x_cpu), initial.clone()
        for r, length in enumerate(lengths[:2]):
            if not length:
                continue
            slot, offset = r + 1, previous[r] - 1
            history = initial[slot, offset : offset + kernel_width - 1]
            sequence = torch.cat((history, x_cpu[ends[r] : ends[r + 1]]))
            for token in range(length):
                value = (sequence[token : token + kernel_width].float() * weights_cpu.float()).sum(0)
                expected[ends[r] + token] = F.silu(value).bfloat16()
            updated = torch.cat((history[1:], x_cpu[ends[r] : ends[r + 1]]))
            expected_state[slot, : updated.shape[0]] = updated
        torch.testing.assert_close(output.cpu(), expected, rtol=2e-2, atol=1e-2)
        torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0)
