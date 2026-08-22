# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.triton.fla.chunk_o import chunk_fwd_o


def _chunk_fwd_o_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> torch.Tensor:
    batch_size, sequence_length, num_query_heads, _ = q.shape
    num_value_heads = v.shape[2]
    chunks_per_sequence = (sequence_length + chunk_size - 1) // chunk_size
    value_heads_per_query_head = num_value_heads // num_query_heads
    output = torch.empty_like(v, dtype=torch.float32)

    for batch_idx in range(batch_size):
        for value_head_idx in range(num_value_heads):
            query_head_idx = value_head_idx // value_heads_per_query_head
            for chunk_idx in range(chunks_per_sequence):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, sequence_length)
                q_chunk = q[batch_idx, start:end, query_head_idx].float()
                k_chunk = k[batch_idx, start:end, query_head_idx].float()
                v_chunk = v[batch_idx, start:end, value_head_idx].float()
                g_chunk = g[batch_idx, start:end, value_head_idx].float()
                h_chunk = h[batch_idx * chunks_per_sequence + chunk_idx, value_head_idx].float()

                state_output = (q_chunk @ h_chunk) * torch.exp(g_chunk).unsqueeze(-1)
                attention = q_chunk @ k_chunk.transpose(0, 1)
                gate_delta = g_chunk.unsqueeze(1) - g_chunk.unsqueeze(0)
                gate = torch.where(gate_delta <= 0, torch.exp(gate_delta), 0.0)
                causal_mask = torch.ones_like(attention, dtype=torch.bool).tril()
                attention = torch.where(causal_mask, attention * gate, 0.0)
                output[batch_idx, start:end, value_head_idx] = scale * (state_output + attention @ v_chunk)

    return output


@pytest.mark.parametrize(
    # (1, 70) covers a tail chunk; (2, 130) additionally exercises the
    # per-batch state offset (boh = i_n * NT) inside the kernel.
    ("batch_size", "sequence_length"),
    [
        (1, 70),
        (2, 130),
    ],
)
def test_chunk_fwd_kernel_o_accuracy(batch_size: int, sequence_length: int):
    torch.manual_seed(2026)
    num_query_heads = 1
    num_value_heads = 2
    key_dim = 128
    value_dim = 96
    chunk_size = 64
    chunks_per_sequence = (sequence_length + chunk_size - 1) // chunk_size
    scale = key_dim**-0.5

    q_cpu = (torch.randn(batch_size, sequence_length, num_query_heads, key_dim) * 0.1).to(torch.bfloat16)
    k_cpu = (torch.randn_like(q_cpu.float()) * 0.1).to(torch.bfloat16)
    v_cpu = (torch.randn(batch_size, sequence_length, num_value_heads, value_dim) * 0.1).to(torch.bfloat16)
    h_cpu = (torch.randn(batch_size * chunks_per_sequence, num_value_heads, key_dim, value_dim) * 0.1).to(
        torch.bfloat16
    )
    # A non-increasing cumulative gate is the domain expected by safe_exp.
    gate_steps = torch.rand(batch_size, sequence_length, num_value_heads) * 0.02
    g_cpu = -torch.cumsum(gate_steps, dim=1)

    expected = _chunk_fwd_o_reference(q_cpu, k_cpu, v_cpu, h_cpu, g_cpu, scale, chunk_size)
    actual = chunk_fwd_o(
        q=q_cpu.npu(),
        k=k_cpu.npu(),
        v=v_cpu.npu(),
        h=h_cpu.npu(),
        g=g_cpu.npu(),
        scale=scale,
        chunk_size=chunk_size,
    )

    torch.testing.assert_close(actual.float().cpu(), expected, rtol=3e-2, atol=2e-2)
