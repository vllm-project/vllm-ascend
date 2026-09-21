# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed KDA sequences must not overwrite adjacent scratch buffers."""

import itertools

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "heads,lengths",
    [
        (16, (65,) * 32),
        (16, (72, 72, 105, 76, 85, 104, 69, 78) * 4),
        (32, (64, 65, 72, 127) * 8),
        (64, (65, 72, 105, 127) * 4),
    ],
)
@torch.inference_mode()
def test_chunk_kda_packed_workspace(dtype, heads, lengths):
    assert enable_custom_op()
    torch.manual_seed(1024)
    dim = 128
    chunk_size = 64
    total_tokens = sum(lengths)
    shape = (1, total_tokens, heads, dim)
    q = F.normalize(torch.randn(shape, device="npu"), dim=-1).to(dtype)
    k = F.normalize(torch.randn(shape, device="npu"), dim=-1).to(dtype)
    v = (torch.randn(shape, device="npu") * 0.05).to(dtype)
    gate = torch.randn(shape, device="npu", dtype=torch.float32)
    beta = torch.rand((1, total_tokens, heads), device="npu")
    initial_state = torch.randn((len(lengths), heads, dim, dim), device="npu") * 0.01
    a_log = torch.zeros(heads, device="npu")
    dt_bias = torch.zeros(heads * dim, device="npu")
    cu_seqlens = tuple(itertools.accumulate(lengths, initial=0))

    def run(q, k, v, gate, beta, state, cu):
        return torch.ops._C_ascend.chunk_kda_fwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            gate.contiguous(),
            beta.contiguous(),
            dim**-0.5,
            chunk_size,
            layout="BSND",
            initial_state=state.contiguous(),
            output_final_state=True,
            cu_seqlens=cu,
            safe_gate=True,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            state_v_first=True,
        )[:2]

    outputs = []
    states = []
    for seq, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        output, state = run(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            gate[:, start:end],
            beta[:, start:end],
            initial_state[seq : seq + 1],
            (0, end - start),
        )
        outputs.append(output)
        states.append(state)
    expected_output = torch.cat(outputs, dim=1)
    expected_state = torch.cat(states, dim=0)
    assert torch.isfinite(expected_output).all()
    assert torch.isfinite(expected_state).all()

    # Repeat with allocator reuse: the old overlapping scratch regions could
    # corrupt different sequences across identical invocations.
    for _ in range(3):
        output, state = run(q, k, v, gate, beta, initial_state, cu_seqlens)
        assert torch.isfinite(output).all()
        assert torch.isfinite(state).all()
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
