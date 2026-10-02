# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 fused prefill must preserve output, VK state, and changed-input replay."""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

from vllm_ascend.ops.kda import run_chunk_kda
from vllm_ascend.utils import enable_custom_op, is_950

if not hasattr(torch.ops._C_ascend, "chunk_kda_fwd"):
    enable_custom_op()


@pytest.mark.parametrize("tokens,requests", [(32, 1), (65, 3), (129, 1), (768, 3)])
def test_k3_prefill_v2_output_state_and_replay(tokens, requests):
    if not is_950():
        pytest.skip("A5 prefill specialization")
    torch.manual_seed(919 + tokens)
    heads, dim = 12, 128
    qkv = torch.randn(1, tokens, heads * dim * 3, dtype=torch.bfloat16, device="npu")
    q, k, v = [x.view(1, tokens, heads, dim) for x in qkv.chunk(3, dim=-1)]
    gate = torch.randn_like(q)
    beta = torch.randn(1, tokens, heads, device="npu").sigmoid()
    state = torch.randn(requests, heads, dim, dim, device="npu") * 0.05
    a_log = torch.randn(heads, device="npu") * 0.1
    bias = torch.randn(heads * dim, device="npu") * 0.1
    lengths = [0, tokens] if requests == 1 else [0, tokens // 3, 2 * (tokens // 3) + 1, tokens]

    def fused():
        return run_chunk_kda(q, k, v, gate, beta, state, lengths, None, a_log, bias, lower_bound=-5.0)

    expected = torch.ops._C_ascend.chunk_kda_fwd(
        l2norm_fwd(q.contiguous()),
        l2norm_fwd(k.contiguous()),
        v.contiguous(),
        gate.contiguous(),
        beta,
        dim**-0.5,
        64,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=lengths,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        A_log=a_log,
        dt_bias=bias,
        state_v_first=True,
    )
    actual = fused()
    for x, y in zip(actual, expected[:2]):
        x, y = x.cpu().double(), y.cpu().double()
        assert torch.isfinite(x).all()
        relative_rms = (x - y).square().mean().sqrt() / y.square().mean().sqrt().clamp_min(1e-8)
        assert relative_rms < 0.01

    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph):
        replay_output, replay_state = fused()
    for _ in range(3):
        qkv.add_(0.03125)
        gate.add_(-0.03125)
        state.add_(0.015625)
        graph.replay()
        eager_output, eager_state = fused()
        torch.npu.synchronize()
        torch.testing.assert_close(replay_output, eager_output, atol=0, rtol=0)
        torch.testing.assert_close(replay_state, eager_state, atol=0, rtol=0)
