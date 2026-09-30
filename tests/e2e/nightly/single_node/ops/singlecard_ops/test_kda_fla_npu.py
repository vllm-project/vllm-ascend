# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the model adapters against an independent CPU recurrence."""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.e2e.nightly.single_node.ops.singlecard_ops.test_kimi_kda_recurrent_ascendc_npu import (
    recurrent_kda_reference,
)
from vllm_ascend.ops.kda import run_chunk_kda, run_recurrent_kda

pytest.importorskip("fla_npu.ops.ascendc", reason="requires a KDA-enabled fla_npu wheel")


@pytest.mark.parametrize("lower_bound", [None, -4.0])
@pytest.mark.parametrize("tokens", [1, 65, 129])
@torch.inference_mode()
def test_chunk_fla_varlen_output_and_vk_state(lower_bound, tokens):
    torch.manual_seed(42)
    cu = (0, tokens, 2 * tokens + 1)
    q, k, v = (torch.randn(1, cu[-1], 2, 128, dtype=torch.bfloat16) for _ in range(3))
    gate = torch.randn_like(q) * 0.1
    beta = torch.randn(1, cu[-1], 2).sigmoid()
    state = torch.randn(2, 2, 128, 128) * 0.01
    a_log, bias = torch.zeros(2), torch.zeros(256)
    # Match the existing external BF16 normalization boundary, independently of Triton.
    qk = [(x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)).to(x.dtype) for x in (q, k)]
    ref_out, ref_state = recurrent_kda_reference(
        *qk,
        v,
        gate,
        beta,
        state,
        cu_seqlens=cu,
        A_log=a_log,
        dt_bias=bias,
        use_gate_in_kernel=True,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound if lower_bound is not None else -5.0,
    )
    chunks = tuple(
        value
        for seq, length in enumerate((tokens, tokens + 1))
        for chunk in range((length + 63) // 64)
        for value in (seq, chunk)
    )
    out, final_state = run_chunk_kda(
        *(x.npu() for x in (q, k, v, gate, beta, state)),
        cu,
        chunks,
        a_log.npu(),
        bias.npu(),
        lower_bound=lower_bound,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref_out, rtol=0.02, atol=0.02)
    torch.testing.assert_close(final_state.cpu(), ref_state, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("lower_bound", [None, -4.0])
@pytest.mark.parametrize("preprocessed", [False, True])
@pytest.mark.parametrize("capture", [False, True])
@torch.inference_mode()
def test_recurrent_fla_mtp_strides_and_graph_replay(lower_bound, preprocessed, capture):
    torch.manual_seed(43)
    q, k, v = (torch.randn(1, 4, 2, 128 + pad, device="npu", dtype=torch.bfloat16)[..., :128] for pad in (16, 32, 64))
    gate = torch.randn_like(q) * 0.1
    beta = torch.randn(1, 4, 2, device="npu")
    if preprocessed:
        beta = beta.sigmoid()
    initial = torch.randn(8, 2, 128, 128) * 0.01
    pool = torch.full((8, 2, 2, 128, 128), 7.0, device="npu")
    state = pool[:, 0]
    state.copy_(initial)
    cu = torch.tensor([0, 2, 4], dtype=torch.int32, device="npu")
    slots = torch.tensor([[2, 3], [5, 6]], dtype=torch.int32, device="npu")
    accepted = torch.tensor([1, 2], dtype=torch.int32, device="npu")
    a_log, bias = torch.zeros(2, device="npu"), torch.zeros(256, device="npu")

    def run():
        return run_recurrent_kda(
            q,
            k,
            v,
            gate,
            beta,
            state,
            cu,
            slots,
            a_log,
            bias,
            lower_bound=lower_bound,
            beta_is_preprocessed=preprocessed,
            num_accepted_tokens=accepted,
        )

    if capture:
        for _ in range(2):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out = run()

    for _ in range(3):
        # Reuse captured addresses with changed inputs and reset the entire state pool.
        q.normal_()
        state.copy_(initial)
        ref_out, ref_state = recurrent_kda_reference(
            *(x.cpu() for x in (q, k, v, gate, beta)),
            initial,
            cu_seqlens=[0, 2, 4],
            ssm_state_indices=slots.cpu(),
            A_log=a_log.cpu(),
            dt_bias=bias.cpu(),
            num_accepted_tokens=accepted.cpu(),
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=not preprocessed,
            safe_gate=lower_bound is not None,
            lower_bound=lower_bound if lower_bound is not None else -5.0,
        )
        if capture:
            graph.replay()
        else:
            out = run()
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), ref_out, rtol=0.02, atol=0.02)
        torch.testing.assert_close(state.cpu(), ref_state, rtol=0.02, atol=0.02)
        torch.testing.assert_close(state[[0, 1, 4, 7]].cpu(), initial[[0, 1, 4, 7]], rtol=0, atol=0)
        assert torch.all(pool[:, 1] == 7)
