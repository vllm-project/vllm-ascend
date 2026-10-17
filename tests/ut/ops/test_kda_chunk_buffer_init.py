# SPDX-License-Identifier: Apache-2.0
# FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the KDA chunked-prefill pipeline's buffer init.

The chunked-prefill kernels (``chunk_gated_delta_rule_fwd_h_kda``,
``chunk_local_cumsum``, ``recompute_w_u_fwd``) write tile-shaped regions and
leave tail elements of their output buffers untouched whenever the sequence
length is not tile-aligned. With ``torch.empty*`` those bytes hold whatever
the caching allocator last served there, so two identical calls can produce
different recurrent states — and the final state is the hand-off that decode
builds on. These tests pin the buffers to zero-initialized behavior:

  1. determinism: identical inputs -> bitwise-identical outputs/states
  2. allocator churn: results are immune to stale pool contents
"""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")  # noqa: F401  (registers the NPU backend)

from vllm_ascend.ops.triton.kda.kda import chunk_kda  # noqa: E402

H, K_DIM, V_DIM = 8, 128, 128


def _kda_inputs(T: int, seed: int = 0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(1, T, H, K_DIM, generator=gen).to("npu", torch.bfloat16)
    k = torch.randn(1, T, H, K_DIM, generator=gen).to("npu", torch.bfloat16)
    v = (
        torch.randn(1, T, H, V_DIM, generator=gen).to("npu", torch.bfloat16) * 0.3
    )
    g = -torch.rand(1, T, H, K_DIM, generator=gen).to("npu", torch.float32)
    beta = torch.rand(1, T, H, K_DIM, generator=gen).to("npu", torch.float32)
    cu = torch.tensor([0, T], dtype=torch.int32, device="npu")
    return q, k, v, g, beta, cu


@pytest.mark.parametrize("T", [771, 968, 1303])
def test_chunk_kda_is_deterministic(T):
    """Identical inputs must give bitwise-identical output and final state."""
    q, k, v, g, beta, cu = _kda_inputs(T)
    init = torch.zeros(1, H, K_DIM, V_DIM, dtype=torch.float32, device="npu")

    outs = []
    for _ in range(3):
        o, state = chunk_kda(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            scale=None,
            initial_state=init.clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu,
        )
        torch.npu.synchronize()
        outs.append((o.clone(), state.clone()))

    assert torch.equal(outs[0][0], outs[1][0]), "output differs between calls"
    assert torch.equal(outs[0][1], outs[1][1]), "final state differs between calls"
    assert torch.equal(outs[0][1], outs[2][1]), "final state differs on 3rd call"


def test_chunk_kda_survives_allocator_churn():
    """Results must not depend on what the caching allocator last served.

    This is the actual failure mode of ``torch.empty*`` buffers: a pool-sized
    tensor filled with garbage is allocated and freed between two identical
    calls, so any unwritten buffer region would read the garbage back.
    """
    T = 968  # not tile-aligned -> tail regions exist in every buffer
    q, k, v, g, beta, cu = _kda_inputs(T, seed=3)
    init = torch.zeros(1, H, K_DIM, V_DIM, dtype=torch.float32, device="npu")

    def run():
        o, state = chunk_kda(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            scale=None,
            initial_state=init.clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu,
        )
        torch.npu.synchronize()
        return o.clone(), state.clone()

    first_out, first_state = run()

    # Dirty the caching allocator with pool-sized garbage between calls.
    for _ in range(3):
        garbage = torch.full(
            (64, 64, 128, 128), 123.0, dtype=torch.bfloat16, device="npu"
        )
        del garbage

    second_out, second_state = run()
    assert torch.equal(first_out, second_out), (
        "output changed after allocator churn (uninitialized buffer read)"
    )
    assert torch.equal(first_state, second_state), (
        "final state changed after allocator churn (uninitialized buffer read)"
    )
