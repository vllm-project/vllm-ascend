# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify/commit correctness, including grouped heads and paged cache strides."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

import vllm_ascend.ops.triton.gdn_recoverssm as recovery  # noqa: E402


def paged(shape, *, dtype=torch.float32):
    elements = 1
    strides = []
    for size in reversed(shape[1:]):
        strides.append(elements)
        elements *= size
    backing = torch.zeros(shape[0] * (elements + 128), dtype=dtype, device="npu")
    return backing.as_strided(shape, (elements + 128, *reversed(strides)))


def reference(q, k, v, g, beta, initial, starts, indices):
    out = torch.zeros_like(v, dtype=torch.float32)
    states = []
    ratio = v.shape[2] // k.shape[2]
    for row, block in enumerate(indices):
        state = initial[block].float().clone()
        history = [state.clone()]
        for token in range(starts[row], starts[row + 1]):
            key = k[0, token].float().repeat_interleave(ratio, 0)
            query = q[0, token].float().repeat_interleave(ratio, 0)
            state = state * g[0, token].float().exp()[:, None, None]
            residual = (v[0, token].float() - torch.einsum("hvk,hk->hv", state, key)) * beta[0, token].float()[:, None]
            state = state + residual[:, :, None] * key[:, None, :]
            out[0, token] = torch.einsum("hvk,hk->hv", state, query) * (key.shape[-1] ** -0.5)
            history.append(state.clone())
        states.append(history)
    return out, states


@pytest.mark.parametrize("accepted", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("hk,hv,dk,dv", [(2, 4, 128, 128), (1, 2, 33, 47)])
def test_verify_and_multilayer_commit_preserve_accepted_prefix(monkeypatch, accepted, dim_first, hk, hv, dk, dv):
    torch.manual_seed(73)
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: dim_first)
    blocks, window, history = 5, 5, 3
    channels = 2 * hk * dk + hv * dv
    layers = []
    for _ in range(2):
        conv = paged((blocks, channels, history + window - 1), dtype=torch.bfloat16)
        conv.copy_(torch.randn_like(conv))
        state = paged((blocks, hv, dv, dk))
        state.copy_(torch.randn_like(state) * 0.1)
        records = (paged((blocks, hv, window, dv)), paged((blocks, hk, window, dk)), paged((blocks, hv, window)))
        layers.append(SimpleNamespace(kv_cache=(conv if dim_first else conv.transpose(-1, -2), state, *records)))
    context = recovery.GDNRecoverSSMCommitContext.create(layers, spec_query_len=window, max_num_reqs=4)
    # Reorder physical pages between rounds; sampler rows are in a different order.
    for block_ids in ([3, 1, 0], [1, 3, 0]):
        indices = torch.tensor(block_ids, dtype=torch.int32, device="npu")
        starts = torch.tensor([0, 5, 8, 8], dtype=torch.int32, device="npu")
        counts = [accepted, min(accepted, 3)]
        expected_states, expected_convs = [], []
        for layer, page in zip(layers, context.pages):
            conv, state, corrections, keys, gates = page
            initial, old_conv = state.clone(), conv.clone()
            # Poison previous-round records: rejected/padded entries must not leak.
            for record in (corrections, keys, gates):
                record.fill_(float("nan"))
            q = torch.nn.functional.normalize(torch.randn(1, 8, hk, dk, device="npu"), dim=-1).to(torch.bfloat16)
            k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).to(torch.bfloat16)
            v = torch.randn(1, 8, hv, dv, device="npu", dtype=torch.bfloat16)
            g = -torch.rand(1, 8, hv, device="npu")
            beta = torch.rand(1, 8, hv, device="npu", dtype=torch.bfloat16)
            expected, histories = reference(q, k, v, g, beta, initial, [0, 5, 8, 8], block_ids)
            observed = recovery.gdn_recoverssm_verify(
                q, k, v, g, beta, state, corrections, keys, gates, starts, indices, window
            )
            torch.testing.assert_close(observed.float(), expected, atol=3e-3, rtol=2e-2)
            torch.testing.assert_close(state, initial, atol=0, rtol=0)
            final = initial.clone()
            conv_final = old_conv.clone()
            for row, count in enumerate(counts):
                block = block_ids[row]
                final[block] = histories[row][count]
                if count:
                    conv_final[block, :, :history] = old_conv[block, :, count - 1 : count - 1 + history]
            expected_states.append(final)
            expected_convs.append(conv_final)
        sampled = torch.tensor([counts[1], 99, counts[0]], dtype=torch.int32, device="npu")
        mapping = torch.tensor([2, 0], dtype=torch.int32, device="npu")
        context.commit(sampled, indices[:2], starts[:3], request_indices=mapping)
        for page, expected_state, expected_conv in zip(context.pages, expected_states, expected_convs):
            torch.testing.assert_close(page[1], expected_state, atol=3e-5, rtol=3e-4)
            torch.testing.assert_close(page[0], expected_conv, atol=0, rtol=0)


def test_empty_verification_does_not_launch_or_mutate():
    q = torch.empty(1, 0, 1, 128, dtype=torch.bfloat16, device="npu")
    v = torch.empty(1, 0, 2, 128, dtype=torch.bfloat16, device="npu")
    g = torch.empty(1, 0, 2, device="npu")
    state = torch.randn(3, 2, 128, 128, device="npu")
    before = state.clone()
    out = recovery.gdn_recoverssm_verify(
        q,
        q,
        v,
        g,
        g,
        state,
        torch.empty(3, 2, 5, 128, device="npu"),
        torch.empty(3, 1, 5, 128, device="npu"),
        torch.empty(3, 2, 5, device="npu"),
        torch.zeros(1, dtype=torch.int32, device="npu"),
        torch.empty(0, dtype=torch.int32, device="npu"),
        5,
    )
    assert out.shape == v.shape
    torch.testing.assert_close(state, before, atol=0, rtol=0)


def test_nonempty_capture_rows_with_null_pages_produce_zero_outputs():
    q = torch.randn(1, 3, 1, 128, dtype=torch.bfloat16, device="npu")
    v = torch.randn(1, 3, 2, 128, dtype=torch.bfloat16, device="npu")
    g = torch.zeros(1, 3, 2, device="npu")
    state = torch.randn(3, 2, 128, 128, device="npu")
    before = state.clone()
    out = recovery.gdn_recoverssm_verify(
        q,
        q,
        v,
        g,
        g,
        state,
        torch.empty(3, 2, 5, 128, device="npu"),
        torch.empty(3, 1, 5, 128, device="npu"),
        torch.empty(3, 2, 5, device="npu"),
        torch.tensor([0, 3], dtype=torch.int32, device="npu"),
        torch.zeros(1, dtype=torch.int32, device="npu"),
        5,
    )
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
    torch.testing.assert_close(state, before, atol=0, rtol=0)
