# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify and commit must agree with independent, multi-step KDA recurrence."""

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.ops.triton.kda.recoverssm import (  # noqa: E402
    kda_recoverssm_commit,
    kda_recoverssm_verify,
)


def reference(q, k, v, raw_g, raw_beta, a_log, bias, lower_bound, state, starts, blocks, accepted):
    output = torch.zeros_like(v, dtype=torch.float32)
    final = state.clone()
    heads, value_dim, key_dim = state.shape[1:]
    for row, block in enumerate(blocks):
        if block == 0:
            continue
        current = state[block].float().clone()
        for position, token in enumerate(range(starts[row], starts[row + 1])):
            key = k[0, token].float()
            key_norm = key.square().sum(-1, keepdim=True)
            key = key * torch.rsqrt(torch.where(key_norm > 0, key_norm, 1.0))
            query = q[0, token].float()
            query_norm = query.square().sum(-1, keepdim=True)
            query = query * torch.rsqrt(torch.where(query_norm > 0, query_norm, 1.0)) * key_dim**-0.5
            gate_input = raw_g[0, token].float() + bias.view(heads, key_dim)
            a = a_log.view(heads, 1).exp()
            if lower_bound is None:
                gate = -a * torch.nn.functional.softplus(gate_input)
            else:
                gate = lower_bound * torch.sigmoid(a * gate_input)
            current = current * gate.exp()[:, None, :]
            correction = raw_beta[0, token].float().sigmoid()[:, None] * (
                v[0, token].float() - (current * key[:, None, :]).sum(-1)
            )
            current = current + correction[:, :, None] * key[:, None, :]
            output[0, token] = (current * query[:, None, :]).sum(-1)
            if position + 1 == accepted[row]:
                final[block] = current
    return output, final


@pytest.mark.parametrize("num_spec", [0, 1, 3, 5, 7])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_recoverssm_preserves_accepted_state_across_steps(num_spec, state_dtype, activation_dtype, lower_bound):
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(42)
    device = "npu"
    heads, dim, capacity = 4, 128, num_spec + 1
    # Permuted slots, rejected-all row, short rows, padding and an empty row.
    blocks = [3, 1, 5, 2, 0, 4]
    lengths = [capacity, 1, capacity - 1, capacity, 1, 0]
    accepted = [0, 1, min(3, capacity - 1), capacity, 1, 0]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    indices = torch.tensor(blocks, dtype=torch.int32, device=device)
    query_starts = torch.tensor(starts, dtype=torch.int64, device=device)
    counts = torch.tensor(accepted, dtype=torch.int32, device=device)
    state = (torch.randn(8, heads, dim, dim, device=device) * 0.05).to(state_dtype)
    correction = torch.full((8, heads, capacity, dim), float("nan"), device=device)
    kg = torch.full((8, heads, capacity, 2 * dim), float("nan"), dtype=activation_dtype, device=device)
    a_log = (0.2 * torch.randn(heads, device=device)).view(1, 1, heads, 1)
    bias = 0.1 * torch.randn(heads * dim, device=device)

    for _ in range(2):
        # Token-strided inputs exercise projected/split tensor layouts.
        projections = torch.randn(1, starts[-1], heads * dim * 4, dtype=activation_dtype, device=device)
        q, k, v, raw_g = (x.view(1, starts[-1], heads, dim) for x in projections.split(heads * dim, dim=-1))
        beta = torch.randn(1, starts[-1], heads, dtype=activation_dtype, device=device)
        initial = state.clone()
        expected_out, expected_state = reference(
            q, k, v, raw_g, beta, a_log, bias, lower_bound, initial, starts, blocks, accepted
        )
        output_buffer = torch.empty_like(v)
        output = kda_recoverssm_verify(
            q,
            k,
            v,
            raw_g,
            beta,
            a_log,
            bias,
            lower_bound,
            state,
            correction,
            kg,
            query_starts,
            indices,
            capacity,
            out=output_buffer,
        )
        assert output is output_buffer
        torch.testing.assert_close(state, initial, atol=0, rtol=0)
        torch.testing.assert_close(output.float(), expected_out, atol=2e-2, rtol=2e-2)
        # Rejected suffix records must never influence the committed state.
        for row, block in enumerate(blocks):
            if block > 0:
                correction[block, :, accepted[row] :] = float("nan")
                kg[block, :, accepted[row] :] = float("nan")
        kda_recoverssm_commit(state, correction, kg, a_log, bias, lower_bound, query_starts, indices, counts)
        tol = 2e-3 if state_dtype == torch.float32 else 2e-2
        torch.testing.assert_close(state.float(), expected_state.float(), atol=tol, rtol=tol)
        # Includes block zero and unused slots as well as zero-acceptance rows.
        torch.testing.assert_close(state[[0, 3, 4, 6, 7]], initial[[0, 3, 4, 6, 7]], atol=0, rtol=0)


@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_recoverssm_empty_batch():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    device, heads, dim, capacity = "npu", 2, 128, 5
    q = torch.empty(1, 0, heads, dim, dtype=torch.bfloat16, device=device)
    beta = torch.empty(1, 0, heads, dtype=q.dtype, device=device)
    state = torch.zeros(1, heads, dim, dim, device=device)
    correction = torch.empty(1, heads, capacity, dim, device=device)
    kg = torch.empty(1, heads, capacity, 2 * dim, dtype=q.dtype, device=device)
    a_log, bias = torch.zeros(heads, device=device), torch.zeros(heads * dim, device=device)
    starts = torch.zeros(1, dtype=torch.int32, device=device)
    indices = torch.empty(0, dtype=torch.int32, device=device)
    output = kda_recoverssm_verify(
        q, q, q, q, beta, a_log, bias, -5.0, state, correction, kg, starts, indices, capacity
    )
    assert output.shape == q.shape
    kda_recoverssm_commit(state, correction, kg, a_log, bias, -5.0, starts, indices, indices)
    assert torch.count_nonzero(state) == 0


@pytest.mark.parametrize("amplitude", [1.0, 1e-2, 1e-3, 1e-6, 1e-8, 0.0])
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_recoverssm_matches_ascendc_small_norms(amplitude):
    from vllm_ascend.ops.kda import run_recurrent_kda

    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(61566)
    device, heads, dim, capacity = "npu", 8, 128, 6
    q, k = (torch.randn(1, capacity, heads, dim, dtype=torch.bfloat16, device=device) * amplitude for _ in range(2))
    v, gate = (torch.randn_like(q) for _ in range(2))
    beta = torch.randn(1, capacity, heads, dtype=q.dtype, device=device)
    a_log, bias = torch.zeros(heads, device=device), torch.zeros(heads * dim, device=device)
    initial = torch.randn(heads, dim, dim, device=device) * 0.05
    native = torch.zeros(capacity + 1, heads, dim, dim, device=device)
    native[1] = initial
    checkpoint = torch.zeros(2, heads, dim, dim, device=device)
    checkpoint[1] = initial
    starts = torch.tensor([0, capacity], dtype=torch.int32, device=device)
    indices = torch.ones(1, dtype=torch.int32, device=device)
    native_indices = torch.arange(1, capacity + 1, dtype=torch.int32, device=device).view(1, capacity)
    expected = run_recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        native,
        starts,
        native_indices,
        a_log,
        bias,
        lower_bound=-5.0,
        beta_is_preprocessed=False,
        num_accepted_tokens=indices,
    )
    correction = torch.empty(2, heads, capacity, dim, device=device)
    kg = torch.empty(2, heads, capacity, 2 * dim, dtype=q.dtype, device=device)
    output = kda_recoverssm_verify(
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        bias,
        -5.0,
        checkpoint,
        correction,
        kg,
        starts,
        indices,
        capacity,
    )
    torch.testing.assert_close(output.float(), expected.float(), atol=1e-4, rtol=2e-4)
    for count in (1, 3, capacity):
        restored = checkpoint.clone()
        accepted = torch.full_like(indices, count)
        kda_recoverssm_commit(restored, correction, kg, a_log, bias, -5.0, starts, indices, accepted)
        torch.testing.assert_close(restored[1], native[count], atol=2e-6, rtol=2e-5)
