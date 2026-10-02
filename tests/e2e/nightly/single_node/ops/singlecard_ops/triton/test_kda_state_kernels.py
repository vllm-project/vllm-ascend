# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""Single-kernel NPU precision tests for the KDA state kernels.

Kernel-to-test mapping (each parameterized case launches only the named raw
Triton kernel and never calls a production wrapper):

* ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64_kda``
  -> ``test_chunk_gated_delta_rule_fwd_kernel_h_blockdim64_kda``
* ``fused_recurrent_gated_delta_rule_fwd_kernel``
  -> ``test_fused_recurrent_gated_delta_rule_fwd_kernel_kda``

All expected values are computed by independent CPU float32 recurrences from
the quantized kernel inputs.
"""

import math

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.kda.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64_kda,
)
from vllm_ascend.ops.triton.kda.fused_recurrent_kda import (
    fused_recurrent_gated_delta_rule_fwd_kernel,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

DEVICE = "npu"
CHUNK_SIZE = 64


@pytest.fixture(scope="module", autouse=True)
def _initialize_triton_device() -> None:
    init_device_properties_triton()


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    return 1e-2, 1e-2


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    actual = actual.detach().to(torch.float32).cpu()
    expected = expected.detach().to(torch.float32).cpu()
    assert torch.isfinite(actual).all(), f"{name}: the kernel produced a non-finite value"
    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=lambda msg: f"{name}: {msg}")


def _chunk_state_reference(
    k: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor,
    sequence_ranges: list[tuple[int, int]],
    chunk_offsets: list[int],
    output_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU FP32 reference for the chunk-state kernel's KDA path."""
    dtype = k.dtype
    _, _, num_k_heads, key_dim = k.shape
    num_heads = u.shape[2]
    value_dim = u.shape[-1]

    flat_k = k.reshape(-1, num_k_heads, key_dim)
    flat_u = u.reshape(-1, num_heads, value_dim)
    flat_w = w.reshape(-1, num_heads, key_dim)
    flat_gk = gk.reshape(-1, num_heads, key_dim)

    total_chunks = chunk_offsets[-1]
    states_at_chunk_start = torch.empty(total_chunks, num_heads, value_dim, key_dim, dtype=dtype)
    new_value = torch.empty_like(flat_u)
    final_state = torch.empty_like(initial_state, dtype=torch.float32)
    heads_per_k_head = num_heads // num_k_heads

    for sequence_index, (bos, eos) in enumerate(sequence_ranges):
        for head in range(num_heads):
            state = initial_state[sequence_index, head].to(torch.float32).clone()
            key_head = head // heads_per_k_head
            for local_chunk, chunk_bos in enumerate(range(bos, eos, CHUNK_SIZE)):
                chunk_eos = min(chunk_bos + CHUNK_SIZE, eos)
                global_chunk = chunk_offsets[sequence_index] + local_chunk
                states_at_chunk_start[global_chunk, head] = state.to(dtype)

                key = flat_k[chunk_bos:chunk_eos, key_head].to(torch.float32)
                value = flat_u[chunk_bos:chunk_eos, head].to(torch.float32)
                weight = flat_w[chunk_bos:chunk_eos, head].to(torch.float32)
                residual = value - torch.matmul(weight, state.transpose(0, 1))
                new_value[chunk_bos:chunk_eos, head] = residual.to(dtype)

                # KDA supplies a base-2 cumulative vector gate. The kernel
                # applies the final gate of each chunk before its state update.
                gate = torch.exp2(flat_gk[chunk_eos - 1, head].to(torch.float32))
                state = state * gate.unsqueeze(0)
                quantized_residual = residual.to(dtype).to(torch.float32)
                state = state + torch.matmul(quantized_residual.transpose(0, 1), key)
            final_state[sequence_index, head] = state

    return states_at_chunk_start.reshape(output_shape), new_value.reshape_as(u), final_state


def _fused_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_states: list[torch.Tensor],
    sequence_ranges: list[tuple[int, int]],
    scale: float,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """CPU FP32 KDA recurrence with per-token state snapshots."""
    dtype = q.dtype
    _, total_tokens, num_qk_heads, _ = q.shape
    num_value_heads = v.shape[2]
    value_dim = v.shape[-1]
    value_heads_per_qk_head = num_value_heads // num_qk_heads
    output = torch.empty(1, total_tokens, num_value_heads, value_dim, dtype=dtype)
    token_states = torch.empty(
        total_tokens,
        num_value_heads,
        value_dim,
        k.shape[-1],
        dtype=torch.float32,
    )
    final_states: list[torch.Tensor] = []

    for sequence_index, (bos, eos) in enumerate(sequence_ranges):
        state = initial_states[sequence_index].to(torch.float32).clone()
        for token in range(bos, eos):
            for value_head in range(num_value_heads):
                qk_head = value_head // value_heads_per_qk_head
                query = q[0, token, qk_head].to(torch.float32)
                key = k[0, token, qk_head].to(torch.float32)
                if use_qk_l2norm:
                    query = query / torch.sqrt(torch.sum(query * query) + 1e-6)
                    key = key / torch.sqrt(torch.sum(key * key) + 1e-6)

                state_head = state[value_head]
                gate = torch.exp(g[0, token, value_head].to(torch.float32))
                state_head = state_head * gate.unsqueeze(0)
                residual = v[0, token, value_head].to(torch.float32) - torch.mv(state_head, key)
                residual = residual * beta[0, token, value_head].to(torch.float32)
                state_head = state_head + residual.unsqueeze(1) * key.unsqueeze(0)
                state[value_head] = state_head
                output[0, token, value_head] = torch.mv(state_head, query * scale).to(dtype)
            token_states[token] = state
        final_states.append(state.clone())

    return output, token_states, final_states


@pytest.mark.parametrize(
    ("dtype", "sequence_lengths"),
    [
        pytest.param(torch.float16, None, id="fixed-batch-fp16"),
        pytest.param(torch.bfloat16, [5, 70], id="varlen-bf16"),
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_chunk_gated_delta_rule_fwd_kernel_h_blockdim64_kda(
    dtype: torch.dtype,
    sequence_lengths: list[int] | None,
) -> None:
    """Directly test fixed-batch and variable-length KDA state layouts."""
    generator = torch.Generator().manual_seed(20260825)
    num_heads, num_k_heads = 2, 1
    key_dim, value_dim = 96, 80

    if sequence_lengths is None:
        batch_size, sequence_length = 2, 70
        num_sequences = batch_size
        sequence_ranges = [(batch * sequence_length, (batch + 1) * sequence_length) for batch in range(batch_size)]
        chunk_offsets = [batch * math.ceil(sequence_length / CHUNK_SIZE) for batch in range(batch_size + 1)]
        cu_seqlens_cpu = None
        chunk_offsets_cpu = None
        output_shape = (
            batch_size,
            math.ceil(sequence_length / CHUNK_SIZE),
            num_heads,
            value_dim,
            key_dim,
        )
    else:
        batch_size, sequence_length = 1, sum(sequence_lengths)
        num_sequences = len(sequence_lengths)
        cumulative = [0]
        chunk_offsets = [0]
        for length in sequence_lengths:
            cumulative.append(cumulative[-1] + length)
            chunk_offsets.append(chunk_offsets[-1] + math.ceil(length / CHUNK_SIZE))
        sequence_ranges = list(zip(cumulative[:-1], cumulative[1:]))
        cu_seqlens_cpu = torch.tensor(cumulative, dtype=torch.int64)
        chunk_offsets_cpu = torch.tensor(chunk_offsets, dtype=torch.int64)
        output_shape = (batch_size, chunk_offsets[-1], num_heads, value_dim, key_dim)

    k_cpu = (torch.randn(batch_size, sequence_length, num_k_heads, key_dim, generator=generator) * 0.15).to(dtype)
    u_cpu = (torch.randn(batch_size, sequence_length, num_heads, value_dim, generator=generator) * 0.15).to(dtype)
    w_cpu = (torch.randn(batch_size, sequence_length, num_heads, key_dim, generator=generator) * 0.03).to(dtype)
    gk_cpu = (-torch.rand(batch_size, sequence_length, num_heads, key_dim, generator=generator) * 0.02).to(dtype)
    h0_cpu = torch.randn(num_sequences, num_heads, value_dim, key_dim, generator=generator) * 0.03

    expected_h, expected_v_new, expected_ht = _chunk_state_reference(
        k_cpu,
        u_cpu,
        w_cpu,
        gk_cpu,
        h0_cpu,
        sequence_ranges,
        chunk_offsets,
        output_shape,
    )

    k = k_cpu.to(DEVICE)
    u = u_cpu.to(DEVICE)
    w = w_cpu.to(DEVICE)
    gk = gk_cpu.to(DEVICE)
    h0 = h0_cpu.to(DEVICE)
    h = torch.empty(output_shape, dtype=dtype, device=DEVICE)
    v_new = torch.empty_like(u)
    ht = torch.empty_like(h0, dtype=torch.float32)
    cu_seqlens = None if cu_seqlens_cpu is None else cu_seqlens_cpu.to(DEVICE)
    chunk_offsets_tensor = None if chunk_offsets_cpu is None else chunk_offsets_cpu.to(DEVICE)

    grid = (math.ceil(value_dim / 64), num_sequences * num_heads)
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64_kda[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=None,
        gk=gk,
        h=h,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets_tensor,
        T=sequence_length,
        H=num_heads,
        Hg=num_k_heads,
        K=key_dim,
        V=value_dim,
        BT=CHUNK_SIZE,
    )

    assert h.shape == expected_h.shape
    assert v_new.shape == expected_v_new.shape
    assert ht.shape == expected_ht.shape
    _assert_close("chunk-start states", h, expected_h, dtype)
    _assert_close("new values", v_new, expected_v_new, dtype)
    _assert_close("final states", ht, expected_ht, dtype)


@pytest.mark.parametrize(
    ("dtype", "continuous_batching"),
    [
        pytest.param(torch.float16, False, id="varlen-noninplace-scalar-beta-l2norm-fp16"),
        pytest.param(torch.bfloat16, True, id="continuous-inplace-vector-beta-bf16"),
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_fused_recurrent_gated_delta_rule_fwd_kernel_kda(
    dtype: torch.dtype,
    continuous_batching: bool,
) -> None:
    """Directly test KDA recurrence, state snapshots, and continuous slots."""
    generator = torch.Generator().manual_seed(20260826)
    sequence_lengths = [2, 3]
    cumulative = [0]
    for length in sequence_lengths:
        cumulative.append(cumulative[-1] + length)
    sequence_ranges = list(zip(cumulative[:-1], cumulative[1:]))

    batch_size, total_tokens = 1, cumulative[-1]
    num_sequences = len(sequence_lengths)
    num_qk_heads, num_value_heads = 2, 4
    # The production kernel loads the KDA gate in full BK vectors without a
    # tail mask, so use the smallest supported aligned key dimension.
    key_dim, value_dim = 64, 20
    scale = key_dim**-0.5
    use_qk_l2norm = not continuous_batching

    q_cpu = (torch.randn(batch_size, total_tokens, num_qk_heads, key_dim, generator=generator) * 0.2).to(dtype)
    k_cpu = (torch.randn(batch_size, total_tokens, num_qk_heads, key_dim, generator=generator) * 0.2).to(dtype)
    v_cpu = (torch.randn(batch_size, total_tokens, num_value_heads, value_dim, generator=generator) * 0.2).to(dtype)
    g_cpu = (-torch.rand(batch_size, total_tokens, num_value_heads, key_dim, generator=generator) * 0.03).to(dtype)
    if continuous_batching:
        beta_cpu = (
            0.2
            + 0.6
            * torch.rand(
                batch_size,
                total_tokens,
                num_value_heads,
                value_dim,
                generator=generator,
            )
        ).to(dtype)
        state_cpu = torch.randn(num_sequences + 1, num_value_heads, value_dim, key_dim, generator=generator) * 0.03
        state_cpu[0].zero_()
        state_indices_cpu = torch.tensor([[1, 1, 0], [2, 2, 2]], dtype=torch.int64)
        initial_states = [state_cpu[1], state_cpu[2]]
    else:
        beta_cpu = (
            0.2
            + 0.6
            * torch.rand(
                batch_size,
                total_tokens,
                num_value_heads,
                generator=generator,
            )
        ).to(dtype)
        state_cpu = torch.randn(total_tokens, num_value_heads, value_dim, key_dim, generator=generator) * 0.03
        state_indices_cpu = None
        initial_states = [state_cpu[bos] for bos, _ in sequence_ranges]

    expected_o, expected_token_states, expected_final_states = _fused_recurrent_reference(
        q_cpu,
        k_cpu,
        v_cpu,
        g_cpu,
        beta_cpu,
        initial_states,
        sequence_ranges,
        scale,
        use_qk_l2norm,
    )

    q = q_cpu.to(DEVICE)
    k = k_cpu.to(DEVICE)
    v = v_cpu.to(DEVICE)
    g = g_cpu.to(DEVICE)
    beta = beta_cpu.to(DEVICE)
    state = state_cpu.to(DEVICE)
    o = torch.empty(batch_size, total_tokens, num_value_heads, value_dim, dtype=dtype, device=DEVICE)
    cu_seqlens = torch.tensor(cumulative, dtype=torch.int64, device=DEVICE)
    state_indices = None if state_indices_cpu is None else state_indices_cpu.to(DEVICE)
    if continuous_batching:
        final_state = state
        stride_indices_seq, stride_indices_tok = state_indices.stride()
    else:
        final_state = torch.empty(total_tokens, num_value_heads, value_dim, key_dim, dtype=torch.float32, device=DEVICE)
        stride_indices_seq, stride_indices_tok = 1, 1

    block_k = _next_power_of_2(key_dim)
    block_v = min(_next_power_of_2(value_dim), 8)
    grid = (math.ceil(key_dim / block_k), math.ceil(value_dim / block_v), num_sequences * num_value_heads)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=None,
        scale=scale,
        N=num_sequences,
        T=total_tokens,
        B=batch_size,
        H=num_qk_heads,
        HV=num_value_heads,
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        stride_init_state_token=state.stride(0),
        stride_final_state_token=final_state.stride(0),
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        INPLACE_FINAL_STATE=continuous_batching,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm,
        IS_KDA=True,
        num_warps=1,
        num_stages=3,
    )

    assert o.shape == expected_o.shape
    _assert_close("recurrent output", o, expected_o, dtype)
    if continuous_batching:
        expected_state = state_cpu.clone()
        expected_state[1] = expected_final_states[0]
        expected_state[2] = expected_final_states[1]
        _assert_close("in-place final states", final_state, expected_state, dtype)
        torch.testing.assert_close(final_state[0].cpu(), torch.zeros_like(final_state[0].cpu()), rtol=0, atol=0)
    else:
        _assert_close("per-token states", final_state, expected_token_states, dtype)
