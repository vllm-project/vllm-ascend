# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.models.glm5next.ops.kda as kda
import vllm_ascend.ops.kda as kda_ops


@pytest.mark.parametrize(
    "accepted,qkv_padding,direct_output",
    [(None, 0, False), (None, 64, False), ([1, 2, 1], 0, False), ([1, 2, 1], 64, False), (None, 64, True)],
)
def test_recurrent_raw_gates_rollback_slots_and_padding(monkeypatch, accepted, qkv_padding, direct_output):
    q, k, v = (torch.ones(1, 4, 1, 128 + qkv_padding * i, dtype=torch.bfloat16)[..., :128] for i in (1, 2, 3))
    gate = q * 2
    beta = torch.zeros(1, 4, 1, dtype=torch.bfloat16)
    state = torch.zeros(8, 1, 128, 128)
    starts = torch.tensor([0, 1, 3], dtype=torch.int32)
    slots = torch.tensor([[2, 3], [5, 6], [0, 0]], dtype=torch.int32)
    accepted_tensor = None if accepted is None else torch.tensor(accepted, dtype=torch.int32)

    def recurrent(q_arg, k_arg, v_arg, gate_arg, beta_arg, state_arg, *, cu_seqlens, ssm_state_indices, **kwargs):
        # Preserve each view's strides and storage without materializing Q/K/V.
        assert q_arg is q and k_arg is k and v_arg is v
        assert kwargs["use_gate_in_kernel"] and kwargs["use_beta_sigmoid_in_kernel"]
        assert kwargs["safe_gate"] and kwargs["lower_bound"] == -4.0
        assert state_arg is state
        torch.testing.assert_close(gate_arg, gate)
        torch.testing.assert_close(beta_arg, beta)
        assert cu_seqlens is starts
        torch.testing.assert_close(ssm_state_indices, slots[:2])
        assert kwargs["state_v_first"] and kwargs["inplace_final_state"]
        assert not kwargs["output_final_state"]
        if accepted_tensor is not None:
            torch.testing.assert_close(kwargs["num_accepted_tokens"], accepted_tensor[:2])
        result = q_arg.clone()
        result[:, 3:] = float("nan")
        return result, None

    monkeypatch.setattr(kda_ops, "recurrent_kda", recurrent)
    destination = torch.full((1, 8, 1, 128), float("nan"), dtype=q.dtype) if direct_output else None

    def writeback(source, target, ends):
        assert target is destination and ends is starts
        assert torch.isnan(source[:, 3:]).all()
        target.zero_()
        target[:, :3].copy_(source[:, :3])

    monkeypatch.setattr(kda, "write_recurrent_output", writeback)
    out = kda.recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        state,
        starts,
        slots,
        torch.zeros(1),
        torch.zeros(128),
        -4,
        accepted_tensor,
        output_buffer=destination,
    )
    if direct_output:
        assert out is destination
    torch.testing.assert_close(out[:, :3], q[:, :3])
    assert torch.count_nonzero(out[:, 3:]) == 0


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compact", [False, True])
def test_chunk_uses_host_descriptors_and_preserves_vk_cache(monkeypatch, state_dtype, compact):
    q = torch.ones(1, 128, 1, 128, dtype=torch.bfloat16)
    state = torch.arange(4 * 128 * 128, dtype=torch.float32).reshape(4, 1, 128, 128).to(state_dtype)
    saved = state.clone()
    indices = torch.tensor([3, 1], dtype=torch.int32)
    has_initial = torch.tensor([True, False])
    keep = torch.tensor([0]) if compact else None
    metadata = SimpleNamespace(
        keep_meta=keep,
        cu_seqlens_host=(0, 128) if compact else (0, 64, 128),
        cu_seqlens_kern=None,
        chunk_indices_chunk64_host=(0, 0, 0, 1) if compact else (0, 0, 1, 0),
    )

    def chunk(q_arg, k_arg, v, g, beta, scale, chunk_size, **kwargs):
        assert kwargs["state_v_first"] and kwargs["use_gate_in_kernel"] and kwargs["safe_gate"]
        assert kwargs["cu_seqlens"] is metadata.cu_seqlens_host
        assert kwargs["chunk_indices"] is metadata.chunk_indices_chunk64_host
        assert kwargs["initial_state"].dtype == torch.float32
        torch.testing.assert_close(kwargs["initial_state"][0], saved[3].float())
        if not compact:
            assert torch.count_nonzero(kwargs["initial_state"][1]) == 0
        torch.testing.assert_close(beta, torch.full_like(beta, 0.5))
        return v, torch.full_like(kwargs["initial_state"], 17)

    monkeypatch.setattr(kda_ops, "chunk_kda_fwd", chunk)
    monkeypatch.setattr(kda_ops, "l2norm_fwd", lambda x: x)
    out = kda.chunk_kda(
        q, q, q, q, torch.zeros(1, 128, 1), state, indices, has_initial, metadata, torch.zeros(1), torch.zeros(128), -4
    )
    torch.testing.assert_close(out, q)
    assert (state[3] == 17).all()
    torch.testing.assert_close(state[0], saved[0])
    torch.testing.assert_close(state[2], saved[2])
    if compact:
        torch.testing.assert_close(state[1], saved[1])
    else:
        assert (state[1] == 17).all()


@pytest.mark.parametrize("compact", [False, True])
def test_chunk_pads_each_sequence_with_neutral_state_updates(monkeypatch, compact):
    q = torch.arange(68, dtype=torch.float32).reshape(1, 68, 1, 1).expand(-1, -1, -1, 128).bfloat16()
    beta = torch.zeros(1, 68, 1)
    state = torch.zeros(5, 1, 128, 128)
    state[4] = 7
    indices = torch.tensor([4, 1, 3], dtype=torch.int32)
    has_initial = torch.tensor([True, False, False])
    metadata = SimpleNamespace(
        keep_meta=torch.tensor([0, 2]) if compact else None,
        cu_seqlens_host=(0, 1, 65, 68),
        cu_seqlens_kern=(0, 1, 4) if compact else None,
        chunk_indices_chunk64_host=(0, 0, 1, 0) if compact else (0, 0, 1, 0, 2, 0),
    )
    if compact:
        q, beta = q[:, :4], beta[:, :4]
    lengths = (1, 3) if compact else (1, 64, 3)

    def chunk(q_arg, k_arg, v_arg, gate, beta_arg, scale, chunk_size, **kwargs):
        assert chunk_size == 64
        assert kwargs["cu_seqlens"] == list(range(0, 64 * (len(lengths) + 1), 64))
        assert kwargs["chunk_indices"] == [index for sequence in range(len(lengths)) for index in (sequence, 0)]
        logical_start = 0
        for sequence, length in enumerate(lengths):
            start = sequence * 64
            for tensor in (q_arg, k_arg, v_arg):
                torch.testing.assert_close(
                    tensor[:, start : start + length], q[:, logical_start : logical_start + length]
                )
                assert torch.count_nonzero(tensor[:, start + length : start + 64]) == 0
            torch.testing.assert_close(beta_arg[:, start : start + length], torch.full((1, length, 1), 0.5))
            assert torch.count_nonzero(beta_arg[:, start + length : start + 64]) == 0
            assert torch.isneginf(gate[:, start + length : start + 64]).all()
            logical_start += length
        torch.testing.assert_close(kwargs["initial_state"][0], state[4])
        assert torch.count_nonzero(kwargs["initial_state"][1:]) == 0
        final = torch.arange(len(lengths), dtype=torch.float32)[:, None, None, None].expand(-1, 1, 128, 128) + 10
        return v_arg, final

    monkeypatch.setattr(kda_ops, "chunk_kda_fwd", chunk)
    monkeypatch.setattr(kda_ops, "l2norm_fwd", lambda x: x)
    out = kda.chunk_kda(q, q, q, q, beta, state, indices, has_initial, metadata, torch.zeros(1), torch.zeros(128), -4)
    torch.testing.assert_close(out, q)
    assert (state[4] == 10).all()
    assert (state[3] == (11 if compact else 12)).all()
    assert (state[1] == (0 if compact else 11)).all()
    assert torch.count_nonzero(state[[0, 2]]) == 0
