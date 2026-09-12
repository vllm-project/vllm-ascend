# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.models.glm5next.ops.kda as kda
import vllm_ascend.ops.kda as kda_ops


@pytest.mark.parametrize("accepted", [None, [1, 2, 1]])
@pytest.mark.parametrize("qkv_padding", [0, 64])
def test_recurrent_raw_gates_rollback_slots_and_padding(monkeypatch, accepted, qkv_padding):
    q, k, v = (torch.ones(1, 4, 1, 128 + qkv_padding * i, dtype=torch.bfloat16)[..., :128] for i in (1, 2, 3))
    gate = q * 2
    beta = torch.zeros(1, 4, 1, dtype=torch.bfloat16)
    state = torch.zeros(8, 1, 128, 128)
    starts = torch.tensor([0, 1, 3], dtype=torch.int32)
    slots = torch.tensor([[2, 3], [5, 6], [0, 0]], dtype=torch.int32)
    accepted_tensor = None if accepted is None else torch.tensor(accepted, dtype=torch.int32)

    def recurrent(q_arg, k_arg, v_arg, gate_arg, beta_arg, state_arg, cu, ids, a_log, bias, **kwargs):
        # Preserve each view's strides and storage without materializing Q/K/V.
        assert q_arg is q and k_arg is k and v_arg is v
        assert kwargs["use_gate_in_kernel"] and kwargs["use_beta_sigmoid_in_kernel"]
        assert kwargs["safe_gate"] and kwargs["lower_bound"] == -4.0
        assert state_arg is state
        torch.testing.assert_close(gate_arg, gate)
        torch.testing.assert_close(beta_arg, beta)
        torch.testing.assert_close(ids, slots[:2])
        if accepted_tensor is not None:
            torch.testing.assert_close(kwargs["num_accepted_tokens"], accepted_tensor[:2])
        result = q_arg.clone()
        result[:, 3:] = float("nan")
        return result

    monkeypatch.setattr(torch.ops._C_ascend, "recurrent_kda", recurrent, raising=False)
    out = kda.recurrent_kda(
        q, k, v, gate, beta, state, starts, slots, torch.zeros(1), torch.zeros(128), -4, accepted_tensor
    )
    torch.testing.assert_close(out[:, :3], q[:, :3])
    assert torch.count_nonzero(out[:, 3:]) == 0


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compact", [False, True])
def test_chunk_uses_host_descriptors_and_preserves_vk_cache(monkeypatch, state_dtype, compact):
    q = torch.ones(1, 3, 1, 128, dtype=torch.bfloat16)
    state = torch.arange(4 * 128 * 128, dtype=torch.float32).reshape(4, 1, 128, 128).to(state_dtype)
    saved = state.clone()
    indices = torch.tensor([3, 1], dtype=torch.int32)
    has_initial = torch.tensor([True, False])
    keep = torch.tensor([0]) if compact else None
    metadata = SimpleNamespace(
        keep_meta=keep,
        cu_seqlens_host=torch.tensor([0, 3] if compact else [0, 1, 3]),
        cu_seqlens_kern=None,
        chunk_indices_chunk64_host=torch.tensor([[0, 0]] if compact else [[0, 0], [1, 0]]),
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

    monkeypatch.setattr(torch.ops._C_ascend, "chunk_kda_fwd", chunk, raising=False)
    monkeypatch.setattr(kda_ops, "l2norm_fwd", lambda x: x)
    out = kda.chunk_kda(
        q, q, q, q, torch.zeros(1, 3, 1), state, indices, has_initial, metadata, torch.zeros(1), torch.zeros(128), -4
    )
    torch.testing.assert_close(out, q)
    assert (state[3] == 17).all()
    torch.testing.assert_close(state[0], saved[0])
    torch.testing.assert_close(state[2], saved[2])
    if compact:
        torch.testing.assert_close(state[1], saved[1])
    else:
        assert (state[1] == 17).all()
