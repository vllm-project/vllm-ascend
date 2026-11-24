# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops import mega_kda


def test_mega_kda_reads_device_metadata_and_merges_token_order(monkeypatch):
    dim = mega_kda.MEGA_KDA_HEAD_DIM
    qkv = torch.randn(4, 3 * dim, dtype=torch.bfloat16)
    raw_gate = torch.randn(1, 4, 1, dim, dtype=torch.bfloat16)
    gate = torch.randn(4, 1, dim, dtype=torch.bfloat16)
    beta = torch.randn(1, 4, 1, dtype=torch.bfloat16)
    result = torch.empty_like(raw_gate)
    spec_indices = torch.tensor([1, 3])
    prefill_indices = torch.tensor([0, 2])
    accepted = torch.tensor([2], dtype=torch.int32)
    spec_slots = torch.tensor([[7, 8]], dtype=torch.int32)
    prefill_slots = torch.tensor([11], dtype=torch.int32)
    no_initial = torch.tensor([False])
    prefix = torch.tensor([0, 2], dtype=torch.int32)
    norm = MagicMock()
    layer = SimpleNamespace(local_num_heads=1, o_norm=SimpleNamespace(forward_oot=norm))
    operator = MagicMock()
    packed_weights = torch.empty(4, 3 * dim, dtype=torch.bfloat16)
    spec_output = torch.ones(2, dim, dtype=torch.bfloat16)
    prefill_output = torch.full_like(spec_output, 2)
    launch = MagicMock(side_effect=[spec_output, prefill_output])
    monkeypatch.setattr(mega_kda, "_run_projected", launch)
    metadata = SimpleNamespace(
        num_actual_tokens=4,
        num_spec_decodes=1,
        num_prefills=1,
        num_decodes=0,
        spec_token_indx=spec_indices,
        non_spec_token_indx=prefill_indices,
        spec_state_indices_tensor=spec_slots,
        non_spec_state_indices_tensor=prefill_slots,
        spec_decode_metadata=SimpleNamespace(
            spec_causal_conv1d=SimpleNamespace(
                query_start_loc=prefix, cache_indices=spec_slots[:, 0], num_accepted_tokens=accepted
            ),
        ),
        non_spec_prefill_metadata=SimpleNamespace(
            causal_conv1d=SimpleNamespace(
                query_start_loc=prefix, cache_indices=prefill_slots, initial_state_mode=no_initial
            ),
        ),
    )

    mega_kda.run_mega_kda(layer, operator, metadata, qkv, raw_gate, gate, beta, result, packed_weights)

    spec_args, prefill_args = [call.args for call in launch.call_args_list]
    torch.testing.assert_close(spec_args[2], qkv[spec_indices])
    torch.testing.assert_close(prefill_args[2], qkv[prefill_indices])
    assert spec_args[6] is prefix
    assert spec_args[7] is spec_slots
    assert spec_args[9] is accepted
    assert prefill_args[10] is no_initial
    expected = torch.stack([prefill_output[0], spec_output[0], prefill_output[1], spec_output[1]]).reshape_as(result)
    torch.testing.assert_close(norm.call_args.args[0], expected)
    assert norm.call_args.kwargs["out"] is result


def test_mega_kda_clears_idle_graph_output(monkeypatch):
    launch = MagicMock()
    monkeypatch.setattr(mega_kda, "_run_projected", launch)
    metadata = SimpleNamespace(num_actual_tokens=4, num_spec_decodes=0, num_prefills=0, num_decodes=0)
    result = torch.full((1, 4, 1, mega_kda.MEGA_KDA_HEAD_DIM), float("nan"))

    mega_kda.run_mega_kda(None, None, metadata, None, None, None, None, result, None)

    assert torch.count_nonzero(result) == 0
    launch.assert_not_called()


def test_mega_kda_requires_projected_wheel(tmp_path):
    with pytest.raises(RuntimeError, match="projected operator wheel"):
        mega_kda.load_mega_kda(str(tmp_path))


def test_upstream_mixed_batch_uses_public_operator_and_shares_spec_plan(monkeypatch):
    from vllm import forward_context

    context = SimpleNamespace(additional_kwargs={})
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    prefix = torch.tensor([0, 2], dtype=torch.int32)
    slot = torch.tensor([4], dtype=torch.int32)
    spec = SimpleNamespace(query_start_loc=prefix, cache_indices=slot, num_accepted_tokens=torch.ones_like(slot))
    decode = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        cache_indices=slot - 1,
        initial_state_mode=torch.zeros_like(slot),
    )
    meta = SimpleNamespace(
        num_actual_tokens=3,
        num_decodes=0,
        num_spec_decodes=1,
        num_prefills=1,
        num_prefill_tokens=1,
        spec_token_indx=torch.tensor([0, 2], dtype=torch.int32),
        non_spec_token_indx=torch.tensor([1], dtype=torch.int32),
        spec_state_indices_tensor=slot[:, None],
        spec_decode_metadata=SimpleNamespace(spec_causal_conv1d=spec),
        non_spec_state_indices_tensor=slot - 1,
        non_spec_prefill_metadata=SimpleNamespace(causal_conv1d=decode),
    )
    hidden = torch.tensor([[2, 4], [3, 5], [7, 9], [12, 13]], dtype=torch.bfloat16)

    def launch(
        *,
        hidden_states,
        conv_state,
        checkpoint_state,
        replay_u,
        replay_k,
        replay_decay,
        decay_projection_a_weight,
        qkv_scale_alg,
        **kwargs,
    ):
        assert qkv_scale_alg == 1 and decay_projection_a_weight.shape == (2, 2)
        replay_u.fill_(1)
        replay_k.fill_(0.25)
        replay_decay.fill_(0.5)
        return hidden_states + 1, conv_state, replay_u, replay_k, replay_decay

    op = mega_kda.UpstreamMegaKDA(
        MagicMock(side_effect=launch), MagicMock(return_value=torch.ones(3, dtype=torch.int32))
    )
    weights = {"qkv_scale_alg": 1, "decay_projection_a_weight": torch.ones(2, 2)}
    for _ in range(2):
        state = torch.full((8, 1, 2, 2), 4.0)
        records = SimpleNamespace(correction=torch.zeros(1, 1, 2, 2), key_gate=torch.zeros(1, 1, 2, 4))
        layer = SimpleNamespace(
            head_dim=2,
            gate_lower_bound=-5.0,
            o_norm=SimpleNamespace(eps=1e-6),
            use_recoverssm=True,
            num_spec=1,
            recoverssm_records=records,
            recoverssm_normalized_key_cache=torch.zeros_like(records.correction),
            recoverssm_decay_cache=torch.zeros_like(records.correction),
            kv_cache=(torch.zeros(8, 4, 6, dtype=torch.bfloat16), state),
        )
        result = mega_kda.run_upstream_mega_kda(layer, op, meta, hidden, weights)
        torch.testing.assert_close(result[:3], hidden[:3] + 1)
        assert result[3].count_nonzero() == 0
        torch.testing.assert_close(state[4], torch.full_like(state[4], 4))
        torch.testing.assert_close(state[3], torch.full_like(state[3], 0.25))
        torch.testing.assert_close(layer.recoverssm_normalized_key_cache, torch.full((1, 1, 2, 2), 0.25))
        torch.testing.assert_close(layer.recoverssm_decay_cache, torch.full((1, 1, 2, 2), 0.5))
    assert op.forward.call_count == 4
    assert len(context.additional_kwargs["mega_kda_state_plans"]) == 2
