# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import create_autospec, patch

import torch
from torch import nn
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.ops.kimi_kda import (
    _PACKED_CONV_WEIGHT_NAME,
    AscendKimiK3DeltaAttention,
    _live_token_counts,
    _prepare_beta,
    _take_live_tokens,
)


def test_live_token_counts_ignore_graph_padded_sequences():
    spec_meta = SimpleNamespace(
        spec_sequence_masks=torch.tensor([True, True, False, False]),
        num_spec_decode_tokens=8,
        num_decode_tokens=0,
        num_prefill_tokens=0,
    )
    decode_meta = SimpleNamespace(
        spec_sequence_masks=None,
        num_spec_decode_tokens=4,
        num_decode_tokens=2,
        num_prefill_tokens=0,
    )

    mixed_meta = SimpleNamespace(
        spec_sequence_masks=torch.tensor([True, False, False]),
        num_spec_decode_tokens=2,
        num_decode_tokens=0,
        num_prefill_tokens=1,
    )

    assert _live_token_counts(spec_meta) == (8, 0)
    assert _live_token_counts(decode_meta) == (0, 2)
    assert _live_token_counts(mixed_meta) == (2, 1)


def test_take_live_tokens_discards_unwritten_kernel_tail():
    output = torch.randn(1, 8, 2, 3)
    expected = output[:, :5].clone()
    output[:, 5:] = torch.nan

    actual = _take_live_tokens(output, 5)

    torch.testing.assert_close(actual, expected)
    assert _take_live_tokens(output, 8) is output
    assert _take_live_tokens(output, 0).shape[1] == 0


def test_kda_output_norm_uses_checkpoint_epsilon():
    def fake_upstream_init(attention, _config, _vllm_config, _prefix):
        nn.Module.__init__(attention)
        attention.o_norm = SimpleNamespace(eps=1e-5)
        attention.conv_size = 4
        attention.local_projection_size = 2
        attention.model_config = SimpleNamespace(dtype=torch.bfloat16)
        attention.conv1d = nn.Module()
        attention.conv1d.weight = nn.Parameter(torch.empty(6, 1, 4))
        attention.conv1d.quant_method = SimpleNamespace(process_weights_after_loading=lambda: None)

    config = SimpleNamespace(rms_norm_eps=1e-6)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            multimodal_config=None,
            enable_prompt_embeds=False,
        )
    )
    with patch(
        "vllm_ascend.ops.kimi_kda.KimiK3DeltaAttention.__init__",
        new=fake_upstream_init,
    ):
        attention = AscendKimiK3DeltaAttention(config, vllm_config)

    assert attention.o_norm.eps == config.rms_norm_eps


def test_prepare_beta_slices_and_applies_sigmoid_in_fp32():
    raw_beta = torch.tensor(
        [[[-20.0], [0.0], [20.0], [100.0]]],
        dtype=torch.bfloat16,
    )

    beta = _prepare_beta(raw_beta, num_actual_tokens=3)

    assert beta.dtype == torch.float32
    assert beta.shape == (1, 3, 1)
    torch.testing.assert_close(beta, raw_beta[:, :3].float().sigmoid())
    assert torch.all((beta >= 0.0) & (beta <= 1.0))


def test_prefill_fuses_raw_gate_and_updates_v_first_state():
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    attention.head_dim = 2
    attention.gate_lower_bound = None
    attention.A_log = nn.Parameter(torch.randn(1))
    attention.dt_bias = nn.Parameter(torch.randn(2))

    q = torch.randn(1, 2, 1, 2)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_gate = torch.randn_like(q)
    beta = torch.randn(1, 2, 1)
    recurrent_state = torch.randn(1, 1, 2, 2)
    state_indices = torch.tensor([0], dtype=torch.int32)
    has_initial_state = torch.tensor([True])
    metadata = SimpleNamespace(
        cu_seqlens_host=(0, 2),
        cu_seqlens_kern=None,
        keep_meta=None,
        chunk_indices_chunk64_host=(0, 0),
    )
    output = torch.randn_like(v)
    final_state = torch.randn(1, 1, 2, 2)

    with (
        patch("vllm_ascend.ops.kimi_kda.clear_ssm_states"),
        patch("vllm_ascend.ops.kimi_kda.l2norm_fwd", side_effect=lambda x: x),
        patch.object(
            torch.ops._C_ascend,
            "chunk_kda_fwd",
            return_value=(output, final_state, *([None] * 10)),
            create=True,
        ) as chunk_kda_fwd,
    ):
        actual = attention._run_prefill(
            q,
            k,
            v,
            raw_gate,
            beta,
            recurrent_state,
            state_indices,
            has_initial_state,
            metadata,
        )

    assert actual is output
    assert chunk_kda_fwd.call_args.args[3] is raw_gate
    assert chunk_kda_fwd.call_args.kwargs["use_gate_in_kernel"] is True
    assert chunk_kda_fwd.call_args.kwargs["state_v_first"] is True
    assert chunk_kda_fwd.call_args.kwargs["safe_gate"] is False
    torch.testing.assert_close(recurrent_state[state_indices], final_state)


def test_kda_empty_forward_context_clears_preallocated_output():
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    core_attn_out = torch.full((1, 4, 2, 3), torch.nan)

    with patch(
        "vllm_ascend.ops.kimi_kda.get_forward_context",
        return_value=SimpleNamespace(attn_metadata=None),
    ):
        attention._forward(
            mixed_qkv=torch.empty(4, 18),
            g1=torch.empty(1, 4, 2, 3),
            g2=torch.empty(4, 2, 3),
            beta=torch.empty(1, 4, 2),
            core_attn_out=core_attn_out,
        )

    assert torch.equal(core_attn_out, torch.zeros_like(core_attn_out))


def _decode_graph_metadata(*, num_actual_tokens: int, num_decode_tokens: int, num_decodes: int):
    query_start_loc = torch.zeros(num_decodes + 1, dtype=torch.int32)
    query_start_loc[1 : num_decode_tokens + 1] = torch.arange(1, num_decode_tokens + 1)
    query_start_loc[num_decode_tokens + 1 :] = num_decode_tokens
    state_indices = torch.full((num_decodes,), -1, dtype=torch.int32)
    state_indices[:num_decode_tokens] = torch.arange(num_decode_tokens)

    conv1d_meta = SimpleNamespace(
        query_start_loc=query_start_loc,
        cache_indices=state_indices,
        initial_state_mode=None,
        num_accepted_tokens=None,
    )
    metadata = create_autospec(GDNAttentionMetadata, instance=True)
    metadata.num_actual_tokens = num_actual_tokens
    metadata.num_prefills = 0
    metadata.num_prefill_tokens = 0
    metadata.num_decodes = num_decodes
    metadata.num_decode_tokens = num_decode_tokens
    metadata.num_spec_decodes = 0
    metadata.num_spec_decode_tokens = 0
    metadata.spec_sequence_masks = None
    metadata.spec_token_indx = None
    metadata.non_spec_token_indx = None
    metadata.spec_query_start_loc = None
    metadata.non_spec_query_start_loc = query_start_loc
    metadata.spec_state_indices_tensor = None
    metadata.non_spec_state_indices_tensor = state_indices
    metadata.spec_decode_metadata = None
    metadata.non_spec_prefill_metadata = None
    metadata.non_spec_decode_metadata = SimpleNamespace(causal_conv1d=conv1d_meta)
    return metadata


def _spec_graph_metadata(*, num_actual_tokens: int, num_spec_decode_tokens: int, num_spec_decodes: int):
    query_start_loc = torch.zeros(num_spec_decodes + 1, dtype=torch.int32)
    query_start_loc[1 : num_spec_decode_tokens + 1] = torch.arange(1, num_spec_decode_tokens + 1)
    query_start_loc[num_spec_decode_tokens + 1 :] = num_spec_decode_tokens
    state_indices = torch.full((num_spec_decodes,), -1, dtype=torch.int32)
    state_indices[:num_spec_decode_tokens] = torch.arange(num_spec_decode_tokens)

    conv1d_meta = SimpleNamespace(
        query_start_loc=query_start_loc,
        cache_indices=state_indices,
        initial_state_mode=None,
        num_accepted_tokens=torch.ones(num_spec_decodes, dtype=torch.int32),
    )
    metadata = create_autospec(GDNAttentionMetadata, instance=True)
    metadata.num_actual_tokens = num_actual_tokens
    metadata.num_prefills = 0
    metadata.num_prefill_tokens = 0
    metadata.num_decodes = 0
    metadata.num_decode_tokens = 0
    metadata.num_spec_decodes = num_spec_decodes
    metadata.num_spec_decode_tokens = num_spec_decode_tokens
    metadata.spec_sequence_masks = torch.tensor(
        [True] * num_spec_decode_tokens + [False] * (num_spec_decodes - num_spec_decode_tokens)
    )
    metadata.spec_token_indx = None
    metadata.non_spec_token_indx = None
    metadata.spec_query_start_loc = query_start_loc
    metadata.non_spec_query_start_loc = None
    metadata.spec_state_indices_tensor = state_indices
    metadata.non_spec_state_indices_tensor = None
    metadata.spec_decode_metadata = SimpleNamespace(spec_causal_conv1d=conv1d_meta)
    metadata.non_spec_prefill_metadata = None
    metadata.non_spec_decode_metadata = None
    return metadata


def _mixed_spec_prefill_metadata():
    spec_token_indx = torch.tensor([0, 2], dtype=torch.long)
    non_spec_token_indx = torch.tensor([1], dtype=torch.long)
    spec_query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    prefill_query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    spec_state_indices = torch.tensor([0, 1], dtype=torch.int32)
    prefill_state_indices = torch.tensor([2], dtype=torch.int32)
    spec_conv1d_meta = SimpleNamespace(
        query_start_loc=spec_query_start_loc,
        cache_indices=spec_state_indices,
        initial_state_mode=None,
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
    )
    prefill_conv1d_meta = SimpleNamespace(
        query_start_loc=prefill_query_start_loc,
        cache_indices=prefill_state_indices,
        initial_state_mode=torch.zeros(1, dtype=torch.int32),
        num_accepted_tokens=None,
    )
    metadata = create_autospec(GDNAttentionMetadata, instance=True)
    metadata.num_actual_tokens = 4
    metadata.num_prefills = 1
    metadata.num_prefill_tokens = 1
    metadata.num_decodes = 0
    metadata.num_decode_tokens = 0
    metadata.num_spec_decodes = 2
    metadata.num_spec_decode_tokens = 2
    metadata.spec_sequence_masks = torch.tensor([True, False, True, False])
    metadata.spec_token_indx = spec_token_indx
    metadata.non_spec_token_indx = non_spec_token_indx
    metadata.spec_query_start_loc = spec_query_start_loc
    metadata.non_spec_query_start_loc = prefill_query_start_loc
    metadata.spec_state_indices_tensor = spec_state_indices
    metadata.non_spec_state_indices_tensor = prefill_state_indices
    metadata.prefill_state_indices = prefill_state_indices
    metadata.prefill_has_initial_state = torch.tensor([False])
    metadata.spec_decode_metadata = SimpleNamespace(spec_causal_conv1d=spec_conv1d_meta)
    metadata.non_spec_prefill_metadata = SimpleNamespace(
        causal_conv1d=prefill_conv1d_meta,
        chunk=SimpleNamespace(),
    )
    metadata.non_spec_decode_metadata = None
    return metadata


def _new_kda_attention():
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    attention.prefix = "layers.0.self_attn"
    attention.head_dim = 3
    attention.kv_cache = (torch.zeros(1), torch.zeros(1))
    attention.get_parameter = lambda _name: torch.zeros(2, 6)
    return attention


def test_kda_forward_norms_only_live_decode_tokens():
    attention = _new_kda_attention()
    captured = {}

    def fake_norm(core, gate):
        captured["tokens"] = core.shape[1]
        captured["gate_nan"] = bool(torch.isnan(gate).any())
        return core

    attention.o_norm = fake_norm
    metadata = _decode_graph_metadata(num_actual_tokens=4, num_decode_tokens=2, num_decodes=4)
    core_attn_out = torch.full((1, 6, 2, 3), torch.nan)
    g2 = torch.full((4, 2, 3), torch.nan)
    g2[:2] = 0.5
    mixed_qkv = torch.ones(4, 18)
    g1 = torch.ones(1, 4, 2, 3)
    beta = torch.ones(1, 4, 2)

    def fake_conv(mixed, *args, **kwargs):
        captured["conv_tokens"] = mixed.shape[0]
        return mixed

    def fake_recurrent(q, *args, **kwargs):
        captured["recurrent_tokens"] = q.shape[1]
        # Simulate a kernel that skips padded sequences and leaves the tail dirty.
        out = torch.ones_like(q)
        out[:, 2:] = torch.nan
        return out

    with (
        patch(
            "vllm_ascend.ops.kimi_kda.get_forward_context",
            return_value=SimpleNamespace(attn_metadata={attention.prefix: metadata}),
        ),
        patch.object(attention, "_run_causal_conv1d", side_effect=fake_conv),
        patch.object(attention, "_run_recurrent", side_effect=fake_recurrent),
    ):
        attention._forward(
            mixed_qkv=mixed_qkv,
            g1=g1,
            g2=g2,
            beta=beta,
            core_attn_out=core_attn_out,
        )

    assert captured["conv_tokens"] == 4
    assert captured["recurrent_tokens"] == 4
    assert captured["tokens"] == 2
    assert captured["gate_nan"] is False
    torch.testing.assert_close(core_attn_out[:, :2], torch.ones(1, 2, 2, 3))
    assert torch.equal(core_attn_out[:, 2:], torch.zeros(1, 4, 2, 3))
    assert torch.isfinite(core_attn_out).all()


def test_kda_forward_norms_only_live_spec_tokens():
    attention = _new_kda_attention()
    captured = {}

    def fake_norm(core, gate):
        captured["tokens"] = core.shape[1]
        captured["gate_nan"] = bool(torch.isnan(gate).any())
        return core

    attention.o_norm = fake_norm
    metadata = _spec_graph_metadata(num_actual_tokens=4, num_spec_decode_tokens=2, num_spec_decodes=4)
    core_attn_out = torch.full((1, 6, 2, 3), torch.nan)
    g2 = torch.full((4, 2, 3), torch.nan)
    g2[:2] = 0.5

    def fake_conv(mixed, *args, **kwargs):
        captured["conv_tokens"] = mixed.shape[0]
        return mixed

    def fake_recurrent(q, *args, **kwargs):
        captured["recurrent_tokens"] = q.shape[1]
        out = torch.ones_like(q)
        out[:, 2:] = torch.nan
        return out

    with (
        patch(
            "vllm_ascend.ops.kimi_kda.get_forward_context",
            return_value=SimpleNamespace(attn_metadata={attention.prefix: metadata}),
        ),
        patch.object(attention, "_run_causal_conv1d", side_effect=fake_conv),
        patch.object(attention, "_run_recurrent", side_effect=fake_recurrent),
    ):
        attention._forward(
            mixed_qkv=torch.ones(4, 18),
            g1=torch.ones(1, 4, 2, 3),
            g2=g2,
            beta=torch.ones(1, 4, 2),
            core_attn_out=core_attn_out,
        )

    assert captured["conv_tokens"] == 4
    assert captured["recurrent_tokens"] == 4
    assert captured["tokens"] == 2
    assert captured["gate_nan"] is False
    torch.testing.assert_close(core_attn_out[:, :2], torch.ones(1, 2, 2, 3))
    assert torch.equal(core_attn_out[:, 2:], torch.zeros(1, 4, 2, 3))
    assert torch.isfinite(core_attn_out).all()


def test_kda_forward_scatters_mixed_tokens_before_live_norm():
    attention = _new_kda_attention()
    captured = {"conv_tokens": []}

    def fake_norm(core, gate):
        captured["tokens"] = core.shape[1]
        captured["gate"] = gate.clone()
        captured["core"] = core.clone()
        captured["gate_nan"] = bool(torch.isnan(gate).any())
        return core

    attention.o_norm = fake_norm
    metadata = _mixed_spec_prefill_metadata()
    core_attn_out = torch.full((1, 6, 2, 3), torch.nan)
    g2 = torch.stack(
        [
            torch.full((2, 3), 10.0),
            torch.full((2, 3), 20.0),
            torch.full((2, 3), 30.0),
            torch.full((2, 3), torch.nan),
        ]
    )

    def fake_conv(mixed, *args, **kwargs):
        captured["conv_tokens"].append(mixed.shape[0])
        return mixed

    def fake_recurrent(q, *args, **kwargs):
        captured["recurrent_tokens"] = q.shape[1]
        return torch.ones_like(q)

    def fake_prefill(q, *args, **kwargs):
        captured["prefill_tokens"] = q.shape[1]
        return torch.full_like(q, 2.0)

    with (
        patch(
            "vllm_ascend.ops.kimi_kda.get_forward_context",
            return_value=SimpleNamespace(attn_metadata={attention.prefix: metadata}),
        ),
        patch.object(attention, "_run_causal_conv1d", side_effect=fake_conv),
        patch.object(attention, "_run_recurrent", side_effect=fake_recurrent),
        patch.object(attention, "_run_prefill", side_effect=fake_prefill),
    ):
        attention._forward(
            mixed_qkv=torch.ones(4, 18),
            g1=torch.ones(1, 4, 2, 3),
            g2=g2,
            beta=torch.ones(1, 4, 2),
            core_attn_out=core_attn_out,
        )

    assert captured["conv_tokens"] == [2, 1]
    assert captured["recurrent_tokens"] == 2
    assert captured["prefill_tokens"] == 1
    assert captured["tokens"] == 3
    assert captured["gate_nan"] is False
    torch.testing.assert_close(captured["core"][0, 0], torch.ones(2, 3))
    torch.testing.assert_close(captured["core"][0, 1], torch.full((2, 3), 2.0))
    torch.testing.assert_close(captured["core"][0, 2], torch.ones(2, 3))
    torch.testing.assert_close(captured["gate"][0], torch.full((2, 3), 10.0))
    torch.testing.assert_close(captured["gate"][1], torch.full((2, 3), 20.0))
    torch.testing.assert_close(captured["gate"][2], torch.full((2, 3), 30.0))
    torch.testing.assert_close(core_attn_out[:, 0], torch.ones(1, 2, 3))
    torch.testing.assert_close(core_attn_out[:, 1], torch.full((1, 2, 3), 2.0))
    torch.testing.assert_close(core_attn_out[:, 2], torch.ones(1, 2, 3))
    assert torch.equal(core_attn_out[:, 3:], torch.zeros(1, 3, 2, 3))
    assert torch.isfinite(core_attn_out).all()


def test_kda_conv_weight_is_packed_once_in_kernel_layout():
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    attention.conv_size = 4
    attention.local_projection_size = 6
    attention.conv1d = nn.Module()
    source = torch.arange(18 * 4, dtype=torch.float32).reshape(18, 1, 4)
    attention.conv1d.weight = nn.Parameter(source)
    attention.register_parameter(
        _PACKED_CONV_WEIGHT_NAME,
        nn.Parameter(torch.empty(4, 18, dtype=torch.bfloat16), requires_grad=False),
    )
    original = attention.get_parameter(_PACKED_CONV_WEIGHT_NAME)
    original_ptr = original.data_ptr()

    attention._pack_conv_weights()

    packed = attention.get_parameter(_PACKED_CONV_WEIGHT_NAME)
    assert packed.data_ptr() == original_ptr
    assert packed.dtype == torch.bfloat16
    assert packed.is_contiguous()
    torch.testing.assert_close(
        packed,
        source[:, 0, :].transpose(0, 1).to(torch.bfloat16),
    )
