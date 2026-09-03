# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from vllm_ascend.ops.kimi_kda import (
    _PACKED_CONV_WEIGHT_NAME,
    AscendKimiK3DeltaAttention,
    _prepare_beta,
    _zero_padded_output,
    _zero_padded_recurrent_output,
)


def test_zero_padded_recurrent_output_clears_uncovered_tail():
    output = torch.randn(1, 8, 2, 3)
    expected = output[:, :5].clone()
    output[:, 5:] = torch.nan

    actual = _zero_padded_recurrent_output(
        output,
        torch.tensor([0, 3, 5, 5], dtype=torch.int32),
    )

    torch.testing.assert_close(actual[:, :5], expected)
    assert torch.equal(actual[:, 5:], torch.zeros_like(actual[:, 5:]))
    assert torch.isfinite(actual).all()


def test_zero_padded_output_uses_combined_live_token_count():
    output = torch.full((1, 8, 1, 1), torch.nan)
    output[:, :6] = torch.arange(6).view(1, 6, 1, 1)

    actual = _zero_padded_output(output, torch.tensor(6, dtype=torch.int32))

    torch.testing.assert_close(actual[:, :6], output[:, :6])
    assert torch.equal(actual[:, 6:], torch.zeros_like(actual[:, 6:]))


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
