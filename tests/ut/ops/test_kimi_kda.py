# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from vllm_ascend.ops.kimi_kda import (
    _PACKED_CONV_WEIGHT_NAME,
    AscendKimiK3DeltaAttention,
    AscendKimiK3MergedGateProjection,
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


def test_run_causal_conv1d_returns_declared_output_alias():
    mixed_qkv = torch.randn(3, 8)
    conv_weights = torch.randn(4, 8)
    conv_state = torch.randn(2, 8, 4)
    query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    cache_indices = torch.tensor([1], dtype=torch.int32)
    returned_alias = torch.full_like(mixed_qkv, 7)

    with patch.object(
        torch.ops._C_ascend,
        "npu_causal_conv1d_custom",
        return_value=returned_alias,
        create=True,
    ) as causal_conv:
        actual = AscendKimiK3DeltaAttention._run_causal_conv1d(
            mixed_qkv,
            conv_weights,
            conv_state,
            query_start_loc,
            cache_indices,
            None,
            run_mode=1,
            num_accepted_tokens=torch.tensor([3], dtype=torch.int32),
        )

    assert actual is returned_alias
    assert causal_conv.call_args.kwargs["query_start_loc_opt"] is query_start_loc
    assert causal_conv.call_args.kwargs["cache_indices_opt"] is cache_indices
    assert causal_conv.call_args.kwargs["initial_state_mode_opt"] is None


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
    with (
        patch(
            "vllm_ascend.ops.kimi_kda.KimiK3DeltaAttention.__init__",
            new=fake_upstream_init,
        ),
        patch("vllm_ascend.ops.kimi_kda.is_vl_model", return_value=False),
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


def test_merged_gate_projection_uses_vllm_shard_loader():
    projection = AscendKimiK3MergedGateProjection.__new__(
        AscendKimiK3MergedGateProjection,
    )
    nn.Module.__init__(projection)
    param = nn.Parameter(torch.empty(4, 3))
    loaded_weight = torch.empty(2, 3)

    with patch(
        "vllm_ascend.ops.kimi_kda._KimiGDNMergedColumnParallelLinear.weight_loader",
        autospec=True,
    ) as weight_loader:
        projection.load_shard_weight(param, loaded_weight, shard_id=2)

    weight_loader.assert_called_once_with(
        projection,
        param,
        loaded_weight,
        2,
    )


def test_kda_output_norm_gate_uses_platform_custom_op_dispatch():
    class FakeNPUTensor:
        device = SimpleNamespace(type="npu")

        @staticmethod
        def numel():
            return 1

    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    expected = object()
    attention.o_norm = MagicMock(return_value=expected)
    core_attn_out = FakeNPUTensor()
    output_gate = object()

    actual = attention._apply_output_norm_gate(core_attn_out, output_gate)

    assert actual is expected
    attention.o_norm.assert_called_once_with(core_attn_out, output_gate)


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

    packed = attention._conv_weights_t()
    assert packed.data_ptr() == original_ptr
    assert packed.dtype == torch.bfloat16
    assert packed.is_contiguous()
    torch.testing.assert_close(
        packed,
        source[:, 0, :].transpose(0, 1).to(torch.bfloat16),
    )
