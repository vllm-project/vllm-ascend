#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.mamba.gdn import kimi_gdn_linear_attn
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_ascend.ops.kimi_kda import (
    _PACKED_CONV_WEIGHT_NAME,
    AscendKimiGatedDeltaNetAttention,
    _load_a_log,
    _zero_padded_spec_output,
)


class _NoopQuantMethod(QuantizeMethodBase):
    def create_weights(self, layer: nn.Module, *args, **kwargs):
        raise NotImplementedError

    def apply(self, layer: nn.Module, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


def _make_conv_pack_attention(
    *,
    local_num_heads: int = 2,
    head_dim: int = 3,
    conv_size: int = 4,
    model_dtype: torch.dtype = torch.bfloat16,
) -> AscendKimiGatedDeltaNetAttention:
    attention = AscendKimiGatedDeltaNetAttention.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(attention)
    attention.local_num_heads = local_num_heads
    attention.head_dim = head_dim
    attention.conv_size = conv_size
    attention.model_config = SimpleNamespace(dtype=model_dtype)

    local_channels = local_num_heads * head_dim
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        conv = nn.Module()
        conv.weight = nn.Parameter(
            torch.empty(
                local_channels,
                1,
                conv_size,
                dtype=torch.float32,
            )
        )
        conv.weight.weight_loader = default_weight_loader
        conv.quant_method = _NoopQuantMethod()
        setattr(attention, name, conv)

    attention.q_conv1d.register_parameter(
        _PACKED_CONV_WEIGHT_NAME,
        nn.Parameter(
            torch.empty(
                attention._packed_conv_shape(),
                dtype=model_dtype,
            ),
            requires_grad=False,
        ),
    )
    for conv in (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d):
        attention._wrap_conv_process_weights(conv)
    return attention


def _process_conv_weights(
    attention: AscendKimiGatedDeltaNetAttention,
) -> None:
    for conv in (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d):
        conv.quant_method.process_weights_after_loading(conv)


def _expected_packed_conv_weights(
    attention: AscendKimiGatedDeltaNetAttention,
) -> torch.Tensor:
    return torch.cat(
        [
            conv.weight[:, 0, :].transpose(0, 1)
            for conv in (
                attention.q_conv1d,
                attention.k_conv1d,
                attention.v_conv1d,
            )
        ],
        dim=1,
    ).to(attention.model_config.dtype)


def test_upstream_kda_dispatch_accepts_beta_keyword_during_profile():
    attention = AscendKimiGatedDeltaNetAttention.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(attention)
    context = SimpleNamespace(
        attn_metadata=None,
        no_compile_layers={"kda": attention},
    )
    qkv = torch.empty(2, 6)
    gate = torch.empty(1, 2, 2, 3)
    raw_beta = torch.empty(1, 2, 2)
    output = torch.randn(1, 2, 2, 3)
    expected = output.clone()

    with (
        patch.object(kimi_gdn_linear_attn, "get_forward_context", return_value=context),
        patch("vllm_ascend.ops.kimi_kda.get_forward_context", return_value=context) as get_context,
    ):
        kimi_gdn_linear_attn.kda_attention(qkv, qkv, qkv, gate, raw_beta, output, "kda")

    get_context.assert_called_once_with()
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("initial_state", [None, [True, False, True, False]])
@pytest.mark.parametrize("cache_layout", ["flat", "block-table"])
def test_causal_conv1d_prefill_keeps_official_packed_varlen_contract(initial_state, cache_layout):
    query_start_loc = torch.tensor([0, 1, 6, 8, 8], dtype=torch.int32)
    cache_indices = torch.tensor([2, 5, 3, -1], dtype=torch.int32)
    if cache_layout == "block-table":
        cache_indices = torch.stack((cache_indices, torch.full_like(cache_indices, 7)), dim=1)
    metadata = SimpleNamespace(query_start_loc=query_start_loc, cache_indices=cache_indices)
    if initial_state is not None:
        # Model metadata can be a non-contiguous boolean view.
        initial_state_table = torch.tensor([[value, False] for value in initial_state], dtype=torch.bool)
        metadata.initial_state_mode = initial_state_table[:, 0]
    mixed_qkv = torch.empty(8, 6)
    conv_weights_t = torch.empty(4, 6)
    conv_state = torch.empty(8, 3, 6)
    expected_output = torch.empty_like(mixed_qkv)

    with patch(
        "vllm_ascend.ops.kimi_kda.torch.ops.cann_ops_transformer.causal_conv1d_fn",
        return_value=expected_output,
    ) as prefill:
        output = AscendKimiGatedDeltaNetAttention._run_causal_conv1d(
            mixed_qkv,
            conv_weights_t,
            conv_state,
            metadata,
            run_mode=0,
        )

    prefill.assert_called_once()
    kwargs = prefill.call_args.kwargs
    assert output is expected_output
    assert kwargs["x"] is mixed_qkv
    assert kwargs["conv_states"] is conv_state
    assert kwargs["weight"] is conv_weights_t
    assert kwargs["bias"] is None
    assert kwargs["query_start_loc"] is query_start_loc
    torch.testing.assert_close(kwargs["cache_indices"], torch.tensor([2, 5, 3, 0], dtype=torch.int32))
    assert kwargs["cache_indices"].is_contiguous()
    expected_initial_state = (
        torch.zeros(4, dtype=torch.int32) if initial_state is None else metadata.initial_state_mode.to(torch.int32)
    )
    torch.testing.assert_close(kwargs["has_initial_state"], expected_initial_state)
    assert kwargs["has_initial_state"].is_contiguous()


@pytest.mark.parametrize(
    ("query_offsets", "cache_ids"),
    [
        pytest.param([0, 1, 2], [2, 4], id="ordinary-decode"),
        pytest.param(
            [0, 1, 2, 2, 2],
            [[2, 3], [4, 5], [0, 0], [0, 0]],
            id="graph-padding",
        ),
    ],
)
def test_causal_conv1d_update_uses_3d_fixed_batch_for_non_spec_decode(query_offsets, cache_ids):
    query_start_loc = torch.tensor(query_offsets, dtype=torch.int32)
    cache_indices = torch.tensor(cache_ids, dtype=torch.int32)
    metadata = SimpleNamespace(query_start_loc=query_start_loc, cache_indices=cache_indices)
    mixed_qkv = torch.empty(query_offsets[-1], 6)
    conv_weights_t = torch.empty(4, 6)
    conv_state = torch.empty(8, 6, 6)
    operator_output = torch.empty(mixed_qkv.shape[0], 1, mixed_qkv.shape[1])

    with patch(
        "vllm_ascend.ops.kimi_kda.torch.ops.cann_ops_transformer.causal_conv1d_update",
        return_value=operator_output,
    ) as update:
        output = AscendKimiGatedDeltaNetAttention._run_causal_conv1d(
            mixed_qkv,
            conv_weights_t,
            conv_state,
            metadata,
            run_mode=1,
        )

    update.assert_called_once()
    kwargs = update.call_args.kwargs
    assert output.shape == mixed_qkv.shape
    assert output.data_ptr() == operator_output.data_ptr()
    assert kwargs["query_start_loc"] is query_start_loc
    assert kwargs["x"].shape == (mixed_qkv.shape[0], 1, mixed_qkv.shape[1])
    assert kwargs["x"].data_ptr() == mixed_qkv.data_ptr()
    assert kwargs["conv_state"] is conv_state
    assert kwargs["num_accepted_tokens"] is None
    expected_indices = cache_indices if cache_indices.ndim == 1 else cache_indices[:, 0]
    torch.testing.assert_close(kwargs["conv_state_indices"], expected_indices.clamp_min(0))
    assert kwargs["conv_state_indices"].is_contiguous()


def test_causal_conv1d_update_keeps_2d_varlen_for_spec_decode():
    query_start_loc = torch.tensor([0, 3, 6, 6], dtype=torch.int32)
    cache_indices = torch.tensor([[2, 3, 4], [5, 6, 7], [-1, -1, -1]], dtype=torch.int32)
    metadata = SimpleNamespace(query_start_loc=query_start_loc, cache_indices=cache_indices)
    mixed_qkv = torch.empty(6, 6)
    conv_weights_t = torch.empty(4, 6)
    conv_state = torch.empty(8, 6, 6)
    accepted = torch.tensor([2, 1, 0], dtype=torch.int32)
    expected_output = torch.empty_like(mixed_qkv)

    with patch(
        "vllm_ascend.ops.kimi_kda.torch.ops.cann_ops_transformer.causal_conv1d_update",
        return_value=expected_output,
    ) as update:
        output = AscendKimiGatedDeltaNetAttention._run_causal_conv1d(
            mixed_qkv,
            conv_weights_t,
            conv_state,
            metadata,
            run_mode=1,
            num_accepted_tokens=accepted,
        )

    update.assert_called_once()
    kwargs = update.call_args.kwargs
    assert output is expected_output
    assert kwargs["x"] is mixed_qkv
    assert kwargs["conv_state"] is conv_state
    assert kwargs["query_start_loc"] is query_start_loc
    assert kwargs["num_accepted_tokens"] is accepted
    torch.testing.assert_close(kwargs["conv_state_indices"], torch.tensor([2, 5, 0], dtype=torch.int32))


@pytest.mark.parametrize("compact_metadata", [False, True])
def test_recipe_prefill_preserves_request_boundaries_and_cached_state(compact_metadata):
    attention = AscendKimiGatedDeltaNetAttention.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(attention)
    attention.local_num_heads = 2
    attention.head_dim = 3
    attention.gate_lower_bound = -5.0
    attention.A_log = nn.Parameter(torch.zeros(2))
    attention.dt_bias = nn.Parameter(torch.zeros(2, 3))

    # Unequal lengths must be padded independently, not to a shared maximum.
    boundaries = (0, 3, 68)
    q, k, v, raw_gate = (torch.randn(1, 68, 2, 3, dtype=torch.bfloat16) for _ in range(4))
    raw_beta = torch.randn(1, 68, 2, dtype=torch.bfloat16)
    recurrent_state = torch.arange(6 * 2 * 3 * 3, dtype=torch.float32).reshape(6, 2, 3, 3).to(torch.bfloat16)
    previous_state = recurrent_state.clone()
    if compact_metadata:
        # The middle metadata row has no tokens and must not update its cache.
        state_indices = torch.tensor([4, 2, 1], dtype=torch.int32)
        has_initial_state = torch.tensor([False, True, True])
        metadata = SimpleNamespace(
            cu_seqlens_host=(0, 3, 3, 68),
            cu_seqlens_kern=torch.tensor(boundaries, dtype=torch.int32),
            keep_meta=torch.tensor([True, False, True]),
        )
    else:
        state_indices = torch.tensor([4, 1], dtype=torch.int32)
        has_initial_state = torch.tensor([False, True])
        metadata = SimpleNamespace(cu_seqlens_host=boundaries, cu_seqlens_kern=None, keep_meta=None)

    expected_initial = [torch.zeros(1, 2, 3, 3), previous_state[1:2].float()]
    calls = []

    def flash_kda(request_q, request_k, request_v, **kwargs):
        request_index = len(calls)
        start, end = boundaries[request_index : request_index + 2]
        length = end - start
        padded_length = (64, 128)[request_index]
        assert request_q.shape == (1, padded_length, 2, 3)
        for actual, source in zip((request_q, request_k, request_v), (q, k, v)):
            torch.testing.assert_close(actual[:, :length], source[:, start:end])
            assert torch.count_nonzero(actual[:, length:]) == 0
            assert actual.is_contiguous()
        torch.testing.assert_close(kwargs["g"][:, :length], raw_gate[:, start:end])
        torch.testing.assert_close(kwargs["beta"][:, :length], raw_beta[:, start:end])
        assert torch.isneginf(kwargs["g"][:, length:]).all()
        assert torch.isneginf(kwargs["beta"][:, length:]).all()
        torch.testing.assert_close(kwargs["initial_state"], expected_initial[request_index])
        assert kwargs["layout_qkv"] == "BSND"
        calls.append(request_index)
        output = torch.full_like(request_q, float("nan"))
        output[:, :length] = request_q[:, :length] + request_v[:, :length]
        return output, kwargs["initial_state"] + 10 * (request_index + 1)

    def clear_initial_states(states, has_state):
        states[~has_state] = 0

    with (
        patch("vllm_ascend.ops.kimi_kda.get_pcp_group", return_value=SimpleNamespace(world_size=1)),
        patch("vllm_ascend.ops.kimi_kda.clear_ssm_states", side_effect=clear_initial_states),
        patch("vllm_ascend.ops.kimi_kda._flash_kda_impl", side_effect=flash_kda),
    ):
        output = attention._run_prefill(
            q, k, v, raw_gate, raw_beta, recurrent_state, state_indices, has_initial_state, metadata
        )

    assert len(calls) == 2
    torch.testing.assert_close(output, q + v)
    expected_state = previous_state.clone()
    expected_state[4] = 10
    expected_state[1] = (previous_state[1].float() + 20).to(recurrent_state.dtype)
    torch.testing.assert_close(recurrent_state, expected_state)


def test_load_a_log_slices_padded_1d_checkpoint_by_tp_rank():
    param = torch.empty(1, 1, 2, 1)
    loaded_weight = torch.arange(6, dtype=torch.float32)

    with patch("vllm_ascend.ops.kimi_kda.get_tensor_model_parallel_rank", return_value=1):
        _load_a_log(param, loaded_weight, num_heads=4)

    torch.testing.assert_close(param, torch.tensor([[[[2.0], [3.0]]]]))


def test_load_a_log_preserves_exact_local_4d_checkpoint():
    param = torch.empty(1, 1, 2, 1)
    loaded_weight = torch.tensor([[[[4.0], [5.0]]]])

    _load_a_log(param, loaded_weight, num_heads=4)

    torch.testing.assert_close(param, loaded_weight)


def test_load_a_log_rejects_unsupported_layout():
    with pytest.raises(ValueError, match="must be 1-D or 4-D"):
        _load_a_log(torch.empty(1, 1, 2, 1), torch.empty(2, 2), num_heads=4)


def test_zero_padded_spec_output_clears_uninitialized_tail():
    output = torch.arange(16 * 2 * 3, dtype=torch.float32).reshape(1, 16, 2, 3)
    output[:, 8:] = torch.nan
    query_start_loc = torch.tensor([0, 8, 8], dtype=torch.int32)

    masked = _zero_padded_spec_output(output, query_start_loc)

    torch.testing.assert_close(masked[:, :8], output[:, :8])
    assert torch.equal(masked[:, 8:], torch.zeros_like(masked[:, 8:]))
    assert torch.isfinite(masked).all()


def test_zero_padded_spec_output_preserves_fully_covered_output():
    output = torch.randn(1, 16, 2, 3)
    query_start_loc = torch.tensor([0, 8, 16], dtype=torch.int32)

    masked = _zero_padded_spec_output(output, query_start_loc)

    torch.testing.assert_close(masked, output)


def test_zero_padded_spec_output_supports_multiple_real_and_dummy_rows():
    output = torch.randn(1, 32, 2, 3)
    expected = output[:, :16].clone()
    output[:, 16:] = torch.nan
    query_start_loc = torch.tensor([0, 8, 16, 16, 16], dtype=torch.int32)

    masked = _zero_padded_spec_output(output, query_start_loc)

    torch.testing.assert_close(masked[:, :16], expected)
    assert torch.equal(masked[:, 16:], torch.zeros_like(masked[:, 16:]))
    assert masked.shape == output.shape
    assert masked.dtype == output.dtype
    assert masked.device == output.device


def test_output_norm_gate_uses_kda_fused_triton_kernel():
    attention = AscendKimiGatedDeltaNetAttention.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(attention)
    attention.o_norm = SimpleNamespace(
        weight=nn.Parameter(torch.randn(3)),
        eps=1e-6,
    )
    core_attn_out = torch.randn(1, 4, 2, 3)
    output_gate = torch.randn(4, 2, 3)
    expected = torch.randn_like(core_attn_out)

    with patch(
        "vllm_ascend.ops.kimi_kda.apply_kda_rms_norm_sigmoid_gate",
        return_value=expected,
    ) as fused_norm_gate:
        actual = attention._apply_output_norm_gate(core_attn_out, output_gate)

    assert actual is expected
    fused_norm_gate.assert_called_once_with(
        core_attn_out,
        output_gate,
        attention.o_norm.weight,
        attention.o_norm.eps,
    )


def test_conv_post_load_processing_packs_kernel_layout_in_place():
    attention = _make_conv_pack_attention()
    convs = (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d)
    for shard_id, conv in enumerate(convs):
        conv.weight.data.copy_(
            torch.arange(
                conv.weight.numel(),
                dtype=torch.float32,
            ).reshape_as(conv.weight)
            + shard_id * 100
        )

    packed = attention.q_conv1d.get_parameter(_PACKED_CONV_WEIGHT_NAME)
    original_ptr = packed.data_ptr()
    _process_conv_weights(attention)

    assert packed.data_ptr() == original_ptr
    torch.testing.assert_close(packed, _expected_packed_conv_weights(attention))
    assert packed.dtype == torch.bfloat16
    assert packed.is_contiguous()
    assert attention._conv_weights_t().data_ptr() == original_ptr
    parameter_name = f"q_conv1d.{_PACKED_CONV_WEIGHT_NAME}"
    assert dict(attention.named_parameters())[parameter_name] is packed
    assert attention.state_dict()[parameter_name].data_ptr() == original_ptr

    convs[1].weight.data.fill_(777)
    convs[1].quant_method.process_weights_after_loading(convs[1])

    assert attention._conv_weights_t().data_ptr() == original_ptr
    torch.testing.assert_close(
        attention._conv_weights_t(),
        _expected_packed_conv_weights(attention),
    )


def test_full_checkpoint_reload_refreshes_packed_weight_in_place():
    attention = _make_conv_pack_attention()
    convs = (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d)
    for shard_id, conv in enumerate(convs, start=1):
        conv.weight.data.fill_(shard_id)
    _process_conv_weights(attention)
    original_packed = attention._conv_weights_t()
    original_ptr = original_packed.data_ptr()

    record_metadata_for_reloading(attention)
    initialize_layerwise_reload(attention)
    for shard_id, conv in enumerate(convs, start=11):
        loaded_weight = torch.full(
            conv.weight.shape,
            shard_id,
            dtype=torch.float32,
        )
        conv.weight.weight_loader(conv.weight, loaded_weight)
    finalize_layerwise_reload(
        attention,
        SimpleNamespace(dtype=torch.bfloat16),
    )

    refreshed = attention._conv_weights_t()
    assert refreshed is original_packed
    assert refreshed.data_ptr() == original_ptr
    torch.testing.assert_close(
        refreshed,
        _expected_packed_conv_weights(attention),
    )


def test_repack_waits_until_all_source_weights_are_materialized():
    attention = _make_conv_pack_attention()
    source_convs = (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d)
    for value, conv in enumerate(source_convs, start=1):
        conv.weight.data.fill_(value)
    packed = attention._conv_weights_t()
    packed.data.fill_(-1)
    before = packed.clone()
    v_shape = attention.v_conv1d.weight.shape
    attention.v_conv1d.weight = nn.Parameter(
        torch.empty(v_shape, device="meta"),
    )

    attention.q_conv1d.quant_method.process_weights_after_loading(attention.q_conv1d)

    torch.testing.assert_close(packed, before)

    attention.v_conv1d.weight = nn.Parameter(
        torch.full(v_shape, 3, dtype=torch.float32),
    )
    attention.v_conv1d.quant_method.process_weights_after_loading(attention.v_conv1d)

    torch.testing.assert_close(packed, _expected_packed_conv_weights(attention))


def test_kernel_format_reload_updates_named_packed_parameter():
    attention = _make_conv_pack_attention(model_dtype=torch.float16)
    for shard_id, conv in enumerate(
        (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d),
        start=1,
    ):
        conv.weight.data.fill_(shard_id)
    _process_conv_weights(attention)

    parameter_name = f"q_conv1d.{_PACKED_CONV_WEIGHT_NAME}"
    packed = attention.get_parameter(parameter_name)
    original_ptr = packed.data_ptr()
    kernel_weight = torch.arange(
        packed.numel(),
        dtype=packed.dtype,
    ).reshape_as(packed)

    with torch.no_grad():
        attention.get_parameter(parameter_name).copy_(kernel_weight)

    assert attention._conv_weights_t().data_ptr() == original_ptr
    torch.testing.assert_close(attention._conv_weights_t(), kernel_weight)
