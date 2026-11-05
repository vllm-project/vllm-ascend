# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.ops.gdn import (
    AscendGatedDeltaNetAttention,
    _pack_conv_weights,
    initialize_packed_conv_weight,
)
from vllm_ascend.ops.gdn_attn_builder import (
    GDNCausalConv1dMetadata,
    GDNDecodeMetadata,
    GDNPrefillMetadata,
    GDNSpecCausalConv1dMetadata,
    GDNSpecDecodeMetadata,
)


def _make_mixed_metadata() -> GDNAttentionMetadata:
    """One decode token followed by a two-token prefill."""
    metadata = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=2,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=3,
        non_spec_query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([0, 1], dtype=torch.int32),
        spec_sequence_masks=None,
    )

    # Metadata consumed by the mixed non-spec execution path.
    metadata.prefill_query_start_loc = torch.tensor(
        [0, 2],
        dtype=torch.int32,
    )
    metadata.prefill_state_indices = torch.tensor(
        [1],
        dtype=torch.int64,
    )
    metadata.prefill_has_initial_state = torch.tensor([True])

    metadata.non_spec_prefill_metadata = GDNPrefillMetadata(
        causal_conv1d=GDNCausalConv1dMetadata(
            # The convolution processes the complete mixed sequence:
            # one decode token followed by two prefill tokens.
            query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
            cache_indices=torch.tensor([0, 1], dtype=torch.int32),
            initial_state_mode=torch.tensor([1, 1], dtype=torch.int32),
        ),
        chunk=Mock(name="chunk_metadata"),
    )
    metadata.non_spec_decode_metadata = GDNDecodeMetadata(
        causal_conv1d=GDNCausalConv1dMetadata(
            query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
            cache_indices=torch.tensor([0], dtype=torch.int32),
            initial_state_mode=None,
        ),
        actual_seq_lengths=torch.tensor([0, 1], dtype=torch.int32),
    )
    return metadata


def _make_layer() -> SimpleNamespace:
    # Packed QKV layout: Q=2, K=2, V=2. Each tensor is reshaped to
    # [1, num_tokens, 1, 2].
    def rearrange_mixed_qkv(
        mixed_qkv: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if mixed_qkv is None:
            return None, None, None

        query, key, value = mixed_qkv.split([2, 2, 2], dim=-1)
        num_tokens = mixed_qkv.shape[0]
        return (
            query.reshape(1, num_tokens, 1, 2),
            key.reshape(1, num_tokens, 1, 2),
            value.reshape(1, num_tokens, 1, 2),
        )

    layer = SimpleNamespace(
        prefix="layers.0.linear_attn",
        activation=None,
        head_k_dim=2,
        head_v_dim=2,
        num_k_heads=1,
        num_v_heads=1,
        tp_size=1,
        key_dim=2,
        value_dim=2,
        num_spec=3,
        conv1d=nn.Conv1d(
            in_channels=1,
            out_channels=6,
            kernel_size=2,
            bias=False,
        ),
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        kv_cache=(
            torch.zeros(2, 1, 2),
            torch.zeros(2, 1, 2, 2),
        ),
    )
    layer.model_config = SimpleNamespace(dtype=layer.conv1d.weight.dtype)
    initialize_packed_conv_weight(layer)
    _pack_conv_weights(layer.conv1d)
    layer.rearrange_mixed_qkv = Mock(
        name="rearrange_mixed_qkv",
        side_effect=rearrange_mixed_qkv,
    )
    return layer


@pytest.mark.parametrize("initial_state", [True, False])
def test_mixed_non_spec_splits_conv_per_slice(initial_state) -> None:
    layer = _make_layer()
    metadata = _make_mixed_metadata()
    if not initial_state:
        metadata.non_spec_prefill_metadata.causal_conv1d.initial_state_mode = None

    # Token rows contain visibly different Q/K/V values so the assertions
    # can detect an incorrect decode/prefill boundary.
    mixed_qkv = torch.tensor(
        [
            [1.0, 2.0, 11.0, 12.0, 21.0, 22.0],  # decode
            [3.0, 4.0, 13.0, 14.0, 23.0, 24.0],  # prefill
            [5.0, 6.0, 15.0, 16.0, 25.0, 26.0],  # prefill
        ]
    )
    a = torch.zeros(3, 1)
    b = torch.zeros(3, 1)
    core_attn_out = torch.empty(3, 1, 2)

    forward_context = ForwardContext(
        no_compile_layers={layer.prefix: layer},
        attn_metadata={layer.prefix: metadata},
        slot_mapping={},
    )

    recurrent_calls: list[dict[str, torch.Tensor]] = []

    def causal_conv1d(
        input_tensor: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        del args, kwargs
        return input_tensor.clone()

    def recurrent_gated_delta_rule(query, key, value, state, **kwargs) -> torch.Tensor:
        recurrent_calls.append({"query": query, "key": key, "value": value, "state": state, **kwargs})
        # Returning V makes the merged output easy to validate.
        return value.clone()

    def chunk_gated_delta_rule(**kwargs):
        return kwargs["v"].clone(), kwargs["initial_state"].clone()

    # Shape [1, num_tokens, num_heads].
    gating = (
        torch.zeros(1, 3, 1),
        torch.ones(1, 3, 1),
    )

    with (
        override_forward_context(forward_context),
        patch(
            "vllm_ascend.ops.gdn.get_pcp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
        patch(
            "vllm_ascend.ops.gdn.get_current_hardware_profile",
            return_value=SimpleNamespace(supports=lambda _: False),
        ),
        patch(
            "vllm_ascend.ops.gdn.DeviceOperator.fused_gdn_gating",
            return_value=gating,
        ),
        patch("vllm_ascend.ops.gdn.l2norm_fwd", side_effect=lambda x: x),
        patch.object(AscendGatedDeltaNetAttention, "_probe_fused_chunk", return_value=True),
        patch.object(AscendGatedDeltaNetAttention, "_chunk_gated_delta_rule_fused") as fused_chunk_mock,
        patch("vllm_ascend.ops.gdn.clear_ssm_states"),
        patch(
            "vllm_ascend.ops.gdn.gather_ssm_states",
            side_effect=lambda state, indices, has_initial_state, **kwargs: state.index_select(
                0, indices.to(torch.long)
            ),
        ),
        patch(
            "vllm_ascend.ops.gdn.chunk_gated_delta_rule",
            side_effect=chunk_gated_delta_rule,
        ) as chunk_mock,
        patch("vllm_ascend.ops.gdn.maybe_save_kv_layer_to_connector"),
        patch(
            "vllm_ascend.ops.gdn.causal_conv1d_fn",
            side_effect=lambda x, *a, **k: causal_conv1d(x, *a, **k),
        ) as prefill_conv,
        patch(
            "vllm_ascend.ops.gdn.causal_conv1d_update",
            side_effect=lambda x, *a, **k: causal_conv1d(x, *a, **k),
        ) as decode_conv,
        patch("vllm_ascend.ops.gdn.recurrent_gated_delta_rule", side_effect=recurrent_gated_delta_rule),
    ):
        AscendGatedDeltaNetAttention._forward_core(
            layer,
            mixed_qkv,
            b,
            a,
            core_attn_out,
        )

    # The recurrent decode kernel must receive the first token from the
    # already-rearranged full tensors.
    assert len(recurrent_calls) == 1
    decode_call = recurrent_calls[0]
    torch.testing.assert_close(
        decode_call["query"],
        torch.tensor([[[1.0, 2.0]]]),
    )
    torch.testing.assert_close(
        decode_call["key"],
        torch.tensor([[[11.0, 12.0]]]),
    )
    torch.testing.assert_close(
        decode_call["value"],
        torch.tensor([[[21.0, 22.0]]]),
    )

    # Each convolution sees only its own token slice and state rows.
    decode_conv.assert_called_once()
    prefill_conv.assert_called_once()
    torch.testing.assert_close(decode_conv.call_args.args[0], mixed_qkv[:1])
    torch.testing.assert_close(prefill_conv.call_args.args[0], mixed_qkv[1:])
    torch.testing.assert_close(decode_conv.call_args.kwargs["conv_state_indices"], torch.tensor([0], dtype=torch.int32))
    torch.testing.assert_close(prefill_conv.call_args.kwargs["cache_indices"], torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(
        prefill_conv.call_args.kwargs["query_start_loc"], torch.tensor([0, 2], dtype=torch.int32)
    )
    if initial_state:
        torch.testing.assert_close(
            prefill_conv.call_args.kwargs["has_initial_state"], torch.tensor([1], dtype=torch.int32)
        )
    else:
        assert prefill_conv.call_args.kwargs["has_initial_state"] is None
    torch.testing.assert_close(decode_conv.call_args.kwargs["query_start_loc"], torch.tensor([0, 1], dtype=torch.int32))
    prefill_call = chunk_mock.call_args.kwargs
    fused_chunk_mock.assert_not_called()
    torch.testing.assert_close(
        prefill_call["q"],
        torch.tensor([[[[3.0, 4.0]], [[5.0, 6.0]]]]),
    )
    torch.testing.assert_close(
        prefill_call["k"],
        torch.tensor([[[[13.0, 14.0]], [[15.0, 16.0]]]]),
    )
    torch.testing.assert_close(
        prefill_call["v"],
        torch.tensor([[[[23.0, 24.0]], [[25.0, 26.0]]]]),
    )

    # Decode and prefill outputs are stitched back in original token order.
    torch.testing.assert_close(
        core_attn_out,
        torch.tensor(
            [
                [[21.0, 22.0]],
                [[23.0, 24.0]],
                [[25.0, 26.0]],
            ]
        ),
    )


@pytest.mark.parametrize("speculative", [False, True])
def test_decode_forward_does_not_reconcat_qkv(speculative):
    layer = _make_layer()
    packed = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    metadata = _make_mixed_metadata()
    metadata.num_prefills = 0
    metadata.num_prefill_tokens = 0
    metadata.num_actual_tokens = 4
    metadata.num_decodes = 0 if speculative else 4
    metadata.num_decode_tokens = 0 if speculative else 4
    metadata.num_spec_decodes = 2 if speculative else 0
    metadata.num_spec_decode_tokens = 4 if speculative else 0
    metadata.spec_sequence_masks = torch.tensor([True, True]) if speculative else None
    metadata.spec_state_indices_tensor = torch.tensor([0, 1], dtype=torch.int32)
    metadata.non_spec_state_indices_tensor = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    metadata.spec_decode_metadata = GDNSpecDecodeMetadata(
        spec_causal_conv1d=GDNSpecCausalConv1dMetadata(
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
            cache_indices=torch.tensor([0, 1], dtype=torch.int32),
            num_accepted_tokens=torch.tensor([1, 2], dtype=torch.int32),
        ),
        actual_seq_lengths=torch.tensor([0, 2, 2], dtype=torch.int32),
    )
    output = torch.empty(4, 1, 2)
    context = ForwardContext(
        no_compile_layers={layer.prefix: layer},
        attn_metadata={layer.prefix: metadata},
        slot_mapping={},
    )

    def recurrent(q, k, v, state, **kwargs):
        torch.testing.assert_close(q, packed[:, :2].reshape(4, 1, 2))
        torch.testing.assert_close(k, packed[:, 2:4].reshape(4, 1, 2))
        torch.testing.assert_close(v, packed[:, 4:].reshape(4, 1, 2))
        if speculative:
            torch.testing.assert_close(kwargs["num_accepted_tokens"], torch.tensor([1, 2], dtype=torch.int32))
        return v.clone()

    # Reject the inherited concat helper for all real decode inputs.
    original = layer.rearrange_mixed_qkv.side_effect
    layer.rearrange_mixed_qkv.side_effect = (
        lambda x: original(x) if x is None else pytest.fail("decode used inherited QKV concat")
    )
    with (
        override_forward_context(context),
        patch("torch.cat", side_effect=AssertionError("unexpected decode cat")),
        patch("vllm_ascend.ops.gdn.get_pcp_group", return_value=SimpleNamespace(world_size=1)),
        patch(
            "vllm_ascend.ops.gdn.DeviceOperator.fused_gdn_gating",
            return_value=(torch.zeros(1, 4, 1), torch.ones(1, 4, 1)),
        ),
        patch("vllm_ascend.ops.gdn.l2norm_fwd", side_effect=lambda x: x),
        patch("vllm_ascend.ops.gdn.causal_conv1d_update", side_effect=lambda x, *a, **k: x.clone()) as conv,
        patch("vllm_ascend.ops.gdn.recurrent_gated_delta_rule", side_effect=recurrent),
        patch("vllm_ascend.ops.gdn.maybe_save_kv_layer_to_connector"),
    ):
        AscendGatedDeltaNetAttention._forward_core(layer, packed, torch.zeros(4, 1), torch.zeros(4, 1), output)
    torch.testing.assert_close(output, packed[:, 4:].reshape(4, 1, 2))
    if speculative:
        torch.testing.assert_close(
            conv.call_args.kwargs["num_accepted_tokens"],
            torch.tensor([1, 2], dtype=torch.int32),
        )


@pytest.mark.parametrize("mixed,head_dim", [(False, 16), (True, 16), (False, 2)])
def test_prefill_conv_layout_for_pure_and_mixed_batches(mixed, head_dim):
    layer = _make_layer()
    layer.head_k_dim = layer.head_v_dim = head_dim
    layer.key_dim = layer.value_dim = head_dim
    layer.kv_cache = (torch.zeros(2, 1, 3 * head_dim), torch.zeros(2, 1, head_dim, head_dim))
    metadata = _make_mixed_metadata()
    if not mixed:
        metadata.num_decodes = metadata.num_decode_tokens = 0
        metadata.num_prefill_tokens = 3
        metadata.prefill_query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
        conv_meta = metadata.non_spec_prefill_metadata.causal_conv1d
        conv_meta.query_start_loc = metadata.prefill_query_start_loc
        conv_meta.cache_indices = torch.tensor([1], dtype=torch.int32)
        conv_meta.initial_state_mode = torch.tensor([1], dtype=torch.int32)
    packed = torch.arange(9 * head_dim, dtype=torch.float32).reshape(3, 3 * head_dim)
    expected_head_num = 3 if head_dim == 16 else 0

    def conv(x, *args, **kwargs):
        assert kwargs["head_num"] == expected_head_num
        torch.testing.assert_close(x, packed[1:] if mixed else packed)
        if expected_head_num:
            return x.reshape(-1, 3, head_dim).transpose(0, 1).contiguous()
        return x.clone()

    def rearrange(x):
        if x is None:
            return None, None, None
        assert not expected_head_num, "head-first prefill used the old QKV rearrangement"
        return tuple(part.reshape(1, -1, 1, head_dim) for part in x.split(head_dim, dim=-1))

    layer.rearrange_mixed_qkv.side_effect = rearrange
    context = ForwardContext(
        no_compile_layers={layer.prefix: layer},
        attn_metadata={layer.prefix: metadata},
        slot_mapping={},
    )
    output = torch.empty(3, 1, head_dim)
    with (
        override_forward_context(context),
        patch("vllm_ascend.ops.gdn.get_pcp_group", return_value=SimpleNamespace(world_size=1)),
        patch("vllm_ascend.ops.gdn._get_fla_gdn_prefill_op", return_value=None),
        patch(
            "vllm_ascend.ops.gdn.DeviceOperator.fused_gdn_gating",
            return_value=(torch.zeros(1, 3, 1), torch.ones(1, 3, 1)),
        ),
        patch("vllm_ascend.ops.gdn.causal_conv1d_fn", side_effect=conv) as conv_mock,
        patch(
            "vllm_ascend.ops.gdn.causal_conv1d_update",
            side_effect=lambda x, *args, **kwargs: x.clone(),
        ) as update_mock,
        patch("vllm_ascend.ops.gdn.l2norm_fwd", side_effect=lambda x: x),
        patch("vllm_ascend.ops.gdn.clear_ssm_states"),
        patch(
            "vllm_ascend.ops.gdn.chunk_gated_delta_rule",
            side_effect=lambda **kw: (kw["v"].clone(), kw["initial_state"]),
        ),
        patch(
            "vllm_ascend.ops.gdn.recurrent_gated_delta_rule",
            side_effect=lambda q, k, v, *args, **kw: v.clone(),
        ),
        patch("vllm_ascend.ops.gdn.maybe_save_kv_layer_to_connector"),
    ):
        AscendGatedDeltaNetAttention._forward_core(layer, packed, torch.zeros(3, 1), torch.zeros(3, 1), output)
    conv_mock.assert_called_once()
    if mixed:
        update_mock.assert_called_once()
        torch.testing.assert_close(update_mock.call_args.args[0], packed[:1])
    else:
        update_mock.assert_not_called()
    torch.testing.assert_close(output, packed[:, 2 * head_dim :].reshape(3, 1, head_dim))
