# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.ops.gdn_attn_builder import (
    GDNCausalConv1dMetadata,
    GDNDecodeMetadata,
    GDNPrefillMetadata,
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
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
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
    layer.rearrange_mixed_qkv = Mock(
        name="rearrange_mixed_qkv",
        side_effect=rearrange_mixed_qkv,
    )
    return layer


def test_mixed_non_spec_reuses_rearranged_qkv() -> None:
    layer = _make_layer()
    metadata = _make_mixed_metadata()

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
        output: torch.Tensor,
        input_tensor: torch.Tensor,
        conv_weights: torch.Tensor,
        **kwargs,
    ) -> None:
        del conv_weights, kwargs
        output.copy_(input_tensor)

    def recurrent_gated_delta_rule(**kwargs) -> torch.Tensor:
        recurrent_calls.append(kwargs)
        # Returning V makes the merged output easy to validate.
        return kwargs["value"].clone()

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
            "vllm_ascend.ops.gdn.DeviceOperator.fused_gdn_gating",
            return_value=gating,
        ),
        patch("vllm_ascend.ops.gdn.l2norm_fwd", side_effect=lambda x: x),
        patch("vllm_ascend.ops.gdn.clear_ssm_states"),
        patch(
            "vllm_ascend.ops.gdn.chunk_gated_delta_rule",
            side_effect=chunk_gated_delta_rule,
        ) as chunk_mock,
        patch("vllm_ascend.ops.gdn.maybe_save_kv_layer_to_connector"),
        patch.object(
            torch.ops._C_ascend,
            "npu_causal_conv1d_custom",
            side_effect=causal_conv1d,
            create=True,
        ),
        patch.object(
            torch.ops._C_ascend,
            "npu_recurrent_gated_delta_rule",
            side_effect=recurrent_gated_delta_rule,
            create=True,
        ),
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

    # The spec and non-spec branches both invoke rearrange_mixed_qkv.
    # In this non-spec mixed batch, the spec invocation receives None.
    # Verify that only one invocation processes an actual tensor, so the
    # decode prefix is not rearranged for a second time.
    non_none_calls = [call for call in layer.rearrange_mixed_qkv.call_args_list if call.args[0] is not None]
    assert len(non_none_calls) == 1

    torch.testing.assert_close(
        non_none_calls[0].args[0],
        mixed_qkv,
    )
    prefill_call = chunk_mock.call_args.kwargs
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
