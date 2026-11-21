# SPDX-License-Identifier: Apache-2.0
# Adapted from upstream commit 722a244be988 (PR #16292).

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.forward_context import ForwardContext, override_forward_context

from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.ops.gdn_attn_builder import GDNSpecCausalConv1dMetadata, GDNSpecDecodeMetadata

from .test_gdn_mixed_batch import _make_layer, _make_mixed_metadata


@pytest.mark.parametrize(
    ("window", "lengths", "accepted"),
    [
        (1, [1, 1], [1, 1]),
        (8, [8, 8], [1, 8]),
        (9, [9, 9], [1, 9]),
        (16, [16, 16], [1, 16]),
        (16, [16, 4, 9], [1, 4, 9]),
        (16, [1, 0, 4], [16, 1, 8]),
    ],
    ids=["single", "native-boundary", "wide-boundary", "wide", "ragged", "shrinking-with-padding"],
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_spec_recurrent_dispatch_preserves_sequence_state(window, lengths, accepted, strided, dtype):
    # Exercise the production forward path, replacing only device operators.
    # These assertions check dispatch/metadata, not device-kernel numerics.
    layer = _make_layer()
    layer.num_spec = window - 1
    metadata = _make_mixed_metadata()
    num_seqs, num_tokens = len(lengths), sum(lengths)
    state_storage = torch.arange(1, 2 * num_seqs * window + 1, dtype=torch.int32).view(num_seqs, 2 * window)
    state_indices = state_storage[:, ::2]
    if not strided:
        state_indices = state_indices.contiguous()
    for row, length in enumerate(lengths):
        if not length:
            state_indices[row].fill_(-1)
    query_start_loc = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    num_accepted_tokens = torch.tensor(accepted, dtype=torch.int32)
    actual_lengths = torch.tensor([0, *lengths], dtype=torch.int32)
    layer.kv_cache = (layer.kv_cache[0], torch.zeros(2 * num_seqs * window + 1, 1, 2, 2))
    metadata.num_prefills = metadata.num_prefill_tokens = 0
    metadata.num_decodes = metadata.num_decode_tokens = 0
    metadata.num_actual_tokens = metadata.num_spec_decode_tokens = num_tokens
    metadata.num_spec_decodes = num_seqs
    metadata.spec_sequence_masks = torch.ones(num_seqs, dtype=torch.bool)
    metadata.spec_state_indices_tensor = state_indices
    metadata.spec_decode_metadata = GDNSpecDecodeMetadata(
        spec_causal_conv1d=GDNSpecCausalConv1dMetadata(
            query_start_loc=query_start_loc,
            cache_indices=state_indices,
            num_accepted_tokens=num_accepted_tokens,
        ),
        actual_seq_lengths=actual_lengths,
    )
    expect_triton = window > 8

    def ascendc(q, k, v, state, **kwargs):
        assert not expect_triton
        assert state is layer.kv_cache[1]
        assert kwargs["actual_seq_lengths"] is actual_lengths
        expected = torch.cat([state_indices[i, :length] for i, length in enumerate(lengths)])
        torch.testing.assert_close(kwargs["ssm_state_indices"], expected)
        torch.testing.assert_close(kwargs["num_accepted_tokens"], num_accepted_tokens)
        return v + 3

    def triton(q, k, v, g, beta, **kwargs):
        assert expect_triton
        assert kwargs["cu_seqlens"] is query_start_loc
        indices = kwargs["ssm_state_indices"]
        assert indices.ndim == 2 and indices.is_contiguous()
        torch.testing.assert_close(indices, state_indices)
        if not strided:
            assert indices is state_indices  # No copy on the normal builder path.
        assert kwargs["num_accepted_tokens"] is num_accepted_tokens
        assert kwargs["initial_state"] is layer.kv_cache[1]
        assert kwargs["initial_state"].dtype == torch.float32
        assert kwargs["inplace_final_state"] is True
        assert kwargs["scale"] == 2**-0.5
        assert q.shape == k.shape == v.shape == (1, num_tokens, 1, 2)
        assert q.dtype == k.dtype == v.dtype == dtype
        return v + 7, kwargs["initial_state"]

    packed = torch.arange(num_tokens * 6, dtype=torch.float32).view(num_tokens, 6).to(dtype)
    output = torch.full((num_tokens + 2, 1, 2), -100, dtype=dtype)
    context = ForwardContext(
        no_compile_layers={layer.prefix: layer},
        attn_metadata={layer.prefix: metadata},
        slot_mapping={},
    )
    with (
        override_forward_context(context),
        patch("vllm_ascend.ops.gdn.get_pcp_group", return_value=SimpleNamespace(world_size=1)),
        patch("vllm_ascend.ops.gdn.wait_for_kv_layer_from_connector"),
        patch("vllm_ascend.ops.gdn.record_attention_compute_start"),
        patch(
            "vllm_ascend.ops.gdn.DeviceOperator.fused_gdn_gating",
            return_value=(torch.zeros(1, num_tokens, 1), torch.ones(1, num_tokens, 1)),
        ),
        patch("vllm_ascend.ops.gdn.l2norm_fwd", side_effect=lambda x: torch.nn.functional.normalize(x, dim=-1)),
        patch("vllm_ascend.ops.gdn.causal_conv1d_update", side_effect=lambda x, *a, **k: x.clone()),
        patch("vllm_ascend.ops.gdn.recurrent_gated_delta_rule", side_effect=ascendc) as native_mock,
        patch("vllm_ascend.ops.gdn.fused_recurrent_gated_delta_rule", side_effect=triton, create=True) as triton_mock,
        patch("vllm_ascend.ops.gdn.maybe_save_kv_layer_to_connector"),
    ):
        AscendGatedDeltaNetAttention._forward_core(
            layer, packed, torch.zeros(num_tokens, 1), torch.zeros(num_tokens, 1), output
        )
    assert native_mock.call_count == int(not expect_triton)
    assert triton_mock.call_count == int(expect_triton)
    torch.testing.assert_close(output[:num_tokens], packed[:, 4:].view(num_tokens, 1, 2) + (7 if expect_triton else 3))
    assert torch.all(output[num_tokens:] == -100)
