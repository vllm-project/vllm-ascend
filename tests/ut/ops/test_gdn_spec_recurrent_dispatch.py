# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.forward_context import ForwardContext, override_forward_context

from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.ops.gdn_attn_builder import (
    GDNSpecCausalConv1dMetadata,
    GDNSpecDecodeMetadata,
)

from .test_gdn_mixed_batch import _make_layer, _make_mixed_metadata


@pytest.mark.parametrize(("window", "expect_upstream"), [(8, False), (9, True), (16, True)])
def test_spec_recurrent_dispatches_by_window_width(window: int, expect_upstream: bool) -> None:
    layer = _make_layer()
    metadata = _make_mixed_metadata()
    num_seqs = 2
    num_tokens = num_seqs * window
    state_indices = torch.arange(num_tokens, dtype=torch.int64).view(num_seqs, window)
    query_start_loc = torch.arange(0, num_tokens + 1, window, dtype=torch.int32)
    num_accepted_tokens = torch.ones(num_seqs, dtype=torch.int32)
    metadata.num_prefills = 0
    metadata.num_prefill_tokens = 0
    metadata.num_decodes = 0
    metadata.num_decode_tokens = 0
    metadata.num_actual_tokens = num_tokens
    metadata.num_spec_decodes = num_seqs
    metadata.num_spec_decode_tokens = num_tokens
    metadata.spec_sequence_masks = torch.ones(num_seqs, dtype=torch.bool)
    metadata.spec_state_indices_tensor = state_indices
    metadata.spec_decode_metadata = GDNSpecDecodeMetadata(
        spec_causal_conv1d=GDNSpecCausalConv1dMetadata(
            query_start_loc=query_start_loc,
            cache_indices=state_indices[:, 0],
            num_accepted_tokens=num_accepted_tokens,
        ),
        actual_seq_lengths=torch.tensor([0] * (num_seqs + 1), dtype=torch.int32),
    )

    def ascendc(q, k, v, state, **kwargs):
        assert not expect_upstream
        return v.clone()

    def triton(q, k, v, g, beta, **kwargs):
        assert expect_upstream
        assert kwargs["cu_seqlens"] is query_start_loc
        assert kwargs["ssm_state_indices"] is state_indices
        assert kwargs["num_accepted_tokens"] is num_accepted_tokens
        return v.clone(), kwargs["initial_state"]

    packed = torch.zeros(num_tokens, 6)
    output = torch.empty(num_tokens, 1, 2)
    context = ForwardContext(
        no_compile_layers={layer.prefix: layer},
        attn_metadata={layer.prefix: metadata},
        slot_mapping={},
    )
    with (
        override_forward_context(context),
        patch("vllm_ascend.ops.gdn.get_pcp_group", return_value=SimpleNamespace(world_size=1)),
        patch(
            "vllm_ascend.ops.gdn.DeviceOperator.fused_gdn_gating",
            return_value=(torch.zeros(1, num_tokens, 1), torch.ones(1, num_tokens, 1)),
        ),
        patch("vllm_ascend.ops.gdn.l2norm_fwd", side_effect=lambda x: x),
        patch("vllm_ascend.ops.gdn.causal_conv1d_update", side_effect=lambda x, *a, **k: x.clone()),
        patch("vllm_ascend.ops.gdn.recurrent_gated_delta_rule", side_effect=ascendc) as ascendc_mock,
        patch("vllm_ascend.ops.gdn.fused_recurrent_gated_delta_rule", side_effect=triton) as triton_mock,
        patch("vllm_ascend.ops.gdn.maybe_save_kv_layer_to_connector"),
    ):
        AscendGatedDeltaNetAttention._forward_core(
            layer, packed, torch.zeros(num_tokens, 1), torch.zeros(num_tokens, 1), output
        )

    assert ascendc_mock.called is not expect_upstream
    assert triton_mock.called is expect_upstream
