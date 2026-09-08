# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.forward_context import ForwardContext, override_forward_context

import vllm_ascend.envs as envs_ascend
from tests.ut.ops.test_gdn_layerwise_kv import _GDNForwardWrapper
from vllm_ascend.ops import gdn as ascend_gdn


def _padded_cpu_state() -> torch.Tensor:
    state_shape = (7, 1, 2, 2)
    state_stride = (10, 4, 2, 1)
    backing = torch.zeros(4 + state_shape[0] * state_stride[0] + 3)
    return torch.as_strided(
        backing,
        state_shape,
        state_stride,
        storage_offset=4,
    )


def _causal_conv1d_passthrough(
    output: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    **kwargs,
) -> None:
    del conv_weights, kwargs
    output.copy_(mixed_qkv)


def _run_core(
    model: _GDNForwardWrapper,
    metadata: SimpleNamespace,
    num_tokens: int,
) -> torch.Tensor:
    forward_context = ForwardContext(
        no_compile_layers={model.prefix: model},
        attn_metadata={model.prefix: metadata},
        slot_mapping={},
    )
    mixed_qkv = torch.arange(num_tokens * 2, dtype=torch.float32).reshape(num_tokens, 2)
    a = torch.zeros(num_tokens, 1)
    b = torch.zeros(num_tokens, 1)
    output = torch.zeros(num_tokens, 1, 2)
    with override_forward_context(forward_context):
        model._forward_core(mixed_qkv, b, a, output)
    return output


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("ascendc", "ascendc"),
        (" ASCENDC ", "ascendc"),
        ("triton-strided", "triton-strided"),
        ("triton", "triton-strided"),
    ],
)
def test_decode_backend_selection_normalizes_supported_values(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    expected: str,
):
    monkeypatch.setattr(
        envs_ascend,
        "VLLM_ASCEND_GDN_DECODE_BACKEND",
        configured,
    )
    monkeypatch.setattr(ascend_gdn, "HAS_TRITON", True)
    assert ascend_gdn._get_gdn_decode_backend() == expected


def test_decode_backend_selection_rejects_unknown_value(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        envs_ascend,
        "VLLM_ASCEND_GDN_DECODE_BACKEND",
        "automatic",
    )
    with pytest.raises(ValueError, match="triton-strided"):
        ascend_gdn._get_gdn_decode_backend()


def test_decode_backend_selection_does_not_silently_fallback_without_triton(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        envs_ascend,
        "VLLM_ASCEND_GDN_DECODE_BACKEND",
        "triton-strided",
    )
    monkeypatch.setattr(ascend_gdn, "HAS_TRITON", False)
    with pytest.raises(RuntimeError, match="requires an active Triton backend"):
        ascend_gdn._get_gdn_decode_backend()


def test_triton_state_layout_rejects_inner_padding():
    backing = torch.zeros(32)
    state = torch.as_strided(
        backing,
        size=(2, 1, 2, 2),
        stride=(12, 6, 2, 1),
    )
    with pytest.raises(ValueError, match="padded leading stride only"):
        ascend_gdn._validate_triton_state_layout(state)


def test_plain_decode_uses_packed_triton_with_strided_state():
    model = _GDNForwardWrapper()
    model.ssm_state = _padded_cpu_state()
    # The final NULL block is a full-graph padding row. It must stay paired with
    # the padded token passed to packed decode rather than being sliced away.
    state_indices = torch.tensor([1, 6, 0], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 2, 2], dtype=torch.int32)
    metadata = SimpleNamespace(
        spec_sequence_masks=None,
        spec_token_indx=None,
        non_spec_token_indx=None,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=state_indices,
        num_actual_tokens=3,
        num_prefills=0,
        num_decodes=2,
        num_decode_tokens=2,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        non_spec_decode_metadata=SimpleNamespace(
            causal_conv1d=SimpleNamespace(
                query_start_loc=query_start_loc,
                cache_indices=state_indices,
            ),
            actual_seq_lengths=torch.tensor([0, 1, 1], dtype=torch.int32),
        ),
    )
    packed = Mock()

    def packed_decode(**kwargs):
        assert kwargs["state"] is model.ssm_state
        assert kwargs["state"].stride(0) > kwargs["state"][0].numel()
        assert kwargs["state"].storage_offset() != 0
        assert kwargs["ssm_state_indices"] is state_indices
        kwargs["out"].fill_(7)
        kwargs["out"][kwargs["ssm_state_indices"] <= 0] = 0
        return kwargs["out"].transpose(0, 1)

    packed.side_effect = packed_decode
    generic = Mock(side_effect=AssertionError("plain decode must use packed Triton"))
    gating = Mock(side_effect=AssertionError("packed decode must fuse gating"))

    with (
        patch.object(ascend_gdn, "GDNAttentionMetadata", SimpleNamespace),
        patch.object(ascend_gdn, "_GDN_DECODE_BACKEND", "triton-strided"),
        patch.object(ascend_gdn, "_run_triton_packed_decode", packed),
        patch.object(ascend_gdn, "_run_triton_recurrent", generic),
        patch.object(ascend_gdn.DeviceOperator, "fused_gdn_gating", gating),
        patch.object(
            torch.ops._C_ascend,
            "npu_causal_conv1d_custom",
            side_effect=_causal_conv1d_passthrough,
            create=True,
        ),
        patch.object(ascend_gdn, "maybe_save_kv_layer_to_connector"),
    ):
        output = _run_core(model, metadata, num_tokens=3)

    packed.assert_called_once()
    generic.assert_not_called()
    gating.assert_not_called()
    torch.testing.assert_close(output[:2], torch.full_like(output[:2], 7))
    torch.testing.assert_close(output[2], torch.zeros_like(output[2]))


def test_spec_decode_uses_generic_triton_and_preserves_state_table():
    model = _GDNForwardWrapper()
    model.ssm_state = _padded_cpu_state()
    # The last request is graph padding: its repeated cu_seqlen and all-NULL
    # state row must reach the generic kernel unchanged.
    spec_query_start_loc = torch.tensor([0, 2, 4, 4], dtype=torch.int32)
    spec_state_indices = torch.tensor(
        [[1, 2], [5, 6], [0, 0]],
        dtype=torch.int32,
    )
    accepted_tokens = torch.tensor([1, 2, 1], dtype=torch.int32)
    metadata = SimpleNamespace(
        spec_sequence_masks=torch.tensor([True, True, False]),
        spec_token_indx=torch.arange(4),
        non_spec_token_indx=torch.empty(0, dtype=torch.int64),
        spec_state_indices_tensor=spec_state_indices,
        non_spec_state_indices_tensor=torch.empty(0, dtype=torch.int32),
        spec_query_start_loc=spec_query_start_loc,
        num_actual_tokens=4,
        num_prefills=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=2,
        num_spec_decode_tokens=4,
        spec_decode_metadata=SimpleNamespace(
            spec_causal_conv1d=SimpleNamespace(
                query_start_loc=spec_query_start_loc,
                cache_indices=spec_state_indices,
                num_accepted_tokens=accepted_tokens,
            ),
            actual_seq_lengths=torch.tensor([0, 2, 2], dtype=torch.int32),
        ),
    )
    packed = Mock(side_effect=AssertionError("spec decode must not use packed Triton"))
    generic = Mock()

    def generic_recurrent(**kwargs):
        assert kwargs["state"] is model.ssm_state
        assert kwargs["state"].stride(0) > kwargs["state"][0].numel()
        assert kwargs["state"].storage_offset() != 0
        assert kwargs["cu_seqlens"] is spec_query_start_loc
        assert kwargs["ssm_state_indices"] is spec_state_indices
        torch.testing.assert_close(kwargs["num_accepted_tokens"], accepted_tokens)
        return torch.full_like(kwargs["value"], 9)

    generic.side_effect = generic_recurrent
    gating = (
        torch.zeros(1, 4, 1),
        torch.ones(1, 4, 1),
    )

    with (
        patch.object(ascend_gdn, "GDNAttentionMetadata", SimpleNamespace),
        patch.object(ascend_gdn, "_GDN_DECODE_BACKEND", "triton-strided"),
        patch.object(ascend_gdn, "_run_triton_packed_decode", packed),
        patch.object(ascend_gdn, "_run_triton_recurrent", generic),
        patch.object(ascend_gdn.DeviceOperator, "fused_gdn_gating", return_value=gating),
        patch.object(ascend_gdn, "l2norm_fwd", side_effect=lambda tensor: tensor),
        patch.object(
            torch.ops._C_ascend,
            "npu_causal_conv1d_custom",
            side_effect=_causal_conv1d_passthrough,
            create=True,
        ),
        patch.object(ascend_gdn, "maybe_save_kv_layer_to_connector"),
    ):
        output = _run_core(model, metadata, num_tokens=4)

    packed.assert_not_called()
    generic.assert_called_once()
    torch.testing.assert_close(output, torch.full_like(output, 9))
