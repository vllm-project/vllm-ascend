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

from vllm_ascend.patch.worker import patch_gdn_prefill_warmup as warmup


class _WarmupLayer:
    prefix = "layers.0.linear_attn"
    num_k_heads = 2
    num_v_heads = 4
    tp_size = 1
    head_k_dim = 2
    head_v_dim = 3

    def __init__(self):
        self.A_log = torch.zeros(self.num_v_heads, dtype=torch.float32)
        self.dt_bias = torch.zeros(self.num_v_heads, dtype=torch.float32)

    def get_state_dtype(self):
        return torch.float16, torch.float32

    def rearrange_mixed_qkv(self, mixed_qkv: torch.Tensor):
        tokens = mixed_qkv.shape[0]
        query = torch.zeros(1, tokens, self.num_k_heads, self.head_k_dim)
        key = torch.zeros_like(query)
        value = torch.zeros(1, tokens, self.num_v_heads, self.head_v_dim)
        return query, key, value


@pytest.fixture(autouse=True)
def clear_warmup_signatures():
    warmup._GDN_PREFILL_WARMUP_SIGNATURES.clear()
    yield
    warmup._GDN_PREFILL_WARMUP_SIGNATURES.clear()


def _gating_outputs():
    return (
        torch.zeros(1, warmup._GDN_PREFILL_WARMUP_TOKENS, 4),
        torch.zeros(1, warmup._GDN_PREFILL_WARMUP_TOKENS, 4),
    )


def test_fallback_prefill_warmup_runs_once_per_signature():
    first_layer = _WarmupLayer()
    second_layer = _WarmupLayer()
    mixed_qkv = torch.zeros(8, 20, dtype=torch.bfloat16)
    fallback = Mock(return_value=(Mock(), Mock()))

    with (
        patch.object(
            warmup.AscendGatedDeltaNetAttention,
            "_probe_fused_chunk",
            return_value=False,
        ),
        patch.object(
            warmup.DeviceOperator,
            "fused_gdn_gating",
            return_value=_gating_outputs(),
        ),
        patch.object(warmup, "get_pcp_group", return_value=SimpleNamespace(world_size=1)),
        patch.object(warmup, "chunk_gated_delta_rule", fallback),
    ):
        warmup._warmup_gdn_prefill_kernels(first_layer, mixed_qkv, 0)
        warmup._warmup_gdn_prefill_kernels(second_layer, mixed_qkv, 0)

    fallback.assert_called_once()
    kwargs = fallback.call_args.kwargs
    assert kwargs["q"].shape == (1, 64, 2, 2)
    assert kwargs["k"].shape == (1, 64, 2, 2)
    assert kwargs["v"].shape == (1, 64, 4, 3)
    assert kwargs["initial_state"].shape == (1, 4, 2, 3)
    assert kwargs["initial_state"].dtype == torch.float32
    assert kwargs["cu_seqlens"].tolist() == [0, 64]
    assert kwargs["use_qk_l2norm_in_kernel"] is True
    assert len(warmup._GDN_PREFILL_WARMUP_SIGNATURES) == 1


def test_fused_prefill_warmup_uses_live_state_layout():
    layer = _WarmupLayer()
    mixed_qkv = torch.zeros(8, 20, dtype=torch.bfloat16)
    fused = Mock(return_value=(Mock(), Mock()))
    fallback = Mock()

    with (
        patch.object(
            warmup.AscendGatedDeltaNetAttention,
            "_probe_fused_chunk",
            return_value=True,
        ),
        patch.object(
            warmup.AscendGatedDeltaNetAttention,
            "_chunk_gated_delta_rule_fused",
            fused,
        ),
        patch.object(
            warmup.DeviceOperator,
            "fused_gdn_gating",
            return_value=_gating_outputs(),
        ),
        patch.object(warmup, "get_pcp_group", return_value=SimpleNamespace(world_size=1)),
        patch.object(warmup, "chunk_gated_delta_rule", fallback),
    ):
        warmup._warmup_gdn_prefill_kernels(layer, mixed_qkv, 0)

    fused.assert_called_once()
    assert fused.call_args.kwargs["initial_state"].shape == (1, 4, 3, 2)
    fallback.assert_not_called()


def test_failed_prefill_warmup_is_retryable():
    layer = _WarmupLayer()
    mixed_qkv = torch.zeros(8, 20, dtype=torch.bfloat16)
    fallback = Mock(side_effect=[RuntimeError("warmup failed"), (Mock(), Mock())])

    with (
        patch.object(
            warmup.AscendGatedDeltaNetAttention,
            "_probe_fused_chunk",
            return_value=False,
        ),
        patch.object(
            warmup.DeviceOperator,
            "fused_gdn_gating",
            return_value=_gating_outputs(),
        ),
        patch.object(warmup, "get_pcp_group", return_value=SimpleNamespace(world_size=1)),
        patch.object(warmup, "chunk_gated_delta_rule", fallback),
    ):
        warmup._warmup_gdn_prefill_kernels(layer, mixed_qkv, 0)
        assert not warmup._GDN_PREFILL_WARMUP_SIGNATURES
        warmup._warmup_gdn_prefill_kernels(layer, mixed_qkv, 0)

    assert fallback.call_count == 2
    assert len(warmup._GDN_PREFILL_WARMUP_SIGNATURES) == 1


def test_profile_forward_invokes_prefill_warmup_instead_of_runtime_core():
    layer = _WarmupLayer()
    mixed_qkv = torch.zeros(8, 20, dtype=torch.bfloat16)
    b = torch.zeros(8, 4)
    a = torch.zeros(8, 4)
    output = torch.zeros(8, 4, 3)

    with (
        patch.object(
            warmup,
            "get_forward_context",
            return_value=SimpleNamespace(attn_metadata=None),
        ),
        patch.object(warmup, "_warmup_gdn_prefill_kernels") as run_warmup,
        patch.object(warmup, "_ASCEND_GDN_FORWARD_CORE") as runtime_core,
    ):
        result = warmup._forward_core_with_prefill_warmup(
            layer,
            mixed_qkv,
            b,
            a,
            output,
        )

    assert result is None
    run_warmup.assert_called_once_with(layer, mixed_qkv, 0)
    runtime_core.assert_not_called()


def test_runtime_forward_delegates_when_metadata_is_available():
    layer = _WarmupLayer()
    mixed_qkv = torch.zeros(8, 20, dtype=torch.bfloat16)
    b = torch.zeros(8, 4)
    a = torch.zeros(8, 4)
    output = torch.zeros(8, 4, 3)
    expected = object()

    with (
        patch.object(
            warmup,
            "get_forward_context",
            return_value=SimpleNamespace(attn_metadata={layer.prefix: object()}),
        ),
        patch.object(warmup, "_warmup_gdn_prefill_kernels") as run_warmup,
        patch.object(
            warmup,
            "_ASCEND_GDN_FORWARD_CORE",
            return_value=expected,
        ) as runtime_core,
    ):
        result = warmup._forward_core_with_prefill_warmup(
            layer,
            mixed_qkv,
            b,
            a,
            output,
        )

    assert result is expected
    run_warmup.assert_not_called()
    runtime_core.assert_called_once_with(layer, mixed_qkv, b, a, output)
