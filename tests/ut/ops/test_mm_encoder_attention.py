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

from unittest import mock

import pytest
import torch
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention


def _make_layer(attn_backend: AttentionBackendEnum) -> AscendMMEncoderAttention:
    layer = AscendMMEncoderAttention.__new__(AscendMMEncoderAttention)
    layer.num_heads = 4
    layer.num_kv_heads = 2
    layer.head_size = 80
    layer.enable_pad = True
    layer.scale_value = layer.head_size**-0.5
    layer.attn_backend = attn_backend
    return layer


def test_mm_encoder_attention_forward_oot_respects_torch_sdpa_backend():
    layer = _make_layer(AttentionBackendEnum.TORCH_SDPA)
    query = torch.randn(2, 3, layer.num_heads, layer.head_size)
    key = torch.randn(2, 3, layer.num_kv_heads, layer.head_size)
    value = torch.randn(2, 3, layer.num_kv_heads, layer.head_size)
    expected = torch.randn_like(query)

    with (
        mock.patch.object(
            AscendMMEncoderAttention,
            "_forward_sdpa",
            autospec=True,
            return_value=expected,
        ) as mock_sdpa,
        mock.patch(
            "vllm_ascend.ops.mm_encoder_attention.DeviceOperator.npu_flash_attention"
        ) as mock_npu_flash_attention,
    ):
        output = layer.forward_oot(query, key, value)

    assert output is expected
    mock_sdpa.assert_called_once_with(layer, query, key, value, None)
    mock_npu_flash_attention.assert_not_called()


def test_mm_encoder_attention_forward_oot_uses_npu_flash_attention_for_flash_backend():
    layer = _make_layer(AttentionBackendEnum.FLASH_ATTN)
    bsz, q_len, kv_len = 2, 3, 3
    query = torch.randn(bsz, q_len, layer.num_heads, layer.head_size)
    key = torch.randn(bsz, kv_len, layer.num_kv_heads, layer.head_size)
    value = torch.randn(bsz, kv_len, layer.num_kv_heads, layer.head_size)
    capture = {}

    def fake_npu_flash_attention(*, query, key, value, seq_lens_cpu, head_num, scale_value, num_kv_heads):
        capture["query_shape"] = query.shape
        capture["key_shape"] = key.shape
        capture["value_shape"] = value.shape
        capture["seq_lens_cpu"] = seq_lens_cpu
        capture["head_num"] = head_num
        capture["scale_value"] = scale_value
        capture["num_kv_heads"] = num_kv_heads
        return query + 1.0

    with mock.patch(
        "vllm_ascend.ops.mm_encoder_attention.DeviceOperator.npu_flash_attention",
        side_effect=fake_npu_flash_attention,
    ):
        output = layer.forward_oot(query, key, value)

    assert capture["query_shape"] == (bsz * q_len, layer.num_heads, 128)
    assert capture["key_shape"] == (bsz * kv_len, layer.num_heads, 128)
    assert capture["value_shape"] == (bsz * kv_len, layer.num_heads, 128)
    torch.testing.assert_close(
        capture["seq_lens_cpu"],
        torch.tensor([q_len, q_len], dtype=torch.int32),
    )
    assert capture["head_num"] == layer.num_heads
    assert capture["scale_value"] == layer.scale_value
    assert capture["num_kv_heads"] == layer.num_kv_heads
    torch.testing.assert_close(output, query + 1.0)


def test_mm_encoder_attention_forward_oot_rejects_unsupported_backend():
    layer = _make_layer(AttentionBackendEnum.TRITON_ATTN)
    query = torch.randn(2, 3, layer.num_heads, layer.head_size)
    key = torch.randn(2, 3, layer.num_kv_heads, layer.head_size)
    value = torch.randn(2, 3, layer.num_kv_heads, layer.head_size)

    with (
        mock.patch(
            "vllm_ascend.ops.mm_encoder_attention.DeviceOperator.npu_flash_attention"
        ) as mock_npu_flash_attention,
        pytest.raises(ValueError, match="Unsupported multi-modal encoder attention backend"),
    ):
        layer.forward_oot(query, key, value)

    mock_npu_flash_attention.assert_not_called()
