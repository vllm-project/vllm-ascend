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

"""Unit tests for the 310P encoder-only (pooling) attention path.

EmbeddingGemma2 routes ENCODER_ONLY attention to a native torch
implementation in AscendAttentionBackendImpl310 to bypass the ATB
SelfAttentionOperation allocator defect on Atlas 300I Duo. These tests
verify the block-diagonal keep-mask (batched pooling sequences must not
attend across each other), the bidirectional sliding-window band mask,
and the GQA head expansion of the torch path.
"""

from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm.v1.attention.backend import AttentionType
from vllm_ascend._310p.attention.attention_v1 import (
    AscendAttentionBackendImpl310,
    AscendAttentionState,
)


def _make_impl(num_heads, num_kv_heads, head_size, hf_text_config=None):
    """Build a minimal AscendAttentionBackendImpl310 for encoder tests."""
    cfg = MagicMock()
    cfg.parallel_config.prefill_context_parallel_size = 1
    with patch("vllm_ascend.attention.attention_v1.get_current_vllm_config", return_value=cfg), patch(
        "vllm_ascend.attention.attention_v1.needs_layer_aware_fia_graph_replay", return_value=False
    ):
        impl = AscendAttentionBackendImpl310(
            num_heads=num_heads,
            head_size=head_size,
            scale=1.0,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=AttentionType.ENCODER_ONLY,
            kv_sharing_target_layer_name=None,
        )
    if hf_text_config is not None:
        impl.vllm_config = cfg
        impl.vllm_config.model_config.hf_config.text_config = hf_text_config
    return impl


def _plain_text_config():
    """No sliding window: every encoder layer attends fully."""
    text_cfg = MagicMock()
    text_cfg.sliding_window = None
    text_cfg.layer_types = []
    text_cfg.per_layer_config = []
    return text_cfg


def _make_metadata(seq_lens):
    metadata = MagicMock()
    metadata.attn_state = AscendAttentionState.PrefillNoCache
    metadata.seq_lens = torch.tensor(seq_lens, dtype=torch.int64)
    return metadata


class TestEncoderBandMask310(TestBase):
    def test_full_attention_layer_returns_none(self):
        """Head size matching a full-attention layer gets no band mask."""
        text_cfg = MagicMock()
        text_cfg.sliding_window = 512
        text_cfg.layer_types = ["sliding_attention", "full_attention"]
        text_cfg.per_layer_config = [MagicMock(head_dim=256), MagicMock(head_dim=512)]
        impl = _make_impl(8, 8, 512, hf_text_config=text_cfg)
        self.assertIsNone(impl._get_encoder_band_mask(64, 512, "cpu"))

    def test_sliding_layer_returns_band_mask(self):
        """Sliding layers get a bidirectional band mask of ±window."""
        text_cfg = MagicMock()
        text_cfg.sliding_window = 4
        text_cfg.layer_types = ["sliding_attention"]
        text_cfg.per_layer_config = [MagicMock(head_dim=256)]
        impl = _make_impl(8, 8, 256, hf_text_config=text_cfg)
        mask = impl._get_encoder_band_mask(10, 256, "cpu")
        self.assertEqual(mask.shape, (10, 10))
        # Within the window: finite; outside: -inf.
        self.assertTrue(torch.isfinite(mask[0, 4]))
        self.assertTrue(torch.isinf(mask[0, 5]))
        # Diagonal band is symmetric and self-attention is always allowed.
        self.assertTrue(torch.isfinite(mask[7, 4]))  # |7-4|=3 <= window
        self.assertTrue(torch.isinf(mask[7, 2]))  # |7-2|=5 > window
        self.assertTrue(torch.all(torch.isfinite(mask.diagonal())))

    def test_mask_is_cached_across_calls(self):
        """The mask is cached and sliced for smaller subsequent calls."""
        text_cfg = MagicMock()
        text_cfg.sliding_window = 4
        text_cfg.layer_types = ["sliding_attention"]
        text_cfg.per_layer_config = [MagicMock(head_dim=256)]
        impl = _make_impl(8, 8, 256, hf_text_config=text_cfg)
        first = impl._get_encoder_band_mask(600, 256, "cpu")
        cached = impl._enc_band_mask
        second = impl._get_encoder_band_mask(16, 256, "cpu")
        self.assertEqual(first.shape, (600, 600))
        self.assertEqual(second.shape, (16, 16))
        self.assertIs(cached, impl._enc_band_mask)

    def test_no_sliding_window_returns_none(self):
        """Without a sliding window in the config, no band mask is built."""
        text_cfg = MagicMock()
        text_cfg.sliding_window = None
        impl = _make_impl(8, 8, 256, hf_text_config=text_cfg)
        self.assertIsNone(impl._get_encoder_band_mask(64, 256, "cpu"))


class TestEncoderAttention310(TestBase):
    def test_single_sequence_reference(self):
        """A single sequence matches a plain softmax(bmm) reference."""
        impl = _make_impl(2, 2, 8, hf_text_config=_plain_text_config())
        torch.manual_seed(0)
        query = torch.randn(6, 2, 8, dtype=torch.float32)
        key = torch.randn(6, 2, 8, dtype=torch.float32)
        value = torch.randn(6, 2, 8, dtype=torch.float32)
        output = torch.empty_like(query)
        metadata = _make_metadata([6])
        result = impl._forward_encoder_attention(query, key, value, metadata, output)
        scores = torch.bmm(query.permute(1, 0, 2), key.permute(1, 0, 2).transpose(1, 2))
        ref = torch.bmm(torch.softmax(scores, dim=-1), value.permute(1, 0, 2)).permute(1, 0, 2)
        self.assertTrue(torch.allclose(result, ref, atol=1e-5))
        self.assertTrue(torch.allclose(output[:6], result, atol=1e-5))

    def test_batched_sequences_do_not_attend_across(self):
        """Block-diagonal keep-mask: sequence 0 tokens never see sequence 1."""
        impl = _make_impl(1, 1, 4, hf_text_config=_plain_text_config())
        torch.manual_seed(0)
        lens = [3, 5]
        total = sum(lens)
        query = torch.randn(total, 1, 4, dtype=torch.float32)
        key = torch.randn(total, 1, 4, dtype=torch.float32)
        value = torch.randn(total, 1, 4, dtype=torch.float32)
        output = torch.empty_like(query)
        metadata = _make_metadata(lens)
        result = impl._forward_encoder_attention(query, key, value, metadata, output)

        # Per-sequence reference.
        offset = 0
        for length in lens:
            q = query[offset : offset + length]
            k = key[offset : offset + length]
            v = value[offset : offset + length]
            scores = torch.bmm(q.permute(1, 0, 2), k.permute(1, 0, 2).transpose(1, 2))
            ref = torch.bmm(torch.softmax(scores, dim=-1), v.permute(1, 0, 2)).permute(1, 0, 2)
            self.assertTrue(torch.allclose(result[offset : offset + length], ref, atol=1e-5))
            offset += length

    def test_gqa_heads_are_expanded(self):
        """num_kv_heads < num_heads replicates K/V heads (GQA)."""
        impl = _make_impl(4, 2, 8, hf_text_config=_plain_text_config())
        torch.manual_seed(0)
        query = torch.randn(4, 4, 8, dtype=torch.float32)
        # One shared K/V head pair per two query heads.
        key = torch.randn(4, 2, 8, dtype=torch.float32)
        value = torch.randn(4, 2, 8, dtype=torch.float32)
        output = torch.empty_like(query)
        metadata = _make_metadata([4])
        result = impl._forward_encoder_attention(query, key, value, metadata, output)

        k = key.permute(1, 0, 2).repeat_interleave(2, dim=0)
        v = value.permute(1, 0, 2).repeat_interleave(2, dim=0)
        scores = torch.bmm(query.permute(1, 0, 2), k.transpose(1, 2))
        ref = torch.bmm(torch.softmax(scores, dim=-1), v).permute(1, 0, 2)
        self.assertTrue(torch.allclose(result, ref, atol=1e-5))

    def test_small_total_uses_fused_path(self):
        """Totals within the 8192-bound take the fused bmm path and still
        match the per-sequence reference exactly."""
        impl = _make_impl(1, 1, 4, hf_text_config=_plain_text_config())
        torch.manual_seed(0)
        lens = [3, 3]
        total = sum(lens)
        query = torch.randn(total, 1, 4, dtype=torch.float32)
        key = torch.randn(total, 1, 4, dtype=torch.float32)
        value = torch.randn(total, 1, 4, dtype=torch.float32)
        output = torch.empty_like(query)
        metadata = _make_metadata(lens)
        result = impl._forward_encoder_attention(query, key, value, metadata, output)
        offset = 0
        for length in lens:
            q = query[offset : offset + length]
            k = key[offset : offset + length]
            v = value[offset : offset + length]
            scores = torch.bmm(q.permute(1, 0, 2), k.permute(1, 0, 2).transpose(1, 2))
            ref = torch.bmm(torch.softmax(scores, dim=-1), v.permute(1, 0, 2)).permute(1, 0, 2)
            self.assertTrue(torch.allclose(result[offset : offset + length], ref, atol=1e-5))
            offset += length

    def test_forward_impl_routes_encoder_only(self):
        """forward_impl dispatches ENCODER_ONLY layers to the torch path."""
        text_cfg = MagicMock()
        text_cfg.sliding_window = None
        impl = _make_impl(2, 2, 8, hf_text_config=text_cfg)
        torch.manual_seed(0)
        query = torch.randn(5, 2, 8, dtype=torch.float32)
        key = torch.randn(5, 2, 8, dtype=torch.float32)
        value = torch.randn(5, 2, 8, dtype=torch.float32)
        output = torch.empty_like(query)
        metadata = _make_metadata([5])
        with patch("torch_npu._npu_flash_attention") as mock_flash:
            result = impl.forward_impl(query, key, value, None, metadata, output)
        # The ATB flash-attention op must not be touched on this path.
        mock_flash.assert_not_called()
        self.assertTrue(torch.allclose(result, output[:5], atol=1e-6))
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))
