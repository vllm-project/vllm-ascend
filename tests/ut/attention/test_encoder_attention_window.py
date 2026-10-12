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
#
"""Regression tests for the windowed encoder-only attention path.

``npu_fusion_attention`` reads an absent ``atten_mask`` at the default
``sparse_mode=0`` as "no masking", so the encoder-only branch used to run with
unbounded attention. Every ``sliding_attention`` layer of a bidirectional
encoder therefore ignored its window: on a Laya (ModernBERT) encoder that is a
max |dlogit| of 12.8 against a fp32 reference of the same checkpoint.
"""

from unittest.mock import MagicMock, patch

import torch

import vllm_ascend.attention.attention_v1 as attn_module
from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

SLIDING_WINDOW = 65
NUM_HEADS = 8
HEAD_SIZE = 64


def _make_impl(sliding_window=SLIDING_WINDOW) -> AscendAttentionBackendImpl:
    with (
        patch.object(attn_module, "get_current_vllm_config", return_value=MagicMock()),
        patch.object(attn_module, "needs_layer_aware_fia_graph_replay", return_value=False),
    ):
        return AscendAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_SIZE,
            scale=1.0,
            num_kv_heads=NUM_HEADS,
            alibi_slopes=None,
            sliding_window=sliding_window,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type="encoder_only",
            kv_sharing_target_layer_name=None,
        )


def _make_metadata(cumulative_lengths: list[int]) -> MagicMock:
    metadata = MagicMock()
    metadata.actual_seq_lengths_q = list(cumulative_lengths)
    metadata.num_actual_tokens = sum(cumulative_lengths)
    return metadata


class TestEncoderAttentionWindow(TestBase):
    def setUp(self):
        super().setUp()
        self.impl = _make_impl()

    def _run(
        self,
        impl: AscendAttentionBackendImpl,
        num_tokens: int,
        cumulative_lengths: list[int],
        output_rows: int | None = None,
    ):
        query = torch.zeros(num_tokens, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
        output = torch.zeros(output_rows or num_tokens, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
        captured: dict = {}

        def fake_fusion_attention(**kwargs):
            captured.update(kwargs)
            return (torch.zeros_like(kwargs["query"]),)

        with patch.object(attn_module.torch_npu, "npu_fusion_attention", side_effect=fake_fusion_attention):
            returned = impl._forward_encoder_attention(query, query, query, _make_metadata(cumulative_lengths), output)
        return captured, returned, output

    def test_without_a_window_the_original_maskless_call_is_kept(self):
        impl = _make_impl(sliding_window=None)
        captured, returned, _ = self._run(impl, 8, [4, 8])

        self.assertNotIn("atten_mask", captured)
        self.assertNotIn("sparse_mode", captured)
        self.assertEqual(captured["actual_seq_qlen"], [4, 8])
        self.assertEqual(captured["input_layout"], "TND")
        self.assertEqual(returned.shape, (8, NUM_HEADS, HEAD_SIZE))

    def test_a_sequence_longer_than_the_window_gets_the_band_mask(self):
        captured, returned, _ = self._run(self.impl, 512, [512])

        mask = captured["atten_mask"]
        self.assertEqual(captured["sparse_mode"], 0)
        self.assertEqual(mask.shape, (512, 512))
        self.assertEqual(mask.dtype, torch.bool)
        # The window boundary is inclusive: |i - j| <= sliding_window - 1.
        self.assertFalse(mask[100, 100 + SLIDING_WINDOW - 1])
        self.assertTrue(mask[100, 100 + SLIDING_WINDOW])
        self.assertEqual(returned.shape, (512, NUM_HEADS, HEAD_SIZE))

    def test_a_sequence_that_fits_in_the_window_stays_maskless(self):
        captured, _, _ = self._run(self.impl, 64, [64])

        self.assertNotIn("atten_mask", captured)
        self.assertNotIn("sparse_mode", captured)

    def test_padding_rows_are_trimmed_before_masking(self):
        # 519 rows are passed down but only 512 tokens are real, as in TND
        # layout with padding; a mask larger than the query is rejected by the
        # fused operator.
        captured, returned, output = self._run(self.impl, 519, [512])

        self.assertEqual(captured["actual_seq_qlen"], [512])
        self.assertEqual(captured["query"].shape[0], 512)
        self.assertEqual(captured["atten_mask"].shape, (512, 512))
        self.assertIs(returned, output)
        self.assertTrue(torch.equal(output[:512], torch.zeros(512, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)))

    def test_masked_and_unmasked_calls_share_the_cached_band(self):
        first, _, _ = self._run(self.impl, 512, [512])
        second, _, _ = self._run(self.impl, 512, [512])

        self.assertTrue(torch.equal(first["atten_mask"], second["atten_mask"]))
