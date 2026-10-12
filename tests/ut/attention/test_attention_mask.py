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

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder


class TestAttentionMaskBuilder(TestBase):
    def test_get_attn_mask(self):
        # if the len is less than max_seq_len, the attn_mask_cache will not be updated
        attention_mask_builder = AttentionMaskBuilder(torch.device("cpu"))
        attn_mask = attention_mask_builder.get_attn_mask(max_seq_len=512, dtype=torch.float16)
        self.assertEqual(attn_mask.shape, (512, 512))
        self.assertEqual(attn_mask[0][-1], torch.tensor(float("-inf"), dtype=torch.float16))
        self.assertEqual(attention_mask_builder._seq_len_cached, 512)
        self.assertEqual(attention_mask_builder.attn_mask_cache.shape, (512, 512))
        self.assertEqual(
            attention_mask_builder.attn_mask_cache[0][-1], torch.tensor(float("-inf"), dtype=torch.float16)
        )

        # if the len is greater than max_seq_len, the attn_mask_cache will be updated
        attn_mask = attention_mask_builder.get_attn_mask(max_seq_len=2048, dtype=torch.float16)
        self.assertEqual(attn_mask.shape, (2048, 2048))
        self.assertEqual(attn_mask[0][-1], torch.tensor(float("-inf"), dtype=torch.float16))
        self.assertEqual(attention_mask_builder._seq_len_cached, 2048)
        self.assertEqual(attention_mask_builder.attn_mask_cache.shape, (2048, 2048))
        self.assertEqual(
            attention_mask_builder.attn_mask_cache[0][-1], torch.tensor(float("-inf"), dtype=torch.float16)
        )

    def test_get_splitfuse_attn_mask(self):
        attention_mask_builder = AttentionMaskBuilder(torch.device("cpu"))
        attn_mask = attention_mask_builder.get_splitfuse_attn_mask()
        self.assertEqual(attn_mask.shape, (2048, 2048))


class TestEncoderBandMask(TestBase):
    """The band mask that honours ``sliding_window`` in encoder-only attention."""

    def setUp(self):
        super().setUp()
        self.device = torch.device("cpu")
        self.builder = AttentionMaskBuilder(self.device)

    def test_band_is_inclusive_and_blocks_beyond_the_window(self):
        window = 65
        num_tokens = 200
        mask = self.builder.get_encoder_band_mask(num_tokens, window, self.device)

        self.assertEqual(mask.shape, (num_tokens, num_tokens))
        self.assertEqual(mask.dtype, torch.bool)
        # A token sees itself and its whole window: |i - j| <= window - 1.
        self.assertFalse(mask[100, 100])
        self.assertFalse(mask[100, 100 + window - 1])
        self.assertFalse(mask[100 + window - 1, 100])
        # One step further out is blocked.
        self.assertTrue(mask[100, 100 + window])
        self.assertTrue(mask[100 + window, 100])

    def test_band_is_mirrored_because_encoder_attention_is_bidirectional(self):
        num_tokens = 96
        mask = self.builder.get_encoder_band_mask(num_tokens, 17, self.device)
        self.assertTrue(torch.equal(mask, mask.T))

    def test_shorter_batch_reuses_the_cached_square(self):
        window = 33
        # 2048 is the rounded-up side, so this call hands back the cached square.
        cached = self.builder.get_encoder_band_mask(2048, window, self.device)
        self.assertIs(cached, self.builder.encoder_band_mask)
        sliced = self.builder.get_encoder_band_mask(1000, window, self.device)

        self.assertEqual(sliced.shape, (1000, 1000))
        self.assertTrue(torch.equal(sliced, cached[:1000, :1000]))
        # Same window, smaller batch: the cached square is not rebuilt.
        self.assertIs(self.builder.encoder_band_mask, cached)

    def test_larger_batch_rebuilds_and_new_window_rebuilds(self):
        narrow = self.builder.get_encoder_band_mask(1024, 17, self.device)
        self.assertTrue(narrow[0, 40])  # 40 >= 17 -> blocked
        grown = self.builder.get_encoder_band_mask(4096, 17, self.device)
        self.assertEqual(grown.shape, (4096, 4096))
        self.assertIsNot(grown, narrow)

        wide = self.builder.get_encoder_band_mask(1024, 65, self.device)
        self.assertFalse(wide[0, 40])  # 40 < 65 -> visible
