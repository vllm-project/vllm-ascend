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

from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310


class TestAttentionMaskBuilder310(TestBase):
    def setUp(self):
        self.max_seqlen = 4096
        self.attention_mask_builder = AttentionMaskBuilder310(torch.device("cpu"), self.max_seqlen)

    @patch("torch_npu.npu_format_cast")
    def test_get_attention_mask_310_for_pooling_returns_none(self, mock_format_cast):
        """Pooling models should get None mask (generated per-batch instead)."""
        model_config = MagicMock()
        model_config.runner_type = "pooling"
        attn_mask = self.attention_mask_builder.get_attention_mask(model_config)
        self.assertIsNone(attn_mask)
        # No mask allocation should have happened
        mock_format_cast.assert_not_called()

    @patch("torch_npu.npu_format_cast")
    def test_get_attention_mask_310(self, mock_format_cast):
        mock_format_cast.side_effect = lambda x, y: x
        model_config = MagicMock()
        model_config.runner_type = "generate"
        attn_mask = self.attention_mask_builder.get_attention_mask(model_config)
        self.assertEqual(attn_mask.shape, (1, self.max_seqlen // 16, self.max_seqlen, 16))
        self.assertEqual(attn_mask[0][-1][0][-1], torch.tensor(float("-inf"), dtype=torch.float16))

    @patch("torch_npu.npu_format_cast")
    def test_get_pooling_mask_shape_and_caching(self, mock_format_cast):
        """Per-batch pooling mask should be correctly sized and cached."""
        mock_format_cast.side_effect = lambda x, y: x
        mask_256 = self.attention_mask_builder.get_pooling_mask(256)
        self.assertEqual(mask_256.shape, (1, 256 // 16, 256, 16))
        self.assertTrue(torch.isinf(mask_256).any())

        # Same size should return cached instance
        mask_256_again = self.attention_mask_builder.get_pooling_mask(256)
        self.assertIs(mask_256, mask_256_again)

        # Different size should return different mask
        mask_512 = self.attention_mask_builder.get_pooling_mask(512)
        self.assertEqual(mask_512.shape, (1, 512 // 16, 512, 16))
        self.assertIsNot(mask_256, mask_512)

    @patch("torch_npu.npu_format_cast")
    def test_get_swa_mask_310(self, mock_format_cast):
        mock_format_cast.side_effect = lambda x, y: x
        swa_mask = self.attention_mask_builder.get_swa_mask(torch.float16, None)
        self.assertIsNone(swa_mask)

        sliding_window = 128
        swa_mask = self.attention_mask_builder.get_swa_mask(torch.float16, sliding_window)
        self.assertEqual(swa_mask.shape, (1, self.max_seqlen // 16, self.max_seqlen, 16))
        self.assertEqual(swa_mask[0][-1][0][-1], torch.tensor(float("-inf"), dtype=torch.float16))
        self.assertEqual(swa_mask[0][0][-1][0], torch.tensor(float("-inf"), dtype=torch.float16))

    @patch("torch_npu.npu_format_cast")
    def test_get_splitfuse_attn_mask_310(self, mock_format_cast):
        mock_format_cast.side_effect = lambda x, y: x
        attn_metadata = MagicMock()
        attn_metadata.query_start_loc = torch.tensor([0, 1, 5])
        attn_metadata.seq_lens = torch.tensor([7, 4])
        attn_mask = self.attention_mask_builder.get_splitfuse_mask(attn_metadata, torch.device("cpu"))
        self.assertEqual(attn_mask.shape, (1, self.max_seqlen // 16, 16, 16))
