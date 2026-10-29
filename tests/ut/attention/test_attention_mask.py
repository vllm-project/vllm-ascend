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

import os
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.utils import RESTORE_FLAG_PATH, is_restore


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

    def test_is_restore_checks_grusflag(self):
        with patch.object(os.path, "exists", return_value=False):
            self.assertFalse(is_restore())
        with patch.object(
            os.path,
            "exists",
            side_effect=lambda path: path == RESTORE_FLAG_PATH,
        ):
            self.assertTrue(is_restore())
