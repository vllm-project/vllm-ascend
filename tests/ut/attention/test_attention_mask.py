#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

    def test_init_attention_mask_builder(self):
        # generate attention_mask_builder with float16
        attention_mask_builder = AttentionMaskBuilder(max_seq_len=1024,
                                                      dtype=torch.float16)
        self.assertEqual(attention_mask_builder._seq_len_cached, 1024)
        self.assertEqual(attention_mask_builder.attn_mask_cache.dtype,
                         torch.float16)
        self.assertEqual(attention_mask_builder.attn_mask_cache.shape,
                         (1024, 1024))
        self.assertEqual(attention_mask_builder.attn_mask_cache[0][-1],
                         torch.tensor(float("-inf"), dtype=torch.float16))

        # generate attention_mask_builder with bfloat16
        attention_mask_builder = AttentionMaskBuilder(max_seq_len=2048,
                                                      dtype=torch.bfloat16)
        self.assertEqual(attention_mask_builder._seq_len_cached, 2048)
        self.assertEqual(attention_mask_builder.attn_mask_cache.dtype,
                         torch.bfloat16)
        self.assertEqual(attention_mask_builder.attn_mask_cache.shape,
                         (2048, 2048))
        self.assertEqual(attention_mask_builder.attn_mask_cache[0][-1],
                         torch.tensor(1, dtype=torch.bfloat16))

    def test_get_mask_scale_factor(self):
        # supported data types
        self.assertEqual(
            AttentionMaskBuilder.get_mask_scale_factor(torch.float16), 1)
        self.assertEqual(
            AttentionMaskBuilder.get_mask_scale_factor(torch.bfloat16), -10000)
        # mask_scale_factor now only supports data types: torch.float16 and torch.bfloat16
        # Otherwise raise ValueError
        with self.assertRaises(ValueError):
            AttentionMaskBuilder.get_mask_scale_factor(torch.int8)

    def test_get_attn_mask(self):
        # if the len is less than max_seq_len, the attn_mask_cache will not be updated
        attention_mask_builder = AttentionMaskBuilder(max_seq_len=1024,
                                                      dtype=torch.float16)
        attn_mask = attention_mask_builder.get_attn_mask(
            max_seq_len=512, dtype=torch.float16, device=torch.device("cpu"))
        self.assertEqual(attn_mask.shape, (512, 512))
        self.assertEqual(attn_mask[0][-1],
                         torch.tensor(float("-inf"), dtype=torch.float16))
        self.assertEqual(attention_mask_builder._seq_len_cached, 1024)
        self.assertEqual(attention_mask_builder.attn_mask_cache.shape,
                         (1024, 1024))
        self.assertEqual(attention_mask_builder.attn_mask_cache[0][-1],
                         torch.tensor(float("-inf"), dtype=torch.float16))

        # if the len is greater than max_seq_len, the attn_mask_cache will be updated
        attn_mask = attention_mask_builder.get_attn_mask(
            max_seq_len=2048, dtype=torch.float16, device=torch.device("cpu"))
        self.assertEqual(attn_mask.shape, (2048, 2048))
        self.assertEqual(attn_mask[0][-1],
                         torch.tensor(float("-inf"), dtype=torch.float16))
        self.assertEqual(attention_mask_builder._seq_len_cached, 2048)
        self.assertEqual(attention_mask_builder.attn_mask_cache.shape,
                         (2048, 2048))
        self.assertEqual(attention_mask_builder.attn_mask_cache[0][-1],
                         torch.tensor(float("-inf"), dtype=torch.float16))

    def test_get_splitfuse_attn_mask(self):
        attention_mask_builder = AttentionMaskBuilder(max_seq_len=1024,
                                                      dtype=torch.float16)
        attn_mask = attention_mask_builder.get_splitfuse_attn_mask(
            seq_lens=torch.tensor([10, 20, 100]),
            position=torch.tensor([7, 8, 9, 18, 19, 99]),
            dtype=torch.float16,
            device=torch.device("cpu"),
        )
        self.assertEqual(attn_mask.shape, (6, 100))
        self.assertEqual(attention_mask_builder._seq_len_cached, 1024)

        attn_mask = attention_mask_builder.get_splitfuse_attn_mask(
            seq_lens=torch.tensor([10, 3000, 2000]),
            position=torch.tensor([7, 8, 9, 2999, 1999]),
            dtype=torch.float16,
            device=torch.device("cpu"),
        )
        self.assertEqual(attn_mask.shape, (5, 3000))
        self.assertEqual(attention_mask_builder._seq_len_cached, 3000)

        # splitfuse_attn_mask now only supports data types: torch.float16 and torch.bfloat16
        # otherwise raise ValueError
        with self.assertRaises(ValueError):
            attn_mask = attention_mask_builder.get_splitfuse_attn_mask(
                seq_lens=torch.tensor([10, 20, 100]),
                position=torch.tensor([7, 8, 9, 18, 19, 99]),
                dtype=torch.int8,
                device=torch.device("cpu"),
            )

    def test_mask_value_cleanliness(self):
        attention_mask_builder = AttentionMaskBuilder(max_seq_len=6,
                                                      dtype=torch.bfloat16)
        self.assertEqual(attention_mask_builder.attn_mask_cache[-2][-1],
                         torch.tensor(1, dtype=torch.bfloat16))

        attn_mask = attention_mask_builder.get_splitfuse_attn_mask(
            seq_lens=torch.tensor([6]),
            position=torch.tensor([3, 4, 5]),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        self.assertEqual(
            attn_mask[-2][-1],
            torch.tensor(-10000, dtype=torch.bfloat16,
                         device=attn_mask.device))
        self.assertEqual(attention_mask_builder.attn_mask_cache[-2][-1],
                         torch.tensor(1, dtype=torch.bfloat16))
