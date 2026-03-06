#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.

import torch

from tests.ut.base import TestBase

from vllm_ascend.patch.worker.patch_multimodal_merge import (
    _merge_multimodal_embeddings,
)


class TestMergeMultimodalEmbeddings(TestBase):

    def test_exact_match(self):
        """Multimodal token count matches placeholder count."""
        inputs_embeds = torch.zeros(10, 4)
        is_multimodal = torch.zeros(10, dtype=torch.bool)
        is_multimodal[2:5] = True  # 3 placeholders
        mm_embeddings = torch.ones(3, 4)

        result = _merge_multimodal_embeddings(
            inputs_embeds, is_multimodal, mm_embeddings
        )
        self.assertTrue(torch.all(result[2:5] == 1.0))
        self.assertTrue(torch.all(result[:2] == 0.0))
        self.assertTrue(torch.all(result[5:] == 0.0))

    def test_fewer_multimodal_than_placeholders(self):
        """Multimodal tokens < placeholders: only first N positions filled."""
        inputs_embeds = torch.zeros(10, 4)
        is_multimodal = torch.zeros(10, dtype=torch.bool)
        is_multimodal[2:7] = True  # 5 placeholders
        mm_embeddings = torch.ones(2, 4)  # only 2 embeddings

        result = _merge_multimodal_embeddings(
            inputs_embeds, is_multimodal, mm_embeddings
        )
        # First 2 placeholder positions (idx 2, 3) should be overwritten
        self.assertTrue(torch.all(result[2:4] == 1.0))
        # Remaining placeholder positions (idx 4, 5, 6) unchanged
        self.assertTrue(torch.all(result[4:7] == 0.0))

    def test_more_multimodal_than_placeholders_raises(self):
        """Multimodal tokens > placeholders should raise ValueError."""
        inputs_embeds = torch.zeros(10, 4)
        is_multimodal = torch.zeros(10, dtype=torch.bool)
        is_multimodal[2:4] = True  # 2 placeholders
        mm_embeddings = torch.ones(5, 4)  # 5 embeddings

        with self.assertRaises(ValueError):
            _merge_multimodal_embeddings(
                inputs_embeds, is_multimodal, mm_embeddings
            )

    def test_nested_embeddings(self):
        """Test with nested tensor list (multiple images)."""
        inputs_embeds = torch.zeros(10, 4)
        is_multimodal = torch.zeros(10, dtype=torch.bool)
        is_multimodal[1:3] = True  # 2 positions for image 1
        is_multimodal[6:8] = True  # 2 positions for image 2
        mm_embeddings = [torch.ones(2, 4), torch.ones(2, 4) * 2]

        result = _merge_multimodal_embeddings(
            inputs_embeds, is_multimodal, mm_embeddings
        )
        self.assertTrue(torch.all(result[1:3] == 1.0))
        self.assertTrue(torch.all(result[6:8] == 2.0))

    def test_dtype_cast(self):
        """Multimodal embeddings are cast to input dtype."""
        inputs_embeds = torch.zeros(5, 4, dtype=torch.float16)
        is_multimodal = torch.zeros(5, dtype=torch.bool)
        is_multimodal[0] = True
        mm_embeddings = torch.ones(1, 4, dtype=torch.float32)

        result = _merge_multimodal_embeddings(
            inputs_embeds, is_multimodal, mm_embeddings
        )
        self.assertEqual(result.dtype, torch.float16)
        self.assertTrue(torch.all(result[0] == 1.0))
