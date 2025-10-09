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
# Adapted from vllm/tests/lora/test_layers.py

import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.ops.logits_processor import AscendLogitsProcessor
from vllm_ascend.ops.vocab_parallel_embedding import AscendParallelLMHead


class TestAscendLogitsProcessor(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 50
        self.num_embeddings = 50
        self.embedding_dim = 10
        self.org_num_embeddings = 40
        self.padding_size = 8

        self.mock_group = MagicMock()
        self.mock_group.world_size = 2
        self.mock_group.rank_in_group = 0
        self.mock_ascend_config = MagicMock()
        self.mock_quant_method = MagicMock()
        self.mock_quant_method.apply = MagicMock(
            return_value=torch.randn(1, self.vocab_size))
        self.patches = [
            patch("vllm_ascend.ascend_config.get_ascend_config",
                  return_value=self.mock_ascend_config),
            patch("vllm_ascend.ops.logits_processor.get_lmhead_tp_group",
                  return_value=self.mock_group),
            patch("vllm_ascend.ops.logits_processor.lmhead_tp_enable",
                  return_value=True),
            patch(
                "vllm_ascend.ops.logits_processor.get_lmhead_tp_group.all_to_all",
                return_value=torch.randn(1, self.vocab_size)),
            patch(
                "vllm_ascend.ops.logits_processor.get_lmhead_tp_group.all_gather",
                return_value=torch.randn(1, self.vocab_size)),
            patch(
                "vllm_ascend.ops.vocab_parallel_embedding.get_lmhead_tp_group",
                return_value=self.mock_group),
            patch("vllm_ascend.ops.vocab_parallel_embedding.lmhead_tp_enable",
                  return_value=True),
            patch(
                "vllm_ascend.ops.vocab_parallel_embedding.get_lmhead_tp_group.all_to_all",
                return_value=torch.randn(1, self.vocab_size)),
            patch(
                "vllm_ascend.ops.vocab_parallel_embedding.get_lmhead_tp_group.all_gather",
                return_value=torch.randn(1, self.vocab_size)),
            patch(
                "vllm_ascend.core.schedule_config.AscendSchedulerConfig.initialize_from_config",
                return_value=MagicMock(max_num_batched_tokens=1000,
                                       max_model_len=512,
                                       enable_chunked_prefill=False))
        ]

        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_create_processor(self):
        processor = AscendLogitsProcessor(vocab_size=self.vocab_size)
        self.assertEqual(processor.vocab_size, self.vocab_size)

    def test_get_logits(self):
        processor = AscendLogitsProcessor(vocab_size=self.vocab_size)
        lmhead = AscendParallelLMHead(num_embeddings=self.num_embeddings,
                                      embedding_dim=self.embedding_dim,
                                      prefix="lm_head")
        lmhead.quant_method = self.mock_quant_method
        lmhead.quant_method.apply = self.mock_quant_method.apply
        hidden_state = torch.randn(1, self.org_num_embeddings)
        processor._get_logits(hidden_state, lmhead)
        self.mock_quant_method.apply.assert_called_once()
