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

from vllm_ascend.ops.vocab_parallel_embedding import (
    AscendLogitsProcessor, AscendParallelLMHead, AscendVocabParallelEmbedding)

VOCAB_PARALLEL_EMBEDDING_TEST_NUM_RANDOM_SEEDS = 128


class TestCustomVocabParallelEmbedding(unittest.TestCase):

    def setUp(self):
        self.num_embeddings = 50
        self.embedding_dim = 10
        self.org_num_embeddings = 40
        self.padding_size = 8

    def _create_layer(self):
        # Patch methods and dependencies for VocabParallelEmbedding
        mock_group = MagicMock()
        mock_group.world_size = 2
        mock_group.rank_in_group = 0
        with patch("vllm_ascend.ops.vocab_parallel_embedding.get_tp_group", return_value=mock_group), \
            patch("vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_rank", return_value=0), \
            patch("vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_world_size", return_value=2), \
            patch("vllm.model_executor.layers.vocab_parallel_embedding.pad_vocab_size", side_effect=lambda x, y: x + y), \
            patch("vllm.model_executor.layers.vocab_parallel_embedding.divide", side_effect=lambda x, y: x // y):

            # Create an instance of VocabParallelEmbedding
            layer = AscendVocabParallelEmbedding(
                num_embeddings=self.num_embeddings,
                embedding_dim=self.embedding_dim,
                org_num_embeddings=self.org_num_embeddings,
                padding_size=self.padding_size,
                quant_config=None,  # Mock quantization config
                prefix="")

            layer.shard_indices = MagicMock()
            layer.shard_indices.org_vocab_start_index = 10
            layer.shard_indices.org_vocab_end_index = 20
            layer.shard_indices.num_org_vocab_padding = 5
            layer.shard_indices.added_vocab_start_index = 30
            layer.shard_indices.added_vocab_end_index = 40

            # Mock the quantization method
            layer.quant_method.embedding = MagicMock(
                side_effect=lambda _, x: torch.randn(x.shape[0], self.
                                                     embedding_dim))
            return layer

    def test_get_masked_input_and_mask(self):
        """Test the mask and offset calculation helper function."""
        layer = self._create_layer()

        input_ = torch.tensor([5, 15, 25, 35, 45])

        masked_input, mask = layer._get_masked_input_and_mask(
            input_,
            org_vocab_start_index=10,
            org_vocab_end_index=20,
            num_org_vocab_padding=5,
            added_vocab_start_index=30,
            added_vocab_end_index=40)

        expected_mask = torch.tensor([True, False, True, False, True])
        self.assertTrue(
            torch.equal(mask, expected_mask),
            f"Mask mismatch. Expected {expected_mask}, got {mask}")

        expected_masked = torch.tensor([0, 5, 0, 20, 0])
        self.assertTrue(
            torch.equal(masked_input, expected_masked),
            f"Masked input mismatch. Expected {expected_masked}, got {masked_input}"
        )

    def test_forward_with_tp_size_1(self):
        """Test forward pass without tensor parallelism."""
        # Create a fresh mock embedding with tp_size=1
        layer = self._create_layer()
        layer.tp_size = 1
        layer.quant_method.embedding = MagicMock(
            return_value=torch.randn(3, layer.embedding_dim))

        input_ = torch.tensor([1, 2, 3])

        with patch(
                "vllm_ascend.ops.vocab_parallel_embedding.tensor_model_parallel_all_reduce",
                side_effect=lambda x: x) as mock_reduce_tp1:
            output = layer.forward(input_)

        # Should just pass through without masking
        layer.quant_method.embedding.assert_called_once_with(
            layer, input_.long())
        self.assertEqual(output.shape, (3, layer.embedding_dim))

        # Verify all_reduce was called once
        mock_reduce_tp1.assert_called_once()

    def test_forward_with_tp(self):
        layer = self._create_layer()
        layer.tp_size = 2

        input_ = torch.tensor([15, 35])  # one org vocab, one added vocab

        with patch(
                "vllm_ascend.ops.vocab_parallel_embedding.tensor_model_parallel_all_reduce",
                side_effect=lambda x: x) as mock_reduce_tp:
            # Call the forward method
            output = layer.forward(input_)

        # Check that masking was applied correctly
        layer.quant_method.embedding.assert_called_once()
        called_input = layer.quant_method.embedding.call_args[0][1]
        expected_input = torch.tensor([5, 20])  # after offset calculation
        self.assertTrue(torch.all(called_input == expected_input))

        # Check that all reduce was called
        mock_reduce_tp.assert_called_once()
        self.assertEqual(output.shape, (2, self.embedding_dim))

    def test_forward_with_invalid_vocab(self):
        """Test that invalid vocab indices are properly masked out."""
        # Create a fresh embedding layer
        layer = self._create_layer()
        input_ = torch.tensor([5, 15, 25, 35, 45])  # includes invalid cases
        # Create predictable mock output
        mock_output = torch.randn(5, self.embedding_dim)
        layer.quant_method.embedding = MagicMock(
            return_value=mock_output.clone())

        # Patch tensor_model_parallel_all_reduce to mock its behavior
        with patch(
                "vllm_ascend.ops.vocab_parallel_embedding.tensor_model_parallel_all_reduce",
                side_effect=lambda x: x):
            # Call the forward method
            output = layer.forward(input_)
        # Check that invalid positions (0, 2, 4) were zeroed out
        self.assertTrue(torch.all(output[0] == 0))
        self.assertTrue(torch.all(output[2] == 0))
        self.assertTrue(torch.all(output[4] == 0))
        self.assertTrue(torch.all(output[1] == mock_output[1]))
        self.assertTrue(torch.all(output[3] == mock_output[3]))
        self.assertEqual(output.shape, (5, self.embedding_dim))

    def test_output_shape(self):
        """Test that output shape is correct."""
        # Create a fresh embedding layer
        layer = self._create_layer()

        test_cases = [
            (torch.tensor([15]), (1, self.embedding_dim)),
            (torch.tensor([15, 35]), (2, self.embedding_dim)),
            (torch.tensor([15, 35, 16, 36]), (4, self.embedding_dim)),
        ]

        for input_, expected_shape in test_cases:
            with self.subTest(input=input_):
                with patch(
                        "vllm_ascend.ops.vocab_parallel_embedding.tensor_model_parallel_all_reduce",
                        side_effect=lambda x: x):
                    # Call the forward method
                    output = layer.forward(input_)
                self.assertEqual(output.shape, expected_shape)


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
