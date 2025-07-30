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

from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.vocab_parallel_embedding import \
    VocabParallelEmbedding

from tests.ut.base import TestBase
from vllm_ascend.ops.vocab_parallel_embedding import (
    get_masked_input_and_mask, vocab_parallel_embedding_forward)

VOCAB_PARALLEL_EMBEDDING_TEST_NUM_RANDOM_SEEDS = 128


class TestGetMaskedInputAndMask(TestBase):

    def setUp(self):
        self.input_ = torch.arange(12)

    def test_get_masked_input_and_mask(self):
        # tp 1 no padding
        input_modified, _ = get_masked_input_and_mask(
            self.input_,
            org_vocab_start_index=0,
            org_vocab_end_index=8,
            added_vocab_start_index=8,
            added_vocab_end_index=12,
            num_org_vocab_padding=0)
        assert torch.equal(self.input_, input_modified)

        # tp 2 no padding
        input_rank_0, _ = get_masked_input_and_mask(self.input_,
                                                    org_vocab_start_index=0,
                                                    org_vocab_end_index=4,
                                                    added_vocab_start_index=8,
                                                    added_vocab_end_index=10,
                                                    num_org_vocab_padding=0)

        input_rank_1, _ = get_masked_input_and_mask(self.input_,
                                                    org_vocab_start_index=4,
                                                    org_vocab_end_index=8,
                                                    added_vocab_start_index=10,
                                                    added_vocab_end_index=12,
                                                    num_org_vocab_padding=0)

        assert torch.equal(input_rank_0,
                           torch.tensor([0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 0, 0]))
        assert torch.equal(input_rank_1,
                           torch.tensor([0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5]))

        # tp 4 no padding
        input_rank_0, _ = get_masked_input_and_mask(self.input_,
                                                    org_vocab_start_index=0,
                                                    org_vocab_end_index=2,
                                                    added_vocab_start_index=8,
                                                    added_vocab_end_index=9,
                                                    num_org_vocab_padding=0)

        input_rank_1, _ = get_masked_input_and_mask(self.input_,
                                                    org_vocab_start_index=2,
                                                    org_vocab_end_index=4,
                                                    added_vocab_start_index=9,
                                                    added_vocab_end_index=10,
                                                    num_org_vocab_padding=0)

        input_rank_2, _ = get_masked_input_and_mask(self.input_,
                                                    org_vocab_start_index=4,
                                                    org_vocab_end_index=6,
                                                    added_vocab_start_index=10,
                                                    added_vocab_end_index=11,
                                                    num_org_vocab_padding=0)

        input_rank_3, _ = get_masked_input_and_mask(self.input_,
                                                    org_vocab_start_index=6,
                                                    org_vocab_end_index=8,
                                                    added_vocab_start_index=11,
                                                    added_vocab_end_index=12,
                                                    num_org_vocab_padding=0)
        assert torch.equal(input_rank_0,
                           torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]))
        assert torch.equal(input_rank_1,
                           torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0]))
        assert torch.equal(input_rank_2,
                           torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0]))
        assert torch.equal(input_rank_3,
                           torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2]))

        # tp 1 with padding
        input_modified, _ = get_masked_input_and_mask(
            self.input_,
            org_vocab_start_index=0,
            org_vocab_end_index=8,
            added_vocab_start_index=8,
            added_vocab_end_index=12,
            num_org_vocab_padding=2)
        assert torch.equal(
            input_modified,
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]))

        # tp 2 with padding
        input_rank_0, _ = get_masked_input_and_mask(self.input_,
                                                    org_vocab_start_index=0,
                                                    org_vocab_end_index=4,
                                                    added_vocab_start_index=8,
                                                    added_vocab_end_index=10,
                                                    num_org_vocab_padding=2)

        input_rank_1, _ = get_masked_input_and_mask(self.input_,
                                                    org_vocab_start_index=4,
                                                    org_vocab_end_index=8,
                                                    added_vocab_start_index=10,
                                                    added_vocab_end_index=12,
                                                    num_org_vocab_padding=2)
        assert torch.equal(input_rank_0,
                           torch.tensor([0, 1, 2, 3, 0, 0, 0, 0, 6, 7, 0, 0]))
        assert torch.equal(input_rank_1,
                           torch.tensor([0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 6, 7]))

        # tp 4 with padding
        input_rank_0, _ = get_masked_input_and_mask(self.input_,
                                                    org_vocab_start_index=0,
                                                    org_vocab_end_index=2,
                                                    added_vocab_start_index=8,
                                                    added_vocab_end_index=9,
                                                    num_org_vocab_padding=2)

        input_rank_1, _ = get_masked_input_and_mask(self.input_,
                                                    org_vocab_start_index=2,
                                                    org_vocab_end_index=4,
                                                    added_vocab_start_index=9,
                                                    added_vocab_end_index=10,
                                                    num_org_vocab_padding=2)

        input_rank_2, _ = get_masked_input_and_mask(self.input_,
                                                    org_vocab_start_index=4,
                                                    org_vocab_end_index=6,
                                                    added_vocab_start_index=10,
                                                    added_vocab_end_index=11,
                                                    num_org_vocab_padding=2)

        input_rank_3, _ = get_masked_input_and_mask(self.input_,
                                                    org_vocab_start_index=6,
                                                    org_vocab_end_index=8,
                                                    added_vocab_start_index=11,
                                                    added_vocab_end_index=12,
                                                    num_org_vocab_padding=2)
        assert torch.equal(input_rank_0,
                           torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0]))
        assert torch.equal(input_rank_1,
                           torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 0, 0]))
        assert torch.equal(input_rank_2,
                           torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0]))
        assert torch.equal(input_rank_3,
                           torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 4]))


class TestVocabParallelEmbedding(TestBase):

    def setUp(self):
        # Create a mock VocabParallelEmbedding instance
        self.mock_embedding = MagicMock(spec=VocabParallelEmbedding)
        self.mock_embedding.tp_size = 2  # Test with tensor parallelism
        self.mock_embedding.shard_indices = MagicMock()
        self.mock_embedding.shard_indices.org_vocab_start_index = 10
        self.mock_embedding.shard_indices.org_vocab_end_index = 20
        self.mock_embedding.shard_indices.num_org_vocab_padding = 5
        self.mock_embedding.shard_indices.added_vocab_start_index = 30
        self.mock_embedding.shard_indices.added_vocab_end_index = 40
        self.mock_embedding.quant_method = MagicMock()

        # Set consistent embedding dimension for all tests
        self.embedding_dim = 10
        # Mock embedding returns tensor with shape (input_length, embedding_dim)
        self.mock_embedding.quant_method.embedding = MagicMock(
            side_effect=lambda _, x: torch.randn(x.shape[0], self.embedding_dim
                                                 ))

    def test_get_masked_input_and_mask(self):
        """Test the mask and offset calculation helper function."""
        input_ = torch.tensor([5, 15, 25, 35, 45])  # includes all cases

        masked_input, mask = get_masked_input_and_mask(
            input_,
            org_vocab_start_index=10,
            org_vocab_end_index=20,
            num_org_vocab_padding=5,
            added_vocab_start_index=30,
            added_vocab_end_index=40)

        # The mask should be True for INVALID tokens (ones we want to mask out)
        expected_mask = torch.tensor([True, False, True, False, True])
        self.assertTrue(
            torch.equal(mask, expected_mask),
            f"Mask mismatch. Expected {expected_mask}, got {mask}")

        # Check masked input values
        expected_masked = torch.tensor([0, 5, 0, 20, 0])
        self.assertTrue(
            torch.equal(masked_input, expected_masked),
            f"Masked input mismatch. Expected {expected_masked}, got {masked_input}"
        )

    def test_forward_with_tp_size_1(self):
        """Test forward pass without tensor parallelism."""
        # Create a fresh mock embedding with tp_size=1
        mock_embedding = MagicMock(spec=VocabParallelEmbedding)
        mock_embedding.tp_size = 1
        mock_embedding.quant_method = MagicMock()
        mock_embedding.quant_method.embedding = MagicMock(
            return_value=torch.randn(3, self.embedding_dim))

        input_ = torch.tensor([1, 2, 3])

        with patch(
                "vllm_ascend.ops.vocab_parallel_embedding.tensor_model_parallel_all_reduce",
                side_effect=lambda x: x) as mock_reduce_tp1:
            output = vocab_parallel_embedding_forward(mock_embedding, input_)

        # Should just pass through without masking
        mock_embedding.quant_method.embedding.assert_called_once_with(
            mock_embedding, input_.long())
        self.assertEqual(output.shape, (3, self.embedding_dim))

        # Verify all_reduce was called once
        mock_reduce_tp1.assert_called_once()

    def test_forward_with_tp(self):
        """Test forward pass with tensor parallelism."""
        input_ = torch.tensor([15, 35])  # one org vocab, one added vocab
        with patch(
                "vllm_ascend.ops.vocab_parallel_embedding.tensor_model_parallel_all_reduce",
                side_effect=lambda x: x) as mock_reduce_tp:
            output = vocab_parallel_embedding_forward(self.mock_embedding,
                                                      input_)

        # Check that masking was applied correctly
        self.mock_embedding.quant_method.embedding.assert_called_once()
        called_input = self.mock_embedding.quant_method.embedding.call_args[0][
            1]
        expected_input = torch.tensor([5, 20])  # after offset calculation
        self.assertTrue(torch.all(called_input == expected_input))

        # Check that all reduce was called
        # self.dist_mock.tensor_model_parallel_all_reduce.assert_called_once()
        mock_reduce_tp.assert_called_once()
        self.assertEqual(output.shape, (2, self.embedding_dim))

    def test_forward_with_invalid_vocab(self):
        """Test that invalid vocab indices are properly masked out."""
        input_ = torch.tensor([5, 15, 25, 35, 45])  # includes invalid cases

        # Create predictable mock output
        mock_output = torch.randn(5, self.embedding_dim)
        self.mock_embedding.quant_method.embedding = MagicMock(
            return_value=mock_output.clone())
        with patch(
                "vllm_ascend.ops.vocab_parallel_embedding.tensor_model_parallel_all_reduce",
                side_effect=lambda x: x):
            output = vocab_parallel_embedding_forward(self.mock_embedding,
                                                      input_)

        # Check that invalid positions (0, 2, 4) were zeroed out
        self.assertTrue(torch.all(output[0] == 0))
        self.assertTrue(torch.all(output[2] == 0))
        self.assertTrue(torch.all(output[4] == 0))
        self.assertTrue(torch.all(output[1] == mock_output[1]))
        self.assertTrue(torch.all(output[3] == mock_output[3]))
        self.assertEqual(output.shape, (5, self.embedding_dim))

    def test_output_shape(self):
        """Test that output shape is correct."""
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
                    output = vocab_parallel_embedding_forward(
                        self.mock_embedding, input_)
                self.assertEqual(output.shape, expected_shape)
