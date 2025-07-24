from unittest.mock import MagicMock

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.multistream.base import MSAttentionMetadataSplitConfig
from vllm_ascend.multistream.ms_split import (compute_split_seq_index,
                                              model_input_split_v1_mla_attn,
                                              split_attn_int_type,
                                              split_attn_tensor_type)


class TestMsSplit(TestBase):

    def test_decode_only(self):
        result = compute_split_seq_index(
            query_lens=None,
            attn_state=AscendAttentionState.DecodeOnly,
            num_tokens=10)
        self.assertEqual(result, [5, 5])

    def test_perfect_balance(self):
        query_lens = [2, 3, 5]
        result = compute_split_seq_index(
            query_lens=query_lens,
            attn_state=AscendAttentionState.PrefillNoCache,
            num_tokens=10)
        self.assertEqual(result, [5, 2])

    def test_imbalance(self):
        query_lens = [1, 2, 3, 4]
        result = compute_split_seq_index(
            query_lens=query_lens,
            attn_state=AscendAttentionState.PrefillNoCache,
            num_tokens=10)
        self.assertEqual(result, [0, 0])

    def test_query_lens_none(self):
        with self.assertRaises(AssertionError):
            compute_split_seq_index(
                query_lens=None,
                attn_state=AscendAttentionState.PrefillNoCache,
                num_tokens=10)

    def test_empty_query_lens(self):
        query_lens: list[int] = []
        result = compute_split_seq_index(
            query_lens=query_lens,
            attn_state=AscendAttentionState.PrefillNoCache,
            num_tokens=10)
        self.assertEqual(result, [0, 0])

    def test_single_query_len(self):
        query_lens = [10]
        result = compute_split_seq_index(
            query_lens=query_lens,
            attn_state=AscendAttentionState.PrefillNoCache,
            num_tokens=10)
        self.assertEqual(result, [0, 0])

    def test_split_attn_tensor_type_middle(self):
        input_tensor = torch.tensor([1, 2, 3, 4, 5])
        index = 3
        expected_result = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        result = split_attn_tensor_type(input_tensor, index)
        self.assertEqual(len(result), 2)
        self.assertTrue(torch.equal(result[0], expected_result[0]))
        self.assertTrue(torch.equal(result[1], expected_result[1]))

    def test_split_attn_tensor_type_start(self):
        input_tensor = torch.tensor([1, 2, 3, 4, 5])
        index = 0
        expected_result = [torch.tensor([]), torch.tensor([1, 2, 3, 4, 5])]
        result = split_attn_tensor_type(input_tensor, index)
        self.assertEqual(len(result), 2)
        self.assertTrue(torch.equal(result[0], expected_result[0]))
        self.assertTrue(torch.equal(result[1], expected_result[1]))

    def test_split_attn_tensor_type_end(self):
        input_tensor = torch.tensor([1, 2, 3, 4, 5])
        index = 5
        expected_result = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([])]
        result = split_attn_tensor_type(input_tensor, index)
        self.assertEqual(len(result), 2)
        self.assertTrue(torch.equal(result[0], expected_result[0]))
        self.assertTrue(torch.equal(result[1], expected_result[1]))

    def test_split_attn_tensor_type_empty_tensor(self):
        input_tensor = torch.tensor([])
        index = 0
        expected_result = [torch.tensor([]), torch.tensor([])]
        result = split_attn_tensor_type(input_tensor, index)
        self.assertEqual(len(result), 2)
        self.assertTrue(torch.equal(result[0], expected_result[0]))
        self.assertTrue(torch.equal(result[1], expected_result[1]))

    def test_split_attn_int_type_index_greater_than_var(self):
        var = 5
        index = 10
        expected_result = [5, 0]
        result = split_attn_int_type(var, index)
        self.assertEqual(result, expected_result)

    def test_split_attn_int_type_index_equal_to_var(self):
        var = 5
        index = 5
        expected_result = [5, 0]
        result = split_attn_int_type(var, index)
        self.assertEqual(result, expected_result)

    def test_split_attn_int_type_index_less_than_var(self):
        var = 10
        index = 5
        expected_result = [5, 5]
        result = split_attn_int_type(var, index)
        self.assertEqual(result, expected_result)

    def test_split_attn_int_type_index_zero(self):
        var = 10
        index = 0
        expected_result = [0, 10]
        result = split_attn_int_type(var, index)
        self.assertEqual(result, expected_result)

    def test_split_attn_int_type_var_zero(self):
        var = 0
        index = 5
        expected_result = [0, 0]
        result = split_attn_int_type(var, index)
        self.assertEqual(result, expected_result)

    def test_split_attn_int_type_both_zero(self):
        var = 0
        index = 0
        expected_result = [0, 0]
        result = split_attn_int_type(var, index)
        self.assertEqual(result, expected_result)

    def test_split_v1_mla_attn_input_none(self):
        attn_metadata = None
        ascendMLAPrefillMetadata = MagicMock()
        ms_split_config = MSAttentionMetadataSplitConfig(num_micro_batches=1)
        result = model_input_split_v1_mla_attn(attn_metadata,
                                               ascendMLAPrefillMetadata,
                                               ms_split_config)
        self.assertEqual(result, [None])
