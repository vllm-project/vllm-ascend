# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_ascend.ops.gdn_attn_builder import _flatten_recurrent_state_indices


@pytest.mark.parametrize("num_rows, total_tokens", [(30, 30), (32, 30), (33, 30), (1, 1)])
def test_1d_indices_sliced_to_token_count(num_rows, total_tokens):
    indices = torch.arange(num_rows, dtype=torch.int32)
    flattened = _flatten_recurrent_state_indices(indices, total_tokens)
    assert flattened.shape[0] == total_tokens
    torch.testing.assert_close(flattened, indices[:total_tokens])
    assert flattened.is_contiguous()


def test_2d_uniform_slots_flatten_without_query_start_loc():
    indices = torch.arange(8, dtype=torch.int32).reshape(4, 2)
    flattened = _flatten_recurrent_state_indices(indices, 8)
    torch.testing.assert_close(flattened, indices.reshape(-1))
    assert flattened.is_contiguous()


def test_2d_non_uniform_slots_drop_unused_draft_slots():
    indices = torch.arange(6, dtype=torch.int32).reshape(3, 2) + 10
    query_start_loc = torch.tensor([0, 2, 3, 5], dtype=torch.int32)
    flattened = _flatten_recurrent_state_indices(indices, 5, query_start_loc)
    torch.testing.assert_close(flattened, torch.tensor([10, 11, 12, 14, 15], dtype=torch.int32))


def test_2d_zero_length_sequences_dropped():
    indices = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 0, 2, 2], dtype=torch.int32)
    flattened = _flatten_recurrent_state_indices(indices, 2, query_start_loc)
    torch.testing.assert_close(flattened, torch.tensor([9, 10], dtype=torch.int32))


def test_2d_query_lens_clamped_to_slot_count():
    indices = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3, 4], dtype=torch.int32)
    flattened = _flatten_recurrent_state_indices(indices, 3, query_start_loc)
    torch.testing.assert_close(flattened, torch.tensor([1, 2, 3], dtype=torch.int32))


def test_2d_nonzero_query_start_offset():
    indices = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    query_start_loc = torch.tensor([4, 6, 7], dtype=torch.int32)
    flattened = _flatten_recurrent_state_indices(indices, 3, query_start_loc)
    torch.testing.assert_close(flattened, torch.tensor([1, 2, 3], dtype=torch.int32))


def test_2d_missing_query_start_loc_raises():
    indices = torch.arange(6, dtype=torch.int32).reshape(3, 2)
    with pytest.raises(ValueError, match="query_start_loc is required"):
        _flatten_recurrent_state_indices(indices, 5)
