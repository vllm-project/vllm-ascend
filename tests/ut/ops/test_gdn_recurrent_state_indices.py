# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from vllm_ascend.ops.gdn import _flatten_recurrent_state_indices


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


def test_2d_missing_query_start_loc_raises():
    indices = torch.arange(6, dtype=torch.int32).reshape(3, 2)
    with pytest.raises(ValueError, match="query_start_loc is required"):
        _flatten_recurrent_state_indices(indices, 5)


@pytest.mark.parametrize(
    "query_lens, num_padding",
    [
        ([1] * 30, 0),
        ([1] * 30, 2),
        ([1] * 16 + [128, 64, 32, 16, 8, 4, 2, 512], 0),
        ([1] * 16 + [100, 50, 25], 3),
    ],
)
def test_decode_indices_match_request_slot_layout(query_lens, num_padding):
    """Every decode token must land in the state slot of its own request.

    Ground truth is the runner-authoritative token layout
    (model_runner_v1.py:1305): tokens are packed by request slot order via
    ``np.repeat(arange(num_rows), num_scheduled_tokens)``, with decode rows
    (query_len == 1) preceding prefill rows and padding rows contributing
    zero tokens (scheduler writes ``num_scheduled_tokens`` only for requests
    with ``num_new_tokens > 0``; FIA/graph padding appends rows after
    ``num_reqs``). The helper's ``[:total_tokens]`` slice must therefore
    reproduce that mapping exactly.
    """
    num_rows = len(query_lens)
    total_rows = num_rows + num_padding
    block_table = torch.arange(total_rows * 8, dtype=torch.int32).reshape(total_rows, 8)
    indices_full = block_table[:, 0]

    scheduled = np.array(query_lens + [0] * num_padding, dtype=np.int64)
    req_indices = np.repeat(np.arange(total_rows), scheduled)
    num_decode_tokens = sum(1 for q in query_lens if q == 1)

    flattened = _flatten_recurrent_state_indices(indices_full, num_decode_tokens)
    expected = block_table[torch.from_numpy(req_indices[:num_decode_tokens]), 0]
    torch.testing.assert_close(flattened, expected)
    assert flattened.tolist() == [block_table[r, 0].item() for r in req_indices[:num_decode_tokens]]


@pytest.mark.parametrize("lens", [[2, 2, 2], [2, 1, 2], [1, 2, 1], [2, 0, 2], [1, 1, 1, 1]])
def test_spec_indices_match_token_slot_layout(lens):
    """Spec token t of sequence s must land in slot (s, t - start_s)."""
    num_slots = 2
    indices = torch.arange(len(lens) * num_slots, dtype=torch.int32).reshape(len(lens), num_slots) + 10
    qsl = torch.tensor(np.concatenate([[0], np.cumsum(lens)]), dtype=torch.int32)
    total = int(sum(lens))

    flattened = _flatten_recurrent_state_indices(indices, total, qsl)
    expected = [indices[s, j].item() for s, length in enumerate(lens) for j in range(length)]
    assert flattened.tolist() == expected
