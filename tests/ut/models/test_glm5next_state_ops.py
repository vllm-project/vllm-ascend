# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.models.glm5next.ops.state_ops import gather_initial_states, scatter_states


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("row_shape", [(2, 3), (1, 4, 4)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_fresh_rows_ignore_poison_without_changing_existing_history(poison, dtype, row_shape, index_dtype):
    state = torch.full((4, *row_shape), 7.0, dtype=dtype)
    state[0] = poison
    state[1] = poison
    saved = state.clone()
    indices = torch.tensor([-1, 3, 100, 1, 0], dtype=index_dtype)
    has_initial = torch.tensor([False, True, False, True, False])
    saved_indices, saved_mask = indices.clone(), has_initial.clone()

    actual = gather_initial_states(state, indices, has_initial)

    expected = torch.zeros((5, *row_shape), dtype=dtype)
    expected[1] = 7.0
    expected[3] = poison
    assert actual.shape == expected.shape and actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.isfinite(actual[[0, 2, 4]]).all()
    assert torch.count_nonzero(actual[[0, 2, 4]]) == 0
    torch.testing.assert_close(state, saved, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(indices, saved_indices)
    torch.testing.assert_close(has_initial, saved_mask)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_empty_batch_preserves_trailing_state_shape(dtype):
    state = torch.full((4, 1, 4, 4), float("nan"), dtype=dtype)
    result = gather_initial_states(state, torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.bool))
    assert result.shape == (0, 1, 4, 4) and result.dtype == dtype


def test_gather_preserves_padded_storage_and_masks_fresh_sentinels():
    raw = torch.full((7, 83), float("nan"))
    state = torch.as_strided(raw, (7, 2, 3, 5), (83, 15, 5, 1), storage_offset=4)
    state[5].copy_(torch.arange(30).view(2, 3, 5))
    indices = torch.tensor([5, 0, -1, 0, 100, 0], dtype=torch.int64)[::2]
    has_initial = torch.tensor([True, True, False, True, False, True])[::2]
    saved = raw.clone()

    actual = gather_initial_states(state, indices, has_initial)

    expected = torch.zeros(3, 2, 3, 5)
    expected[0] = state[5]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()
    torch.testing.assert_close(raw, saved, rtol=0, atol=0, equal_nan=True)


def test_scatter_preserves_padded_storage_and_unselected_rows():
    raw = torch.full((7, 83), -19.0)
    state = torch.as_strided(raw, (7, 2, 3, 5), (83, 15, 5, 1), storage_offset=4)
    source = torch.arange(120, dtype=torch.float32).reshape(4, 2, 3, 5)[::2]
    indices = torch.tensor([5, 0, 2, 0], dtype=torch.int32)[::2]
    expected = raw.clone()
    expected_state = torch.as_strided(expected, state.shape, state.stride(), storage_offset=4)
    expected_state[5].copy_(source[0])
    expected_state[2].copy_(source[1])

    scatter_states(state, source, indices)
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
    scatter_states(state, source[:0], indices[:0])
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
