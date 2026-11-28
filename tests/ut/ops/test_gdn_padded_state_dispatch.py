# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_ascend.ops.gdn import _is_padded_state_cache

NUM_SLOTS = 8
STATE_SHAPE = (128, 32, 32)
PADDING_ELEMENTS = 8


def _make_state(padding: int, dtype=torch.bfloat16) -> torch.Tensor:
    row = torch.Size(STATE_SHAPE).numel()
    stride = row + padding
    storage = torch.empty((NUM_SLOTS - 1) * stride + row, dtype=dtype)
    inner_strides = []
    running = 1
    for dim in reversed(STATE_SHAPE):
        inner_strides.append(running)
        running *= dim
    inner_strides.reverse()
    return torch.as_strided(
        storage,
        size=(NUM_SLOTS, *STATE_SHAPE),
        stride=(stride, *inner_strides),
    )


def test_contiguous_cache_is_not_padded():
    assert not _is_padded_state_cache(torch.empty((NUM_SLOTS, *STATE_SHAPE)))


def test_hybrid_cache_manager_view_is_padded():
    # The hybrid KV-cache manager hands out a state cache whose leading stride is
    # larger than one state row, which is what makes advanced indexing materialize
    # the whole cache on NPU.
    state = _make_state(PADDING_ELEMENTS)
    assert state.stride(0) > state[0].numel()
    assert _is_padded_state_cache(state)


@pytest.mark.parametrize("padding", [1, PADDING_ELEMENTS, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_any_leading_padding_is_detected(padding, dtype):
    assert _is_padded_state_cache(_make_state(padding, dtype=dtype))


def test_sliced_packed_view_stays_packed():
    # A packed cache that is itself a slice keeps stride(0) == one row, so the plain
    # index path (and its cheaper gather) stays in use.
    packed = torch.empty((2 * NUM_SLOTS, *STATE_SHAPE))
    view = packed[:NUM_SLOTS]
    assert _is_padded_state_cache(view) is False
