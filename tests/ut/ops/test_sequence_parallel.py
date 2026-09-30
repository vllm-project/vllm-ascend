# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm_ascend.models.common.ops.sequence_parallel as sp_module
from vllm_ascend.models.common.ops.sequence_parallel import (
    _ascend_sp_padding_mask_impl,
    _ascend_sp_shard_fake,
    _ascend_sp_shard_impl,
)


@pytest.fixture
def set_tp(monkeypatch: pytest.MonkeyPatch):
    def _set(tp_size: int, tp_rank: int) -> None:
        monkeypatch.setattr(sp_module, "get_tensor_model_parallel_world_size", lambda: tp_size)
        monkeypatch.setattr(sp_module, "get_tensor_model_parallel_rank", lambda: tp_rank)

    return _set


def test_sp_shard_pads_token_axis_of_3d_input(set_tp):
    tp_size = 8
    x = torch.arange(6 * 2 * 3, dtype=torch.float32).reshape(6, 2, 3)
    for tp_rank in range(tp_size):
        set_tp(tp_size, tp_rank)
        out = _ascend_sp_shard_impl(x)
        assert out.shape == (1, 2, 3)
        if tp_rank < 6:
            assert torch.equal(out[0], x[tp_rank])
        else:
            assert torch.equal(out[0], torch.zeros(2, 3))


def test_sp_shard_chunks_2d_input_without_padding(set_tp):
    set_tp(8, 3)
    x = torch.arange(16 * 4, dtype=torch.float32).reshape(16, 4)
    out = _ascend_sp_shard_impl(x)
    assert out.shape == (2, 4)
    assert torch.equal(out, x[6:8])
    # No-pad path: a functional custom op must not return a view of its input.
    assert out.data_ptr() != x.data_ptr()


def test_sp_shard_shards_1d_input_ids(set_tp):
    set_tp(8, 5)
    input_ids = torch.tensor([10, 11, 12, 13, 14, 15])
    out = _ascend_sp_shard_impl(input_ids)
    assert out.shape == (1,)
    assert out.item() == 15


def test_sp_shard_fake_derives_cdiv_rows(set_tp):
    set_tp(8, 0)
    x = torch.empty(6, 2, 3)
    assert _ascend_sp_shard_fake(x).shape == (1, 2, 3)
    x = torch.empty(16, 4)
    assert _ascend_sp_shard_fake(x).shape == (2, 4)


def test_sp_padding_mask_pads_true_rows(set_tp):
    set_tp(4, 3)
    is_padding = torch.tensor([False, True, False])
    out = _ascend_sp_padding_mask_impl(is_padding)
    assert out.shape == (1,)
    assert bool(out[0]) is True  # rank chunk falls on the padded True row


def test_sp_padding_mask_no_pad_returns_fresh_chunk(set_tp):
    set_tp(4, 1)
    is_padding = torch.zeros(4, dtype=torch.bool)
    out = _ascend_sp_padding_mask_impl(is_padding)
    assert out.shape == (1,)
    assert not bool(out[0])
    # No-pad path must not alias the input (functional custom op contract).
    assert out.data_ptr() != is_padding.data_ptr()
