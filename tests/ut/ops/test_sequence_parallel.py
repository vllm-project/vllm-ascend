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


@pytest.mark.parametrize("tp_size", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [1, 7, 8, 12, 16, 50, 4096])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_prepared_routing_ids_match_gathered_sp_shards(set_tp, tp_size, num_tokens, dtype):
    input_ids = torch.arange(num_tokens, dtype=dtype) + 10
    input_ids[0] = -1
    original = input_ids.clone()
    shards = []
    for rank in range(tp_size):
        set_tp(tp_size, rank)
        shards.append(_ascend_sp_shard_impl(input_ids).to(torch.int64))
    gathered = torch.cat(shards)
    expected = torch.where(gathered == -1, 0, gathered)

    actual = sp_module._prepare_sp_routing_input_ids_impl(input_ids)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(input_ids, original, rtol=0, atol=0)
    assert actual.dtype == torch.int64
    assert actual.shape == sp_module._prepare_sp_routing_input_ids_fake(input_ids).shape
    assert actual.data_ptr() != input_ids.data_ptr()


def test_prepared_routing_ids_follow_updated_input_buffer(set_tp):
    set_tp(8, 0)
    input_ids = torch.arange(12, dtype=torch.int32)
    first = sp_module._prepare_sp_routing_input_ids_impl(input_ids)
    input_ids.add_(100)
    second = sp_module._prepare_sp_routing_input_ids_impl(input_ids)
    torch.testing.assert_close(first[:12], torch.arange(12), rtol=0, atol=0)
    torch.testing.assert_close(second[:12], torch.arange(12) + 100, rtol=0, atol=0)
    assert torch.equal(second[12:], torch.zeros(4, dtype=torch.int64))


def test_sp_reduce_scatter_appends_zero_rows_with_cat(set_tp, monkeypatch: pytest.MonkeyPatch):
    set_tp(2, 0)
    monkeypatch.setattr(sp_module, "_custom_collective", lambda name, value: value)
    original_cat = torch.cat
    cat_calls = []

    def record_cat(tensors, dim=0):
        cat_calls.append((tensors, dim))
        return original_cat(tensors, dim=dim)

    monkeypatch.setattr(sp_module.torch, "cat", record_cat)
    x = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)

    out = sp_module._ascend_sp_reduce_scatter_impl(x)

    assert out.shape == (4, 4)
    assert torch.equal(out[:3], x)
    assert torch.equal(out[3], torch.zeros(4))
    assert len(cat_calls) == 1
    assert cat_calls[0][1] == 0


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
