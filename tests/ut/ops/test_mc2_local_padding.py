# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.fused_moe import prepare_finalize as module


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_tokens", [0, 1, 2, 3, 4, 7, 8, 9, 17])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("extra_rows", [0, 3])
def test_local_shard_matches_full_padding(tp_size, num_tokens, dtype, extra_rows):
    # Include uneven tensor_split lengths and DP peers with more local tokens.
    padded_len = ((num_tokens + tp_size - 1) // tp_size + 1) * tp_size + extra_rows
    x = torch.arange(num_tokens * 16, dtype=dtype).reshape(num_tokens, 16)[:, ::2]
    reference = F.pad(x, (0, 0, 0, padded_len - num_tokens))
    for rank, expected in enumerate(torch.tensor_split(reference, tp_size, dim=0)):
        actual = module._pad_and_split_tokens(x, padded_len, tp_size, rank)
        assert actual.dtype == x.dtype
        assert actual.device == x.device
        assert actual.is_contiguous()
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("num_tokens,padded_len", [(0, 4), (1, 4), (3, 4), (5, 8), (9, 16)])
def test_prepare_preserves_mask_and_global_shape(monkeypatch, num_tokens, padded_len):
    tp_size = 4
    hidden = torch.arange(num_tokens * 8, dtype=torch.float32).reshape(num_tokens, 8)
    logits = torch.arange(num_tokens * 4, dtype=torch.float32).reshape(num_tokens, 4)
    mask = torch.arange(padded_len) < num_tokens
    monkeypatch.setattr(module, "_EXTRA_CTX", SimpleNamespace(mc2_mask=mask, padded_num_tokens=padded_len))
    for rank in range(tp_size):
        prepare = object.__new__(module.PrepareAndFinalizeWithMC2)
        prepare.tp_size, prepare.tp_rank = tp_size, rank
        result = prepare.prepare(hidden, logits)
        expected_h = F.pad(hidden, (0, 0, 0, padded_len - num_tokens)).chunk(tp_size)[rank]
        expected_r = F.pad(logits, (0, 0, 0, padded_len - num_tokens)).chunk(tp_size)[rank]
        assert torch.equal(result.hidden_states, expected_h)
        assert torch.equal(result.router_logits, expected_r)
        assert torch.equal(result.mc2_mask, mask.chunk(tp_size)[rank])
        assert result.padded_hidden_states_shape == torch.Size((padded_len, 8))
        assert prepare.num_tokens == num_tokens


def test_zero_shards_do_not_share_mutable_storage():
    x = torch.ones(1, 8)
    first = module._pad_and_split_tokens(x, 4, 4, 1)
    first.fill_(9)
    second = module._pad_and_split_tokens(x, 4, 4, 1)
    assert torch.count_nonzero(second) == 0
    assert torch.equal(x, torch.ones_like(x))


def test_fully_valid_shard_does_not_launch_padding(monkeypatch):
    def unexpected_pad(*args):
        raise AssertionError("A fully valid local shard must not pad the full tensor")

    monkeypatch.setattr(module, "_pad_tokens_with_cat", unexpected_pad)
    x = torch.arange(8).reshape(1, 8)
    result = module._pad_and_split_tokens(x, 4, 4, 0)
    assert result.data_ptr() == x.data_ptr()
    assert torch.equal(result, x)
