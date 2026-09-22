# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

import pytest
import torch

from vllm_ascend.ops import rotary_embedding


@pytest.fixture
def rope(monkeypatch):
    for name in ("_cos_cache", "_sin_cache", "_cos_mla", "_sin_mla"):
        monkeypatch.setattr(rotary_embedding, name, None)
    return vars(rotary_embedding)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [32, 64, 128])
def test_layout_values_and_registration(rope, dtype, dim):
    table = torch.randn(257, dim, dtype=dtype)
    rope["_record_cos_and_sin_cache_interleaved"](table)
    expected = [half.repeat(1, 2) for half in table.chunk(2, -1)]
    for name, ref in zip(["_cos_cache", "_sin_cache"], expected):
        actual = rope[name]
        assert torch.equal(actual, ref)
        assert actual.is_contiguous()
        assert actual.stride() == (dim, 1)
    pointers = [rope[k].data_ptr() for k in ["_cos_cache", "_sin_cache"]]
    rope["_record_cos_and_sin_cache_interleaved"](torch.zeros_like(table))
    assert pointers == [rope[k].data_ptr() for k in ["_cos_cache", "_sin_cache"]]
    positions = torch.tensor([256, 0, 99, 99, 1])
    result = rope["get_cos_and_sin_mla"](positions)
    for actual, ref in zip(result, expected):
        assert torch.equal(actual, ref[positions, None, None])


@pytest.mark.parametrize("batch", [0, 1, 4, 48])
def test_persistent_buffer_updates(rope, batch):
    table = torch.randn(257, 64, dtype=torch.bfloat16)
    rope["_record_cos_and_sin_cache_interleaved"](table)
    for key in ["_cos_mla", "_sin_mla"]:
        rope[key] = torch.empty(48, 1, 1, 64, dtype=table.dtype)
    for offset in [0, 17, 129]:
        pos = (torch.arange(batch) + offset) % 257
        result = rope["get_cos_and_sin_mla"](pos, use_cache=True)
        for actual, name, cache in zip(result, ["_cos_mla", "_sin_mla"], ["_cos_cache", "_sin_cache"]):
            assert actual.untyped_storage().data_ptr() == rope[name].untyped_storage().data_ptr()
            assert torch.equal(actual, rope[cache][pos, None, None])


def test_partial_registration_guard(rope):
    existing = torch.ones(2, 64)
    rope["_cos_cache"] = existing
    rope["_record_cos_and_sin_cache_interleaved"](torch.randn(257, 64))
    assert rope["_cos_cache"] is existing
    assert rope["_sin_cache"] is None
