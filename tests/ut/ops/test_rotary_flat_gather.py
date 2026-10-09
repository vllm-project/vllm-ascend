#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
#

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

import vllm_ascend.ops.rotary_embedding as rope

NUM_POSITIONS = 64
ROTARY_DIM = 64


def _make_cache(num_positions=NUM_POSITIONS, rotary_dim=ROTARY_DIM, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_positions, rotary_dim, generator=generator, dtype=torch.float32).to(torch.bfloat16)


def _make_owner(cos_sin_cache):
    owner = nn.Module()
    owner.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)
    return owner


def _register_interleaved(monkeypatch, cos_sin_cache):
    monkeypatch.setattr(rope, "_cos_cache", None)
    monkeypatch.setattr(rope, "_sin_cache", None)
    monkeypatch.setattr(rope, "_cos_sin_flat", None)
    rope._record_cos_and_sin_cache_interleaved(_make_owner(cos_sin_cache), cos_sin_cache)


def _patch_rope_config(monkeypatch, rope_flat_gather):
    monkeypatch.setattr(
        rope, "get_ascend_config", lambda: SimpleNamespace(rope_flat_gather=rope_flat_gather)
    )


class TestFlatRegistration:
    def test_flat_view_row_identity(self, monkeypatch):
        _register_interleaved(monkeypatch, _make_cache())
        assert rope._cos_sin_flat is not None
        assert torch.equal(rope._cos_cache, rope._cos_sin_flat[0::2])
        assert torch.equal(rope._sin_cache, rope._cos_sin_flat[1::2])

    def test_flat_shares_storage_with_half_tables(self, monkeypatch):
        _register_interleaved(monkeypatch, _make_cache())
        assert rope._cos_sin_flat.untyped_storage().data_ptr() == rope._cos_cache.untyped_storage().data_ptr()


class TestFlatGatherEquivalence:
    def _gather(self, positions):
        return rope.get_cos_and_sin_mla(positions)

    def test_bitwise_equal_with_duplicates(self, monkeypatch):
        _register_interleaved(monkeypatch, _make_cache())
        _patch_rope_config(monkeypatch, rope_flat_gather=True)
        positions = torch.tensor([0, 5, 5, 63, 7, 0], dtype=torch.int64)
        cos_fast, sin_fast = self._gather(positions)
        _patch_rope_config(monkeypatch, rope_flat_gather=False)
        cos_switch_off, sin_switch_off = self._gather(positions)
        assert torch.equal(cos_fast, cos_switch_off)
        assert torch.equal(sin_fast, sin_switch_off)
        monkeypatch.setattr(rope, "_cos_sin_flat", None)
        cos_flat_none, sin_flat_none = self._gather(positions)
        assert torch.equal(cos_fast, cos_flat_none)
        assert torch.equal(sin_fast, sin_flat_none)
        assert cos_fast.shape == (6, 1, 1, ROTARY_DIM)
        assert cos_fast.dtype == torch.bfloat16

    def test_bitwise_equal_all_positions(self, monkeypatch):
        _register_interleaved(monkeypatch, _make_cache())
        _patch_rope_config(monkeypatch, rope_flat_gather=True)
        positions = torch.arange(NUM_POSITIONS, dtype=torch.int64)
        cos_fast, sin_fast = self._gather(positions)
        _patch_rope_config(monkeypatch, rope_flat_gather=False)
        cos_base, sin_base = self._gather(positions)
        assert torch.equal(cos_fast, cos_base)
        assert torch.equal(sin_fast, sin_base)

    def test_bitwise_equal_empty_positions(self, monkeypatch):
        _register_interleaved(monkeypatch, _make_cache())
        _patch_rope_config(monkeypatch, rope_flat_gather=True)
        positions = torch.empty(0, dtype=torch.int64)
        cos_fast, sin_fast = self._gather(positions)
        monkeypatch.setattr(rope, "_cos_sin_flat", None)
        cos_base, sin_base = self._gather(positions)
        assert cos_fast.shape == (0, 1, 1, ROTARY_DIM)
        assert torch.equal(cos_fast, cos_base)
        assert torch.equal(sin_fast, sin_base)

    def test_bitwise_equal_boundary_position(self, monkeypatch):
        _register_interleaved(monkeypatch, _make_cache())
        _patch_rope_config(monkeypatch, rope_flat_gather=True)
        positions = torch.tensor([NUM_POSITIONS - 1], dtype=torch.int64)
        cos_fast, sin_fast = self._gather(positions)
        _patch_rope_config(monkeypatch, rope_flat_gather=False)
        cos_base, sin_base = self._gather(positions)
        assert torch.equal(cos_fast, cos_base)
        assert torch.equal(sin_fast, sin_base)

    def test_use_cache_path_bitwise_equal(self, monkeypatch):
        _register_interleaved(monkeypatch, _make_cache())
        monkeypatch.setattr(
            rope, "_cos_mla", torch.zeros(NUM_POSITIONS, 1, 1, ROTARY_DIM, dtype=torch.bfloat16)
        )
        monkeypatch.setattr(
            rope, "_sin_mla", torch.zeros(NUM_POSITIONS, 1, 1, ROTARY_DIM, dtype=torch.bfloat16)
        )
        _patch_rope_config(monkeypatch, rope_flat_gather=True)
        positions = torch.tensor([1, 2, 3], dtype=torch.int64)
        cos_fast = rope.get_cos_and_sin_mla(positions, use_cache=True)[0].clone()
        sin_fast = rope.get_cos_and_sin_mla(positions, use_cache=True)[1].clone()
        _patch_rope_config(monkeypatch, rope_flat_gather=False)
        cos_base = rope.get_cos_and_sin_mla(positions, use_cache=True)[0].clone()
        sin_base = rope.get_cos_and_sin_mla(positions, use_cache=True)[1].clone()
        assert torch.equal(cos_fast, cos_base)
        assert torch.equal(sin_fast, sin_base)


class TestRegistrationOverride:
    def test_non_interleaved_overwrite_disables_fast_path(self, monkeypatch):
        _register_interleaved(monkeypatch, _make_cache())
        assert rope._cos_sin_flat is not None
        new_cos = torch.randn(NUM_POSITIONS, ROTARY_DIM).to(torch.bfloat16)
        new_sin = torch.randn(NUM_POSITIONS, ROTARY_DIM).to(torch.bfloat16)
        rope._record_cos_and_sin_cache(new_cos, new_sin)
        assert rope._cos_sin_flat is None
        assert rope._cos_cache is new_cos
        assert rope._sin_cache is new_sin

    def test_interleaved_registration_is_first_wins(self, monkeypatch):
        _register_interleaved(monkeypatch, _make_cache(seed=1))
        first_cache = rope._cos_cache
        second = _make_cache(seed=2)
        rope._record_cos_and_sin_cache_interleaved(_make_owner(second), second)
        assert rope._cos_cache is first_cache
