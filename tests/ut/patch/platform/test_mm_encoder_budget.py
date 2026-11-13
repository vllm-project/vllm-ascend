# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the mm encoder budget patch (per-request residency sizing)."""

from unittest.mock import MagicMock

import pytest

import vllm_ascend.patch.platform.patch_mm_encoder_budget as patch_mm_encoder_budget


@pytest.fixture(autouse=True)
def stub_orig_init(monkeypatch):
    """Neutralize the wrapped MultiModalBudget.__init__: only the
    amplification logic added by the patch is under test."""
    monkeypatch.setattr(
        patch_mm_encoder_budget, "_orig_init", lambda self, *args, **kwargs: None
    )


def _make_budget(mm_max_toks_per_item, mm_limits, encoder_cache_size):
    budget = MagicMock()
    budget.mm_max_toks_per_item = mm_max_toks_per_item
    budget.mm_limits = mm_limits
    budget.encoder_cache_size = encoder_cache_size
    return budget


def test_amplifies_cache_to_per_request_residency():
    """Two large images: 16817 tokens/item x 2 must raise the cache floor."""
    budget = _make_budget({"image": 16817}, {"image": 2}, 16817)
    patch_mm_encoder_budget._patched_init(budget)
    assert budget.encoder_cache_size == 16817 * 2


def test_keeps_size_when_residency_fits():
    """Single-item-per-request models must keep the upstream size."""
    budget = _make_budget({"image": 5000}, {"image": 1}, 8192)
    patch_mm_encoder_budget._patched_init(budget)
    assert budget.encoder_cache_size == 8192


def test_never_shrinks_the_cache():
    """A residency below the configured cache must not lower it."""
    budget = _make_budget({"image": 100}, {"image": 2}, 8192)
    patch_mm_encoder_budget._patched_init(budget)
    assert budget.encoder_cache_size == 8192


def test_unlimited_mm_limit_keeps_legacy_sizing():
    """None (unlimited) mm_limit has unbounded residency: fall back to the
    legacy single-item sizing instead of amplifying without bound."""
    budget = _make_budget({"image": 5000}, {"image": None}, 8192)
    patch_mm_encoder_budget._patched_init(budget)
    assert budget.encoder_cache_size == 8192


def test_sums_across_modalities():
    """Multiple tower modalities must all count toward the residency."""
    budget = _make_budget(
        {"image": 1000, "video": 500}, {"image": 2, "video": 3}, 2000
    )
    patch_mm_encoder_budget._patched_init(budget)
    assert budget.encoder_cache_size == 1000 * 2 + 500 * 3


def test_missing_mm_limit_entry_defaults_to_one():
    """A modality absent from mm_limits counts as a single item."""
    budget = _make_budget({"image": 1000}, {}, 500)
    patch_mm_encoder_budget._patched_init(budget)
    assert budget.encoder_cache_size == 1000
