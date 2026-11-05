# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Zero-width (NoPE) rope tables must not enter the module-global registries.

A text-only GLM-5.3-Flash target builds its sparse-indexer rope first, and
that rope is NoPE (rotary_dim == 0). The interleaved registry's de-interleave
reshape (``view(-1, 2, 0)``) raises on the empty table during model
initialization, so the engine cannot start. Multimodal GLM-5.3-Flash
checkpoints mask the crash only by construction order: the vision tower's
real rope records first and the NoPE table exits at the occupied-slot check.

These tests live in their own module: tests/ut/ops/test_rotary_embedding.py
patches the two record functions for every test via an autouse fixture.
"""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

import vllm_ascend.ops.rotary_embedding as rope_module
from vllm_ascend.ops.rotary_embedding import (
    AscendRotaryEmbedding,
    _record_cos_and_sin_cache_interleaved,
    _record_cos_sin_cache,
)

HEAD_SIZE = 64
MAX_POS = 2048
BASE = 10000.0
DTYPE = torch.bfloat16

REGISTRY_GLOBALS = (
    "_cos_sin_cache",
    "_cos_cache",
    "_sin_cache",
    "_cos_sin_flat",
    "_cos_mla",
    "_sin_mla",
    "_cos",
    "_sin",
    "_cos_slice",
    "_sin_slice",
)


@pytest.fixture(autouse=True)
def _reset_singletons():
    saved = {name: getattr(rope_module, name, None) for name in REGISTRY_GLOBALS}
    for name in REGISTRY_GLOBALS:
        setattr(rope_module, name, None)
    yield
    for name, value in saved.items():
        setattr(rope_module, name, value)


def test_zero_width_rope_builds_without_registering():
    """Regression for the text-only GLM-5.3-Flash startup crash.

    A NoPE indexer rope (rotary_dim == 0) reaches AscendRotaryEmbedding
    through the CustomOp dispatch; the interleaved registry's de-interleave
    reshape used to raise on its empty table. Without the guards this
    construction fails the same way it did in production.
    """
    vllm_config = MagicMock()
    vllm_config.compilation_config.custom_ops = ["all"]

    with set_current_vllm_config(vllm_config):
        emb = RotaryEmbedding(HEAD_SIZE, 0, MAX_POS, BASE, True, DTYPE)

    assert isinstance(emb, AscendRotaryEmbedding)
    assert emb.cos_sin_cache.shape[-1] == 0
    assert rope_module._cos_sin_cache is None
    assert rope_module._cos_cache is None
    assert rope_module._sin_cache is None
    assert rope_module._cos_sin_flat is None


def test_cos_sin_cache_slot_skips_empty_then_accepts_real():
    empty = torch.zeros(8, 0)
    real = torch.zeros(8, 16)

    _record_cos_sin_cache(empty)
    assert rope_module._cos_sin_cache is None
    assert rope_module._cos_sin_flat is None

    _record_cos_sin_cache(real)
    assert rope_module._cos_sin_cache is real


def test_interleaved_pair_slots_skip_empty_then_accept_real():
    empty = torch.zeros(8, 0)
    real = torch.zeros(8, 16)
    owner = torch.nn.Module()

    # The empty table must not reshape-crash, occupy the slots, publish the
    # flat view, or leave derived buffers on the owner module (RFC #16558).
    _record_cos_and_sin_cache_interleaved(owner, empty)
    assert rope_module._cos_cache is None
    assert rope_module._sin_cache is None
    assert rope_module._cos_sin_flat is None
    assert not list(owner.named_buffers())

    _record_cos_and_sin_cache_interleaved(owner, real)
    assert rope_module._cos_cache is not None
    assert rope_module._sin_cache is not None
    assert rope_module._cos_cache.shape[-1] == real.shape[-1]
    assert rope_module._cos_sin_flat is not None
    assert rope_module._cos_sin_flat.shape[0] == 2 * real.shape[0]
    buffers = dict(owner.named_buffers())
    assert buffers["_rope_derived_cos_cache"] is rope_module._cos_cache
    assert buffers["_rope_derived_sin_cache"] is rope_module._sin_cache


def test_empty_then_real_through_both_registries():
    """The production build order: the NoPE indexer rope hits BOTH registries
    first, and the real rope built afterwards must record through both."""
    empty = torch.zeros(8, 0)
    real = torch.zeros(8, 16)

    _record_cos_sin_cache(empty)
    _record_cos_and_sin_cache_interleaved(torch.nn.Module(), empty)
    assert rope_module._cos_sin_cache is None
    assert rope_module._cos_cache is None
    assert rope_module._cos_sin_flat is None

    _record_cos_sin_cache(real)
    _record_cos_and_sin_cache_interleaved(torch.nn.Module(), real)
    assert rope_module._cos_sin_cache is real
    assert rope_module._cos_cache is not None
    assert rope_module._sin_cache is not None
    assert rope_module._cos_sin_flat is not None
