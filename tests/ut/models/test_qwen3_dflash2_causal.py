# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Regression tests for DFlash2 MLA draft attention causality resolution.

The draft causality must come from the checkpoint config, not from layer
attributes that Ascend attention modules never set. Before this fix every
MLA DFlash2 draft layer was silently treated as non-causal, which routed
spec-decode attention through the bidirectional FIA path and collapsed the
token acceptance rate (observed ~2.5 mean acceptance length on
RadixArk/Kimi-K3.1-Dflash2-Preview instead of the expected ~3.5+).
"""

from types import SimpleNamespace

import pytest

pytest.importorskip(
    "vllm.model_executor.models.qwen3_dflash",
    reason="DFlash drafters are unavailable on this vLLM release",
)

from vllm_ascend.models.qwen3_dflash2 import (
    DFlash2Qwen3ForCausalLM,
    _draft_uses_causal_attention,
    _resolve_draft_layer_causal,
)

_NUM_LAYERS = 3
_ALL_SLIDING = ["sliding_attention"] * _NUM_LAYERS
_ALL_FULL = ["full_attention"] * _NUM_LAYERS


def _config(**overrides):
    fields = {
        "num_hidden_layers": _NUM_LAYERS,
        "dflash_config": {},
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestResolveDraftLayerCausal:
    def test_top_level_is_causal_wins(self):
        config = _config(is_causal=True, layer_types=_ALL_FULL)
        assert all(_resolve_draft_layer_causal(config, i) for i in range(_NUM_LAYERS))

    def test_top_level_is_causal_false_wins(self):
        config = _config(is_causal=False, layer_types=_ALL_SLIDING)
        assert not any(_resolve_draft_layer_causal(config, i) for i in range(_NUM_LAYERS))

    def test_dflash_config_causal_override(self):
        config = _config(dflash_config={"causal": False}, layer_types=_ALL_SLIDING)
        assert not any(_resolve_draft_layer_causal(config, i) for i in range(_NUM_LAYERS))

    def test_layer_types_sliding_fallback(self):
        config = _config(layer_types=_ALL_SLIDING)
        assert all(_resolve_draft_layer_causal(config, i) for i in range(_NUM_LAYERS))

    def test_layer_types_full_attention_is_non_causal(self):
        config = _config(layer_types=_ALL_FULL)
        assert not any(_resolve_draft_layer_causal(config, i) for i in range(_NUM_LAYERS))

    def test_legacy_default_without_any_field(self):
        config = _config()
        assert not any(_resolve_draft_layer_causal(config, i) for i in range(_NUM_LAYERS))

    def test_mixed_layer_types_resolve_per_layer(self):
        layer_types = ["sliding_attention", "full_attention", "sliding_attention"]
        config = _config(layer_types=layer_types)
        resolved = [_resolve_draft_layer_causal(config, i) for i in range(_NUM_LAYERS)]
        assert resolved == [True, False, True]


class TestDraftUsesCausalAttention:
    def test_kimi_k31_style_checkpoint_is_causal(self):
        config = _config(is_causal=True, layer_types=_ALL_SLIDING)
        assert _draft_uses_causal_attention(config)

    def test_legacy_non_causal_checkpoint(self):
        assert not _draft_uses_causal_attention(_config())


class TestGetDraftAttnCausal:
    def _draft_model(self, config):
        draft = DFlash2Qwen3ForCausalLM.__new__(DFlash2Qwen3ForCausalLM)
        draft.config = config
        draft.model = SimpleNamespace(layers=[object()] * _NUM_LAYERS)
        return draft

    def test_resolves_from_config_not_layer_attributes(self):
        config = _config(is_causal=True, layer_types=_ALL_SLIDING)
        draft = self._draft_model(config)
        assert draft.get_draft_attn_causal() == [True] * _NUM_LAYERS

    def test_non_causal_checkpoint_stays_non_causal(self):
        config = _config(dflash_config={"causal": False}, layer_types=_ALL_SLIDING)
        draft = self._draft_model(config)
        assert draft.get_draft_attn_causal() == [False] * _NUM_LAYERS

    def test_layer_types_fallback_per_layer(self):
        layer_types = ["sliding_attention", "full_attention", "sliding_attention"]
        draft = self._draft_model(_config(layer_types=layer_types))
        assert draft.get_draft_attn_causal() == [True, False, True]
