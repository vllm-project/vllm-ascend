# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for mixed-causal DFlash draft layers in one KV cache group.

The pinned upstream DFlash speculator stores draft causality per KV cache
group, so the last layer's flag would win for the whole group. Ascend DFlash2
MLA draft layers publish identical KV cache specs and collapse into one group
even for mixed-causal checkpoints; the validation must reject that instead of
silently corrupting draft attention (review follow-up on PR #21675).
"""

from types import SimpleNamespace

import pytest

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    validate_grouped_draft_causal,
)


def _kv_cache_config(*groups):
    return SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=list(names)) for names in groups])


def _draft_model(layer_names, causal):
    return SimpleNamespace(
        get_draft_kv_cache_layer_names=lambda: list(layer_names),
        get_draft_attn_causal=lambda: list(causal),
    )


class TestValidateGroupedDraftCausal:
    def test_homogeneous_causal_layers_pass(self):
        model = _draft_model(["a", "b", "c"], [True, True, True])
        validate_grouped_draft_causal(model, _kv_cache_config(["a", "b", "c"]))

    def test_homogeneous_non_causal_layers_pass(self):
        model = _draft_model(["a", "b"], [False, False])
        validate_grouped_draft_causal(model, _kv_cache_config(["a", "b"]))

    def test_mixed_causal_in_one_group_raises(self):
        model = _draft_model(["a", "b", "c"], [True, False, True])
        with pytest.raises(ValueError, match="mixed attention causality"):
            validate_grouped_draft_causal(model, _kv_cache_config(["a", "b", "c"]))

    def test_mixed_causal_split_across_groups_passes(self):
        model = _draft_model(["a", "b", "c"], [True, False, True])
        validate_grouped_draft_causal(model, _kv_cache_config(["a", "c"], ["b"]))

    def test_model_without_hooks_is_skipped(self):
        model = SimpleNamespace()
        validate_grouped_draft_causal(model, _kv_cache_config(["a"]))

    def test_unknown_layer_names_are_ignored(self):
        model = _draft_model(["x"], [True])
        validate_grouped_draft_causal(model, _kv_cache_config(["a"]))
