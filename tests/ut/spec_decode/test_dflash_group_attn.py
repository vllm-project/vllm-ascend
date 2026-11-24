# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for mixed-mode DFlash draft layers in one KV cache group.

The pinned upstream DFlash speculator stores draft attention metadata per KV
cache group, so the last layer's flag would win for the whole group. Ascend
DFlash2 MLA draft layers publish identical KV cache specs and collapse into
one group even for mixed-mode checkpoints; set_attn validates the resolved
layers at startup and rejects modes the per-group metadata cannot express
(review follow-up on PR #21675).
"""

from types import SimpleNamespace

import pytest

import vllm_ascend.worker.v2.spec_decode.dflash.speculator as spec_module
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import validate_grouped_draft_attn


class FakeMLAImpl:
    sliding_window = None


class FakeDCPImpl(FakeMLAImpl):
    pass


@pytest.fixture
def mla_impls(monkeypatch):
    monkeypatch.setattr(spec_module, "AscendMLAImpl", FakeMLAImpl)
    monkeypatch.setattr(spec_module, "AscendMlaDCPImpl", FakeDCPImpl)


def _kv_cache_config(*groups):
    return SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=list(names)) for names in groups])


def _draft_model(layer_names, causal):
    return SimpleNamespace(
        get_draft_kv_cache_layer_names=lambda: list(layer_names),
        get_draft_attn_causal=lambda: list(causal),
    )


def _mla_layer(window, dcp=False):
    impl = FakeDCPImpl() if dcp else FakeMLAImpl()
    impl.sliding_window = window
    return SimpleNamespace(impl=impl)


class TestValidateGroupedDraftCausal:
    def test_homogeneous_causal_layers_pass(self):
        model = _draft_model(["a", "b", "c"], [True, True, True])
        validate_grouped_draft_attn(model, _kv_cache_config(["a", "b", "c"]), {})

    def test_mixed_causal_in_one_group_raises(self):
        model = _draft_model(["a", "b", "c"], [True, False, True])
        with pytest.raises(ValueError, match="mixed attention causality"):
            validate_grouped_draft_attn(model, _kv_cache_config(["a", "b", "c"]), {})

    def test_mixed_causal_split_across_groups_passes(self):
        model = _draft_model(["a", "b", "c"], [True, False, True])
        validate_grouped_draft_attn(model, _kv_cache_config(["a", "c"], ["b"]), {})

    def test_model_without_hooks_is_skipped(self):
        validate_grouped_draft_attn(SimpleNamespace(), _kv_cache_config(["a"]), {})

    def test_unknown_layer_names_are_ignored(self):
        model = _draft_model(["x"], [True])
        validate_grouped_draft_attn(model, _kv_cache_config(["a"]), {})


class TestValidateWindowedMlaLayers:
    """Windowed MLA layers must be causal and stay off the DCP implementation.

    AscendMLAImpl applies the sparse_mode=4 pre_tokens band only in the
    causal decode branch (the non-causal branch runs sparse_mode=0 with no
    mask), and AscendMlaDCPImpl applies no band at all, so either combination
    would silently attend the full context. Windowed GQA drafts keep the
    upstream per-group behavior and are not affected.
    """

    def test_windowed_causal_mla_layer_passes(self, mla_impls):
        model = _draft_model(["a"], [True])
        validate_grouped_draft_attn(model, _kv_cache_config(["a"]), {"a": _mla_layer(1024)})

    def test_windowed_non_causal_mla_layer_raises(self, mla_impls):
        model = _draft_model(["a"], [False])
        with pytest.raises(ValueError, match="must be causal"):
            validate_grouped_draft_attn(model, _kv_cache_config(["a"]), {"a": _mla_layer(1024)})

    def test_windowed_non_causal_gqa_layer_passes(self, mla_impls):
        model = _draft_model(["a"], [False])
        validate_grouped_draft_attn(
            model, _kv_cache_config(["a"]), {"a": SimpleNamespace(impl=SimpleNamespace(sliding_window=1024))}
        )

    def test_windowed_mla_layer_under_dcp_raises(self, mla_impls):
        model = _draft_model(["a"], [True])
        with pytest.raises(ValueError, match="decode context parallelism"):
            validate_grouped_draft_attn(model, _kv_cache_config(["a"]), {"a": _mla_layer(1024, dcp=True)})

    def test_full_attention_mla_layer_under_dcp_passes(self, mla_impls):
        model = _draft_model(["a"], [True])
        validate_grouped_draft_attn(model, _kv_cache_config(["a"]), {"a": _mla_layer(None, dcp=True)})
