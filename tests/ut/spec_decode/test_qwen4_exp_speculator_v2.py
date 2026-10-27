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
"""QSA (Qwen3.8-Flash-Next) wiring in the V2 autoregressive speculator."""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.models.qwen4_exp.qsa import AscendQSABackend
from vllm_ascend.worker.v2.spec_decode.autoregressive import speculator as speculator_module
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)


def _speculator_with_backend(monkeypatch, backend, model=None):
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.model = model if model is not None else SimpleNamespace(forward=MagicMock())
    speculator.attn_vllm_config = object()
    speculator.attn_groups = object()
    monkeypatch.setattr(speculator_module, "set_current_vllm_config", lambda _cfg: contextlib.nullcontext())
    monkeypatch.setattr(speculator_module, "_get_graph_update_backend", lambda _groups: backend)
    monkeypatch.setattr(speculator_module.AutoRegressiveSpeculator, "set_attn", MagicMock())
    return speculator


def test_set_attn_classifies_qsa_and_binds_draft_context_flag(monkeypatch):
    speculator = _speculator_with_backend(monkeypatch, AscendQSABackend)
    original_forward = speculator.model.forward

    speculator.set_attn(None, None, None, None, None)

    assert speculator.attn_architecture == "QSA"
    assert speculator.model.forward is not original_forward


def test_draft_context_flag_tags_context_and_delegates_to_original_forward(monkeypatch):
    sentinel_output = object()
    captured = {}

    def original_forward(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel_output

    model = SimpleNamespace(forward=original_forward)
    speculator = _speculator_with_backend(monkeypatch, AscendQSABackend, model=model)

    speculator.set_attn(None, None, None, None, None)

    forward_context = SimpleNamespace()
    monkeypatch.setattr(speculator_module, "get_forward_context", lambda: forward_context)
    result = speculator.model.forward(3, 4, step=5)

    assert result is sentinel_output
    # The QSA draft RoPE bypass reads this flag; upstream set_forward_context
    # never sets it on the V2 runner, so the speculator must tag every draft
    # forward itself.
    assert forward_context.is_draft_model is True
    assert captured["args"] == (3, 4)
    assert captured["kwargs"] == {"step": 5}


def test_set_attn_keeps_gqa_classification_without_binding(monkeypatch):
    speculator = _speculator_with_backend(monkeypatch, AscendAttentionBackend)
    original_forward = speculator.model.forward

    speculator.set_attn(None, None, None, None, None)

    assert speculator.attn_architecture == "GQA"
    # Only QSA drafts need the context flag; other architectures keep the
    # model's forward untouched.
    assert speculator.model.forward is original_forward


def test_set_attn_rejects_unsupported_backend(monkeypatch):
    class UnknownBackend:
        pass

    speculator = _speculator_with_backend(monkeypatch, UnknownBackend)

    with pytest.raises(ValueError, match="Unsupported attention backend"):
        speculator.set_attn(None, None, None, None, None)


def test_update_decode_attn_metadata_skips_host_patch_for_qsa():
    """QSA has no host-side FIA fields to patch: the update must return
    before touching any per-layer metadata."""
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.attn_architecture = "QSA"
    metadata = MagicMock()

    result = speculator._update_decode_attn_metadata({"layer.0": metadata}, step=1, num_reqs=2)

    assert result is None
    assert not metadata.mock_calls


def test_qsa_backend_is_descriptor_native():
    """The hybrid spec builder routes descriptor-native owners through the
    mamba-page alignment branch keyed by this flag."""
    assert getattr(AscendQSABackend, "uses_descriptor_cache_view", False) is True
