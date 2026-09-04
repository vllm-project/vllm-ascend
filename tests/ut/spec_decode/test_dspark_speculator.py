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
"""Unit tests for ``AscendDSparkSpeculator.load_draft_model`` fc rotation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.models.qwen3_dspark import AscendQwen3DSparkForCausalLM
from vllm_ascend.models.qwen3_dspark import _get_draft_rotation_path
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
    DSPARK_AUX_HIDDEN_FORMAT_MATERIALIZED,
    DSPARK_AUX_HIDDEN_FORMAT_RAW,
    AscendDSparkSpeculator,
)

_HIDDEN = 8
_FC_IN = 5 * _HIDDEN  # concatenated aux hidden states
def _spec(vllm_config: SimpleNamespace) -> AscendDSparkSpeculator:
    """Bypass the heavy ``__init__``; ``load_draft_model`` only reads
    ``self.vllm_config`` and the patched parent call."""
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.vllm_config = vllm_config
    return spec


def _fake_draft() -> SimpleNamespace:
    fc = torch.nn.Linear(_FC_IN, _HIDDEN, bias=False)
    with torch.no_grad():
        fc.weight.copy_(torch.randn_like(fc.weight))
    return SimpleNamespace(model=SimpleNamespace(fc=fc))


def _quarot_config() -> SimpleNamespace:
    quarot = {"rotation_map": {"global_rotation": "x.safetensors"}}
    return SimpleNamespace(
        quant_config=SimpleNamespace(quant_description={"optional": {"quarot": quarot}}),
        model_config=SimpleNamespace(model="/fake"),
    )


def _bf16_config() -> SimpleNamespace:
    return SimpleNamespace(quant_config=None, model_config=SimpleNamespace())


class TestLoadDraftModel:
    """``load_draft_model`` rotates fc for a QuaRot target and is a no-op otherwise."""

    @pytest.fixture
    def captured(self, monkeypatch):
        """Stub the heavy parent ``load_draft_model`` to return a fake draft and
        snapshot its fc weight before the override mutates it in place."""
        out: dict = {}

        def _load(self, target_model, target_attn_layer_names):
            draft = _fake_draft()
            out["before"] = draft.model.fc.weight.data.clone()
            out["draft"] = draft
            return draft

        monkeypatch.setattr(DSparkSpeculator, "load_draft_model", _load)
        return out

    def test_noop_for_bf16_target(self, captured, monkeypatch):
        draft = _spec(_bf16_config()).load_draft_model(MagicMock(), set())
        assert torch.equal(draft.model.fc.weight.data, captured["before"])

    def test_configures_capture_before_loading_draft(self, monkeypatch):
        events = []
        target = SimpleNamespace(
            set_dspark_aux_capture_materialized=lambda enabled: events.append(
                ("capture", enabled)
            )
        )

        def _load(self, target_model, target_attn_layer_names):
            events.append(("load", None))
            return _fake_draft()

        monkeypatch.setattr(DSparkSpeculator, "load_draft_model", _load)
        spec = _spec(_bf16_config())
        spec.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                dspark_aux_hidden_state_format=DSPARK_AUX_HIDDEN_FORMAT_MATERIALIZED
            )
        )

        spec.load_draft_model(target, set())

        assert events == [("capture", True), ("load", None)]

    def test_injects_target_rotation_before_draft_load(self, monkeypatch):
        draft_hf_config = SimpleNamespace(
            architectures=["Qwen3DSparkModel"],
            model_type="qwen3",
            dspark_aux_hidden_state_format=DSPARK_AUX_HIDDEN_FORMAT_MATERIALIZED,
        )
        target_embed = object()
        target_head = object()
        target = SimpleNamespace(
            model=SimpleNamespace(embed_tokens=target_embed),
            lm_head=target_head,
            set_dspark_aux_capture_materialized=lambda enabled: None,
        )
        draft = SimpleNamespace(
            model=SimpleNamespace(embed_tokens=object()),
            lm_head=object(),
            has_own_embed_tokens=True,
            has_own_lm_head=True,
            dspark_aux_hidden_state_format=DSPARK_AUX_HIDDEN_FORMAT_MATERIALIZED,
        )

        def _load(self, target_model, target_attn_layer_names):
            assert draft_hf_config._ascend_target_rotation_path == "/rotation"
            return draft

        monkeypatch.setattr(DSparkSpeculator, "load_draft_model", _load)
        monkeypatch.setattr(
            "vllm_ascend.worker.v2.spec_decode.dspark.speculator.get_rotation_path",
            lambda config: "/rotation",
        )
        spec = _spec(_quarot_config())
        spec.draft_model_config = SimpleNamespace(hf_config=draft_hf_config)

        assert spec.load_draft_model(target, set()) is draft
        assert not hasattr(draft_hf_config, "_ascend_target_rotation_path")

    def test_rejects_shared_quarot_embedding(self, monkeypatch):
        draft_hf_config = SimpleNamespace(
            architectures=["Qwen3DSparkModel"],
            model_type="qwen3",
            dspark_aux_hidden_state_format=DSPARK_AUX_HIDDEN_FORMAT_MATERIALIZED,
        )
        shared_embed = object()
        target = SimpleNamespace(
            model=SimpleNamespace(embed_tokens=shared_embed),
            lm_head=object(),
            set_dspark_aux_capture_materialized=lambda enabled: None,
        )
        draft = SimpleNamespace(
            model=SimpleNamespace(embed_tokens=shared_embed),
            lm_head=object(),
            has_own_embed_tokens=False,
            has_own_lm_head=True,
            dspark_aux_hidden_state_format=DSPARK_AUX_HIDDEN_FORMAT_MATERIALIZED,
        )
        monkeypatch.setattr(
            DSparkSpeculator,
            "load_draft_model",
            lambda self, target_model, target_attn_layer_names: draft,
        )
        monkeypatch.setattr(
            "vllm_ascend.worker.v2.spec_decode.dspark.speculator.get_rotation_path",
            lambda config: "/rotation",
        )
        spec = _spec(_quarot_config())
        spec.draft_model_config = SimpleNamespace(hf_config=draft_hf_config)

        with pytest.raises(RuntimeError, match="must not share target embed_tokens"):
            spec.load_draft_model(target, set())


def test_injected_rotation_path_does_not_require_draft_quant_config():
    config = SimpleNamespace(_ascend_target_rotation_path="/rotation")
    vllm_config = SimpleNamespace(quant_config=None)

    assert _get_draft_rotation_path(vllm_config, config) == Path("/rotation")


def test_process_weight_orthogonal_basis_equivalence():
    from vllm_ascend.models.qwen3_dspark import process_weight

    generator = torch.Generator().manual_seed(7)
    q, _ = torch.linalg.qr(
        torch.randn(_HIDDEN, _HIDDEN, dtype=torch.float64, generator=generator)
    )
    x = torch.randn(3, 5, _HIDDEN, dtype=torch.float64, generator=generator)
    weight = torch.randn(_HIDDEN, _FC_IN, dtype=torch.float64, generator=generator)
    x_rotated = x @ q

    expected = torch.nn.functional.linear(x.reshape(3, _FC_IN), weight)
    fused_weight = process_weight(weight, q)
    actual = torch.nn.functional.linear(x_rotated.reshape(3, _FC_IN), fused_weight)

    # process_weight intentionally performs the fusion in FP32 even when this
    # test builds the reference in FP64.
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-6)


class TestAuxHiddenStateFormatContract:
    def test_qwen3_gqa_draft_declares_materialized_format(self):
        assert (
            AscendQwen3DSparkForCausalLM.dspark_aux_hidden_state_format
            == DSPARK_AUX_HIDDEN_FORMAT_MATERIALIZED
        )

    @pytest.mark.parametrize(
        ("aux_hidden_format", "expected_materialized"),
        [
            (DSPARK_AUX_HIDDEN_FORMAT_MATERIALIZED, True),
            (DSPARK_AUX_HIDDEN_FORMAT_RAW, False),
        ],
    )
    def test_configures_target_capture_mode(
        self,
        aux_hidden_format,
        expected_materialized,
    ):
        set_capture_mode = MagicMock()
        target = SimpleNamespace(
            set_dspark_aux_capture_materialized=set_capture_mode,
        )
        draft = SimpleNamespace(
            dspark_aux_hidden_state_format=aux_hidden_format,
        )

        actual = AscendDSparkSpeculator._configure_target_aux_hidden_state_format(
            target,
            draft,
        )

        assert actual == aux_hidden_format
        set_capture_mode.assert_called_once_with(expected_materialized)

    @pytest.mark.parametrize(
        "config",
        [
            SimpleNamespace(architectures=["Qwen3DSparkModel"]),
            SimpleNamespace(architectures=["DSparkDraftModel"], model_type="qwen3"),
            SimpleNamespace(model_type="qwen3"),
        ],
    )
    def test_qwen3_config_declares_materialized(self, config):
        set_capture_mode = MagicMock()
        target = SimpleNamespace(
            set_dspark_aux_capture_materialized=set_capture_mode,
        )

        actual = AscendDSparkSpeculator._configure_target_aux_hidden_state_format(
            target,
            config,
        )

        assert actual == DSPARK_AUX_HIDDEN_FORMAT_MATERIALIZED
        set_capture_mode.assert_called_once_with(True)

    def test_undeclared_format_preserves_existing_target_behavior(self):
        set_capture_mode = MagicMock()
        target = SimpleNamespace(
            set_dspark_aux_capture_materialized=set_capture_mode,
        )

        actual = AscendDSparkSpeculator._configure_target_aux_hidden_state_format(
            target,
            SimpleNamespace(),
        )

        assert actual is None
        set_capture_mode.assert_not_called()

    def test_rejects_unknown_format(self):
        with pytest.raises(ValueError, match="Unsupported DSpark auxiliary"):
            AscendDSparkSpeculator._configure_target_aux_hidden_state_format(
                SimpleNamespace(),
                SimpleNamespace(dspark_aux_hidden_state_format="unknown"),
            )
