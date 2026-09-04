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
"""Unit tests for Ascend DSpark model loading and target contracts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.models.dspark_aux import (
    DSparkAuxHiddenContract,
    DSparkAuxHiddenFormat,
)
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
    AscendDSparkSpeculator,
)

_HIDDEN = 8
_FC_IN = 5 * _HIDDEN  # concatenated aux hidden states
# Patch where load_draft_model looks it up (the speculator module binding).
_ROT_MATRIX = "vllm_ascend.worker.v2.spec_decode.dspark.speculator.get_rotation_matrix"


def _spec(vllm_config: SimpleNamespace) -> AscendDSparkSpeculator:
    """Bypass the heavy ``__init__``; ``load_draft_model`` only reads
    ``self.vllm_config`` and the patched parent call."""
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.vllm_config = vllm_config
    spec.aux_hidden_contract = None
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


def _no_call(*args, **kwargs):
    raise AssertionError("get_rotation_matrix must not be called without a rotation path")


class TestLoadDraftModel:
    """Cover QuaRot post-load handling and Aux Hidden negotiation."""

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

    def test_rotates_fc_for_quarot_target(self, captured, monkeypatch):
        # R = 2*I -> W @ R == 2*W, an expectation independent of process_weight.
        monkeypatch.setattr(_ROT_MATRIX, lambda path: torch.eye(_HIDDEN) * 2.0)
        draft = _spec(_quarot_config()).load_draft_model(MagicMock(), set())
        before = captured["before"]
        assert draft is captured["draft"]
        assert torch.allclose(draft.model.fc.weight.data, 2.0 * before, atol=1e-6)
        assert not torch.allclose(draft.model.fc.weight.data, before)

    def test_noop_for_bf16_target(self, captured, monkeypatch):
        monkeypatch.setattr(_ROT_MATRIX, _no_call)
        draft = _spec(_bf16_config()).load_draft_model(MagicMock(), set())
        assert torch.equal(draft.model.fc.weight.data, captured["before"])

    def test_negotiates_draft_declared_aux_hidden_contract(self, monkeypatch):
        contract = DSparkAuxHiddenContract(
            format=DSparkAuxHiddenFormat.RAW_PREFIX_SUM,
            layer_ids=(1,),
            capture_point="post_layer_raw_prefix_sum",
            target_hidden_size=_HIDDEN,
            dtype=torch.bfloat16,
        )
        draft = _fake_draft()
        draft.get_required_dspark_aux_hidden_state_contract = lambda: contract
        monkeypatch.setattr(
            DSparkSpeculator,
            "load_draft_model",
            lambda *_args: draft,
        )
        target = MagicMock()

        loaded = _spec(_bf16_config()).load_draft_model(target, set())

        assert loaded is draft
        target.configure_dspark_aux_hidden_state_contract.assert_called_once_with(contract)

    def test_rejects_target_without_required_aux_capability(self, monkeypatch):
        contract = DSparkAuxHiddenContract(
            format=DSparkAuxHiddenFormat.RAW_PREFIX_SUM,
            layer_ids=(1,),
            capture_point="post_layer_raw_prefix_sum",
            target_hidden_size=_HIDDEN,
            dtype=torch.bfloat16,
        )
        draft = _fake_draft()
        draft.get_required_dspark_aux_hidden_state_contract = lambda: contract
        monkeypatch.setattr(
            DSparkSpeculator,
            "load_draft_model",
            lambda *_args: draft,
        )

        with pytest.raises(ValueError, match="does not expose"):
            _spec(_bf16_config()).load_draft_model(SimpleNamespace(), set())
