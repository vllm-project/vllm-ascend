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
"""CPU tests for DSpark weight rotation and parallel-draft profiling."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

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


@pytest.mark.parametrize("dummy_run,skip_attn", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("has_dp_sync", [False, True])
@pytest.mark.parametrize("release_vllm", [False, True])
def test_profiling_does_not_reuse_target_dp_counts(dummy_run, skip_attn, has_dp_sync, release_vllm):
    speculator_cls = AscendDSparkSpeculator
    spec = speculator_cls.__new__(speculator_cls)
    spec.input_buffers = SimpleNamespace(positions=torch.empty(0, device="cpu"))
    spec.max_num_tokens = 64
    batch = SimpleNamespace(is_prefilling_np=np.zeros(8, dtype=np.bool_))
    counts = torch.full((2,), 64, dtype=torch.int32, device="cpu")
    dp_sync = SimpleNamespace(num_tokens_across_dp=counts) if has_dp_sync else None
    module = sys.modules[speculator_cls.__module__]
    with (
        patch.object(speculator_cls.__bases__[0], "propose", return_value=object()) as propose,
        patch.object(module, "vllm_version_is", return_value=release_vllm),
        patch.object(module, "build_attn_metadata_wrapper", return_value=nullcontext()),
        patch.object(module, "build_draft_attn_metadata_factory", return_value=nullcontext(), create=True),
    ):
        result = spec.propose(
            batch,
            *([None] * 10),
            dp_sync=dp_sync,
            num_tokens_across_dp=counts,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn,
            is_profile=dummy_run and skip_attn,
        )

    assert result is propose.return_value
    propose.assert_called_once()
    expected_sync = None if dummy_run and skip_attn else (counts if release_vllm else dp_sync)
    assert propose.call_args.args[11] is expected_sync
    assert propose.call_args.args[12:14] == (dummy_run, skip_attn)
    assert torch.equal(counts, torch.full_like(counts, 64))
