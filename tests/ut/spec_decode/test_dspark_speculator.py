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
"""Unit tests for ``AscendDSparkSpeculator.load_draft_model``.

The override rotates the drafter's ``fc`` projection after load so it matches a
QuaRot-quantized target's rotated aux hidden states: upstream
``load_dspark_model`` overrides the drafter's ``quant_config`` with
``get_draft_quant_config`` (None for a bf16 drafter), so the drafter's
``__init__`` derives ``rotation_path=None`` and ``fc`` is loaded unrotated. The
speculator still holds the target's ``vllm_config`` (``quant_config=QuaRot``),
so the override recomputes ``rotation_path`` from it and rotates ``fc`` in
place. These tests pin that behaviour and the CPU-side matmul that avoids a
cross-device (NPU vs CPU) ``torch.matmul``.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
    AscendDSparkSpeculator,
)

# Small shapes that mirror the real layout (fc is [hidden, num_aux * hidden]
# for the concatenated aux hidden states) but keep the CPU matmul trivial.
_HIDDEN_SIZE = 8
_NUM_AUX_LAYERS = 5
_FC_IN = _NUM_AUX_LAYERS * _HIDDEN_SIZE

# The speculator module binds get_rotation_matrix at import time; patch it there
# so load_draft_model sees the stub instead of reading a rotation file.
_ROTATION_MATRIX_ATTR = (
    "vllm_ascend.worker.v2.spec_decode.dspark.speculator.get_rotation_matrix"
)


def _quarot_vllm_config() -> SimpleNamespace:
    """A vllm_config whose quant_config exposes a QuaRot global_rotation, so
    the real ``get_rotation_path`` returns a non-None path. The matrix itself
    is stubbed per-test, so no rotation file is read from disk."""
    quant_config = SimpleNamespace(
        quant_description={
            "optional": {
                "quarot": {
                    "rotation_map": {
                        "global_rotation": "optional/quarot.safetensors"
                    }
                }
            }
        }
    )
    return SimpleNamespace(
        quant_config=quant_config,
        model_config=SimpleNamespace(model="/fake/target"),
    )


def _bf16_vllm_config() -> SimpleNamespace:
    """A bf16 target: ``quant_config`` is None, so ``get_rotation_path``
    returns None immediately (the no-op path)."""
    return SimpleNamespace(quant_config=None, model_config=SimpleNamespace())


def _non_quarot_vllm_config() -> SimpleNamespace:
    """A quantized-but-not-QuaRot target: ``quant_config`` is set but its
    description has no ``quarot.rotation_map``, so ``get_rotation_path``
    returns None via KeyError (the other no-op path)."""
    quant_config = SimpleNamespace(quant_description={"optional": {}})
    return SimpleNamespace(
        quant_config=quant_config,
        model_config=SimpleNamespace(model="/fake/target"),
    )


def _make_speculator(vllm_config: SimpleNamespace) -> AscendDSparkSpeculator:
    """Bypass the heavy ``DSparkSpeculator.__init__``; ``load_draft_model``
    only reads ``self.vllm_config`` and the (patched) parent call."""
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.vllm_config = vllm_config
    return spec


def _fake_draft_model() -> SimpleNamespace:
    """A draft model exposing ``model.fc`` as a non-sharded ``nn.Linear`` whose
    ``weight.data`` ``load_draft_model`` can rotate in place, mirroring the
    real ``DFlashQwen3Model.fc`` ``ReplicatedLinear``."""
    fc = torch.nn.Linear(_FC_IN, _HIDDEN_SIZE, bias=False)
    with torch.no_grad():
        fc.weight.copy_(torch.randn_like(fc.weight))
    return SimpleNamespace(model=SimpleNamespace(fc=fc))


class _LoadDraftModelTestBase:
    """Shared fixtures for ``load_draft_model`` rotation tests."""

    @pytest.fixture
    def stub_parent_load(self, monkeypatch):
        """Replace the heavy parent ``load_draft_model`` (which builds/loads a
        real drafter via ``load_dspark_model``) with one returning a fresh fake
        draft model, and snapshot its ``fc`` weight before the override can
        mutate it in place."""

        captured: dict = {}

        def _fake_load(
            self, target_model, target_attn_layer_names
        ) -> SimpleNamespace:
            draft = _fake_draft_model()
            captured["fc_before"] = draft.model.fc.weight.data.clone()
            captured["draft"] = draft
            return draft

        monkeypatch.setattr(DSparkSpeculator, "load_draft_model", _fake_load)
        return captured


# fmt: off
class TestLoadDraftModelRotatesFC(_LoadDraftModelTestBase):
    """``load_draft_model`` rotates the drafter's ``fc`` projection to match a
    QuaRot-quantized target's rotated aux hidden states, and is a no-op for
    bf16 / non-QuaRot targets (``get_rotation_path`` returns None)."""

    def test_rotates_fc_for_quarot_target(self, stub_parent_load, monkeypatch):
        """QuaRot target: ``fc.weight`` is overwritten with ``W @ R`` and the
        returned model is the same object the parent loaded."""
        spec = _make_speculator(_quarot_vllm_config())
        # diag(2): W @ (2*I) == 2 * W, an expected value that is independent of
        # process_weight's internal chunked matmul.
        rotation = torch.eye(_HIDDEN_SIZE, dtype=torch.float32) * 2.0
        monkeypatch.setattr(_ROTATION_MATRIX_ATTR, lambda rotation_path: rotation)

        draft = spec.load_draft_model(
            target_model=MagicMock(), target_attn_layer_names=set()
        )

        assert draft is stub_parent_load["draft"]
        rotated = draft.model.fc.weight.data
        assert rotated.shape == (_HIDDEN_SIZE, _FC_IN)
        fc_before = stub_parent_load["fc_before"]
        assert torch.allclose(rotated, 2.0 * fc_before, atol=1e-6)
        # the weight actually changed (R is not the identity).
        assert not torch.allclose(rotated, fc_before)

    def test_noop_for_bf16_target(self, stub_parent_load, monkeypatch):
        """bf16 target (``quant_config=None``): ``get_rotation_path`` is None,
        so ``fc`` is left exactly as loaded and the matrix is never read."""
        spec = _make_speculator(_bf16_vllm_config())

        def _fail(*args, **kwargs):
            raise AssertionError(
                "get_rotation_matrix must not be called for a bf16 target"
            )

        monkeypatch.setattr(_ROTATION_MATRIX_ATTR, _fail)

        draft = spec.load_draft_model(
            target_model=MagicMock(), target_attn_layer_names=set()
        )

        fc_before = stub_parent_load["fc_before"]
        assert torch.equal(draft.model.fc.weight.data, fc_before)

    def test_noop_for_non_quarot_quant(self, stub_parent_load, monkeypatch):
        """Quantized-but-not-QuaRot target (no ``quarot.rotation_map``):
        ``get_rotation_path`` returns None via KeyError, so ``fc`` is untouched
        and the matrix is never read."""
        spec = _make_speculator(_non_quarot_vllm_config())

        def _fail(*args, **kwargs):
            raise AssertionError(
                "get_rotation_matrix must not be called for a non-QuaRot target"
            )

        monkeypatch.setattr(_ROTATION_MATRIX_ATTR, _fail)

        draft = spec.load_draft_model(
            target_model=MagicMock(), target_attn_layer_names=set()
        )

        fc_before = stub_parent_load["fc_before"]
        assert torch.equal(draft.model.fc.weight.data, fc_before)

    def test_rotation_runs_on_cpu(self):
        """The drafter is already on device after load while the rotation
        matrix is loaded from disk to CPU, so the matmul must take ``fc`` on CPU
        (``fc.weight.data.cpu()``) and derive ``rotation_path`` from the
        target's ``vllm_config``. A refactor that drops the ``.cpu()`` regresses
        the cross-device matmul, so fail loudly here rather than silently."""
        src = inspect.getsource(AscendDSparkSpeculator.load_draft_model)
        assert "process_weight(fc.weight.data.cpu()" in src
        assert "get_rotation_path(self.vllm_config)" in src
# fmt: on
