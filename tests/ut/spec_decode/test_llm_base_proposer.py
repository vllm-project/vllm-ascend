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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.spec_decode.utils import _disable_flash_comm_v1_context

# CUDAGraphMode values whose ``has_full_cudagraphs()`` is True: FULL plus the
# two composite modes that mix FULL with NONE / PIECEWISE.
FULL_CUDAGRAPH_MODES = [
    CUDAGraphMode.FULL,
    CUDAGraphMode.FULL_DECODE_ONLY,
    CUDAGraphMode.FULL_AND_PIECEWISE,
]

# Modes without a full cudagraph.
NON_FULL_CUDAGRAPH_MODES = [
    CUDAGraphMode.NONE,
    CUDAGraphMode.PIECEWISE,
]


def test_draft_vllm_config_uses_draft_model_config():
    draft_model_config = SimpleNamespace(runner_type="draft")
    base_vllm_config = SimpleNamespace(model_config=SimpleNamespace(runner_type="generate"))
    expected_vllm_config = SimpleNamespace(model_config=draft_model_config)
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.speculative_config = SimpleNamespace(
        draft_model_config=draft_model_config,
    )

    with (
        patch(
            "vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer._create_draft_vllm_config",
            return_value=base_vllm_config,
        ),
        patch(
            "vllm_ascend.spec_decode.llm_base_proposer.replace",
            return_value=expected_vllm_config,
        ) as mock_replace,
    ):
        draft_vllm_config = proposer._create_draft_vllm_config()

    assert draft_vllm_config is expected_vllm_config
    mock_replace.assert_called_once_with(
        base_vllm_config,
        model_config=draft_model_config,
    )


class TestDisablePaddedDrafterBatchWithFullGraph:
    """Guard: ``disable_padded_drafter_batch=True`` + cuda graph + any full
    cudagraph mode must raise ``NotImplementedError``.
    """

    @staticmethod
    def _make_proposer(
        *,
        disable_padded_drafter_batch: bool,
        use_cuda_graph: bool,
        cudagraph_mode: CUDAGraphMode,
    ) -> AscendSpecDecodeBaseProposer:
        """Bypass ``__init__`` and set only the three attrs the guard reads.

        ``cudagraph_mode`` is a real enum value so ``has_full_cudagraphs()`` is
        exercised, not stubbed.
        """
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.speculative_config = SimpleNamespace(
            disable_padded_drafter_batch=disable_padded_drafter_batch,
        )
        proposer.use_cuda_graph = use_cuda_graph
        proposer.compilation_config = SimpleNamespace(cudagraph_mode=cudagraph_mode)
        return proposer

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_raises_when_padded_drafter_batch_disabled_with_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        """The bad combo: disable_padded + cuda graph + any full-cudagraph mode
        is intercepted with ``NotImplementedError``."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        with pytest.raises(NotImplementedError, match="disable_padded_drafter_batch"):
            proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", NON_FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_without_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        """NONE / PIECEWISE never trip the guard, even with disable_padded + cuda graph."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        # Must not raise.
        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_when_padded_drafter_batch_enabled(self, cudagraph_mode: CUDAGraphMode):
        """Padded drafter batch on (the default) is fine with any full cudagraph."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=False,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    def test_guard_does_not_raise_when_eager(self):
        """``enforce_eager`` -> ``use_cuda_graph=False`` short-circuits the guard."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=False,
            cudagraph_mode=CUDAGraphMode.FULL,
        )

        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()


class TestDisableFlashCommV1Context:
    """``_disable_flash_comm_v1_context`` temporarily clears
    ``forward_context.flash_comm_v1_enabled`` while MarkovHead runs -- MarkovHead
    operates in the all-gathered full space, so SP's reduce-scatter must not
    split ``markov_emb`` -- then restores the original value on exit, including
    on exception. See commit c62ef687b ([BugFix] Fix `sp` in dspark).
    """

    @staticmethod
    def _patch_forward_context(monkeypatch, flash_comm_v1_enabled: bool):
        ctx = SimpleNamespace(flash_comm_v1_enabled=flash_comm_v1_enabled)
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.utils.get_forward_context",
            lambda: ctx,
        )
        return ctx

    def test_clears_while_inside_when_sp_on(self, monkeypatch):
        ctx = self._patch_forward_context(monkeypatch, True)
        with _disable_flash_comm_v1_context():
            assert ctx.flash_comm_v1_enabled is False

    def test_restores_true_on_exit(self, monkeypatch):
        ctx = self._patch_forward_context(monkeypatch, True)
        with _disable_flash_comm_v1_context():
            pass
        assert ctx.flash_comm_v1_enabled is True

    def test_restores_false_on_exit(self, monkeypatch):
        """SP already off -> clearing is a no-op, original False preserved."""
        ctx = self._patch_forward_context(monkeypatch, False)
        with _disable_flash_comm_v1_context():
            assert ctx.flash_comm_v1_enabled is False
        assert ctx.flash_comm_v1_enabled is False

    def test_restores_on_exception(self, monkeypatch):
        ctx = self._patch_forward_context(monkeypatch, True)
        with pytest.raises(RuntimeError, match="boom"), _disable_flash_comm_v1_context():
            raise RuntimeError("boom")
        assert ctx.flash_comm_v1_enabled is True


class TestQuaRotDraftBoundaries:
    """CPU tests for QuaRot alignment at every draft-model boundary."""

    @staticmethod
    def _make_proposer() -> AscendSpecDecodeBaseProposer:
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.method = "dspark"
        proposer.device = torch.device("cpu")
        proposer.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model="target"))
        return proposer

    def test_anti_rotates_each_aux_hidden_state_fc_block(self, monkeypatch):
        proposer = self._make_proposer()
        rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        fc = nn.Linear(10, 2, bias=False)
        initial_weight = torch.arange(1.0, 21.0).reshape(2, 10)
        fc.weight.data.copy_(initial_weight)
        proposer.model = SimpleNamespace(model=SimpleNamespace(fc=fc))
        monkeypatch.setattr(proposer, "_load_quarot_rotation", lambda _: rotation)

        proposer._maybe_anti_rotate_fc()

        expected = torch.matmul(initial_weight.reshape(2, 5, 2), rotation).reshape(2, 10)
        torch.testing.assert_close(fc.weight, expected)
        torch.testing.assert_close(proposer._quarot_rotation, rotation)

    def test_anti_rotates_k3_mla_context_projection(self, monkeypatch):
        proposer = self._make_proposer()
        rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        context_proj = nn.Linear(10, 2, bias=False)
        initial_weight = torch.arange(1.0, 21.0).reshape(2, 10)
        context_proj.weight.data.copy_(initial_weight)
        proposer.model = SimpleNamespace(model=SimpleNamespace(context_proj=context_proj))
        monkeypatch.setattr(proposer, "_load_quarot_rotation", lambda _: rotation)

        proposer._maybe_anti_rotate_fc()

        expected = torch.matmul(initial_weight.reshape(2, 5, 2), rotation).reshape(2, 10)
        torch.testing.assert_close(context_proj.weight, expected)
        torch.testing.assert_close(proposer._quarot_rotation, rotation)

    def test_copies_unrotated_shared_weight_without_mutating_target(self):
        proposer = self._make_proposer()
        proposer._quarot_rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        target = nn.Linear(2, 3, bias=False)
        draft = nn.Linear(2, 3, bias=False)
        target_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        target.weight.data.copy_(target_weight)
        original_target = target.weight.detach().clone()

        copied = proposer._copy_unrotated_shared_weight(draft, target, "draft embed_tokens.weight")

        assert copied is True
        torch.testing.assert_close(draft.weight, target_weight @ proposer._quarot_rotation.T)
        torch.testing.assert_close(target.weight, original_target)

    def test_shared_weight_shape_mismatch_keeps_draft_weight(self):
        proposer = self._make_proposer()
        proposer._quarot_rotation = torch.eye(2)
        target = nn.Linear(2, 3, bias=False)
        draft = nn.Linear(3, 3, bias=False)
        original_draft = draft.weight.detach().clone()

        copied = proposer._copy_unrotated_shared_weight(draft, target, "draft lm_head.weight")

        assert copied is False
        torch.testing.assert_close(draft.weight, original_draft)

    def test_materializes_unrotated_layer_for_none_placeholder(self):
        proposer = self._make_proposer()
        proposer._quarot_rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        target = nn.Linear(2, 3, bias=False)
        target_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        target.weight.data.copy_(target_weight)
        original_target = target.weight.detach().clone()

        prepared = proposer._prepare_unrotated_shared_layer(
            None,
            target,
            "draft embed_tokens.weight",
        )

        assert prepared is not None
        assert prepared is not target
        torch.testing.assert_close(
            prepared.weight,
            target_weight @ proposer._quarot_rotation.T,
        )
        torch.testing.assert_close(target.weight, original_target)

    def test_materialized_layer_reuses_noncopyable_comm_group(self):
        class NonCopyableCommGroup:
            def __deepcopy__(self, memo):
                del memo
                raise TypeError("cannot pickle ProcessGroup")

        proposer = self._make_proposer()
        proposer._quarot_rotation = torch.eye(2)
        target = nn.Linear(2, 3, bias=False)
        target.comm_group = NonCopyableCommGroup()

        prepared = proposer._prepare_unrotated_shared_layer(
            None,
            target,
            "draft embed_tokens.weight",
        )

        assert prepared is not None
        assert prepared.comm_group is target.comm_group
        assert prepared.weight is not target.weight
        assert prepared.weight.data_ptr() != target.weight.data_ptr()
        torch.testing.assert_close(prepared.weight, target.weight)

    def test_incompatible_quarot_shared_layer_fails_instead_of_aliasing(self):
        proposer = self._make_proposer()
        proposer._quarot_rotation = torch.eye(2)
        target = nn.Linear(2, 3, bias=False)
        draft = nn.Linear(3, 3, bias=False)

        with pytest.raises(ValueError, match="refusing to alias"):
            proposer._prepare_unrotated_shared_layer(
                draft,
                target,
                "draft lm_head.weight",
            )
