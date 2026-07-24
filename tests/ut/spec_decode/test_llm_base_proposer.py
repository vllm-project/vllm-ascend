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

import pytest
from vllm.config import CUDAGraphMode

from vllm_ascend.spec_decode.llm_base_proposer import (
    AscendSpecDecodeBaseProposer,
    _disable_flash_comm_v1_context,
)

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
            "vllm_ascend.spec_decode.llm_base_proposer.get_forward_context",
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
