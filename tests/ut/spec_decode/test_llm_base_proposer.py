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
from unittest.mock import Mock

import pytest
import torch
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


class TestDynamicSpeculativeDecoding:
    """Dynamic SD: ``_propose`` must honor the per-step K carried on
    ``scheduler_output.num_spec_tokens_to_schedule``.

    Key contracts:
    * ``self.num_speculative_tokens`` is refreshed to the per-step K so all
      downstream loops / reshapes see it.
    * ``self.decode_threshold`` stays at the configured max (1 + max K): it
      only feeds the FULL-graph ``slicing_length``, which must match the
      static max-K graph bucket.
    * K == 0 returns an empty draft so the target runs a plain decode.
    """

    MAX_K = 3

    @staticmethod
    def _make_proposer() -> AscendSpecDecodeBaseProposer:
        """Bypass ``__init__`` and set only the attrs the DSD entry path reads."""
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.device = torch.device("cpu")
        proposer.method = "mtp"
        proposer.num_speculative_tokens = TestDynamicSpeculativeDecoding.MAX_K
        # decode_threshold as set once at init from the configured max K.
        proposer.decode_threshold = 1 + TestDynamicSpeculativeDecoding.MAX_K
        return proposer

    @staticmethod
    def _propose_with_k(proposer: AscendSpecDecodeBaseProposer, per_step_k: int | None, batch_size: int = 4):
        """Enter ``_propose`` far enough to exercise the DSD entry path.

        ``per_step_k=None`` models the non-DSD path (no scheduler_output).
        """
        common_attn_metadata = SimpleNamespace(batch_size=lambda: batch_size)
        scheduler_output = None if per_step_k is None else SimpleNamespace(num_spec_tokens_to_schedule=per_step_k)
        return proposer._propose(
            target_token_ids=torch.zeros(0, dtype=torch.long),
            target_positions=torch.zeros(0, dtype=torch.long),
            target_hidden_states=torch.zeros(0, 0),
            next_token_ids=torch.zeros(batch_size, dtype=torch.long),
            token_indices_to_sample=torch.zeros(batch_size, dtype=torch.long),
            common_attn_metadata=common_attn_metadata,
            target_model_batch_desc=None,
            sampling_metadata=None,
            scheduler_output=scheduler_output,
        )

    def test_zero_per_step_k_returns_empty_draft(self):
        """DSD chose K=0 for this batch size: empty (batch_size, 0) int64 draft."""
        proposer = self._make_proposer()
        scheduler_k = 0

        draft = self._propose_with_k(proposer, scheduler_k, batch_size=4)

        assert draft.shape == (4, 0)
        assert draft.dtype == torch.int64
        assert proposer.num_speculative_tokens == 0

    def test_decode_threshold_not_refreshed_to_per_step_k(self):
        """Regression: decode_threshold must stay at 1 + max K even when the
        per-step K is smaller (it sizes the static max-K FULL graph bucket)."""
        proposer = self._make_proposer()

        self._propose_with_k(proposer, per_step_k=0)

        assert proposer.num_speculative_tokens == 0
        assert proposer.decode_threshold == 1 + self.MAX_K

    def test_per_step_k_refresh_happens_before_propose(self):
        """A smaller per-step K replaces the configured max before any drafting
        work runs (verified via a sentinel raised at the first downstream use)."""
        proposer = self._make_proposer()
        proposer.set_inputs_first_pass = Mock(side_effect=RuntimeError("sentinel"))

        with pytest.raises(RuntimeError, match="sentinel"):
            self._propose_with_k(proposer, per_step_k=2)

        assert proposer.num_speculative_tokens == 2
        assert proposer.decode_threshold == 1 + self.MAX_K

    def test_no_scheduler_output_keeps_configured_k(self):
        """Non-DSD path: without scheduler_output the configured max K is used."""
        proposer = self._make_proposer()
        proposer.set_inputs_first_pass = Mock(side_effect=RuntimeError("sentinel"))

        with pytest.raises(RuntimeError, match="sentinel"):
            self._propose_with_k(proposer, per_step_k=None)

        assert proposer.num_speculative_tokens == self.MAX_K
        assert proposer.decode_threshold == 1 + self.MAX_K
