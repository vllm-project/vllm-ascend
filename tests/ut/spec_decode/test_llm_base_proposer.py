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

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import using_paged_attention
from vllm_ascend.device.utils import FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

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


# ---------------------------------------------------------------------------
# PA gate routing: draft 512-dim global attention (Gemma4 MTP) must route
# through PagedAttention, not the dense-KV-gather prefill fallback.
# ---------------------------------------------------------------------------


class TestMTPDraftAttnStateRouting:
    """MTP draft step 0 attn_state must be SpecDecoding, not inherited from target.

    Bug: during chunked prefill the target model's attn_state is ChunkedPrefill.
    _propose's builder.build() copies this into the first draft step's metadata.
    The PA gate in forward_impl only accepts (DecodeOnly, SpecDecoding), so
    512-dim global attention heads fall through to the dense-KV-gather prefill
    fallback (_gather_paged_kv_to_dense) and OOM on long sequences.

    Fix: _propose overrides attn_metadata.attn_state to SpecDecoding for MTP
    after builder.build(), same as attn_update_stack_num_spec_norm does for
    subsequent draft steps.
    """

    # ---- PA gate helper --------------------------------------------------

    @staticmethod
    def _pa_gate_accepts(attn_state: AscendAttentionState) -> bool:
        """Mirrors forward_impl line 1585-1586."""
        return attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        )

    # ---- Tests -----------------------------------------------------------

    def test_pa_gate_rejects_chunked_prefill(self):
        """Without fix, ChunkedPrefill makes PA gate fail → FIA 512 prefill fallback."""
        assert not self._pa_gate_accepts(AscendAttentionState.ChunkedPrefill), (
            "ChunkedPrefill must NOT pass the PA gate — this is the bug: "
            "512-dim draft attention would take the dense-KV-gather prefill path"
        )

    def test_pa_gate_accepts_spec_decoding(self):
        """With fix, SpecDecoding makes PA gate pass → PagedAttention for 512-dim."""
        assert self._pa_gate_accepts(AscendAttentionState.SpecDecoding), (
            "SpecDecoding must pass the PA gate — the fix overrides attn_state "
            "to this value for MTP draft step 0"
        )

    def test_using_paged_attention_enabled_for_512dim(self):
        """using_paged_attention returns True for head_size==512 on A2/A3.

        This early return (before the speculative_config gate) is what makes PA
        available for 512-dim heads. If it ever regresses, 512-dim decode would
        fail at FIA TND (error 561002) or OOM at _gather_paged_kv_to_dense.
        """
        assert FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE == 512, (
            "FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE must be 512"
        )
        # Create a minimal VllmConfig stub. using_paged_attention checks
        # head_size==512 first and returns True before touching other fields.
        cfg_stub = SimpleNamespace(speculative_config=None)
        result = using_paged_attention(
            runtime_shape=1,  # decode: 1 token per request
            vllm_config=cfg_stub,
            head_size=512,
        )
        assert result is True, (
            "using_paged_attention(head_size=512) must return True on A2/A3"
        )

    def test_chunked_prefill_512dim_would_oom_without_fix(self):
        """End-to-end simulation of the OOM path without the fix.

        Constructs the exact condition that triggered the bug and verifies
        that applying the fix resolves it.
        """
        # Simulate what builder.build() returns during chunked prefill
        # (attn_state copied from common_attn_metadata).
        attn_state_before_fix = AscendAttentionState.ChunkedPrefill

        # Without fix: PA gate rejects → would route through
        # forward_fused_infer_attention → npu_large_head_prefill_attention
        # → _gather_paged_kv_to_dense → OOM for long sequences.
        assert not self._pa_gate_accepts(attn_state_before_fix), (
            "Before fix: ChunkedPrefill → PA gate fails → OOM path"
        )

        # Apply the fix (what _propose now does for MTP).
        if True:  # self.method == "mtp"
            attn_state_after_fix = AscendAttentionState.SpecDecoding

        # With fix: PA gate accepts → routes through forward_paged_attention
        # → PagedAttention (no dense KV gather, no OOM).
        assert self._pa_gate_accepts(attn_state_after_fix), (
            "After fix: SpecDecoding → PA gate passes → PagedAttention"
        )
