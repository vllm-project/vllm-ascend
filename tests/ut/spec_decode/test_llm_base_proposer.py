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

import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode

import vllm_ascend.spec_decode.llm_base_proposer as proposer_module
from vllm_ascend.ops.triton.spec_decode.utils import (
    prepare_inputs_padded_kernel,
    prepare_next_token_padded_kernel,
)
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


def test_prepare_inputs_padded_uses_one_program_per_request():
    source = inspect.getsource(prepare_inputs_padded_kernel.fn)

    assert "req_idx = tl.program_id(axis=0)" in source
    assert "if req_idx == 0" in source
    assert "cu_num_draft_tokens_ptr + req_idx - 1" in source
    assert "tl.arange" not in source


def test_prepare_next_token_padded_counts_int32_mask_values():
    source = inspect.getsource(prepare_next_token_padded_kernel.fn)

    assert "tl.sum(is_valid.to(tl.int32), axis=0)" in source


def test_prepare_inputs_padded_ignores_graph_only_request_rows(monkeypatch):
    monkeypatch.setattr(proposer_module, "HAS_TRITON", False)
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.pcp_size = 1
    proposer.arange = torch.arange(8, dtype=torch.int32)
    proposer.runner = SimpleNamespace(
        actual_seq_lengths_q=[3, 3],
        attn_state=None,
        decode_token_per_req=1,
    )
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3, 6], dtype=torch.int32),
        seq_lens=torch.tensor([5, 0], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([5, 0], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([5, 0], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([5, 0], dtype=torch.int32),
        num_computed_tokens_cpu=None,
        _num_computed_tokens_cpu=None,
        num_reqs=2,
        num_actual_tokens=3,
        num_input_tokens=6,
        block_table_tensor=None,
        slot_mapping=torch.tensor([0, 1, 2, -1, -1, -1], dtype=torch.int32),
        positions=torch.arange(6, dtype=torch.int32),
        positions_cpu=None,
        is_prefilling=None,
        group_len=None,
        group_key_idx=None,
        group_key_cache_idx=None,
    )
    spec_metadata = SimpleNamespace(
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int32),
    )

    _, token_indices, token_indices_to_sample, num_rejected = proposer.prepare_inputs_padded(
        metadata,
        spec_metadata,
        torch.tensor([1], dtype=torch.int32),
    )

    assert token_indices.tolist() == [0, 1, 2, 3, 4, 5]
    assert token_indices_to_sample.tolist() == [0]
    assert num_rejected.tolist() == [2]


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


def test_mtp_without_own_lm_head_shares_target_head():
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    draft_head = object()
    target_head = object()
    proposer.method = "mtp"
    proposer.model = SimpleNamespace(
        has_own_lm_head=False,
        lm_head=draft_head,
        model=SimpleNamespace(layers={"43": SimpleNamespace()}),
    )
    proposer.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(is_deepseek_mla=True),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
    )
    proposer.use_cuda_graph = False
    proposer.use_eagle = False
    proposer.enable_enpu = False

    proposer._maybe_share_lm_head(SimpleNamespace(lm_head=target_head))

    assert proposer.model.lm_head is target_head
