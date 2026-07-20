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

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

FULL_CUDAGRAPH_MODES = [
    CUDAGraphMode.FULL,
    CUDAGraphMode.FULL_DECODE_ONLY,
    CUDAGraphMode.FULL_AND_PIECEWISE,
]

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
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.speculative_config = SimpleNamespace(
            disable_padded_drafter_batch=disable_padded_drafter_batch,
        )
        proposer.use_cuda_graph = use_cuda_graph
        proposer.compilation_config = SimpleNamespace(cudagraph_mode=cudagraph_mode)
        return proposer

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_raises_when_padded_drafter_batch_disabled_with_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )
        with pytest.raises(NotImplementedError, match="disable_padded_drafter_batch"):
            proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", NON_FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_without_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )
        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_when_padded_drafter_batch_enabled(self, cudagraph_mode: CUDAGraphMode):
        proposer = self._make_proposer(
            disable_padded_drafter_batch=False,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )
        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    def test_guard_does_not_raise_when_eager(self):
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=False,
            cudagraph_mode=CUDAGraphMode.FULL,
        )
        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()


class TestDrafterBlockSizeDispatch:
    """Verify model_runner passes the correct block_size type to draft proposers.

    Gemma4 MTP (if branch): receives kernel_block_sizes as list (per-group).
    Others (else branch): receives block_size as single int.

    Mirrors the if/else dispatch in model_runner_v1.py initialize_metadata_builders.
    """

    @staticmethod
    def _dispatch(is_gemma4: bool, kernel_block_sizes):
        if is_gemma4:
            return kernel_block_sizes if isinstance(kernel_block_sizes, list) else [kernel_block_sizes]
        else:
            return kernel_block_sizes[0] if isinstance(kernel_block_sizes, list) else kernel_block_sizes

    @pytest.mark.parametrize("kernel_block_sizes, expected", [
        ([64, 128], 64),
        (64, 64),
    ])
    def test_else_branch_returns_int(self, kernel_block_sizes, expected):
        """Non-Gemma4 proposers (Eagle/DFlash/etc.) enter the else branch
        and receive a single int block_size."""
        result = self._dispatch(is_gemma4=False, kernel_block_sizes=kernel_block_sizes)
        assert isinstance(result, int)
        assert result == expected

    @pytest.mark.parametrize("kernel_block_sizes, expected", [
        ([64, 128], [64, 128]),
        (64, [64]),
    ])
    def test_if_branch_returns_list(self, kernel_block_sizes, expected):
        """Gemma4 MTP enters the if branch and receives a list of block_sizes."""
        result = self._dispatch(is_gemma4=True, kernel_block_sizes=kernel_block_sizes)
        assert isinstance(result, list)
        assert result == expected
