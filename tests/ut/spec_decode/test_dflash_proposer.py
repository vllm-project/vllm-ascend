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
"""Unit tests for the DFlash speculative-decoding proposer."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.spec_decode.dflash2_proposer import AscendDflash2Proposer
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

_MAX_NUM_TOKENS = 256
_NUM_SPECULATIVE_TOKENS = 3
_HIDDEN_SIZE = 8


@pytest.mark.parametrize("proposer_cls", [AscendDflashProposer, AscendDflash2Proposer])
@pytest.mark.parametrize(
    ("max_num_seqs", "max_capture_size", "num_input_tokens"),
    [
        pytest.param(33, 136, 136, id="padded-query"),
        pytest.param(32, 128, 128, id="capture-aligned-query"),
        pytest.param(33, 132, 132, id="equal-capacity"),
        pytest.param(33, 128, 132, id="query-exceeds-capture"),
        pytest.param(33, None, 132, id="unset-capture-size"),
        pytest.param(33, 0, 132, id="disabled-capture"),
    ],
)
def test_query_buffers_cover_execution_shape(proposer_cls, max_num_seqs, max_capture_size, num_input_tokens):
    """Graph padding must not silently truncate query positions or slots."""
    device = torch.device("cpu")
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=max_capture_size),
        speculative_config=SimpleNamespace(draft_sample_method="greedy"),
    )

    def init_base(self, vllm_config, device, runner=None):
        self.max_batch_size = max_num_seqs
        self.num_speculative_tokens = _NUM_SPECULATIVE_TOKENS
        self.max_num_tokens = _MAX_NUM_TOKENS
        self.hidden_size = _HIDDEN_SIZE
        self.dtype = torch.float32
        self.device = device
        self.input_ids = torch.zeros(_MAX_NUM_TOKENS, dtype=torch.int32, device=device)
        self.uses_mrope = False
        self.uses_xdrope_dim = 0
        self.draft_model_config = SimpleNamespace(hf_config=SimpleNamespace(dflash_config={"selector_top_k": 2}))

    with (
        patch.object(AscendEagleProposer, "__init__", init_base),
        patch(
            "vllm_ascend.spec_decode.dflash_proposer.get_ascend_config",
            return_value=SimpleNamespace(dynamic_spec_config=SimpleNamespace(method=None)),
        ),
    ):
        proposer = proposer_cls(config, device)

    input_ids = proposer.input_ids[:num_input_tokens]
    assert input_ids.shape == (num_input_tokens,)
    assert proposer._get_positions(num_input_tokens).shape == input_ids.shape
    assert proposer._slot_mapping_buffer[:num_input_tokens].shape == input_ids.shape
    for buffer in (proposer.positions, proposer._slot_mapping_buffer):
        assert buffer.dtype == torch.int32
        assert buffer.device == device

    # Padding changes storage capacity, not the logical query or context limits.
    assert proposer.max_query_tokens == max_num_seqs * (1 + _NUM_SPECULATIVE_TOKENS)
    assert proposer.max_positions == _MAX_NUM_TOKENS + num_input_tokens
    assert proposer.arange_dflash.shape == (proposer.max_positions + 1,)
    assert proposer._context_positions_buffer.shape == (_MAX_NUM_TOKENS,)
    assert proposer._context_slot_mapping_buffers.shape == (_MAX_NUM_TOKENS,)
    assert proposer._dflash_hidden_states.shape == (_MAX_NUM_TOKENS, _HIDDEN_SIZE)


class TestDummySlotMappingCleanup:
    def test_dummy_run_clears_single_group_buffers_before_forward(self, monkeypatch):
        proposer = AscendDflashProposer.__new__(AscendDflashProposer)
        proposer._slot_mapping_buffer = torch.full((8,), 123, dtype=torch.int32)
        proposer._context_slot_mapping_buffers = torch.full((8,), 456, dtype=torch.int32)

        @contextmanager
        def forward_context(*args, **kwargs):
            yield

        def runnable(**kwargs):
            assert torch.all(proposer._slot_mapping_buffer == -1)
            assert torch.all(proposer._context_slot_mapping_buffers == -1)

        monkeypatch.setattr("vllm_ascend.spec_decode.dflash_proposer.set_ascend_forward_context", forward_context)
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dflash_proposer.get_forward_context",
            lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE),
        )
        proposer.max_query_tokens = 8
        proposer.runner = SimpleNamespace(
            _sync_metadata_across_dp=lambda num_tokens, **kwargs: (num_tokens, None, None),
        )
        proposer.use_cuda_graph = False
        proposer.num_speculative_tokens = 4
        proposer._context_positions_buffer = torch.zeros(8, dtype=torch.int32)
        proposer.hidden_states = torch.zeros((8, 8), dtype=torch.float32)
        proposer.token_indices_to_sample = torch.zeros(4, dtype=torch.int32)
        proposer.vllm_config = SimpleNamespace()
        proposer._get_positions = lambda num_tokens: torch.zeros(num_tokens, dtype=torch.int32)
        proposer._runnable = MagicMock(side_effect=runnable)

        proposer.dummy_run(num_tokens=5, num_reqs=1, is_profile=False)

        assert torch.all(proposer._slot_mapping_buffer == -1)
        assert torch.all(proposer._context_slot_mapping_buffers == -1)
        proposer._runnable.assert_called_once()
