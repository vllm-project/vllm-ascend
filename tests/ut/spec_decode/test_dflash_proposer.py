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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

_MAX_NUM_TOKENS = 256
_NUM_SPECULATIVE_TOKENS = 3
_HIDDEN_SIZE = 8


@pytest.fixture
def make_proposer():
    def init_base(self, vllm_config, device, runner=None):
        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.num_speculative_tokens = _NUM_SPECULATIVE_TOKENS
        self.max_num_tokens = _MAX_NUM_TOKENS
        self.hidden_size = _HIDDEN_SIZE
        self.dtype = torch.float32
        self.device = device
        self.input_ids = torch.zeros(_MAX_NUM_TOKENS, dtype=torch.int32, device=device)
        self.uses_mrope = False
        self.uses_xdrope_dim = 0
        self.draft_uses_xdrope_dim = 0

    def make(config):
        return AscendDflashProposer(config, torch.device("cpu"))

    with patch.object(AscendEagleProposer, "__init__", init_base):
        yield make


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
def test_query_buffers_cover_execution_shape(make_proposer, max_num_seqs, max_capture_size, num_input_tokens):
    """Backport the allocation regression from upstream vllm-ascend PR #17379."""
    device = torch.device("cpu")
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=max_capture_size),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )
    proposer = make_proposer(config)

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
    assert proposer._context_slot_mapping_buffer.shape == (_MAX_NUM_TOKENS,)
    assert proposer._dflash_hidden_states.shape == (_MAX_NUM_TOKENS, _HIDDEN_SIZE)


@pytest.mark.parametrize(
    ("graph_mode", "expected_mode", "expected_tokens"),
    [
        pytest.param(CUDAGraphMode.FULL, CUDAGraphMode.FULL, 136, id="padded-dispatch"),
        pytest.param(CUDAGraphMode.PIECEWISE, CUDAGraphMode.PIECEWISE, 136, id="piecewise-dispatch"),
        pytest.param(CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.NONE, 132, id="decode-only-fallback"),
    ],
)
def test_model_inputs_cover_dispatcher_extent(make_proposer, graph_mode, expected_mode, expected_tokens):
    """Use an actual dispatch result, including decode-only's unpadded fallback."""
    config = SimpleNamespace(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=graph_mode,
            cudagraph_capture_sizes=[4, 8, 16, 32, 64, 128, 136],
            max_cudagraph_capture_size=136,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=33),
        speculative_config=SimpleNamespace(num_speculative_tokens=_NUM_SPECULATIVE_TOKENS),
        lora_config=None,
    )
    proposer = make_proposer(config)
    config.compilation_config.set_splitting_ops_for_v1("allgather_reducescatter")
    dispatcher = CudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(graph_mode, 1 + _NUM_SPECULATIVE_TOKENS)
    runtime_mode, descriptor = dispatcher.dispatch(proposer.max_query_tokens, uniform_decode=True)
    assert runtime_mode == expected_mode
    assert descriptor.num_tokens == expected_tokens

    # Stop at input assembly: no model, context-KV calculation or graph replay.
    proposer._dflash_num_context = 1
    proposer.model = SimpleNamespace(precompute_and_store_context_kv=Mock())
    inputs = proposer.build_model_inputs_first_pass(descriptor.num_tokens)
    assert inputs["input_ids"].shape == (descriptor.num_tokens,)
    assert inputs["positions"].shape == inputs["input_ids"].shape
