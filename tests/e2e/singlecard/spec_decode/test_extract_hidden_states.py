#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""E2E tests for extract_hidden_states speculative decoding method.

This test file follows the pattern from vllm's test_extraction.py,
testing that hidden states are correctly extracted and saved.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
from safetensors import safe_open
from vllm import LLM, SamplingParams

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Use Qwen3-8B as the test model (standard HuggingFace format)
MODEL_NAME = "Qwen/Qwen3-8B"
# Layer indices for hidden state extraction (Qwen3-8B has 36 layers)
EAGLE_AUX_HIDDEN_STATE_LAYER_IDS = [2, 18, 34]


def _verify_output(output, expected_shape):
    """Verify hidden states output (matches vllm's get_and_check_output pattern)."""
    assert output.kv_transfer_params is not None
    hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
    assert hidden_states_path is not None
    assert os.path.exists(hidden_states_path)

    with safe_open(hidden_states_path, "pt") as f:
        tensor_names = f.keys()
        assert "hidden_states" in tensor_names

        hidden_states = f.get_tensor("hidden_states")
        assert hidden_states.shape == expected_shape

        # Verify hidden_states are not all zeros
        assert not torch.allclose(hidden_states, torch.zeros_like(hidden_states))

    return hidden_states


@pytest.fixture
def sampling_config():
    return SamplingParams(temperature=0, max_tokens=1)


def test_extract_hidden_states_eager_mode(sampling_config):
    """
    Test extract_hidden_states with enforce_eager=True.

    This extracts hidden states from the target model and saves them to disk.
    Pattern matches vllm's test_extract_hidden_states_with_predictable_dummy_model.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=1,
            enforce_eager=True,
            max_num_seqs=16,
            gpu_memory_utilization=0.8,
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": EAGLE_AUX_HIDDEN_STATE_LAYER_IDS,
                    }
                },
            },
            kv_transfer_config={
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": tmpdirname,
                },
            },
        )

        prompts = [
            "Hello, how are you?",
            "What is machine learning?",
            "Explain quantum computing briefly.",
        ]

        outputs = llm.generate(prompts, sampling_config)
        hidden_size = llm.llm_engine.model_config.get_hidden_size()
        num_layers = len(EAGLE_AUX_HIDDEN_STATE_LAYER_IDS)

        assert len(outputs) == len(prompts)

        for output in outputs:
            num_tokens = len(output.prompt_token_ids)
            expected_shape = (num_tokens, num_layers, hidden_size)
            _verify_output(output, expected_shape)


def test_extract_hidden_states_aclgraph_mode(sampling_config):
    """
    Test extract_hidden_states with enforce_eager=False (ACL graph mode).

    This tests that ACL graph capture works correctly with extract_hidden_states.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=1,
            enforce_eager=False,
            max_num_seqs=16,
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": EAGLE_AUX_HIDDEN_STATE_LAYER_IDS,
                    }
                },
            },
            kv_transfer_config={
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": tmpdirname,
                },
            },
        )

        prompts = [
            "Hello, how are you?",
            "What is machine learning?",
        ]

        outputs = llm.generate(prompts, sampling_config)
        hidden_size = llm.llm_engine.model_config.get_hidden_size()
        num_layers = len(EAGLE_AUX_HIDDEN_STATE_LAYER_IDS)

        assert len(outputs) == len(prompts)

        hidden_states_count = 0
        for output in outputs:
            num_tokens = len(output.prompt_token_ids)
            expected_shape = (num_tokens, num_layers, hidden_size)
            _verify_output(output, expected_shape)
            hidden_states_count += 1

        assert hidden_states_count > 0, "No hidden states were saved"
