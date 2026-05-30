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
"""E2E smoke test for extract_hidden_states on a hybrid attention model.

Mirrors upstream vLLM PR #39949 test_extract_hidden_states_qwen35_hybrid_smoke,
but exercises the Ascend NPU code path. Validates that:

1. The hidden-state cache layer is allocated as a single tensor and is
   not hijacked by the hybrid-attn-with-mamba branch.
2. With ExampleHiddenStatesConnector (which subclasses SupportsHMA in
   upstream vLLM), HMA stays enabled so unify_hybrid_kv_cache_specs is
   not invoked, and the Mamba + FullAttention + HiddenState spec mix no
   longer trips the "failed to convert the KV cache specs to one
   unified type" ValueError.
3. Extracted hidden states have the expected shape
   (num_prompt_tokens, len(layer_ids), hidden_size).

The test uses load_format="dummy" so no checkpoint download is required;
only shapes and code paths are validated.
"""

from __future__ import annotations

import gc
import os
import tempfile

import pytest
import torch
from safetensors import safe_open
from vllm import LLM, SamplingParams

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Qwen3.5-0.8B is a hybrid model with interleaved linear_attention
# (GatedDeltaNet / Mamba-style) and full_attention layers.
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
EAGLE_AUX_HIDDEN_STATE_LAYER_IDS = [5, 11, 17]
HIDDEN_SIZE = 1024  # Qwen3.5-0.8B hidden_size


@pytest.fixture
def sampling_config():
    return SamplingParams(temperature=0.0, max_tokens=1)


def test_extract_hidden_states_qwen35_hybrid_smoke(sampling_config):
    """Smoke test for Qwen3.5 hybrid + extract_hidden_states on NPU."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=1,
            max_model_len=256,
            enforce_eager=True,
            gpu_memory_utilization=0.4,
            load_format="dummy",
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
            "Hello world",
            "Test prompt with several tokens",
        ]
        outputs = llm.generate(prompts, sampling_config)
        del llm
        gc.collect()

        assert len(outputs) == len(prompts)
        for output in outputs:
            assert output.kv_transfer_params is not None
            hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
            assert hidden_states_path is not None
            assert os.path.exists(hidden_states_path)

            with safe_open(hidden_states_path, "pt") as f:
                token_ids = f.get_tensor("token_ids")
                hidden_states = f.get_tensor("hidden_states")

            assert torch.equal(token_ids, torch.tensor(output.prompt_token_ids))
            assert hidden_states.shape == (
                len(output.prompt_token_ids),
                len(EAGLE_AUX_HIDDEN_STATE_LAYER_IDS),
                HIDDEN_SIZE,
            )
