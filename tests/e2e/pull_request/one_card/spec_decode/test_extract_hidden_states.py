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
"""E2E tests for the extract_hidden_states speculative decoding method.

Follows the pattern from vllm's test_extraction.py, validating that hidden
states are correctly extracted and saved on the Ascend NPU. Parametrized over:

* a dense model (Qwen3-8B) in both eager and ACL graph modes, using real
  weights so outputs can be checked to be non-zero, and
* a hybrid attention model (Qwen3.5-0.8B, GatedDeltaNet + full_attention)
  loaded with dummy weights as a shape/round-trip smoke test. The hybrid case
  mirrors upstream vLLM PR #39949.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import pytest
import torch
from vllm import LLM, SamplingParams

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.23.0"):
    from safetensors import safe_open
else:
    from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector


os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

DENSE_MODEL = "Qwen/Qwen3-8B"
# Qwen3-8B has 36 layers; pick a spread of layer indices to extract.
DENSE_AUX_HIDDEN_STATE_LAYER_IDS = [2, 18, 34]

HYBRID_MODEL = "Qwen/Qwen3.5-0.8B"
HYBRID_AUX_HIDDEN_STATE_LAYER_IDS = [5, 11, 17]


@dataclass
class ExtractHiddenStatesCase:
    model_name: str
    aux_hidden_state_layer_ids: list[int]
    prompts: list[str]
    enforce_eager: bool
    # ``None`` means "do not pass the argument", preserving each model's
    # original defaults.
    gpu_memory_utilization: float | None = None
    max_num_seqs: int | None = None
    max_model_len: int | None = None
    load_format: str | None = None
    # Dummy-weight runs can't assert non-zero outputs; real-weight runs can.
    verify_nonzero: bool = True
    # Hybrid smoke test additionally checks the token_ids round-trip.
    verify_token_ids: bool = False


CASES = [
    pytest.param(
        ExtractHiddenStatesCase(
            model_name=DENSE_MODEL,
            aux_hidden_state_layer_ids=DENSE_AUX_HIDDEN_STATE_LAYER_IDS,
            prompts=[
                "Hello, how are you?",
                "What is machine learning?",
                "Explain quantum computing briefly.",
            ],
            enforce_eager=True,
            gpu_memory_utilization=0.8,
            max_num_seqs=16,
        ),
        id="dense_eager",
    ),
    pytest.param(
        ExtractHiddenStatesCase(
            model_name=DENSE_MODEL,
            aux_hidden_state_layer_ids=DENSE_AUX_HIDDEN_STATE_LAYER_IDS,
            prompts=[
                "Hello, how are you?",
                "What is machine learning?",
            ],
            enforce_eager=False,
            max_num_seqs=16,
        ),
        id="dense_aclgraph",
    ),
    pytest.param(
        ExtractHiddenStatesCase(
            model_name=HYBRID_MODEL,
            aux_hidden_state_layer_ids=HYBRID_AUX_HIDDEN_STATE_LAYER_IDS,
            prompts=[
                "Hello world",
                "Test prompt with several tokens",
            ],
            enforce_eager=True,
            gpu_memory_utilization=0.4,
            max_model_len=256,
            load_format="dummy",
            verify_nonzero=False,
            verify_token_ids=True,
        ),
        id="hybrid_dummy_eager",
    ),
]


@pytest.fixture
def sampling_config():
    return SamplingParams(temperature=0, max_tokens=1)


def _verify_output(output, expected_shape, *, verify_nonzero, verify_token_ids):
    """Verify a single hidden-states dump (matches vllm's check pattern)."""
    assert output.kv_transfer_params is not None
    hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
    assert hidden_states_path is not None

    if vllm_version_is("0.23.0"):
        assert os.path.exists(hidden_states_path)
        with safe_open(hidden_states_path, "pt") as f:
            tensor_names = f.keys()
            assert "hidden_states" in tensor_names
            hidden_states = f.get_tensor("hidden_states")
            assert hidden_states.shape == expected_shape

            if verify_token_ids:
                token_ids = f.get_tensor("token_ids")
                assert torch.equal(token_ids, torch.tensor(output.prompt_token_ids))

            if verify_nonzero:
                assert not torch.allclose(hidden_states, torch.zeros_like(hidden_states))
    else:
        obj = example_hidden_states_connector.load_hidden_states(hidden_states_path)
        example_hidden_states_connector.cleanup_hidden_states(hidden_states_path)
        hidden_states = obj["hidden_states"]
        assert hidden_states.shape == expected_shape

        if verify_token_ids:
            token_ids = obj["token_ids"]
            assert torch.equal(token_ids, torch.tensor(output.prompt_token_ids))

        if verify_nonzero:
            assert not torch.allclose(hidden_states, torch.zeros_like(hidden_states))


@pytest.mark.e2e_features("dense", "eager_mode", "pd_disaggregation", "eagle_3", "extract_hidden_states")
@pytest.mark.e2e_model("Qwen/Qwen3-8B", "Qwen/Qwen3.5-0.8B")
@pytest.mark.parametrize("case", CASES)
def test_extract_hidden_states(case: ExtractHiddenStatesCase, sampling_config):
    """Extract hidden states from the target model and validate the dump."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        llm_kwargs = dict(
            model=case.model_name,
            tensor_parallel_size=1,
            enforce_eager=case.enforce_eager,
            enable_chunked_prefill=False,
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": case.aux_hidden_state_layer_ids,
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
        if case.gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = case.gpu_memory_utilization
        if case.max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = case.max_num_seqs
        if case.max_model_len is not None:
            llm_kwargs["max_model_len"] = case.max_model_len
        if case.load_format is not None:
            llm_kwargs["load_format"] = case.load_format

        llm = LLM(**llm_kwargs)

        outputs = llm.generate(case.prompts, sampling_config)
        hidden_size = llm.llm_engine.model_config.get_hidden_size()
        num_layers = len(case.aux_hidden_state_layer_ids)

        assert len(outputs) == len(case.prompts)

        for output in outputs:
            num_tokens = len(output.prompt_token_ids)
            expected_shape = (num_tokens, num_layers, hidden_size)
            _verify_output(
                output,
                expected_shape,
                verify_nonzero=case.verify_nonzero,
                verify_token_ids=case.verify_token_ids,
            )
