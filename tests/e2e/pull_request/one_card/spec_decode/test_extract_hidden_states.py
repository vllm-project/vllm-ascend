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
* Model Runner V1 (Ascend default) and Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`),
  covering the Ascend adaptation of upstream vLLM PR #49811 on the 0828 pin.
* token-in / token-out via ``skip_tokenizer_init`` + ``TokensPrompt`` on the
  text-only dense model (dummy weights). Qwen3.5 is multimodal, so skipping
  tokenizer init leaves ``tokenizer=None`` and ``Qwen3VLProcessor`` crashes.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import pytest
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector
from vllm.inputs import TokensPrompt

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

DENSE_MODEL = "Qwen/Qwen3-8B"
# Qwen3-8B has 36 layers; pick a spread of layer indices to extract.
DENSE_AUX_HIDDEN_STATE_LAYER_IDS = [2, 18, 34]

HYBRID_MODEL = "Qwen/Qwen3.5-0.8B"
HYBRID_AUX_HIDDEN_STATE_LAYER_IDS = [5, 11, 17]

# In-vocab dummy sequences for skip_tokenizer_init (Qwen3 vocab >> 500).
TOKEN_IN_PROMPTS = [
    [100, 200, 300, 400, 500],
    [7, 8, 9, 10, 11, 12, 13, 14],
]


@dataclass
class ExtractHiddenStatesCase:
    model_name: str
    aux_hidden_state_layer_ids: list[int]
    enforce_eager: bool
    prompts: list[str] | None = None
    token_prompts: list[list[int]] | None = None
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
    # When True, force Model Runner V2 via VLLM_USE_V2_MODEL_RUNNER.
    use_v2_model_runner: bool = False
    # Token-in / token-out: skip tokenizer init and pass TokensPrompt.
    skip_tokenizer_init: bool = False


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
    pytest.param(
        ExtractHiddenStatesCase(
            model_name=DENSE_MODEL,
            aux_hidden_state_layer_ids=DENSE_AUX_HIDDEN_STATE_LAYER_IDS,
            prompts=[
                "Hello, how are you?",
                "What is machine learning?",
            ],
            enforce_eager=True,
            gpu_memory_utilization=0.8,
            max_num_seqs=16,
            use_v2_model_runner=True,
        ),
        id="dense_eager_mrv2",
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
            use_v2_model_runner=True,
        ),
        id="hybrid_dummy_eager_mrv2",
    ),
    pytest.param(
        ExtractHiddenStatesCase(
            model_name=DENSE_MODEL,
            aux_hidden_state_layer_ids=DENSE_AUX_HIDDEN_STATE_LAYER_IDS,
            token_prompts=TOKEN_IN_PROMPTS,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
            max_num_seqs=16,
            max_model_len=256,
            load_format="dummy",
            verify_nonzero=False,
            verify_token_ids=True,
            skip_tokenizer_init=True,
        ),
        id="dense_dummy_token_in_token_out",
    ),
    pytest.param(
        ExtractHiddenStatesCase(
            model_name=DENSE_MODEL,
            aux_hidden_state_layer_ids=DENSE_AUX_HIDDEN_STATE_LAYER_IDS,
            token_prompts=TOKEN_IN_PROMPTS,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
            max_num_seqs=16,
            max_model_len=256,
            load_format="dummy",
            verify_nonzero=False,
            verify_token_ids=True,
            use_v2_model_runner=True,
            skip_tokenizer_init=True,
        ),
        id="dense_dummy_token_in_token_out_mrv2",
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

    obj = example_hidden_states_connector.load_hidden_states(hidden_states_path)
    try:
        hidden_states = obj["hidden_states"]
        assert hidden_states.shape == expected_shape

        assert not torch.isnan(hidden_states).any(), "hidden_states contains NaN"
        assert not torch.isinf(hidden_states).any(), "hidden_states contains Inf"

        if verify_token_ids:
            token_ids = obj["token_ids"]
            assert torch.equal(token_ids, torch.tensor(output.prompt_token_ids))

        if verify_nonzero:
            assert not torch.allclose(hidden_states, torch.zeros_like(hidden_states))
    finally:
        example_hidden_states_connector.cleanup_hidden_states(hidden_states_path)


def _generate_inputs(case: ExtractHiddenStatesCase):
    if case.skip_tokenizer_init:
        assert case.token_prompts is not None
        return [TokensPrompt(prompt_token_ids=ids) for ids in case.token_prompts]
    assert case.prompts is not None
    return case.prompts


def _verify_token_in_token_out(output, token_prompt: list[int], *, max_tokens: int):
    """Input token ids round-trip; generated ids are present without detokenizing."""
    assert list(output.prompt_token_ids) == token_prompt
    assert not output.outputs[0].text
    assert len(output.outputs[0].token_ids) == max_tokens


@pytest.mark.parametrize("case", CASES)
def test_extract_hidden_states(case: ExtractHiddenStatesCase, sampling_config, monkeypatch):
    """Extract hidden states from the target model and validate the dump."""
    if case.use_v2_model_runner:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    else:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)

    generate_inputs = _generate_inputs(case)
    if case.skip_tokenizer_init:
        sampling = SamplingParams(temperature=0, max_tokens=1, detokenize=False)
    else:
        sampling = sampling_config

    with tempfile.TemporaryDirectory() as tmpdirname:
        llm_kwargs = dict(
            model=case.model_name,
            tensor_parallel_size=1,
            enforce_eager=case.enforce_eager,
            enable_chunked_prefill=True,
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
        if case.skip_tokenizer_init:
            llm_kwargs["skip_tokenizer_init"] = True

        llm = LLM(**llm_kwargs)

        outputs = llm.generate(generate_inputs, sampling)
        hidden_size = llm.llm_engine.model_config.get_hidden_size()
        num_layers = len(case.aux_hidden_state_layer_ids)
        vocab_size = llm.llm_engine.model_config.get_vocab_size()

        assert len(outputs) == len(generate_inputs)

        for idx, output in enumerate(outputs):
            num_tokens = len(output.prompt_token_ids)
            expected_shape = (num_tokens, num_layers, hidden_size)
            if case.skip_tokenizer_init:
                assert case.token_prompts is not None
                assert sampling.max_tokens is not None
                _verify_token_in_token_out(
                    output,
                    case.token_prompts[idx],
                    max_tokens=sampling.max_tokens,
                )
                assert all(0 <= token_id < vocab_size for token_id in output.outputs[0].token_ids)
            _verify_output(
                output,
                expected_shape,
                verify_nonzero=case.verify_nonzero,
                verify_token_ids=case.verify_token_ids,
            )
