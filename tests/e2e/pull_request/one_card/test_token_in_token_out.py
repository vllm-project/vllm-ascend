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
"""E2E tests for token-in / token-out inference on Ascend.

``skip_tokenizer_init=True`` skips tokenizer and detokenizer init. Prompts
must be ``TokensPrompt`` (valid ``prompt_token_ids``); outputs contain token
ids and empty text. This is the typical RL / EAGLE-dump collection path.
"""

from __future__ import annotations

import os

import pytest
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 8
# In-vocab dummy sequences (Qwen3 vocab >> 500).
TOKEN_PROMPTS = [
    [100, 200, 300, 400, 500],
    [7, 8, 9, 10, 11, 12, 13, 14],
]


@pytest.mark.parametrize(
    "use_v2_model_runner",
    [
        pytest.param(False, id="mrv1"),
        pytest.param(True, id="mrv2"),
    ],
)
def test_token_in_token_out(use_v2_model_runner: bool, monkeypatch):
    """Generate from token ids and return token ids without a tokenizer."""
    if use_v2_model_runner:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    else:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        enforce_eager=True,
        load_format="dummy",
        skip_tokenizer_init=True,
        max_model_len=256,
        gpu_memory_utilization=0.4,
        max_num_seqs=4,
    )
    sampling = SamplingParams(
        temperature=0,
        max_tokens=MAX_TOKENS,
        ignore_eos=True,
        detokenize=False,
    )
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in TOKEN_PROMPTS]
    outputs = llm.generate(prompts, sampling)
    vocab_size = llm.llm_engine.model_config.get_vocab_size()

    assert len(outputs) == len(TOKEN_PROMPTS)
    for output, token_prompt in zip(outputs, TOKEN_PROMPTS):
        assert list(output.prompt_token_ids) == token_prompt
        assert not output.outputs[0].text
        generated = list(output.outputs[0].token_ids)
        assert len(generated) == MAX_TOKENS
        assert all(0 <= token_id < vocab_size for token_id in generated)
