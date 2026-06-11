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
# This file is a part of the vllm-ascend project.
#

"""
E2E test for the token-in/token-out inference endpoint `/inference/v1/generate`.

This endpoint is the RL-oriented "tokens in <> tokens out" API: the client sends
already-tokenized `token_ids` and (with `detokenize=False`) receives the raw
generated `token_ids` back, bypassing string (de)tokenization. This avoids
retokenization drift between inference and training in agent RL workflows.

A lightweight model (`Qwen/Qwen3-0.6B`) is used so the test fits comfortably on a
single Ascend card.

Run `pytest tests/e2e/pull_request/one_card/test_token_in_token_out.py`.
"""

import pytest
import requests
from transformers import AutoTokenizer
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"
MAX_TOKENS = 16


@wait_until_npu_memory_free()
def test_token_in_token_out_generate():
    """The `/inference/v1/generate` endpoint must accept input `token_ids` and,
    with `detokenize=False`, return raw generated `token_ids` that decode back to
    non-empty text."""
    port = get_open_port()
    server_args = [
        "--max-model-len",
        "1024",
        "--gpu-memory-utilization",
        "0.4",
        "--enforce-eager",
        "--trust-remote-code",
        "--port",
        str(port),
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_token_ids = tokenizer(PROMPT).input_ids
    assert prompt_token_ids, "Tokenizer produced an empty prompt; cannot test token-in path"

    with RemoteOpenAIServer(MODEL, server_args, server_port=port, auto_port=False) as server:
        generate_url = server.url_for("inference", "v1", "generate")
        response = requests.post(
            generate_url,
            json={
                "model": MODEL,
                "token_ids": prompt_token_ids,
                # detokenize=False -> token-out: response carries raw token ids.
                "sampling_params": {
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "detokenize": False,
                },
                "stream": False,
            },
            timeout=600,
        )

        # If the running vLLM build predates the tokens-in/tokens-out endpoint,
        # skip rather than hard-fail so the suite stays version-tolerant.
        if response.status_code == 404:
            pytest.skip("`/inference/v1/generate` endpoint is not available in this vLLM build")

        response.raise_for_status()
        data = response.json()

        choices = data.get("choices")
        assert choices, f"Response is missing `choices`: {data}"

        output_token_ids = choices[0].get("token_ids")
        assert output_token_ids, f"Response is missing generated `token_ids`: {data}"
        assert all(isinstance(token_id, int) for token_id in output_token_ids), (
            f"Generated `token_ids` must all be integers, got: {output_token_ids}"
        )
        assert len(output_token_ids) <= MAX_TOKENS, (
            f"Generated more tokens ({len(output_token_ids)}) than requested ({MAX_TOKENS})"
        )

        # The returned token ids must decode back into non-empty text, proving the
        # token-out path produced a usable continuation.
        decoded_text = tokenizer.decode(output_token_ids)
        assert decoded_text.strip(), f"Decoded token ids produced empty text: {output_token_ids}"
