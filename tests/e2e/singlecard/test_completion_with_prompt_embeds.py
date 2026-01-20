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
import os

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.e2e.conftest import VllmRunner

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODELS = ["Qwen/Qwen3-0.6B"]


def get_prompt_embeds(chat, tokenizer, embedding_layer):
    """Convert chat messages to prompt embeddings."""
    token_ids = tokenizer.apply_chat_template(chat,
                                              add_generation_prompt=True,
                                              return_tensors='pt')
    prompt_embeds = embedding_layer(token_ids).squeeze(0)
    return prompt_embeds


@pytest.mark.parametrize("model_name", MODELS)
def test_mixed_prompt_embeds_and_text(model_name):
    """Test mixed inputs with both prompt embeddings and text prompts."""
    # Prepare prompt embeddings for first request
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformers_model = AutoModelForCausalLM.from_pretrained(model_name)
    embedding_layer = transformers_model.get_input_embeddings()

    chat = [{"role": "user", "content": "What is AI?"}]
    prompt_embeds = get_prompt_embeds(chat, tokenizer, embedding_layer)

    # Prepare text prompt for second request
    text_prompt = "What is machine learning?"

    # Run inference with mixed inputs
    with VllmRunner(
            model_name,
            enable_prompt_embeds=True,
            cudagraph_capture_sizes=[1, 2, 4, 8],
    ) as vllm_runner:
        # Test prompt embeddings
        embeds_output = vllm_runner.model.generate({
            "prompt_embeds":
            prompt_embeds,
        })

        # Test text prompt
        text_output = vllm_runner.model.generate(text_prompt)

    # Verify both types of inputs work
    assert len(embeds_output) == 1
    assert len(text_output) == 1
    assert len(embeds_output[0].outputs[0].text) > 0
    assert len(text_output[0].outputs[0].text) > 0

    print("\n[Prompt Embeds Output]:", embeds_output[0].outputs[0].text)
    print("[Text Prompt Output]:", text_output[0].outputs[0].text)
