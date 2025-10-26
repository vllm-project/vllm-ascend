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

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct"]


def get_prompt_embeds(chat, tokenizer, embedding_layer):
    """Convert chat messages to prompt embeddings."""
    token_ids = tokenizer.apply_chat_template(chat,
                                              add_generation_prompt=True,
                                              return_tensors='pt')
    prompt_embeds = embedding_layer(token_ids).squeeze(0)
    return prompt_embeds


@pytest.mark.parametrize("model_name", MODELS)
def test_single_prompt_embeds_inference(model_name):
    """Test single prompt inference with prompt embeddings."""
    # Prepare prompt embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformers_model = AutoModelForCausalLM.from_pretrained(model_name)
    embedding_layer = transformers_model.get_input_embeddings()

    chat = [{
        "role": "user",
        "content": "Please tell me about the capital of France."
    }]
    prompt_embeds = get_prompt_embeds(chat, tokenizer, embedding_layer)

    # Run inference with prompt embeddings
    with VllmRunner(
            model_name,
            enable_prompt_embeds=True,
            enforce_eager=True,
    ) as vllm_runner:
        outputs = vllm_runner.model.generate({
            "prompt_embeds": prompt_embeds,
        })

    # Verify output
    assert len(outputs) == 1
    assert len(outputs[0].outputs) > 0
    assert len(outputs[0].outputs[0].text) > 0
    print(f"\n[Single Inference Output]: {outputs[0].outputs[0].text}")


@pytest.mark.parametrize("model_name", MODELS)
def test_batch_prompt_embeds_inference(model_name):
    """Test batch prompt inference with prompt embeddings."""
    # Prepare prompt embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformers_model = AutoModelForCausalLM.from_pretrained(model_name)
    embedding_layer = transformers_model.get_input_embeddings()

    chats = [[{
        "role": "user",
        "content": "Please tell me about the capital of France."
    }],
             [{
                 "role": "user",
                 "content": "When is the day longest during the year?"
             }],
             [{
                 "role": "user",
                 "content": "Where is bigger, the moon or the sun?"
             }]]

    prompt_embeds_list = [
        get_prompt_embeds(chat, tokenizer, embedding_layer) for chat in chats
    ]

    # Run batch inference with prompt embeddings
    with VllmRunner(
            model_name,
            enable_prompt_embeds=True,
            enforce_eager=True,
    ) as vllm_runner:
        outputs = vllm_runner.model.generate([{
            "prompt_embeds": embeds
        } for embeds in prompt_embeds_list])

    # Verify outputs
    assert len(outputs) == len(chats)
    for i, output in enumerate(outputs):
        assert len(output.outputs) > 0
        assert len(output.outputs[0].text) > 0
        print(f"\nQ{i+1}: {chats[i][0]['content']}")
        print(f"A{i+1}: {output.outputs[0].text}")


@pytest.mark.parametrize("model_name", MODELS)
def test_prompt_embeds_with_aclgraph(model_name):
    """Test prompt embeddings with ACL graph enabled vs disabled."""
    # Prepare prompt embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformers_model = AutoModelForCausalLM.from_pretrained(model_name)
    embedding_layer = transformers_model.get_input_embeddings()

    chat = [{"role": "user", "content": "What is the capital of China?"}]
    prompt_embeds = get_prompt_embeds(chat, tokenizer, embedding_layer)

    # Run with ACL graph enabled (enforce_eager=False)
    with VllmRunner(
            model_name,
            enable_prompt_embeds=True,
            enforce_eager=False,
    ) as vllm_aclgraph_runner:
        aclgraph_outputs = vllm_aclgraph_runner.model.generate({
            "prompt_embeds":
            prompt_embeds,
        })

    # Run with ACL graph disabled (enforce_eager=True)
    with VllmRunner(
            model_name,
            enable_prompt_embeds=True,
            enforce_eager=True,
    ) as vllm_eager_runner:
        eager_outputs = vllm_eager_runner.model.generate({
            "prompt_embeds":
            prompt_embeds,
        })

    # Verify both produce valid outputs
    assert len(aclgraph_outputs) == 1
    assert len(eager_outputs) == 1
    assert len(aclgraph_outputs[0].outputs[0].text) > 0
    assert len(eager_outputs[0].outputs[0].text) > 0

    print("\n[ACL Graph Output]:", aclgraph_outputs[0].outputs[0].text)
    print("[Eager Output]:", eager_outputs[0].outputs[0].text)

    # Note: Outputs may differ slightly due to different execution paths,
    # but both should be valid responses


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
            enforce_eager=True,
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
