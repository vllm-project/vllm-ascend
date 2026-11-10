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
"""Serving checks for InternLM2-20B-Chat on two cards.

The model is a text-only dense GQA decoder whose checkpoint selects dynamic
NTK rotary embeddings. These tests cover what that combination can break on
the NPU: greedy determinism under tensor parallelism, seeded sampling,
retrieval from a long context, and the ChatML template with its two stop
tokens.

Each test loads a 20B checkpoint, so the offline checks share one engine.
"""

import pytest
import requests
from vllm import SamplingParams
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, VllmRunner, wait_until_npu_memory_free

MODEL = "Shanghai_AI_Laboratory/internlm2-chat-20b"
TENSOR_PARALLEL_SIZE = 2

SHORT_PROMPTS = [
    "The capital of France is",
    "请用一句话解释什么是机器学习。",
]

# Retrieval over a long prompt: a mis-scaled rotary table degrades gradually
# with position, so a fact placed far from the question is the cheapest signal
# that positions beyond the short-prompt range still resolve.
NEEDLE = "The access code for the north gate is 73915."
FILLER = "The weather report for that day was unremarkable. " * 260
LONG_PROMPT = f"{FILLER}\n{NEEDLE}\n{FILLER}\nQuestion: What is the access code for the north gate?\nAnswer:"


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="long_sequence",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="piecewise",
)
@wait_until_npu_memory_free()
def test_offline_generation():
    """Greedy determinism, seeded sampling, and long-context retrieval.

    Greedy decoding has a single correct answer per prompt, so repeating the
    batch must reproduce it token for token. Under tensor parallelism a rotary
    or collective defect usually shows up here first, as a reply that is
    fluent but not reproducible.
    """
    sampling = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=48, seed=1234)
    other_seed = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=48, seed=4321)

    with VllmRunner(
        MODEL,
        max_model_len=8192,
        dtype="bfloat16",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=0.8,
        max_num_seqs=4,
    ) as runner:
        first = runner.generate_greedy(SHORT_PROMPTS, max_tokens=32)
        second = runner.generate_greedy(SHORT_PROMPTS, max_tokens=32)
        seeded = runner.generate(SHORT_PROMPTS[:1], sampling)
        seeded_again = runner.generate(SHORT_PROMPTS[:1], sampling)
        reseeded = runner.generate(SHORT_PROMPTS[:1], other_seed)
        long_context = runner.generate_greedy([LONG_PROMPT], max_tokens=16)

    # generate_greedy returns one (token_ids, text) pair per prompt, with the
    # prompt itself prepended to the text.
    for (_, first_text), (_, second_text), prompt in zip(first, second, SHORT_PROMPTS):
        assert first_text[len(prompt) :].strip(), f"empty greedy completion for {prompt!r}"
        assert first_text == second_text, f"greedy decoding is not reproducible for {prompt!r}"

    # Two properties together, because either alone can hide a defect: a broken
    # sampler can be reproducible by always emitting the argmax, and one
    # reading uninitialised state can vary without honouring the seed.
    assert seeded[0][1][0] == seeded_again[0][1][0], "sampling with a fixed seed is not reproducible"
    assert seeded[0][1][0] != reseeded[0][1][0], "sampling ignores the seed: two seeds gave identical text"

    answer = long_context[0][1][len(LONG_PROMPT) :]
    assert "73915" in answer, f"long-context retrieval failed, model answered: {answer!r}"


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="piecewise",
)
@wait_until_npu_memory_free()
def test_online_multi_turn_chat_and_completions():
    """Online serving: completions, a three-turn chat, and the stop tokens.

    The chat turns go through the checkpoint's own ChatML template, so this
    also covers `trust_remote_code` tokenizer loading and the
    `eos_token_id: [2, 92542]` pair from `generation_config.json`. Without
    92542 the assistant turn never terminates on its own, which shows up as a
    reply truncated by `max_tokens` rather than stopped.
    """
    port = get_open_port()
    server_args = [
        "--trust-remote-code",
        "--max-model-len",
        "4096",
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.8",
        "--port",
        str(port),
    ]

    with RemoteOpenAIServer(MODEL, server_args, server_port=port, auto_port=False) as server:
        completion = requests.post(
            server.url_for("v1", "completions"),
            json={"model": MODEL, "prompt": "Shanghai is", "max_tokens": 32, "temperature": 0.0},
        )
        assert completion.status_code == 200
        assert completion.json()["choices"][0]["text"].strip(), "empty online completion"

        messages = [{"role": "user", "content": "My favourite colour is teal. Please remember it."}]
        replies: list[str] = []
        for follow_up in ("What did I say my favourite colour was?", "用中文再回答一次。"):
            chat = requests.post(
                server.url_for("v1", "chat", "completions"),
                json={"model": MODEL, "messages": messages, "max_tokens": 96, "temperature": 0.0},
            )
            assert chat.status_code == 200
            choice = chat.json()["choices"][0]
            reply = choice["message"]["content"]
            assert reply.strip(), f"empty reply at turn {len(replies) + 1}"
            # A turn that ends on its own proves `<|im_end|>` reached the
            # sampler; one cut off by the token budget proves nothing.
            assert choice["finish_reason"] == "stop", (
                f"turn {len(replies) + 1} did not stop on an end-of-turn token: {choice}"
            )
            replies.append(reply)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": follow_up})

        # The colour was stated in turn one and asked for in turn two, so a
        # miss here means the turns were not assembled into a single prompt.
        assert "teal" in replies[1].lower(), f"multi-turn context was lost, model answered: {replies[1]!r}"
