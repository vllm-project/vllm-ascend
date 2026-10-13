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
#

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODELS = ["Qwen/Qwen3-0.6B"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [True])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_qwen3_dense_eager_mode(
    model: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.5,
        top_p=0.95,
        top_k=10,
        repetition_penalty=1.03,
        logprobs=2,
        prompt_logprobs=2,
        logit_bias={0: -1.0, 1: 0.5},
        min_p=0.01,
        bad_words=["the", " the"],
    )
    with VllmRunner(
        model,
        max_model_len=1024,
        enforce_eager=enforce_eager,
        async_scheduling=True,
    ) as runner:
        runner.generate(prompts, sampling_params)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [False])
@pytest.mark.parametrize(
    "compilation_config",
    [
        pytest.param({"cudagraph_mode": "FULL_DECODE_ONLY"}, id="full_decode_only"),
        pytest.param({}, id="default_full_and_piecewise"),
    ],
)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen3_dense_graph_mode(
    model: str,
    max_tokens: int,
    enforce_eager: bool,
    compilation_config: dict,
) -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
        model,
        max_model_len=1024,
        enforce_eager=enforce_eager,
        compilation_config=compilation_config,
    ) as runner:
        outputs = runner.model.generate(prompts, sampling_params)

    expected_outputs = [
        " Lina. I'm a 22-year-old student from China.",
        " the same as the president of the United Nations. This is because the president",
        " Paris. The capital of France is also the capital of the Republic of France",
        " not just about the technology itself but also about the human aspect-how we",
    ]

    matches = 0
    misses = 0
    for output, expected_output in zip(outputs, expected_outputs):
        if output.outputs[0].text[:10] == expected_output[:10]:
            matches += 1
        else:
            misses += 1
            print(f"output: {output.outputs[0].text}")
            print(f"expected_output: {expected_output}")

    assert misses == 0


@pytest.mark.parametrize("model", MODELS)
@patch.dict(os.environ, {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
def test_qwen3_mrv1_mrv2_seeded_tokens_and_logprobs(model: str) -> None:
    # Different prefill lengths exercise FIA's packed-query offset sensitivity.
    prompts = [
        "Compute 12 * 13.",
        "Explain the purpose of a cache.",
        "Write a Python loop.",
        ("A cache reuses earlier computation. " * 300) + "Summarize.",
    ]
    settings = [
        SamplingParams(temperature=0.0, seed=0, max_tokens=32, ignore_eos=True, logprobs=5),
        SamplingParams(temperature=0.6, seed=0, top_k=20, top_p=0.95, max_tokens=32, ignore_eos=True, logprobs=5),
        SamplingParams(temperature=0.6, seed=17, top_k=20, top_p=0.95, max_tokens=32, ignore_eos=True, logprobs=5),
    ]
    results = []
    for runner_version in ["0", "1"]:
        with (
            patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": runner_version}),
            VllmRunner(
                model,
                dtype="bfloat16",
                max_model_len=4096,
                max_num_batched_tokens=4096,
                max_num_seqs=16,
                enforce_eager=True,
                enable_prefix_caching=False,
                async_scheduling=False,
                gpu_memory_utilization=0.85,
            ) as runner,
        ):
            assert runner.model.llm_engine.vllm_config.use_v2_model_runner == (runner_version == "1")
            cases = []
            # Repeating seeded requests also exercises reused request slots.
            for params in settings + settings[1:]:
                outputs = runner.model.generate(prompts, params, use_tqdm=False)
                cases.append([output.outputs[0] for output in outputs])
            results.append(cases)
    for v1_case, v2_case in zip(*results, strict=True):
        for v1_output, v2_output in zip(v1_case, v2_case, strict=True):
            assert v1_output.token_ids == v2_output.token_ids
            assert v1_output.logprobs is not None and v2_output.logprobs is not None
            for v1_step, v2_step in zip(v1_output.logprobs, v2_output.logprobs, strict=True):
                assert v1_step.keys() == v2_step.keys()
                for token_id in v1_step:
                    assert v1_step[token_id].rank == v2_step[token_id].rank
                    assert v1_step[token_id].logprob == v2_step[token_id].logprob


def _generate_with_preemption(llm):
    prompts = [
        "Compute 12 * 13.",
        "Explain matrix multiplication.",
        ("A cache reuses previous results. " * 100) + "Summarize.",
        "Write a short Python loop.",
    ]
    engine = llm.llm_engine
    tokenizer = llm.get_tokenizer()
    for index, (prompt, seed, temperature) in enumerate(zip(prompts, [0, 17, 42, 0], [0.6, 0.6, 0.6, 0.0])):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        settings = SamplingParams(
            temperature=temperature, seed=seed, top_k=20, top_p=0.95, max_tokens=48, ignore_eos=True, logprobs=5
        )
        engine.add_request(str(index), text, settings)
    requests = list(engine.engine_core.engine_core.scheduler.requests.values())
    outputs = {}
    preempted = False
    while engine.has_unfinished_requests():
        for output in engine.step():
            outputs[output.request_id] = output
        if not preempted and len(outputs) == len(prompts):
            if min(len(output.outputs[0].token_ids) for output in outputs.values()) >= 8:
                assert engine.reset_prefix_cache(reset_running_requests=True)
                preempted = True
    assert preempted and all(request.num_preemptions > 0 for request in requests)
    assert all(output.finished for output in outputs.values())
    return [outputs[str(index)].outputs[0] for index in range(len(prompts))]


@pytest.mark.parametrize("model", MODELS)
@patch.dict(os.environ, {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
def test_qwen3_mrv1_mrv2_sampling_after_preemption(model: str) -> None:
    results = []
    for runner_version in ["0", "1"]:
        with (
            patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": runner_version}),
            VllmRunner(
                model,
                dtype="bfloat16",
                max_model_len=2048,
                max_num_batched_tokens=256,
                max_num_seqs=8,
                enforce_eager=True,
                enable_prefix_caching=False,
                async_scheduling=False,
                gpu_memory_utilization=0.85,
            ) as runner,
        ):
            results.append(_generate_with_preemption(runner.model))
    # Compare runners under the same recomputation schedule. Recomputing KV
    # can change floating-point results compared with uninterrupted decoding.
    for v1_output, v2_output in zip(*results, strict=True):
        assert v1_output.token_ids == v2_output.token_ids
        assert v1_output.logprobs is not None and v2_output.logprobs is not None
        for v1_step, v2_step in zip(v1_output.logprobs, v2_output.logprobs, strict=True):
            assert v1_step.keys() == v2_step.keys()
            for token_id in v1_step:
                assert v1_step[token_id].rank == v2_step[token_id].rank
                assert v1_step[token_id].logprob == v2_step[token_id].logprob


@pytest.mark.parametrize("model", MODELS)
@patch.dict(os.environ, {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
def test_qwen3_mrv1_mrv2_unseeded_greedy(model: str) -> None:
    prompts = ["Compute 12 * 13.", "Explain the purpose of a cache.", ("A cache reuses previous results. " * 100)]
    results = []
    for runner_version in ["0", "1"]:
        with (
            patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": runner_version}),
            VllmRunner(
                model,
                dtype="bfloat16",
                max_model_len=4096,
                max_num_batched_tokens=256,
                max_num_seqs=8,
                enforce_eager=True,
                enable_prefix_caching=False,
                async_scheduling=False,
                gpu_memory_utilization=0.85,
            ) as runner,
        ):
            cases = []
            for logprobs in [None, 5]:
                settings = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True, logprobs=logprobs)
                outputs = runner.model.generate(prompts, settings, use_tqdm=False)
                cases.append([(x.outputs[0].token_ids, x.outputs[0].logprobs) for x in outputs])
            results.append(cases)
    assert results[0] == results[1]
