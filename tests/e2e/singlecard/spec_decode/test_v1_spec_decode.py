# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import os
import random
from typing import Any

import pytest
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODELS = {
    "eagle": {
        "main": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        "spec": "vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B",
    },
    "eagle3": {
        "main": "Qwen/Qwen3-8B",
        "spec": "RedHatAI/Qwen3-8B-speculator.eagle3",
    },
}

# NOTE: golden may change (eagle_proposer only runs in eager mode currently),
# thus please update it if ci fails but you have better acceptance
BASELINES = {
    "eagle": [0.74, 0.44, 0.29],
    "eagle3": [0.68, 0.40, 0.18],
}

BASELINES_SP = {
    "eagle3": [0.68, 0.40, 0.18],
}


@pytest.fixture
def test_prompts():
    prompt_types = ["repeat", "sentence"]
    num_prompts = 100
    prompts = []

    random.seed(0)
    random_prompt_type_choices = random.choices(prompt_types, k=num_prompts)

    # Generate a mixed batch of prompts, some of which can be easily
    # predicted by n-gram matching and some which likely cannot.
    for kind in random_prompt_type_choices:
        word_choices = ["test", "temp", "hello", "where"]
        word = random.choice(word_choices)
        if kind == "repeat":
            prompt = f"""
            please repeat the word '{word}' 10 times.
            give no other output than the word at least ten times in a row,
            in lowercase with spaces between each word and without quotes.
            """
        elif kind == "sentence":
            prompt = f"""
            please give a ten-word sentence that
            uses the word {word} at least once.
            give no other output than that simple sentence without quotes.
            """
        else:
            raise ValueError(f"Unknown prompt type: {kind}")
        prompts.append([{"role": "user", "content": prompt}])

    return prompts


@pytest.fixture
def sampling_config():
    return SamplingParams(temperature=0, max_tokens=10, ignore_eos=False)


@pytest.fixture
def model_name():
    return "LLM-Research/Meta-Llama-3.1-8B-Instruct"


def eagle_model_name():
    return "vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B"


def eagle3_model_name():
    return "vllm-ascend/EAGLE3-LLaMA3.1-Instruct-8B"


def test_ngram_correctness(
    test_prompts: list[list[dict[str, Any]]],
    sampling_config: SamplingParams,
    model_name: str,
):
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using ngram speculative decoding.
    '''

    with VllmRunner(
            model_name,
            max_model_len=1024,
            cudagraph_capture_sizes=[1, 2, 4, 8],
    ) as ref_llm:
        ref_outputs = ref_llm.model.chat(test_prompts, sampling_config)

    with VllmRunner(
            model_name,
            speculative_config={
                "method": "ngram",
                "prompt_lookup_max": 5,
                "prompt_lookup_min": 3,
                "num_speculative_tokens": 3,
            },
            max_model_len=1024,
            cudagraph_capture_sizes=[1, 2, 4, 8],
    ) as runner:
        spec_outputs = runner.model.chat(test_prompts, sampling_config)
    matches = 0
    misses = 0
    for ref_output, spec_output in zip(ref_outputs, spec_outputs):
        if ref_output.outputs[0].text == spec_output.outputs[0].text:
            matches += 1
        else:
            misses += 1
            print(f"ref_output: {ref_output.outputs[0].text}")
            print(f"spec_output: {spec_output.outputs[0].text}")

    # Heuristic: expect at least 70% of the prompts to match exactly
    # Upon failure, inspect the outputs to check for inaccuracy.
    assert matches > int(0.66 * len(ref_outputs))


def test_suffix_correctness(
    test_prompts: list[list[dict[str, Any]]],
    sampling_config: SamplingParams,
    model_name: str,
):
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using ngram speculative decoding.
    '''
    with VllmRunner(model_name,
                    max_model_len=1024,
                    cudagraph_capture_sizes=[1, 2, 4, 8]) as ref_llm:
        ref_outputs = ref_llm.model.chat(test_prompts, sampling_config)

    with VllmRunner(model_name,
                    speculative_config={
                        "method": "suffix",
                        "num_speculative_tokens": 8,
                    },
                    cudagraph_capture_sizes=[1, 2, 4, 8],
                    max_model_len=1024) as runner:
        spec_outputs = runner.model.chat(test_prompts, sampling_config)
    matches = 0
    misses = 0
    for ref_output, spec_output in zip(ref_outputs, spec_outputs):
        if ref_output.outputs[0].text == spec_output.outputs[0].text:
            matches += 1
        else:
            misses += 1
            print(f"ref_output: {ref_output.outputs[0].text}")
            print(f"spec_output: {spec_output.outputs[0].text}")

    # Heuristic: expect at least 70% of the prompts to match exactly
    # Upon failure, inspect the outputs to check for inaccuracy.
    assert matches > int(0.66 * len(ref_outputs))


def test_suffix_acceptance(
    test_prompts: list[list[dict[str, Any]]],
    sampling_config: SamplingParams,
    model_name: str,
):
    '''
    Check that suffix decoding caching takes effect and improves acceptance
    lengths and acceptance rates over multiple runs of the same prompts.
    '''
    num_draft = []
    num_accept = []
    with VllmRunner(model_name,
                    speculative_config={
                        "method": "suffix",
                        "suffix_decoding_max_spec_factor": 2.0,
                        "suffix_decoding_max_cached_requests": 1000,
                        "num_speculative_tokens": 10,
                    },
                    max_model_len=1024,
                    cudagraph_capture_sizes=[1, 2, 4, 8],
                    disable_log_stats=False) as runner:
        for i in range(10):
            runner.model.chat(test_prompts[i], sampling_config)
            metrics = runner.model.get_metrics()
            for metric in metrics:
                print(metric)
                if metric.name == "vllm:spec_decode_num_draft_tokens":
                    num_draft.append(metric.value)
                if metric.name == "vllm:spec_decode_num_accepted_tokens":
                    num_accept.append(metric.value)
    # Calculate the acceptance rates for the first and last runs.
    first_accept_tokens = num_accept[0]
    first_draft_tokens = num_draft[0]
    first_accept_rate = first_accept_tokens / first_draft_tokens

    # Take the diff since the stats are cumulative.
    last_accept_tokens = num_accept[-1] - num_accept[-2]
    last_draft_tokens = num_draft[-1] - num_draft[-2]
    last_accept_rate = last_accept_tokens / last_draft_tokens

    # Expect the acceptance length to improve.
    assert first_accept_tokens < last_accept_tokens

    # Expect the acceptance rate to improve.
    assert first_accept_rate < last_accept_rate

    # Heuristic: expect at least 80% acceptance rate at the end.
    assert last_accept_rate > 0.60


@pytest.mark.parametrize("use_eagle3", [True], ids=["eagle3"])
def test_eagle_logprobs(
    model_name: str,
    use_eagle3: bool,
):
    prompt = {"role": "user", "content": "Hello world " * 10}
    sampling_params = SamplingParams(temperature=0,
                                     logprobs=1,
                                     max_tokens=10,
                                     ignore_eos=False)

    ref_llm = LLM(model=model_name, max_model_len=2048)
    ref_outputs = ref_llm.chat([prompt], sampling_params)
    ref_logprobs = []
    for output in ref_outputs[0].outputs:
        for logprobs in output.logprobs:
            for token_id in logprobs:
                ref_logprobs.append(logprobs[token_id])
    del ref_llm

    spec_model_name = eagle3_model_name() if use_eagle3 else eagle_model_name()
    with VllmRunner(
            model_name,
            max_num_seqs=1,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=0.6,
            speculative_config={
                "method": "eagle3" if use_eagle3 else "eagle",
                "model": spec_model_name,
                "num_speculative_tokens": 2,
                "max_model_len": 128,
            },
            max_model_len=128,
            cudagraph_capture_sizes=[1, 2, 4, 8],
    ) as runner:
        spec_outputs = runner.model.chat([prompt], sampling_params)

    # Collect logprobs outputs from spec decode LLM.
    spec_logprobs = []
    for output in spec_outputs[0].outputs:
        for logprobs in output.logprobs:
            for token_id in logprobs:
                spec_logprobs.append(logprobs[token_id])

    for ref_logprob, spec_logprob in zip(ref_logprobs, spec_logprobs):
        assert math.isclose(ref_logprob.logprob,
                            spec_logprob.logprob,
                            rel_tol=5e-2,
                            abs_tol=1e-1)
        assert ref_logprob.rank == spec_logprob.rank
        assert ref_logprob.decoded_token == spec_logprob.decoded_token


@pytest.mark.parametrize("method", MODELS.keys())
@pytest.mark.parametrize("num_speculative_tokens", [3])
@pytest.mark.parametrize("disable_padded_drafter_batch", [True, False])
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_llama_qwen_eagle_acceptance(
    method: str,
    num_speculative_tokens: int,
    disable_padded_drafter_batch: bool,
    async_scheduling: bool,
):
    if disable_padded_drafter_batch and async_scheduling:
        pytest.skip(
            "skip disable_padded_drafter_batch=True and async_scheduling=True",
        )

    main_model_name = MODELS[method]["main"]
    spec_model_name = MODELS[method]["spec"]

    tokenizer = AutoTokenizer.from_pretrained(
        main_model_name,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=256,
    )

    prompts = [
        {
            "role": "user",
            "content": "Hello, my name is",
        },
        {
            "role": "user",
            "content": "The president of the United States is",
        },
        {
            "role": "user",
            "content": "The capital of France is",
        },
        {
            "role": "user",
            "content": "The future of AI is",
        },
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [prompt],
            tokenize=False,
            add_generation_prompt=True,
        ) for prompt in prompts
    ]

    speculative_config = {
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
        "disable_padded_drafter_batch": disable_padded_drafter_batch,
        "model": spec_model_name,
    }

    compilation_config = CompilationConfig(cudagraph_capture_sizes=[12])

    with VllmRunner(
            main_model_name,
            max_model_len=2048,
            disable_log_stats=False,
            tensor_parallel_size=1,
            max_num_seqs=256,
            distributed_executor_backend="mp",
            gpu_memory_utilization=0.7,
            speculative_config=speculative_config,
            compilation_config=compilation_config,
            async_scheduling=async_scheduling,
    ) as llm:
        _ = llm.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()

    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * num_speculative_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                num_accepted_tokens_per_pos[pos] += metric.values[pos]

    acceptance_per_pos = [
        num_accepted_tokens / num_drafts
        for num_accepted_tokens in num_accepted_tokens_per_pos
    ]
    golden = BASELINES[method]

    match = all(abs(a - b) < 0.06 for a, b in zip(acceptance_per_pos, golden))
    if not match:
        print(f"acceptance_per_pos: {acceptance_per_pos}")
        print(f"golden: {golden}")

    assert match


# TODO the function of sp in eagle3 is improving gradually,
# there are still problems when enable sp + dp and some unknown scenes.
# this e2e should also be improving gradually.
@pytest.mark.parametrize("method", ["eagle3"])
@pytest.mark.parametrize("num_speculative_tokens", [3])
@pytest.mark.parametrize("disable_padded_drafter_batch", [True, False])
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_eagle3_sp_acceptance(
    method: str,
    num_speculative_tokens: int,
    disable_padded_drafter_batch: bool,
    async_scheduling: bool,
):
    if disable_padded_drafter_batch and async_scheduling:
        pytest.skip(
            "skip disable_padded_drafter_batch=True and async_scheduling=True",
        )

    main_model_name = MODELS[method]["main"]
    spec_model_name = MODELS[method]["spec"]

    tokenizer = AutoTokenizer.from_pretrained(
        main_model_name,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=256,
    )

    # sp will only be enabled when query_lens > 1000
    prompts = [
        {
            "role": "user",
            "content": " " * 1000 + "Hello, my name is",
        },
        {
            "role": "user",
            "content": " " * 1000 + "The president of the United States is",
        },
        {
            "role": "user",
            "content": " " * 1000 + "The capital of France is",
        },
        {
            "role": "user",
            "content": " " * 1000 + "The future of AI is",
        },
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [prompt],
            tokenize=False,
            add_generation_prompt=True,
        ) for prompt in prompts
    ]

    speculative_config = {
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
        "disable_padded_drafter_batch": disable_padded_drafter_batch,
        "model": spec_model_name,
    }

    compilation_config = CompilationConfig(cudagraph_capture_sizes=[12])

    with VllmRunner(
            main_model_name,
            enforce_eager=True,
            max_model_len=8192,
            disable_log_stats=False,
            tensor_parallel_size=1,
            max_num_seqs=256,
            distributed_executor_backend="mp",
            gpu_memory_utilization=0.7,
            speculative_config=speculative_config,
            compilation_config=compilation_config,
            async_scheduling=async_scheduling,
    ) as llm:
        _ = llm.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()

    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * num_speculative_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                num_accepted_tokens_per_pos[pos] += metric.values[pos]

    acceptance_per_pos = [
        num_accepted_tokens / num_drafts
        for num_accepted_tokens in num_accepted_tokens_per_pos
    ]
    golden = BASELINES_SP[method]

    match = all(abs(a - b) < 0.06 for a, b in zip(acceptance_per_pos, golden))
    if not match:
        print(f"acceptance_per_pos: {acceptance_per_pos}")
        print(f"golden: {golden}")

    assert match
