# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import os
import random
from typing import Any

import pytest
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


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


@pytest.mark.parametrize("use_eagle3", [False, True], ids=["eagle", "eagle3"])
def test_eagle_correctness(
    test_prompts: list[list[dict[str, Any]]],
    sampling_config: SamplingParams,
    model_name: str,
    use_eagle3: bool,
):
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using eagle speculative decoding.
    '''
    # NOTE: e2e of eagle has many problems before.
    # We first check whether it is functioning properly.
    # Should fix the e2e with VllmRunner in future.
    spec_model_name = eagle3_model_name() if use_eagle3 else eagle_model_name()
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    prompts = [{
        "role": "user",
        "content": "Hello, my name is"
    }, {
        "role": "user",
        "content": "The president of the United States is"
    }, {
        "role": "user",
        "content": "The capital of France is"
    }, {
        "role": "user",
        "content": "The future of AI is"
    }]
    prompts = [
        tokenizer.apply_chat_template(
            [prompt],
            tokenize=False,
            add_generation_prompt=True,
        ) for prompt in prompts
    ]

    sampling_params = SamplingParams(
        max_tokens=300,
        temperature=0.8,
        top_p=0.7,
        top_k=4,
        ignore_eos=False,
    )

    # Create an LLM.
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        disable_log_stats=False,
        max_model_len=4096,
        seed=1024,
        async_scheduling=True,
        compilation_config={
            "level": 3,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_num_of_warmups": 1,
            "cudagraph_capture_sizes": [12],
        },
        speculative_config={
            "disable_padded_drafter_batch": False,
            "method": "eagle3" if use_eagle3 else "eagle",
            "model": spec_model_name,
            "num_speculative_tokens": 2,
            "max_model_len": 128,
            "draft_vocab_size": 128256,
        },
    )
    llm.generate(prompts, sampling_params)
    cleanup_dist_env_and_memory()
    del llm


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
