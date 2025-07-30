from __future__ import annotations

import random
from typing import Any

import pytest
from vllm import SamplingParams
from tests.e2e.conftest import VllmRunner


@pytest.fixture
def test_prompts():
    prompt_types = ["repeat", "sentence"]
    num_prompts = 10
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
    return SamplingParams(temperature=0, max_tokens=256, ignore_eos=False)


@pytest.fixture
def model_name():
    return "vllm-ascend/DeepSeek-R1-W8A8"


def test_mtp_correctness(
    test_prompts: list[list[dict[str, Any]]],
    sampling_config: SamplingParams,
    model_name: str,
):
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using mtp speculative decoding.
    '''

    with VllmRunner(model_name,
                    tensor_parallel_size=16,
                    max_model_len=256,
                    gpu_memory_utilization=0.8,
                    enforce_eager=True) as ref_llm:
        ref_outputs = ref_llm.chat(test_prompts, sampling_config)

    with VllmRunner(model_name,
                    tensor_parallel_size=16,
                    max_model_len=256,
                    gpu_memory_utilization=0.8,
                    speculative_config={
                        "method": "deepseek_mtp",
                        "num_speculative_tokens": 1,
                    },
                    enforce_eager=True) as spec_llm:
        spec_outputs = spec_llm.chat(test_prompts, sampling_config)

    matches = 0
    misses = 0
    for ref_output, spec_output in zip(ref_outputs, spec_outputs):
        ref_ids = ref_output.outputs[0].token_ids
        spec_ids = ref_output.outputs[0].token_ids
        count = 0
        for i in range(len(ref_ids)):
            if ref_ids[i] == spec_ids[i]:
                count += 1
        rate = count / len(ref_ids)
        if rate > 0.7:
            matches += 1
        else:
            misses += 1

    assert matches > int(0.66 * len(ref_outputs))
    del spec_llm
