from __future__ import annotations

import random
from typing import Any

import pytest
from vllm import LLM, SamplingParams


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
    return "wemaster/deepseek_mtp_main_random_bf16"


@pytest.mark.skipif(
    True, reason="TODO: Enable me after test_mtp_correctness is fixed")
def test_mtp_correctness(
    test_prompts: list[list[dict[str, Any]]],
    sampling_config: SamplingParams,
    model_name: str,
):
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using mtp speculative decoding.
    '''
    ref_llm = LLM(model=model_name, max_model_len=256, enforce_eager=True)
    ref_outputs = ref_llm.chat(test_prompts, sampling_config)
    del ref_llm

    spec_llm = LLM(model=model_name,
                   trust_remote_code=True,
                   speculative_config={
                       "method": "deepseek_mtp",
                       "num_speculative_tokens": 1,
                   },
                   max_model_len=256,
                   enforce_eager=True)
    spec_outputs = spec_llm.chat(test_prompts, sampling_config)
    matches = 0
    misses = 0
    for ref_output, spec_output in zip(ref_outputs, spec_outputs):
        if ref_output.outputs[0].text == spec_output.outputs[0].text:
            matches += 1
        else:
            misses += 1
            print(f"ref_output: {ref_output.outputs[0].text}")
            print(f"spec_output: {spec_output.outputs[0].text}")

    # Heuristic: expect at least 66% of the prompts to match exactly
    # Upon failure, inspect the outputs to check for inaccuracy.
    assert matches > int(0.66 * len(ref_outputs))
    del spec_llm
