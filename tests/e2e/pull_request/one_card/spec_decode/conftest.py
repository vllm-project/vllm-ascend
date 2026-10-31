from __future__ import annotations

import random
from typing import Any

import pytest
from vllm import SamplingParams


@pytest.fixture
def test_prompts() -> list[list[dict[str, Any]]]:
    prompt_types = ["repeat", "sentence"]
    num_prompts = 100
    prompts = []

    random.seed(0)
    random_prompt_type_choices = random.choices(prompt_types, k=num_prompts)

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
def sampling_config() -> SamplingParams:
    return SamplingParams(temperature=0, max_tokens=10, ignore_eos=False)


@pytest.fixture
def model_name() -> str:
    return "LLM-Research/Meta-Llama-3.1-8B-Instruct"


@pytest.fixture
def vl_model_name() -> str:
    return "Qwen/Qwen3-VL-8B-Instruct"
