"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/basic_correctness/test_basic_correctness.py`.
"""
import os

import pytest
import vllm  # noqa: F401
from conftest import VllmRunner

import vllm_ascend  # noqa: F401

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
]

TARGET_TEST_SUITE = os.environ.get("TARGET_TEST_SUITE", "L4")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half", "float16"])
@pytest.mark.parametrize("max_tokens", [5])
def test_models(
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    os.environ["VLLM_ATTENTION_BACKEND"] = "ASCEND"

    # 5042 tokens for gemma2
    # gemma2 has alternating sliding window size of 4096
    # we need a prompt with more than 4096 tokens to test the sliding window
    prompt = "The following numbers of the sequence " + ", ".join(
        str(i) for i in range(1024)) + " are:"
    example_prompts = [prompt]

    with VllmRunner(model,
                    max_model_len=8192,
                    dtype=dtype,
                    enforce_eager=False,
                    gpu_memory_utilization=0.7) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
