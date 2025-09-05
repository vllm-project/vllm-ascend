from __future__ import annotations

import os

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


@pytest.fixture
def sampling_config():
    return SamplingParams(temperature=0, max_tokens=256, ignore_eos=False)


@pytest.fixture
def model_name():
    return "wemaster/deepseek_mtp_main_random_bf16"


def mtp_correctness(
    sampling_config: SamplingParams,
    model_name: str,
    num_speculative_tokens: int,
):
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    '''
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using mtp speculative decoding.
    '''
    with VllmRunner(model_name,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.7,
                    max_model_len=256,
                    enforce_eager=True) as ref_llm:
        ref_outputs = ref_llm.generate(example_prompts, sampling_config)

    with VllmRunner(
            model_name,
            tensor_parallel_size=1,
            max_num_seqs=256,
            gpu_memory_utilization=0.7,
            distributed_executor_backend="mp",
            enable_expert_parallel=True,
            speculative_config={
                "method": "deepseek_mtp",
                "num_speculative_tokens": num_speculative_tokens,
            },
            enforce_eager=True,
            max_model_len=2000,
            additional_config={"ascend_scheduler_config": {
                "enabled": False
            }}) as spec_llm:
        spec_outputs = spec_llm.generate(example_prompts, sampling_config)

    matches = 0
    misses = 0
    for ref_output, spec_output in zip(ref_outputs, spec_outputs):
        ref_token_ids = ref_output[0][0]
        spec_token_ids = spec_output[0][0]
        if ref_token_ids == spec_token_ids[:len(ref_token_ids)]:
            matches += 1
        else:
            misses += 1
            print(f"ref_output: {ref_output[1][0]}")
            print(f"spec_output: {spec_output[1][0]}")

    # Heuristic: expect at least 66% of the prompts to match exactly
    # Upon failure, inspect the outputs to check for inaccuracy.
    assert matches > int(0.66 * len(ref_outputs))
    del spec_llm


def test_mtp1_correctness(
    sampling_config: SamplingParams,
    model_name: str,
):
    mtp_correctness(sampling_config, model_name, 1)


def test_mtp2_correctness(
    sampling_config: SamplingParams,
    model_name: str,
):
    mtp_correctness(sampling_config, model_name, 2)
