"""
End-to-end tests for BatchJobAwareScheduler.

This test module verifies the correctness of the BatchJobAwareScheduler
by comparing outputs with the default scheduler.
"""

import pytest

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODEL = "Qwen/Qwen3-0.6B"

example_prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Extended prompts for testing with 20 samples
extended_prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "Machine learning is a subset of",
    "The largest ocean in the world is",
    "Python is a programming language that",
    "The speed of light is approximately",
    "The human brain consists of",
    "Climate change is primarily caused by",
    "The Eiffel Tower is located in",
    "Artificial intelligence can be defined as",
    "The Great Wall of China was built to",
    "Quantum computing uses principles of",
    "The theory of relativity was proposed by",
    "Solar energy is generated through",
    "The human heart pumps approximately",
    "Blockchain technology enables",
    "The Amazon rainforest is home to",
    "Electric vehicles are powered by",
]


@pytest.mark.parametrize("max_tokens", [4])
def test_batch_job_aware_scheduler_basic(max_tokens: int) -> None:
    """Test basic functionality of BatchJobAwareScheduler.

    Compare outputs between BatchJobAwareScheduler and default scheduler
    to ensure correctness.
    """
    with VllmRunner(
        MODEL,
        additional_config={
            "scheduler_config": {
                "batch_job_sched_config": {
                    "enabled": True,
                },
            },
        },
        max_model_len=2048,
        gpu_memory_utilization=0.7,
    ) as vllm_model:
        batch_job_output = vllm_model.generate_greedy(example_prompts, max_tokens)

    with VllmRunner(
        MODEL,
        max_model_len=2048,
        gpu_memory_utilization=0.7,
        async_scheduling=False,
    ) as vllm_model:
        vllm_default_output = vllm_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_default_output,
        outputs_1_lst=batch_job_output,
        name_0="default_scheduler",
        name_1="batch_job_aware_scheduler",
    )


@pytest.mark.parametrize("max_tokens", [4])
def test_batch_job_aware_scheduler_with_async_scheduling(max_tokens: int) -> None:
    """Test basic functionality of BatchJobAwareAsyncScheduler.

    Compare outputs between BatchJobAwareAsyncScheduler and default scheduler
    to ensure correctness.
    """
    with VllmRunner(
        MODEL,
        additional_config={
            "scheduler_config": {
                "batch_job_sched_config": {
                    "enabled": True,
                },
            },
        },
        max_model_len=2048,
        gpu_memory_utilization=0.7,
        async_scheduling=True,
    ) as vllm_model:
        batch_job_output = vllm_model.generate_greedy(example_prompts, max_tokens)

    with VllmRunner(
        MODEL,
        max_model_len=2048,
        gpu_memory_utilization=0.7,
    ) as vllm_model:
        vllm_default_output = vllm_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_default_output,
        outputs_1_lst=batch_job_output,
        name_0="default_scheduler",
        name_1="batch_job_aware_async_scheduler",
    )


@pytest.mark.parametrize("max_tokens", [4])
@pytest.mark.parametrize("chunked_prefill_token_size", [16])
def test_batch_job_aware_scheduler_with_chunked_prefill(max_tokens: int, chunked_prefill_token_size: int) -> None:
    """Test BatchJobAwareScheduler with chunked prefill enabled.

    Verify that BatchJobAwareScheduler works correctly with chunked prefill,
    comparing outputs with the default scheduler using 20 test samples.
    """
    max_num_seqs = chunked_prefill_token_size
    max_num_batched_tokens = chunked_prefill_token_size

    with VllmRunner(
        MODEL,
        additional_config={
            "scheduler_config": {
                "batch_job_sched_config": {
                    "enabled": True,
                },
            },
        },
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=2048,
        gpu_memory_utilization=0.7,
        enable_chunked_prefill=True,
        async_scheduling=False,
    ) as vllm_model:
        batch_job_output = vllm_model.generate_greedy(extended_prompts, max_tokens)

    with VllmRunner(
        MODEL,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=2048,
        gpu_memory_utilization=0.7,
        enable_chunked_prefill=True,
    ) as vllm_model:
        default_output = vllm_model.generate_greedy(extended_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=default_output,
        outputs_1_lst=batch_job_output,
        name_0="default_scheduler_chunked_prefill",
        name_1="batch_job_aware_scheduler_chunked_prefill",
    )


@pytest.mark.parametrize("max_tokens", [4])
def test_batch_job_aware_scheduler_with_custom_config(max_tokens: int) -> None:
    """Test BatchJobAwareScheduler with custom configuration parameters.

    Verify that BatchJobAwareScheduler works correctly with custom EWMA
    parameters and other configuration options using 20 test samples.
    """
    with VllmRunner(
        MODEL,
        additional_config={
            "scheduler_config": {
                "batch_job_sched_config": {
                    "enabled": True,
                    "max_jobs": 10,
                    "reserve_margin_blocks": 4,
                    "reserve_max_blocks": 12,
                    "low_available_tokens_threshold": 2048,
                    "short_decode_token_threshold": 32,
                },
            },
        },
        max_model_len=2048,
        gpu_memory_utilization=0.7,
    ) as vllm_model:
        batch_job_output = vllm_model.generate_greedy(extended_prompts, max_tokens)

    with VllmRunner(
        MODEL,
        max_model_len=2048,
        gpu_memory_utilization=0.7,
        async_scheduling=False,
    ) as vllm_model:
        default_output = vllm_model.generate_greedy(extended_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=default_output,
        outputs_1_lst=batch_job_output,
        name_0="default_scheduler",
        name_1="batch_job_aware_scheduler_custom_config",
    )
