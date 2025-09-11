# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODEL = "Qwen/Qwen3-0.6B"


def test_concurrent_partial_prefill():
    with VllmRunner(MODEL,
                    additional_config={
                        'ascend_scheduler_config': {
                            'enabled': True,
                        },
                    },
                    max_num_seqs=3,
                    max_num_batched_tokens=2048,
                    enforce_eager=True,
                    max_model_len=2048,
                    gpu_memory_utilization=0.7) as vllm_model:
        outputs = vllm_model.model.generate(["Hello my name is Robert and I"] *
                                            3)
        assert len(outputs) == 3
        for output in outputs:
            assert len(output.outputs) == 1


def test_prefix_cache_stats_is_recorded():
    with VllmRunner(MODEL,
                    additional_config={
                        'ascend_scheduler_config': {
                            'enabled': True,
                        },
                    },
                    max_num_seqs=3,
                    max_num_batched_tokens=2048,
                    enforce_eager=True,
                    max_model_len=2048,
                    gpu_memory_utilization=0.7) as vllm_model:
        # 17 tokens will make sure first 16 tokens are cached in a block
        input_tokens = {"prompt_token_ids": [101] * 129}
        _ = vllm_model.model.generate([input_tokens])
        outputs = vllm_model.model.generate([input_tokens])
        assert outputs[0].num_cached_tokens == 128


@pytest.mark.parametrize("max_tokens",
                         [4])  # cannot align results when max_tokens > 4
@pytest.mark.parametrize("chunked_prefill_token_size", [16])
def test_chunked_prefill_with_ascend_scheduler(
        max_tokens: int, chunked_prefill_token_size: int) -> None:
    example_prompts = [
        "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs."
    ]
    max_num_seqs = chunked_prefill_token_size
    max_num_batched_tokens = chunked_prefill_token_size
    with VllmRunner(MODEL,
                    additional_config={
                        'ascend_scheduler_config': {
                            'enabled': True,
                            'enable_chunked_prefill': True,
                        },
                    },
                    max_num_seqs=max_num_seqs,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_model_len=2048,
                    gpu_memory_utilization=0.7) as vllm_model:
        chunked_prefill_output = vllm_model.generate_greedy(
            example_prompts, max_tokens)

    with VllmRunner(MODEL,
                    additional_config={
                        'ascend_scheduler_config': {
                            'enabled': True,
                        },
                    },
                    max_model_len=2048,
                    gpu_memory_utilization=0.7) as vllm_model:
        vllm_output = vllm_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_output,
        outputs_1_lst=chunked_prefill_output,
        name_0="vllm_output",
        name_1="chunked_prefill_output",
    )


def test_async_scheduling() -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] * 10
    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=10,
                                     stop_token_ids=None)

    with VllmRunner(
            "Qwen/Qwen2.5-0.5B-Instruct",
            max_model_len=4096,
            max_num_seqs=50,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            async_scheduling=True,
    ) as vllm_model:
        vllm_model.generate(prompts, sampling_params=sampling_params)
