# SPDX-License-Identifier: Apache-2.0
"""Compare the with and without chunked prefill on AscendScheduler

It tests chunked prefill. Chunked prefill can be enabled by 
`additional_config={'ascend_scheduler_config': {'enabled': True, 'enable_chunked_prefill': True,},}`. 
If prefill size exceeds max_num_batched_tokens, prefill requests are chunked.

Run `pytest tests/e2e/singlecard/core/ascend_scheduler/test_chunk_prefill.py`.
"""
import pytest

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODELS = [
    "Qwen/Qwen3-0.6B-Base",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens",
                         [4])  # cannot align results when max_tokens > 4
@pytest.mark.parametrize("chunked_prefill_token_size", [16])
def test_chunked_prefill_with_ascend_scheduler(
        example_prompts, model: str, max_tokens: int,
        chunked_prefill_token_size: int) -> None:
    max_num_seqs = chunked_prefill_token_size
    max_num_batched_tokens = chunked_prefill_token_size
    with VllmRunner(model,
                    additional_config={
                        'ascend_scheduler_config': {
                            'enabled': True,
                            'enable_chunked_prefill': True,
                        },
                    },
                    max_num_seqs=max_num_seqs,
                    max_num_batched_tokens=max_num_batched_tokens,
                    enforce_eager=True,
                    max_model_len=2048,
                    gpu_memory_utilization=0.7) as vllm_model:
        chunked_prefill_output = vllm_model.generate_greedy(
            example_prompts, max_tokens)

    with VllmRunner(model,
                    additional_config={
                        'ascend_scheduler_config': {
                            'enabled': True,
                        },
                    },
                    enforce_eager=True,
                    max_model_len=2048,
                    gpu_memory_utilization=0.7) as vllm_model:
        vllm_output = vllm_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_output,
        outputs_1_lst=chunked_prefill_output,
        name_0="vllm_output",
        name_1="chunked_prefill_output",
    )
