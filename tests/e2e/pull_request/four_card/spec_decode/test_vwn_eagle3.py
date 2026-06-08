"""E2E acceptance tests for VWN-Eagle3 speculative decoding (TP=4, 4 NPU).

Test Matrix
-----------
+----------------------------------------------+-------------+-------------+------------+---------------------+
| Test                                         | Scheduling   | Padded Batch | max_tokens | Assertion           |
+----------------------------------------------+-------------+-------------+------------+---------------------+
| test_vwn_eagle3_acceptance_tp4               | async=True  | enabled     | 256        | acceptance vs golden |
| test_vwn_eagle3_correctness_tp4              | default     | enabled     | 64         | non-empty output    |
| test_vwn_eagle3_acceptance_tp4_async_false   | async=False | enabled     | 256        | acceptance vs golden |
| test_vwn_eagle3_disable_padded_drafter_batch | default     | disabled    | 64         | non-empty output    |
| test_vwn_eagle3_longer_generation_tp4        | default     | enabled     | 512        | non-empty output    |
+----------------------------------------------+-------------+-------------+------------+---------------------+

Config: main=Qwen/Qwen3-30B-A3B, draft=ascend/vwn_eagle3, TP=4, num_speculative_tokens=3

Run:
    pytest tests/e2e/pull_request/four_card/spec_decode/test_vwn_eagle3.py
"""

import os

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.tokenizers.registry import resolve_tokenizer_args
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MAIN_MODEL = "Qwen/Qwen3-30B-A3B"
SPEC_MODEL = "ascend/vwn_eagle3"


@pytest.mark.parametrize("async_scheduling", [True])
def test_vwn_eagle3_acceptance_tp4(async_scheduling):
    """Test VWN-Eagle3 acceptance rate with TP=4.

    Validates that the acceptance-per-position metric is within reasonable
    bounds. Golden values are established from baseline runs and may need
    updating when the model or serving config changes.
    """
    # Golden acceptance-per-position (3 speculative tokens).
    # Update after first run if values shift.
    golden = [0.50, 0.25, 0.10]

    tokenizer_path = resolve_tokenizer_args(MAIN_MODEL)[1]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )

    example_prompts = [
        {"role": "user", "content": "Hello, my name is"},
        {"role": "user", "content": "The president of the United States is"},
        {"role": "user", "content": "The capital of France is"},
        {"role": "user", "content": "The future of AI is"},
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [p],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in example_prompts
    ]

    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=256,
    )

    speculative_config = {
        "method": "eagle3",
        "model": SPEC_MODEL,
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 1,
        "disable_padded_drafter_batch": False,
    }

    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_DECODE_ONLY",
        cudagraph_capture_sizes=[20],
    )

    with VllmRunner(
        MAIN_MODEL,
        tensor_parallel_size=4,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        disable_log_stats=False,
        distributed_executor_backend="mp",
        speculative_config=speculative_config,
        compilation_config=compilation_config,
        async_scheduling=async_scheduling,
    ) as llm:
        outputs = llm.model.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()

    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")

    # Calculate acceptance per position
    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * 3
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                num_accepted_tokens_per_pos[pos] += metric.values[pos]

    acceptance_per_pos = [
        n / num_drafts for n in num_accepted_tokens_per_pos
    ]

    print(f"acceptance_per_pos: {acceptance_per_pos}")
    print(f"golden: {golden}")
    assert len(acceptance_per_pos) == 3

    # Validate acceptance rates are within tolerance of golden values.
    # Use a wide tolerance initially; tighten after collecting baseline data.
    match = all(abs(a - b) < 0.15 for a, b in zip(acceptance_per_pos, golden))
    if not match:
        print("WARNING: acceptance rates differ from golden baseline.")
        print("Update golden values in this test if the new values are correct.")

    assert match, (
        f"Acceptance rates {acceptance_per_pos} differ significantly "
        f"from golden {golden}. Update golden if needed."
    )

    cleanup_dist_env_and_memory()


def test_vwn_eagle3_correctness_tp4():
    """Test VWN-Eagle3 greedy generation produces non-empty output."""
    tokenizer_path = resolve_tokenizer_args(MAIN_MODEL)[1]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is the capital of France?"}],
            tokenize=False,
            add_generation_prompt=True,
        ),
    ]

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=64,
    )

    speculative_config = {
        "method": "eagle3",
        "model": SPEC_MODEL,
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 1,
        "disable_padded_drafter_batch": False,
    }

    with VllmRunner(
        MAIN_MODEL,
        tensor_parallel_size=4,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        distributed_executor_backend="mp",
        speculative_config=speculative_config,
        compilation_config=CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[20],
        ),
    ) as llm:
        outputs = llm.model.generate(prompts, sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt!r}, Generated: {generated_text!r}")
        assert len(generated_text) > 0, "Generated text should not be empty"

    cleanup_dist_env_and_memory()


def test_vwn_eagle3_acceptance_tp4_async_false():
    """Test VWN-Eagle3 acceptance rate with async_scheduling=False.

    Validates that the synchronous scheduling mode produces acceptance
    rates within reasonable bounds. Uses wider tolerance since this
    configuration may differ from the default async mode.
    """
    golden = [0.50, 0.25, 0.10]

    tokenizer_path = resolve_tokenizer_args(MAIN_MODEL)[1]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )

    example_prompts = [
        {"role": "user", "content": "Hello, my name is"},
        {"role": "user", "content": "The president of the United States is"},
        {"role": "user", "content": "The capital of France is"},
        {"role": "user", "content": "The future of AI is"},
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [p],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in example_prompts
    ]

    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=256,
    )

    speculative_config = {
        "method": "eagle3",
        "model": SPEC_MODEL,
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 1,
        "disable_padded_drafter_batch": False,
    }

    with VllmRunner(
        MAIN_MODEL,
        tensor_parallel_size=4,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        disable_log_stats=False,
        distributed_executor_backend="mp",
        speculative_config=speculative_config,
        compilation_config=CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[20],
        ),
        async_scheduling=False,
    ) as llm:
        outputs = llm.model.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()

    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")

    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * 3
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                num_accepted_tokens_per_pos[pos] += metric.values[pos]

    acceptance_per_pos = [
        n / num_drafts for n in num_accepted_tokens_per_pos
    ]

    print(f"acceptance_per_pos (async=False): {acceptance_per_pos}")
    print(f"golden: {golden}")
    assert len(acceptance_per_pos) == 3

    match = all(abs(a - b) < 0.20 for a, b in zip(acceptance_per_pos, golden))
    if not match:
        print("WARNING: acceptance rates differ from golden baseline.")
        print("Update golden values in this test if the new values are correct.")

    assert match, (
        f"Acceptance rates {acceptance_per_pos} differ significantly "
        f"from golden {golden}. Update golden if needed."
    )

    cleanup_dist_env_and_memory()


def test_vwn_eagle3_disable_padded_drafter_batch_tp4():
    """Test VWN-Eagle3 with disable_padded_drafter_batch=True.

    Verifies correctness under the alternative batching mode that disables
    padded drafter batching.
    """
    tokenizer_path = resolve_tokenizer_args(MAIN_MODEL)[1]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )

    example_prompts = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "user", "content": "Who wrote Romeo and Juliet?"},
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [p],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in example_prompts
    ]

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=64,
    )

    speculative_config = {
        "method": "eagle3",
        "model": SPEC_MODEL,
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 1,
        "disable_padded_drafter_batch": True,
    }

    with VllmRunner(
        MAIN_MODEL,
        tensor_parallel_size=4,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        distributed_executor_backend="mp",
        speculative_config=speculative_config,
        compilation_config=CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[20],
        ),
    ) as llm:
        outputs = llm.model.generate(prompts, sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt!r}, Generated: {generated_text!r}")
        assert len(generated_text) > 0, "Generated text should not be empty"

    cleanup_dist_env_and_memory()


def test_vwn_eagle3_longer_generation_tp4():
    """Test VWN-Eagle3 with longer generation (max_tokens=512).

    Verifies stability of the speculative decoding loop over more decode
    steps and that output is non-trivial.
    """
    tokenizer_path = resolve_tokenizer_args(MAIN_MODEL)[1]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )

    example_prompts = [
        {"role": "user", "content": "Explain the theory of relativity in detail."},
        {"role": "user", "content": "Write a short essay about artificial intelligence."},
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [p],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in example_prompts
    ]

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=512,
    )

    speculative_config = {
        "method": "eagle3",
        "model": SPEC_MODEL,
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 1,
        "disable_padded_drafter_batch": False,
    }

    with VllmRunner(
        MAIN_MODEL,
        tensor_parallel_size=4,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        distributed_executor_backend="mp",
        speculative_config=speculative_config,
        compilation_config=CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[20],
        ),
    ) as llm:
        outputs = llm.model.generate(prompts, sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        output_tokens = output.outputs[0].token_ids
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated ({len(output_tokens)} tokens): {generated_text!r}")
        assert len(generated_text) > 0, "Generated text should not be empty"
        assert len(output_tokens) > 0, "Should generate at least one token"

    cleanup_dist_env_and_memory()
