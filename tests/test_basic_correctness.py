# SPDX-License-Identifier: Apache-2.0
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/basic_correctness/test_basic_correctness.py`.
"""
import os

import pytest

from vllm import LLM
from vllm.platforms import current_platform

from conftest import VllmRunner
from model_utils import check_outputs_equal

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct"
]

TARGET_TEST_SUITE = os.environ.get("TARGET_TEST_SUITE", "L4")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", ["ASCEND"])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("enforce_eager", [False])
def test_models(
    monkeypatch: pytest.MonkeyPatch,
    hf_runner,
    model: str,
    backend: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    with monkeypatch.context() as m:
        m.setenv("VLLM_ATTENTION_BACKEND", backend)

        # 5042 tokens for gemma2
        # gemma2 has alternating sliding window size of 4096
        # we need a prompt with more than 4096 tokens to test the sliding window
        prompt = "The following numbers of the sequence " + ", ".join(
            str(i) for i in range(1024)) + " are:"
        example_prompts = [prompt]
        with hf_runner(model) as hf_model:
            hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

        with VllmRunner(model,
                        max_model_len=8192,
                        enforce_eager=enforce_eager,
                        gpu_memory_utilization=0.7) as vllm_model:
            vllm_outputs = vllm_model.generate_greedy(example_prompts,
                                                      max_tokens)

        check_outputs_equal(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )
        print(hf_outputs)
        print(vllm_outputs)

@pytest.mark.parametrize(
    "model, distributed_executor_backend, attention_backend, "
    "tp", [
        ("Qwen/QwQ-32B", "mp", "ASCEND", 4),
        # ("distilbert/distilgpt2", "mp", "FLASHINFER", "A100"),
        # ("meta-llama/Meta-Llama-3-8B", "ray", "FLASHINFER", "A100"),
    ])
def test_models_distributed(
    monkeypatch: pytest.MonkeyPatch,
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    distributed_executor_backend: str,
    attention_backend: str,
    tp: int,
) -> None:

    # if test_suite != TARGET_TEST_SUITE:
    #     pytest.skip(f"Skip test for {test_suite}")

    with monkeypatch.context() as monkeypatch_context:

        if attention_backend:
            monkeypatch_context.setenv(
                "VLLM_ATTENTION_BACKEND",
                attention_backend,
            )

        max_tokens = 5

        # NOTE: take care of the order. run vLLM first, and then run HF.
        # vLLM needs a fresh new process without cuda initialization.
        # if we run HF first, the cuda initialization will be done and it
        # will hurt multiprocessing backend with fork method
        # (the default method).
        with vllm_runner(
                model,
                tensor_parallel_size=tp,
                distributed_executor_backend=distributed_executor_backend,
        ) as vllm_model:
            vllm_outputs = vllm_model.generate_greedy(example_prompts,
                                                      max_tokens)

        with hf_runner(model) as hf_model:
            hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

        check_outputs_equal(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )