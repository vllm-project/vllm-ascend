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
