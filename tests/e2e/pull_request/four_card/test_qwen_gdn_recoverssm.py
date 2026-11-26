# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.utils import assert_valid_generation_outputs


@pytest.mark.parametrize("cache_mode", ["none", "align"])
def test_qwen36_recoverssm_matches_mtp_baseline_and_graph(cache_mode):
    prompts = ["What is 17 times 23?", "Explain why the sky is blue.", "Write a Python function to reverse a list."]
    if cache_mode == "align":
        prefix = "This is shared background context for the following question. " * 128
        prompts = [prefix + prompt for prompt in prompts]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
    reuse_prompts = prompts if cache_mode == "align" else list(reversed(prompts))
    for use_replayssm, eager in [(False, True), (True, True), (True, False)]:
        with VllmRunner(
            "Qwen/Qwen3.6-35B-A3B",
            tensor_parallel_size=4,
            dtype="bfloat16",
            mamba_ssm_cache_dtype="float32",
            mamba_cache_mode=cache_mode,
            enable_prefix_caching=cache_mode == "align",
            use_replayssm=use_replayssm,
            speculative_config={"method": "mtp", "num_speculative_tokens": 4},
            enforce_eager=eager,
            max_model_len=4096,
            max_num_seqs=8,
            max_num_batched_tokens=1024,
            gpu_memory_utilization=0.9,
        ) as runner:
            model = runner.model
            vocab_size = model.llm_engine.model_config.get_vocab_size()
            result = model.generate(prompts, sampling_params)
            # Repeat requests to exercise slot reuse and prefix caching.
            reused = model.generate(reuse_prompts, sampling_params)

        assert_valid_generation_outputs(result, len(prompts), vocab_size)
        assert_valid_generation_outputs(reused, len(reuse_prompts), vocab_size)
