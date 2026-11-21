# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
from unittest.mock import patch

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = os.environ.get("FLASHCOMM1_TEST_MODEL", "Qwen/Qwen3.5-35B-A3B")


@pytest.mark.parametrize("enabled", [False, True])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
@wait_until_npu_memory_free(0.8)
def test_flashcomm1_config(enabled):
    # Exercise legacy fallback and explicit override in real spawned Workers.
    with (
        patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "0" if enabled else "1"}),
        VllmRunner(
            MODEL,
            tensor_parallel_size=2,
            data_parallel_size=1,
            enable_expert_parallel=True,
            additional_config={"enable_flashcomm1": enabled},
            enforce_eager=True,
            max_model_len=1024,
            max_num_batched_tokens=1024,
            max_num_seqs=2,
            gpu_memory_utilization=0.85,
            distributed_executor_backend="mp",
        ) as runner,
    ):
        assert bool(runner.model.llm_engine.vllm_config.parallel_config.use_sequence_parallel_moe) is enabled
        for prompts in (["The capital of France is", "1 + 1 ="], ["The capital of France is"]):
            outputs = runner.generate_greedy_logprobs(prompts, max_tokens=8, num_logprobs=5)
            assert len(outputs) == len(prompts)
            for tokens, text, logprobs in outputs:
                assert tokens and text and logprobs
                assert len(tokens) == len(logprobs)
                assert all(math.isfinite(value.logprob) for row in logprobs for value in row.values())
