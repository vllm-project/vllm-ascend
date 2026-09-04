#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""DeepSeek-V4 DSA-PCP accuracy and MTP acceptance test for Model Runner V2.

Run `pytest tests/e2e/pull_request/four_card/context_parallel/test_deepseek_v4.py`.
"""

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.outputs import RequestOutput
from vllm.v1.metrics.reader import Counter, Metric, Vector

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.one_card.model_runner_v2.utils import calculate_acceptance_per_pos

MODEL = "gdydems/DeepSeek-V4-Flash-w4a8-mtp"
MAX_NUM_SEQS = 4
NUM_SPECULATIVE_TOKENS = 3

FULL_DECODE_GRAPH_CONFIG = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [MAX_NUM_SEQS, 2 * MAX_NUM_SEQS],
}

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "What is the meaning of life?",
]
MAX_TOKENS = 1024
EXPECTED_OUTPUT_PREFIXES = {
    "Hello, my name is": "Hello, my name is {name} and I",
    "What is the meaning of life?": 'What is the meaning of life?",\n    "What is',
}

MIN_ACCEPTANCE_RATES = [0.85, 0.65, 0.35]
ACCEPTANCE_RATE_TOLERANCE = 0.03


def _assert_output_accuracy(outputs: list[RequestOutput]) -> None:
    assert len(outputs) == len(PROMPTS), f"Expected {len(PROMPTS)} outputs, got {len(outputs)}"
    outputs_by_prompt = {output.prompt: output for output in outputs}

    for prompt, expected_prefix in EXPECTED_OUTPUT_PREFIXES.items():
        assert prompt in outputs_by_prompt, f"Missing output for prompt {prompt!r}"
        request_output = outputs_by_prompt[prompt]
        assert len(request_output.outputs) == 1, (
            f"Expected one completion for prompt {prompt!r}, got {len(request_output.outputs)}"
        )

        output_text = prompt + request_output.outputs[0].text
        assert output_text.startswith(expected_prefix), (
            f"Unexpected output prefix for prompt {prompt!r}: "
            f"got {output_text[: len(expected_prefix)]!r}, expected {expected_prefix!r}"
        )


def _assert_acceptance_rates(metrics: list[Metric]) -> None:
    acceptance_rates = calculate_acceptance_per_pos(
        metrics,
        NUM_SPECULATIVE_TOKENS,
        Counter,
        Vector,
    )
    assert len(acceptance_rates) == len(MIN_ACCEPTANCE_RATES), (
        f"Expected {len(MIN_ACCEPTANCE_RATES)} acceptance rates, got {len(acceptance_rates)}"
    )
    for position, (actual, minimum) in enumerate(zip(acceptance_rates, MIN_ACCEPTANCE_RATES, strict=True)):
        assert actual >= minimum or minimum - actual < ACCEPTANCE_RATE_TOLERANCE, (
            f"Acceptance rate at draft position {position} is {actual}, below minimum {minimum}"
        )


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dsa_pcp,mtp",
    parallel="TP,EP,PCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_BATCH_INVARIANT": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "2560",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v4_dsa_pcp_mtp_full_decode_only() -> None:
    """Verify output accuracy and MTP acceptance for DSA-PCP graph execution."""
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        seed=0,
    )

    with VllmRunner(
        MODEL,
        max_model_len=8192,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=4096,
        dtype="auto",
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.9,
        quantization="ascend",
        tokenizer_mode="deepseek_v4",
        block_size=128,
        enforce_eager=False,
        compilation_config=FULL_DECODE_GRAPH_CONFIG,
        disable_log_stats=False,
        speculative_config={
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            "method": "mtp",
        },
        additional_config={
            "enable_dsa_cp": False,
            "enable_prefill_mc2": True,
        },
    ) as runner:
        outputs = runner.model.generate(PROMPTS, sampling_params)
        metrics = runner.model.get_metrics()

    _assert_output_accuracy(outputs)
    _assert_acceptance_rates(metrics)
