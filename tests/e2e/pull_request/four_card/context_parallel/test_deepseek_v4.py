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
"""DeepSeek-V4 DSA-PCP tests for Model Runner V2.

Run `pytest tests/e2e/pull_request/four_card/context_parallel/test_deepseek_v4.py`.
"""

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.one_card.model_runner_v2.utils import calculate_acceptance_per_pos
from tests.e2e.pull_request.utils import PROMPTS_SHORT

MTP_MODEL = "gdydems/DeepSeek-V4-Flash-w4a8-mtp"
DSPARK_MODEL = "UploadWeight/DeepSeek-V4-Flash-DSpark-w4a8-test"

MAX_NUM_SEQS = 4
MAX_TOKENS = 1024

MTP_NUM_SPECULATIVE_TOKENS = 3
DSPARK_NUM_SPECULATIVE_TOKENS = 5

COMMON_ENV = {
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_BATCH_INVARIANT": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "HCCL_BUFFSIZE": "2560",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "CLOSE_MATMUL_K_SHIFT": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
}

MTP_EXPECTED_OUTPUT_PREFIXES = {
    "The president of the United States is": ("The president of the United States is the head of the executive branch"),
}
DSPARK_EXPECTED_OUTPUT_PREFIXES = {
    "The president of the United States is": ("The president of the United States is the head of the executive branch"),
}

MTP_MIN_ACCEPTANCE_RATES = [0.85, 0.65, 0.35]
DSPARK_MIN_ACCEPTANCE_RATES = [0.73, 0.64, 0.55, 0.49, 0.42]
ACCEPTANCE_RATE_TOLERANCE = 0.03


def _run_test(
    model: str,
    minimum_rates: list[float] | None,
    speculative_config: dict,
    compilation_config: dict,
    expected_output_prefixes: dict[str, str],
) -> None:
    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, seed=0)
    with VllmRunner(
        model,
        max_model_len=8192,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=1024,
        dtype="auto",
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.9,
        quantization="ascend",
        tokenizer_mode="deepseek_v4",
        block_size=128,
        enforce_eager=False,
        compilation_config=compilation_config,
        disable_log_stats=False,
        speculative_config=speculative_config,
        additional_config={
            "enable_dsa_cp": False,
            "enable_prefill_mc2": True,
        },
    ) as runner:
        outputs = runner.generate(PROMPTS_SHORT, sampling_params)
        for prompt, (_, texts) in zip(PROMPTS_SHORT, outputs, strict=True):
            print(f"Model: {model}, Prompt: {prompt!r}, Outputs: {texts!r}", flush=True)
        metrics = runner.model.get_metrics()

    assert len(outputs) == len(PROMPTS_SHORT), f"Expected {len(PROMPTS_SHORT)} outputs, got {len(outputs)}"
    for prompt, (_, texts) in zip(PROMPTS_SHORT, outputs, strict=True):
        assert len(texts) == 1, f"Expected one completion for prompt {prompt!r}, got {len(texts)}"
        expected_prefix = expected_output_prefixes.get(prompt)
        if expected_prefix is not None:
            assert texts[0].startswith(expected_prefix), (
                f"Unexpected output prefix for prompt {prompt!r}: "
                f"got {texts[0][: len(expected_prefix)]!r}, expected {expected_prefix!r}"
            )

    acceptance_rates = calculate_acceptance_per_pos(
        metrics,
        speculative_config["num_speculative_tokens"],
        Counter,
        Vector,
    )
    print(f"Model: {model}, Acceptance rates per draft position: {acceptance_rates}", flush=True)
    if minimum_rates is None:
        return

    assert len(acceptance_rates) == len(minimum_rates), (
        f"Expected {len(minimum_rates)} acceptance rates, got {len(acceptance_rates)}"
    )
    for position, (actual, minimum) in enumerate(zip(acceptance_rates, minimum_rates, strict=True)):
        assert actual >= minimum or minimum - actual < ACCEPTANCE_RATE_TOLERANCE, (
            f"Acceptance rate at draft position {position} is {actual}, below minimum {minimum}"
        )


@pytest.mark.e2e_model(MTP_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dsa_pcp,mtp",
    parallel="TP,EP,PCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@patch.dict(os.environ, COMMON_ENV)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v4_dsa_pcp_mtp_full_decode_only() -> None:
    """Verify output accuracy and MTP acceptance for DSA-PCP graph execution."""
    _run_test(
        MTP_MODEL,
        minimum_rates=MTP_MIN_ACCEPTANCE_RATES,
        expected_output_prefixes=MTP_EXPECTED_OUTPUT_PREFIXES,
        speculative_config={
            "num_speculative_tokens": MTP_NUM_SPECULATIVE_TOKENS,
            "method": "mtp",
        },
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [MAX_NUM_SEQS, 2 * MAX_NUM_SEQS],
        },
    )


@pytest.mark.e2e_model(DSPARK_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dsa_pcp,dspark",
    parallel="TP,EP,PCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@patch.dict(os.environ, COMMON_ENV)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v4_dsa_pcp_dspark() -> None:
    """Verify output accuracy and DSpark acceptance for DSA-PCP graph execution."""
    _run_test(
        DSPARK_MODEL,
        # Skip acceptance checks until the DSpark acceptance issue is resolved.
        minimum_rates=None,
        expected_output_prefixes=DSPARK_EXPECTED_OUTPUT_PREFIXES,
        speculative_config={
            "num_speculative_tokens": DSPARK_NUM_SPECULATIVE_TOKENS,
            "method": "dspark",
            "enable_adaptive_verification": False,
            "draft_sample_method": "greedy",
        },
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            # The target verifies N+1 tokens while the replicated DSpark draft executes N
            # query tokens per request. Include the 1-, 2-, and 4-request shapes for both.
            "cudagraph_capture_sizes": [5, 6, 10, 12, 20, 24],
        },
    )
