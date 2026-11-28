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
# This file is a part of the vllm-ascend project.
#
import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.one_card.model_runner_v2.utils import calculate_acceptance_per_pos

os.environ["HCCL_BUFFSIZE"] = "2048"
DSPARK_MAIN_MODEL = ["UploadWeight/DeepSeek-V4-Flash-DSpark-w4a8-test"]

MODEL = "gdydems/DeepSeek-V4-Flash-w4a8-mtp"


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free()
@pytest.mark.parametrize("num_speculative_tokens", [1, 3], ids=["one_step", "three_steps"])
def test_deepseek_v4_mtp_full_decode_only(num_speculative_tokens: int):
    """Verify DeepSeek V4 draft prefill capture and MTP acceptance with ModelRunner V2."""
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "What is the meaning of life?",
    ]
    one_step = num_speculative_tokens == 1
    max_tokens = 32 if one_step else 1024
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0, seed=0, ignore_eos=one_step)
    compilation_config: dict = {"cudagraph_mode": "FULL_DECODE_ONLY"}
    if one_step:
        # Capture the uniform two-query target batch and the draft prefill
        # for the five prompts above without capturing every default size.
        compilation_config["cudagraph_capture_sizes"] = [2, 4, 8, 16]

    with VllmRunner(
        MODEL,
        max_model_len=8192,
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        dtype="auto",
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.9,
        quantization="ascend",
        tokenizer_mode="deepseek_v4",
        block_size=128,
        enforce_eager=False,
        compilation_config=compilation_config,
        disable_log_stats=False,
        async_scheduling=True,
        speculative_config={
            "num_speculative_tokens": num_speculative_tokens,
            "method": "mtp",
        },
        attention_config={"indexer_kv_dtype": "int8"},
    ) as runner:
        outputs = runner.model.generate(prompts, sampling_params)
        metrics = runner.model.get_metrics()

    if one_step:
        assert len(outputs) == len(prompts)
        assert all(len(output.outputs[0].token_ids) == max_tokens for output in outputs)
        # A successful generate alone could silently bypass speculation.
        assert any(
            metric.name == "vllm:spec_decode_num_drafts" and metric.value > 0
            for metric in metrics
            if isinstance(metric, Counter)
        )
        return

    acceptance_per_pos = calculate_acceptance_per_pos(
        metrics,
        num_speculative_tokens,
        Counter,
        Vector,
    )
    golden = [0.85, 0.65, 0.35]
    match = all((a >= b) or (b - a < 0.03) for a, b in zip(acceptance_per_pos, golden))
    assert match, f"acceptance_per_pos {acceptance_per_pos} below golden {golden}"


@pytest.mark.parametrize("model", DSPARK_MAIN_MODEL)
@pytest.mark.parametrize("max_tokens", [1024])
@pytest.mark.parametrize("enforce_eager", [False])
@pytest.mark.parametrize(
    ("compilation_config", "enable_adaptive_verification"),
    [
        pytest.param(
            {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [6, 12]},
            False,
            id="full_decode_only",
        ),
        pytest.param({}, False, id="default_full_and_piecewise"),
        pytest.param(
            {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [6, 12]},
            True,
            id="full_decode_only-adaptive",
        ),
    ],
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dspark_spec_decoding(
    model: str,
    max_tokens: int,
    enforce_eager: bool,
    enable_adaptive_verification: bool,
    compilation_config: dict,
) -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    num_speculative_tokens = 5
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
        model,
        max_model_len=4096,
        max_num_seqs=len(prompts),
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        enforce_eager=enforce_eager,
        disable_log_stats=False,
        async_scheduling=True,
        attention_config={"indexer_kv_dtype": "int8"},
        speculative_config={
            "method": "dspark",
            "num_speculative_tokens": num_speculative_tokens,
            **({"enable_adaptive_verification": True} if enable_adaptive_verification else {}),
        },
        compilation_config=compilation_config,
    ) as runner:
        runner.model.generate(prompts, sampling_params)
        metrics = runner.model.get_metrics()

    if enable_adaptive_verification:
        return

    acceptance_per_pos = calculate_acceptance_per_pos(
        metrics,
        num_speculative_tokens,
        Counter,
        Vector,
    )
    golden = [0.73, 0.64, 0.55, 0.49, 0.42]
    match = all((a >= b) or (b - a < 0.03) for a, b in zip(acceptance_per_pos, golden))
    assert match, f"acceptance_per_pos {acceptance_per_pos} below golden {golden}"
