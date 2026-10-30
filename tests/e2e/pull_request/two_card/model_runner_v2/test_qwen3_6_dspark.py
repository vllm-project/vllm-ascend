#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Qwen3.6-27B BF16 DSPARK acceptance on MRV2 over two NPUs.

Run "pytest tests/e2e/pull_request/two_card/model_runner_v2/test_qwen3_6_dspark.py".
"""

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.config import CompilationConfig

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.utils import SPEC_DECODE_PROMPTS, _check_decode_token, _run_speculative_decoding

QWEN36_DENSE_MODEL = "Qwen/Qwen3.6-27B"
QWEN36_DSPARK_DRAFT_MODEL = "rainney/Qwen3.6-27B-DSpark"
MODELS = [QWEN36_DENSE_MODEL]
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MAX_MODEL_LEN = 72320
MAX_NUM_BATCHED_TOKENS = 16384
GPU_MEMORY_UTILIZATION = 0.95


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize(
    ("expected_acceptance_length", "num_speculative_tokens", "additional_config"),
    [
        pytest.param(
            3.47,
            7,
            {"ascend_compilation_config": {"enable_npugraph_ex": False}},
            id="dspark-qwen36-27b",
        ),
    ],
)
@patch.dict(
    os.environ,
    {
        "OMP_NUM_THREADS": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_BUFFSIZE": "1024",
        "TASK_QUEUE_ENABLE": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "LCCL_DETERMINISTIC": "1",
        "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "HCCL_DETERMINISTIC": "true",
        "CLOSE_MATMUL_K_SHIFT": "1",
        "VLLM_ASCEND_ENABLE_FUSED_MC2": "1",
    },
)
def test_qwen36_27b_dspark_acceptance_tp2(
    model_name,
    expected_acceptance_length,
    num_speculative_tokens,
    additional_config,
):
    _run_speculative_decoding(
        model_name=model_name,
        speculative_config={
            "method": "dspark",
            "model": QWEN36_DSPARK_DRAFT_MODEL,
            "num_speculative_tokens": num_speculative_tokens,
        },
        example_prompts=SPEC_DECODE_PROMPTS,
        expected_acceptance_length=expected_acceptance_length,
        runner_kwargs={
            "tensor_parallel_size": 2,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "generation_config": "vllm",
            "compilation_config": CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY"),
            "additional_config": additional_config,
            "enable_prefix_caching": False,
            "async_scheduling": True,
        },
        is_moe=False,
        max_tokens=8192,
    )


def _install_verification_cost_probe(worker, alternate):
    """Control only the hardware cost input; keep vLLM's allocator and sampler."""
    import numpy as np

    runner = worker.model_runner
    manager = runner.adaptive_verification
    assert manager is not None
    if not hasattr(runner, "_test_original_get_num_tokens"):
        runner._test_original_get_num_tokens = manager.get_num_tokens
    original = runner._test_original_get_num_tokens
    draft_cost, verify_cost = manager.cost_tables
    draft_cost = np.zeros_like(draft_cost)
    verify_cost = np.ones_like(verify_cost)
    observations = []
    runner._test_verify_budgets = observations

    def get_num_tokens(scheduled, drafts):
        scheduled_drafts = sum(len(drafts.get(req_id, ())) for req_id in scheduled)
        base_tokens = sum(scheduled.values()) - scheduled_drafts
        verify_cost.fill(1.0)
        if alternate and len(observations) % 2:
            # A deliberately expensive verify curve forces a zero-draft step
            # immediately after a step that admitted drafts.
            verify_cost.fill(1e12)
            verify_cost[base_tokens] = 1.0
        manager.cost_tables = draft_cost, verify_cost
        total = original(scheduled, drafts)
        observations.append((scheduled_drafts, total - base_tokens))
        return total

    manager.get_num_tokens = get_num_tokens


def _read_verification_cost_probe(worker):
    return worker.model_runner._test_verify_budgets


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_qwen36_dspark_adaptive_gdn_full_to_zero_graph_replay():
    """Compare target distributions while repeatedly changing the verify graph shape."""
    prompts = [
        "Write a Python function that computes the Fibonacci sequence and explain it.",
        "Explain how photosynthesis works, step by step.",
    ]
    sampling = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True, logprobs=20)
    with VllmRunner(
        QWEN36_DENSE_MODEL,
        tensor_parallel_size=2,
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        generation_config="vllm",
        enable_prefix_caching=False,
        disable_log_stats=False,
        compilation_config=CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32]
        ),
        additional_config={"ascend_compilation_config": {"enable_npugraph_ex": False}},
        speculative_config={
            "method": "dspark",
            "model": QWEN36_DSPARK_DRAFT_MODEL,
            "num_speculative_tokens": 7,
            "enable_adaptive_verification": True,
        },
    ) as runner:
        runner.model.collective_rpc(_install_verification_cost_probe, args=(False,))
        reference = runner.model.generate(prompts, sampling)
        full_budgets = runner.model.collective_rpc(_read_verification_cost_probe)
        assert all(
            budgets and all(scheduled == admitted for scheduled, admitted in budgets) for budgets in full_budgets
        )

        runner.model.collective_rpc(_install_verification_cost_probe, args=(True,))
        adaptive = runner.model.generate(prompts, sampling)
        for budgets in runner.model.collective_rpc(_read_verification_cost_probe):
            assert any(admitted > 0 for _, admitted in budgets)
            assert any(scheduled > 0 and admitted == 0 for scheduled, admitted in budgets)

    for prompt_idx, (baseline, trimmed) in enumerate(zip(reference, adaptive, strict=True)):
        base_seq, trimmed_seq = baseline.outputs[0], trimmed.outputs[0]
        assert len(base_seq.token_ids) == len(trimmed_seq.token_ids) == sampling.max_tokens
        for token_idx, (base_id, trimmed_id) in enumerate(zip(base_seq.token_ids, trimmed_seq.token_ids, strict=True)):
            _check_decode_token(base_seq, trimmed_seq, token_idx, prompt_idx, decode_atol=0.0689)
            if base_id != trimmed_id:
                # A permitted near-tie changes subsequent contexts. Compare
                # distributions only while both runs share the same prefix.
                break
