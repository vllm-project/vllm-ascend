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

import math
import os
from copy import deepcopy
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.config import CompilationConfig

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.utils import (
    SPEC_DECODE_PROMPTS,
    _check_decode_token,
    _check_prefill_token,
    _run_speculative_decoding,
)

QWEN36_DENSE_MODEL = os.getenv("PR20361_TARGET_MODEL", "Qwen/Qwen3.6-27B")
QWEN36_DSPARK_DRAFT_MODEL = os.getenv("PR20361_DRAFT_MODEL", "rainney/Qwen3.6-27B-DSpark")
MODELS = [QWEN36_DENSE_MODEL]
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MAX_MODEL_LEN = 72320
MAX_NUM_BATCHED_TOKENS = 16384
GPU_MEMORY_UTILIZATION = 0.95
# Match compare_logprobs(): decode permits twice the prefill tolerance when
# different BF16 execution/query shapes are compared. Keep prefill stricter.
_GDN_PREFILL_LOGPROB_ATOL = 0.0689
_GDN_DECODE_LOGPROB_ATOL = 2 * _GDN_PREFILL_LOGPROB_ATOL


def _check_gdn_verification_logprobs(base_seq, comp_seq, prompt_idx):
    for name, seq in (("baseline", base_seq), ("comparison", comp_seq)):
        assert seq.logprobs is not None, f"{name}: missing logprobs for prompt {prompt_idx}"
        assert len(seq.logprobs) == len(seq.token_ids), f"{name}: incomplete logprobs for prompt {prompt_idx}"
        # Check the entire sequence even if a permitted near-tie later makes
        # the contexts diverge and ends the distribution comparison.
        for token_idx, token_id in enumerate(seq.token_ids):
            assert token_id in seq.logprobs[token_idx], (
                f"{name}: missing sampled-token logprob at prompt {prompt_idx}, token {token_idx}"
            )
            assert math.isfinite(seq.logprobs[token_idx][token_id].logprob), (
                f"{name}: non-finite logprob at prompt {prompt_idx}, token {token_idx}"
            )

    _check_prefill_token(base_seq, comp_seq, prompt_idx, atol=_GDN_PREFILL_LOGPROB_ATOL)
    for token_idx in range(1, len(base_seq.token_ids)):
        _check_decode_token(base_seq, comp_seq, token_idx, prompt_idx, decode_atol=_GDN_DECODE_LOGPROB_ATOL)
        if base_seq.token_ids[token_idx] != comp_seq.token_ids[token_idx]:
            # A permitted near-tie changes subsequent contexts. Compare
            # distributions only while both runs share the same prefix.
            break


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


def _install_natural_verification_budget_probe(worker):
    """Record the unmodified adaptive allocator's decisions."""
    runner = worker.model_runner
    manager = runner.adaptive_verification
    assert manager is not None
    runner._test_original_get_num_tokens = manager.get_num_tokens
    original = runner._test_original_get_num_tokens
    observations: list[tuple[int, int]] = []
    runner._test_verify_budgets = observations

    def get_num_tokens(scheduled, drafts):
        scheduled_drafts = sum(len(drafts.get(req_id, ())) for req_id in scheduled)
        base_tokens = sum(scheduled.values()) - scheduled_drafts
        total = original(scheduled, drafts)
        observations.append((scheduled_drafts, total - base_tokens))
        return total

    manager.get_num_tokens = get_num_tokens


def _install_verification_cost_probe(worker, alternate):
    """Control only the hardware cost input; keep vLLM's allocator and sampler."""
    import numpy as np

    runner = worker.model_runner
    manager = runner.adaptive_verification
    assert manager is not None
    original = runner._test_original_get_num_tokens
    draft_cost, verify_cost = manager.cost_tables
    draft_cost = np.zeros_like(draft_cost)
    verify_cost = np.ones_like(verify_cost)
    observations: list[tuple[int, int]] = []
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


@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        # collective_rpc transports the test-only cost probe callable across
        # the engine boundary; this fallback is needed only for this test.
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    },
)
def test_qwen36_dspark_adaptive_gdn_full_to_zero_graph_replay():
    """Compare target distributions while repeatedly changing the verify graph shape."""
    num_speculative_tokens = int(os.getenv("PR20361_NUM_SPECULATIVE_TOKENS", "7"))
    reproduce_server = os.getenv("PR20361_REPRO_SERVER") == "1"
    prompts = [
        "Explain how speculative decoding works in detail.",
        "Write a Python function that computes the Fibonacci sequence and explain it.",
        "Explain how photosynthesis works, step by step.",
    ]
    if os.getenv("PR20361_SINGLE_REQUEST") == "1":
        prompts = prompts[:1]
    sampling = SamplingParams(
        temperature=0,
        max_tokens=32 if reproduce_server else 64,
        ignore_eos=not reproduce_server,
        logprobs=20,
    )
    runner_kwargs = dict(
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
    )
    if reproduce_server:
        # Match the reported serving run, including async scheduling and the
        # default FULL_DECODE_ONLY graph capture sizes.
        runner_kwargs.update(
            max_model_len=32768,
            gpu_memory_utilization=0.8,
            async_scheduling=True,
            dtype="bfloat16",
            compilation_config=CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY"),
        )
        runner_kwargs.pop("max_num_seqs")

    def assert_exact_greedy_output(name, baseline, comparison, budgets=None):
        for prompt_idx, (base, current) in enumerate(zip(baseline, comparison, strict=True)):
            base_ids = base.outputs[0].token_ids
            current_ids = current.outputs[0].token_ids
            if base_ids != current_ids:
                first = next(
                    (i for i, (x, y) in enumerate(zip(base_ids, current_ids)) if x != y),
                    min(len(base_ids), len(current_ids)),
                )
                pytest.fail(
                    f"{name}: prompt={prompt_idx}, first differing token={first}, "
                    f"plain_id={base_ids[first : first + 1]}, actual_id={current_ids[first : first + 1]}, "
                    f"plain_text={base.outputs[0].text!r}, actual_text={current.outputs[0].text!r}, "
                    f"budgets={budgets}"
                )

    # A non-speculative target is an independent reference: comparing two
    # adaptive runs alone could miss an error common to both recurrent paths.
    with VllmRunner(QWEN36_DENSE_MODEL, **deepcopy(runner_kwargs)) as runner:
        plain_reference = runner.model.generate(prompts, sampling)
    # Fixed speculation exercises the shared DSpark/GDN path independently of
    # the adaptive budget. A high acceptance length can still hide wrong text.
    with VllmRunner(
        QWEN36_DENSE_MODEL,
        **deepcopy(runner_kwargs),
        speculative_config={
            "method": "dspark",
            "model": QWEN36_DSPARK_DRAFT_MODEL,
            "num_speculative_tokens": num_speculative_tokens,
            "enable_adaptive_verification": False,
            "enforce_eager": True,
        },
    ) as runner:
        fixed_reference = runner.model.generate(prompts, sampling)
    if reproduce_server:
        assert_exact_greedy_output("fixed", plain_reference, fixed_reference)
    with VllmRunner(
        QWEN36_DENSE_MODEL,
        **deepcopy(runner_kwargs),
        speculative_config={
            "method": "dspark",
            "model": QWEN36_DSPARK_DRAFT_MODEL,
            "num_speculative_tokens": num_speculative_tokens,
            "enable_adaptive_verification": True,
            "enforce_eager": reproduce_server,
        },
    ) as runner:
        runner.model.collective_rpc(_install_natural_verification_budget_probe)
        natural = runner.model.generate(prompts, sampling)
        natural_budgets = runner.model.collective_rpc(_read_verification_cost_probe)
        runner.model.collective_rpc(_install_verification_cost_probe, args=(False,))
        reference = runner.model.generate(prompts, sampling)
        full_budgets = runner.model.collective_rpc(_read_verification_cost_probe)
        assert all(
            budgets and all(scheduled == admitted for scheduled, admitted in budgets) for budgets in full_budgets
        )

        runner.model.collective_rpc(_install_verification_cost_probe, args=(True,))
        adaptive = runner.model.generate(prompts, sampling)
        for budgets in runner.model.collective_rpc(_read_verification_cost_probe):
            assert any(
                prev_admitted > 0 and scheduled > 0 and admitted == 0
                for (_, prev_admitted), (scheduled, admitted) in zip(budgets, budgets[1:])
            )

    if reproduce_server:
        assert_exact_greedy_output("adaptive full budget", plain_reference, reference, full_budgets)
        assert_exact_greedy_output("adaptive natural budget", plain_reference, natural, natural_budgets)

    for comparison_name, reference_outputs, comparison in (
        ("fixed", plain_reference, fixed_reference),
        ("adaptive natural budget", plain_reference, natural),
        ("adaptive full budget", plain_reference, reference),
        ("adaptive varying budget", plain_reference, adaptive),
        ("adaptive full versus varying budget", reference, adaptive),
    ):
        if reproduce_server:
            assert_exact_greedy_output(comparison_name, reference_outputs, comparison)
        for prompt_idx, (baseline, trimmed) in enumerate(zip(reference_outputs, comparison, strict=True)):
            base_seq, trimmed_seq = baseline.outputs[0], trimmed.outputs[0]
            assert len(base_seq.token_ids) == len(trimmed_seq.token_ids) == sampling.max_tokens
            try:
                _check_gdn_verification_logprobs(base_seq, trimmed_seq, prompt_idx)
            except AssertionError as exc:
                pytest.fail(f"{comparison_name}: {exc}")
