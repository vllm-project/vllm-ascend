# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in real-checkpoint parity and changing-budget verification on MRV2."""

import math
import os
from copy import deepcopy
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


def _install_budget_probe(worker, policy):
    import numpy as np

    runner = worker.model_runner
    manager = runner.adaptive_verification
    assert manager is not None
    if not hasattr(manager, "_test_v41_original_budget"):
        manager._test_v41_original_budget = manager.get_num_tokens
        manager._test_v41_cost_tables = manager.cost_tables
    original = manager._test_v41_original_budget
    manager.cost_tables = manager._test_v41_cost_tables
    observations: list[tuple[int, int]] = []
    runner._test_v41_budgets = observations
    runner._test_v41_graph_modes = []
    if not hasattr(runner, "_test_v41_original_prepare"):
        runner._test_v41_original_prepare = runner.prepare_inputs

        def prepare_inputs(scheduled, batch, descriptor, *args, **kwargs):
            result = runner._test_v41_original_prepare(scheduled, batch, descriptor, *args, **kwargs)
            if not batch.has_prefill:
                runner._test_v41_graph_modes.append(descriptor.cg_mode.name)
            return result

        runner.prepare_inputs = prepare_inputs

    def get_num_tokens(scheduled, drafts):
        scheduled_drafts = sum(len(drafts.get(req, ())) for req in scheduled)
        base_tokens = sum(scheduled.values()) - scheduled_drafts
        if policy != "natural":
            draft_cost, verify_cost = manager._test_v41_cost_tables
            # A decreasing curve makes full verification win even if the
            # trained confidence rounds to zero at every draft position.
            verify_cost = np.linspace(2.0, 1.0, verify_cost.size, dtype=verify_cost.dtype)
            if policy == "alternating" and len(observations) % 2:
                verify_cost.fill(1e12)
                verify_cost[base_tokens] = 1.0
            manager.cost_tables = np.zeros_like(draft_cost), verify_cost
        total = original(scheduled, drafts)
        observations.append((scheduled_drafts, total - base_tokens))
        return total

    manager.get_num_tokens = get_num_tokens


def _read_budget_probe(worker):
    runner = worker.model_runner
    return runner._test_v41_budgets, runner._test_v41_graph_modes


def _assert_output_parity(name, reference, outputs):
    assert len(outputs) == len(reference)
    for prompt, (baseline, actual) in enumerate(zip(reference, outputs)):
        base, result = baseline.outputs[0], actual.outputs[0]
        assert result.token_ids == base.token_ids, (
            f"{name}, prompt={prompt}: plain={base.text!r}, actual={result.text!r}, "
            f"plain_ids={base.token_ids}, actual_ids={result.token_ids}"
        )
        assert result.logprobs is not None and len(result.logprobs) == len(result.token_ids)
        for token, topk in zip(result.token_ids, result.logprobs):
            assert math.isfinite(topk[token].logprob), f"{name}: non-finite sampled-token logprob"


@pytest.mark.skipif(
    not os.getenv("DSV41_TARGET_MODEL"), reason="Set DSV41_TARGET_MODEL to a real V4.1 DSpark checkpoint"
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        # Test-only RPC callables install cost probes; serving needs no such flag.
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    },
)
def test_deepseek_v41_dspark_adaptive_output_and_graph_replay():
    model = os.environ["DSV41_TARGET_MODEL"]
    prompts = [
        "Explain how speculative decoding works in detail.",
        "Write a Python function that computes the Fibonacci sequence.",
        "The capital of France is",
    ]
    sampling = SamplingParams(temperature=0, seed=0, max_tokens=64, ignore_eos=True, logprobs=1)
    kwargs = dict(
        tensor_parallel_size=int(os.getenv("DSV41_TP_SIZE", "4")),
        enable_expert_parallel=True,
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.9,
        generation_config="vllm",
        enable_prefix_caching=False,
        async_scheduling=True,
        # Varlen graphs above max_num_seqs * (K+1) are not captured. Include
        # 24 explicitly so the full 3-request/18-token budget really replays.
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 6, 8, 12, 16, 24],
        },
        additional_config={"ascend_compilation_config": {"enable_npugraph_ex": False}},
    )
    if quantization := os.getenv("DSV41_QUANTIZATION"):
        kwargs["quantization"] = quantization
    spec = dict(
        method="dspark",
        num_speculative_tokens=5,
        enforce_eager=True,
        enable_adaptive_verification=False,
    )
    with VllmRunner(model, **deepcopy(kwargs)) as runner:
        reference = runner.model.generate(prompts, sampling)
    with VllmRunner(model, speculative_config=spec, **deepcopy(kwargs)) as runner:
        fixed = runner.model.generate(prompts, sampling)
    _assert_output_parity("fixed", reference, fixed)
    spec["enable_adaptive_verification"] = True
    with VllmRunner(model, speculative_config=spec, **deepcopy(kwargs)) as runner:
        for policy in ("natural", "full", "alternating"):
            runner.model.collective_rpc(_install_budget_probe, args=(policy,))
            outputs = runner.model.generate(prompts, sampling)
            _assert_output_parity(policy, reference, outputs)
            probes = runner.model.collective_rpc(_read_budget_probe)
            assert probes and all(probes)
            assert all(probe == probes[0] for probe in probes)
            raw_budgets, graph_modes = probes[0]
            assert graph_modes and all(mode == "FULL" for mode in graph_modes), graph_modes
            budgets = [(scheduled, admitted) for scheduled, admitted in raw_budgets if scheduled > 0]
            assert budgets, f"{policy}: did not exercise speculative verification"
            if policy == "full":
                assert all(admitted == scheduled for scheduled, admitted in budgets)
            if policy == "alternating":
                assert any(admitted > 0 for _, admitted in budgets)
                assert any(admitted == 0 for _, admitted in budgets)
