# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Qwen3.6 BF16 MTP acceptance on MRV2 with fixed goldens on two NPUs.

Use the same 40 MT-Bench prompts, chat formatting, output limit and
acceptance-length tolerance as vllm-ascend#13960. The fixed goldens use
the supplied MRV2 measurements with greedy decoding (temperature 0).

Run with:
    pytest -sv tests/e2e/pull_request/two_card/model_runner_v2/test_qwen3_6_mtp.py
"""

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.inputs import TokensPrompt
from vllm.v1.metrics.reader import Counter, Metric, Vector

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.utils import ACCEPTANCE_LENGTH_RTOL, SPEC_DECODE_PROMPTS

QWEN36_MOE_MODEL = "Qwen/Qwen3.6-35B-A3B"
QWEN36_DENSE_MODEL = "Qwen/Qwen3.6-27B"
QWEN36_MOE_EXPECTED_ACCEPTANCE_LENGTH = 3.141271769947347
QWEN36_MOE_EXPECTED_ACCEPTANCE_PER_POS = (0.8663426488456865, 0.7049817739975699, 0.5699473471040907)
QWEN36_DENSE_EXPECTED_ACCEPTANCE_LENGTH = 3.1607901975493875
QWEN36_DENSE_EXPECTED_ACCEPTANCE_PER_POS = (0.8668833875135451, 0.7126781695423856, 0.5812286404934567)
NUM_SPECULATIVE_TOKENS = 3
MAX_TOKENS = 1024
MAX_MODEL_LEN = 4096
SEED = 42
TEMPERATURE = 0.0


def _read_mtp_counters(metrics: list[Metric]) -> tuple[int, int, list[int]]:
    num_drafts = 0
    num_accepted_tokens = 0
    accepted_per_pos = [0] * NUM_SPECULATIVE_TOKENS
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            assert len(metric.values) == NUM_SPECULATIVE_TOKENS, (
                f"Expected {NUM_SPECULATIVE_TOKENS} MTP positions, got {metric.values}"
            )
            for pos, value in enumerate(metric.values):
                accepted_per_pos[pos] += value
    return num_drafts, num_accepted_tokens, accepted_per_pos


@patch.dict(
    os.environ,
    {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "HCCL_BUFFSIZE": "1024",
        "LCCL_DETERMINISTIC": "1",
        "HCCL_DETERMINISTIC": "true",
        "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
        "CLOSE_MATMUL_K_SHIFT": "1",
    },
)
@wait_until_npu_memory_free()
def _run_qwen3_6_mtp(model_name: str, is_moe: bool) -> tuple[float, list[float]]:
    with VllmRunner(
        model_name,
        dtype="bfloat16",
        tensor_parallel_size=2,
        enable_expert_parallel=is_moe,
        distributed_executor_backend="mp",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=len(SPEC_DECODE_PROMPTS),
        gpu_memory_utilization=0.9,
        disable_log_stats=False,
        enable_prefix_caching=False,
        async_scheduling=True,
        seed=SEED,
        generation_config="vllm",
        speculative_config={
            "method": "qwen3_5_mtp",
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        },
        compilation_config=CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[4, 8, 16, 32, 64, 128, 160],
        ),
    ) as vllm_model:
        assert vllm_model.model.llm_engine.vllm_config.use_v2_model_runner, f"Expected MRV2 for {model_name}"
        tokenizer = vllm_model.model.get_tokenizer()
        prompts = [
            TokensPrompt(
                prompt_token_ids=tokenizer.encode(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    ),
                    add_special_tokens=False,
                )
            )
            for prompt in SPEC_DECODE_PROMPTS
        ]
        # Use deltas so other engines in this pytest process cannot
        # contaminate the fixed-golden comparison through old counters.
        before = _read_mtp_counters(vllm_model.model.get_metrics())
        outputs = vllm_model.model.generate(
            prompts,
            sampling_params=SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=SEED),
        )
        after = _read_mtp_counters(vllm_model.model.get_metrics())

    assert len(outputs) == len(prompts), f"Expected {len(prompts)} outputs, got {len(outputs)}"
    for index, output in enumerate(outputs):
        assert output.finished and output.outputs and output.outputs[0].token_ids, (
            f"{model_name}, MRV2, temperature={TEMPERATURE}: prompt {index} returned an unfinished or empty output"
        )
    num_drafts = after[0] - before[0]
    num_accepted_tokens = after[1] - before[1]
    accepted_per_pos = [a - b for a, b in zip(after[2], before[2])]
    assert num_drafts > 0, "MTP did not generate any drafts"
    assert sum(accepted_per_pos) == num_accepted_tokens, "Inconsistent MTP acceptance counters"
    assert all(0 < count <= num_drafts for count in accepted_per_pos), (
        f"Invalid acceptance counters: drafts={num_drafts}, accepted_per_pos={accepted_per_pos}"
    )
    assert accepted_per_pos == sorted(accepted_per_pos, reverse=True), (
        f"Acceptance counters must not increase with draft position: {accepted_per_pos}"
    )

    acceptance_per_pos = [count / num_drafts for count in accepted_per_pos]
    acceptance_length = 1 + num_accepted_tokens / num_drafts
    print(
        f"{model_name}, MRV2, temperature={TEMPERATURE}:\n"
        f"num_drafts={num_drafts}\n"
        f"num_accepted_tokens_per_pos={accepted_per_pos}\n"
        f"acceptance_per_pos={acceptance_per_pos}\n"
        f"acceptance_len={acceptance_length}"
    )
    return acceptance_length, acceptance_per_pos


def _check_qwen3_6_mtp(
    model_name: str,
    is_moe: bool,
    expected_acceptance_length: float,
    expected_acceptance_per_pos: tuple[float, ...],
) -> None:
    actual_length, actual_per_pos = _run_qwen3_6_mtp(model_name, is_moe)

    # Follow #13960's relative-error check against fixed goldens. Check each
    # position as well: the 1D MRoPE input bug already degraded the first draft.
    assert len(expected_acceptance_per_pos) == len(actual_per_pos) == NUM_SPECULATIVE_TOKENS
    for name, expected, actual in zip(
        ["acc_len", *(f"position_{pos + 1}" for pos in range(NUM_SPECULATIVE_TOKENS))],
        [expected_acceptance_length, *expected_acceptance_per_pos],
        [actual_length, *actual_per_pos],
    ):
        relative_error = abs(actual - expected) / expected
        assert relative_error <= ACCEPTANCE_LENGTH_RTOL, (
            f"{model_name}, MRV2, temperature={TEMPERATURE}: {name} does not match the fixed golden; "
            f"expected={expected:.6f}, actual={actual:.6f}, "
            f"relative error={relative_error:.2%}, tolerance={ACCEPTANCE_LENGTH_RTOL:.0%}"
        )


@pytest.mark.e2e_model(QWEN36_MOE_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,aclgraph",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
def test_qwen3_6_35b_a3b_mtp_acceptance_tp2() -> None:
    _check_qwen3_6_mtp(
        QWEN36_MOE_MODEL,
        is_moe=True,
        expected_acceptance_length=QWEN36_MOE_EXPECTED_ACCEPTANCE_LENGTH,
        expected_acceptance_per_pos=QWEN36_MOE_EXPECTED_ACCEPTANCE_PER_POS,
    )


@pytest.mark.e2e_model(QWEN36_DENSE_MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="mtp,aclgraph",
    parallel="TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
def test_qwen3_6_27b_mtp_acceptance_tp2() -> None:
    _check_qwen3_6_mtp(
        QWEN36_DENSE_MODEL,
        is_moe=False,
        expected_acceptance_length=QWEN36_DENSE_EXPECTED_ACCEPTANCE_LENGTH,
        expected_acceptance_per_pos=QWEN36_DENSE_EXPECTED_ACCEPTANCE_PER_POS,
    )
