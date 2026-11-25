# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Qwen3.6 BF16 MTP acceptance and RecoverSSM functionality on two NPUs.

Use the same 40 MT-Bench prompts, chat formatting and output limit as
vllm-ascend#13960, with a 10% acceptance-length tolerance. The fixed goldens
use the supplied MRV2 measurements with greedy decoding (temperature 0).
Check generation and draft activity independently in each RecoverSSM mode.

Run with:
    pytest -sv tests/e2e/pull_request/two_card/model_runner_v2/test_qwen3_6_mtp.py
"""

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Metric

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.utils import SPEC_DECODE_PROMPTS, assert_valid_generation_outputs

QWEN36_MOE_MODEL = "Qwen/Qwen3.6-35B-A3B"
QWEN36_DENSE_MODEL = "Qwen/Qwen3.6-27B"
QWEN36_MOE_EXPECTED_ACCEPTANCE_LENGTH = 3.141271769947347
QWEN36_DENSE_EXPECTED_ACCEPTANCE_LENGTH = 3.1607901975493875
ACCEPTANCE_LENGTH_RTOL = 0.10
NUM_SPECULATIVE_TOKENS = 3
MAX_TOKENS = 1024
MAX_MODEL_LEN = 4096
SEED = 42


def _read_mtp_counters(metrics: list[Metric]) -> tuple[int, int]:
    num_drafts = 0
    num_accepted_tokens = 0
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
    return num_drafts, num_accepted_tokens


def _run_mtp_case(model_name: str, is_moe: bool, use_replayssm: bool) -> tuple[int, int]:
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
        mamba_cache_mode="none",
        mamba_ssm_cache_dtype="float32",
        use_replayssm=use_replayssm,
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
        model = vllm_model.model
        tokenizer = model.get_tokenizer()
        prompt_ids = [
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                add_special_tokens=False,
            )
            for prompt in SPEC_DECODE_PROMPTS
        ]
        vocab_size = model.llm_engine.model_config.get_vocab_size()
        # Use deltas so other engines in this pytest process cannot
        # contaminate the comparison through old counters.
        before = _read_mtp_counters(model.get_metrics())
        outputs = model.generate(
            vllm_model.get_inputs(prompt_ids),
            sampling_params=SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS),
        )
        after = _read_mtp_counters(model.get_metrics())

    assert_valid_generation_outputs(outputs, len(prompt_ids), vocab_size)
    return after[0] - before[0], after[1] - before[1]


@pytest.mark.parametrize(
    "model_name, is_moe, expected_acceptance_length",
    [
        pytest.param(
            QWEN36_MOE_MODEL,
            True,
            QWEN36_MOE_EXPECTED_ACCEPTANCE_LENGTH,
            id="qwen3_6_35b_a3b",
            marks=[
                pytest.mark.e2e_model(QWEN36_MOE_MODEL),
                pytest.mark.e2e_coverage(
                    arch="moe",
                    feature="mtp,aclgraph",
                    parallel="TP,EP",
                    deploy="pd_mix",
                    hardware="A3",
                    quantization="BF16",
                    graph_mode="full_decode_only",
                ),
            ],
        ),
        pytest.param(
            QWEN36_DENSE_MODEL,
            False,
            QWEN36_DENSE_EXPECTED_ACCEPTANCE_LENGTH,
            id="qwen3_6_27b",
            marks=[
                pytest.mark.e2e_model(QWEN36_DENSE_MODEL),
                pytest.mark.e2e_coverage(
                    arch="dense",
                    feature="mtp,aclgraph",
                    parallel="TP",
                    deploy="pd_mix",
                    hardware="A3",
                    quantization="BF16",
                    graph_mode="full_decode_only",
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize("use_replayssm", [False, True], ids=["replayssm_off", "replayssm_on"])
@patch.dict(
    os.environ,
    {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "1024",
        "LCCL_DETERMINISTIC": "1",
        "HCCL_DETERMINISTIC": "true",
        "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
        "CLOSE_MATMUL_K_SHIFT": "1",
    },
)
@wait_until_npu_memory_free()
def test_qwen3_6_mtp_acceptance_tp2(
    model_name: str,
    is_moe: bool,
    expected_acceptance_length: float,
    use_replayssm: bool,
) -> None:
    num_drafts, num_accepted_tokens = _run_mtp_case(model_name, is_moe, use_replayssm)
    assert num_drafts > 0, f"{model_name}, use_replayssm={use_replayssm}: no draft activity"
    assert num_accepted_tokens > 0, f"{model_name}, use_replayssm={use_replayssm}: no accepted drafts"

    if not use_replayssm:
        acceptance_length = 1 + num_accepted_tokens / num_drafts
        relative_error = abs(acceptance_length - expected_acceptance_length) / expected_acceptance_length
        assert relative_error <= ACCEPTANCE_LENGTH_RTOL, (
            f"{model_name}, MRV2: acc_len does not match the fixed golden; "
            f"expected={expected_acceptance_length:.6f}, actual={acceptance_length:.6f}, "
            f"relative error={relative_error:.2%}, tolerance={ACCEPTANCE_LENGTH_RTOL:.0%}\n"
            f"num_drafts={num_drafts}\n"
            f"num_accepted_tokens={num_accepted_tokens}"
        )
