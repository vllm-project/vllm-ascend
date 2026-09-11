# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import os
from typing import Any
from unittest.mock import patch

import pytest
import regex as re
from vllm import SamplingParams

from tests.e2e.conftest import DPVllmRunner, wait_until_npu_memory_free
from vllm_ascend.distributed.eplb.state import ASYNC_EPLB_CYCLE_COMMITTED_LOG

MODEL = os.environ.get("QWEN3_MRV2_EPLB_MODEL_PATH", "vllm-ascend/Qwen3-30B-A3B-W8A8")
PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water freezes at a temperature of",
    "The author of Pride and Prejudice is",
    "The chemical symbol for gold is",
    "The square root of 144 is",
    "The opposite of hot is",
    "The first month of the year is",
]
EXPECTED_ANSWER_PREFIXES = [
    ("Paris",),
    ("Jupiter",),
    ("32 degrees Fahrenheit", "32°F", "0 degrees Celsius", "0°C", "0 °C"),
    ("Jane Austen",),
    ("Au",),
    ("12",),
    ("cold",),
    ("January",),
]
ASYNC_EPLB_CYCLE_CHUNK_TOKENS = 32
ASYNC_EPLB_CYCLE_MAX_CHUNKS = 12


def _assert_expected_answers(outputs, name: str) -> None:
    assert len(outputs) == len(PROMPTS) == len(EXPECTED_ANSWER_PREFIXES)
    for prompt_idx, (prompt, (_, output_text), expected_prefixes) in enumerate(
        zip(PROMPTS, outputs, EXPECTED_ANSWER_PREFIXES)
    ):
        assert output_text.startswith(prompt), (
            f"{name} returned text that does not start with its prompt for "
            f"prompt {prompt_idx}: expected prefix {prompt!r}, got {output_text!r}"
        )
        completion = output_text[len(prompt) :].lstrip()
        matching_prefix = next(
            (prefix for prefix in expected_prefixes if completion.startswith(prefix)),
            None,
        )
        assert matching_prefix is not None, (
            f"{name} produced an incorrect answer for prompt {prompt_idx}: "
            f"expected the completion to start with one of {expected_prefixes!r}, "
            f"got {completion!r}"
        )
        suffix = completion[len(matching_prefix) :]
        assert not suffix or not (suffix[0].isalnum() or suffix[0] == "_"), (
            f"{name} only matched an answer as part of a longer word for "
            f"prompt {prompt_idx}: matched {matching_prefix!r}, got {completion!r}"
        )


def _run_dp2_tp2(capfd: pytest.CaptureFixture[str]):
    runner_kwargs: dict[str, Any] = {
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "enable_expert_parallel": True,
        "max_model_len": 2048,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 2048,
        "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"},
        "quantization": "ascend",
        "distributed_executor_backend": "mp",
        "async_scheduling": True,
        "gpu_memory_utilization": 0.7,
        "block_size": 128,
        "enable_prefix_caching": False,
        "dp_start_timeout": 1800,
        "dp_request_timeout": 1800,
        "enable_eplb": True,
        "eplb_config": {
            "window_size": 2,
            "step_interval": 2,
            "num_redundant_experts": 4,
            "log_balancedness": False,
            "use_async": True,
        },
        "additional_config": {
            "eplb_config": {
                "load_collection_phase": "prefill",
            },
        },
    }

    captured_output = ""
    with DPVllmRunner(MODEL, **runner_kwargs) as runner:
        outputs = runner.generate_greedy(PROMPTS, max_tokens=16)
        captured = capfd.readouterr()
        captured_output += captured.out + captured.err
        for _ in range(ASYNC_EPLB_CYCLE_MAX_CHUNKS):
            if ASYNC_EPLB_CYCLE_COMMITTED_LOG in captured_output:
                break
            # The upstream async worker commits one of this model's 48 MoE
            # layers per forward step. Decode in bounded chunks so weight
            # transfers have time to finish, stopping as soon as the cycle is
            # observable.
            runner.generate(
                [PROMPTS[0]],
                SamplingParams(
                    temperature=0.0,
                    max_tokens=ASYNC_EPLB_CYCLE_CHUNK_TOKENS,
                    ignore_eos=True,
                ),
            )
            captured = capfd.readouterr()
            captured_output += captured.out + captured.err
    return outputs, captured_output


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="eplb",
    parallel="DP,TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="full_and_piecewise",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_LOGGING_LEVEL": "INFO",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "1024",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.7, max_wait_seconds=180)
def test_qwen3_moe_w8a8_dp2_tp2_async_eplb_accuracy(
    capfd: pytest.CaptureFixture[str],
):
    eplb_outputs, output = _run_dp2_tp2(capfd)
    _assert_expected_answers(eplb_outputs, "MRV2 asynchronous EPLB")
    captured = capfd.readouterr()
    output += captured.out + captured.err
    committed_cycle = re.search(
        rf"{re.escape(ASYNC_EPLB_CYCLE_COMMITTED_LOG)}: model=.+",
        output,
    )
    eplb_log_lines = [line for line in output.splitlines() if "eplb" in line.lower()]
    assert committed_cycle is not None, "No asynchronous EPLB cycle completed.\n" + "\n".join(eplb_log_lines)
