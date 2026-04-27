# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""End-to-end tests for the Ascend ``SimpleCPUOffloadConnector``.

The simple CPU offloading scheduler/worker pair is reused from upstream
vLLM; here we only exercise the NPU-native worker path
(``aclrtMemcpyBatchAsync`` + ``torch.npu`` streams) to confirm that
KV blocks are stored to and reloaded from CPU correctly on Ascend.
"""

import os
import time

import pytest
from vllm import SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig

from tests.e2e.conftest import VllmRunner

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _build_kv_transfer_config(
    cpu_bytes_to_use: int,
    lazy_offload: bool = False,
) -> KVTransferConfig:
    return KVTransferConfig(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": cpu_bytes_to_use,
            "lazy_offload": lazy_offload,
        },
    )


def test_simple_cpu_offload_accuracy() -> None:
    """Reset GPU prefix cache after a cold run; verify the CPU-loaded KV
    cache reproduces the cold-run output deterministically."""
    sampling_params = SamplingParams(max_tokens=1, temperature=0)

    # Long enough prompt to occupy multiple full KV blocks.
    prompt = "hi " * 500 + "Let's count to ten. One, two, three, "

    with VllmRunner(
        "Qwen/Qwen3-0.6B",
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=True,
        kv_transfer_config=_build_kv_transfer_config(1 << 30),  # 1 GiB
        enforce_eager=True,
    ) as runner:
        llm = runner.model

        # Cold run — populates GPU cache and triggers CPU offload.
        cold_output = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
        expected = cold_output.outputs[0].text

        success = 0
        attempts = 5
        for _ in range(attempts):
            # Let the engine core drain pending store transfers.
            time.sleep(2)
            # Reset GPU prefix cache so the next run must reload from CPU.
            if not llm.reset_prefix_cache():
                continue
            output = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
            if output.outputs[0].text == expected:
                success += 1

        assert success >= int(0.5 * attempts), (
            f"CPU-load accuracy too low: {success}/{attempts} matched "
            f"baseline output {expected!r}"
        )


@pytest.mark.parametrize("lazy", [False, True])
def test_simple_cpu_offload_no_crash_on_repeat(lazy: bool) -> None:
    """Smoke test: many short generations exercise both eager and lazy
    offload paths without errors and yield non-empty outputs."""
    sampling_params = SamplingParams(max_tokens=4, temperature=0)
    prompt_token_ids = [0] * 257

    with VllmRunner(
        "Qwen/Qwen3-0.6B",
        max_model_len=2048,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=True,
        kv_transfer_config=_build_kv_transfer_config(
            cpu_bytes_to_use=512 * (1 << 20),  # 512 MiB
            lazy_offload=lazy,
        ),
        enforce_eager=True,
    ) as runner:
        llm = runner.model
        for i in range(8):
            prompt_token_ids[0] = i
            prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)]
            outs = llm.generate(prompts, sampling_params, use_tqdm=False)
            assert outs and len(outs[0].outputs[0].token_ids) > 0
