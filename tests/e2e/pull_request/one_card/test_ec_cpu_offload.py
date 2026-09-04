# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Single-NPU ECCPUConnector correctness test."""

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.assets.image import ImageAsset
from vllm.config import ECTransferConfig

from tests.e2e.conftest import (
    VllmRunner,
    qwen_prompt,
    wait_until_npu_memory_free,
)

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
EC_CPU_BYTES = 500 << 20
MMAP_GLOB = "vllm_ec_*.mmap"
MMAP_CLEANUP_TIMEOUT = 10.0


def _ec_mmap_paths() -> set[Path]:
    return set(Path("/dev/shm").glob(MMAP_GLOB))


def _wait_for_ec_ready(runner: VllmRunner) -> None:
    """Advance one scheduler step so the completed save becomes ready."""
    runner.generate_greedy(["hi"], max_tokens=1, use_tqdm=False)


def _reset_device_encoder_cache(runner: VllmRunner) -> None:
    runner.model.llm_engine.reset_encoder_cache()


def _assert_ec_round_trip(runner: VllmRunner, image, prompt: str) -> None:
    sampling_params = SamplingParams(max_tokens=8, temperature=0)

    _reset_device_encoder_cache(runner)
    cold_output = runner.generate(
        prompts=[prompt],
        images=[image],
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    _wait_for_ec_ready(runner)
    _reset_device_encoder_cache(runner)
    loaded_output = runner.generate(
        prompts=[prompt],
        images=[image],
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    assert loaded_output == cold_output


def _assert_mmaps_removed(paths: set[Path]) -> None:
    deadline = time.monotonic() + MMAP_CLEANUP_TIMEOUT
    while remaining := {path for path in paths if path.exists()}:
        if time.monotonic() >= deadline:
            pytest.fail(f"EC mmap files were not removed at shutdown: {remaining}")
        time.sleep(0.1)


def _run_ec_cpu_offloading() -> None:
    image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
    prompt = qwen_prompt(["Describe this image briefly."])[0]
    ec_transfer_config = ECTransferConfig(
        ec_connector="ECCPUConnector",
        ec_role="ec_both",
        ec_connector_extra_config={"ec_cpu_bytes": EC_CPU_BYTES},
    )

    existing_mmaps = _ec_mmap_paths()
    created_mmaps: set[Path]
    with VllmRunner(
        MODEL,
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=0.3,
        enable_prefix_caching=False,
        ec_transfer_config=ec_transfer_config,
    ) as runner:
        _assert_ec_round_trip(runner, image, prompt)
        created_mmaps = _ec_mmap_paths() - existing_mmaps
        assert created_mmaps, "ECCPUConnector did not create an EC mmap file"

    _assert_mmaps_removed(created_mmaps)


@pytest.mark.e2e_model("Qwen/Qwen2.5-VL-7B-Instruct")
@pytest.mark.e2e_coverage(
    arch="multimodal",
    feature="cpu_offloading",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="eager",
)
@pytest.mark.parametrize("use_v2_model_runner", [False, True], ids=["v1", "v2"])
@wait_until_npu_memory_free()
def test_ec_cpu_offloading(use_v2_model_runner: bool) -> None:
    env = {
        "VLLM_USE_V2_MODEL_RUNNER": "1" if use_v2_model_runner else "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    with patch.dict(os.environ, env):
        _run_ec_cpu_offloading()
