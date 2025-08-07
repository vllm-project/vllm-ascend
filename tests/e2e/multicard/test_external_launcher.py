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
#
"""
Compare the outputs of vLLM with and without aclgraph.

Run `pytest tests/multicard/test_external_launcher.py`.
"""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch_npu

MODELS = ["Qwen/Qwen3-0.6B"]
MOE_MODELS = ["Qwen/Qwen3-30B-A3B"]
DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


@pytest.mark.parametrize("model", MODELS)
def test_external_launcher(model):
    script = Path(
        __file__
    ).parent.parent.parent.parent / "examples" / "offline_external_launcher.py"
    env = os.environ.copy()
    # TODO: Change to 2 when ci machine has 4 cards
    cmd = [
        sys.executable,
        str(script),
        "--model",
        model,
        "--tp-size",
        "1",
        "--node-size",
        "1",
        "--node-rank",
        "0",
        "--proc-per-node",
        "2",
        "--trust-remote-code",
    ]

    print(f"Running subprocess: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
    )
    output = proc.stdout.decode()

    print(output)

    assert "TP RANKS: [0]" in output
    assert "TP RANKS: [1]" in output
    assert "Generated text:" in output
    assert proc.returncode == 0


@pytest.mark.parametrize("model", MOE_MODELS)
def test_moe_external_launcher(model):
    script = Path(
        __file__
    ).parent.parent.parent.parent / "examples" / "offline_external_launcher.py"
    env = os.environ.copy()
    # TODO: Change to 2 when ci machine has 4 cards
    cmd = [
        sys.executable,
        str(script), "--model", model, "--tp-size", "2", "--node-size", "1",
        "--node-rank", "0", "--proc-per-node", "2", "--trust-remote-code",
        "--enable-expert-parallel"
    ]

    print(f"Running subprocess: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
    )
    output = proc.stdout.decode()

    print(output)

    assert "TP RANKS: [0, 1]" in output
    assert "Generated text:" in output
    assert proc.returncode == 0


def test_external_launcher_and_sleepmode():
    script = Path(
        __file__
    ).parent.parent.parent.parent / "examples" / "offline_external_launcher.py"
    env = os.environ.copy()
    # TODO: Change to 2 when ci machine has 4 cards
    cmd = [
        sys.executable,
        str(script),
        "--model",
        "Qwen/Qwen3-8B",
        "--tp-size",
        "1",
        "--node-size",
        "1",
        "--node-rank",
        "0",
        "--proc-per-node",
        "2",
        "--trust-remote-code",
        "--enable-sleep-mode",
        "--temperature",
        "0",
        "--model-weight-gib",
        "16",
    ]

    print(f"Running subprocess: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    output = proc.stdout.decode()

    print(output)

    assert "TP RANKS: [0]" in output
    assert "TP RANKS: [1]" in output
    assert "Generated text:" in output
    assert "Sleep and wake up successfully!!" in output
    assert proc.returncode == 0


@pytest.mark.skipif(
    DEVICE_NAME != "Ascend910B",
    reason="This test is only for Ascend910B devices.",
)
@pytest.mark.parametrize("model", MODELS)
@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "1"})
def test_mm_allreduce(model):
    script = Path(
        __file__
    ).parent.parent.parent.parent / "examples" / "offline_external_launcher.py"
    env = os.environ.copy()
    cmd = [
        sys.executable,
        str(script),
        "--model",
        model,
        "--trust-remote-code",
    ]

    print(f"Running subprocess: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
    )

    output = proc.stdout.decode()
    print(output)

    assert "Generated text:" in output
    assert proc.returncode == 0
