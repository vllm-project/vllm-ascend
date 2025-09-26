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

import pytest
import torch_npu

MOE_MODELS = ["Qwen/Qwen3-30B-A3B"]
MODELS = ["Qwen/Qwen3-8B"]
DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


@pytest.mark.parametrize("model", MOE_MODELS)
def test_external_launcher_eager(model):
    script = script = "/usr/local/python3.11.13/bin/python3.11/__w/vllm-ascend/tests/examples/test_weight_loader.py"
    env = os.environ.copy()
    # TODO: Change to 2 when ci machine has 4 cards
    cmd = [
        sys.executable,
        str(script),
        "--model",
        model,
        "--tp-size",
        "2",
        "--proc-per-node",
        "2",
        "--trust-remote-code",
        "--enforce-eager",
        "--enable-expert-parallel",
        "--enable-sleep-mode",
        "--model-weight-gib",
        "20",
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
def test_external_launcher_aclgraph(model):
    script = "/usr/local/python3.11.13/bin/python3.11/__w/vllm-ascend/tests/examples/test_weight_loader.py"
    env = os.environ.copy()
    # TODO: Change to 2 when ci machine has 4 cards
    cmd = [
        sys.executable,
        str(script),
        "--model",
        model,
        "--tp-size",
        "2",
        "--proc-per-node",
        "2",
        "--trust-remote-code",
        "--enable-expert-parallel",
        "--enable-sleep-mode",
        "--model-weight-gib",
        "20",
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


@pytest.mark.parametrize("model", MODELS)
def test_external_launcher_dense(model):
    script = "/usr/local/python3.11.13/bin/python3.11/__w/vllm-ascend/tests/examples/test_weight_loader.py"
    env = os.environ.copy()
    # TODO: Change to 2 when ci machine has 4 cards
    cmd = [
        sys.executable,
        str(script),
        "--model",
        model,
        "--tp-size",
        "2",
        "--proc-per-node",
        "2",
        "--trust-remote-code",
        "--enable-sleep-mode",
        "--model-weight-gib",
        "20",
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


@pytest.mark.parametrize("model", MODELS)
def test_external_launcher_dense_eager(model):
    script = "/usr/local/python3.11.13/bin/python3.11/__w/vllm-ascend/tests/examples/test_weight_loader.py"
    env = os.environ.copy()
    # TODO: Change to 2 when ci machine has 4 cards
    cmd = [
        sys.executable,
        str(script),
        "--model",
        model,
        "--tp-size",
        "2",
        "--proc-per-node",
        "2",
        "--trust-remote-code",
        "--enforce-eager",
        "--enable-sleep-mode",
        "--model-weight-gib",
        "20",
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
