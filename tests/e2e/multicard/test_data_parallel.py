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

Run `pytest tests/multicard/test_data_parallel.py`.
"""

import os
import subprocess
import sys
from unittest.mock import patch

import pytest

MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-30B-A3B"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1"})
def test_data_parallel_inference(model, max_tokens):
    script = "examples/offline_data_parallel.py"

    env = os.environ.copy()

    cmd = [
        sys.executable,
        script,
        "--model",
        model,
        "--dp-size",
        "2",
        "--tp-size",
        "1",
        "--node-size",
        "1",
        "--node-rank",
        "0",
        "--trust-remote-code",
        "--enforce-eager",
    ]
    if model == "Qwen/Qwen3-30B-A3B":
        cmd.append("--enable-expert-parallel")

    print(f"Running subprocess: {' '.join(cmd)}")
    proc = subprocess.run(cmd,
                          env=env,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,
                          timeout=600)
    output = proc.stdout.decode()

    print(output)

    assert "DP rank 0 needs to process" in output
    assert "DP rank 1 needs to process" in output
    assert "Generated text:" in output
    assert proc.returncode == 0
