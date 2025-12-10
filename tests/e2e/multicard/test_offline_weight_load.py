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
Run `pytest tests/multicard/test_offline_load_weight.py`.
"""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

MODELS = ["Qwen/Qwen3-30B-A3B"]


@pytest.mark.parametrize("model", MODELS)
@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "0"})
def test_offline_weight_load_and_sleepmode(model):
    script = Path(
        __file__
    ).parent.parent.parent.parent / "examples" / "offline_external_launcher.py"
    env = os.environ.copy()
    cmd = [
        sys.executable,
        str(script),
        "--model",
        model,
        "--tp-size",
        "2",
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
        "0.8",
    ]

    print(f"Running subprocess: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
    )
    output = proc.stdout.decode(errors='ignore')

    print(output)

    assert "Generated text:" in output
    assert "Sleep and wake up successfully!!" in output
    assert proc.returncode == 0
