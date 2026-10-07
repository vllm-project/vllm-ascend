# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""The launcher must preserve device assignment and paired source precedence."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("tp,devices", [("1", "5"), ("4", "0,1,2,3")])
def test_startlux_launcher_preserves_arguments(tmp_path, tp, devices):
    launcher = Path(__file__).parents[2] / "examples/models/startlux_decision/serve.sh"
    capture = tmp_path / "capture"
    capture.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" "$ASCEND_RT_VISIBLE_DEVICES" "$VLLM_ENABLE_V1_MULTIPROCESSING" "$PYTHONPATH"\n'
    )
    capture.chmod(0o755)
    env = dict(
        os.environ,
        MODEL_PATH="/models/decision checkpoint",
        VLLM_SOURCE="/source/vllm",
        STARTLUX_SOURCE="/source/renderer",
        ASCEND_RT_VISIBLE_DEVICES=devices,
        TENSOR_PARALLEL_SIZE=tp,
        PYTHON_BIN=str(capture),
        PYTHONPATH="/bindings",
        PORT="18000",
        MAX_MODEL_LEN="4096",
    )
    env.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
    run = subprocess.run(
        ["bash", str(launcher), "--enforce-eager"], env=env, capture_output=True, text=True, check=True
    )
    args = run.stdout.splitlines()
    assert args[:10] == [
        "/source/vllm/examples/pooling/classify/serve_startlux_decision.py",
        "--model",
        "/models/decision checkpoint",
        "--tensor-parallel-size",
        tp,
        "--max-model-len",
        "4096",
        "--port",
        "18000",
        "--enforce-eager",
    ]
    assert args[10:12] == [devices, "0"]
    assert args[12] == f"/source/vllm:{launcher.parents[3]}:/source/renderer:/bindings"


def test_startlux_launcher_requires_explicit_devices():
    launcher = Path(__file__).parents[2] / "examples/models/startlux_decision/serve.sh"
    env = dict(os.environ, MODEL_PATH="/model", VLLM_SOURCE="/vllm", STARTLUX_SOURCE="/renderer")
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    run = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
    assert run.returncode != 0
    assert "Select the assigned NPU devices" in run.stderr
