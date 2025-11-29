"""
Run `pytest tests/e2e/multicard/test_data_parallel_tp2.py`.
"""

import os
import subprocess
import sys
from unittest.mock import patch

import pytest

MODELS = ["Qwen/Qwen3-0.6B"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3"})
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
        "2",
        "--node-size",
        "1",
        "--node-rank",
        "0",
        "--trust-remote-code",
    ]

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
