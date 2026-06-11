import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.e2e.conftest import wait_until_npu_memory_free

MODELS = ["Qwen/Qwen3-30B-A3B"]
REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_PARALLEL_SCRIPT = REPO_ROOT / "examples" / "offline_data_parallel.py"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3"})
@wait_until_npu_memory_free(target_free_percentage=0.95)
def test_qwen3_inference_dp2_tp2(model, max_tokens):
    env = os.environ.copy()

    cmd = [
        sys.executable,
        str(DATA_PARALLEL_SCRIPT),
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
        "--enforce-eager",
    ]

    print(f"Running subprocess: {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    output = proc.stdout.decode(errors="ignore")

    print(output)

    assert "DP rank 0 needs to process" in output
    assert "DP rank 1 needs to process" in output
    assert "Generated text:" in output
    assert proc.returncode == 0
