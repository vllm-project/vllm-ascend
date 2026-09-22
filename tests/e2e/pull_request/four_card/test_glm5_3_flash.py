# SPDX-License-Identifier: Apache-2.0
"""Opt-in TP4 Flash smoke. Requires checkpoint metadata, not full weight shards.

Run with --glm53flash-checkpoint PATH. Missing configuration is a visible skip;
the CI workflow must supply the artifact path before enabling this job.
This functional smoke does not replace calibrated precision/performance gates.
"""

import json
import os
import signal
import subprocess
import sys
from contextlib import suppress

import pytest

from tools.ci.glm53flash_config import build


@pytest.mark.parametrize("mode", ["graph", "eager"])
def test_glm53flash_tp4(request, tmp_path, mode):
    source = request.config.getoption("--glm53flash-checkpoint")
    if source is None:
        pytest.skip("Supply --glm53flash-checkpoint; Flash CI assets are not configured")
    model = tmp_path / "model"
    build(source, model, layers=9, mtp=False, multimodal=False)
    output = tmp_path / mode
    command = [
        sys.executable,
        "-m",
        "tools.ci.glm53flash_collect",
        "--model",
        str(model),
        "--output",
        str(output),
        "--mode",
        mode,
        "--smoke",
    ]
    with (tmp_path / f"{mode}.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            assert process.wait(timeout=900) == 0, f"Inspect {tmp_path / f'{mode}.log'}"
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "PASS"
    assert len(result["replays"]) == 4
    assert all(n > 0 if mode == "graph" else n == 0 for n in result["replays"])
