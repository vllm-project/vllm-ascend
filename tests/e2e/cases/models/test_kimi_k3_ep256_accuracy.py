# SPDX-License-Identifier: Apache-2.0
"""Single-node A5 eight-card EP256 GSM8K first-500 accuracy regression."""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.e2e.conftest import RemoteOpenAIServer
from tools.kimi_k3.accuracy_case import service_arguments, validate_inputs, validate_results


@pytest.mark.e2e_model("Kimi-K3")
def test_kimi_k3_ep256_gsm8k_accuracy():
    required = ("KIMI_EP256_MODEL", "KIMI_GSM8K_DATASET", "KIMI_EP256_BASELINE", "KIMI_EP256_TOLERANCE")
    missing = [key for key in required if not os.environ.get(key)]
    assert not missing, f"Provision runner inputs before running: {missing}"
    model = Path(os.environ["KIMI_EP256_MODEL"]).resolve()
    dataset = Path(os.environ["KIMI_GSM8K_DATASET"]).resolve()
    baseline, tolerance = validate_inputs(model, dataset, os.environ[required[2]], os.environ[required[3]])
    executable = shutil.which("ais_bench")
    assert executable, "AISBench must be installed in the target runtime"
    repo = Path(__file__).resolve().parents[4]
    script = (repo / "tools/kimi_k3/serve_ep256.sh").read_text()
    args = service_arguments(script)
    artifacts = repo / "benchmark_results" / "Kimi-K3-W4A8-EP256-A5"
    artifacts.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=artifacts))
    env = os.environ.copy()
    env.update(
        KIMI_EP256_MODEL=str(model),
        KIMI_GSM8K_DATASET=str(dataset),
        KIMI_API_HOST="127.0.0.1",
        VLLM_USE_V2_MODEL_RUNNER="1",
        HCCL_BUFFSIZE="800",
        PYTORCH_NPU_ALLOC_CONF="expandable_segments:True",
        OMP_NUM_THREADS="1",
        VLLM_ENGINE_READY_TIMEOUT_S="7200",
        VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="3000",
    )
    env.pop("HCCL_OP_EXPANSION_MODE", None)
    # env_dict updates the parent environment; explicitly unset this optional flag.
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)
        with RemoteOpenAIServer(
            str(model), args, server_host="127.0.0.1", env_dict=env, max_wait_seconds=7200
        ) as server:
            env["KIMI_API_PORT"] = str(server.port)
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo), env.get("PYTHONPATH", "")]))
            config = repo / "tests/e2e/cases/models/configs/Kimi/gsm8k_ep256_first500.py"
            work = output / "aisbench"
            with (output / "aisbench.log").open("w") as log:
                subprocess.run(
                    [
                        executable,
                        str(config),
                        "--mode",
                        "all",
                        "--num-warmups",
                        "0",
                        "--max-num-workers",
                        "1",
                        "--dump-eval-details",
                        "--work-dir",
                        str(work),
                    ],
                    cwd=repo,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=14400,
                )
    summary = validate_results(work, dataset)
    summary.update(baseline=baseline, tolerance_percentage_points=tolerance)
    (output / "accuracy-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    assert summary["request_failures"] == 0, "Infrastructure request failures invalidate this run"
    assert abs(summary["accuracy"] - baseline) <= tolerance, summary
