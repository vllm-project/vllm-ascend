# SPDX-License-Identifier: Apache-2.0
"""Strict first-500 GSM8K accuracy benchmark called by the shared YAML runner."""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from tools.kimi_k3.accuracy_case import validate_inputs, validate_results


def run_accuracy(model, port, host_ip, config):
    if (config.get("num_prompts"), config.get("max_out_len"), config.get("batch_size")) != (500, 1024, 60):
        raise ValueError("This recorded accuracy contract requires 500 questions, output 1024 and concurrency 60")
    required = ("KIMI_EP256_MODEL", "KIMI_GSM8K_DATASET", "KIMI_EP256_BASELINE", "KIMI_EP256_TOLERANCE")
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise ValueError(f"Provision runner inputs before running: {missing}")
    if Path(model).resolve() != Path(os.environ[required[0]]).resolve():
        raise ValueError("Service model differs from the requested EP256 checkpoint")
    dataset = Path(os.environ[required[1]]).resolve()
    baseline, tolerance = validate_inputs(model, dataset, os.environ[required[2]], os.environ[required[3]])
    executable = shutil.which("ais_bench")
    if not executable:
        raise RuntimeError("AISBench must be installed in the target runtime")
    repo = Path(__file__).resolve().parents[2]
    artifacts = repo / "benchmark_results" / "Kimi-K3-W4A8-EP256-A5"
    artifacts.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=artifacts))
    env = os.environ.copy()
    env.update(
        KIMI_EP256_MODEL=str(Path(model).resolve()),
        KIMI_GSM8K_DATASET=str(dataset),
        KIMI_API_HOST=host_ip,
        KIMI_API_PORT=str(port),
    )
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo), env.get("PYTHONPATH", "")]))
    work = output / "aisbench"
    with (output / "aisbench.log").open("w") as log:
        subprocess.run(
            [
                executable,
                str(repo / config["config_file"]),
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
    # Let the shared report use the same actual acceptance values.
    config.update(baseline=baseline, threshold=tolerance)
    if summary["request_failures"]:
        raise AssertionError("Infrastructure request failures invalidate this run")
    if abs(summary["accuracy"] - baseline) > tolerance:
        raise AssertionError(summary)
    return summary["accuracy"]
