# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One server configuration for baseline capture and the regression test."""

import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.glm53_flash.checkpoint import object_hash, read_json, write_json
from tests.e2e.glm53_flash.logprobs import (
    LOGPROB_ATOL,
    REPLAY_ROUNDS,
    SERVED_MODEL,
    compare_completion,
    completion_request,
    curl_completion,
)

FIXTURES = Path(__file__).parent / "fixtures"
RUNTIME_PREFIX = "GLM53_CI_RUNTIME="
STARTUP_TIMEOUT_SECONDS = 900


def deterministic_env() -> dict[str, str]:
    return {
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "LCCL_DETERMINISTIC": "1",
        "HCCL_DETERMINISTIC": "true",
        "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
        "ATB_LLM_LCOC_ENABLE": "0",
        "CLOSE_MATMUL_K_SHIFT": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        # This test is single-host. Avoid DNS-dependent Gloo initialization.
        "GLOO_SOCKET_IFNAME": "lo",
        "VLLM_LOGGING_LEVEL": "INFO",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }


def serve_args() -> list[str]:
    return [
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--tensor-parallel-size",
        "2",
        "--data-parallel-size",
        "1",
        "--enable-expert-parallel",
        "--seed",
        "0",
        "--generation-config",
        "vllm",
        "--quantization",
        "ascend",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "512",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        "--no-disable-hybrid-kv-cache-manager",
        "--mamba-cache-mode",
        "none",
        "--kv-cache-memory-bytes",
        str(1024**3),
        "--enforce-eager",
        "--language-model-only",
        "--logprobs-mode",
        "raw_logprobs",
        "--max-logprobs",
        "5",
        "--worker-cls",
        "tests.e2e.glm53_flash.worker.PrecisionWorker",
    ]


def contract(source: dict, projection: dict, prompts: dict) -> dict:
    return {
        "source_id": source["source_id"],
        "projection_sha256": object_hash(projection),
        "prompts_sha256": object_hash(prompts),
        "serve_args": serve_args(),
        "environment": deterministic_env(),
        "request_parameters": {key: value for key, value in completion_request([]).items() if key != "prompt"},
        "comparison": {"atol": LOGPROB_ATOL, "rtol": 0, "replay_rounds": REPLAY_ROUNDS},
    }


def environment_info() -> dict:
    packages = {}
    for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "transformers", "triton-ascend", "safetensors"):
        packages[name] = importlib.metadata.version(name)
    result = {"packages": packages}
    for name, command in (
        ("npu_smi", ["npu-smi", "info"]),
        ("board", ["npu-smi", "info", "-t", "board", "-i", "0", "-c", "0"]),
        ("ascend_commit", ["git", "rev-parse", "HEAD"]),
    ):
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=30)
            result[name] = process.stdout.strip() if process.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = f"unavailable: {error}"
    for name, path in (
        ("driver", "/usr/local/Ascend/driver/version.info"),
        ("cann_compiler", "/usr/local/Ascend/ascend-toolkit/latest/compiler/version.info"),
        ("cann_opp", "/usr/local/Ascend/ascend-toolkit/latest/opp/version.info"),
    ):
        result[name] = Path(path).read_text() if Path(path).is_file() else "unavailable"
    return result


class LoggedServer(RemoteOpenAIServer):
    """Reuse server readiness and process-tree cleanup, redirect only logging."""

    def __init__(self, model: Path, artifacts: Path):
        self.log_path = artifacts / "server.log"
        port = get_open_port()
        try:
            super().__init__(
                str(model),
                serve_args() + ["--port", str(port)],
                server_host="127.0.0.1",
                server_port=port,
                auto_port=False,
                env_dict=deterministic_env(),
                max_wait_seconds=STARTUP_TIMEOUT_SECONDS,
            )
        except BaseException:
            if hasattr(self, "proc"):
                self._terminate_server()
            raise

    def _start_server(self, model: str, server_cmd: list[str], env_dict: dict[str, str] | None) -> None:
        environment = os.environ.copy()
        environment.update(env_dict or {})
        # Console scripts and spawned workers must resolve this checkout's
        # test-only worker; tests are intentionally not installed as a package.
        source_root = str(Path(__file__).resolve().parents[3])
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, environment.get("PYTHONPATH"))))
        # Disable user-specific serving overrides; the test owns its configuration.
        environment.pop("VLLM_BATCH_INVARIANT", None)
        with self.log_path.open("w", encoding="utf-8") as log:
            self.proc = subprocess.Popen(server_cmd, env=environment, stdout=log, stderr=subprocess.STDOUT)


def read_runtime(log: Path) -> list[dict]:
    runtimes = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if RUNTIME_PREFIX in line:
            value = line.split(RUNTIME_PREFIX, 1)[1]
            runtimes.append(json.JSONDecoder().raw_decode(value)[0])
    if sorted(item["rank"] for item in runtimes) != [0, 1]:
        raise AssertionError(f"Missing/duplicated effective-configuration checks from both workers: {log}")
    return sorted(runtimes, key=lambda item: item["rank"])


def run_suite(model: Path, prompts: dict, artifacts: Path, golden: dict | None = None) -> dict:
    artifacts.mkdir(parents=True, exist_ok=False)
    write_json(artifacts / "environment.json", environment_info())
    results = {}
    timing = time.monotonic()
    errors = []
    report = {"max_abs_error": 0.0, "completed_requests": 0, "completed_rounds": 0}
    try:
        with LoggedServer(model, artifacts) as server:
            runtime = read_runtime(artifacts / "server.log")
            write_json(artifacts / "runtime.json", {"workers": runtime})
            report["startup_seconds"] = time.monotonic() - timing
            if golden is not None:
                expected_devices = [worker["device"] for worker in golden["workers"]]
                if [worker["device"] for worker in runtime] != expected_devices:
                    raise AssertionError("Hardware differs from the reviewed golden")
            for repeat in range(REPLAY_ROUNDS):
                for case in prompts["cases"]:
                    name = case["name"]
                    result = curl_completion(
                        server.url_for("v1/completions"), case["token_ids"], artifacts / f"round-{repeat}" / name
                    )
                    if golden is not None or repeat:
                        reference = golden["cases"][name] if golden is not None else results[name]
                        try:
                            error = compare_completion(reference, result, f"{name}/round-{repeat}")
                            report["max_abs_error"] = max(report["max_abs_error"], error)
                        except AssertionError as error:
                            errors.append(str(error))
                    if repeat == 0:
                        results[name] = result
                    report["completed_requests"] += 1
                report["completed_rounds"] += 1
            write_json(artifacts / "observed.json", {"cases": results, "workers": runtime})
        peaks = {}
        for line in (artifacts / "server.log").read_text(encoding="utf-8", errors="replace").splitlines():
            if "GLM53_CI_PEAK=" in line:
                record = json.JSONDecoder().raw_decode(line.split("GLM53_CI_PEAK=", 1)[1])[0]
                rank = str(record["rank"])
                peaks[rank] = max(peaks.get(rank, 0), record["allocated_bytes"])
        report["peak_torch_allocated_bytes_by_rank"] = peaks
    except BaseException as error:
        report["fatal_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["total_seconds"] = time.monotonic() - timing
        report["errors"] = errors
        write_json(artifacts / "report.json", report)
    if errors:
        raise AssertionError("\n".join(errors) + f"\nArtifacts: {artifacts}")
    return {"cases": results, "workers": runtime, "report": report}


def load_prompts() -> dict:
    prompts = read_json(FIXTURES / "prompts.json")
    names = [case["name"] for case in prompts["cases"]]
    if len(names) != len(set(names)) or len(names) != 11:
        raise ValueError("Expected eleven distinct reviewed prompt cases")
    for case in prompts["cases"]:
        ids = case["token_ids"]
        if not ids or any(type(token) is not int or not 0 <= token < 154880 for token in ids):
            raise ValueError(f"Invalid input IDs: {case['name']}")
        if case["name"].startswith("boundary_") and len(ids) != int(case["name"].split("_")[1]):
            raise ValueError(f"Boundary fixture length mismatch: {case['name']}")
    return prompts
