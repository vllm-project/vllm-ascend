#!/usr/bin/env python3
"""Run one reproducible vLLM serving accuracy/performance case."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


PROMPTS = [
    "The capital of France is",
    "Explain why the sky appears blue in two sentences:",
    "Write a Python function that returns the factorial of a non-negative integer:",
    "Solve step by step: If 3x + 7 = 28, then x =",
    "Translate to Chinese: Speculative decoding can improve inference throughput.",
    "List three practical ways to reduce household energy consumption:",
    "Complete the sequence and explain the pattern: 2, 6, 12, 20, 30,",
    "Summarize the water cycle in one short paragraph:",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("target", "fixed", "adaptive", "physical"), required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--ascend-source", required=True)
    parser.add_argument("--vllm-source", required=True)
    parser.add_argument(
        "--custom-opp-root",
        default=(
            "/home/w00664509/dspark-vllm/hardware-vllm-ascend/"
            "vllm-ascend-spec1-sync-20260915/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
        ),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-prompts", type=int, default=16)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--physical-min-k", type=int)
    parser.add_argument("--device-offset", type=int, default=0)
    return parser.parse_args()


def git_head(path: str) -> str:
    result = subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_ready(base_url: str, process: subprocess.Popen, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before readiness with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = str(error)
        time.sleep(2)
    raise TimeoutError(f"server was not ready after {timeout}s: {last_error}")


def ensure_port_free(port: int) -> None:
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"port {port} is already in use")


def speculative_config(case: str) -> dict | None:
    if case == "target":
        return None
    config = {
        "method": "dspark",
        "model": "/home/vllm/weights/Qwen3-4b-dspark",
        "num_speculative_tokens": 5,
    }
    if case in ("adaptive", "physical"):
        config["enable_adaptive_verification"] = True
    return config


def serve_command(args: argparse.Namespace) -> list[str]:
    command = [
        "vllm",
        "serve",
        "/home/vllm/weights/Qwen3-4B/",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--max-model-len",
        "256",
        "--served-model-name",
        "auto",
        "--gpu-memory-utilization",
        "0.86",
        "--tensor-parallel-size",
        str(args.tp),
        "--max-num-seqs",
        "16",
        "--async-scheduling",
        "--compilation-config",
        json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY"}, separators=(",", ":")),
        "--safetensors-load-strategy",
        "prefetch",
    ]
    spec = speculative_config(args.case)
    if spec is not None:
        command.extend(("--speculative-config", json.dumps(spec, separators=(",", ":"))))
    if args.case == "physical":
        physical_config = {}
        if args.physical_min_k is not None:
            physical_config["min_k"] = args.physical_min_k
        command.extend(
            (
                "--additional-config",
                json.dumps(
                    {
                        "dynamic_spec_config": {
                            "method": "dspark",
                            "physical_k": physical_config,
                        }
                    },
                    separators=(",", ":"),
                ),
            )
        )
    return command


def benchmark_command(args: argparse.Namespace, output_dir: Path, filename: str, prompts: int) -> list[str]:
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        "auto",
        "--tokenizer",
        "/home/vllm/weights/Qwen3-4B/",
        "--dataset-name",
        "random",
        "--random-input-len",
        "128",
        "--random-output-len",
        "128",
        "--num-prompts",
        str(prompts),
        "--max-concurrency",
        "16",
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--seed",
        "20261013",
        "--save-result",
        "--result-dir",
        str(output_dir),
        "--result-filename",
        filename,
    ]


def terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def validate_benchmark(path: Path, expected_prompts: int) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    completed = int(result.get("completed", -1))
    failed = int(result.get("failed", -1))
    if completed != expected_prompts or failed != 0:
        raise RuntimeError(
            f"invalid benchmark result {path}: "
            f"completed={completed}, failed={failed}, expected={expected_prompts}"
        )


def main() -> int:
    args = parse_args()
    if args.tp not in (1, 2, 4, 8, 16):
        raise ValueError("tp must be one of 1, 2, 4, 8, 16")
    if args.device_offset < 0 or args.device_offset + args.tp > 16:
        raise ValueError("device-offset must select devices within 0..15")
    ensure_port_free(args.port)

    output_dir = Path(args.output_root) / f"tp{args.tp}" / args.case
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    source_path = f"{args.ascend_source}:{args.vllm_source}"
    inherited_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{source_path}:{inherited_path}" if inherited_path else source_path
    env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(
        str(index) for index in range(args.device_offset, args.device_offset + args.tp)
    )
    env["VLLM_PLUGINS"] = "ascend"
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    inherited_opp_path = env.get("ASCEND_CUSTOM_OPP_PATH")
    env["ASCEND_CUSTOM_OPP_PATH"] = (
        f"{args.custom_opp_root}:{inherited_opp_path}" if inherited_opp_path else args.custom_opp_root
    )

    command = serve_command(args)
    metadata = {
        "case": args.case,
        "tp": args.tp,
        "ascend_source": args.ascend_source,
        "ascend_commit": git_head(args.ascend_source),
        "vllm_source": args.vllm_source,
        "vllm_commit": git_head(args.vllm_source),
        "custom_opp_root": args.custom_opp_root,
        "visible_devices": env["ASCEND_RT_VISIBLE_DEVICES"],
        "device_offset": args.device_offset,
        "physical_min_k": args.physical_min_k,
        "serve_command": command,
        "num_prompts": args.num_prompts,
        "runs": args.runs,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    server_log_path = output_dir / "server.log"
    with server_log_path.open("w", encoding="utf-8") as server_log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            base_url = f"http://127.0.0.1:{args.port}"
            wait_ready(base_url, process, args.startup_timeout)

            accuracy = []
            for prompt in PROMPTS:
                response = post_json(
                    f"{base_url}/v1/completions",
                    {
                        "model": "auto",
                        "prompt": prompt,
                        "temperature": 0,
                        "max_tokens": 64,
                        "seed": 20261013,
                    },
                )
                choice = response["choices"][0]
                accuracy.append(
                    {
                        "prompt": prompt,
                        "text": choice["text"],
                        "finish_reason": choice.get("finish_reason"),
                        "usage": response.get("usage"),
                    }
                )
            (output_dir / "accuracy.json").write_text(
                json.dumps(accuracy, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            warmup_name = "warmup.json"
            with (output_dir / "warmup.log").open("w", encoding="utf-8") as warmup_log:
                subprocess.run(
                    benchmark_command(args, output_dir, warmup_name, args.warmup_prompts),
                    env=env,
                    check=True,
                    stdout=warmup_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            validate_benchmark(output_dir / warmup_name, args.warmup_prompts)

            for run_index in range(1, args.runs + 1):
                filename = f"benchmark-{run_index}.json"
                with (output_dir / f"benchmark-{run_index}.log").open("w", encoding="utf-8") as bench_log:
                    subprocess.run(
                        benchmark_command(args, output_dir, filename, args.num_prompts),
                        env=env,
                        check=True,
                        stdout=bench_log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                validate_benchmark(output_dir / filename, args.num_prompts)
            metadata["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            metadata["status"] = "passed"
            (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return 0
        except Exception as error:
            metadata["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            metadata["status"] = "failed"
            metadata["error"] = repr(error)
            (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            raise
        finally:
            terminate_process_group(process)


if __name__ == "__main__":
    sys.exit(main())
