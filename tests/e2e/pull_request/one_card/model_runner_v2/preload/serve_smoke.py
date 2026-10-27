# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exercise the public serve CLI and completions API against ND preload."""

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
from contextlib import nullcontext
from pathlib import Path

from .validate import DAEMON_READY_TIMEOUT_S, daemon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--mode", choices=("zero_copy", "disk"), default="zero_copy")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    name = "serve-zero" if args.mode == "zero_copy" else "serve-disk"
    producer = daemon(args.output_dir, args.model, name, "bfloat16") if args.mode == "zero_copy" else nullcontext()
    with producer as socket_dir:
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            args.model,
            "--host=127.0.0.1",
            f"--port={port}",
            "--dtype=bfloat16",
            "--seed=0",
            "--max-model-len=2048",
            "--max-num-seqs=8",
            "--max-num-batched-tokens=2048",
            "--gpu-memory-utilization=0.55",
            "--shutdown-timeout=30",
            "--enforce-eager",
            "--no-enable-prefix-caching",
            "--additional-config",
            '{"weight_nz_mode":0}',
        ]
        if args.mode == "zero_copy":
            command.extend(
                [
                    "--load-format=ipc_cache",
                    "--model-loader-extra-config",
                    json.dumps({"socket_dir": socket_dir, "mode": "zero_copy", "fallback": False}),
                ]
            )
        with (args.output_dir / f"{name}.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            started = time.perf_counter()
            try:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Serve process exited; see {name}.log")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                            if response.status == 200:
                                break
                    except (OSError, urllib.error.URLError):
                        pass
                    if time.perf_counter() - started > DAEMON_READY_TIMEOUT_S:
                        raise TimeoutError("Serve readiness timed out")
                    time.sleep(0.5)
                ready_s = time.perf_counter() - started
                payload = json.dumps(
                    {
                        "model": args.model,
                        "prompt": "The capital of France is",
                        "temperature": 0,
                        "seed": 0,
                        "max_tokens": 32,
                        "ignore_eos": True,
                        "return_token_ids": True,
                    }
                ).encode()
                expected = json.loads(args.baseline.read_text())["token_ids"][0]
                for _ in range(2):
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{port}/v1/completions",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request, timeout=120) as response:
                        result = json.load(response)
                    assert result["choices"][0]["token_ids"] == expected, "Serve tokens differ from ND baseline"
                result["server_ready_s"] = ready_s
                (args.output_dir / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                print(f"PASS {args.mode} serve API, ready={ready_s:.2f}s", flush=True)
            finally:
                if process.poll() is None:
                    # Let the API server stop its engine before terminating
                    # the producer. Use the upstream shutdown grace period
                    # instead of its default immediate-abort behavior.
                    process.terminate()
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()


if __name__ == "__main__":
    main()
