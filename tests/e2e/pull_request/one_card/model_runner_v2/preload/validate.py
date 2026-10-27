# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Run the single-NPU preload acceptance matrix in isolated processes."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

RUNNER_MODULE = "tests.e2e.pull_request.one_card.model_runner_v2.preload.runner"
DAEMON_READY_TIMEOUT_S = 600
ENGINE_TIMEOUT_S = 900
MEMORY_DRIFT_LIMIT_BYTES = 256 * 1024 * 1024
KV_BUDGET_TOLERANCE_BYTES = 64 * 1024 * 1024


def cached_bytes(socket_dir):
    from vllm.model_executor.model_loader.weight_cache.protocol import get_socket_path, recv_msg, send_msg

    from vllm_ascend.platform import NPUPlatform

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(5)
        connection.connect(get_socket_path(NPUPlatform.get_device_uuid(0), socket_dir))
        send_msg(connection, {"cmd": "get_memory"})
        response = recv_msg(connection)
    assert response["status"] == "ok", response
    return response["memory_bytes"]


def run_engine(
    output_dir,
    model,
    name,
    *,
    mode="disk",
    dtype="bfloat16",
    graph=False,
    socket_dir=None,
    fallback=False,
    nz=0,
    hf_overrides=None,
):
    command = [
        sys.executable,
        "-m",
        RUNNER_MODULE,
        "--model",
        model,
        "--output",
        str(output_dir / f"{name}.json"),
        "--mode",
        mode,
        "--dtype",
        dtype,
        "--nz",
        str(nz),
    ]
    if graph:
        command.append("--graph")
    if socket_dir:
        command.extend(["--socket-dir", socket_dir])
    if fallback:
        command.append("--fallback")
    if hf_overrides:
        command.extend(["--hf-overrides", json.dumps(hf_overrides)])
    print(f"START {name}", flush=True)
    with (output_dir / f"{name}.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=ENGINE_TIMEOUT_S)
    if result.returncode:
        raise RuntimeError(f"{name} failed: see {output_dir / f'{name}.log'}")
    data = json.loads((output_dir / f"{name}.json").read_text())
    print(f"PASS {name}: ready={data['ready_s']:.2f}s, tokens/s={data['tokens_per_second']:.2f}", flush=True)
    return data


@contextmanager
def daemon(output_dir, model, name, dtype, *, hf_overrides=None):
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    # AF_UNIX limits the entire path; upstream's UUID-hashed name is long.
    with tempfile.TemporaryDirectory(prefix="ap-", dir="/tmp") as socket_dir:
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "preload",
            "--model",
            model,
            "--dtype",
            dtype,
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "2048",
            "--enforce-eager",
            "--additional-config",
            '{"weight_nz_mode":0}',
            "--weight-cache-socket-dir",
            socket_dir,
            "--weight-cache-health-host",
            "127.0.0.1",
            "--weight-cache-health-port",
            str(port),
        ]
        if hf_overrides:
            command.extend(["--hf-overrides", json.dumps(hf_overrides)])
        with (output_dir / f"{name}-daemon.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            started = time.perf_counter()
            try:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"{name} daemon exited; see {name}-daemon.log")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                            if response.status == 200:
                                break
                    except (OSError, urllib.error.URLError):
                        pass
                    if time.perf_counter() - started > DAEMON_READY_TIMEOUT_S:
                        raise TimeoutError(f"{name} daemon readiness timed out")
                    time.sleep(0.5)
                ready_s = time.perf_counter() - started
                before_bytes = cached_bytes(socket_dir)
                print(f"READY {name} daemon in {ready_s:.2f}s", flush=True)
                yield socket_dir
                (output_dir / f"{name}-daemon.json").write_text(
                    json.dumps(
                        {
                            "ready_s": ready_s,
                            "exported_bytes_before": before_bytes,
                            "exported_bytes_after": cached_bytes(socket_dir),
                        }
                    )
                    + "\n"
                )
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()


def check_tokens(actual, baseline):
    assert actual["token_ids"] == baseline["token_ids"], "Preload token IDs differ from the matching ND baseline"


def expect_engine_failure(output_dir, model, name, expected, **kwargs):
    try:
        run_engine(output_dir, model, name, **kwargs)
    except RuntimeError:
        log = (output_dir / f"{name}.log").read_text()
        assert expected in log, f"{name} failed for an unexpected reason"
        print(f"PASS {name}: rejected as expected", flush=True)
    else:
        raise AssertionError(f"{name} unexpectedly loaded a model")


def validate_dtype(output_dir, model, dtype, restarts):
    import torch

    baselines = {}
    for graph in (False, True):
        name = f"{dtype}-disk-{'graph' if graph else 'eager'}"
        baselines[graph] = run_engine(output_dir, model, name, dtype=dtype, graph=graph)
    free_after_restart = []
    with daemon(output_dir, model, f"{dtype}-zero", dtype) as socket_dir:
        for iteration in range(restarts):
            data = run_engine(
                output_dir,
                model,
                f"{dtype}-zero-eager-{iteration}",
                mode="zero_copy",
                dtype=dtype,
                socket_dir=socket_dir,
            )
            check_tokens(data, baselines[False])
            # DeviceMemoryProfiler should see only consumer-owned state.
            assert data["worker_memory"][0]["model_memory_usage"] < (
                baselines[False]["worker_memory"][0]["model_memory_usage"] * 0.2
            ), "zero_copy allocated a substantial fraction of another model"
            free_after_restart.append(torch.npu.mem_get_info(0)[0])
        check_tokens(
            run_engine(
                output_dir,
                model,
                f"{dtype}-zero-graph",
                mode="zero_copy",
                dtype=dtype,
                graph=True,
                socket_dir=socket_dir,
            ),
            baselines[True],
        )
    assert max(free_after_restart) - min(free_after_restart) <= MEMORY_DRIFT_LIMIT_BYTES, free_after_restart
    (output_dir / f"{dtype}-restart-memory.json").write_text(json.dumps(free_after_restart) + "\n")
    validate_copy_modes(output_dir, model, dtype, baselines)
    return baselines


def validate_copy_modes(output_dir, model, dtype, baselines):
    import torch

    for graph in (False, True):
        name = f"{dtype}-copy-{'graph' if graph else 'eager'}"
        with daemon(output_dir, model, name, dtype) as socket_dir:
            before_free = torch.npu.mem_get_info(0)[0]
            before_bytes = cached_bytes(socket_dir)
            data = run_engine(output_dir, model, name, mode="copy", dtype=dtype, graph=graph, socket_dir=socket_dir)
            check_tokens(data, baselines[graph])
            assert data["worker_memory"][0]["kv_cache_memory"] <= (
                baselines[graph]["worker_memory"][0]["kv_cache_memory"] + KV_BUDGET_TOLERANCE_BYTES
            ), "copy received KV budget credit for the released producer weights"
            memory = data["worker_memory"][0]
            assert memory["model_memory_usage"] + memory["kv_cache_memory"] <= memory["requested_memory"]
            after_free = torch.npu.mem_get_info(0)[0]
            (output_dir / f"{name}-release-memory.json").write_text(json.dumps([before_free, after_free]) + "\n")
            assert cached_bytes(socket_dir) == 0, "copy handoff did not release the daemon cache"
            assert after_free - before_free >= before_bytes * 0.8, "daemon retained most weight storage after copy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    bf16 = validate_dtype(args.output_dir, args.model, "bfloat16", restarts=5)
    fp16 = validate_dtype(args.output_dir, args.model, "float16", restarts=1)
    with tempfile.TemporaryDirectory(prefix="ap-", dir="/tmp") as unavailable:
        expect_engine_failure(
            args.output_dir,
            args.model,
            "strict-no-daemon",
            "did not become ready within",
            mode="zero_copy",
            socket_dir=unavailable,
        )
        check_tokens(
            run_engine(
                args.output_dir,
                args.model,
                "fallback-no-daemon",
                mode="zero_copy",
                socket_dir=unavailable,
                fallback=True,
            ),
            bf16[False],
        )
    with daemon(args.output_dir, args.model, "mismatch", "bfloat16") as socket_dir:
        expect_engine_failure(
            args.output_dir,
            args.model,
            "strict-dtype-mismatch",
            "WeightCacheKey mismatch",
            mode="zero_copy",
            dtype="float16",
            socket_dir=socket_dir,
        )
        check_tokens(
            run_engine(
                args.output_dir,
                args.model,
                "fallback-dtype-mismatch",
                mode="zero_copy",
                dtype="float16",
                socket_dir=socket_dir,
                fallback=True,
            ),
            fp16[False],
        )
    run_engine(args.output_dir, args.model, "bfloat16-disk-nz-eager", nz=2)
    print("ALL PRELOAD ACCEPTANCE CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
