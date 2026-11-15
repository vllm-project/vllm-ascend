# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared process and assertion helpers for preload pytest suites."""

import gc
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
from contextlib import contextmanager, suppress

RUNNER_MODULE = "tests.e2e.pull_request.one_card.model_runner_v2.preload.runner"
DAEMON_READY_TIMEOUT_S = 600
ENGINE_TIMEOUT_S = 900
MEMORY_DRIFT_LIMIT_BYTES = 256 * 1024 * 1024
KV_BUDGET_TOLERANCE_BYTES = 64 * 1024 * 1024
ZERO_COPY_MAX_WEIGHT_FRACTION = 0.2
COPY_MIN_RELEASE_FRACTION = 0.8
PROCESS_SHUTDOWN_TIMEOUT_S = 60
MEMORY_RELEASE_TIMEOUT_S = 30
MEMORY_RELEASE_POLL_INTERVAL_S = 0.5


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
    log_path = output_dir / f"{name}.log"
    with managed_process(command, log_path) as process:
        try:
            returncode = process.wait(timeout=ENGINE_TIMEOUT_S)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(f"{name} engine timed out; see {log_path}") from error
    if returncode:
        raise RuntimeError(f"{name} failed: see {log_path}\n{log_path.read_text()[-4000:]}")
    data = json.loads((output_dir / f"{name}.json").read_text())
    print(f"PASS {name}: ready={data['ready_s']:.2f}s, tokens/s={data['tokens_per_second']:.2f}", flush=True)
    return data


@contextmanager
def managed_process(command, log_path, *, graceful=False):
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield process
        finally:
            if process.poll() is None:
                if graceful:
                    process.terminate()
                else:
                    signal_process_group(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=PROCESS_SHUTDOWN_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    signal_process_group(process.pid, signal.SIGKILL)
                    process.wait()
            # The leader can exit before its engine workers do.
            signal_process_group(process.pid, signal.SIGKILL)


def signal_process_group(pid, sig):
    with suppress(ProcessLookupError):
        os.killpg(pid, sig)


def wait_for_health(process, port, log_path):
    started = time.perf_counter()
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"Process exited; see {log_path}\n{log_path.read_text()[-4000:]}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if response.status == 200:
                    return time.perf_counter() - started
        except (OSError, urllib.error.URLError):
            pass
        if time.perf_counter() - started > DAEMON_READY_TIMEOUT_S:
            raise TimeoutError(f"Readiness timed out; see {log_path}")
        time.sleep(0.5)


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
        log_path = output_dir / f"{name}-daemon.log"
        with managed_process(command, log_path) as process:
            ready_s = wait_for_health(process, port, log_path)
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


def check_tokens(actual, baseline):
    assert actual["token_ids"] == baseline["token_ids"], "Preload token IDs differ from the matching ND baseline"


def free_device_memory():
    import torch

    return torch.npu.mem_get_info(0)[0]


def check_preload_result(actual, baseline, mode, socket_dir, before_free, before_bytes):
    check_tokens(actual, baseline)
    memory = actual["worker_memory"][0]
    baseline_memory = baseline["worker_memory"][0]
    if mode == "zero_copy":
        assert memory["model_memory_usage"] < baseline_memory["model_memory_usage"] * ZERO_COPY_MAX_WEIGHT_FRACTION
    else:
        assert memory["kv_cache_memory"] <= baseline_memory["kv_cache_memory"] + KV_BUDGET_TOLERANCE_BYTES
        assert memory["model_memory_usage"] + memory["kv_cache_memory"] <= memory["requested_memory"]
        assert cached_bytes(socket_dir) == 0, "copy handoff did not release the daemon cache"
        # Graph process exit can precede the driver's device-memory reclamation.
        # Keep the release threshold, but allow that reclamation to finish.
        required_release = before_bytes * COPY_MIN_RELEASE_FRACTION
        deadline = time.monotonic() + MEMORY_RELEASE_TIMEOUT_S
        after_free = free_device_memory()
        while after_free - before_free < required_release and time.monotonic() < deadline:
            time.sleep(MEMORY_RELEASE_POLL_INTERVAL_S)
            after_free = free_device_memory()
        assert after_free - before_free >= required_release, (
            f"daemon retained most weight storage after copy: released {after_free - before_free} bytes, "
            f"expected at least {required_release:.0f} within {MEMORY_RELEASE_TIMEOUT_S}s"
        )


def expect_engine_failure(output_dir, model, name, expected, **kwargs):
    try:
        run_engine(output_dir, model, name, **kwargs)
    except RuntimeError:
        log = (output_dir / f"{name}.log").read_text()
        assert expected in log, f"{name} failed for an unexpected reason"
        print(f"PASS {name}: rejected as expected", flush=True)
    else:
        raise AssertionError(f"{name} unexpectedly loaded a model")


def check_serve_api(output_dir, model, baseline, socket_dir):
    """Check repeated HTTP requests against the same offline disk baseline."""
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    name = "serve-zero"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
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
        "--load-format=ipc_cache",
        "--model-loader-extra-config",
        json.dumps({"socket_dir": socket_dir, "mode": "zero_copy", "fallback": False}),
    ]
    log_path = output_dir / f"{name}.log"
    with managed_process(command, log_path, graceful=True) as process:
        ready_s = wait_for_health(process, port, log_path)
        payload = json.dumps(
            {
                "model": model,
                "prompt": "The capital of France is",
                "temperature": 0,
                "seed": 0,
                "max_tokens": 32,
                "ignore_eos": True,
                "return_token_ids": True,
            }
        ).encode()
        expected = baseline["token_ids"][0]
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
        (output_dir / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"PASS zero_copy serve API, ready={ready_s:.2f}s", flush=True)


def ipc_consumer(connection):
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    entry, expected, stride, offset = connection.recv()
    tensor = entry.rebuild(0)
    torch.testing.assert_close(tensor.cpu(), torch.tensor(expected, dtype=tensor.dtype))
    assert tensor.stride() == stride
    assert tensor.storage_offset() == offset
    tensor.fill_(7)
    torch.npu.synchronize()
    del tensor
    gc.collect()
    connection.send("released")
    connection.close()


def ipc_alias_consumer(connection):
    import torch
    import torch_npu  # noqa: F401
    from vllm.config.load import LoadConfig

    from vllm_ascend.model_loader.preload.loader import AscendIpcModelLoader

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    state = connection.recv()
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty(16, 16, dtype=torch.bfloat16, device="meta"))
    model.tied = model.weight
    model.register_buffer("view", torch.empty(8, 8, dtype=torch.bfloat16, device="meta"))
    loader = AscendIpcModelLoader(LoadConfig(load_format="ipc_cache", model_loader_extra_config={"fallback": False}))
    loader._apply_entries(model, state, 0)
    assert model.weight is model.tied
    assert model.weight.untyped_storage().data_ptr() == model.view.untyped_storage().data_ptr()
    assert model.view.stride() == (16, 1) and model.view.storage_offset() == 33
    model.view.fill_(7)
    torch.npu.synchronize()
    torch.testing.assert_close(model.weight[2:10, 1:9], torch.full_like(model.view, 7))
    del model
    gc.collect()
    connection.send("released")
    connection.close()


def check_rope_state(data):
    assert data["rope_state"], "No rotary embedding state was reported"
    for state in data["rope_state"]:
        assert state["types"] == ["AscendYaRNRotaryEmbedding"], state
        assert state["global_cache_device"].startswith("npu"), state
        assert state["global_matches_model_buffer"], state
