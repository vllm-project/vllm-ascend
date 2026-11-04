# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Failed-rank behaviour for the MRV1 pipeline-parallel runner.

Killing one PP worker must be *detected*: EngineCore logs
``Worker proc ... died unexpectedly`` and shuts the executor down.

Known limitation (vLLM core behaviour, unchanged by this PR): the driver-side
``LLM.generate`` does not return after the death (fail-slow).  This test
therefore asserts detection + executor shutdown, and tears the engine down
itself instead of waiting for the client call.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

MODEL = os.environ.get("QWEN3_5_4B_MODEL_PATH", "Qwen/Qwen3.5-4B")

_ENGINE = r"""
import os
from vllm import LLM, SamplingParams

llm = LLM(
    model=os.environ["QWEN3_5_4B_MODEL_PATH"],
    max_model_len=1024,
    max_num_seqs=1,
    tensor_parallel_size=1,
    pipeline_parallel_size=2,
    distributed_executor_backend="mp",
    gpu_memory_utilization=0.8,
    enforce_eager=True,
    enable_prefix_caching=False,
    speculative_config={"method": "mtp", "num_speculative_tokens": 3, "enforce_eager": True},
    hf_overrides={"language_model_only": True},
)
print("ENGINE_READY", flush=True)
llm.generate(
    ["Hello, my name is"],
    SamplingParams(temperature=0.0, max_tokens=512, ignore_eos=True),
)
print("GENERATE_DONE", flush=True)
"""


def _npu_count() -> int:
    try:
        import torch
        import torch_npu  # noqa: F401

        return int(torch.npu.device_count())
    except Exception:  # pragma: no cover - environment dependent
        return 0


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="spec_decode",
    parallel="TP,PP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager",
)
@pytest.mark.skipif(_npu_count() < 2, reason="PP2 needs 2 NPU devices")
def test_mrv1_pp_worker_death_is_detected_and_executor_shuts_down() -> None:
    import psutil

    env = os.environ.copy()
    env.update(
        {
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "ASCEND_RT_VISIBLE_DEVICES": "0,1",
            "QWEN3_5_4B_MODEL_PATH": MODEL,
        }
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", _ENGINE],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )

    lines: list[str] = []
    lock = threading.Lock()

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            with lock:
                lines.append(line.rstrip("\n"))

    threading.Thread(target=_reader, daemon=True).start()

    def _tail() -> str:
        with lock:
            return "\n".join(lines[-60:])

    def _wait_for(needle: str, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with lock:
                if any(needle in line for line in lines):
                    return True
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        with lock:
            return any(needle in line for line in lines)

    try:
        assert _wait_for("ENGINE_READY", 300), f"engine did not become ready:\n{_tail()}"

        root = psutil.Process(proc.pid)
        victims: list[psutil.Process] = []
        deadline = time.time() + 30
        while time.time() < deadline and not victims:
            for child in root.children(recursive=True):
                try:
                    name = child.name()
                    cmd = " ".join(child.cmdline())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                if "Worker_PP" in name or "VllmWorker" in name or "Worker_PP" in cmd or "VllmWorker" in cmd:
                    victims.append(child)
            if not victims:
                time.sleep(1)
        assert victims, f"no VllmWorker process found under the engine:\n{_tail()}"

        victims[0].send_signal(signal.SIGKILL)
        victims[0].wait(timeout=30)

        assert _wait_for("died unexpectedly", 120), f"worker death was not detected within 120s:\n{_tail()}"
        assert _wait_for("shutting down executor", 30), f"executor did not shut down after the death:\n{_tail()}"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
