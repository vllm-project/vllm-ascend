# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""The launcher must preserve device assignment and paired source precedence."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("tp,devices", [("1", "5"), ("4", "0,1,2,3")])
def test_startlux_launcher_preserves_arguments(tmp_path, tp, devices):
    launcher = Path(__file__).resolve().parents[2] / "examples/models/startlux_decision/serve.sh"
    capture = tmp_path / "capture"
    capture.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" "$ASCEND_RT_VISIBLE_DEVICES" '
        '"$VLLM_ENABLE_V1_MULTIPROCESSING" "$PYTHONPATH"\n'
    )
    capture.chmod(0o755)
    env = dict(
        os.environ,
        MODEL_PATH="/models/decision checkpoint",
        VLLM_SOURCE="/source/vllm",
        STARTLUX_SOURCE="/source/renderer",
        ASCEND_RT_VISIBLE_DEVICES=devices,
        TENSOR_PARALLEL_SIZE=tp,
        PYTHON_BIN=str(capture),
        PYTHONPATH="/bindings",
        PORT="18000",
        MAX_MODEL_LEN="4096",
    )
    env.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
    run = subprocess.run(
        ["bash", str(launcher), "--enforce-eager"], env=env, capture_output=True, text=True, check=True
    )
    args = run.stdout.splitlines()
    assert args[:10] == [
        "/source/vllm/examples/pooling/classify/serve_startlux_decision.py",
        "--model",
        "/models/decision checkpoint",
        "--tensor-parallel-size",
        tp,
        "--max-model-len",
        "4096",
        "--port",
        "18000",
        "--enforce-eager",
    ]
    assert args[10:12] == [devices, "0"]
    assert args[12] == f"/source/vllm:{launcher.parents[3]}:/source/renderer:/bindings"


def test_startlux_launcher_requires_explicit_devices():
    launcher = Path(__file__).resolve().parents[2] / "examples/models/startlux_decision/serve.sh"
    env = dict(os.environ, MODEL_PATH="/model", VLLM_SOURCE="/vllm", STARTLUX_SOURCE="/renderer")
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    run = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
    assert run.returncode != 0
    assert "Select the assigned NPU devices" in run.stderr


@pytest.mark.parametrize(
    "execution",
    [
        ["--enforce-eager"],
        ["--compilation-config", '{"mode":3,"cudagraph_mode":"PIECEWISE","cudagraph_allow_padding":false}'],
    ],
)
def test_instant_launcher_preserves_mode_and_raw_logits(tmp_path, execution):
    launcher = Path(__file__).resolve().parents[2] / "examples/models/startlux_decision/serve_instant.sh"
    capture = tmp_path / "capture"
    capture.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    capture.chmod(0o755)
    env = dict(
        os.environ,
        MODEL_PATH="/models/Instant checkpoint",
        ASCEND_RT_VISIBLE_DEVICES="0",
        PYTHON_BIN=str(capture),
        TENSOR_PARALLEL_SIZE="1",
    )
    result = subprocess.run(["bash", str(launcher), *execution], env=env, text=True, capture_output=True, check=True)
    args = result.stdout.splitlines()
    assert args[:4] == ["-m", "vllm.entrypoints.cli.main", "serve", "/models/Instant checkpoint"]
    assert args[args.index("--pooler-config") + 1] == '{"task":"classify","use_activation":false}'
    # Bidirectional pooling must see the complete prompt, even in graph mode.
    assert "--no-enable-chunked-prefill" in args
    assert "--enable-chunked-prefill" not in args
    assert args[args.index("--max-num-batched-tokens") + 1] == args[args.index("--max-model-len") + 1]
    assert args[-6 - len(execution) : -6] == execution


@pytest.mark.parametrize(
    "override",
    [
        ["--enable-chunked-prefill"],
        ["--enable-chunked-prefill=true"],
        ["--enable-prefix-caching"],
        ["--enable-prefix-caching=true"],
        ["--max-model-len", "16384"],
        ["--max-num-batched-tokens=128"],
        ["--enable_chunked_prefill"],
        ["--enable-chunked"],
    ],
)
def test_instant_launcher_rejects_incomplete_prompt_overrides(tmp_path, override):
    """Reject incompatible overrides before invoking the model process."""
    launcher = Path(__file__).resolve().parents[2] / "examples/models/startlux_decision/serve_instant.sh"
    invoked = tmp_path / "invoked"
    capture = tmp_path / "capture"
    capture.write_text('#!/bin/bash\ntouch "$INVOKED_MARKER"\n')
    capture.chmod(0o755)
    env = dict(
        os.environ,
        MODEL_PATH="/models/Instant checkpoint",
        ASCEND_RT_VISIBLE_DEVICES="0",
        PYTHON_BIN=str(capture),
        INVOKED_MARKER=str(invoked),
    )
    result = subprocess.run(["bash", str(launcher), *override], env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "Instant requires a complete prompt" in result.stderr
    assert not invoked.exists()
