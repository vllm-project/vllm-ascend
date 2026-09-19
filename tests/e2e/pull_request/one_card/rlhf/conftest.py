#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/tests/entrypoints/serve/dev/rlhf/conftest.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Shared fixtures and helpers for RLHF end-to-end tests.

This module provides the server harness, HTTP helpers, NPU memory query
utilities, and common fixtures used across all RL test files under
``tests/e2e/pull_request/rlhf/``.
"""

import contextlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Model / server defaults
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B")

# Lightweight args for state-machine / protocol tests that don't need real
# weights (avoids spending time downloading a 1B checkpoint in T0 tests).
_DUMMY_ARGS = [
    "--dtype",
    "bfloat16",
    "--max-model-len",
    "128",
    "--max-num-seqs",
    "8",
    "--gpu-memory-utilization",
    "0.5",
    "--enable-sleep-mode",
    "--enforce-eager",
    "--load-format",
    "dummy",
]


# ---------------------------------------------------------------------------
# Sleep/wake model profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SleepWakeModelProfile:
    """A single-card deployment profile for the sleep/wake precision tests.

    The protocol-level tests only need the tiny Qwen3-0.6B default, but the
    sleep -> wake weight-restore path is exactly where large production models
    (MoE, MLA/DSA sparse attention, hybrid linear attention, quantized
    checkpoints) tend to break.  Each profile therefore describes how to bring
    one such model up on a *single* NPU card so that ``TestOutputCorrectness``
    and ``TestLogprobsPrecision`` can guard it.

    Two ways of shrinking a model onto one card are supported:

    * ``model`` points at a checkpoint that has already been layer-reduced on
      disk.  Real weights are preferred because sleep/wake then also has to
      restore the real (de)quantization metadata; DeepSeek-V4-Flash and
      Qwen3.5-35B-A3B are staged this way.  Producing such a checkpoint only
      requires dropping every ``layers.<i>`` tensor with ``i >= N`` (plus the
      draft/MTP tensors, which a non-speculative server never loads) and
      setting ``num_hidden_layers = N`` in ``config.json``.
    * ``load_format="dummy"`` together with ``hf_overrides`` reducing
      ``num_hidden_layers``.  This keeps only ``config.json`` + tokenizer on
      disk and is the only workable recipe when the checkpoint cannot be
      staged at all, e.g. GLM-5.2 whose BF16 export is ~1 TB.  Note that a
      config-only layer reduction is *invalid* with real weights: vLLM rejects
      any checkpoint tensor whose module the config no longer declares
      ("There is no module or parameter named 'layers.N'"), so real-weight
      profiles must reduce the checkpoint itself.

    Attributes:
        name: pytest id reported for this profile.
        model: Default model directory or hub id.
        env_var: Environment variable overriding ``model`` with a staged
            checkpoint directory.
        extra_args: Extra server CLI flags appended after the shared base flags.
        hf_overrides: Value forwarded through ``--hf-overrides``.
        load_format: Value forwarded through ``--load-format``.  ``"dummy"``
            also means no weight files are needed, only the model config.
        max_model_len: ``--max-model-len`` for the served engine.
        max_num_seqs: ``--max-num-seqs`` for the served engine.
        gpu_memory_utilization: ``--gpu-memory-utilization`` for the engine.
        requires_local_path: When True the profile is skipped unless ``model``
            resolves to an existing local directory.  The default Qwen3-0.6B
            profile sets this to False so CI can fetch it from the hub.
    """

    name: str
    model: str
    env_var: str
    extra_args: tuple[str, ...] = ()
    hf_overrides: dict | None = None
    load_format: str | None = None
    max_model_len: int = 2048
    max_num_seqs: int = 32
    gpu_memory_utilization: float = 0.75
    requires_local_path: bool = True

    @property
    def resolved_model(self) -> str:
        """Model path/id actually served for this profile."""
        return os.environ.get(self.env_var, self.model)

    @property
    def weights_required(self) -> bool:
        """True unless dummy weights make the checkpoint files unnecessary."""
        return self.load_format != "dummy"

    def unavailable_reason(self) -> str | None:
        """Return why this profile cannot run here, or None when it can.

        Large checkpoints are staged per machine, so a profile whose weights
        are missing is skipped instead of failing: upstream CI, which only
        carries Qwen3-0.6B, keeps running the default profile unchanged.
        """
        path = Path(self.resolved_model)
        if path.is_dir():
            if not self.weights_required or _has_weight_files(path):
                return None
            return f"no weight files under {path}"
        if self.requires_local_path:
            return f"{self.resolved_model} is not staged locally; set {self.env_var}=<model dir> to enable this profile"
        return None

    def serve_args(self) -> list[str]:
        """Server CLI flags for this profile."""
        args = [
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--enable-sleep-mode",
            "--enforce-eager",
        ]
        if self.load_format is not None:
            args += ["--load-format", self.load_format]
        if self.hf_overrides is not None:
            args += ["--hf-overrides", json.dumps(self.hf_overrides)]
        args += list(self.extra_args)
        return args


def _has_weight_files(path: Path) -> bool:
    """True when ``path`` holds a sharded or monolithic checkpoint."""
    if any(path.glob("*.safetensors")):
        return True
    if any(path.glob("*.bin")):
        return True
    return (path / "model.safetensors.index.json").exists()


# The default profile is what upstream CI exercises; it must stay runnable
# everywhere.  The large-model profiles below are single-card, layer-reduced
# deployments and are skipped (with a clear reason) wherever their checkpoint
# is not staged.
DEFAULT_PROFILE = SleepWakeModelProfile(
    name="qwen3-0.6b",
    model=MODEL_NAME,
    env_var="VLLM_TEST_MODEL",
    requires_local_path=False,
)

# DeepSeek-V4-Flash: 3 of its 44 decoder layers (~39 GiB of real BF16 weights)
# fit a single 64 GiB card, and layer 2 is a Compress-4 layer, so the DeepSeek
# sparse attention + compressor/indexer caches stay under test.  The upstream
# export keeps 4 layers (~51 GiB), which no longer fits once the MoE weights
# are re-laid-out during loading.
DSV4_FLASH_PROFILE = SleepWakeModelProfile(
    name="deepseek-v4-flash-3l",
    model="RedHatAI/DeepSeek-V4-Flash-BF16",
    env_var="VLLM_TEST_SLEEP_DSV4_FLASH_MODEL",
    extra_args=("--tokenizer-mode", "deepseek_v4", "--block-size", "32"),
    max_num_seqs=8,
    gpu_memory_utilization=0.9,
)

# Qwen3.5-35B-A3B: MoE with hybrid GDN linear attention and a full-attention
# layer every 4 layers, so 8 kept layers still contain 2 full-attention layers.
QWEN35_35B_PROFILE = SleepWakeModelProfile(
    name="qwen3.5-35b-a3b-8l",
    model="Qwen/Qwen3.5-35B-A3B",
    env_var="VLLM_TEST_SLEEP_QWEN35_35B_MODEL",
    max_num_seqs=8,
    gpu_memory_utilization=0.85,
)

# GLM-5.2: the BF16 checkpoint is ~1 TB and its w4a8/w8a8c8 exports still need
# several nodes, so only config + tokenizer are staged and the layers are
# dummy-initialised.  Sleep/wake still offloads and remaps the real parameter
# tensors, which is what this profile guards.
GLM52_PROFILE = SleepWakeModelProfile(
    name="glm-5.2-4l-dummy",
    model="zai-org/GLM-5.2",
    env_var="VLLM_TEST_SLEEP_GLM52_MODEL",
    hf_overrides={"text_config": {"num_hidden_layers": 4}},
    load_format="dummy",
    max_num_seqs=8,
    gpu_memory_utilization=0.85,
)

# Profiles exercised by the sleep/wake precision tests, in report order.
SLEEP_WAKE_PRECISION_PROFILES = (
    DEFAULT_PROFILE,
    DSV4_FLASH_PROFILE,
    QWEN35_35B_PROFILE,
    GLM52_PROFILE,
)


# ---------------------------------------------------------------------------
# Server harness
# ---------------------------------------------------------------------------


@contextmanager
def server(
    extra_args=None,
    port: int = 8770,
    timeout: float = 180.0,
    dummy_weights: bool = False,
    profile: SleepWakeModelProfile | None = None,
):
    """Launch a vLLM server with the dev router; yield its base URL.

    Args:
        extra_args:      Additional CLI flags appended after the base args.
        port:            HTTP port to bind (caller is responsible for uniqueness).
        timeout:         Seconds to wait for /health before giving up.
        dummy_weights:   If True, use --load-format dummy (fast, no real weights).
        profile:         Model profile to serve.  Defaults to the Qwen3-0.6B
                         profile used by the protocol-level tests.
    """
    profile = profile or DEFAULT_PROFILE
    env = {
        **os.environ,
        "VLLM_SERVER_DEV_MODE": "1",
        "HF_HUB_OFFLINE": "1",
    }
    if dummy_weights:
        base = _DUMMY_ARGS
    else:
        base = profile.serve_args()
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        profile.resolved_model,
        "--port",
        str(port),
        "--served-model-name",
        "m",
        "--additional-config",
        '{"weight_nz_mode": 0}',
        *(base + (extra_args or [])),
    ]
    # Establish the test process's NPU device context while the card is still
    # idle, before the server reserves most of device memory. Otherwise the
    # first in-process mem_get_info() in npu_free_bytes() would cold-init the
    # device against a running server — the pattern that made the old
    # subprocess helper time out on CI.
    import torch

    torch.npu.mem_get_info(0)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    url = f"http://localhost:{port}"
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read(4000).decode(errors="replace") if proc.stderr else ""
                raise RuntimeError(f"vllm server exited during startup:\n{err}")
            with contextlib.suppress(Exception):
                if requests.get(f"{url}/health", timeout=3).status_code == 200:
                    break
            time.sleep(1)
        else:
            proc.terminate()
            raise RuntimeError("vllm server did not start in time")
        yield url
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        if proc.poll() is None:
            proc.kill()


# ---------------------------------------------------------------------------
# Polling helper (200-lie workaround)
# ---------------------------------------------------------------------------


def poll_until(
    predicate: Callable[[], bool],
    timeout: float = 10.0,
    interval: float = 0.5,
) -> bool:
    """Poll predicate() until it returns True or timeout expires.

    Workaround for the vLLM sleep/wake "200-lie" — the HTTP endpoints may
    return 200 before the underlying operation is complete, so callers that
    need to verify state *after* an operation can use this helper instead of
    assuming the 200 means completion.

    Returns True if predicate became true within timeout, False otherwise.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# HTTP helpers — generation
# ---------------------------------------------------------------------------


def gen(url, prompt="The capital of France is", max_tokens=8, timeout=30):
    """Fire a /v1/completions request; return JSON or None on any error."""
    try:
        r = requests.post(
            f"{url}/v1/completions",
            json={
                "model": "m",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=timeout,
        )
        return r.json()
    except Exception:
        return None


def gen_with_logprobs(url, prompt="The capital of France is", max_tokens=8, logprobs=5, timeout=30):
    """Fire a /v1/completions request with logprobs; return JSON or None."""
    try:
        r = requests.post(
            f"{url}/v1/completions",
            json={
                "model": "m",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "logprobs": logprobs,
            },
            timeout=timeout,
        )
        return r.json()
    except Exception:
        return None


def ok(resp) -> bool:
    """True iff resp is a successful completion (has choices, no error key)."""
    return resp is not None and "choices" in resp and bool(resp["choices"]) and "error" not in resp


# ---------------------------------------------------------------------------
# HTTP helpers — sleep / wake / pause / resume
# ---------------------------------------------------------------------------


def sleep(url, level=1, mode="abort"):
    return requests.post(f"{url}/sleep", params={"level": level, "mode": mode}, timeout=15).status_code


def wake(url, tags=None):
    params = {"tags": tags} if tags else {}
    return requests.post(f"{url}/wake_up", params=params, timeout=20).status_code


def pause(url, mode="abort", clear_cache=True):
    return requests.post(
        f"{url}/pause",
        params={"mode": mode, "clear_cache": clear_cache},
        timeout=60,
    ).status_code


def resume(url):
    return requests.post(f"{url}/resume", timeout=10).status_code


def is_sleeping(url) -> bool:
    return requests.get(f"{url}/is_sleeping", timeout=5).json()["is_sleeping"]


def is_paused(url) -> bool:
    return requests.get(f"{url}/is_paused", timeout=5).json()["is_paused"]


def health(url) -> int:
    try:
        return requests.get(f"{url}/health", timeout=5).status_code
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# HTTP helpers — weight transfer
# ---------------------------------------------------------------------------


def start_weight_update(url, is_checkpoint_format=True):
    return requests.post(
        f"{url}/start_weight_update",
        json={"is_checkpoint_format": is_checkpoint_format},
        timeout=10,
    )


def finish_weight_update(url):
    return requests.post(f"{url}/finish_weight_update", timeout=10)


def get_world_size(url, include_dp=True):
    return requests.get(
        f"{url}/get_world_size",
        params={"include_dp": include_dp},
        timeout=5,
    )


# ---------------------------------------------------------------------------
# NPU / metrics helpers
# ---------------------------------------------------------------------------


def npu_free_bytes(device: int = 0) -> int:
    """Read NPU free bytes in-process (device-wide free memory).

    Do NOT shell out to a fresh python subprocess as the upstream CUDA test
    does: on Ascend a cold ``import torch`` + device-context init in a child
    process can exceed a 10s timeout while the vLLM server holds the same card
    busy (torch_npu init is far heavier than CUDA's). ``mem_get_info`` reports
    device-wide free bytes, so an in-process query returns the same value
    without paying subprocess startup cost. The device context is warmed up in
    ``server()`` while the card is still idle.
    """
    import torch

    return int(torch.npu.mem_get_info(device)[0])


def sleep_metrics(url):
    """Return (awake, weights_offloaded, discard_all) from /metrics."""
    try:
        from prometheus_client.parser import text_string_to_metric_families
    except ImportError:
        return None, None, None

    r = requests.get(f"{url}/metrics", timeout=5)
    vals: dict = {}
    for family in text_string_to_metric_families(r.text):
        if family.name == "vllm:engine_sleep_state":
            for s in family.samples:
                vals[s.labels.get("sleep_state", "")] = s.value
    return vals.get("awake"), vals.get("weights_offloaded"), vals.get("discard_all")
