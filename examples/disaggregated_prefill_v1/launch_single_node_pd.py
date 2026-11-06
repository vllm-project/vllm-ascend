#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""Single-node PD disaggregation launcher for vLLM-Ascend.

Orchestrates Prefill engine(s), Decode engine(s), and a load-balance proxy
on ONE Atlas node. Each engine is an independent ``vllm serve`` subprocess;
NPU chips are partitioned per engine through ``ASCEND_RT_VISIBLE_DEVICES``.

Usage (Qwen3-30B-A3B W8A8 on A3, 1P1D, TP4+TP4):

    python launch_single_node_pd.py \
        --model vllm-ascend/Qwen3-30B-A3B-W8A8 \
        --prefill-npus 0,1,2,3 \
        --decode-npus 4,5,6,7 \
        --tp-size 4 \
        --enable-expert-parallel \
        --quantization ascend

Requires Mooncake (built with -DUSE_ASCEND_DIRECT=ON) for KV transfer.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PROXY_SCRIPT = str(Path(__file__).parent / "load_balance_proxy_server_example.py")

DEFAULT_ENV = {
    "HCCL_BUFFSIZE": "1024",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "VLLM_USE_MODELSCOPE": "True",
    "VLLM_MOONCAKE_CONNECTOR_TIMEOUT": "1800",
    "VLLM_RPC_TIMEOUT": "3600000",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "30000",
    "VLLM_LOGGING_LEVEL": "INFO",
    "OMP_PROC_BIND": "false",
    "OMP_NUM_THREADS": "1",
}

# Fallback locations scanned for libmooncake*.so (no hardcoded python version).
_MOONCAKE_FALLBACK_DIRS = [
    "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake",
    "/usr/local/lib64/python3.12/site-packages/mooncake",
    "/usr/local/lib/python3.12/dist-packages/mooncake",
]


def _mooncake_lib_path() -> str:
    """Locate the directory holding the mooncake shared libraries."""
    candidates: list[Path] = []
    try:
        import mooncake  # type: ignore[import-not-found]

        pkg = Path(mooncake.__file__).resolve().parent
        candidates.extend([pkg / "lib", pkg, pkg.parent])
    except ImportError:
        pass
    candidates.extend(Path(p) for p in _MOONCAKE_FALLBACK_DIRS)
    for cand in candidates:
        if cand.is_dir() and any(cand.glob("libmooncake*")):
            return str(cand)
    return ""


@dataclass
class EngineConfig:
    """Configuration for one PD engine (prefill or decode)."""

    role: str
    visible_devices: str
    host: str
    port: int
    tp_size: int
    kv_role: str
    kv_port: str
    engine_id: str
    model: str
    max_model_len: int = 32768
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 64
    gpu_memory_utilization: float = 0.9
    quantization: str | None = None
    trust_remote_code: bool = True
    enable_prefix_caching: bool = False
    enable_expert_parallel: bool = False
    kv_connector: str = "MooncakeConnectorV2"
    compilation_config: str | None = None  # only set for decode by default
    additional_config: str | None = None
    hf_overrides: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)

    def build_kv_transfer_config(self) -> str:
        return json.dumps(
            {
                "kv_connector": self.kv_connector,
                "kv_role": self.kv_role,
                "kv_port": self.kv_port,
                "engine_id": self.engine_id,
                "kv_connector_extra_config": {
                    "use_ascend_direct": True,
                    "prefill": {"dp_size": 1, "tp_size": self.tp_size},
                    "decode": {"dp_size": 1, "tp_size": self.tp_size},
                },
            }
        )

    def build_command(self) -> list[str]:
        cmd = [
            "vllm",
            "serve",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.tp_size),
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-batched-tokens",
            str(self.max_num_batched_tokens),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--distributed-executor-backend",
            "mp",
        ]
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if not self.enable_prefix_caching:
            cmd.append("--no-enable-prefix-caching")
        if self.quantization:
            cmd.extend(["--quantization", self.quantization])
        if self.enable_expert_parallel:
            cmd.append("--enable-expert-parallel")
        if self.compilation_config:
            cmd.extend(["--compilation-config", self.compilation_config])
        if self.additional_config:
            cmd.extend(["--additional-config", self.additional_config])
        if self.hf_overrides:
            cmd.extend(["--hf-overrides", self.hf_overrides])
        cmd.extend(["--kv-transfer-config", self.build_kv_transfer_config()])
        return cmd

    def build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(DEFAULT_ENV)
        env.update(self.extra_env)
        env["ASCEND_RT_VISIBLE_DEVICES"] = self.visible_devices
        mooncake_lib = _mooncake_lib_path()
        if mooncake_lib:
            ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{mooncake_lib}:{ld}" if ld else mooncake_lib
        return env


def wait_for_http(url: str, timeout: int = 600, interval: float = 2.0) -> bool:
    """Poll an HTTP endpoint until it responds 200 or timeout."""
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - transient errors while warming up
            pass
        time.sleep(interval)
    return False


def launch_process(cmd: list[str], env: dict[str, str] | None, log_file: str, name: str) -> subprocess.Popen:
    """Launch one subprocess with its own process group; log to file."""
    print(f"[{name}] Launching: {' '.join(cmd)}")
    print(f"[{name}] Log: {log_file}")
    with open(log_file, "w") as log_handle:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    # log_handle is closed here; the child keeps its own inherited fd
    return proc


def terminate_process(proc: subprocess.Popen, name: str) -> None:
    """Gracefully terminate a process group."""
    if proc.poll() is not None:
        return
    print(f"[{name}] Terminating...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    print(f"[{name}] Stopped.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-node PD disaggregation launcher for vLLM-Ascend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    p.add_argument("--model", required=True, help="Model path or HuggingFace/ModelScope ID")
    p.add_argument("--quantization", default=None, help="Quantization method (e.g. ascend)")
    p.add_argument("--enable-expert-parallel", action="store_true", help="Enable expert parallelism (MoE)")

    # Parallelism
    p.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size per engine")
    p.add_argument("--prefill-npus", required=True, help="Comma-separated NPU chip IDs for prefill (e.g. '0,1,2,3')")
    p.add_argument("--decode-npus", required=True, help="Comma-separated NPU chip IDs for decode (e.g. '4,5,6,7')")

    # Ports
    p.add_argument("--prefill-port", type=int, default=13700)
    p.add_argument("--decode-port", type=int, default=13701)
    p.add_argument("--proxy-port", type=int, default=8080)
    p.add_argument("--prefill-kv-port", type=str, default="30000")
    p.add_argument("--decode-kv-port", type=str, default="30100")

    # Host
    p.add_argument("--host", default="0.0.0.0", help="Bind host for engines")

    # Model serving params
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--prefill-max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--decode-max-num-batched-tokens", type=int, default=256)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)

    # Extra configs
    p.add_argument("--additional-config", default=None, help="JSON string for --additional-config")
    p.add_argument(
        "--compilation-config",
        default='{"cudagraph_mode": "FULL_DECODE_ONLY"}',
        help="JSON string for --compilation-config (decode engine only)",
    )
    p.add_argument("--hf-overrides", default=None, help="JSON string for --hf-overrides")
    p.add_argument(
        "--kv-connector",
        default="MooncakeConnectorV2",
        help="KV connector (MooncakeConnectorV2 or MooncakeConnectorV1)",
    )
    p.add_argument("--no-trust-remote-code", action="store_true")

    # Proxy
    p.add_argument("--proxy-script", default=DEFAULT_PROXY_SCRIPT, help="Path to load_balance_proxy_server_example.py")
    p.add_argument("--no-proxy", action="store_true", help="Skip launching the proxy (engines only)")

    # Startup
    p.add_argument("--startup-timeout", type=int, default=600, help="Seconds to wait for each engine")
    p.add_argument("--log-dir", default="/tmp/vllm_pd_logs", help="Directory for engine and proxy log files")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    prefill_cfg = EngineConfig(
        role="prefill",
        visible_devices=args.prefill_npus,
        host=args.host,
        port=args.prefill_port,
        tp_size=args.tp_size,
        kv_role="kv_producer",
        kv_port=args.prefill_kv_port,
        engine_id="0",
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.prefill_max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
        trust_remote_code=not args.no_trust_remote_code,
        enable_expert_parallel=args.enable_expert_parallel,
        kv_connector=args.kv_connector,
        compilation_config=None,  # cudagraph only makes sense on the decode engine
        additional_config=args.additional_config,
        hf_overrides=args.hf_overrides,
    )
    decode_cfg = EngineConfig(
        role="decode",
        visible_devices=args.decode_npus,
        host=args.host,
        port=args.decode_port,
        tp_size=args.tp_size,
        kv_role="kv_consumer",
        kv_port=args.decode_kv_port,
        engine_id="1",
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.decode_max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
        trust_remote_code=not args.no_trust_remote_code,
        enable_expert_parallel=args.enable_expert_parallel,
        kv_connector=args.kv_connector,
        compilation_config=args.compilation_config,
        additional_config=args.additional_config,
        hf_overrides=args.hf_overrides,
    )

    processes: list[tuple[subprocess.Popen, str]] = []

    def _raise_interrupt(*_args: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _raise_interrupt)
    signal.signal(signal.SIGTERM, _raise_interrupt)

    try:
        processes.append(
            (
                launch_process(
                    prefill_cfg.build_command(), prefill_cfg.build_env(), str(log_dir / "prefill.log"), "prefill"
                ),
                "prefill",
            )
        )
        time.sleep(2)
        processes.append(
            (
                launch_process(
                    decode_cfg.build_command(), decode_cfg.build_env(), str(log_dir / "decode.log"), "decode"
                ),
                "decode",
            )
        )

        print(f"\n[launcher] Waiting for prefill engine on port {args.prefill_port}...")
        if not wait_for_http(f"http://127.0.0.1:{args.prefill_port}/health", timeout=args.startup_timeout):
            print(f"[launcher] ERROR: prefill engine did not become healthy. Check {log_dir / 'prefill.log'}")
            return 1
        print("[launcher] Prefill engine is healthy.")

        print(f"[launcher] Waiting for decode engine on port {args.decode_port}...")
        if not wait_for_http(f"http://127.0.0.1:{args.decode_port}/health", timeout=args.startup_timeout):
            print(f"[launcher] ERROR: decode engine did not become healthy. Check {log_dir / 'decode.log'}")
            return 1
        print("[launcher] Decode engine is healthy.")

        if not args.no_proxy:
            proxy_cmd = [
                sys.executable,
                args.proxy_script,
                "--host",
                "0.0.0.0",
                "--port",
                str(args.proxy_port),
                "--prefiller-hosts",
                "127.0.0.1",
                "--prefiller-ports",
                str(args.prefill_port),
                "--decoder-hosts",
                "127.0.0.1",
                "--decoder-ports",
                str(args.decode_port),
            ]
            processes.append((launch_process(proxy_cmd, None, str(log_dir / "proxy.log"), "proxy"), "proxy"))
            print(f"[launcher] Waiting for proxy on port {args.proxy_port}...")
            if not wait_for_http(f"http://127.0.0.1:{args.proxy_port}/healthcheck", timeout=60):
                print("[launcher] WARNING: proxy health check failed, but it may still work.")

        print("\n" + "=" * 70)
        print("  Single-node PD disaggregation is ready!")
        print("=" * 70)
        if not args.no_proxy:
            print(f"\n  Proxy endpoint:  http://127.0.0.1:{args.proxy_port}/v1/chat/completions")
            print(f"  Health check:    http://127.0.0.1:{args.proxy_port}/healthcheck")
            print(
                f"\n  Quick test:\n\n"
                f"    curl http://127.0.0.1:{args.proxy_port}/v1/chat/completions \\\n"
                f'        -H "Content-Type: application/json" \\\n'
                f'        -d \'{{"model": "{args.model}",'
                f' "messages": [{{"role": "user", "content": "Hello!"}}],'
                f' "max_completion_tokens": 64}}\'\n'
            )
        else:
            print(f"\n  Prefill: http://127.0.0.1:{args.prefill_port}/v1/chat/completions")
            print(f"  Decode:  http://127.0.0.1:{args.decode_port}/v1/chat/completions")
        print(f"  Logs:    {log_dir}/")
        print("\n  Press Ctrl-C to shut down.")
        print("=" * 70)

        # Wait until any process exits (a healthy setup runs until Ctrl-C).
        while True:
            for proc, name in processes:
                code = proc.poll()
                if code is not None:
                    print(f"[launcher] {name} exited with code {code}.")
                    return 0 if code == 0 else 1
            time.sleep(2)
    except KeyboardInterrupt:
        return 0
    finally:
        print("\n[launcher] Shutting down all processes...")
        for proc, name in reversed(processes):
            terminate_process(proc, name)


if __name__ == "__main__":
    sys.exit(main())
