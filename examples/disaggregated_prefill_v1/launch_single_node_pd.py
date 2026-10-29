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

Orchestrates a Prefill engine, a Decode engine, and a load-balance proxy
on a single Atlas 800I A3 (or A2) node.  Each engine runs as an independent
``vllm serve`` subprocess with MooncakeConnectorV1 for KV-cache transfer.

Usage (Qwen3-30B-A3B W8A8 on A3, 1P1D):

    python launch_single_node_pd.py \\
        --model /model/Qwen3-30B-A3B-w8a8 \\
        --served-model-name qwen3 \\
        --prefill-npus 0,1 \\
        --decode-npus 2,3 \\
        --tp-size 1 \\
        --quantization ascend \\
        --additional-config '{"weight_nz_mode": 2, "enable_flashcomm1": true}'

The script:
  1. Starts the Prefill engine (kv_producer) on --prefill-npus.
  2. Starts the Decode engine (kv_consumer) on --decode-npus.
  3. Waits for both engines to be healthy.
  4. Starts the load-balance proxy.
  5. Forwards Ctrl-C to shut everything down cleanly.

Requires Mooncake to be installed (https://github.com/kvcache-ai/Mooncake).
"""

from __future__ import annotations

import argparse
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
    "VLLM_LOGGING_LEVEL": "INFO",
    "OMP_PROC_BIND": "false",
    "OMP_NUM_THREADS": "10",
}


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
    model: str
    served_model_name: str
    max_model_len: int = 32768
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 100
    gpu_memory_utilization: float = 0.9
    quantization: str | None = None
    trust_remote_code: bool = True
    enable_prefix_caching: bool = False
    distributed_executor_backend: str = "mp"
    compilation_config: str = '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
    additional_config: str | None = None
    enable_expert_parallel: bool = False
    hf_overrides: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)

    @property
    def kv_connector(self) -> str:
        return "MooncakeConnectorV1"

    def build_kv_transfer_config(self) -> str:
        return json.dumps(
            {
                "kv_connector": self.kv_connector,
                "kv_role": self.kv_role,
                "kv_port": self.kv_port,
                "kv_buffer_device": "npu",
                "kv_connector_extra_config": {
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
            "--served-model-name",
            self.served_model_name,
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
            "--distributed_executor_backend",
            self.distributed_executor_backend,
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
        ld = env.get("LD_LIBRARY_PATH", "")
        try:
            import mooncake
            mooncake_ld = os.path.dirname(mooncake.__file__)
        except ImportError:
            mooncake_ld = "/usr/local/1ib64/python3.12/site-packages/mooncake"
        env["LD_LIBRARY_PATH"] = f"{mooncake_ld}:{ld}" if ld else mooncake_ld


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
        except Exception:
            pass
        time.sleep(interval)
    return False


def launch_engine(config: EngineConfig, log_file: str) -> subprocess.Popen:
    """Launch one vLLM serve subprocess and return its process handle."""
    cmd = config.build_command()
    env = config.build_env()
    print(f"[{config.role}] Launching: {' '.join(cmd)}")
    print(f"[{config.role}] ASCEND_RT_VISIBLE_DEVICES={config.visible_devices}")
    print(f"[{config.role}] Log: {log_file}")
    with open(log_file, "w") as log_handle:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
    return proc


def launch_proxy(
    script: str,
    host: str,
    port: int,
    prefill_host: str,
    prefill_port: int,
    decode_host: str,
    decode_port: int,
    log_file: str,
) -> subprocess.Popen:
    """Launch the load-balance proxy subprocess."""
    cmd = [
        sys.executable, script,
        "--host",
        host,
        "--port",
        str(port),
        "--prefiller-hosts",
        prefill_host,
        "--prefiller-ports",
        str(prefill_port),
        "--decoder-hosts",
        decode_host,
        "--decoder-ports",
        str(decode_port),
    ]
    print(f"[proxy] Launching: {' '.join(cmd)}")
    print(f"[proxy] Log: {log_file}")
    log_handle = open(log_file, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    return proc


def terminate_process(proc: subprocess.Popen, name: str) -> None:
    """Gracefully terminate a process group."""
    if proc.poll() is not None:
        return
    print(f"[{name}] Terminating...")
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=15)
    except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, OSError):
            pass
    print(f"[{name}] Stopped.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-node PD disaggregation launcher for vLLM-Ascend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    p.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    p.add_argument("--served-model-name", default="qwen3", help="Served model name")
    p.add_argument("--quantization", default=None, help="Quantization method (e.g. ascend)")
    p.add_argument("--enable-expert-parallel", action="store_true", help="Enable expert parallelism (MoE)")

    # Parallelism
    p.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size per engine")
    p.add_argument("--prefill-npus", required=True,help="Comma-separated NPU chip IDs for prefill (e.g. '0,1')")
    p.add_argument("--decode-npus", required=True,help="Comma-separated NPU chip IDs for decode (e.g. '2,3')")

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
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--max-num-seqs", type=int, default=100)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)

    # Extra configs
    p.add_argument(
        "--additional-config", default=None,help='JSON string for --additional-config (e.g. \'{"weight_nz_mode": 2}\')'
    )
    p.add_argument(
        "--compilation-config",default='{"cudagraph_mode": "FULL_DECODE_ONLY"}',help="JSON string for --compilation-config"
    )
    p.add_argument("--hf-overrides", default=None, help="JSON string for --hf-overrides")
    p.add_argument("--no-trust-remote-code", action="store_true")

    # Proxy
    p.add_argument("--proxy-script", default=DEFAULT_PROXY_SCRIPT,help="Path to load_balance_proxy_server_example.py")
    p.add_argument("--no-proxy", action="store_true",help="Skip launching the proxy (engines only)")

    # Startup
    p.add_argument("--startup-timeout", type=int, default=600,help="Seconds to wait for each engine to become healthy")
    p.add_argument("--log-dir", default="/tmp/vllm_pd_logs",help="Directory for engine and proxy log files")

    return p.parse_args()


def main():
    args = parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    common_kwargs = dict(
        model=args.model,
        served_model_name=args.served_model_name,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
        trust_remote_code=not args.no_trust_remote_code,
        distributed_executor_backend="mp",
        compilation_config=args.compilation_config,
        additional_config=args.additional_config,
        enable_expert_parallel=args.enable_expert_parallel,
        hf_overrides=args.hf_overrides,
    )

    prefill_cfg = EngineConfig(
        role="prefill",
        visible_devices=args.prefill_npus,
        host=args.host,
        port=args.prefill_port,
        tp_size=args.tp_size,
        kv_role="kv_producer",
        kv_port=args.prefill_kv_port,
        **common_kwargs,
    )

    decode_cfg = EngineConfig(
        role="decode",
        visible_devices=args.decode_npus,
        host=args.host,
        port=args.decode_port,
        tp_size=args.tp_size,
        kv_role="kv_consumer",
        kv_port=args.decode_kv_port,
        **common_kwargs,
    )

    processes: list[tuple[subprocess.Popen, str]] = []

    def cleanup(*_):
        print("\n[launcher] Shutting down all processes...")
        for proc, name in reversed(processes):
            terminate_process(proc, name)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # --- Launch prefill engine ---
    prefill_proc = launch_engine(prefill_cfg, str(log_dir / "prefill.log"))
    processes.append((prefill_proc, "prefill"))

    # --- Launch decode engine ---
    time.sleep(2)
    decode_proc = launch_engine(decode_cfg, str(log_dir / "decode.log"))
    processes.append((decode_proc, "decode"))

    # --- Wait for both engines to be healthy ---
    print(f"\n[launcher] Waiting for prefill engine on port {args.prefill_port}...")
    if not wait_for_http(f"http://127.0.0.1:{args.prefill_port}/health", timeout=args.startup_timeout):
        print("[launcher] ERROR: Prefill engine did not become healthy in time.")
        print(f"  Check log: {log_dir / 'prefill.log'}")
        cleanup()
    print("[launcher] Prefill engine is healthy.")

    print(f"[launcher] Waiting for decode engine on port {args.decode_port}...")
    if not wait_for_http(f"http://127.0.0.1:{args.decode_port}/health", timeout=args.startup_timeout):
        print("[launcher] ERROR: Decode engine did not become healthy in time.")
        print(f"  Check log: {log_dir / 'decode.log'}")
        cleanup()
    print("[launcher] Decode engine is healthy.")

    # --- Launch proxy ---
    if not args.no_proxy:
        proxy_host = "127.0.0.1"
        proxy_proc = launch_proxy(
            script=args.proxy_script,
            host=args.host,
            port=args.proxy_port,
            prefill_host=proxy_host,
            prefill_port=args.prefill_port,
            decode_host=proxy_host,
            decode_port=args.decode_port,
            log_file=str(log_dir / "proxy.log"),
        )
        processes.append((proxy_proc, "proxy"))

        print(f"[launcher] Waiting for proxy on port {args.proxy_port}...")
        if not wait_for_http(f"http://127.0.0.1:{args.proxy_port}/healthcheck", timeout=60):
            print("[launcher] WARNING: Proxy health check failed, but it may still work.")
        print("[launcher] Proxy is ready.")

    # --- Print usage ---
    print("\n" + "=" * 70)
    print("  Single-node PD disaggregation is ready!")
    print("=" * 70)
    if not args.no_proxy:
        print(f"\n  Proxy endpoint:  http://127.0.0.1:{args.proxy_port}/v1/chat/completions")
        print(f"  Health check:    http://127.0.0.1:{args.proxy_port}/healthcheck")
        print(f"""
  Quick test:

    curl http://127.0.0.1:{args.proxy_port}/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{{
            "model": "{args.served_model_name}",
            "messages": [{{"role": "user", "content": "Hello!"}}],
            "max_completion_tokens": 64
        }}'
""")
    else:
        print(f"\n  Prefill: http://127.0.0.1:{args.prefill_port}/v1/chat/completions")
        print(f"  Decode:  http://127.0.0.1:{args.decode_port}/v1/chat/completions")
    print(f"  Logs:    {log_dir}/")
    print("\n  Press Ctrl-C to shut down.\n")
    print("=" * 70)

    # --- Wait for processes ---
    try:
        for proc, name in processes:
            proc.wait()
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
