"""在独立进程和全新缓存目录中验证 eager 与 DeepSeek 启动。"""

import argparse
import contextlib
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
from pathlib import Path

import yaml

STARTUP_TIMEOUT_SECONDS = 3000
POLL_SECONDS = 5


def eager_probe(evidence: Path) -> None:
    # Import the backend only in the isolated eager process.
    import torch
    import torch_npu

    torch.npu.set_device(0)
    torch.manual_seed(20260905)
    print(torch.__version__, torch_npu.__version__, flush=True)
    for rows in (1, 48, 4096):
        x1 = torch.randn(rows, 7168, dtype=torch.bfloat16).npu()
        x2 = torch.randn_like(x1)
        gamma = torch.ones(7168, dtype=torch.bfloat16).npu()
        outputs = torch.ops.npu.npu_add_rms_norm_dynamic_quant(x1, x2, gamma, epsilon=1e-6, output_mask=[True, False])
        torch.npu.synchronize()
        print(json.dumps({"rows": rows, "outputs": [list(t.shape) for t in outputs]}), flush=True)
    mappings = Path("/proc/self/maps").read_text()
    (evidence / "eager-loaded-libraries.txt").write_text(
        "\n".join(line for line in mappings.splitlines() if "Ascend" in line or "cann-" in line) + "\n"
    )


def probe_startup(case: dict, evidence: Path, fusion: bool) -> dict:
    label = "fusion-on" if fusion else "fusion-off"
    cache = Path(tempfile.mkdtemp(prefix=f"cann920-{label}-", dir="/tmp"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in case["envs"].items()})
    env["SERVER_PORT"] = str(port)
    env["VLLM_CACHE_ROOT"] = str(cache)
    args = [str(arg).replace("$SERVER_PORT", str(port)) for arg in case["server_cmd"]]
    index = args.index("--compilation-config") + 1
    config = json.loads(args[index])
    config["cache_dir"] = str(cache / "torch_compile_cache")
    config.setdefault("pass_config", {})["fuse_norm_quant"] = fusion
    args[index] = json.dumps(config)
    command = ["vllm", "serve", case["model"], *args]
    result = {"fusion": fusion, "cache": str(cache), "command": command, "status": "timeout"}
    (evidence / f"{label}-command.json").write_text(json.dumps(result, indent=2) + "\n")
    with (evidence / f"{label}-server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    result.update(status="process_exited", returncode=process.returncode)
                    break
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                        if response.status == 200:
                            result["status"] = "ready"
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(POLL_SECONDS)
        finally:
            # All children belong to this one experiment server's process group.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=30)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    if args.eager:
        eager_probe(args.evidence)
        return
    config_path = args.baseline / "tests/e2e/nightly/single_node/models/configs/DeepSeek-V3.2-W8A8.yaml"
    case = yaml.safe_load(config_path.read_text())["test_cases"][0]
    eager_env = os.environ.copy()
    eager_env.update({str(key): str(value) for key, value in case["envs"].items()})
    with (args.evidence / "eager.log").open("w") as log:
        eager = subprocess.run(
            [sys.executable, __file__, "--eager", "--evidence", str(args.evidence)],
            env=eager_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
    results = {"eager_returncode": eager.returncode, "startup": []}
    for fusion in (True, False):
        results["startup"].append(probe_startup(case, args.evidence, fusion))
        (args.evidence / "result.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    if eager.returncode or any(item["status"] != "ready" for item in results["startup"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
