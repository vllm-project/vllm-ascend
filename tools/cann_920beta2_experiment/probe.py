"""在独立进程和全新缓存目录中验证 eager 与 DeepSeek 启动。"""

import argparse
import contextlib
import json
import os
import shutil
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


def probe_startup(case: dict, evidence: Path, fusion: bool, request_smoke: bool = False) -> dict:
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
    args[index] = json.dumps(config)
    # Ascend's pass manager reads additional_config, not upstream pass_config.
    index = args.index("--additional-config") + 1
    additional = json.loads(args[index])
    additional.setdefault("ascend_compilation_config", {})["fuse_norm_quant"] = fusion
    args[index] = json.dumps(additional)
    command = [sys.executable, str(Path(__file__).with_name("server.py")), "serve", case["model"], *args]
    result = {"fusion": fusion, "cache": str(cache), "command": command, "status": "timeout"}
    (evidence / f"{label}-command.json").write_text(json.dumps(result, indent=2) + "\n")
    log_path = evidence / f"{label}-server.log"
    with log_path.open("w") as log, log_path.open() as monitor:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
            recent_log = ""
            while time.monotonic() < deadline:
                recent_log += monitor.read()
                if "Segfault encountered" in recent_log or "Fatal Python error: Segmentation fault" in recent_log:
                    result["status"] = "native_crash"
                    break
                recent_log = recent_log[-128:]
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
            if result["status"] == "ready" and request_smoke:
                payload = {"model": case["model"], "prompt": "Hello", "max_tokens": 8, "temperature": 0}
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/completions",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=180) as response:
                        completion = json.load(response)
                    (evidence / "completion-smoke.json").write_text(json.dumps(completion, indent=2) + "\n")
                    assert completion.get("choices") and completion.get("usage", {}).get("completion_tokens", 0) > 0
                    result["completion_smoke"] = "passed"
                except (urllib.error.URLError, TimeoutError, ValueError, AssertionError) as error:
                    result.update(status="request_failed", completion_error=str(error))
        finally:
            # All children belong to this one experiment server's process group.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=30)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    diagnostics = cache / "diagnostics"
    if diagnostics.exists():
        shutil.copytree(diagnostics, evidence / f"{label}-diagnostics", dirs_exist_ok=True)
    if result["status"] == "ready" and env.get("ARDQ_EXPECTED_VENDOR"):
        expected_workers = int(args[args.index("--tensor-parallel-size") + 1]) * int(
            args[args.index("--data-parallel-size") + 1]
        )
        configs = [json.loads(p.read_text()) for p in diagnostics.glob("fusion-config-*.json")]
        successes = [json.loads(p.read_text()) for p in diagnostics.glob("first-success-*.json")]
        beta_successes = [
            r
            for r in successes
            if r["kwargs"].get("beta") is not None or (len(r["args"]) > 5 and r["args"][5] is not None)
        ]
        result["fixed_workers"] = len(beta_successes)
        verified = (
            len(configs) >= expected_workers
            and len(beta_successes) >= expected_workers
            and all(r["actual_fuse_norm_quant"] and "AddRMSNormQuantFusionPass" in r["passes"] for r in configs)
            and all(r.get("fixed_libraries") for r in beta_successes)
        )
        if not verified:
            result["status"] = "fixed_library_or_fusion_evidence_missing"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--fusion-on-only", action="store_true")
    parser.add_argument("--request-smoke", action="store_true")
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--minimal-only", action="store_true")
    phases.add_argument("--startup-only", action="store_true")
    args = parser.parse_args()
    if args.eager:
        eager_probe(args.evidence)
        return
    config_path = args.baseline / "tests/e2e/nightly/single_node/models/configs/DeepSeek-V3.2-W8A8.yaml"
    case = yaml.safe_load(config_path.read_text())["test_cases"][0]
    if args.startup_only:
        result_path = args.evidence / "result.json"
        results = json.loads(result_path.read_text()) if result_path.exists() else {"startup": []}
        for fusion in (True,) if args.fusion_on_only else (True, False):
            results["startup"].append(probe_startup(case, args.evidence, fusion, args.request_smoke))
            (args.evidence / "result.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2), flush=True)
        if any(item["status"] != "ready" for item in results["startup"]):
            raise SystemExit(1)
        return
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
    results = {"eager_returncode": eager.returncode, "minimal": [], "startup": []}
    # Each mode gets a new process: native crashes must not stop the comparisons.
    for mode in ("eager-inference", "compile", "compile-no-smooth", "compile-init0"):
        minimal_env = eager_env.copy()
        minimal_env["VLLM_CACHE_ROOT"] = tempfile.mkdtemp(prefix=f"cann920-{mode}-", dir="/tmp")
        if mode == "compile-init0":
            minimal_env["ACL_OP_INIT_MODE"] = "0"
        with (args.evidence / f"minimal-{mode}.log").open("w") as log:
            try:
                child = subprocess.run(
                    [sys.executable, str(Path(__file__).with_name("minimal.py")), mode, str(args.evidence)],
                    env=minimal_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                    check=False,
                )
                result = {"mode": mode, "returncode": child.returncode}
            except subprocess.TimeoutExpired:
                result = {"mode": mode, "status": "timeout"}
        results["minimal"].append(result)
        (args.evidence / "result.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.minimal_only:
        print(json.dumps(results, indent=2), flush=True)
        return
    for fusion in (True, False):
        results["startup"].append(probe_startup(case, args.evidence, fusion))
        (args.evidence / "result.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    if eager.returncode or any(item["status"] != "ready" for item in results["startup"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
