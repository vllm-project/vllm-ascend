"""确认同次 A3 CI 中原包仍崩溃、修复包的带 beta 调用可以执行。"""

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path


def child(variant: str, mode: str, evidence: Path) -> None:
    import torch
    import torch_npu
    from diagnostic_hooks import check_fixed_libraries, install_trace

    install_trace(evidence / f"{variant}-{mode}-diagnostics")
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(20260907)
    hidden = 4 if variant == "stock" else 7168
    gamma = torch.nn.Parameter(torch.ones(hidden, dtype=torch.bfloat16, device="npu"))

    def forward(x1, x2, gamma, beta):
        return torch.ops.npu.npu_add_rms_norm_dynamic_quant(
            x1, x2, gamma, beta=beta, epsilon=1e-6, output_mask=[True, False]
        )

    if mode == "compile":
        import npugraph_ex as nge

        forward = torch.compile(forward, backend=nge.get_npu_backend(), fullgraph=True, dynamic=True)
    print(torch.__version__, torch_npu.__version__, variant, mode, flush=True)
    with torch.inference_mode():
        for rows in (1,) if variant == "stock" else (4096, 48, 1):
            x1 = torch.randn(rows, hidden, dtype=torch.bfloat16, device="npu")
            x2 = torch.randn_like(x1)
            beta = torch.randn(hidden, dtype=torch.bfloat16, device="npu")
            print(json.dumps({"stage": "before", "rows": rows, "beta": True}), flush=True)
            result = forward(x1, x2, gamma, beta)
            torch.npu.synchronize()
            assert all(torch.isfinite(result[i].cpu()).all().item() for i in (0, 2, 3))
            if variant == "fixed" and mode == "eager":
                # The package also provides its RmsNorm dependency; cover that exported entry.
                rms_result = torch.ops.npu.npu_rms_norm(x1, gamma, epsilon=1e-6)
                torch.npu.synchronize()
                assert torch.isfinite(rms_result[0].cpu()).all().item()
            print(json.dumps({"stage": "after", "rows": rows}), flush=True)
    if variant == "fixed":
        print(json.dumps({"fixed_libraries": check_fixed_libraries()}), flush=True)


def read_fault(diagnostics: Path, pid: int) -> dict:
    fault_path = diagnostics / f"python-fault-{pid}.log"
    fault = fault_path.read_text(errors="replace") if fault_path.exists() else ""
    if "Fatal Python error: Segmentation fault" not in fault:
        return {}
    call_path = diagnostics / f"first-call-{pid}.json"
    try:
        call = json.loads(call_path.read_text())
    except (OSError, ValueError):
        call = {}
    kwargs = call.get("kwargs", {})
    target_call = (
        call.get("pid") == pid
        and call.get("op") in ("npu.npu_add_rms_norm_dynamic_quant", "npu.npu_add_rms_norm_dynamic_quant.default")
        and len(call.get("args", [])) == 3
        and isinstance(kwargs.get("beta"), dict)
        and kwargs.get("smooth_scale1") is None
        and kwargs.get("smooth_scale2") is None
        and kwargs.get("output_mask") == [True, False]
        and "fixed_probe.py" in fault
        and "diagnostic_hooks.py" in fault
        and " in __call__" in fault
    )
    return {"observed_segfault": True, "target_beta_call": target_call, "fault_log": str(fault_path)}


def run_child(command: list[str], env: dict, log, diagnostics: Path, timeout: float = 600) -> dict:
    proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    started = time.monotonic()
    fault_started = None
    status = "completed"
    terminated = False
    while proc.poll() is None:
        fault = read_fault(diagnostics, proc.pid)
        if fault:
            if fault_started is None:
                fault_started = time.monotonic()
            # Allow a partially written fault dump to finish before checking its call-site evidence.
            if not fault["target_beta_call"] and time.monotonic() - fault_started < 1:
                time.sleep(0.1)
                continue
            status = "segmentation_fault"
        elif time.monotonic() - started >= timeout:
            status = "timeout"
        else:
            time.sleep(0.2)
            continue
        # A native signal handler may leave the faulted process alive. Reap only this probe's process group.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            terminated = True
        except ProcessLookupError:
            pass
        break
    returncode = proc.wait(timeout=10)
    fault = read_fault(diagnostics, proc.pid)
    if fault:
        status = "segmentation_fault"
    return {"returncode": returncode, "status": status, "terminated_by_monitor": terminated, **fault}


def probe_passed(variant: str, result: dict) -> bool:
    if variant == "stock":
        return result.get("observed_segfault", False) and result.get("target_beta_call", False)
    return result.get("returncode") == 0 and result.get("status") == "completed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("stock", "fixed"), required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--child-mode", choices=("eager", "compile"))
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if args.child_mode:
        child(args.variant, args.child_mode, args.evidence)
        return
    results = []
    env = os.environ.copy()
    env.update(ACL_OP_INIT_MODE="1", TASK_QUEUE_ENABLE="1")
    for mode in ("eager",) if args.variant == "stock" else ("eager", "compile"):
        with (args.evidence / f"{args.variant}-beta-{mode}.log").open("w") as log:
            result = run_child(
                [
                    sys.executable,
                    __file__,
                    "--variant",
                    args.variant,
                    "--evidence",
                    str(args.evidence),
                    "--child-mode",
                    mode,
                ],
                env,
                log,
                args.evidence / f"{args.variant}-{mode}-diagnostics",
            )
            result.update(variant=args.variant, mode=mode, passed=probe_passed(args.variant, result))
        results.append(result)
        (args.evidence / f"{args.variant}-beta-result.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    assert all(item["passed"] for item in results), results


if __name__ == "__main__":
    main()
