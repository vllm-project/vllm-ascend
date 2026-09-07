"""确认同次 A3 CI 中原包仍崩溃、修复包的带 beta 调用可以执行。"""

import argparse
import json
import os
import resource
import subprocess
import sys
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("stock", "fixed"), required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--child-mode", choices=("eager", "compile"))
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if args.child_mode:
        child(args.variant, args.child_mode, args.evidence)
        return
    results = []
    env = os.environ.copy()
    env.update(ACL_OP_INIT_MODE="1", TASK_QUEUE_ENABLE="1")
    for mode in ("eager",) if args.variant == "stock" else ("eager", "compile"):
        with (args.evidence / f"{args.variant}-beta-{mode}.log").open("w") as log:
            try:
                proc = subprocess.run(
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
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                    check=False,
                )
                result = {"variant": args.variant, "mode": mode, "returncode": proc.returncode}
            except subprocess.TimeoutExpired:
                result = {"variant": args.variant, "mode": mode, "status": "timeout"}
        results.append(result)
        (args.evidence / f"{args.variant}-beta-result.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    expected = -11 if args.variant == "stock" else 0
    assert all(item.get("returncode") == expected for item in results), results


if __name__ == "__main__":
    main()
