# SPDX-License-Identifier: Apache-2.0
"""Compare frozen/candidate DCP combine kernels; optionally export ascend_pt traces.

Use --reference-root for an immutable baseline checkout, --world for the number
of contributions in one receive buffer, and --profile-dir for operator traces.
This single-device benchmark excludes HCCL and whole-model execution.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu
from dcp_four_point_ab import measure

from vllm_ascend.ops.triton.dcp import dcp_a2a
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


def combine_cases(args):
    from vllm_ascend.device.device_config import AscendDeviceType, get_ascend_device_type

    frozen_dir = args.reference_root / "vllm_ascend/ops/triton/dcp"
    frozen_path = frozen_dir / "dcp_a2a_batched.py"
    spec = importlib.util.spec_from_file_location("frozen_combine_kernels", frozen_path)
    frozen_kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen_kernels)
    tree = ast.parse((frozen_dir / "dcp_a2a.py").read_text())
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "fused_dcp_lse_combine"
    ]
    scope = dict(vars(dcp_a2a))
    scope.update(
        AscendDeviceType=AscendDeviceType,
        get_ascend_device_type=get_ascend_device_type,
        _fused_dcp_lse_combine_batched_kernel=frozen_kernels._fused_dcp_lse_combine_batched_kernel,
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(frozen_dir / "dcp_a2a.py"), "exec"), scope)
    baseline_combine = scope["fused_dcp_lse_combine"]
    torch.manual_seed(55)
    r, t, h, d = args.world, args.rows, args.heads, 512
    history = torch.randn(r, t, h, d, dtype=torch.bfloat16)
    logs = torch.randn(r, t, h, 1) * 12
    history[:, 0, 0] = torch.nan
    logs[:, 0, 0] = -torch.inf
    history[0, -1, -1] = torch.nan
    logs[0, -1, -1] = torch.nan
    local_cpu = torch.randn(t, h, d, dtype=torch.bfloat16) if args.local else None
    local_logs = torch.randn(t, h, 1) * 12 if args.local else None
    if args.local:
        local_cpu[0, 0] = torch.nan
        local_logs[0, 0] = -torch.inf
    values = history.double()
    lses = logs.double()
    if args.local:
        values = torch.cat((values, local_cpu.double()[None]))
        lses = torch.cat((lses, local_logs.double()[None]))
    valid = torch.isfinite(lses)
    maximum = torch.where(valid, lses, -torch.inf).amax(0)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0)
    weights = torch.where(valid, (lses - maximum).exp(), 0)
    total = weights.sum(0)
    expected = (torch.where(valid, values, 0) * weights).sum(0) / torch.where(total > 0, total, 1)
    recv = dcp_a2a.pack_dcp_output_lse(
        history.transpose(0, 1).reshape(t, r * h, d).npu(),
        logs.transpose(0, 1).reshape(t, r * h, 1).npu(),
        r,
        1,
    )
    local = local_cpu.npu() if args.local else None
    local_lse = local_logs.npu() if args.local else None

    cases = [
        ("combine_baseline", lambda: baseline_combine(recv, d, 1, local_output=local, local_lse=local_lse), expected),
        (
            "combine_candidate",
            lambda: dcp_a2a.fused_dcp_lse_combine(recv, d, 1, local_output=local, local_lse=local_lse),
            expected,
        ),
    ]
    return cases


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=48)
    parser.add_argument("--world", type=int, default=16)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    if min(args.rows, args.world, args.heads) <= 0:
        parser.error("rows/world/heads must be positive")
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    init_device_properties_triton()
    root = Path(dcp_a2a.__file__).resolve().parents[4]
    result = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": torch.npu.get_device_name(),
        "cores": get_vectorcore_num(),
        "hashes": {},
        "measurement": {"warmups": 5, "graph_warmups": 3, "kernels_per_replay": 20, "samples": 7},
        "cases": [],
    }
    for relative in ("ops/triton/dcp/dcp_a2a_batched.py", "ops/triton/dcp/dcp_a2a.py"):
        path = root / "vllm_ascend" / relative
        result["hashes"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["reference_hashes"] = {}
    for relative in result["hashes"]:
        path = args.reference_root / "vllm_ascend" / relative
        result["reference_hashes"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    cases = combine_cases(args)
    if args.reverse:
        cases.reverse()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, fn, expected in cases:
        actual = fn()
        torch.testing.assert_close(actual.cpu().double(), expected, rtol=0.016, atol=0.001)
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                captured = fn()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(captured.cpu().double(), expected, rtol=0.016, atol=0.001)
        finally:
            graph.reset()
        entry = {"name": name, "correctness": "eager_and_graph_pass"}
        for mode in ("graph",):
            entry[mode] = measure(fn, mode, 20, 7)
        result["cases"].append(entry)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
        if args.profile_dir:
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.profile_dir / name)),
                record_shapes=True,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1
                ),
            ) as profiler:
                for _ in range(4):
                    with torch.autograd.profiler.record_function(name):
                        fn()
                    profiler.step()


if __name__ == "__main__":
    main()
