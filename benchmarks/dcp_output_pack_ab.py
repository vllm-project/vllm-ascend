# SPDX-License-Identifier: Apache-2.0
"""Compare scalar and batched output/LSE pack on identical received values."""

import argparse
import ast
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from dcp_four_point_ab import measure

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.triton.dcp import dcp_a2a
from vllm_ascend.ops.triton.dcp.dcp_a2a import pack_dcp_output_lse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=48)
    parser.add_argument("--world", type=int, default=16)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.manual_seed(56)
    values = torch.randn(args.rows, args.world * args.heads, 512, dtype=torch.bfloat16).npu()
    lse = (torch.randn(args.rows, args.world * args.heads, 1) * 12).npu()
    frozen = (args.reference_root / "vllm_ascend/ops/triton/dcp").resolve()
    scope = dict(vars(dcp_a2a))
    scope["is_950"] = is_950
    spec = importlib.util.spec_from_file_location("frozen_pack_batched", frozen / "dcp_a2a_batched.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scope["_pack_dcp_output_lse_batched_kernel"] = module._pack_dcp_output_lse_batched_kernel
    tree = ast.parse((frozen / "dcp_a2a.py").read_text())
    funcs = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("_pack_dcp_output_lse_kernel", "pack_dcp_output_lse")
    ]
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(frozen / "dcp_a2a.py"), "exec"), scope)
    baseline = scope["pack_dcp_output_lse"]
    bits = lse.cpu().contiguous().view(torch.int32)
    exponent = (bits >> 23) & 255
    sig = (bits & 0x7FFFFF) | torch.where(exponent == 0, 0, 0x800000)
    sig = torch.where(exponent < 255, sig, 0)
    code = exponent.clamp_min(1) + 1
    code = torch.where(bits < 0, -code, code)
    code = torch.where((bits & 0x7FFFFFFF) == 0, 128, code)
    code = torch.where(exponent < 255, code, 0)
    fields = torch.cat((code, (sig >> 16) & 255, (sig >> 8) & 255, sig & 255), -1).to(torch.bfloat16)
    exact = (
        torch.cat((values.cpu(), fields), -1)
        .reshape(args.rows, args.world, args.heads, 516)
        .permute(1, 2, 0, 3)
        .contiguous()
    )
    result = []
    cases = [
        ("baseline", lambda: baseline(values, lse, args.world, 1)),
        ("candidate", lambda: pack_dcp_output_lse(values, lse, args.world, 1)),
    ]
    if args.reverse:
        cases.reverse()
    for name, fn in cases:
        for _ in range(5):
            actual = fn()
        torch.npu.synchronize()
        cpu = actual.cpu()
        torch.testing.assert_close(cpu[..., :512], exact[..., :512], rtol=0, atol=0)
        if name == "candidate":
            torch.testing.assert_close(cpu, exact, rtol=0, atol=0)
        code, hi, mid, lo = cpu[..., 512:].double().unbind(-1)
        decoded = torch.sign(code) * torch.ldexp(hi * 65536 + mid * 256 + lo, (code.abs() - 151).int())
        expected_lse = lse.cpu().reshape(args.rows, args.world, args.heads).permute(1, 2, 0).double()
        torch.testing.assert_close(decoded, expected_lse, rtol=1e-5, atol=1e-6)
        entry = dict(
            name=name,
            graph=measure(fn, "graph", args.iterations, args.repeats),
            correctness="payload_exact_and_lse_oracle",
            max_lse_error=(decoded - expected_lse).abs().max().item(),
        )
        result.append(entry)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
        if args.profile_dir:
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.profile_dir / name)),
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1
                ),
            ) as profiler:
                for _ in range(4):
                    fn()
                    profiler.step()


if __name__ == "__main__":
    main()
