"""Benchmark 310P chunk_gated_delta_rule implementations.

Run this script from a built vllm-ascend checkout on a 310P machine.  For the
three-way comparison, run the same command on commits 2ce60baa, 4576fe1f, and
6-1-cgdr-next, using --mode pytorch or --mode ascend depending on what that
checkout exposes.
"""

from __future__ import annotations

import argparse
import subprocess
import time

import torch
import torch_npu

from vllm_ascend._310p.ops.fla import chunk_gated_delta_rule as cgdr_mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pytorch", "ascend", "both"), default="both")
    parser.add_argument("--seq-lens", default="1024", help="Comma-separated varlen sequence lengths.")
    parser.add_argument("--h-qk", type=int, default=4)
    parser.add_argument("--h-v", type=int, default=8)
    parser.add_argument("--k-dim", type=int, default=192)
    parser.add_argument("--v-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def make_inputs(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    device = torch.device(f"npu:{args.device}")
    seq_lens = [int(item) for item in args.seq_lens.split(",") if item]
    total_tokens = sum(seq_lens)
    cu_seqlens = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int64, device=device)

    q = torch.randn(1, total_tokens, args.h_qk, args.k_dim, dtype=torch.float16, device=device)
    k = torch.randn(1, total_tokens, args.h_qk, args.k_dim, dtype=torch.float16, device=device)
    v = torch.randn(1, total_tokens, args.h_v, args.v_dim, dtype=torch.float16, device=device)
    g = -torch.rand(1, total_tokens, args.h_v, dtype=torch.float32, device=device) * 0.2
    beta = (0.1 + 0.4 * torch.rand(1, total_tokens, args.h_v, dtype=torch.float32, device=device)).to(torch.float16)
    initial_state = (
        torch.randn(len(seq_lens), args.h_v, args.v_dim, args.k_dim, dtype=torch.float16, device=device) * 0.01
    )
    return q, k, v, g, beta, initial_state, cu_seqlens


def run_once(fn, inputs):
    q, k, v, g, beta, initial_state, cu_seqlens = inputs
    return fn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )


def bench(label: str, fn, inputs, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        run_once(fn, inputs)
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        run_once(fn, inputs)
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
    print(f"{label:>8}: {elapsed_ms:.3f} ms/iter")
    return elapsed_ms


def main() -> int:
    args = parse_args()
    torch_npu.npu.set_device(args.device)
    inputs = make_inputs(args)

    print(f"commit: {current_commit()}")
    print(f"device: {torch_npu.npu.get_device_name(args.device)}")
    print(f"shape: seq_lens={args.seq_lens} h_qk={args.h_qk} h_v={args.h_v} k_dim={args.k_dim} v_dim={args.v_dim}")
    print(f"warmup={args.warmup} iters={args.iters}")

    funcs = {}
    if args.mode in ("pytorch", "both"):
        funcs["pytorch"] = cgdr_mod.chunk_gated_delta_rule_pytorch
    if args.mode in ("ascend", "both"):
        ascend_fn = getattr(cgdr_mod, "chunk_gated_delta_rule_310", None)
        if ascend_fn is None:
            raise RuntimeError("This checkout does not expose chunk_gated_delta_rule_310.")
        funcs["ascend"] = ascend_fn

    results = {name: bench(name, fn, inputs, args.warmup, args.iters) for name, fn in funcs.items()}
    if "pytorch" in results and "ascend" in results:
        speedup = results["pytorch"] / results["ascend"]
        print(f"speedup ascend_vs_pytorch: {speedup:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
