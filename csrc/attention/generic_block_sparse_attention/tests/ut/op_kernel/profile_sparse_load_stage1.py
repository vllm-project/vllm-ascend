#!/usr/bin/env python3
"""Reproducible msprof op input for baseline/stage-1 comparison (no golden)."""

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "build.sh").exists())
sys.path.insert(0, str(REPO_ROOT / "torch_extension"))
sys.path.insert(0, str(REPO_ROOT / "attention/generic_block_sparse_attention/torch_extension"))
import generic_block_sparse_attention as gbsa_extension  # noqa: E402

_builder = gbsa_extension.generic_block_sparse_attention_op_builder
_builder.category = None
_builder.sources = lambda: [str(REPO_ROOT / "attention/generic_block_sparse_attention/torch_extension/csrc/generic_block_sparse_attention.cpp")]


def make_sparse(qs, kvs, block_y, topk, residual_mode):
    """Choose an ordered 80.078125% prefix sample from each causal row."""
    total_blocks = kvs // block_y
    visible = kvs - qs + torch.arange(qs, dtype=torch.int64) + 1
    full_blocks = visible // block_y
    explicit_count = full_blocks * topk // total_blocks
    cols = torch.arange(topk, dtype=torch.int64).unsqueeze(0)
    values = cols * full_blocks.unsqueeze(1) // explicit_count.clamp_min(1).unsqueeze(1)
    valid = cols < explicit_count.unsqueeze(1)
    idx = torch.where(valid, values, -1).to(torch.int32)
    if residual_mode == 0:
        has_tail = visible % block_y != 0
        rows = torch.nonzero(has_tail, as_tuple=True)[0]
        idx[rows, explicit_count[rows]] = full_blocks[rows].to(torch.int32)
        explicit_count += has_tail.to(torch.int64)
    assert int(explicit_count.max()) <= topk
    return idx.unsqueeze(0).contiguous(), explicit_count.to(torch.int32).unsqueeze(0).contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("decode", "prefill"), required=True)
    parser.add_argument("--residual-mode", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--group", type=int, default=16)
    parser.add_argument("--qs", type=int, help="override prefill length for D/resource sweeps")
    parser.add_argument("--precision", type=int, choices=(0, 1), default=0)
    parser.add_argument("--bench-repeats", type=int, default=0,
                        help="synchronized wall-clock samples; 0 keeps one call for msprof op")
    args = parser.parse_args()
    torch.set_num_threads(2)

    torch.manual_seed(20260924)
    torch_npu.npu.set_device(args.device)
    device = f"npu:{args.device}"
    qs = 1 if args.scenario == "decode" else 16384
    if args.qs is not None:
        qs = args.qs
    kvs, dim, q_heads, kv_heads, block_y, topk = 16384, args.dim, args.group, 1, 16, 820
    dtype = torch.float16
    q = (torch.randn(qs, q_heads, dim, dtype=torch.float32) * 0.2).to(dtype).to(device)
    k = (torch.randn(kvs, kv_heads, dim, dtype=torch.float32) * 0.2).to(dtype).to(device)
    v = (torch.randn(kvs, kv_heads, dim, dtype=torch.float32) * 0.2).to(dtype).to(device)
    idx, count = make_sparse(qs, kvs, block_y, topk, args.residual_mode)
    idx, count = idx.to(device), count.to(device)
    cu_q = torch.tensor([0, qs], dtype=torch.int64).to(device)
    cu_kv = torch.tensor([0, kvs], dtype=torch.int64).to(device)
    used_kv = torch.tensor([kvs], dtype=torch.int32).to(device)

    def make_metadata():
        return gbsa_extension.generic_block_sparse_attention_metadata(
            idx, count, q_heads, kv_heads, dim, [1, block_y],
            cu_seqlens_q=cu_q, cu_seqlens_kv=cu_kv, seqused_kv=used_kv,
            max_seqlen_q=qs, max_seqlen_kv=kvs,
            layout_q="TND", layout_kv="TND", layout_sparse_pattern=4,
            mask_mode=1, quant_mode=0, softmax_precision=args.precision,
            win_left=-1, win_right=-1, residual_block_mode=args.residual_mode,
            is_consistent_topk=False,
        )

    def invoke(metadata):
        return gbsa_extension.generic_block_sparse_attention(
            q, k, v, idx, count, [1, block_y], metadata=metadata,
            cu_seqlens_q=cu_q, cu_seqlens_kv=cu_kv, seqused_kv=used_kv,
            layout_q="TND", layout_kv="TND", layout_sparse_pattern=4,
            softmax_scale=1.0 / math.sqrt(dim), mask_mode=1, quant_mode=0,
            softmax_precision=args.precision, win_left=-1, win_right=-1,
            return_softmax_lse=False, residual_block_mode=args.residual_mode,
            is_consistent_topk=False,
        )
    with torch.inference_mode():
        metadata = make_metadata()
        out, _ = invoke(metadata)
        torch_npu.npu.synchronize()
        if args.bench_repeats:
            for _ in range(3):
                invoke(make_metadata())
            torch_npu.npu.synchronize()
            for include_metadata in (False, True):
                samples = []
                for _ in range(args.bench_repeats):
                    start = time.perf_counter_ns()
                    current_metadata = make_metadata() if include_metadata else metadata
                    out, _ = invoke(current_metadata)
                    torch_npu.npu.synchronize()
                    samples.append((time.perf_counter_ns() - start) / 1000)
                label = "metadata+attention+sync" if include_metadata else "attention+sync"
                print(f"WALL_US scope={label} median={statistics.median(samples):.3f} "
                      f"min={min(samples):.3f} max={max(samples):.3f} n={len(samples)}", flush=True)
    print(
        f"DONE scenario={args.scenario} B=1 QS={qs} KVS={kvs} D={dim} "
        f"Nq:Nkv={q_heads}:{kv_heads} blockY={block_y} topK={topk} "
        f"sparsity={(1 - topk / (kvs // block_y)) * 100:.6f}% "
        f"softmax_precision={args.precision} residualBlockMode={args.residual_mode} "
        f"FD_cores={int(metadata[5].cpu())} outputShape={tuple(out.shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
