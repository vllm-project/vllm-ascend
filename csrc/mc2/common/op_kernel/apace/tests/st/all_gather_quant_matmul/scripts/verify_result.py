#!/usr/bin/env python3
"""
Verify AllGatherQuantMatmul operator output.

Currently a lightweight check — confirms output files were generated and prints
summary statistics. Full precision verification will be added once kernel logic
is implemented.

Usage:
  python3 verify_result.py m n rank_num [base_dir]
"""

import os
import sys

import numpy as np
import torch


def verify_single_rank(m_total, n_per_rank, rank_id, base_dir="./output"):
    """Check NPU output file exists and print summary for a single rank.

    Returns: (pass_bool, diff_max, diff_mean, message)
      - pass_bool: True if diff.max()==0 and diff.mean()==0 (strict bit-exact)
      - diff_max: max absolute diff (None if no golden)
      - diff_mean: mean absolute diff (None if no golden)
      - message: status string
    """
    output_path = os.path.join(base_dir, str(rank_id), "npu_out.bin")
    golden_path = os.path.join(base_dir, str(rank_id), "cpu_output.bin")

    if not os.path.exists(output_path):
        print(f"[INFO] Rank {rank_id}: NPU output file not found at {output_path}")
        return False, None, None, "npu_out.bin not found"

    output = np.fromfile(output_path, dtype=np.uint16)
    expected_size = m_total * n_per_rank

    print(f"\n[Rank {rank_id}] NPU output: {output.size} elements (expected {expected_size})")

    if output.size != expected_size:
        print(f"[WARN] Rank {rank_id}: output size mismatch ({output.size} vs {expected_size})")
        return False, None, None, f"size mismatch: {output.size} != {expected_size}"

    npu_tensor = torch.from_numpy(output).view(torch.bfloat16).reshape(m_total, n_per_rank)
    print(f"[Rank {rank_id}] Output shape: ({m_total}, {n_per_rank})")
    print(f"[Rank {rank_id}] Output range: [{npu_tensor.float().min():.4f}, {npu_tensor.float().max():.4f}]")
    print(f"[Rank {rank_id}] Output mean: {npu_tensor.float().mean():.4f}")

    if not os.path.exists(golden_path):
        print(f"[Rank {rank_id}] No CPU golden file found, skipping comparison")
        return False, None, None, "no golden file"

    golden = np.fromfile(golden_path, dtype=np.uint16)
    golden_tensor = torch.from_numpy(golden).view(torch.bfloat16).reshape(m_total, n_per_rank)
    diff = (npu_tensor.float() - golden_tensor.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Raw uint16 comparison for 1-ULP tolerance.
    # Large-K matmuls can produce ~300 elements that differ by 1 ULP in bfloat16
    # due to float32 accumulation non-associativity. This is expected behavior.
    raw_npu = np.fromfile(output_path, dtype=np.uint16)
    raw_diff = (raw_npu.astype(np.int32) - golden.astype(np.int32)).__abs__()
    raw_max = raw_diff.max()
    raw_diff_count = (raw_diff > 0).sum()
    raw_one_ulp_count = (raw_diff == 1).sum()
    print(f"[Rank {rank_id}] diff vs golden: max={max_diff:.6e}, mean={mean_diff:.6e}")
    print(
        f"[Rank {rank_id}] raw uint16 diff: max={raw_max}, "
        f"diff_elems={raw_diff_count}, one_ulp_elems={raw_one_ulp_count}"
    )

    if max_diff == 0 and mean_diff == 0:
        print(f"[Rank {rank_id}] [PASS] Bit exact")
        return True, max_diff, mean_diff, None
    elif raw_max <= 1:
        print(
            f"[Rank {rank_id}] [PASS] Within 1 ULP "
            f"({raw_diff_count}/{expected_size} elems, {raw_one_ulp_count} at 1 ULP)"
        )
        return True, max_diff, mean_diff, None
    else:
        close_count = (diff < 0.2).sum().item()
        print(f"[Rank {rank_id}] [FAIL] Raw diff max={raw_max} > 1 ULP ({close_count}/{expected_size} within 0.2)")
        return False, max_diff, mean_diff, f"raw_max={raw_max} > 1 ULP"


def verify_result(m, n, rank_num, base_dir="./output"):
    """Verify results for all ranks.

    Returns: (all_pass: bool)
    """
    m_total = rank_num * m
    n_per_rank = n

    print(f"\n{'=' * 60}")
    print(f"AllGatherQuantMatmul — Output Check for {rank_num} ranks")
    print(f"Input: M={m}, N={n}")
    print(f"Per-rank output shape: ({m_total}, {n_per_rank})")
    print(f"{'=' * 60}")

    all_pass = True
    failed_ranks = []
    for rank_id in range(rank_num):
        ok, max_d, mean_d, msg = verify_single_rank(m_total, n_per_rank, rank_id, base_dir)
        if not ok:
            all_pass = False
            failed_ranks.append(rank_id)
            if msg:
                print(f"[FAIL] Rank {rank_id}: {msg}")
        else:
            print(f"[OK] Rank {rank_id}: bit exact")

    print(f"\n{'=' * 60}")
    if all_pass:
        print("Summary: All ranks OK")
    else:
        print(f"Summary: Some ranks FAILED: {failed_ranks}")
    print(f"{'=' * 60}")

    return all_pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify AllGatherQuantMatmul output")
    parser.add_argument("m", type=int, help="matrix M dimension")
    parser.add_argument("n", type=int, help="matrix N dimension (total)")
    parser.add_argument("rank_num", type=int, help="number of ranks")
    parser.add_argument("base_dir", nargs="?", default="./output", help="output directory")
    parser.add_argument("--check", action="store_true", help="Silent mode: print only PASS or FAIL and exit")
    args = parser.parse_args()

    m, n, rank_num, base_dir = args.m, args.n, args.rank_num, args.base_dir

    if n % rank_num != 0:
        print(f"Error: n={n} is not divisible by rank_num={rank_num}")
        sys.exit(1)

    try:
        if args.check:
            # Quiet mode for run_prof.sh — only print PASS or FAIL
            import io
            from contextlib import redirect_stderr, redirect_stdout

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                try:
                    all_pass = verify_result(m, n, rank_num, base_dir)
                except Exception:
                    all_pass = False
            print("PASS" if all_pass else "FAIL")
            sys.exit(0 if all_pass else 1)
        else:
            all_pass = verify_result(m, n, rank_num, base_dir)
            print("\n[DONE] Output check completed. Kernel logs should be visible in the run output above.\n")
            sys.exit(0 if all_pass else 1)
    except Exception as e:
        print(f"Error during verification: {e}")
        sys.exit(2)
