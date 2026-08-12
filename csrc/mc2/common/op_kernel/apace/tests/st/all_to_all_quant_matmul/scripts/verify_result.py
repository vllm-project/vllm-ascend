#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import sys

import numpy as np
import torch

# FP8(MXFP8) 量化输入 + BF16 输出，理论相对误差 ~0.5-1%。
# atol/rtol 都用 1e-2 对齐低精度量化算子的通用验收门槛。
ERROR_TOL = 1e-2
DATA_TYPE = np.uint16
FULL_TENSOR_PRINT_MAX_ELEMENTS = 1024
CORNER_ROWS = 4
CORNER_COLS = 4


def _print_large_tensor_summary(golden_tensor: torch.Tensor, npu_output_tensor: torch.Tensor, m: int, n: int, rank_id: int = -1) -> None:
    g = golden_tensor.float()
    p = npu_output_tensor.float()
    diff = p - g
    abs_err = diff.abs()
    denom = g.abs().clamp_min(1e-8)
    rel_err = abs_err / denom

    numel = m * n
    over_tol = (abs_err > ERROR_TOL).sum().item()

    rank_prefix = f"[Rank {rank_id}] " if rank_id >= 0 else ""
    print(f"\n{rank_prefix}[verify] shape=({m}, {n}), elements={numel} - summary (large matrix, full tensors omitted)")
    print(
        f"  abs_err: max={abs_err.max().item():.6e}, mean={abs_err.mean().item():.6e}, "
        f"rmse={(diff.pow(2).mean().sqrt()).item():.6e}"
    )
    print(f"  rel_err: max={rel_err.max().item():.6e}")
    print(f"  count(|abs_err| > {ERROR_TOL:g}): {over_tol} / {numel}")

    cr = min(CORNER_ROWS, m)
    cc = min(CORNER_COLS, n)
    if cr > 0 and cc > 0:
        print(f"  cpu golden (top-left {cr}x{cc}):\n{golden_tensor[:cr, :cc]}")
        print(f"  npu output (top-left {cr}x{cc}):\n{npu_output_tensor[:cr, :cc]}")


def verify_single_rank(m, n, rank_id, base_dir="./output"):
    output_path = os.path.join(base_dir, str(rank_id), "npu_out.bin")
    golden_path = os.path.join(base_dir, str(rank_id), "cpu_output.bin")
    
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"NPU output file not found: {output_path}")
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"CPU golden file not found: {golden_path}")

    output = np.fromfile(output_path, dtype=DATA_TYPE)
    golden = np.fromfile(golden_path, dtype=DATA_TYPE)

    expected_size = m * n
    if output.size != expected_size:
        raise ValueError(f"[Rank {rank_id}] npu output size {output.size} != expected size {expected_size}")
    if golden.size != expected_size:
        raise ValueError(f"[Rank {rank_id}] cpu output size {golden.size} != expected size {expected_size}")

    npu_output_tensor = torch.from_numpy(output).view(torch.bfloat16).reshape(m, n)
    golden_tensor = torch.from_numpy(golden).view(torch.bfloat16).reshape(m, n)

    numel = m * n
    if numel <= FULL_TENSOR_PRINT_MAX_ELEMENTS:
        print(f"\n[Rank {rank_id}] cpu golden:\n", golden_tensor)
        print(f"[Rank {rank_id}] npu output:\n", npu_output_tensor)
    else:
        _print_large_tensor_summary(golden_tensor, npu_output_tensor, m, n, rank_id)

    return torch.allclose(
        golden_tensor, npu_output_tensor, rtol=ERROR_TOL, atol=ERROR_TOL, equal_nan=True
    )


def verify_result(m, n, rank_num, base_dir="./output"):
    all_pass = True
    results = []
    
    print(f"\n{'='*60}")
    print(f"Verifying outputs for {rank_num} ranks")
    print(f"Matrix shape: M={m}, N={n}")
    print(f"All ranks should produce the same result after All2All")
    print(f"{'='*60}")
    
    for rank_id in range(rank_num):
        try:
            res = verify_single_rank(m, n, rank_id, base_dir)
            results.append((rank_id, bool(res), None if res else "allclose returned False"))
            if res:
                print(f"[PASS] Rank {rank_id}: NPU results match CPU golden")
            else:
                print(f"[FAIL] Rank {rank_id}: allclose returned False (rtol=atol={ERROR_TOL})")
                all_pass = False
        except Exception as e:
            results.append((rank_id, False, str(e)))
            print(f"[FAIL] Rank {rank_id}: {e}")
            all_pass = False
    
    print(f"\n{'='*60}")
    print("Verification Summary:")
    print(f"{'='*60}")
    for rank_id, passed, error in results:
        status = "PASS" if passed else "FAIL"
        print(f"  Rank {rank_id}: {status}")
        if error:
            print(f"    Error: {error}")
    
    return all_pass


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 verify_result.py m n rank_num [base_dir]")
        print("  m: matrix M dimension")
        print("  n: matrix N dimension")
        print("  rank_num: number of ranks")
        print("  base_dir: output directory (default: ./output)")
        print("\nExpected file structure:")
        print("  {base_dir}/0/npu_out.bin      - NPU output from rank 0")
        print("  {base_dir}/0/cpu_output.bin   - CPU golden for rank 0")
        print("  {base_dir}/1/npu_out.bin      - NPU output from rank 1")
        print("  {base_dir}/1/cpu_output.bin   - CPU golden for rank 1")
        print("  ...")
        sys.exit(1)

    m = int(sys.argv[1])
    n = int(sys.argv[2])
    rank_num = int(sys.argv[3])
    base_dir = sys.argv[4] if len(sys.argv) > 4 else "./output"
    
    try:
        all_pass = verify_result(m, n, rank_num, base_dir)
        if not all_pass:
            raise ValueError("[ERROR] Some NPU results differ from CPU.\n")
        print(f"\n[ALL PASS] All {rank_num} ranks passed verification!\n")

    except Exception as e:
        print(e)
        sys.exit(1)