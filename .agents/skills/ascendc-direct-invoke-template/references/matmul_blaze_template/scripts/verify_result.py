# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import sys

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import numpy as np
import torch

POINT_ERROR_TOL = 1e-1
RATIO_POINT_ERROR_TOL = 1e-3
ERROR_RATIO_TOL = 1e-3
DATA_TYPE = np.uint16

FULL_TENSOR_PRINT_MAX_ELEMENTS = 128
CORNER_ROWS = 4
CORNER_COLS = 4


def _print_large_tensor_summary(golden_tensor: torch.Tensor, npu_output_tensor: torch.Tensor, m: int, n: int) -> None:
    g = golden_tensor.float()
    p = npu_output_tensor.float()
    diff = p - g
    abs_err = diff.abs()
    denom = g.abs().clamp_min(1e-8)
    rel_err = abs_err / denom

    numel = m * n
    over_tol = (abs_err > RATIO_POINT_ERROR_TOL).sum().item()

    print(f"\n[verify] shape=({m}, {n}), elements={numel} - summary (large matrix, full tensors omitted)")
    print(
        f"  abs_err: max={abs_err.max().item():.6e}, mean={abs_err.mean().item():.6e}, "
        f"rmse={(diff.pow(2).mean().sqrt()).item():.6e}"
    )
    print(f"  rel_err: max={rel_err.max().item():.6e}")
    print(f"  count(|abs_err| > {RATIO_POINT_ERROR_TOL:g}): {over_tol} / {numel}")

    cr = min(CORNER_ROWS, m)
    cc = min(CORNER_COLS, n)
    if cr > 0 and cc > 0:
        print(f"  cpu golden (top-left {cr}x{cc}):\n{golden_tensor[:cr, :cc]}")
        print(f"  npu output (top-left {cr}x{cc}):\n{npu_output_tensor[:cr, :cc]}")


def verify_result(m, n):
    output = np.fromfile("./output/npu_out.bin", dtype=DATA_TYPE)
    golden = np.fromfile("./output/cpu_output.bin", dtype=DATA_TYPE)

    if output.size != golden.size:
        raise ValueError("npu output size != cpu output size")

    npu_output_tensor = torch.from_numpy(output).view(torch.bfloat16).reshape(m, n)
    golden_tensor = torch.from_numpy(golden).view(torch.bfloat16).reshape(m, n)

    numel = m * n
    if numel <= FULL_TENSOR_PRINT_MAX_ELEMENTS:
        print("\ncpu golden:\n", golden_tensor)
        print("npu output:\n", npu_output_tensor)
    else:
        _print_large_tensor_summary(golden_tensor, npu_output_tensor, m, n)

    golden_f32 = golden_tensor.to(torch.float32)
    npu_f32 = npu_output_tensor.to(torch.float32)
    abs_diff = torch.abs(golden_f32 - npu_f32)
    non_finite_mask = ~(torch.isfinite(golden_f32) & torch.isfinite(npu_f32) & torch.isfinite(abs_diff))
    abs_golden = torch.abs(golden_f32)
    rel_diff = torch.where(
        abs_golden > 0,
        abs_diff / abs_golden,
        torch.where(abs_diff == 0, torch.zeros_like(abs_diff), torch.full_like(abs_diff, float("inf"))),
    )
    point_error_mask = (rel_diff > POINT_ERROR_TOL) | non_finite_mask
    ratio_error_mask = (abs_diff > RATIO_POINT_ERROR_TOL) | non_finite_mask
    point_error_count = int(point_error_mask.sum().item())
    error_count = int(ratio_error_mask.sum().item())
    error_ratio = error_count / numel if numel else 0.0

    print(f"max abs diff: {abs_diff.max().item() if numel else 0.0}")
    print(f"point error count(>{POINT_ERROR_TOL}): {point_error_count}/{numel}")
    if point_error_count > 0:
        point_error_indices = torch.nonzero(point_error_mask, as_tuple=False)
        print(f"point error details(rel diff > {POINT_ERROR_TOL} or non-finite):")
        for idx in point_error_indices:
            row = int(idx[0].item())
            col = int(idx[1].item())
            golden_val = float(golden_f32[row, col].item())
            npu_val = float(npu_f32[row, col].item())
            diff_val = float(abs_diff[row, col].item())
            rel_val = float(rel_diff[row, col].item())
            print(
                f"  (row={row}, col={col}) "
                f"golden={golden_val}, npu={npu_val}, abs_diff={diff_val}, rel_diff={rel_val}"
            )
    print(
        f"ratio error count(>{RATIO_POINT_ERROR_TOL}): {error_count}/{numel}, "
        f"error ratio: {error_ratio:.6f}"
    )

    return point_error_count == 0 and error_ratio <= ERROR_RATIO_TOL


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 verify_result.py m n")
        sys.exit(1)

    m = int(sys.argv[1])
    n = int(sys.argv[2])
    try:
        res = verify_result(m, n)
        if not res:
            raise ValueError(
                f"[ERROR] NPU results differ from CPU. "
                f"Single-point relative error (abs_diff/abs(golden)) must be <= {POINT_ERROR_TOL}, "
                f"and the ratio of points with absolute error > {RATIO_POINT_ERROR_TOL} "
                f"must be <= {ERROR_RATIO_TOL}.\n"
            )
        print("[PASS] NPU results are consistent with CPU.\n")

    except Exception as e:
        print(e)
        sys.exit(1)
