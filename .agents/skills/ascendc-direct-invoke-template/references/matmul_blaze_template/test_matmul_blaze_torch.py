#!/usr/bin/python3
# coding=utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import math
import os
import sys

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import numpy as np
import torch
import torch_npu
from en_dtypes import float8_e8m0
from ml_dtypes import float8_e4m3fn

SO_NAME = "libmatmul_blaze_ops.so"
OP_NAME = "matmul_blaze"
POINT_ERROR_TOL = 1e-1
RATIO_POINT_ERROR_TOL = 1e-3
ERROR_RATIO_TOL = 1e-3


def build_scale_broadcast(scale, target_shape, chunk_axis):
    scale_repeat = np.repeat(scale.astype(np.float32), 32, axis=-1)
    if chunk_axis == 1:
        scale_broadcast = scale_repeat.reshape(scale.shape[0], -1)[..., : target_shape[1]]
    elif chunk_axis == 0:
        scale_broadcast = np.transpose(scale_repeat, (0, 2, 1)).reshape(-1, scale.shape[1])[: target_shape[0], ...]
    else:
        raise ValueError(f"Invalid chunk_axis={chunk_axis}")
    return scale_broadcast


def dequant_mxfp8(fp8_input, scale, chunk_axis):
    scale_broadcast = build_scale_broadcast(scale, fp8_input.shape, chunk_axis)
    return fp8_input.astype(np.float32) * scale_broadcast


def compute_golden(m, k, n, trans_a, trans_b):
    a_shape = (k, m) if trans_a else (m, k)
    a_ori = np.random.uniform(1, 8, a_shape).astype(float8_e4m3fn)

    b_shape = (n, k) if trans_b else (k, n)
    b_ori = np.random.uniform(1, 8, b_shape).astype(float8_e4m3fn)

    a_scale_shape = (math.ceil(k / 64), m, 2) if trans_a else (m, math.ceil(k / 64), 2)
    b_scale_shape = (n, math.ceil(k / 64), 2) if trans_b else (math.ceil(k / 64), n, 2)
    a_scale = np.random.uniform(1, 8, size=a_scale_shape).astype(float8_e8m0)
    b_scale = np.random.uniform(1, 8, size=b_scale_shape).astype(float8_e8m0)

    a_chunk_axis = 0 if trans_a else 1
    b_chunk_axis = 1 if trans_b else 0

    a_dequant = dequant_mxfp8(a_ori, a_scale, a_chunk_axis)
    b_dequant = dequant_mxfp8(b_ori, b_scale, b_chunk_axis)

    a_matmul = np.swapaxes(a_dequant, -1, -2) if trans_a else a_dequant
    b_matmul = np.swapaxes(b_dequant, -1, -2) if trans_b else b_dequant

    golden = torch.matmul(torch.from_numpy(a_matmul), torch.from_numpy(b_matmul)).to(torch.bfloat16)

    return a_ori, b_ori, a_scale, b_scale, golden


def run_test(m, k, n, trans_a, trans_b):
    a_ori, b_ori, a_scale, b_scale, golden = compute_golden(m, k, n, trans_a, trans_b)

    a_t = torch.from_numpy(a_ori.view(np.uint8)).view(torch.float8_e4m3fn).to("npu")
    b_t = torch.from_numpy(b_ori.view(np.uint8)).view(torch.float8_e4m3fn).to("npu")
    sA_t = torch.from_numpy(a_scale.view(np.uint8)).to("npu")
    sB_t = torch.from_numpy(b_scale.view(np.uint8)).to("npu")

    op_fn = getattr(torch.ops.npu, OP_NAME)
    c = op_fn(a_t, b_t, sA_t, sB_t, trans_a, trans_b)

    diff = (c.cpu().float() - golden.float()).abs()
    max_diff = diff.max().item()

    rel_diff = diff / golden.float().abs().clamp_min(1e-8)
    point_error_count = int((rel_diff > POINT_ERROR_TOL).sum().item())
    error_count = int((diff > RATIO_POINT_ERROR_TOL).sum().item())
    numel = m * n
    error_ratio = error_count / numel if numel else 0.0

    passed = point_error_count == 0 and error_ratio <= ERROR_RATIO_TOL
    return passed, max_diff, error_ratio


def parse_bool_arg(value, name):
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "t"):
        return True
    if normalized in ("0", "false", "f"):
        return False
    raise ValueError(f"Invalid {name}: {value}. Expected one of 0/1/true/false.")


def main():
    if len(sys.argv) not in (1, 4, 6):
        print("Usage: python3 test_matmul_blaze_torch.py [m k n [transA transB]]")
        print("Example: python3 test_matmul_blaze_torch.py 16 128 16384 false true")
        sys.exit(1)

    m = int(sys.argv[1]) if len(sys.argv) >= 4 else 16
    k = int(sys.argv[2]) if len(sys.argv) >= 4 else 128
    n = int(sys.argv[3]) if len(sys.argv) >= 4 else 16384
    trans_a = parse_bool_arg(sys.argv[4], "transA") if len(sys.argv) >= 6 else False
    trans_b = parse_bool_arg(sys.argv[5], "transB") if len(sys.argv) >= 6 else True

    so_path = SO_NAME
    if not os.path.exists(so_path):
        so_path = os.path.join("build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {SO_NAME} not found. Build matmul_blaze_ops first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    print(f"Running torch test: M={m} K={k} N={n} transA={trans_a} transB={trans_b}")
    passed, max_diff, error_ratio = run_test(m, k, n, trans_a, trans_b)

    print(f"max abs diff: {max_diff}")
    print(f"error ratio: {error_ratio:.6f}")
    if passed:
        print("[PASS] PyTorch results are consistent with CPU.")
    else:
        print(f"[FAIL] NPU results differ from CPU. "
              f"Single-point relative error must be <= {POINT_ERROR_TOL}, "
              f"and error ratio must be <= {ERROR_RATIO_TOL}.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
