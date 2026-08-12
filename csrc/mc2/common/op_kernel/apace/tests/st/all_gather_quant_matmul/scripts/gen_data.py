#!/usr/bin/env python3
"""
Generate MXFP8 test data for AllGatherQuantMatmul operator.

Inputs:
  - x1 (A): fp8e4m3fn_t, ND layout, shape [M, K]
  - x2 (B): fp8e4m3fn_t, DN layout (col-major ND), shape [K, N]
  - scale1: fp8e8m0_t scale for A, shape [M, ceil(K/64), 2]
  - scale2: fp8e8m0_t scale for B, shape [ceil(K/64), 2, N]

Output:
  - output: bfloat16_t, ND layout, shape [M, N/rankNum]

Usage:
  python3 gen_data.py m k n rank_num
"""

import math
import os
import sys

import numpy as np
import torch


def write_artifacts(base_dir, rank_id, a_fp8, b_fp8, a_scale, b_scale, out):
    """Write test data and golden output to input/output directories."""
    input_dir = os.path.join(base_dir, "input", str(rank_id))
    output_dir = os.path.join(base_dir, "output", str(rank_id))
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # A: ND layout (row-major), store as-is
    a_fp8.tofile(os.path.join(input_dir, "input_a.bin"))
    # B: DN layout (col-major), store by transposing first
    np.ascontiguousarray(b_fp8.T).tofile(os.path.join(input_dir, "input_b.bin"))
    # ScaleA: ND layout
    a_scale.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    # ScaleB: [ceil(K/64), 2, N] -> store as [N, ceil(K/64), 2] for DN layout
    np.ascontiguousarray(b_scale.transpose(2, 0, 1)).tofile(os.path.join(input_dir, "input_scaleB.bin"))
    # Output: bf16 as uint16
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def float8_e4m3fn_to_float(data_uint8):
    """Convert FP8E4M3FN (uint8) to FP32 with NPU bias=7."""
    data = data_uint8.astype(np.uint8)
    sign = (data >> 7) & 1
    exp = (data >> 3) & 15
    mant = data & 7

    fp32 = np.zeros_like(data, dtype=np.float32)
    zero_mask = (exp == 0) & (mant == 0)
    fp32[zero_mask] = 0.0
    normal_mask = ~zero_mask
    fp32[normal_mask] = (
        (1.0 - 2.0 * sign[normal_mask]) * np.power(2.0, exp[normal_mask] - 7.0) * (1.0 + mant[normal_mask] / 8.0)
    )
    return fp32


def float8_e8m0_to_float(scale_uint8):
    """Convert FP8E8M0 (uint8) to FP32 as power of 2."""
    exp = scale_uint8.astype(np.float32)
    return np.power(2.0, exp - 127.0)


def float_to_fp8_e4m3fn_vec(fp32_arr):
    """Vectorized FP32 -> FP8E4M3FN with NPU bias=7."""
    fp32 = fp32_arr.astype(np.float32)
    sign = (fp32 < 0).astype(np.uint8)
    abs_val = np.abs(fp32)

    MAX_NORMAL = 1.875 * 128.0  # 240.0
    MIN_NORMAL = 2.0 ** (-6)  # 0.015625

    log2_val = np.log2(np.clip(abs_val, 1e-30, None))
    e = np.floor(log2_val).astype(np.int32) + 7  # bias = 7
    e = np.clip(e, 1, 14)

    normalized = abs_val / np.power(2.0, (e - 7).astype(np.float32))
    mant = np.clip(np.round((normalized - 1.0) * 8.0).astype(np.int32), 0, 7)

    overflow = abs_val >= MAX_NORMAL
    underflow = abs_val < MIN_NORMAL
    e[overflow] = 14
    mant[overflow] = 7
    e[underflow] = 0
    mant[underflow] = 0
    sign[underflow] = 0

    result = (sign << 7) | (e.astype(np.uint8) << 3) | mant.astype(np.uint8)
    result[fp32 == 0.0] = 0
    return result


def generate_fp8_e4m3fn_data(shape, min_val=1.0, max_val=8.0):
    """Generate random FP8E4M3FN data."""
    fp32_data = np.random.uniform(min_val, max_val, shape).astype(np.float32)
    fp8_data = float_to_fp8_e4m3fn_vec(fp32_data)
    return fp8_data, fp32_data


def generate_fp8_e8m0_scale(shape):
    """Generate FP8E8M0 scale values (exp ~127-129 → scale ~1-4)."""
    exp_values = np.random.randint(127, 129, shape).astype(np.uint8)
    return exp_values


def dequantize_mxfp8(data_fp8, scale_fp8, divisor=64, c0=2):
    """
    MXFP8 dequantization: FP8 E4M3FN × FP8 E8M0 scale → FP32.
    Per-64-element group quantization with c0=2 scales per group.
    """
    sub_group = divisor // c0
    fp32_data = float8_e4m3fn_to_float(data_fp8.astype(np.uint8))
    fp32_scale = float8_e8m0_to_float(scale_fp8.astype(np.uint8))

    is_a_scale = scale_fp8.ndim == 3 and scale_fp8.shape[-1] == c0 and scale_fp8.shape[-2] > c0
    is_b_scale = scale_fp8.ndim == 3 and scale_fp8.shape[1] == c0

    k_shape = data_fp8.shape[1] if is_a_scale else data_fp8.shape[0]
    n_groups = (k_shape + divisor - 1) // divisor
    k_indices = np.arange(k_shape)
    group_idx = np.minimum(k_indices // divisor, n_groups - 1)
    sub_idx = np.clip((k_indices % divisor) // sub_group, 0, c0 - 1)

    scales = np.zeros(data_fp8.shape, dtype=np.float32)

    if is_a_scale:
        M = data_fp8.shape[0]
        for i in range(M):
            for k in range(k_shape):
                scales[i, k] = fp32_scale[i, group_idx[k], sub_idx[k]]
    elif is_b_scale:
        for k in range(k_shape):
            scales[k, :] = fp32_scale[group_idx[k], sub_idx[k], :]
    else:
        raise ValueError(f"Unexpected scale shape: {scale_fp8.shape}")

    return fp32_data * scales


BASE_SEED = 42


def gen_golden_data_all_gather_matmul(m, k, n, rank_num):
    """
    Generate golden data for AllGatherQuantMatmul operator.

    Each rank i holds independent A_i [M, K], B_i [K, N].
    Data flow:
      1. AllGather: A_all = concat([A_0, ..., A_{R-1}]) → [rankNum*M, K]
      2. Matmul:   C_i = A_all × B_i → [rankNum*M, N]
    Each rank produces DIFFERENT output (different B_i per rank).
    """
    M = m
    K = k
    N = n
    M_total = rank_num * M

    print(f"  M_per_rank={M}, K={K}, N={N}, M_total={M_total}")

    # 每个 rank 的 A / scaleA / B / scaleB 都用独立 seed 生成
    a_fp8_list = []
    a_scale_list = []
    b_fp8_list = []
    b_scale_list = []
    for rank_id in range(rank_num):
        np.random.seed(BASE_SEED + rank_id)
        a_fp8, _ = generate_fp8_e4m3fn_data((M, K), 1.0, 8.0)
        a_scale = generate_fp8_e8m0_scale((M, math.ceil(K / 64), 2))
        b_fp8, _ = generate_fp8_e4m3fn_data((K, N), 1.0, 8.0)
        b_scale = generate_fp8_e8m0_scale((math.ceil(K / 64), 2, N))
        a_fp8_list.append(a_fp8)
        a_scale_list.append(a_scale)
        b_fp8_list.append(b_fp8)
        b_scale_list.append(b_scale)

    print(f"  [DIAG] A[0]: fp8.shape={a_fp8_list[0].shape}, scale.shape={a_scale_list[0].shape}")
    print(f"  [DIAG] B[0]: fp8.shape={b_fp8_list[0].shape}, scale.shape={b_scale_list[0].shape}")

    # AllGather: concatenate all A_i → [rankNum*M, K]
    a_deq_parts = []
    for rank_id in range(rank_num):
        a_deq = dequantize_mxfp8(a_fp8_list[rank_id], a_scale_list[rank_id])
        a_deq_parts.append(a_deq)
    a_all_deq = np.concatenate(a_deq_parts, axis=0)

    print(f"  [DIAG] a_all_deq.shape={a_all_deq.shape} (expect ({M_total},{K}))")

    base_dir = os.getcwd()
    for rank_id in range(rank_num):
        b_deq = dequantize_mxfp8(b_fp8_list[rank_id], b_scale_list[rank_id])
        a_cpu = torch.from_numpy(a_all_deq)
        b_cpu = torch.from_numpy(b_deq)
        c_full = torch.matmul(a_cpu, b_cpu)  # [rankNum*M, N]
        c_bf16 = c_full.to(torch.bfloat16)

        if rank_id == 0:
            print(f"  [DIAG] a_all_deq.mean={a_all_deq.mean():.3f}, b_deq[0].mean={b_deq.mean():.3f}")
            print(f"  Golden C[0] shape {c_full.shape}, range [{c_full.min():.2f}, {c_full.max():.2f}]")

        write_artifacts(
            base_dir,
            rank_id,
            a_fp8_list[rank_id],
            b_fp8_list[rank_id],
            a_scale_list[rank_id],
            b_scale_list[rank_id],
            c_bf16,
        )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 gen_data.py m k n rank_num")
        print("  m: matrix A row dimension M")
        print("  k: matrix A column dimension K")
        print("  n: matrix B column dimension N")
        print("  rank_num: number of ranks")
        print("\nExample: python3 gen_data.py 2048 3584 4096 4")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    rank_num = int(sys.argv[4])

    if k % 32 != 0:
        print(f"Error: K={k} not divisible by 32")
        sys.exit(1)

    if math.ceil(k / 64) % 2 != 0:
        print(f"Error: ceil(K/64)={math.ceil(k / 64)} not even")
        sys.exit(1)

    print("Generating AllGatherQuantMatmul test data:")
    print(f"  M={m}, K={k}, N={n}, rankSize={rank_num}")

    gen_golden_data_all_gather_matmul(m, k, n, rank_num)

    print("Test data generation completed!")
