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

import math
import os
import sys

import numpy as np
import torch
from en_dtypes import float8_e8m0
from ml_dtypes import float8_e4m3fn


def write_artifacts(base_dir, rank_id, a_segment_fp8, b_fp8, a_segment_scale, b_scale, out):
    input_dir = os.path.join(base_dir, "input", str(rank_id))
    output_dir = os.path.join(base_dir, "output", str(rank_id))
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    a_segment_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_a.bin"))
    b_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_b.bin"))
    a_segment_scale.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    b_scale.tofile(os.path.join(input_dir, "input_scaleB.bin"))
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_golden_data_all_to_all_matmul(m, k, n, rank_num):
    M_total = m * rank_num
    K = k
    N = n
    Ka = K // rank_num

    if K % rank_num != 0:
        raise ValueError(f"K={K} must be divisible by rank_num={rank_num}")

    # [NOTE] 2026-06-22: fixed seed 方便与 sanity_check.py 对比 shmem_A.bin。
    np.random.seed(42)

    b_ori = np.random.uniform(1, 8, (N, K)).astype(float8_e4m3fn)
    b_ori_transpose = np.swapaxes(b_ori, -1, -2)
    
    b_scale_ori = np.random.uniform(1, 8, size=(N, math.ceil(K / 64), 2)).astype(float8_e8m0)
    
    b_scale_reshape = b_scale_ori.reshape(N, -1)
    b_scale_broadcast = np.repeat(b_scale_reshape, 32, axis=-1)[..., :K]
    b_scale_broadcast_transpose = np.swapaxes(b_scale_broadcast, -1, -2)
    b_dequant = b_ori_transpose.astype(np.float32) * b_scale_broadcast_transpose.astype(np.float32)

    # [FIX] 2026-06-22: TransB=true 时 kernel 用 ScaleBDNLayoutPtn (column-major)
    # 访问 scaleB，layoutScaleB = (rankDim*kScaleSize, N) = (K/32, N) 按列主序读。
    # host 必须写成 [N, K/32] C-order (= [K/32, N] 列主序) 才能被正确解读。
    # 之前 .transpose(1, 0, 2).reshape(K//32, N) 写的是行主序，被 kernel 按列主序
    # 读成乱序字节，导致最终输出与 golden rel_err 288% 不相关。
    b_scale_all = b_scale_ori.reshape(N, K // 32).flatten()

    a_local_list = []
    a_scale_list = []

    for rank_id in range(rank_num):
        a_local = np.random.uniform(1, 8, (M_total, Ka)).astype(float8_e4m3fn)
        a_scale = np.random.uniform(1, 8, size=(M_total, math.ceil(Ka / 64), 2)).astype(float8_e8m0)
        
        a_local_list.append(a_local)
        a_scale_list.append(a_scale)

    a_dequant_list = []
    for rank_id in range(rank_num):
        a_local = a_local_list[rank_id]
        a_scale = a_scale_list[rank_id]
        a_scale_reshape = a_scale.reshape(M_total, -1)
        a_scale_broadcast = np.repeat(a_scale_reshape, 32, axis=-1)[..., :Ka]
        a_dequant = a_local.astype(np.float32) * a_scale_broadcast.astype(np.float32)
        a_dequant_list.append(a_dequant)

    a_after_alltoall_list = []
    for dst_rank in range(rank_num):
        a_after_alltoall = np.zeros((M_total, Ka), dtype=np.float32)
        for src_rank in range(rank_num):
            chunk_row_start = dst_rank * m
            chunk_row_end = chunk_row_start + m
            chunk = a_dequant_list[src_rank][chunk_row_start:chunk_row_end, :]
            
            dst_row_start = src_rank * m
            dst_row_end = dst_row_start + m
            a_after_alltoall[dst_row_start:dst_row_end, :] = chunk
        a_after_alltoall_list.append(a_after_alltoall)

    out_all_ranks = []
    for dst_rank in range(rank_num):
        a_permuted = np.zeros((m, K), dtype=np.float32)
        for src_rank in range(rank_num):
            src_row_start = src_rank * m
            src_row_end = src_row_start + m
            src_chunk = a_after_alltoall_list[dst_rank][src_row_start:src_row_end, :]
            
            dst_col_start = src_rank * Ka
            dst_col_end = dst_col_start + Ka
            a_permuted[:, dst_col_start:dst_col_end] = src_chunk
        
        a_cpu = torch.from_numpy(a_permuted)
        b_cpu = torch.from_numpy(b_dequant)
        out = torch.matmul(a_cpu, b_cpu).to(torch.bfloat16)
        out_all_ranks.append(out)

    # [FIX] 2026-06-22: TransB=true 时 kernel 用 DNExtLayoutPtn (column-major)
    # 访问 B，layoutB = (rankDim*Ka, N) = (K, N) 按列主序读。
    # host 必须写成 [N, K] C-order (= [K, N] 列主序)，即直接写 b_ori 不转置。
    # 之前传 b_ori_transpose 写的是 [K, N] 行主序，被 kernel 按列主序读成乱序字节。
    # golden 计算里仍用 b_ori_transpose (逻辑 B = [K, N])，不动。
    current_dir = os.getcwd()
    for rank_id in range(rank_num):
        write_artifacts(current_dir, rank_id, a_local_list[rank_id], b_ori,
                       a_scale_list[rank_id], b_scale_all, out_all_ranks[rank_id])


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 gen_data.py m k n rank_num")
        print("  m: matrix M dimension")
        print("  k: matrix K dimension (total, must be divisible by rank_num)")
        print("  n: matrix N dimension")
        print("  rank_num: number of ranks")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    rank_num = int(sys.argv[4])

    gen_golden_data_all_to_all_matmul(m, k, n, rank_num)