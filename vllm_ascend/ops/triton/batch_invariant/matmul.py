# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/batch_invariant.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import torch

from vllm.triton_utils import triton, tl


@triton.jit
def matmul_bias_persistent_kernel(
    # 输入张量指针
    x_ptr,
    y_ptr,
    bias_ptr,
    output_ptr,
    # 矩阵维度
    M,
    N,
    K,
    # 步长信息
    stride_xm,
    stride_xk,  # x的步长: [M, K]
    stride_yk,
    stride_yn,  # y的步长: [K, N]  
    stride_bias,  # bias的步长: [N]
    stride_outm,
    stride_outn,  # 输出的步长: [M, N]
    # 是否使用偏置
    has_bias: tl.constexpr,
    # 分块大小（常量表达式）
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 获取程序ID（2D网格）
    pid_m = tl.program_id(0)  # 行分块ID
    pid_n = tl.program_id(1)  # 列分块ID

    # 计算当前块在矩阵中的起始位置
    rm_start = pid_m * BLOCK_M
    rn_start = pid_n * BLOCK_N

    # 创建索引范围
    rm = rm_start + tl.arange(0, BLOCK_M)  # 行索引范围 [BLOCK_M]
    rn = rn_start + tl.arange(0, BLOCK_N)  # 列索引范围 [BLOCK_N]
    rk = tl.arange(0, BLOCK_K)  # K维度索引范围 [BLOCK_K]

    # 初始化累加器为0
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 在K维度上进行循环，每次处理BLOCK_K个元素
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k * BLOCK_K
        # 计算x的指针偏移量（行主序）
        x_ptrs = x_ptr + rm[:, None] * stride_xm + (rk[None, :] +
                                                    k_start) * stride_xk
        # 计算y的指针偏移量（行主序）
        y_ptrs = y_ptr + (rk[:, None] +
                          k_start) * stride_yk + rn[None, :] * stride_yn

        # 创建掩码以防止越界访问
        x_mask = (rm[:, None] < M) & ((rk[None, :] + k_start) < K)
        y_mask = ((rk[:, None] + k_start) < K) & (rn[None, :] < N)

        # 从全局内存加载数据块
        x_chunk = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
        y_chunk = tl.load(y_ptrs, mask=y_mask, other=0.0).to(tl.float32)

        # 计算矩阵乘法累加
        acc += tl.dot(x_chunk, y_chunk, allow_tf32=False)

    # 根据has_bias标志决定是否添加偏置
    if has_bias:
        # 加载偏置值（广播到所有行）
        bias_ptrs = bias_ptr + rn * stride_bias
        bias_mask = rn < None
        bias_vals = tl.load(bias_ptrs, mask=bias_mask,
                            other=0.0).to(tl.float32)
        # 将偏置加到累加器上（自动广播）
        acc += bias_vals[None, :]

    # 计算输出指针位置
    out_ptrs = output_ptr + rm[:,
                               None] * stride_outm + rn[None, :] * stride_outn
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)

    # 将结果存储到全局内存
    tl.store(out_ptrs, acc.to(out_ptrs.dtype.element_ty), mask=out_mask)


def matmul_persistent(x, y, bias=None):
    """
    使用Triton实现矩阵乘法加可选偏置: x @ y + bias (如果bias不为None)
                
    参数:
        x: torch.Tensor, 形状为 [M, K]
        y: torch.Tensor, 形状为 [K, N] 
        bias: torch.Tensor, 形状为 [N] 或 None
                                                
    返回:
        output: torch.Tensor, 形状为 [M, N]
    """
    # 验证输入形状
    assert x.dim() == 2, "x必须是2D张量"
    assert y.dim() == 2, "y必须是2D张量"
    assert x.shape[1] == y.shape[
        0], f"矩阵维度不匹配: x.shape[1]={x.shape[1]}, y.shape[0]={y.shape[0]}"

    M, K = x.shape
    _, N = y.shape
    # 验证bias形状（如果不为None）
    if bias is not None:
        assert bias.dim() == 1, "bias必须是1D张量"
        assert y.shape[1] == bias.shape[
            0], f"偏置维度不匹配: y.shape[1]={y.shape[1]}, bias.shape[0]={bias.shape[0]}"

    # 分配输出张量（与x相同的数据类型）
    output = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # 定义分块大小（可根据硬件调整）
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128

    # 计算网格大小（每个分块一个线程）
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # 处理bias为None的情况
    if bias is None:
        # 创建一个虚拟的bias张量（不会被使用，因为has_bias=False）
        dummy_bias = torch.empty(0, dtype=x.dtype, device=x.device)
        has_bias = False
        bias_stride = 0
        bias_to_pass = dummy_bias
    else:
        has_bias = True
        bias_stride = bias.stride(0)
        bias_to_pass = bias
    # 启动kernel
    matmul_bias_persistent_kernel[grid](
        x,
        y,
        bias_to_pass,
        output,  # 输入输出张量
        M,
        N,
        K,  # 矩阵维度
        x.stride(0),
        x.stride(1),  # x的步长
        y.stride(0),
        y.stride(1),  # y的步长  
        bias_stride,  # bias的步长（如果bias为None则为0）
        output.stride(0),
        output.stride(1),  # 输出的步长
        has_bias,  # 是否使用偏置的标志
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output


@triton.jit
def linear_persistent_kernel(
        a_ptr,  # 指针指向张量 a，形状 [M, K]
        b_ptr,  # 指针指向张量 b，形状 [N, K]
        c_ptr,  # 指针指向输出张量 c，形状 [M, N]
        M,  # 张量 a 的行数
        N,  # 张量 b 的行数（输出 c 的列数）
        K,  # 张量 a 的列数和张量 b 的列数
        stride_am,  # 张量 a 在维度 M 的步长（通常为 K）
        stride_ak,  # 张量 a 在维度 K 的步长（通常为 1）
        stride_bn,  # 张量 b 在维度 N 的步长（通常为 K）
        stride_bk,  # 张量 b 在维度 K 的步长（通常为 1）
        stride_cm,  # 张量 c 在维度 M 的步长（通常为 N）
        stride_cn,  # 张量 c 在维度 N 的步长（通常为 1）
        BLOCK_M: tl.constexpr,  # 阻塞大小 for M 维度
        BLOCK_N: tl.constexpr,  # 阻塞大小 for N 维度
        BLOCK_K: tl.constexpr,  # 阻塞大小 for K 维度
        NUM_BLOCKS_M: tl.constexpr,  # 新增：M 维度的块数
        NUM_BLOCKS_N: tl.constexpr,  # 新增：N 维度的块数
        GRID_SIZE: tl.constexpr,  # 新增：固定的一维网格大小
):
    # 获取当前程序的一维索引（一维网格）
    pid = tl.program_id(0)
    total_blocks = NUM_BLOCKS_M * NUM_BLOCKS_N  # 总输出块数

    # 循环处理分配给当前 program 的多个块（类似知识片段7的循环策略）
    for block_index in range(pid, total_blocks, GRID_SIZE):
        # 将一维块索引转换为二维坐标 (m_block, n_block)
        m_block = block_index // NUM_BLOCKS_N
        n_block = block_index % NUM_BLOCKS_N

        # 计算当前输出块的起始索引
        start_m = m_block * BLOCK_M
        start_n = n_block * BLOCK_N

        # 创建当前块内的行和列索引范围
        m_indices = start_m + tl.arange(0, BLOCK_M)
        n_indices = start_n + tl.arange(0, BLOCK_N)

        # 创建掩码以处理边界
        m_mask = m_indices < M
        n_mask = n_indices < N

        # 初始化累加器为0
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # 循环遍历 K 维度，以 BLOCK_K 为步长进行阻塞
        for k_offset in range(0, K, BLOCK_K):
            k_indices = k_offset + tl.arange(0, BLOCK_K)
            k_mask = k_indices < K

            # 加载张量 a 的块：形状 [BLOCK_M, BLOCK_K]
            a_ptrs = a_ptr + m_indices[:, None] * stride_am + k_indices[
                None, :] * stride_ak
            a_vals = tl.load(a_ptrs,
                             mask=m_mask[:, None] & k_mask[None, :],
                             other=0.0)

            # 加载张量 b 的块：形状 [BLOCK_N, BLOCK_K]
            b_ptrs = b_ptr + n_indices[:, None] * stride_bn + k_indices[
                None, :] * stride_bk
            b_vals = tl.load(b_ptrs,
                             mask=n_mask[:, None] & k_mask[None, :],
                             other=0.0)

            # 使用 tl.trans 显式转置 b 矩阵：形状变为 [BLOCK_K, BLOCK_N]
            b_vals_transposed = tl.trans(b_vals)

            # 计算矩阵乘法：a_vals × b_vals_transposed
            product = tl.dot(a_vals, b_vals_transposed)
            acc += product
        # 将结果存储到输出张量 c
        c_ptrs = c_ptr + m_indices[:, None] * stride_cm + n_indices[
            None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def linear_persistent(x, y):
    """
    使用Triton实现矩阵乘法加可选偏置: x @ y^T
    使用固定大小的一维网格
                    
    参数:
        x: torch.Tensor, 形状为 [M, K]
        y: torch.Tensor, 形状为 [N, K] 
                                                    
    返回:
        output: torch.Tensor, 形状为 [M, N]
    """
    # 验证输入形状
    assert x.dim() == 2, "x必须是2D张量"
    assert y.dim() == 2, "y必须是2D张量"
    assert x.shape[1] == y.shape[
        1], f"矩阵维度不匹配: x.shape[1]={x.shape[1]}, y.shape[1]={y.shape[1]}"

    M, K = x.shape
    N, _ = y.shape

    # 分配输出张量（与x相同的数据类型）
    output = torch.zeros((M, N), dtype=x.dtype, device=x.device)

    # 定义分块大小（可根据硬件调整）
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128

    # 计算每个维度的块数（向上取整）
    num_blocks_m = triton.cdiv(M, BLOCK_M)
    num_blocks_n = triton.cdiv(N, BLOCK_N)

    # 设置固定的一维网格
    grid_size = driver.active.utils.get_device_properties(
        torch.npu.current_device())["num_vectorcore"] // 2
    grid = (grid_size, )

    # 启动kernel
    linear_persistent_kernel[grid](
        a_ptr=x,
        b_ptr=y,
        c_ptr=output,
        M=M,
        N=N,
        K=K,
        stride_am=x.stride(0),
        stride_ak=x.stride(1),
        stride_bn=y.stride(0),
        stride_bk=y.stride(1),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_BLOCKS_M=num_blocks_m,  # 传入M维度块数
        NUM_BLOCKS_N=num_blocks_n,  # 传入N维度块数
        GRID_SIZE=grid_size,  # 传入固定网格大小
    )

    return output


def mm_batch_invariant(a, b):
    return matmul_persistent(a, b)


def bmm_batch_invariant(a, b, *, out=None):
    # Batched matrix multiply: (B, M, K) x (B, K, N) -> (B, M, N)
    # Process each batch separately with our persistent kernel
    if a.ndim == 3 and b.ndim == 3:
        results = []
        for i in range(a.shape[0]):
            results.append(matmul_persistent(a[i], b[i]))
        result = torch.stack(results, dim=0)

        if out is not None:
            out.copy_(result)
            return out
        return result
    else:
        raise ValueError(f"bmm_batch_invariant expects 3D tensors, "
                         f"got shapes {a.shape} and {b.shape}")


def addmm_batch_invariant(bias, a, b):
    return matmul_persistent(a, b, bias=bias)


def matmul_batch_invariant(a, b, *, out=None):
    # torch.matmul can handle various dimensions
    # For 2D x 2D, it's the same as matmul
    if a.ndim == 2 and b.ndim == 2:
        result = matmul_persistent(a, b)
        if out is not None:
            out.copy_(result)
            return out
        return result
    elif a.ndim == 3 and b.ndim == 3:
        # Handle batched case like bmm
        return bmm_batch_invariant(a, b, out=out)
    elif a.ndim == 3 and b.ndim == 2:
        # Handle 3D x 2D: common for linear layers
        # (batch, seq, hidden) @ (hidden, out) -> (batch, seq, out)
        # Reshape to 2D, do mm, reshape back
        batch, seq, hidden = a.shape
        a_2d = a.reshape(-1, hidden)
        result_2d = matmul_persistent(a_2d, b)
        result = result_2d.reshape(batch, seq, -1)
        if out is not None:
            out.copy_(result)
            return out
        return result
    elif a.ndim == 2 and b.ndim == 3:
        # Handle 2D x 3D: (M, K) @ (B, K, N) -> (B, M, N)
        # By broadcasting `a` to 3D, we can reuse the batched matrix
        # multiplication logic.
        a_expanded = a.unsqueeze(0).expand(b.shape[0], -1, -1)
        return bmm_batch_invariant(a_expanded, b, out=out)
    elif a.ndim == 4 and b.ndim == 4:
        # Handle 4D attention tensors: [batch, heads, seq, dim]
        # Reshape to 3D, process, reshape back
        batch, heads, seq_a, dim_a = a.shape
        _, _, dim_b, seq_b = b.shape

        # Reshape to [batch*heads, seq_a, dim_a]
        a_3d = a.reshape(batch * heads, seq_a, dim_a)
        b_3d = b.reshape(batch * heads, dim_b, seq_b)

        # Do batched matmul
        result_3d = bmm_batch_invariant(a_3d, b_3d)

        # Reshape back to [batch, heads, seq_a, seq_b]
        result = result_3d.reshape(batch, heads, seq_a, seq_b)

        if out is not None:
            out.copy_(result)
            return out
        return result
    else:
        raise ValueError(
            f"matmul_batch_invariant currently only supports 2D x 2D, 3D x 3D, "
            f"3D x 2D, 2D x 3D, and 4D x 4D, "
            f"got shapes {a.shape} and {b.shape}")


def linear_batch_invariant(input, weight, bias=None):
    output = linear_persistent(input, weight)

    if bias is not None:
        output = output + bias
    return output
