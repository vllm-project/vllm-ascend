/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_situ_quant_activation.h
 * \brief SiTU-GLU 激活（Kimi K3 SiTUGLU）在 A2/A3 UB 上的向量实现。
 *
 *   out = beta * tanh(gate / beta) * sigmoid(gate) * up
 *
 * 其中 up 在 linear_beta > 0 时先做一次线性缩放：
 *   up  = linear_beta * tanh(up / linear_beta)
 *
 * 与 GroupedMatmulSwigluQuantV2 的 A8W4 MSD epilogue 相比，本实现是**原地**的：
 * 结果直接写回行块的前半区，不再经 reduceWorkspace 中转一次 DataCopy，省掉
 * 每行 H 个 float 的 UB->UB 搬运。同时默认参数下（beta=1、linear_beta<=0）
 * 会自动跳过 up 变换与 beta 缩放共 5 遍向量计算。
 */
#ifndef OP_KERNEL_GROUPED_MATMUL_SITU_QUANT_ACTIVATION_H
#define OP_KERNEL_GROUPED_MATMUL_SITU_QUANT_ACTIVATION_H

#include "kernel_operator.h"
#include "adv_api/math/tanh.h"

namespace GroupedMatmulSituQuantActivation {
using namespace AscendC;

// Tanh 高阶 API 内部按 tmpBufferSize/32*32 计算 stackSize 并据此分轮，
// tmpBuffer 小于 32 个 float 时 stackSize 会退化为 0 并触发除零。
// 因此 tmp 至少要给到 32 个 float。
constexpr uint32_t SITU_TANH_TMP_MIN_ELEMS = 32;

// 快速路径里 u = exp(-gate) 的上限。u 只在 gate 很负时才大，钳到 1e18 后
// u² = 1e36 不会上溢成 inf，(1-u²)/(1+u²) 此时已经等于 -1，不引入可见偏差。
constexpr float SITU_EXP_U_CLAMP = 1e18f;

// SiTU 的参数集，与 tiling 侧 GMMSituQuant 的对应字段一致。
struct SituParams {
    float beta = 1.0f;          // 必须 > 0
    float linearBeta = 0.0f;    // <= 0 表示不启用 up 分支变换
    int64_t activateLeft = 1;   // 1: gate 取前半、up 取后半；0: 反之
};

// 计算 Tanh 所需的 tmp 元素个数（单位：float 元素）。
__aicore__ inline uint32_t SituTanhTmpElems(uint32_t halfLen)
{
    return halfLen < SITU_TANH_TMP_MIN_ELEMS ? SITU_TANH_TMP_MIN_ELEMS : halfLen;
}

/*!
 * \brief 对行块 x（长度 2*halfLen）原地做 SiTU-GLU，结果写到 x[0, halfLen)。
 * \param x        行块，前 halfLen 为前半区、后 halfLen 为后半区；结果写回前半区
 * \param work     长度 >= halfLen 的 float 暂存
 * \param tanhTmp  长度 >= SituTanhTmpElems(halfLen)*sizeof(float) 字节的 Tanh 暂存
 * \param halfLen  半区长度 H（= tokenLen/2）
 * \param params   SiTU 参数
 *
 * 说明：AscendC 的 Tanh 实现会先把 src 钳位后拷到独立的 tmpClip，之后不再读 src，
 * 因此 dst == src 的原地调用是安全的。
 * 另外 gate 会被内部钳位到 [-8.8, 8.8]，由于 tanh 在 |x|>8.8 处已饱和到 1
 * （tanh(8.8) = 0.99999995，相对误差 5e-8，小于 fp32 eps），该钳位不引入可见偏差。
 */
__aicore__ inline void SituGluCompute(const LocalTensor<float>& x, const LocalTensor<float>& work,
                                      const LocalTensor<uint8_t>& tanhTmp, uint32_t halfLen,
                                      const SituParams& params)
{
    const uint32_t gateOffset = (params.activateLeft == 1) ? 0 : halfLen;
    const uint32_t upOffset = (params.activateLeft == 1) ? halfLen : 0;

    LocalTensor<float> gate = x[gateOffset];
    LocalTensor<float> up = x[upOffset];
    // 结果固定落在前半区，与 Quant 的读取偏移（行块起点）保持一致。
    LocalTensor<float> out = x;

    // 1. up 分支：仅当 linear_beta > 0 时启用，默认路径直接跳过。
    if (params.linearBeta > 0.0f) {
        Muls(up, up, 1.0f / params.linearBeta, halfLen);
        PipeBarrier<PIPE_V>();
        Tanh(up, up, tanhTmp, halfLen);
        PipeBarrier<PIPE_V>();
        Muls(up, up, params.linearBeta, halfLen);
        PipeBarrier<PIPE_V>();
    }

    if (params.beta == 1.0f) {
        // 快速路径（beta == 1，即默认参数与 vllm-ascend 的调用约定）：
        //
        //   out = up * tanh(g) * sigmoid(g)，令 u = exp(-g)，则
        //     sigmoid(g) = 1/(1+u)，tanh(g) = (1-u²)/(1+u²)
        //     ⇒ tanh(g) * sigmoid(g) = (1-u) / (1+u²)
        //   整条激活只要 1 次 Exp + 1 次 Div。
        //
        // 原来的写法是 Exp(-g) 求 sigmoid + Tanh 高阶 API 求 tanh；后者内部就是
        // Mins/Maxs/Muls/Exp/Adds/Adds/Div 七遍（见 adv_api/.../tanh_common_impl.h），
        // 加起来 2 次 Exp + 2 次 Div。Exp/Div 是向量单元里最贵的两条，这一步把它们
        // 各砍掉一次 —— 实测 prefill NZ（E=14/M=2048）的 Vector ALU 79.0 → 68.7 µs、
        // aiv_time 359.8 → 349.2 µs、Task 365.4 → 355.2 µs。
        //
        // g → -∞ 时 u → +∞，u² 会上溢成 inf、再算出 NaN，所以对 u 做一次 Mins 钳位：
        // u ≤ 1e18 时 u² ≤ 1e36 仍在 fp32 范围内，而此时 (1-u²)/(1+u²) 已经精确等于
        // -1，与 Tanh API 把 g 钳到 -8.8 的语义一致（两者量化后不可区分）。
        //
        // 注意 out 与 gate 是同一块内存（activateLeft == 1 时 gateOffset == 0），
        // 所以分母 (1+u²) 必须放进 scratch，不能留在 gate 里 —— 否则写 out 时会把它盖掉。
        LocalTensor<float> scratch = tanhTmp.ReinterpretCast<float>();
        Muls(work, gate, -1.0f, halfLen);
        PipeBarrier<PIPE_V>();
        Exp(work, work, halfLen); // work = u
        PipeBarrier<PIPE_V>();
        Mins(work, work, SITU_EXP_U_CLAMP, halfLen);
        PipeBarrier<PIPE_V>();
        Mul(scratch, work, work, halfLen);
        PipeBarrier<PIPE_V>();
        Adds(scratch, scratch, 1.0f, halfLen); // scratch = 1 + u²（分母）
        PipeBarrier<PIPE_V>();
        Muls(work, work, -1.0f, halfLen);
        PipeBarrier<PIPE_V>();
        Adds(work, work, 1.0f, halfLen); // work = 1 - u（分子）
        PipeBarrier<PIPE_V>();
        Mul(out, up, work, halfLen); // out = up * (1-u)，此处 gate 已用完
        PipeBarrier<PIPE_V>();
        Div(out, out, scratch, halfLen); // out /= (1+u²)
        PipeBarrier<PIPE_V>();
        return;
    }

    // 2. work = 1 + exp(-gate)。gate 很负时 exp 上溢为 inf，Div 后得到 0，
    //    恰好是 sigmoid 的极限值，无需额外钳位。
    Muls(work, gate, -1.0f, halfLen);
    PipeBarrier<PIPE_V>();
    Exp(work, work, halfLen);
    PipeBarrier<PIPE_V>();
    Adds(work, work, 1.0f, halfLen);
    PipeBarrier<PIPE_V>();

    // 3. work = up * sigmoid(gate) = up / (1 + exp(-gate))
    Div(work, up, work, halfLen);
    PipeBarrier<PIPE_V>();


    // 4. gate = beta * tanh(gate / beta)，beta == 1 时省掉两次 Muls。
    if (params.beta != 1.0f) {
        Muls(gate, gate, 1.0f / params.beta, halfLen);
        PipeBarrier<PIPE_V>();
    }
    Tanh(gate, gate, tanhTmp, halfLen);
    PipeBarrier<PIPE_V>();
    if (params.beta != 1.0f) {
        Muls(gate, gate, params.beta, halfLen);
        PipeBarrier<PIPE_V>();
    }

    // 5. out = work * gate
    Mul(out, work, gate, halfLen);
    PipeBarrier<PIPE_V>();
}

} // namespace GroupedMatmulSituQuantActivation

#endif // OP_KERNEL_GROUPED_MATMUL_SITU_QUANT_ACTIVATION_H
