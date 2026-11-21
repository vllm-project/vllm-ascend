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
 * \file situ_activation.h
 * \brief GroupedMatmulSituQuant 的激活策略：SiTU。
 *
 * 复用 grouped_matmul_swiglu_quant_v2 的 MxA8W4 VCV 基础块（GMMSQWeightQuantVcvBasicBlock）与 MX 动态量化流程，
 * 仅把门控激活由 SwiGLU 换成 SiTU：
 *     situ_a = beta * tanh(gate / beta) * sigmoid(gate)
 *     up'    = linearBeta * tanh(up / linearBeta)        (linearBeta != 0 时)
 *     out    = situ_a * up'
 *
 * 数值约定（与仓库内生产 SiTU 实现 mc2/mega_moe .../activation/situglu_activation.h 逐指令一致）：
 *   1) GMM 输出为缩小 64 倍的 F32，激活前先乘回 64（与 SwiGLU 路径的 mulFactor 同源）；
 *   2) 乘回后按 BF16 舍入再参与激活（生产路径经 UB 落 BF16 再读回，此处用寄存器 Cast 往返达到同等舍入）；
 *   3) tanh 两路径：|x| < 0.6 用 9 次多项式 Horner；|x| >= 0.6 用 2 * sigmoid(2x) - 1；
 *   4) 激活结果落 BF16 平面，后续 ComputeMaxExp/ComputeScale/ComputeDataForQuantTargetFp8
 *      与 SwiGLU 路径完全共用（量化与搬运零改动）。
 */
#ifndef GMSQ_SITU_ACTIVATION_H
#define GMSQ_SITU_ACTIVATION_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

namespace GMMSQWeightQuant {

namespace SituReg = AscendC::Reg;

// tanh 分段近似系数与切换阈值，取自生产 SiTU 实现（数值不可调整，否则与 golden 失配）
static constexpr float SITU_TANH_C1 = -0.333327681f;
static constexpr float SITU_TANH_C2 = 0.133152977f;
static constexpr float SITU_TANH_C3 = -0.0523039624f;
static constexpr float SITU_TANH_C4 = 0.0157396831f;
static constexpr float SITU_TANH_THRESHOLD = 0.60000002384185791016f;
// GMM 输出缩小 64 倍，激活前补齐
static constexpr float SITU_DEQ_FACTOR = 64.0f;

static constexpr SituReg::CastTrait SITU_CAST_FP32_TO_BF16 = {
    SituReg::RegLayout::ZERO, SituReg::SatMode::NO_SAT, SituReg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr SituReg::CastTrait SITU_CAST_BF16_TO_FP32 = {
    SituReg::RegLayout::ZERO, SituReg::SatMode::UNKNOWN, SituReg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

/*!
 * \brief SiTU 激活 VF：F32 中继（前 nSize/2 为 gate 支路、后 nSize/2 为 up 支路）-> BF16 结果。
 *
 * 行独立逐元素运算，与量化阶段的 per-32 归约互不跨行，故按行/按向量块循环不改变位级结果。
 *
 * \param[in] gateSrc           gate 支路 F32 起点（各行跨度 nSrcUbAligned）
 * \param[in] upSrc             up 支路 F32 起点（各行跨度 nSrcUbAligned）
 * \param[in] situResAddr       输出 BF16 平面（各行跨度 nDstUbAligned）
 * \param[in] mSize             行数
 * \param[in] nSize             每行元素数（nSize/2 为输出列数，64 的整数倍）
 */
__simd_vf__ inline void GmmSituVf(__ubuf__ float *gateSrc, __ubuf__ float *upSrc, __ubuf__ bfloat16_t *situResAddr,
                                  uint16_t mSize, uint16_t nSize, uint32_t nSrcUbAligned, uint32_t nDstUbAligned,
                                  uint16_t oneRowRepeatTimes, uint16_t sizePerRepeat, float beta, float invBeta,
                                  float linearBeta, float invLinearBeta)
{
    SituReg::RegTensor<float> gateF, upF;
    SituReg::RegTensor<float> zReg, squareReg, polyReg, tanhReg, tanhSelReg;
    SituReg::RegTensor<float> expReg, denReg, negReg, outF, oneReg, c1Reg, c2Reg, betaReg;
    SituReg::RegTensor<bfloat16_t> gateB, upB, outB;
    SituReg::MaskReg cmpMask;

    SituReg::Duplicate(oneReg, 1.0f);
    SituReg::Duplicate(c1Reg, SITU_TANH_C1);
    SituReg::Duplicate(c2Reg, SITU_TANH_C2);

    for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
        uint32_t elementNum = nSize;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < oneRowRepeatTimes; ++vfBlockIdx) {
            SituReg::MaskReg mask = SituReg::UpdateMask<float>(elementNum);
            uint32_t srcOff = mIdx * nSrcUbAligned + vfBlockIdx * sizePerRepeat;
            uint32_t dstOff = mIdx * nDstUbAligned + vfBlockIdx * sizePerRepeat;

            // GMM 输出补齐 64 倍，并按 BF16 舍入（对齐生产路径的落盘 BF16 往返）
            SituReg::LoadAlign(gateF, gateSrc + srcOff);
            SituReg::Muls(gateF, gateF, SITU_DEQ_FACTOR, mask);
            SituReg::LoadAlign(upF, upSrc + srcOff);
            SituReg::Muls(upF, upF, SITU_DEQ_FACTOR, mask);
            SituReg::Cast<bfloat16_t, float, SITU_CAST_FP32_TO_BF16>(gateB, gateF, mask);
            SituReg::Cast<bfloat16_t, float, SITU_CAST_FP32_TO_BF16>(upB, upF, mask);
            SituReg::Cast<float, bfloat16_t, SITU_CAST_BF16_TO_FP32>(gateF, gateB, mask);
            SituReg::Cast<float, bfloat16_t, SITU_CAST_BF16_TO_FP32>(upF, upB, mask);

            // ---- tanh(gate / beta)：多项式路径 ----
            SituReg::Muls(zReg, gateF, invBeta, mask);
            SituReg::Mul(squareReg, zReg, zReg, mask);
            SituReg::Muls(polyReg, squareReg, SITU_TANH_C4, mask);
            SituReg::Adds(polyReg, polyReg, SITU_TANH_C3, mask);
            SituReg::MulDstAdd(polyReg, squareReg, c2Reg, mask);
            SituReg::MulDstAdd(polyReg, squareReg, c1Reg, mask);
            SituReg::Mul(polyReg, polyReg, squareReg, mask);
            SituReg::MulDstAdd(polyReg, zReg, zReg, mask);
            // ---- tanh(gate / beta)：sigmoid 路径 2 * sigmoid(2z) - 1 ----
            SituReg::Muls(expReg, zReg, -2.0f, mask);
            SituReg::Exp(expReg, expReg, mask);
            SituReg::Adds(expReg, expReg, 1.0f, mask);
            SituReg::Div(tanhReg, oneReg, expReg, mask);
            SituReg::Muls(tanhReg, tanhReg, 2.0f, mask);
            SituReg::Adds(tanhReg, tanhReg, -1.0f, mask);
            // 按 |z| >= 0.6 选路
            SituReg::Abs(squareReg, zReg, mask);
            SituReg::CompareScalar<float, AscendC::CMPMODE::GE>(cmpMask, squareReg, SITU_TANH_THRESHOLD, mask);
            SituReg::Select(tanhSelReg, tanhReg, polyReg, cmpMask);

            // ---- situ_a = tanh(gate/beta) * (beta / (1 + exp(-gate))) ----
            SituReg::Muls(negReg, gateF, -1.0f, mask);
            SituReg::Exp(expReg, negReg, mask);
            SituReg::Adds(denReg, expReg, 1.0f, mask);
            SituReg::Duplicate(betaReg, beta, mask);
            SituReg::Div(expReg, betaReg, denReg, mask);
            SituReg::Mul(outF, tanhSelReg, expReg, mask);

            // ---- up' = linearBeta * tanh(up / linearBeta) ----
            SituReg::Muls(zReg, upF, invLinearBeta, mask);
            SituReg::Mul(squareReg, zReg, zReg, mask);
            SituReg::Muls(polyReg, squareReg, SITU_TANH_C4, mask);
            SituReg::Adds(polyReg, polyReg, SITU_TANH_C3, mask);
            SituReg::MulDstAdd(polyReg, squareReg, c2Reg, mask);
            SituReg::MulDstAdd(polyReg, squareReg, c1Reg, mask);
            SituReg::Mul(polyReg, polyReg, squareReg, mask);
            SituReg::MulDstAdd(polyReg, zReg, zReg, mask);
            SituReg::Muls(expReg, zReg, -2.0f, mask);
            SituReg::Exp(expReg, expReg, mask);
            SituReg::Adds(expReg, expReg, 1.0f, mask);
            SituReg::Div(tanhReg, oneReg, expReg, mask);
            SituReg::Muls(tanhReg, tanhReg, 2.0f, mask);
            SituReg::Adds(tanhReg, tanhReg, -1.0f, mask);
            SituReg::Abs(squareReg, zReg, mask);
            SituReg::CompareScalar<float, AscendC::CMPMODE::GE>(cmpMask, squareReg, SITU_TANH_THRESHOLD, mask);
            SituReg::Select(tanhSelReg, tanhReg, polyReg, cmpMask);
            SituReg::Muls(tanhSelReg, tanhSelReg, linearBeta, mask);
            SituReg::Mul(outF, outF, tanhSelReg, mask);

            SituReg::Cast<bfloat16_t, float, SITU_CAST_FP32_TO_BF16>(outB, outF, mask);
            SituReg::StoreAlign<bfloat16_t, SituReg::StoreDist::DIST_PACK_B32>(situResAddr + dstOff, outB, mask);
        }
    }
}

/*!
 * \brief 激活策略：SiTU（作为 GMMSQWeightQuantVcvBasicBlock 的 actType 模板实参传入）。
 *
 * 签名与 GmmVecActivationSwiGLU 完全一致，因此 VCV 基础块/MX 量化/搬运路径无需任何改动。
 * linearBeta 由 host tiling 保证 > 0（默认 1.0，即 up 支路退化为标准 tanh）。
 */
struct GmmVecActivationSitu {
    __aicore__ inline void Compute(__ubuf__ float *firstSrc, __ubuf__ float *secondSrc, __ubuf__ bfloat16_t *actRes,
                                   uint16_t mSize, uint16_t nSize, uint32_t nSrcUbAligned, uint32_t nDstUbAligned,
                                   uint16_t oneRowRepeatTimes, uint16_t sizePerRepeat, float beta, float invBeta,
                                   float linearBeta, float invLinearBeta)
    {
        GmmSituVf(firstSrc, secondSrc, actRes, mSize, nSize, nSrcUbAligned, nDstUbAligned, oneRowRepeatTimes,
                  sizePerRepeat, beta, invBeta, linearBeta, invLinearBeta);
    }
};

} // namespace GMMSQWeightQuant

#endif // GMSQ_SITU_ACTIVATION_H
