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
 * \file vf_quant_perchannel.h
 * \brief
 */

#ifndef VF_QUANT_PERCHANNEL_H
#define VF_QUANT_PERCHANNEL_H
#include "kernel_tensor.h"

namespace MlaProlog {

template <typename T, typename C, typename O>
__simd_vf__ void QuantChannelVFImpl(__ubuf__ O *yAddr, __ubuf__ T *xAddr, __ubuf__ C *quantScaleAddr,
                                    const uint32_t floatRepSize, uint32_t dLoops, uint32_t dTail, uint32_t dTailLoop,
                                    uint32_t row, uint32_t col, uint32_t stride)
{
    AscendC::MicroAPI::RegTensor<T> vregInput;
    AscendC::MicroAPI::RegTensor<C> vregQuantScale;
    AscendC::MicroAPI::RegTensor<O> vregOutput;
    AscendC::MicroAPI::RegTensor<half> vregOutputHalf; // float-->half-->int8
    AscendC::MicroAPI::MaskReg fullMask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg tailMask;
    tailMask = AscendC::MicroAPI::UpdateMask<float>(dTail);
    constexpr static AscendC::MicroAPI::CastTrait castTraitPack2 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32ToHalf = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ODD};

    uint32_t colOffset = 0;
    uint32_t rowOffset = 0;
    for (uint32_t j = 0; j < dLoops; j++) {
        AscendC::MicroAPI::LoadAlign<C, AscendC::MicroAPI::LoadDist::DIST_NORM>(vregQuantScale,
                                                                                quantScaleAddr + colOffset);
        rowOffset = 0;
        for (uint32_t i = 0; i < row; i++) {
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_NORM>(vregInput,
                                                                                    xAddr + colOffset + rowOffset);
            AscendC::MicroAPI::Mul(vregInput, vregInput, vregQuantScale, fullMask);
            AscendC::MicroAPI::Cast<half, float, castTraitF32ToHalf>(vregOutputHalf, vregInput, fullMask);
            AscendC::MicroAPI::Cast<O, half, castTraitPack2>(vregOutput, vregOutputHalf, fullMask);
            AscendC::MicroAPI::StoreAlign<O, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                yAddr + colOffset + rowOffset, vregOutput, fullMask);
            rowOffset += stride;
        }
        colOffset += floatRepSize;
    }

    if (dTailLoop > 0) {
        rowOffset = 0;
        AscendC::MicroAPI::LoadAlign<C, AscendC::MicroAPI::LoadDist::DIST_NORM>(vregQuantScale,
                                                                                quantScaleAddr + dLoops * floatRepSize);
        for (uint32_t i = 0; i < row; i++) {
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                vregInput, xAddr + dLoops * floatRepSize + rowOffset);
            AscendC::MicroAPI::Mul(vregInput, vregInput, vregQuantScale, tailMask);
            AscendC::MicroAPI::Cast<half, float, castTraitF32ToHalf>(vregOutputHalf, vregInput, tailMask);
            AscendC::MicroAPI::Cast<O, half, castTraitPack2>(vregOutput, vregOutputHalf, tailMask);
            AscendC::MicroAPI::StoreAlign<O, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                yAddr + dLoops * floatRepSize + rowOffset, vregOutput, tailMask);
            rowOffset += stride;
        }
    }
}

/**
 * @brief QuantPerChannelVF 同时对row进行FP32到int8的per-channel量化操作。一行中的每一列用不同的量化参数。
          outLocal[i , j] = inputLocal[i , j] * quantScaleLocal[j]
 * @param outputLocal 输出tensor [row, col]
 * @param inputLocal 输入tensor [row, col]
 * @param quantScaleLocal 量化参数tensor [1, col]
 * @param row 处理数据的行数 默认为1 后续可扩展
 * @param col 处理数据的列数
 * @param stride 待处理数据一行的真实长度
 */
template <typename T, typename C, typename O>
__aicore__ inline void QuantPerChannelVf(const LocalTensor<O> &outputLocal, const LocalTensor<T> &inputLocal,
                                         const LocalTensor<C> &quantScaleLocal, const uint32_t row, const uint32_t col,
                                         const uint32_t stride)
{
    __ubuf__ O *outputUb = (__ubuf__ O *)outputLocal.GetPhyAddr();
    __ubuf__ T *inputUb = (__ubuf__ T *)inputLocal.GetPhyAddr();
    __ubuf__ C *quantScaleUb = (__ubuf__ C *)quantScaleLocal.GetPhyAddr();

    const uint32_t floatRepSize = 64; // 一个寄存器能够存放64个FP32
    uint32_t dLoops = col / floatRepSize;
    uint32_t dTail = col % floatRepSize;
    uint32_t dTailLoop = dTail > 0 ? 1 : 0;

    QuantChannelVFImpl(outputUb, inputUb, quantScaleUb, floatRepSize, dLoops, dTail, dTailLoop, row, col, stride);
}
} // namespace MlaProlog
#endif