/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_regbase.h
 * \brief
 */
#ifndef CAUSAL_CONV1D_REGBASE_H
#define CAUSAL_CONV1D_REGBASE_H

namespace NsCausalConv1d {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint16_t V_LENGTH = VECTOR_REG_WIDTH / sizeof(float);

constexpr CastTrait castTraitB16ToB32 = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <typename T, bool hasActivation>
__aicore__ inline void ComputeFnRollingOutputRegbase(LocalTensor<T> ring, LocalTensor<float> currF, 
    LocalTensor<float> state0F, LocalTensor<float> weightF, uint32_t dataCount) 
{
    __ubuf__ T* ringAddr = (__ubuf__ T*)ring.GetPhyAddr();
    __ubuf__ float* currFAddr = (__ubuf__ float*)currF.GetPhyAddr();
    __ubuf__ float* state0FAddr = (__ubuf__ float*)state0F.GetPhyAddr();
    __ubuf__ float* weightFAddr = (__ubuf__ float*)weightF.GetPhyAddr();

    uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(dataCount, V_LENGTH));
    __VEC_SCOPE__
    {
        RegTensor<T> ring;
        RegTensor<float> currF;
        RegTensor<float> state0F;
        RegTensor<float> weightF;
        RegTensor<float> tmp;
        MaskReg pregLoop;
        for (uint16_t j = 0; j < colLoopTimes; j++) {
            pregLoop = UpdateMask<float>(dataCount);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(ring, ringAddr + j * V_LENGTH);
            DataCopy(state0F, state0FAddr + j * V_LENGTH);
            DataCopy(weightF, weightFAddr + j * V_LENGTH);
            Cast<float, T, castTraitB16ToB32>(currF, ring, pregLoop);
            Mul(currF, currF, weightF, pregLoop);
            Add(state0F, state0F, currF, pregLoop);
            if constexpr (hasActivation) {
                Muls(tmp, state0F, -1.0f, pregLoop);
                Exp(tmp, tmp, pregLoop);
                Adds(tmp, tmp, 1.0f, pregLoop);
                Div(currF, state0F, tmp, pregLoop);
                DataCopy(currFAddr + j * V_LENGTH, currF, pregLoop);
            } else {
                DataCopy(state0FAddr + j * V_LENGTH, state0F, pregLoop);
            }
        }
    }
}

template <typename T>
static __simd_vf__ inline void AdvanceFnLocalPartialsWidthTwo(__ubuf__ T* ringAddr, __ubuf__ float* weight0FAddr, 
    __ubuf__ float* state0FAddr, uint32_t dataCount, uint16_t colLoopTimes)
{
    RegTensor<T> ring;
    RegTensor<float> currF;
    RegTensor<float> weight0F;
    RegTensor<float> state0F;
    MaskReg pregLoop;
    for (uint16_t j = 0; j < colLoopTimes; j++) {
        pregLoop = UpdateMask<float>(dataCount);
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(ring, ringAddr + j * V_LENGTH);
        DataCopy(weight0F, weight0FAddr + j * V_LENGTH);
        Cast<float, T, castTraitB16ToB32>(currF, ring, pregLoop);
        Mul(state0F, currF, weight0F, pregLoop);
        DataCopy(state0FAddr + j * V_LENGTH, state0F, pregLoop);
    }
}

template <typename T>
static __simd_vf__ inline void AdvanceFnLocalPartialsWidthThree(__ubuf__ T* ringAddr, __ubuf__ float* weight0FAddr, 
    __ubuf__ float* weight1FAddr, __ubuf__ float* state0FAddr, __ubuf__ float* state1FAddr, uint32_t dataCount, 
    uint16_t colLoopTimes)
{
    RegTensor<T> ring;
    RegTensor<float> currF;
    RegTensor<float> weight0F;
    RegTensor<float> weight1F;
    RegTensor<float> state0F;
    RegTensor<float> state1F;
    MaskReg pregLoop;
    for (uint16_t j = 0; j < colLoopTimes; j++) {
        pregLoop = UpdateMask<float>(dataCount);
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(ring, ringAddr + j * V_LENGTH);
        DataCopy(state1F, state1FAddr + j * V_LENGTH);
        Cast<float, T, castTraitB16ToB32>(currF, ring, pregLoop);
        DataCopy(weight1F, weight1FAddr + j * V_LENGTH);
        Mul(state0F, currF, weight1F, pregLoop);
        DataCopy(weight0F, weight0FAddr + j * V_LENGTH);
        Add(state0F, state0F, state1F, pregLoop);
        Mul(state1F, currF, weight0F, pregLoop);
        DataCopy(state0FAddr + j * V_LENGTH, state0F, pregLoop);
        DataCopy(state1FAddr + j * V_LENGTH, state1F, pregLoop);
    }
}

template <typename T>
static __simd_vf__ inline void AdvanceFnLocalPartialsWidthFour(__ubuf__ T* ringAddr, __ubuf__ float* weight0FAddr, 
    __ubuf__ float* weight1FAddr, __ubuf__ float* weight2FAddr, __ubuf__ float* state0FAddr, __ubuf__ float* state1FAddr, 
    __ubuf__ float* state2FAddr, uint32_t dataCount, uint16_t colLoopTimes)
{
    RegTensor<T> ring;
    RegTensor<float> currF;
    RegTensor<float> weight0F;
    RegTensor<float> weight1F;
    RegTensor<float> weight2F;
    RegTensor<float> state0F;
    RegTensor<float> state1F;
    RegTensor<float> state2F;
    MaskReg pregLoop;
    for (uint16_t j = 0; j < colLoopTimes; j++) {
        pregLoop = UpdateMask<float>(dataCount);
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(ring, ringAddr + j * V_LENGTH);
        DataCopy(state1F, state1FAddr + j * V_LENGTH);
        DataCopy(state2F, state2FAddr + j * V_LENGTH);
        Cast<float, T, castTraitB16ToB32>(currF, ring, pregLoop);
        DataCopy(weight2F, weight2FAddr + j * V_LENGTH);
        Mul(state0F, currF, weight2F, pregLoop);
        DataCopy(weight1F, weight1FAddr + j * V_LENGTH);
        Add(state0F, state0F, state1F, pregLoop);
        Mul(state1F, currF, weight1F, pregLoop);
        DataCopy(weight0F, weight0FAddr + j * V_LENGTH);
        Add(state1F, state1F, state2F, pregLoop);
        Mul(state2F, currF, weight0F, pregLoop);
        DataCopy(state0FAddr + j * V_LENGTH, state0F, pregLoop);
        DataCopy(state1FAddr + j * V_LENGTH, state1F, pregLoop);
        DataCopy(state2FAddr + j * V_LENGTH, state2F, pregLoop);
    }
}

template <typename T, int32_t kTemplateWidth>
__aicore__ inline void AdvanceFnLocalPartialsRegbase(LocalTensor<T> ring, LocalTensor<float> weightF, 
    LocalTensor<float> state0F, LocalTensor<float> state1F, LocalTensor<float> state2F, uint32_t dataCount, 
    uint32_t weightStep)
{
    uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(dataCount, V_LENGTH));

    __ubuf__ T* ringAddr = (__ubuf__ T*)ring.GetPhyAddr();
    __ubuf__ float* weight0FAddr = (__ubuf__ float*)weightF.GetPhyAddr();
    __ubuf__ float* state0FAddr = (__ubuf__ float*)state0F.GetPhyAddr();
    if constexpr (kTemplateWidth == 2) {
        AscendC::VF_CALL<AdvanceFnLocalPartialsWidthTwo<T>>(ringAddr, weight0FAddr, state0FAddr, dataCount, colLoopTimes);
    } else if constexpr (kTemplateWidth == 3) {
        __ubuf__ float* weight1FAddr = weight0FAddr + weightStep;
        __ubuf__ float* state1FAddr = (__ubuf__ float*)state1F.GetPhyAddr();
        AscendC::VF_CALL<AdvanceFnLocalPartialsWidthThree<T>>(ringAddr, weight0FAddr, weight1FAddr, state0FAddr, 
            state1FAddr, dataCount, colLoopTimes);
    } else if constexpr (kTemplateWidth == 4) {
        __ubuf__ float* weight1FAddr = weight0FAddr + weightStep;
        __ubuf__ float* weight2FAddr = weight1FAddr + weightStep;
        __ubuf__ float* state1FAddr = (__ubuf__ float*)state1F.GetPhyAddr();
        __ubuf__ float* state2FAddr = (__ubuf__ float*)state2F.GetPhyAddr();
        AscendC::VF_CALL<AdvanceFnLocalPartialsWidthFour<T>>(ringAddr, weight0FAddr, weight1FAddr, weight2FAddr, 
            state0FAddr, state1FAddr, state2FAddr, dataCount, colLoopTimes);
    }
}

} // namespace NsCausalConv1d

#endif // CAUSAL_CONV1D_REGBASE_H
