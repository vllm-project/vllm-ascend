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
 * \file causal_conv1d_common.h
 */

#ifndef CAUSAL_CONV1D_COMMON_H
#define CAUSAL_CONV1D_COMMON_H

#include "kernel_operator.h"

namespace NsCausalConv1dCommon {

constexpr int32_t MAX_WIDTH = 4;
constexpr int32_t MAX_BLOCK_DIM = 3072;
constexpr int32_t RING_SLOTS = 5;

// This function should only be called if one KNOWS that the input is non-negative.
template <uint32_t N>
__aicore__ inline constexpr int32_t UnsignedMod(int32_t num) {
    const uint32_t uNum = static_cast<uint32_t>(num);
    const int32_t retVal = static_cast<int32_t>(uNum % N);
    return retVal;
}

__aicore__ inline constexpr int32_t SlotCurr(int32_t t)
{
    return UnsignedMod<RING_SLOTS>(t + 3);
}

__aicore__ inline constexpr int32_t SlotHist(int32_t t, int32_t i)
{
    return UnsignedMod<RING_SLOTS>(t + 3 - i);
}

__aicore__ inline constexpr int32_t SlotPrefetch(int32_t t)
{
    return UnsignedMod<RING_SLOTS>(t + 4);
}

struct CalcBufLayout {
    AscendC::LocalTensor<float> weightF;
    AscendC::LocalTensor<float> biasF;
    AscendC::LocalTensor<float> accF;
    AscendC::LocalTensor<float> tmpF;
    AscendC::LocalTensor<float> currF;

    __aicore__ inline CalcBufLayout() = default;

    __aicore__ static inline CalcBufLayout FromCalcBuf(AscendC::TBuf<AscendC::QuePosition::VECCALC> &calcBuf)
    {
        CalcBufLayout layout;
        AscendC::LocalTensor<float> calc = calcBuf.template Get<float>();
        layout.weightF = calc;
        layout.biasF = calc[MAX_WIDTH * MAX_BLOCK_DIM];
        layout.accF = layout.biasF[MAX_BLOCK_DIM];
        layout.tmpF = layout.accF[MAX_BLOCK_DIM];
        layout.currF = layout.tmpF[MAX_BLOCK_DIM];
        return layout;
    }
};

} // namespace NsCausalConv1dCommon

#endif // CAUSAL_CONV1D_COMMON_H
