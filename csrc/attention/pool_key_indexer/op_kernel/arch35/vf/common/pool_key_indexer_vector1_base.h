/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef POOL_KEY_INDEXER_VECTOR1_BASE_H
#define POOL_KEY_INDEXER_VECTOR1_BASE_H

#include "kernel_operator.h"

namespace vector1 {

template <typename T>
struct FloatSortTraits;

// fp32
template <>
struct FloatSortTraits<float> {
    using UInt = uint32_t;
    static constexpr UInt ZERO = 0x00000000;
    static constexpr UInt SIGN_MASK = 0x80000000;
    static constexpr UInt NAN_MASK = 0x7FC00000;
    static constexpr UInt ALL_ONE = 0xFFFFFFFF;
};

// bf16
template <>
struct FloatSortTraits<bfloat16_t> {
    using UInt = uint16_t;
    static constexpr UInt ZERO = 0x0000;
    static constexpr UInt SIGN_MASK = 0x8000;
    static constexpr UInt NAN_MASK = 0x7FC0;
    static constexpr UInt ALL_ONE = 0xFFFF;
};

template <typename FloatT>
struct FloatSortConstCtx {
    using Traits = FloatSortTraits<FloatT>;
    using UInt = typename Traits::UInt;
    AscendC::Reg::RegTensor<UInt> zeros;
    AscendC::Reg::RegTensor<UInt> allOnes;
    AscendC::Reg::RegTensor<UInt> signMask;
    AscendC::Reg::RegTensor<UInt> nan;
};

template <typename FloatT>
__simd_callee__ inline void InitFloatSortConstCtx(FloatSortConstCtx<FloatT> &ctx, AscendC::Reg::MaskReg &maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    AscendC::Reg::Duplicate(ctx.zeros, Traits::ZERO, maskAll);
    AscendC::Reg::Duplicate(ctx.allOnes, Traits::ALL_ONE, maskAll);
    AscendC::Reg::Duplicate(ctx.signMask, Traits::SIGN_MASK, maskAll);
    AscendC::Reg::Duplicate(ctx.nan, Traits::NAN_MASK, maskAll);
}

template <typename FloatT>
__simd_callee__ inline void FloatToSortableKey(AscendC::Reg::RegTensor<typename FloatSortTraits<FloatT>::UInt> &outKey,
                                               AscendC::Reg::RegTensor<FloatT> &inVal, FloatSortConstCtx<FloatT> &ctx,
                                               AscendC::Reg::MaskReg &maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    using UInt = typename Traits::UInt;

    AscendC::Reg::RegTensor<UInt> regTemp;
    AscendC::Reg::RegTensor<UInt> regMask;
    AscendC::Reg::MaskReg regSelectNan;
    AscendC::Reg::MaskReg regSelectSign;

    auto &inBits = (AscendC::Reg::RegTensor<UInt> &)inVal;

    // 1. NaN check
    AscendC::Reg::Compare<UInt, CMPMODE::EQ>(regSelectNan, inBits, ctx.nan, maskAll);

    // 2. NaN -> ALL_ONE
    AscendC::Reg::Select(outKey, ctx.allOnes, inBits, regSelectNan);

    // 3. sign bit
    AscendC::Reg::And(regTemp, outKey, ctx.signMask, maskAll);

    AscendC::Reg::Compare<UInt, CMPMODE::GT>(regSelectSign, regTemp, ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::Reg::Select(regMask, ctx.allOnes, ctx.signMask, regSelectSign);
    AscendC::Reg::Xor(outKey, outKey, regMask, maskAll);
}

template <typename FloatT>
__simd_callee__ inline void FloatX2ToSortableKey(
    AscendC::Reg::RegTensor<typename FloatSortTraits<FloatT>::UInt> &outKey0,
    AscendC::Reg::RegTensor<typename FloatSortTraits<FloatT>::UInt> &outKey1, AscendC::Reg::RegTensor<FloatT> &inVal0,
    AscendC::Reg::RegTensor<FloatT> &inVal1, FloatSortConstCtx<FloatT> &ctx, AscendC::Reg::MaskReg &maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    using UInt = typename Traits::UInt;

    AscendC::Reg::RegTensor<UInt> regTemp[2];
    AscendC::Reg::RegTensor<UInt> regMask[2];
    AscendC::Reg::MaskReg regSelectNan[2];
    AscendC::Reg::MaskReg regSelectSign[2];

    auto &inBits0 = (AscendC::Reg::RegTensor<UInt> &)inVal0;
    auto &inBits1 = (AscendC::Reg::RegTensor<UInt> &)inVal1;

    // 1. NaN check
    AscendC::Reg::Compare<UInt, CMPMODE::EQ>(regSelectNan[0], inBits0, ctx.nan, maskAll);
    AscendC::Reg::Compare<UInt, CMPMODE::EQ>(regSelectNan[1], inBits1, ctx.nan, maskAll);

    // 2. NaN -> ALL_ONE
    AscendC::Reg::Select(outKey0, ctx.allOnes, inBits0, regSelectNan[0]);
    AscendC::Reg::Select(outKey1, ctx.allOnes, inBits1, regSelectNan[1]);

    // 3. sign bit
    AscendC::Reg::And(regTemp[0], outKey0, ctx.signMask, maskAll);
    AscendC::Reg::And(regTemp[1], outKey1, ctx.signMask, maskAll);

    AscendC::Reg::Compare<UInt, CMPMODE::GT>(regSelectSign[0], regTemp[0], ctx.zeros, maskAll);
    AscendC::Reg::Compare<UInt, CMPMODE::GT>(regSelectSign[1], regTemp[1], ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::Reg::Select(regMask[0], ctx.allOnes, ctx.signMask, regSelectSign[0]);
    AscendC::Reg::Select(regMask[1], ctx.allOnes, ctx.signMask, regSelectSign[1]);
    AscendC::Reg::Xor(outKey0, outKey0, regMask[0], maskAll);
    AscendC::Reg::Xor(outKey1, outKey1, regMask[1], maskAll);
}

template <typename T, size_t N>
__simd_callee__ inline void DuplicateZero(AscendC::Reg::RegTensor<T> (&regArray)[N], AscendC::Reg::MaskReg &mask)
{
    static_assert(N <= 4, "N must be <= 4");
    // 不能用循环, 会导致fatal error: error in backend: Unsupported Inst must be hoisted.
    if constexpr (N >= 1) {
        AscendC::Reg::Duplicate(regArray[0], static_cast<T>(0), mask);
    }
    if constexpr (N >= 2) {
        AscendC::Reg::Duplicate(regArray[1], static_cast<T>(0), mask);
    }
    if constexpr (N >= 3) {
        AscendC::Reg::Duplicate(regArray[2], static_cast<T>(0), mask);
    }
    if constexpr (N >= 4) {
        AscendC::Reg::Duplicate(regArray[3], static_cast<T>(0), mask);
    }
}

template <typename T, size_t N, bool ApplyRelu = true>
__simd_callee__ inline void WeightedAccum(AscendC::Reg::RegTensor<T> (&accum)[N],
                                          AscendC::Reg::RegTensor<T> (&input)[N], AscendC::Reg::RegTensor<T> &weight,
                                          AscendC::Reg::MaskReg &mask)
{
    static_assert(N <= 2, "N must be <= 2");
    // ---- Relu block ----
    if constexpr (ApplyRelu) {
        if constexpr (N >= 1) {
            AscendC::Reg::Relu(input[0], input[0], mask);
        }
        if constexpr (N >= 2) {
            AscendC::Reg::Relu(input[1], input[1], mask);
        }
    }
    // ---- MulAdd block ----
    if constexpr (N >= 1) {
        AscendC::Reg::MulAddDst(accum[0], input[0], weight, mask);
    }
    if constexpr (N >= 2) {
        AscendC::Reg::MulAddDst(accum[1], input[1], weight, mask);
    }
}

__simd_callee__ inline void BroadcastLane(AscendC::Reg::RegTensor<float> &dst, AscendC::Reg::RegTensor<float> &src,
                                          uint16_t laneIdx)
{
    AscendC::Reg::RegTensor<uint32_t> brcGatherIndex;
    AscendC::Reg::Duplicate(brcGatherIndex, laneIdx);
    AscendC::Reg::Gather(dst, src, brcGatherIndex);
}

} // namespace vector1
#endif
