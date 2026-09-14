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
 * \file lightning_indexer_v2_vector1_base.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_V2_VECTOR1_BASE_H
#define LIGHTNING_INDEXER_V2_VECTOR1_BASE_H

#include "basic_api/kernel_basic_intf.h"

namespace liV2Vector1 {
template <typename T>
struct UIntSortTraits;

template <>
struct UIntSortTraits<float> {
    using UInt = uint32_t;
    static constexpr UInt ZERO = 0x00000000;
    static constexpr UInt SIGN_MASK = 0x80000000;
    static constexpr UInt NAN_MASK = 0xFFC00000;
    static constexpr UInt ALL_ONE = 0xFFFFFFFF;
};

template <>
struct UIntSortTraits<bfloat16_t> {
    using UInt = uint16_t;
    static constexpr UInt ZERO = 0x0000;
    static constexpr UInt SIGN_MASK = 0x8000;
    static constexpr UInt NAN_MASK = 0xFFC0;
    static constexpr UInt ALL_ONE = 0xFFFF;
};

template <typename FloatT>
struct UIntSortConstCtx {
    using Traits = UIntSortTraits<FloatT>;
    using UInt = typename Traits::UInt;
    AscendC::MicroAPI::RegTensor<UInt> zeros;
    AscendC::MicroAPI::RegTensor<UInt> allOnes;
    AscendC::MicroAPI::RegTensor<UInt> signMask;
    AscendC::MicroAPI::RegTensor<UInt> nan;
};

template <typename FloatT>
__simd_callee__ inline void InitUIntSortConstCtx(UIntSortConstCtx<FloatT> &ctx, AscendC::MicroAPI::MaskReg &maskAll)
{
    using Traits = UIntSortTraits<FloatT>;
    AscendC::MicroAPI::Duplicate(ctx.zeros, Traits::ZERO, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.allOnes, Traits::ALL_ONE, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.signMask, Traits::SIGN_MASK, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.nan, Traits::NAN_MASK, maskAll);
}

template <typename FloatT>
__simd_callee__ inline void UIntToSortableKey(
    AscendC::MicroAPI::RegTensor<FloatT> &outKey,
    AscendC::MicroAPI::RegTensor<typename UIntSortConstCtx<FloatT>::UInt> &inVal, UIntSortConstCtx<FloatT> &ctx,
    AscendC::MicroAPI::MaskReg &maskAll)
{
    using Traits = UIntSortTraits<FloatT>;
    using UInt = typename Traits::UInt;

    AscendC::MicroAPI::RegTensor<UInt> regTemp;
    AscendC::MicroAPI::RegTensor<UInt> regMask;
    AscendC::MicroAPI::MaskReg regSelectZero;
    AscendC::MicroAPI::MaskReg regSelectSign;

    auto &inBits = inVal;

    // 1. 0 check
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectZero, inBits, ctx.zeros, maskAll);

    // 2. 0 -> -NAN
    AscendC::MicroAPI::Select((AscendC::MicroAPI::RegTensor<UInt> &)outKey, ctx.nan, inBits, regSelectZero);

    // 3. sign bit
    AscendC::MicroAPI::And(regTemp, (AscendC::MicroAPI::RegTensor<UInt> &)outKey, ctx.signMask, maskAll);

    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign, regTemp, ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::MicroAPI::Select(regMask, ctx.signMask, ctx.allOnes, regSelectSign);
    AscendC::MicroAPI::Xor((AscendC::MicroAPI::RegTensor<UInt> &)outKey, (AscendC::MicroAPI::RegTensor<UInt> &)outKey,
                           regMask, maskAll);
}

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
    AscendC::MicroAPI::RegTensor<UInt> zeros;
    AscendC::MicroAPI::RegTensor<UInt> allOnes;
    AscendC::MicroAPI::RegTensor<UInt> signMask;
    AscendC::MicroAPI::RegTensor<UInt> nan;
};

template <typename FloatT>
__simd_callee__ inline void InitFloatSortConstCtx(FloatSortConstCtx<FloatT> &ctx, AscendC::MicroAPI::MaskReg &maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    AscendC::MicroAPI::Duplicate(ctx.zeros, Traits::ZERO, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.allOnes, Traits::ALL_ONE, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.signMask, Traits::SIGN_MASK, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.nan, Traits::NAN_MASK, maskAll);
}

template <typename FloatT>
__simd_callee__ inline void FloatToSortableKey(MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt> &outKey,
                                               MicroAPI::RegTensor<FloatT> &inVal, FloatSortConstCtx<FloatT> &ctx,
                                               MicroAPI::MaskReg &maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    using UInt = typename Traits::UInt;

    AscendC::MicroAPI::RegTensor<UInt> regTemp;
    AscendC::MicroAPI::RegTensor<UInt> regMask;
    AscendC::MicroAPI::MaskReg regSelectNan;
    AscendC::MicroAPI::MaskReg regSelectSign;

    auto &inBits = (AscendC::MicroAPI::RegTensor<UInt> &)inVal;

    // 1. NaN check
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectNan, inBits, ctx.nan, maskAll);

    // 2. NaN -> ALL_ONE
    AscendC::MicroAPI::Select(outKey, ctx.allOnes, inBits, regSelectNan);

    // 3. sign bit
    AscendC::MicroAPI::And(regTemp, outKey, ctx.signMask, maskAll);

    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign, regTemp, ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::MicroAPI::Select(regMask, ctx.allOnes, ctx.signMask, regSelectSign);
    AscendC::MicroAPI::Xor(outKey, outKey, regMask, maskAll);
}

template <typename FloatT>
__simd_callee__ inline void FloatX2ToSortableKey(MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt> &outKey0,
                                                 MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt> &outKey1,
                                                 MicroAPI::RegTensor<FloatT> &inVal0,
                                                 MicroAPI::RegTensor<FloatT> &inVal1, FloatSortConstCtx<FloatT> &ctx,
                                                 MicroAPI::MaskReg &maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    using UInt = typename Traits::UInt;

    AscendC::MicroAPI::RegTensor<UInt> regTemp[2];
    AscendC::MicroAPI::RegTensor<UInt> regMask[2];
    AscendC::MicroAPI::MaskReg regSelectNan[2];
    AscendC::MicroAPI::MaskReg regSelectSign[2];

    auto &inBits0 = (AscendC::MicroAPI::RegTensor<UInt> &)inVal0;
    auto &inBits1 = (AscendC::MicroAPI::RegTensor<UInt> &)inVal1;

    // 1. NaN check
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectNan[0], inBits0, ctx.nan, maskAll);
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectNan[1], inBits1, ctx.nan, maskAll);

    // 2. NaN -> ALL_ONE
    AscendC::MicroAPI::Select(outKey0, ctx.allOnes, inBits0, regSelectNan[0]);
    AscendC::MicroAPI::Select(outKey1, ctx.allOnes, inBits1, regSelectNan[1]);

    // 3. sign bit
    AscendC::MicroAPI::And(regTemp[0], outKey0, ctx.signMask, maskAll);
    AscendC::MicroAPI::And(regTemp[1], outKey1, ctx.signMask, maskAll);

    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign[0], regTemp[0], ctx.zeros, maskAll);
    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign[1], regTemp[1], ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::MicroAPI::Select(regMask[0], ctx.allOnes, ctx.signMask, regSelectSign[0]);
    AscendC::MicroAPI::Select(regMask[1], ctx.allOnes, ctx.signMask, regSelectSign[1]);
    AscendC::MicroAPI::Xor(outKey0, outKey0, regMask[0], maskAll);
    AscendC::MicroAPI::Xor(outKey1, outKey1, regMask[1], maskAll);
}

template <typename T, size_t N>
__simd_callee__ inline void DuplicateZero(AscendC::MicroAPI::RegTensor<T> (&regArray)[N],
                                          AscendC::MicroAPI::MaskReg &mask)
{
    static_assert(N <= 4, "N must be <= 4");
    // 不能用循环, 会导致fatal error: error in backend: Unsupported Inst must be hoisted.
    if constexpr (N >= 1) {
        AscendC::MicroAPI::Duplicate(regArray[0], static_cast<T>(0), mask);
    }
    if constexpr (N >= 2) {
        AscendC::MicroAPI::Duplicate(regArray[1], static_cast<T>(0), mask);
    }
    if constexpr (N >= 3) {
        AscendC::MicroAPI::Duplicate(regArray[2], static_cast<T>(0), mask);
    }
    if constexpr (N >= 4) {
        AscendC::MicroAPI::Duplicate(regArray[3], static_cast<T>(0), mask);
    }
}

template <typename T, size_t N, bool ApplyRelu = true>
__simd_callee__ inline void WeightedAccum(AscendC::MicroAPI::RegTensor<T> (&accum)[N],
                                          AscendC::MicroAPI::RegTensor<T> (&input)[N],
                                          AscendC::MicroAPI::RegTensor<T> &weight, AscendC::MicroAPI::MaskReg &mask)
{
    static_assert(N <= 2, "N must be <= 2");
    // ---- Relu block ----
    if constexpr (ApplyRelu) {
        if constexpr (N >= 1) {
            AscendC::MicroAPI::Relu(input[0], input[0], mask);
        }
        if constexpr (N >= 2) {
            AscendC::MicroAPI::Relu(input[1], input[1], mask);
        }
    }
    // ---- MulAdd block ----
    if constexpr (N >= 1) {
        AscendC::MicroAPI::MulAddDst(accum[0], input[0], weight, mask);
    }
    if constexpr (N >= 2) {
        AscendC::MicroAPI::MulAddDst(accum[1], input[1], weight, mask);
    }
}

__simd_callee__ inline void BroadcastLane(AscendC::MicroAPI::RegTensor<float> &dst,
                                          AscendC::MicroAPI::RegTensor<float> &src, uint16_t laneIdx)
{
    AscendC::MicroAPI::RegTensor<uint32_t> brcGatherIndex;
    AscendC::MicroAPI::Duplicate(brcGatherIndex, laneIdx);
    AscendC::MicroAPI::Gather(dst, src, brcGatherIndex);
}

__simd_callee__ inline void BroadcastLane(AscendC::MicroAPI::RegTensor<bfloat16_t> &dst,
                                          AscendC::MicroAPI::RegTensor<bfloat16_t> &src, uint16_t laneIdx)
{
    AscendC::MicroAPI::RegTensor<uint16_t> brcGatherIndex;
    AscendC::MicroAPI::Duplicate(brcGatherIndex, laneIdx);
    AscendC::MicroAPI::Gather(dst, src, brcGatherIndex);
}

__simd_callee__ inline void BroadcastLane(AscendC::MicroAPI::RegTensor<float>& dst,
                                          __ubuf__ float* src,
                                          uint16_t laneIdx)
{
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(dst, src + laneIdx);
}

} // namespace liV2Vector1
#endif
