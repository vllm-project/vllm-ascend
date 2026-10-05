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
 * \file lightning_indexer_vector1.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_VECTOR1_H
#define LIGHTNING_INDEXER_VECTOR1_H

#include "kernel_operator.h"

namespace vector1 {

template <typename T>
struct FloatSortTraits;

template <typename T>
struct UIntSortTraits;

// fp32
template <>
struct FloatSortTraits<float> {
    using UInt = uint32_t;
    static constexpr UInt ZERO      = 0x00000000;
    static constexpr UInt SIGN_MASK = 0x80000000;
    static constexpr UInt NAN_MASK  = 0x7FC00000;
    static constexpr UInt ALL_ONE   = 0xFFFFFFFF;
};

// bf16
template <>
struct FloatSortTraits<bfloat16_t> {
    using UInt = uint16_t;
    static constexpr UInt ZERO      = 0x0000;
    static constexpr UInt SIGN_MASK = 0x8000;
    static constexpr UInt NAN_MASK  = 0x7FC0;
    static constexpr UInt ALL_ONE   = 0xFFFF;
};


template <typename FloatT>
struct FloatSortConstCtx {
    using Traits = FloatSortTraits<FloatT>;
    using UInt   = typename Traits::UInt;
    AscendC::MicroAPI::RegTensor<UInt> zeros;
    AscendC::MicroAPI::RegTensor<UInt> all_one;
    AscendC::MicroAPI::RegTensor<UInt> signMask;
    AscendC::MicroAPI::RegTensor<UInt> nan;
};


template <typename FloatT>
__simd_callee__ inline void InitFloatSortConstCtx(FloatSortConstCtx<FloatT>& ctx, AscendC::MicroAPI::MaskReg& maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    AscendC::MicroAPI::Duplicate(ctx.zeros,    Traits::ZERO,      maskAll);
    AscendC::MicroAPI::Duplicate(ctx.all_one,   Traits::ALL_ONE,   maskAll);
    AscendC::MicroAPI::Duplicate(ctx.signMask, Traits::SIGN_MASK, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.nan,      Traits::NAN_MASK,  maskAll);
}


template <typename FloatT>
__simd_callee__ inline void FloatToSortableKey(
                                               AscendC::MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt>&
                                                outKey,
                                               AscendC::MicroAPI::RegTensor<FloatT>& inVal,
                                               FloatSortConstCtx<FloatT>& ctx,
                                               AscendC::MicroAPI::MaskReg& maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    using UInt   = typename Traits::UInt;

    AscendC::MicroAPI::RegTensor<UInt> regTemp;
    AscendC::MicroAPI::RegTensor<UInt> regMask;
    AscendC::MicroAPI::MaskReg regSelectNan;
    AscendC::MicroAPI::MaskReg regSelectSign;

    auto& inBits = (AscendC::MicroAPI::RegTensor<UInt>&)inVal;

    // 1. NaN check
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectNan, inBits, ctx.nan, maskAll);

    // 2. NaN -> ALL_ONE
    AscendC::MicroAPI::Select(outKey, ctx.all_one, inBits, regSelectNan);

    // 3. sign bit
    AscendC::MicroAPI::And(regTemp, outKey, ctx.signMask, maskAll);

    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign, regTemp, ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::MicroAPI::Select(regMask, ctx.all_one, ctx.signMask, regSelectSign);
    AscendC::MicroAPI::Xor(outKey, outKey, regMask, maskAll);
}

// uint16-bf16
template <>
struct UIntSortTraits<bfloat16_t> {
    using UInt = uint16_t;
    static constexpr UInt ZERO      = 0x0000;
    static constexpr UInt SIGN_MASK = 0x8000;
    static constexpr UInt NAN_MASK  = 0xFFC0;
    static constexpr UInt ALL_ONE   = 0xFFFF;
};

template <typename FloatT>
struct UIntSortConstCtx {
    using Traits = UIntSortTraits<FloatT>;
    using UInt   = typename Traits::UInt;
    AscendC::MicroAPI::RegTensor<UInt> zeros;
    AscendC::MicroAPI::RegTensor<UInt> all_one;
    AscendC::MicroAPI::RegTensor<UInt> signMask;
    AscendC::MicroAPI::RegTensor<UInt> nan;
};

template <typename FloatT>
__simd_callee__ inline void InitUIntSortConstCtx(UIntSortConstCtx<FloatT>& ctx, AscendC::MicroAPI::MaskReg& maskAll)
{
    using Traits = UIntSortTraits<FloatT>;
    AscendC::MicroAPI::Duplicate(ctx.zeros,    Traits::ZERO,      maskAll);
    AscendC::MicroAPI::Duplicate(ctx.all_one,   Traits::ALL_ONE,   maskAll);
    AscendC::MicroAPI::Duplicate(ctx.signMask, Traits::SIGN_MASK, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.nan,      Traits::NAN_MASK,  maskAll);
}

template <typename FloatT>
__simd_callee__ inline void UIntToSortableKey(AscendC::MicroAPI::RegTensor<FloatT>& outKey,
                                              AscendC::MicroAPI::RegTensor<typename UIntSortConstCtx<FloatT>::UInt>&
                                                inVal,
                                              UIntSortConstCtx<FloatT>& ctx,
                                              AscendC::MicroAPI::MaskReg& maskAll)
{
    using Traits = UIntSortTraits<FloatT>;
    using UInt   = typename Traits::UInt;

    AscendC::MicroAPI::RegTensor<UInt> regTemp;
    AscendC::MicroAPI::RegTensor<UInt> regMask;
    AscendC::MicroAPI::MaskReg regSelectZero;
    AscendC::MicroAPI::MaskReg regSelectSign;

    auto& inBits = inVal;

    // 1. 0 check
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectZero, inBits, ctx.zeros, maskAll);

    // 2. 0 -> -NAN
    AscendC::MicroAPI::Select((AscendC::MicroAPI::RegTensor<UInt>&)outKey, ctx.nan, inBits, regSelectZero);

    // 3. sign bit
    AscendC::MicroAPI::And(regTemp, (AscendC::MicroAPI::RegTensor<UInt>&)outKey, ctx.signMask, maskAll);

    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign, regTemp, ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::MicroAPI::Select(regMask, ctx.signMask, ctx.all_one, regSelectSign);
    AscendC::MicroAPI::Xor((AscendC::MicroAPI::RegTensor<UInt>&)outKey,
                           (AscendC::MicroAPI::RegTensor<UInt>&)outKey, regMask, maskAll);
}

__aicore__ inline void UIntToFloatReturnValue(const LocalTensor<bfloat16_t> &out_,
                                              const LocalTensor<uint16_t> &in,
                                              const uint32_t topK)
{
    auto outBuf = (__local_mem__ bfloat16_t*)out_.GetPhyAddr();
    auto inBuf = (__local_mem__ uint16_t*)in.GetPhyAddr();

    const uint16_t repeatSize16 = 128;
    uint16_t topkLoopNum = (topK + repeatSize16 - 1) / repeatSize16;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> regIn;
        AscendC::MicroAPI::RegTensor<bfloat16_t> regOut;
        AscendC::MicroAPI::MaskReg maskAllB16 =
                                    AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();

        for (uint16_t i = 0; i < topkLoopNum; ++i) {
            AscendC::MicroAPI::LoadAlign<uint16_t>(regIn, inBuf + i * 128);

            UIntSortConstCtx<bfloat16_t> uint16Ctx;
            InitUIntSortConstCtx(uint16Ctx, maskAllB16);

            UIntToSortableKey<bfloat16_t>(regOut, regIn, uint16Ctx, maskAllB16);

            AscendC::MicroAPI::StoreAlign<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(
                outBuf + i * 128,
                regOut,
                maskAllB16);
        }
    }
}

__aicore__ inline void UIntToFloatReturnValue(const LocalTensor<half> &out_,
                                              const LocalTensor<uint16_t> &in,
                                              const uint32_t topK)
{
    auto outBuf = (__local_mem__ half*)out_.GetPhyAddr();
    auto inBuf = (__local_mem__ uint16_t*)in.GetPhyAddr();

    const uint16_t repeatSize16 = 128;
    uint16_t topkLoopNum = (topK + repeatSize16 - 1) / repeatSize16;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> regIn;
        AscendC::MicroAPI::RegTensor<bfloat16_t> regOut;
        AscendC::MicroAPI::RegTensor<half> regOutHalf;
        AscendC::MicroAPI::MaskReg maskAllB16 =
                                    AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskAllHalf =
                                    AscendC::MicroAPI::CreateMask<half, AscendC::MicroAPI::MaskPattern::ALL>();
        constexpr static MicroAPI::CastTrait castTraitBF16ToHalf =
                                    {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                     MicroAPI::MaskMergeMode::ZEROING,
                                     RoundMode::CAST_RINT};

        for (uint16_t i = 0; i < topkLoopNum; ++i) {
            AscendC::MicroAPI::LoadAlign<uint16_t>(regIn, inBuf + i * repeatSize16);

            UIntSortConstCtx<bfloat16_t> uint16Ctx;
            InitUIntSortConstCtx(uint16Ctx, maskAllB16);

            UIntToSortableKey<bfloat16_t>(regOut, regIn, uint16Ctx, maskAllB16);

            AscendC::MicroAPI::Cast<half, bfloat16_t, castTraitBF16ToHalf>(regOutHalf, regOut, maskAllB16);

            AscendC::MicroAPI::StoreAlign<half, AscendC::MicroAPI::StoreDist::DIST_NORM>(
                outBuf + i * repeatSize16,
                regOutHalf,
                maskAllHalf);
        }
    }
}

template <typename FloatT>
__simd_callee__ inline void FloatX2ToSortableKey(AscendC::MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt>&
                                                     outKey0,
                                                 AscendC::MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt>&
                                                     outKey1,
                                                 AscendC::MicroAPI::RegTensor<FloatT>& inVal0,
                                                 AscendC::MicroAPI::RegTensor<FloatT>& inVal1,
                                                 FloatSortConstCtx<FloatT>& ctx,
                                                 AscendC::MicroAPI::MaskReg& maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    using UInt   = typename Traits::UInt;

    AscendC::MicroAPI::RegTensor<UInt> regTemp[2];
    AscendC::MicroAPI::RegTensor<UInt> regMask[2];
    AscendC::MicroAPI::MaskReg regSelectNan[2];
    AscendC::MicroAPI::MaskReg regSelectSign[2];

    auto& inBits0 = (AscendC::MicroAPI::RegTensor<UInt>&)inVal0;
    auto& inBits1 = (AscendC::MicroAPI::RegTensor<UInt>&)inVal1;

    // 1. NaN check
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectNan[0], inBits0, ctx.nan, maskAll);
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectNan[1], inBits1, ctx.nan, maskAll);

    // 2. NaN -> ALL_ONE
    AscendC::MicroAPI::Select(outKey0, ctx.all_one, inBits0, regSelectNan[0]);
    AscendC::MicroAPI::Select(outKey1, ctx.all_one, inBits1, regSelectNan[1]);

    // 3. sign bit
    AscendC::MicroAPI::And(regTemp[0], outKey0, ctx.signMask, maskAll);
    AscendC::MicroAPI::And(regTemp[1], outKey1, ctx.signMask, maskAll);

    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign[0], regTemp[0], ctx.zeros, maskAll);
    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign[1], regTemp[1], ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::MicroAPI::Select(regMask[0], ctx.all_one, ctx.signMask, regSelectSign[0]);
    AscendC::MicroAPI::Select(regMask[1], ctx.all_one, ctx.signMask, regSelectSign[1]);
    AscendC::MicroAPI::Xor(outKey0, outKey0, regMask[0], maskAll);
    AscendC::MicroAPI::Xor(outKey1, outKey1, regMask[1], maskAll);
}


template <typename T, size_t N>
__simd_callee__ inline void DuplicateZero(AscendC::MicroAPI::RegTensor<T> (&regArray)[N],
                                          AscendC::MicroAPI::MaskReg& mask)
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
                                          AscendC::MicroAPI::RegTensor<T>& weight,
                                          AscendC::MicroAPI::MaskReg& mask)
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


__simd_callee__ inline void BroadcastLane(AscendC::MicroAPI::RegTensor<float>& dst,
                                          AscendC::MicroAPI::RegTensor<float>& src,
                                          uint16_t laneIdx)
{
    AscendC::MicroAPI::RegTensor<uint32_t> brcGatherIndex;
    AscendC::MicroAPI::Duplicate(brcGatherIndex, laneIdx);
    AscendC::MicroAPI::Gather(dst, src, brcGatherIndex);
}

__simd_callee__ inline void BroadcastLane(AscendC::MicroAPI::RegTensor<float>& dst,
                                          __local_mem__ float* src,
                                          uint16_t laneIdx)
{
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(dst, src + laneIdx);
}

// float in uint16 out
__simd_vf__ inline void MulWeightAndReduceSum(__ubuf__ uint16_t* out_,
                                              __ubuf__ float* qk_,
                                              const uint32_t qkVLStride,
                                              __ubuf__ float* weight_,
                                              const int gSize)
{
    AscendC::MicroAPI::RegTensor<float> regwBrc;
    AscendC::MicroAPI::RegTensor<float> regQK[2];
    AscendC::MicroAPI::RegTensor<float> regW;

    AscendC::MicroAPI::RegTensor<float> regSum0[2];
    AscendC::MicroAPI::RegTensor<float> regSum1[2];
    AscendC::MicroAPI::MaskReg maskAllB32 =
                     AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg maskAllB16 =
                     AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();

    FloatSortConstCtx<bfloat16_t> bf16Ctx;
    InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN =
                                    {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                     MicroAPI::MaskMergeMode::MERGING,
                                     RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD =
                                    {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                     MicroAPI::MaskMergeMode::ZEROING,
                                     RoundMode::CAST_ROUND};

    AscendC::MicroAPI::LoadAlign<float>(regW, weight_);
    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);

    // unroll2
    for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i += 2) {
        MicroAPI::LoadAlign<float>(regQK[0], qk_ + 128 * i); // RowStride是128, 行都落在一个bank上
        MicroAPI::LoadAlign<float>(regQK[1], qk_ + 128 * i + qkVLStride);
        BroadcastLane(regwBrc, regW, i);
        WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

        MicroAPI::LoadAlign<float>(regQK[0], qk_ + 128 * i + 128);
        MicroAPI::LoadAlign<float>(regQK[1], qk_ + 128 * i + 128 + qkVLStride);
        BroadcastLane(regwBrc, regW, i + 1);
        WeightedAccum(regSum1, regQK, regwBrc, maskAllB32);
    }

    AscendC::MicroAPI::Add(regSum0[0], regSum0[0], regSum1[0], maskAllB32);
    AscendC::MicroAPI::Add(regSum0[1], regSum0[1], regSum1[1], maskAllB32);

    AscendC::MicroAPI::RegTensor<bfloat16_t> regSumBF16;
    // interleave cast ==> regSum[1] high regSum[0] low
    AscendC::MicroAPI::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum0[1], maskAllB32);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum0[0], maskAllB32);

    AscendC::MicroAPI::RegTensor<uint16_t> regOut;
    FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
    // normal store
    AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out_, regOut, maskAllB16);
}

// float in uint16 out
__simd_vf__ inline void MulWeightAndReduceSum(__ubuf__ uint16_t* out_,
                                              __ubuf__ float* qk_,
                                              const uint32_t qkVLStride,
                                              __ubuf__ bfloat16_t* weight_,
                                              const int gSize)
{
    AscendC::MicroAPI::RegTensor<float> regwBrc;
    AscendC::MicroAPI::RegTensor<float> regQK[2];
    AscendC::MicroAPI::RegTensor<bfloat16_t> regWBF16;
    AscendC::MicroAPI::RegTensor<float> regW;

    AscendC::MicroAPI::RegTensor<float> regSum0[2];
    AscendC::MicroAPI::RegTensor<float> regSum1[2];
    AscendC::MicroAPI::MaskReg maskAllB32 =
                     AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg maskAllB16 =
                     AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();

    FloatSortConstCtx<bfloat16_t> bf16Ctx;
    InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                    MicroAPI::MaskMergeMode::MERGING,
                                                                    RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                                    MicroAPI::MaskMergeMode::ZEROING,
                                                                    RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitBF16ToFP32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                MicroAPI::MaskMergeMode::ZEROING,
                                                                RoundMode::UNKNOWN};

    AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(regWBF16, weight_);
    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitBF16ToFP32>(regW, regWBF16, maskAllB16);

    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);

    // unroll2
    for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i += 2) {
        MicroAPI::LoadAlign<float>(regQK[0], qk_ + 128 * i); // RowStride是128, 行都落在一个bank上
        MicroAPI::LoadAlign<float>(regQK[1], qk_ + 128 * i + qkVLStride);
        BroadcastLane(regwBrc, regW, i);
        WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

        MicroAPI::LoadAlign<float>(regQK[0], qk_ + 128 * i + 128);
        MicroAPI::LoadAlign<float>(regQK[1], qk_ + 128 * i + 128 + qkVLStride);
        BroadcastLane(regwBrc, regW, i + 1);
        WeightedAccum(regSum1, regQK, regwBrc, maskAllB32);
    }

    AscendC::MicroAPI::Add(regSum0[0], regSum0[0], regSum1[0], maskAllB32);
    AscendC::MicroAPI::Add(regSum0[1], regSum0[1], regSum1[1], maskAllB32);

    AscendC::MicroAPI::RegTensor<bfloat16_t> regSumBF16;
    // interleave cast ==> regSum[1] high regSum[0] low
    AscendC::MicroAPI::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum0[1], maskAllB32);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum0[0], maskAllB32);

    AscendC::MicroAPI::RegTensor<uint16_t> regOut;
    FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
    // normal store
    AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out_, regOut, maskAllB16);
}

// float in uint16 out
__simd_vf__ inline void MulWeightAndReduceSum(__ubuf__ uint16_t* out_,
                                              __ubuf__ float* qk_,
                                              const uint32_t qkVLStride,
                                              __ubuf__ half* weight_,
                                              const int gSize)
{
    AscendC::MicroAPI::RegTensor<float> regwBrc;
    AscendC::MicroAPI::RegTensor<float> regQK[2];
    AscendC::MicroAPI::RegTensor<float> regW;
    AscendC::MicroAPI::RegTensor<half> regWFP16;
    AscendC::MicroAPI::RegTensor<float> regSum0[2];
    AscendC::MicroAPI::RegTensor<float> regSum1[2];
    AscendC::MicroAPI::MaskReg maskAllB32 =
                     AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg maskAllB16 =
                     AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();

    FloatSortConstCtx<bfloat16_t> bf16Ctx;
    InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                    MicroAPI::MaskMergeMode::MERGING,
                                                                    RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                                    MicroAPI::MaskMergeMode::ZEROING,
                                                                    RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitFP16ToFP32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                MicroAPI::MaskMergeMode::ZEROING,
                                                                RoundMode::UNKNOWN};

    AscendC::MicroAPI::LoadAlign<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(regWFP16, weight_);
    AscendC::MicroAPI::Cast<float, half, castTraitFP16ToFP32>(regW, regWFP16, maskAllB16);

    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);

    // unroll2
    for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i += 2) {
        MicroAPI::LoadAlign<float>(regQK[0], qk_ + 128 * i); // RowStride是128, 行都落在一个bank上
        MicroAPI::LoadAlign<float>(regQK[1], qk_ + 128 * i + qkVLStride);
        BroadcastLane(regwBrc, regW, i);
        WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

        MicroAPI::LoadAlign<float>(regQK[0], qk_ + 128 * i + 128);
        MicroAPI::LoadAlign<float>(regQK[1], qk_ + 128 * i + 128 + qkVLStride);
        BroadcastLane(regwBrc, regW, i + 1);
        WeightedAccum(regSum1, regQK, regwBrc, maskAllB32);
    }

    AscendC::MicroAPI::Add(regSum0[0], regSum0[0], regSum1[0], maskAllB32);
    AscendC::MicroAPI::Add(regSum0[1], regSum0[1], regSum1[1], maskAllB32);

    AscendC::MicroAPI::RegTensor<bfloat16_t> regSumBF16;
    // interleave cast ==> regSum[1] high regSum[0] low
    AscendC::MicroAPI::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum0[1], maskAllB32);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum0[0], maskAllB32);

    AscendC::MicroAPI::RegTensor<uint16_t> regOut;
    FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
    // normal store
    AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out_, regOut, maskAllB16);
}

// 计算S1=2
// float in uint16 out
__simd_vf__ inline void MulWeightAndReduceSum2(__ubuf__ uint16_t* out0_,
                                               __ubuf__ uint16_t* out1_,
                                               uint32_t outStride,
                                               __ubuf__ float* qk0_,
                                               __ubuf__ float* qk1_,
                                               uint32_t qkVLStride,
                                               uint32_t qkStride,
                                               __ubuf__ float* weight0_,
                                               __ubuf__ float* weight1_,
                                               uint32_t weightStride,
                                               __ubuf__ float* weightFloat_,
                                               const int gSize)
{
    AscendC::MicroAPI::RegTensor<float> regwBrc[2];
    AscendC::MicroAPI::RegTensor<float> regQK0[2];
    AscendC::MicroAPI::RegTensor<float> regQK1[2];
    AscendC::MicroAPI::RegTensor<float> regW[2];

    AscendC::MicroAPI::RegTensor<float> regSum0[2];
    AscendC::MicroAPI::RegTensor<float> regSum1[2];
    AscendC::MicroAPI::MaskReg maskAllB32 =
                        AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg maskAllB16 =
                        AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();

    FloatSortConstCtx<bfloat16_t> bf16Ctx;
    InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                    MicroAPI::MaskMergeMode::MERGING,
                                                                    RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                                    MicroAPI::MaskMergeMode::ZEROING,
                                                                    RoundMode::CAST_ROUND};

    AscendC::MicroAPI::LoadAlign<float>(regW[0], weight0_);
    AscendC::MicroAPI::LoadAlign<float>(regW[1], weight1_);
    // regW[0]与weight1混合使用
    AscendC::MicroAPI::StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM>(weight1_, regW[1], maskAllB32);
    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);

    for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i++) {
        MicroAPI::LoadAlign<float>(regQK0[0], qk0_ + 128 * i);
        MicroAPI::LoadAlign<float>(regQK0[1], qk0_ + 128 * i + qkVLStride);
        MicroAPI::LoadAlign<float>(regQK1[0], qk1_ + 128 * i);
        MicroAPI::LoadAlign<float>(regQK1[1], qk1_ + 128 * i + qkVLStride);
        // 混合使用对整体性能更好
        BroadcastLane(regwBrc[0], regW[0], i);
        // Weight无bank冲突，用LoadAlign来提取weight标量
        BroadcastLane(regwBrc[1], weight1_, i);
        AscendC::MicroAPI::Relu(regQK0[0], regQK0[0], maskAllB32);
        AscendC::MicroAPI::Relu(regQK0[1], regQK0[1], maskAllB32);
        AscendC::MicroAPI::Relu(regQK1[0], regQK1[0], maskAllB32);
        AscendC::MicroAPI::Relu(regQK1[1], regQK1[1], maskAllB32);
        AscendC::MicroAPI::MulAddDst(regSum0[0], regQK0[0], regwBrc[0], maskAllB32);
        AscendC::MicroAPI::MulAddDst(regSum0[1], regQK0[1], regwBrc[0], maskAllB32);
        AscendC::MicroAPI::MulAddDst(regSum1[0], regQK1[0], regwBrc[1], maskAllB32);
        AscendC::MicroAPI::MulAddDst(regSum1[1], regQK1[1], regwBrc[1], maskAllB32);
    }

    // Convert to bfloat16 and store output channel
    AscendC::MicroAPI::RegTensor<bfloat16_t> regSumBF16[2];
    AscendC::MicroAPI::RegTensor<uint16_t> regOut[2];
    AscendC::MicroAPI::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
    AscendC::MicroAPI::DeInterleave(regSum1[0], regSum1[1], regSum1[0], regSum1[1]);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[0], regSum0[1], maskAllB32);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[1], regSum1[1], maskAllB32);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[0], regSum0[0], maskAllB32);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[1], regSum1[0], maskAllB32);

    FloatX2ToSortableKey<bfloat16_t>(regOut[0], regOut[1], regSumBF16[0], regSumBF16[1], bf16Ctx, maskAllB16);
    AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out0_, regOut[0], maskAllB16);
    AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out1_, regOut[1], maskAllB16);
}

// 计算S1=2
// float in uint16 out
__simd_vf__ inline void MulWeightAndReduceSum2(__ubuf__ uint16_t* out0_,
                                               __ubuf__ uint16_t* out1_,
                                               uint32_t outStride,
                                               __ubuf__ float* qk0_,
                                               __ubuf__ float* qk1_,
                                               uint32_t qkVLStride,
                                               uint32_t qkStride,
                                               __ubuf__ bfloat16_t* weight0_,
                                               __ubuf__ bfloat16_t* weight1_,
                                               uint32_t weightStride,
                                               __ubuf__ float* weightFloat_,
                                               const int gSize)
{
    AscendC::MicroAPI::RegTensor<float> regwBrc[2];
    AscendC::MicroAPI::RegTensor<float> regQK0[2];
    AscendC::MicroAPI::RegTensor<float> regQK1[2];
    AscendC::MicroAPI::RegTensor<float> regW[2];
    AscendC::MicroAPI::RegTensor<bfloat16_t> regWBF16[2];

    AscendC::MicroAPI::RegTensor<float> regSum0[2];
    AscendC::MicroAPI::RegTensor<float> regSum1[2];
    AscendC::MicroAPI::MaskReg maskAllB32 =
                     AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg maskAllB16 =
                     AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();

    FloatSortConstCtx<bfloat16_t> bf16Ctx;
    InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                    MicroAPI::MaskMergeMode::MERGING,
                                                                    RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                                    MicroAPI::MaskMergeMode::ZEROING,
                                                                    RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitBF16ToFP32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                MicroAPI::MaskMergeMode::ZEROING,
                                                                RoundMode::UNKNOWN};

    AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(regWBF16[0], weight0_);
    AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(regWBF16[1], weight1_);
    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitBF16ToFP32>(regW[0], regWBF16[0], maskAllB16);
    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitBF16ToFP32>(regW[1], regWBF16[1], maskAllB16);

    // regW[0]与weight1混合使用
    AscendC::MicroAPI::StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM>(weightFloat_, regW[1], maskAllB32);
    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);

    for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i++) {
        MicroAPI::LoadAlign<float>(regQK0[0], qk0_ + 128 * i);
        MicroAPI::LoadAlign<float>(regQK0[1], qk0_ + 128 * i + qkVLStride);
        MicroAPI::LoadAlign<float>(regQK1[0], qk1_ + 128 * i);
        MicroAPI::LoadAlign<float>(regQK1[1], qk1_ + 128 * i + qkVLStride);
        // 混合使用对整体性能更好
        BroadcastLane(regwBrc[0], regW[0], i);
        // Weight无bank冲突，用LoadAlign来提取weight标量
        BroadcastLane(regwBrc[1], weightFloat_, i);
        AscendC::MicroAPI::Relu(regQK0[0], regQK0[0], maskAllB32);
        AscendC::MicroAPI::Relu(regQK0[1], regQK0[1], maskAllB32);
        AscendC::MicroAPI::Relu(regQK1[0], regQK1[0], maskAllB32);
        AscendC::MicroAPI::Relu(regQK1[1], regQK1[1], maskAllB32);
        AscendC::MicroAPI::MulAddDst(regSum0[0], regQK0[0], regwBrc[0], maskAllB32);
        AscendC::MicroAPI::MulAddDst(regSum0[1], regQK0[1], regwBrc[0], maskAllB32);
        AscendC::MicroAPI::MulAddDst(regSum1[0], regQK1[0], regwBrc[1], maskAllB32);
        AscendC::MicroAPI::MulAddDst(regSum1[1], regQK1[1], regwBrc[1], maskAllB32);
    }

    // Convert to bfloat16 and store output channel
    AscendC::MicroAPI::RegTensor<bfloat16_t> regSumBF16[2];
    AscendC::MicroAPI::RegTensor<uint16_t> regOut[2];
    AscendC::MicroAPI::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
    AscendC::MicroAPI::DeInterleave(regSum1[0], regSum1[1], regSum1[0], regSum1[1]);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[0], regSum0[1], maskAllB32);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[1], regSum1[1], maskAllB32);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[0], regSum0[0], maskAllB32);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[1], regSum1[0], maskAllB32);

    FloatX2ToSortableKey<bfloat16_t>(regOut[0], regOut[1], regSumBF16[0], regSumBF16[1], bf16Ctx, maskAllB16);
    AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out0_, regOut[0], maskAllB16);
    AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out1_, regOut[1], maskAllB16);
}

// 计算S1=2
// float in uint16 out
__simd_vf__ inline void MulWeightAndReduceSum2(__ubuf__ uint16_t* out0_,
                                               __ubuf__ uint16_t* out1_,
                                               uint32_t outStride,
                                               __ubuf__ float* qk0_,
                                               __ubuf__ float* qk1_,
                                               uint32_t qkVLStride,
                                               uint32_t qkStride,
                                               __ubuf__ half* weight0_,
                                               __ubuf__ half* weight1_,
                                               uint32_t weightStride,
                                               __ubuf__ float* weightFloat_,
                                               const int gSize)
{
    AscendC::MicroAPI::RegTensor<float> regwBrc[2];
    AscendC::MicroAPI::RegTensor<float> regQK0[2];
    AscendC::MicroAPI::RegTensor<float> regQK1[2];
    AscendC::MicroAPI::RegTensor<float> regW[2];
    AscendC::MicroAPI::RegTensor<half> regWFP16[2];

    AscendC::MicroAPI::RegTensor<float> regSum0[2];
    AscendC::MicroAPI::RegTensor<float> regSum1[2];
    AscendC::MicroAPI::MaskReg maskAllB32 =
                     AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg maskAllB16 =
                     AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();

    FloatSortConstCtx<bfloat16_t> bf16Ctx;
    InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                    MicroAPI::MaskMergeMode::MERGING,
                                                                    RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                                    MicroAPI::MaskMergeMode::ZEROING,
                                                                    RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitFP16ToFP32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                MicroAPI::MaskMergeMode::ZEROING,
                                                                RoundMode::UNKNOWN};

    AscendC::MicroAPI::LoadAlign<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(regWFP16[0], weight0_);
    AscendC::MicroAPI::LoadAlign<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(regWFP16[1], weight1_);
    AscendC::MicroAPI::Cast<float, half, castTraitFP16ToFP32>(regW[0], regWFP16[0], maskAllB16);
    AscendC::MicroAPI::Cast<float, half, castTraitFP16ToFP32>(regW[1], regWFP16[1], maskAllB16);

    // regW[0]与weight1混合使用
    AscendC::MicroAPI::StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM>(weightFloat_, regW[1], maskAllB32);
    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);

    for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i++) {
        MicroAPI::LoadAlign<float>(regQK0[0], qk0_ + 128 * i);
        MicroAPI::LoadAlign<float>(regQK0[1], qk0_ + 128 * i + qkVLStride);
        MicroAPI::LoadAlign<float>(regQK1[0], qk1_ + 128 * i);
        MicroAPI::LoadAlign<float>(regQK1[1], qk1_ + 128 * i + qkVLStride);
        // 混合使用对整体性能更好
        BroadcastLane(regwBrc[0], regW[0], i);
        // Weight无bank冲突，用LoadAlign来提取weight标量
        BroadcastLane(regwBrc[1], weightFloat_, i);
        AscendC::MicroAPI::Relu(regQK0[0], regQK0[0], maskAllB32);
        AscendC::MicroAPI::Relu(regQK0[1], regQK0[1], maskAllB32);
        AscendC::MicroAPI::Relu(regQK1[0], regQK1[0], maskAllB32);
        AscendC::MicroAPI::Relu(regQK1[1], regQK1[1], maskAllB32);
        AscendC::MicroAPI::MulAddDst(regSum0[0], regQK0[0], regwBrc[0], maskAllB32);
        AscendC::MicroAPI::MulAddDst(regSum0[1], regQK0[1], regwBrc[0], maskAllB32);
        AscendC::MicroAPI::MulAddDst(regSum1[0], regQK1[0], regwBrc[1], maskAllB32);
        AscendC::MicroAPI::MulAddDst(regSum1[1], regQK1[1], regwBrc[1], maskAllB32);
    }

    // Convert to bfloat16 and store output channel
    AscendC::MicroAPI::RegTensor<bfloat16_t> regSumBF16[2];
    AscendC::MicroAPI::RegTensor<uint16_t> regOut[2];
    AscendC::MicroAPI::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
    AscendC::MicroAPI::DeInterleave(regSum1[0], regSum1[1], regSum1[0], regSum1[1]);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[0], regSum0[1], maskAllB32);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[1], regSum1[1], maskAllB32);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[0], regSum0[0], maskAllB32);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[1], regSum1[0], maskAllB32);

    FloatX2ToSortableKey<bfloat16_t>(regOut[0], regOut[1], regSumBF16[0], regSumBF16[1], bf16Ctx, maskAllB16);
    AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out0_, regOut[0], maskAllB16);
    AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out1_, regOut[1], maskAllB16);
}

template<typename QK_T, typename W_T, typename SCORE_T>
__aicore__ inline void BatchMulWeightAndReduceSum(const LocalTensor<SCORE_T> &out_,   // out    [S2Base]     [128   ]
                                                  uint32_t outStride,
                                                  const LocalTensor<QK_T> &qk_,       // q*k^t  [G, S2Base]  [64 128]
                                                  uint32_t qkVLStride,
                                                  uint32_t qkStride,
                                                  const LocalTensor<W_T> &weight_,   // w      [G]          [64    ]
                                                  uint32_t weightStride,
                                                  const LocalTensor<float> &weightFloat_,
                                                  const int gSize,                     // G 64
                                                  const int batch)
{
    // 暂只支持这两种情况, 后续改成循环
    if (batch != 2 && batch != 1) {
        return;
    }
    auto weight = (__ubuf__ W_T *)weight_.GetPhyAddr();
    auto weightFloat = (__ubuf__ float *)weightFloat_.GetPhyAddr();
    auto qk = (__ubuf__ float *)qk_.GetPhyAddr();
    auto out = (__ubuf__ uint16_t *)out_.GetPhyAddr();
    if (batch == 2) {
        auto weight1 = weight + weightStride;
        auto qk1 = qk + qkStride;
        auto out1 = out + outStride;
        MulWeightAndReduceSum2(out, out1, outStride,
                               qk, qk1, qkVLStride, qkStride,
                               weight, weight1, weightStride, weightFloat,
                               gSize);
    } else {
        MulWeightAndReduceSum(out, qk, qkVLStride, weight, gSize);
    }
}

}

#endif
