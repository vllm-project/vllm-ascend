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
 * \file quant_lightning_indexer_vector1.h
 * \brief
 */
#ifndef quant_lightning_indexer_VECTOR1_H
#define quant_lightning_indexer_VECTOR1_H

#include "kernel_operator.h"

namespace vector1 {

template <typename T>
struct FloatSortTraits;

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
    AscendC::MicroAPI::RegTensor<UInt> allOnes;
    AscendC::MicroAPI::RegTensor<UInt> signMask;
    AscendC::MicroAPI::RegTensor<UInt> nan;
};


template <typename FloatT>
__simd_callee__ inline void InitFloatSortConstCtx(FloatSortConstCtx<FloatT>& ctx, AscendC::MicroAPI::MaskReg& maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    AscendC::MicroAPI::Duplicate(ctx.zeros,    Traits::ZERO,      maskAll);
    AscendC::MicroAPI::Duplicate(ctx.allOnes,   Traits::ALL_ONE,   maskAll);
    AscendC::MicroAPI::Duplicate(ctx.signMask, Traits::SIGN_MASK, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.nan,      Traits::NAN_MASK,  maskAll);
}


template <typename FloatT>
__simd_callee__ inline void FloatToSortableKey(AscendC::MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt>& outKey,
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
    AscendC::MicroAPI::Select(outKey, ctx.allOnes, inBits, regSelectNan);

    // 3. sign bit
    AscendC::MicroAPI::And(regTemp, outKey, ctx.signMask, maskAll);

    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign, regTemp, ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::MicroAPI::Select(regMask, ctx.allOnes, ctx.signMask, regSelectSign);
    AscendC::MicroAPI::Xor(outKey, outKey, regMask, maskAll);
}

template <typename FloatT>
__simd_callee__ inline void FloatX2ToSortableKey(AscendC::MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt>& outKey0,
                                                 AscendC::MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt>& outKey1,
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
__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint16_t> &out_,   // out    [S2Base]     [128   ]
                                             const LocalTensor<float> &qk_,       // q*k^t  [G, S2Base]  [64 128]
                                             const uint32_t qkVLStride,
                                             const LocalTensor<float> &weight_,   // w      [G]          [64    ]
                                             const LocalTensor<float> &kScale_,   // kScale [S2Base]     [128   ]
                                             const LocalTensor<float> &qScale_,   // qScale [G]          [64    ]
                                             const int gSize)                     // G 64
{
    auto weight = (__local_mem__ float*)weight_.GetPhyAddr();
    auto qScale = (__local_mem__ float*)qScale_.GetPhyAddr();
    auto kScale = (__local_mem__ float*)kScale_.GetPhyAddr();
    auto qk = (__local_mem__ float*)qk_.GetPhyAddr();
    auto out = (__local_mem__ uint16_t*)out_.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> regwBrc;
        AscendC::MicroAPI::RegTensor<float> regQK[2];
        AscendC::MicroAPI::RegTensor<float> regW;

        AscendC::MicroAPI::RegTensor<float> regQScale;
        AscendC::MicroAPI::RegTensor<float> regKScale[2];
        AscendC::MicroAPI::RegTensor<float> regSum0[2];
        AscendC::MicroAPI::RegTensor<float> regSum1[2];
        AscendC::MicroAPI::MaskReg maskAllB32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskAllB16 = AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();

        FloatSortConstCtx<bfloat16_t> bf16Ctx;
        InitFloatSortConstCtx(bf16Ctx, maskAllB16);

        constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                       MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
        constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                                      MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

        AscendC::MicroAPI::LoadAlign<float>(regW, weight);
        AscendC::MicroAPI::LoadAlign<float>(regQScale, qScale);
        AscendC::MicroAPI::Mul(regW, regW, regQScale, maskAllB32);

        DuplicateZero(regSum0, maskAllB32);
        DuplicateZero(regSum1, maskAllB32);

        MicroAPI::LoadAlign<float>(regKScale[0], kScale);
        MicroAPI::LoadAlign<float>(regKScale[1], kScale + 64);

        // unroll2
        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i += 2) {
            MicroAPI::LoadAlign<float>(regQK[0], qk + 128 * i); // RowStride是128, 行都落在一个bank上
            MicroAPI::LoadAlign<float>(regQK[1], qk + 128 * i + qkVLStride);
            BroadcastLane(regwBrc, regW, i);
            WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

            MicroAPI::LoadAlign<float>(regQK[0], qk + 128 * i + 128);
            MicroAPI::LoadAlign<float>(regQK[1], qk + 128 * i + 128 + qkVLStride);
            BroadcastLane(regwBrc, regW, i + 1);
            WeightedAccum(regSum1, regQK, regwBrc, maskAllB32);
        }

        AscendC::MicroAPI::Add(regSum0[0], regSum0[0], regSum1[0], maskAllB32);
        AscendC::MicroAPI::Add(regSum0[1], regSum0[1], regSum1[1], maskAllB32);

        AscendC::MicroAPI::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
        AscendC::MicroAPI::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);

        AscendC::MicroAPI::RegTensor<bfloat16_t> regSumBF16;
        // interleave cast ==> regSum[1] high regSum[0] low
        AscendC::MicroAPI::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum0[1], maskAllB32);
        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum0[0], maskAllB32);

        AscendC::MicroAPI::RegTensor<uint16_t> regOut;
        FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
        // normal store
        AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out, regOut, maskAllB16);
    }
}


// bfloat16_t in uint16 out
__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint16_t> &out_,   // out    [S2Base]     [128   ]
                                             const LocalTensor<bfloat16_t> &qk_,  // q*k^t  [G, S2Base]  [64 128]
                                             const uint32_t qkVLStride,           // unused for bfloat16
                                             const LocalTensor<float> &weight_,   // w      [G]          [64    ]
                                             const LocalTensor<float> &kScale_,   // kScale [S2Base]     [128   ]
                                             const LocalTensor<float> &qScale_,   // qScale [G]          [64    ]
                                             const int gSize)                     // G 64
{
    auto weight = (__local_mem__ float*)weight_.GetPhyAddr();
    auto qScale = (__local_mem__ float*)qScale_.GetPhyAddr();
    auto qk = (__local_mem__ bfloat16_t*)qk_.GetPhyAddr();
    auto kScale = (__local_mem__ float*)kScale_.GetPhyAddr();
    auto out = (__local_mem__ uint16_t*)out_.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> regQK[4];
        AscendC::MicroAPI::RegTensor<bfloat16_t> regQKB16[2];
        AscendC::MicroAPI::RegTensor<float> regW;
        AscendC::MicroAPI::RegTensor<float> regwBrc[2];
        AscendC::MicroAPI::RegTensor<float> regQScale;
        AscendC::MicroAPI::RegTensor<float> regKScale[2];
        AscendC::MicroAPI::RegTensor<float> regSum[2];

        AscendC::MicroAPI::MaskReg maskAllB32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskAllB16 = AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::RegTensor<bfloat16_t> regSumBF16;

        FloatSortConstCtx<bfloat16_t> bf16Ctx;
        InitFloatSortConstCtx(bf16Ctx, maskAllB16);


        using CastTrait = AscendC::MicroAPI::CastTrait;
        static constexpr CastTrait castTraitB162B32_EVEN = {AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
                                                            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr CastTrait castTraitB162B32_ODD  = {AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
                                                            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        constexpr static CastTrait castTraitF32ToF16_EVEN = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                             MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
        constexpr static CastTrait castTraitF32ToF16_ODD  = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

        AscendC::MicroAPI::LoadAlign<float>(regW, weight);
        AscendC::MicroAPI::LoadAlign<float>(regQScale, qScale);
        AscendC::MicroAPI::Mul(regW, regW, regQScale, maskAllB32);
        AscendC::MicroAPI::StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM>(weight, regW, maskAllB32);
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

        DuplicateZero(regSum, maskAllB32);

        // interleave load
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(regKScale[0], regKScale[1], kScale);

        // Duplicate + Gather方法劣化
        // Relu在cube随路做
        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i++) {
            AscendC::MicroAPI::LoadAlign<bfloat16_t>(regQKB16[0], qk + 256 * i); // RowStride是256, 行都落在一个bank上
            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(regwBrc[0], weight + i);
            // interleave cast
            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitB162B32_EVEN>(regQK[0], regQKB16[0], maskAllB16);
            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitB162B32_ODD>(regQK[1], regQKB16[0], maskAllB16);
            AscendC::MicroAPI::MulAddDst(regSum[0], regQK[0], regwBrc[0], maskAllB32);
            AscendC::MicroAPI::MulAddDst(regSum[1], regQK[1], regwBrc[0], maskAllB32);
        }

        AscendC::MicroAPI::Mul(regSum[0], regSum[0], regKScale[0], maskAllB32);
        AscendC::MicroAPI::Mul(regSum[1], regSum[1], regKScale[1], maskAllB32);
        // interleave cast back
        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum[1], maskAllB32);
        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum[0], maskAllB32);

        AscendC::MicroAPI::RegTensor<uint16_t> regOut;
        FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
        // norm load
        AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out, regOut, maskAllB16);
    }
}


// 计算S1=2
// float in uint16 out
__aicore__ inline void MulWeightAndReduceSum2(const LocalTensor<uint16_t> &out_,   // out    [2, S2Base]     [128   ]
                                              uint32_t outStride,
                                              const LocalTensor<float> &qk_,       // q*k^t  [2, G, S2Base]  [64 128]
                                              uint32_t qkVLStride,
                                              uint32_t qkStride,
                                              const LocalTensor<float> &weight_,   // w      [2, G]          [64    ]
                                              uint32_t weightStride,
                                              const LocalTensor<float> &kScale_,   // kScale [S2Base]        [128   ]
                                              uint32_t kScaleStride,
                                              const LocalTensor<float> &qScale_,   // qScale [2, G]          [64    ]
                                              uint32_t qScaleStride,
                                              const int gSize)                     // G 64
{
    auto weight0 = (__local_mem__ float*)weight_.GetPhyAddr();
    auto qScale0 = (__local_mem__ float*)qScale_.GetPhyAddr();
    auto kScale0 = (__local_mem__ float*)kScale_.GetPhyAddr();
    auto qk0 = (__local_mem__ float*)qk_.GetPhyAddr();
    auto out0 = (__local_mem__ uint16_t*)out_.GetPhyAddr();

    auto weight1 = weight0 + weightStride;
    auto qScale1 = qScale0 + qScaleStride;
    auto qk1 = qk0 + qkStride;
    // kScaleStride is zero
    auto out1 = out0 + outStride;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> regwBrc[2];
        AscendC::MicroAPI::RegTensor<float> regQK0[2];
        AscendC::MicroAPI::RegTensor<float> regQK1[2];
        AscendC::MicroAPI::RegTensor<float> regW[2];

        AscendC::MicroAPI::RegTensor<float> regQScale[2];
        AscendC::MicroAPI::RegTensor<float> regKScale[2];
        AscendC::MicroAPI::RegTensor<float> regSum0[2];
        AscendC::MicroAPI::RegTensor<float> regSum1[2];
        AscendC::MicroAPI::MaskReg maskAllB32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskAllB16 = AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();

        FloatSortConstCtx<bfloat16_t> bf16Ctx;
        InitFloatSortConstCtx(bf16Ctx, maskAllB16);

        constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                       MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
        constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                                      MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

        AscendC::MicroAPI::LoadAlign<float>(regW[0], weight0);
        AscendC::MicroAPI::LoadAlign<float>(regW[1], weight1);
        AscendC::MicroAPI::LoadAlign<float>(regQScale[0], qScale0);
        AscendC::MicroAPI::LoadAlign<float>(regQScale[1], qScale1);
        AscendC::MicroAPI::Mul(regW[0], regW[0], regQScale[0], maskAllB32);
        AscendC::MicroAPI::Mul(regW[1], regW[1], regQScale[1], maskAllB32);
        // regW[0]与weight1混合使用
        AscendC::MicroAPI::StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM>(weight1, regW[1], maskAllB32);
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        DuplicateZero(regSum0, maskAllB32);
        DuplicateZero(regSum1, maskAllB32);

        MicroAPI::LoadAlign<float>(regKScale[0], kScale0);
        MicroAPI::LoadAlign<float>(regKScale[1], kScale0 + 64);

        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i++) {
            MicroAPI::LoadAlign<float>(regQK0[0], qk0 + 128 * i);
            MicroAPI::LoadAlign<float>(regQK0[1], qk0 + 128 * i + qkVLStride);
            MicroAPI::LoadAlign<float>(regQK1[0], qk1 + 128 * i);
            MicroAPI::LoadAlign<float>(regQK1[1], qk1 + 128 * i + qkVLStride);
            // 混合使用对整体性能更好
            BroadcastLane(regwBrc[0], regW[0], i);
            // Weight无bank冲突，用LoadAlign来提取weight标量
            BroadcastLane(regwBrc[1], weight1, i);
            AscendC::MicroAPI::Relu(regQK0[0], regQK0[0], maskAllB32);
            AscendC::MicroAPI::Relu(regQK0[1], regQK0[1], maskAllB32);
            AscendC::MicroAPI::Relu(regQK1[0], regQK1[0], maskAllB32);
            AscendC::MicroAPI::Relu(regQK1[1], regQK1[1], maskAllB32);
            AscendC::MicroAPI::MulAddDst(regSum0[0], regQK0[0], regwBrc[0], maskAllB32);
            AscendC::MicroAPI::MulAddDst(regSum0[1], regQK0[1], regwBrc[0], maskAllB32);
            AscendC::MicroAPI::MulAddDst(regSum1[0], regQK1[0], regwBrc[1], maskAllB32);
            AscendC::MicroAPI::MulAddDst(regSum1[1], regQK1[1], regwBrc[1], maskAllB32);
        }

        // Apply kScale scaling
        AscendC::MicroAPI::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
        AscendC::MicroAPI::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);
        AscendC::MicroAPI::Mul(regSum1[0], regSum1[0], regKScale[0], maskAllB32);
        AscendC::MicroAPI::Mul(regSum1[1], regSum1[1], regKScale[1], maskAllB32);


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
        AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out0, regOut[0], maskAllB16);
        AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out1, regOut[1], maskAllB16);
    }
}


// 计算S1=2
// bfloat16 in uint16 out
__aicore__ inline void MulWeightAndReduceSum2(const LocalTensor<uint16_t> &out_,   // out    [2, S2Base]     [128   ]
                                              uint32_t outStride,
                                              const LocalTensor<bfloat16_t> &qk_,  // q*k^t  [2, G, S2Base]  [64 128]
                                              uint32_t qkVLStride,
                                              uint32_t qkStride,   // gSize * 256
                                              const LocalTensor<float> &weight_,   // w      [2, G]          [64    ]
                                              uint32_t weightStride,
                                              const LocalTensor<float> &kScale_,   // kScale [S2Base]        [128   ]
                                              uint32_t kScaleStride,
                                              const LocalTensor<float> &qScale_,   // qScale [2, G]          [64    ]
                                              uint32_t qScaleStride,
                                              const int gSize)                     // G 64
{
    auto weight0 = (__local_mem__ float*)weight_.GetPhyAddr();
    auto qScale0 = (__local_mem__ float*)qScale_.GetPhyAddr();
    auto kScale0 = (__local_mem__ float*)kScale_.GetPhyAddr();
    auto qk0 = (__local_mem__ bfloat16_t*)qk_.GetPhyAddr();
    auto out0 = (__local_mem__ uint16_t*)out_.GetPhyAddr();

    auto weight1 = weight0 + weightStride;
    auto qScale1 = qScale0 + qScaleStride;
    auto qk1 = qk0 + qkStride;
    // kScaleStride is zero
    auto out1 = out0 + outStride;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> regwBrc[2];
        AscendC::MicroAPI::RegTensor<float> regQK0[2];
        AscendC::MicroAPI::RegTensor<float> regQK1[2];
        AscendC::MicroAPI::RegTensor<float> regW[2];
        AscendC::MicroAPI::RegTensor<bfloat16_t> regQKB16[2];

        AscendC::MicroAPI::RegTensor<float> regQScale[2];
        AscendC::MicroAPI::RegTensor<float> regKScale[2];
        AscendC::MicroAPI::RegTensor<float> regSum0[2];
        AscendC::MicroAPI::RegTensor<float> regSum1[2];
        AscendC::MicroAPI::MaskReg maskAllB32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskAllB16 = AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();

        FloatSortConstCtx<bfloat16_t> bf16Ctx;
        InitFloatSortConstCtx(bf16Ctx, maskAllB16);

        using CastTrait = AscendC::MicroAPI::CastTrait;
        static constexpr CastTrait castTraitB162B32_EVEN = {AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
                                                            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr CastTrait castTraitB162B32_ODD  = {AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
                                                            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                       MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
        constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                                      MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

        AscendC::MicroAPI::LoadAlign<float>(regW[0], weight0);
        AscendC::MicroAPI::LoadAlign<float>(regW[1], weight1);
        AscendC::MicroAPI::LoadAlign<float>(regQScale[0], qScale0);
        AscendC::MicroAPI::LoadAlign<float>(regQScale[1], qScale1);
        AscendC::MicroAPI::Mul(regW[0], regW[0], regQScale[0], maskAllB32);
        AscendC::MicroAPI::Mul(regW[1], regW[1], regQScale[1], maskAllB32);
        // 读写依赖，寄存器可以保序
        AscendC::MicroAPI::StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM>(weight0, regW[0], maskAllB32);
        AscendC::MicroAPI::StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM>(weight1, regW[1], maskAllB32);
        DuplicateZero(regSum0, maskAllB32);
        DuplicateZero(regSum1, maskAllB32);

        // interleave load
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(regKScale[0], regKScale[1], kScale0);

        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i++) {
            AscendC::MicroAPI::LoadAlign<bfloat16_t>(regQKB16[0], qk0 + 256 * i); // RowStride是256, 行都落在一个bank上
            AscendC::MicroAPI::LoadAlign<bfloat16_t>(regQKB16[1], qk1 + 256 * i); // RowStride是256, 行都落在一个bank上
            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(regwBrc[0], weight0 + i);
            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(regwBrc[1], weight1 + i);
            // interleave cast
            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitB162B32_EVEN>(regQK0[0], regQKB16[0], maskAllB32);
            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitB162B32_ODD>(regQK0[1], regQKB16[0], maskAllB32);
            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitB162B32_EVEN>(regQK1[0], regQKB16[1], maskAllB32);
            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitB162B32_ODD>(regQK1[1], regQKB16[1], maskAllB32);
            AscendC::MicroAPI::MulAddDst(regSum0[0], regQK0[0], regwBrc[0], maskAllB32);
            AscendC::MicroAPI::MulAddDst(regSum0[1], regQK0[1], regwBrc[0], maskAllB32);
            AscendC::MicroAPI::MulAddDst(regSum1[0], regQK1[0], regwBrc[1], maskAllB32);
            AscendC::MicroAPI::MulAddDst(regSum1[1], regQK1[1], regwBrc[1], maskAllB32);
        }

        // Apply kScale scaling
        AscendC::MicroAPI::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
        AscendC::MicroAPI::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);
        AscendC::MicroAPI::Mul(regSum1[0], regSum1[0], regKScale[0], maskAllB32);
        AscendC::MicroAPI::Mul(regSum1[1], regSum1[1], regKScale[1], maskAllB32);

        // Convert to bfloat16 and store output channel
        AscendC::MicroAPI::RegTensor<bfloat16_t> regSumBF16[2];
        AscendC::MicroAPI::RegTensor<uint16_t> regOut[2];
        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[0], regSum0[1], maskAllB32);
        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[1], regSum1[1], maskAllB32);
        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[0], regSum0[0], maskAllB32);
        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[1], regSum1[0], maskAllB32);

        FloatX2ToSortableKey<bfloat16_t>(regOut[0], regOut[1], regSumBF16[0], regSumBF16[1], bf16Ctx, maskAllB16);
        AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out0, regOut[0], maskAllB16);
        AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out1, regOut[1], maskAllB16);
    }
}


template<typename QK_T, typename SCORE_T>
__aicore__ inline void BatchMulWeightAndReduceSum(const LocalTensor<SCORE_T> &out_,   // out    [S2Base]     [128   ]
                                                  uint32_t outStride,
                                                  const LocalTensor<QK_T> &qk_,       // q*k^t  [G, S2Base]  [64 128]
                                                  uint32_t qkVLStride,
                                                  uint32_t qkStride,
                                                  const LocalTensor<float> &weight_,   // w      [G]          [64    ]
                                                  uint32_t weightStride,
                                                  const LocalTensor<float> &kScale_,   // kScale [S2Base]     [128   ]
                                                  uint32_t kScaleStride,
                                                  const LocalTensor<float> &qScale_,   // qScale [G]          [64    ]
                                                  uint32_t qScaleStride,
                                                  const int gSize,                     // G 64
                                                  const int batch)
{
    // 暂只支持这两种情况, 后续改成循环
    if (batch != 2 && batch != 1) {
        return;
    }
    if (batch == 2) {
        MulWeightAndReduceSum2(out_, outStride,
                               qk_, qkVLStride, qkStride,
                               weight_, weightStride,
                               kScale_, kScaleStride,
                               qScale_, qScaleStride,
                               gSize);
    } else {
        MulWeightAndReduceSum(out_, qk_, qkVLStride, weight_, kScale_, qScale_, gSize);
    }
}

}

#endif