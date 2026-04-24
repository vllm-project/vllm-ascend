/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
    AscendC::MicroAPI::RegTensor<UInt> allOne;
    AscendC::MicroAPI::RegTensor<UInt> signMask;
    AscendC::MicroAPI::RegTensor<UInt> nan;
};


template <typename FloatT>
__simd_callee__ inline void InitFloatSortConstCtx(FloatSortConstCtx<FloatT>& ctx, AscendC::MicroAPI::MaskReg& maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    AscendC::MicroAPI::Duplicate(ctx.zeros,    Traits::ZERO,      maskAll);
    AscendC::MicroAPI::Duplicate(ctx.allOne,   Traits::ALL_ONE,   maskAll);
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
    AscendC::MicroAPI::Select(outKey, ctx.allOne, inBits, regSelectNan);

    // 3. sign bit
    AscendC::MicroAPI::And(regTemp, outKey, ctx.signMask, maskAll);

    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign, regTemp, ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::MicroAPI::Select(regMask, ctx.allOne, ctx.signMask, regSelectSign);
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
    AscendC::MicroAPI::Select(outKey0, ctx.allOne, inBits0, regSelectNan[0]);
    AscendC::MicroAPI::Select(outKey1, ctx.allOne, inBits1, regSelectNan[1]);

    // 3. sign bit
    AscendC::MicroAPI::And(regTemp[0], outKey0, ctx.signMask, maskAll);
    AscendC::MicroAPI::And(regTemp[1], outKey1, ctx.signMask, maskAll);

    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign[0], regTemp[0], ctx.zeros, maskAll);
    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign[1], regTemp[1], ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::MicroAPI::Select(regMask[0], ctx.allOne, ctx.signMask, regSelectSign[0]);
    AscendC::MicroAPI::Select(regMask[1], ctx.allOne, ctx.signMask, regSelectSign[1]);
    AscendC::MicroAPI::Xor(outKey0, outKey0, regMask[0], maskAll);
    AscendC::MicroAPI::Xor(outKey1, outKey1, regMask[1], maskAll);
}


// W * Relu(ScaleQ * Q * (ScaleK * K)^T)
// W * ScaleQ * Relu(Q * K^T) * ScaleK
// float in uint32 out
__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint32_t> &out,   // out    [S2Base]     [128   ] 2
                                             const LocalTensor<float> &qk,       // q*k^t  [G, S2Base]  [64 128] 2
                                             const LocalTensor<float> &weight,   // w      [G]          [64    ] 1
                                             const LocalTensor<float> &kScale,   // kScale [S2Base]     [128   ] 2 
                                             const LocalTensor<float> &qScale,   // qScale [G]          [64    ] 1
                                             const int gSize)                    // G 64
{
    __local_mem__ float* weight_ = (__local_mem__ float*)weight.GetPhyAddr();
    __local_mem__ float* qScale_ = (__local_mem__ float*)qScale.GetPhyAddr();

    constexpr uint32_t VL = 64; // vector length

    auto qk0 = (__local_mem__ float*)qk.GetPhyAddr();;
    auto qk1 = qk0 + VL;
    auto kScale0 = (__local_mem__ float*)kScale.GetPhyAddr();
    auto kScale1 = kScale0 + VL;
    auto out0 = (__local_mem__ uint32_t*)out.GetPhyAddr();
    auto out1 = out0 + VL;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> brcGatherIndex;
        AscendC::MicroAPI::RegTensor<float> regQK[2];
        AscendC::MicroAPI::RegTensor<float> regW;
        AscendC::MicroAPI::RegTensor<float> regwBrc;        
        AscendC::MicroAPI::RegTensor<float> regQScale;
        AscendC::MicroAPI::RegTensor<float> regKScale[2];
        AscendC::MicroAPI::RegTensor<float> regSum[2];

        AscendC::MicroAPI::MaskReg maskAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

        FloatSortConstCtx<float> fp32Ctx;
        InitFloatSortConstCtx(fp32Ctx, maskAll);

        AscendC::MicroAPI::LoadAlign<float>(regW, weight_);
        AscendC::MicroAPI::LoadAlign<float>(regQScale, qScale_);

        AscendC::MicroAPI::Duplicate(regSum[0], 0.0f, maskAll);
        AscendC::MicroAPI::Duplicate(regSum[1], 0.0f, maskAll);

        AscendC::MicroAPI::LoadAlign<float>(regKScale[0], kScale0);
        AscendC::MicroAPI::LoadAlign<float>(regKScale[1], kScale1); 
        AscendC::MicroAPI::Mul(regW, regW, regQScale, maskAll);

        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); ++i) {
            AscendC::MicroAPI::Duplicate(brcGatherIndex, i);
            AscendC::MicroAPI::LoadAlign<float>(regQK[0], qk0 + 128 * i);
            AscendC::MicroAPI::LoadAlign<float>(regQK[1], qk1 + 128 * i);
            AscendC::MicroAPI::Gather(regwBrc, regW, brcGatherIndex);

            AscendC::MicroAPI::Relu(regQK[0], regQK[0], maskAll);
            AscendC::MicroAPI::Relu(regQK[1], regQK[1], maskAll);

            AscendC::MicroAPI::MulAddDst(regSum[0], regQK[0], regwBrc, maskAll);
            AscendC::MicroAPI::MulAddDst(regSum[1], regQK[1], regwBrc, maskAll);
        }

        AscendC::MicroAPI::Mul(regSum[0], regSum[0], regKScale[0], maskAll);
        AscendC::MicroAPI::Mul(regSum[1], regSum[1], regKScale[1], maskAll);


        AscendC::MicroAPI::RegTensor<uint32_t> regOut[2];
        FloatX2ToSortableKey<float>(regOut[0], regOut[1], regSum[0], regSum[1], fp32Ctx, maskAll);

        AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out0, regOut[0], maskAll);
        AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out1, regOut[1], maskAll);
    }
}




// float in uint16 out
__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint16_t> &out_,   // out    [S2Base]     [128   ] 2
                                             const LocalTensor<float> &qk_,       // q*k^t  [G, S2Base]  [64 128] 2
                                             const LocalTensor<float> &weight_,    // w      [G]          [64    ] 1
                                             const LocalTensor<float> &kScale_,    // kScale [S2Base]     [128   ] 2 
                                             const LocalTensor<float> &qScale_,    // qScale [G]          [64    ] 1
                                             const int gSize)                     // G 64
{
    auto weight = (__local_mem__ float*)weight_.GetPhyAddr();
    auto qScale = (__local_mem__ float*)qScale_.GetPhyAddr();
    auto kScale = (__local_mem__ float*)kScale_.GetPhyAddr();
    auto qk = (__local_mem__ float*)qk_.GetPhyAddr();
    auto out = (__local_mem__ uint16_t*)out_.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> brcGatherIndex;
        AscendC::MicroAPI::RegTensor<float> regwBrc;
        AscendC::MicroAPI::RegTensor<float> regQK[2];
        AscendC::MicroAPI::RegTensor<float> regW;

        AscendC::MicroAPI::RegTensor<float> regQScale;
        AscendC::MicroAPI::RegTensor<float> regKScale[2];
        AscendC::MicroAPI::RegTensor<float> regSum[4];
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

        AscendC::MicroAPI::Duplicate(regSum[0], 0.0f, maskAllB32);
        AscendC::MicroAPI::Duplicate(regSum[1], 0.0f, maskAllB32);
        AscendC::MicroAPI::Duplicate(regSum[2], 0.0f, maskAllB32);
        AscendC::MicroAPI::Duplicate(regSum[3], 0.0f, maskAllB32);

        AscendC::MicroAPI::Mul(regW, regW, regQScale, maskAllB32);

        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(regKScale[0], regKScale[1], kScale);

        // unroll2
        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i += 2) {
            AscendC::MicroAPI::Duplicate(brcGatherIndex, i);
            // interleave load
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(regQK[0], regQK[1], qk + 128 * i);
            AscendC::MicroAPI::Gather(regwBrc, regW, brcGatherIndex);
            AscendC::MicroAPI::Relu(regQK[0], regQK[0], maskAllB32);
            AscendC::MicroAPI::Relu(regQK[1], regQK[1], maskAllB32);
            AscendC::MicroAPI::MulAddDst(regSum[0], regQK[0], regwBrc, maskAllB32);
            AscendC::MicroAPI::MulAddDst(regSum[1], regQK[1], regwBrc, maskAllB32);

            AscendC::MicroAPI::Duplicate(brcGatherIndex, i + 1);
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(regQK[0], regQK[1], qk + 128 * i + 128);
            AscendC::MicroAPI::Gather(regwBrc, regW, brcGatherIndex);
            AscendC::MicroAPI::Relu(regQK[0], regQK[0], maskAllB32);
            AscendC::MicroAPI::Relu(regQK[1], regQK[1], maskAllB32);
            AscendC::MicroAPI::MulAddDst(regSum[2], regQK[0], regwBrc, maskAllB32);
            AscendC::MicroAPI::MulAddDst(regSum[3], regQK[1], regwBrc, maskAllB32);
        }

        AscendC::MicroAPI::Add(regSum[0], regSum[0], regSum[2], maskAllB32);
        AscendC::MicroAPI::Add(regSum[1], regSum[1], regSum[3], maskAllB32);

        AscendC::MicroAPI::Mul(regSum[0], regSum[0], regKScale[0], maskAllB32);
        AscendC::MicroAPI::Mul(regSum[1], regSum[1], regKScale[1], maskAllB32);

        AscendC::MicroAPI::RegTensor<bfloat16_t> regSumBF16;
        // interleave back ==> regSum[1] high regSum[0] low
        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum[1], maskAllB32);
        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum[0], maskAllB32);

        AscendC::MicroAPI::RegTensor<uint16_t> regOut;
        FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
        AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out, regOut, maskAllB16);
    }
}


// W * Relu(ScaleQ * Q * (ScaleK * K)^T)
// W * ScaleQ * Relu(Q * K^T) * ScaleK
// float in uint16 out
__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint16_t> &out_,   // out    [S2Base]     [128   ] 2
                                             const LocalTensor<bfloat16_t> &qk_,  // q*k^t  [G, S2Base]  [64 128] 2
                                             const LocalTensor<float> &weight_,   // w      [G]          [64    ] 1
                                             const LocalTensor<float> &kScale_,   // kScale [S2Base]     [128   ] 2 
                                             const LocalTensor<float> &qScale_,   // qScale [G]          [64    ] 1
                                             const int gSize)                     // G 64
{
    // AscendC::DumpTensor(qk_, 0, gSize * 128);
    __local_mem__ float* weight = (__local_mem__ float*)weight_.GetPhyAddr();
    __local_mem__ float* qScale = (__local_mem__ float*)qScale_.GetPhyAddr();
    auto qk = (__local_mem__ bfloat16_t*)qk_.GetPhyAddr();
    auto kScale = (__local_mem__ float*)kScale_.GetPhyAddr();
    auto out = (__local_mem__ uint16_t*)out_.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> brcGatherIndex;
        AscendC::MicroAPI::RegTensor<float> regQK[4];
        AscendC::MicroAPI::RegTensor<bfloat16_t> regQKB16[2];
        AscendC::MicroAPI::RegTensor<float> regW;
        AscendC::MicroAPI::RegTensor<float> regwBrc[2];        
        AscendC::MicroAPI::RegTensor<float> regQScale;
        AscendC::MicroAPI::RegTensor<float> regKScale[2];
        AscendC::MicroAPI::RegTensor<float> regSum[4];

        AscendC::MicroAPI::MaskReg maskAllB32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskAllB16 = AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::RegTensor<bfloat16_t> regSumBF16;

        FloatSortConstCtx<bfloat16_t> bf16Ctx;
        InitFloatSortConstCtx(bf16Ctx, maskAllB16);


        using CastTrait = AscendC::MicroAPI::CastTrait;
        static constexpr CastTrait castTraitB162B32_EVEN = {AscendC::MicroAPI::RegLayout::ZERO,
                                                            AscendC::MicroAPI::SatMode::UNKNOWN, 
                                                            AscendC::MicroAPI::MaskMergeMode::ZEROING,
                                                            RoundMode::UNKNOWN};
        static constexpr CastTrait castTraitB162B32_ODD  = {AscendC::MicroAPI::RegLayout::ONE,
                                                            AscendC::MicroAPI::SatMode::UNKNOWN, 
                                                            AscendC::MicroAPI::MaskMergeMode::ZEROING,
                                                            RoundMode::UNKNOWN};

        constexpr static CastTrait castTraitF32ToF16_EVEN = {MicroAPI::RegLayout::ZERO,
                                                             MicroAPI::SatMode::NO_SAT,
                                                             MicroAPI::MaskMergeMode::MERGING,
                                                             RoundMode::CAST_ROUND};
        constexpr static CastTrait castTraitF32ToF16_ODD  = {MicroAPI::RegLayout::ONE,
                                                             MicroAPI::SatMode::NO_SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING,
                                                             RoundMode::CAST_ROUND};

        AscendC::MicroAPI::LoadAlign<float>(regW, weight);
        AscendC::MicroAPI::LoadAlign<float>(regQScale, qScale);
        AscendC::MicroAPI::Mul(regW, regW, regQScale, maskAllB32);

        AscendC::MicroAPI::Duplicate(regSum[0], 0.0f, maskAllB32);
        AscendC::MicroAPI::Duplicate(regSum[1], 0.0f, maskAllB32);
        AscendC::MicroAPI::Duplicate(regSum[2], 0.0f, maskAllB32);
        AscendC::MicroAPI::Duplicate(regSum[3], 0.0f, maskAllB32);

        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(regKScale[0], regKScale[1], kScale);

        AscendC::MicroAPI::StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM>(weight, regW, maskAllB32);

        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        
        // Duplicate + Gather方法劣化
        // Relu在cube随路做
        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i++) {
            AscendC::MicroAPI::LoadAlign<bfloat16_t>(regQKB16[0], qk + 128 * i);
            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(regwBrc[0], weight + i);
            // cast interleave
            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitB162B32_EVEN>(regQK[0], regQKB16[0], maskAllB16);
            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitB162B32_ODD>(regQK[1], regQKB16[0], maskAllB16);
            AscendC::MicroAPI::MulAddDst(regSum[0], regQK[0], regwBrc[0], maskAllB32);
            AscendC::MicroAPI::MulAddDst(regSum[1], regQK[1], regwBrc[0], maskAllB32);
        }

        AscendC::MicroAPI::Mul(regSum[0], regSum[0], regKScale[0], maskAllB32);
        AscendC::MicroAPI::Mul(regSum[1], regSum[1], regKScale[1], maskAllB32);
        // cast interleave back
        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum[1], maskAllB32);
        AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum[0], maskAllB32);

        AscendC::MicroAPI::RegTensor<uint16_t> regOut;
        FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
        AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out, regOut, maskAllB16);
    }
}

}

#endif