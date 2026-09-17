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
 * \file pool_key_indexer_vector1.h
 * \brief
 */
#ifndef POOL_KEY_INDEXER_VECTOR1_H
#define POOL_KEY_INDEXER_VECTOR1_H

#include "kernel_operator.h"
#include "common/pool_key_indexer_vector1_base.h"

namespace vector1 {

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

template <typename FloatT>
struct UIntSortConstCtx {
    using Traits = UIntSortTraits<FloatT>;
    using UInt = typename Traits::UInt;
    AscendC::Reg::RegTensor<UInt> zeros;
    AscendC::Reg::RegTensor<UInt> allOnes;
    AscendC::Reg::RegTensor<UInt> signMask;
    AscendC::Reg::RegTensor<UInt> nan;
};

template <typename FloatT>
__simd_callee__ inline void InitUIntSortConstCtx(UIntSortConstCtx<FloatT> &ctx, AscendC::Reg::MaskReg &maskAll)
{
    using Traits = UIntSortTraits<FloatT>;
    AscendC::Reg::Duplicate(ctx.zeros, Traits::ZERO, maskAll);
    AscendC::Reg::Duplicate(ctx.allOnes, Traits::ALL_ONE, maskAll);
    AscendC::Reg::Duplicate(ctx.signMask, Traits::SIGN_MASK, maskAll);
    AscendC::Reg::Duplicate(ctx.nan, Traits::NAN_MASK, maskAll);
}

template <typename FloatT>
__simd_callee__ inline void UIntToSortableKey(AscendC::Reg::RegTensor<FloatT> &outKey,
                                              AscendC::Reg::RegTensor<typename UIntSortConstCtx<FloatT>::UInt> &inVal,
                                              UIntSortConstCtx<FloatT> &ctx, AscendC::Reg::MaskReg &maskAll)
{
    using Traits = UIntSortTraits<FloatT>;
    using UInt = typename Traits::UInt;

    AscendC::Reg::RegTensor<UInt> regTemp;
    AscendC::Reg::RegTensor<UInt> regMask;
    AscendC::Reg::MaskReg regSelectZero;
    AscendC::Reg::MaskReg regSelectSign;

    auto &inBits = inVal;

    // 1. 0 check
    AscendC::Reg::Compare<UInt, CMPMODE::EQ>(regSelectZero, inBits, ctx.zeros, maskAll);

    // 2. 0 -> -NAN
    AscendC::Reg::Select((AscendC::Reg::RegTensor<UInt> &)outKey, ctx.nan, inBits, regSelectZero);

    // 3. sign bit
    AscendC::Reg::And(regTemp, (AscendC::Reg::RegTensor<UInt> &)outKey, ctx.signMask, maskAll);

    AscendC::Reg::Compare<UInt, CMPMODE::GT>(regSelectSign, regTemp, ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::Reg::Select(regMask, ctx.signMask, ctx.allOnes, regSelectSign);
    AscendC::Reg::Xor((AscendC::Reg::RegTensor<UInt> &)outKey, (AscendC::Reg::RegTensor<UInt> &)outKey, regMask,
                      maskAll);
}

__aicore__ inline void UIntToFloatReturnValue(const LocalTensor<bfloat16_t> &out_, const LocalTensor<uint32_t> &in,
                                              const uint32_t topK)
{
    auto outBuf = (__local_mem__ bfloat16_t *)out_.GetPhyAddr();
    auto inBuf = (__local_mem__ uint32_t *)in.GetPhyAddr();

    const uint16_t repeatSize32 = 128;
    uint16_t topkLoopNum = (topK + repeatSize32 - 1) / repeatSize32;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint32_t> regIn[2];
        AscendC::Reg::RegTensor<float> regOut[2];
        AscendC::Reg::RegTensor<bfloat16_t> regOutBF16[2];
        AscendC::Reg::RegTensor<bfloat16_t> regOutValue;
        AscendC::Reg::RegTensor<bfloat16_t> regInvalid;
        AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();
        constexpr static Reg::CastTrait castTraitFP32ToBF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < topkLoopNum; ++i) {
            AscendC::Reg::LoadAlign<uint32_t>(regIn[0], inBuf + i * repeatSize32);
            AscendC::Reg::LoadAlign<uint32_t>(regIn[1], inBuf + i * repeatSize32 + 64);
            UIntSortConstCtx<float> uint32Ctx;
            InitUIntSortConstCtx(uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[0], regIn[0], uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[1], regIn[1], uint32Ctx, maskAllB32);

            AscendC::Reg::Cast<bfloat16_t, float, castTraitFP32ToBF16>(regOutBF16[0], regOut[0], maskAllB32);
            AscendC::Reg::Cast<bfloat16_t, float, castTraitFP32ToBF16>(regOutBF16[1], regOut[1], maskAllB32);

            AscendC::Reg::DeInterleave(regOutValue, regInvalid, regOutBF16[0], regOutBF16[1]);

            AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_NORM>(outBuf + i * repeatSize32,
                                                                                     regOutValue, maskAllB16);
        }
    }
}

__aicore__ inline void UIntToFloatReturnValue(const LocalTensor<half> &out_, const LocalTensor<uint32_t> &in,
                                              const uint32_t topK)
{
    auto outBuf = (__local_mem__ half *)out_.GetPhyAddr();
    auto inBuf = (__local_mem__ uint32_t *)in.GetPhyAddr();

    const uint16_t repeatSize32 = 128;
    uint16_t topkLoopNum = (topK + repeatSize32 - 1) / repeatSize32;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint32_t> regIn[2];
        AscendC::Reg::RegTensor<float> regOut[2];
        AscendC::Reg::RegTensor<half> regOutFP16[2];
        AscendC::Reg::RegTensor<half> regOutValue;
        AscendC::Reg::RegTensor<half> regInvalid;
        AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<half, AscendC::Reg::MaskPattern::ALL>();
        constexpr static Reg::CastTrait castTraitFP32ToFP16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < topkLoopNum; ++i) {
            AscendC::Reg::LoadAlign<uint32_t>(regIn[0], inBuf + i * repeatSize32);
            AscendC::Reg::LoadAlign<uint32_t>(regIn[1], inBuf + i * repeatSize32 + 64);
            UIntSortConstCtx<float> uint32Ctx;
            InitUIntSortConstCtx(uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[0], regIn[0], uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[1], regIn[1], uint32Ctx, maskAllB32);

            AscendC::Reg::Cast<half, float, castTraitFP32ToFP16>(regOutFP16[0], regOut[0], maskAllB32);
            AscendC::Reg::Cast<half, float, castTraitFP32ToFP16>(regOutFP16[1], regOut[1], maskAllB32);

            AscendC::Reg::DeInterleave(regOutValue, regInvalid, regOutFP16[0], regOutFP16[1]);

            AscendC::Reg::StoreAlign<half, AscendC::Reg::StoreDist::DIST_NORM>(outBuf + i * repeatSize32, regOutValue,
                                                                               maskAllB16);
        }
    }
}

// PKI: float output version (no Cast needed, values output is FLOAT)
__aicore__ inline void UIntToFloatReturnValue(const LocalTensor<float> &out_, const LocalTensor<uint32_t> &in,
                                              const uint32_t topK)
{
    auto outBuf = (__local_mem__ float *)out_.GetPhyAddr();
    auto inBuf = (__local_mem__ uint32_t *)in.GetPhyAddr();

    const uint16_t repeatSize32 = 128;
    uint16_t topkLoopNum = (topK + repeatSize32 - 1) / repeatSize32;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint32_t> regIn[2];
        AscendC::Reg::RegTensor<float> regOut[2];
        AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskAllB32f = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        for (uint16_t i = 0; i < topkLoopNum; ++i) {
            AscendC::Reg::LoadAlign<uint32_t>(regIn[0], inBuf + i * repeatSize32);
            AscendC::Reg::LoadAlign<uint32_t>(regIn[1], inBuf + i * repeatSize32 + 64);
            UIntSortConstCtx<float> uint32Ctx;
            InitUIntSortConstCtx(uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[0], regIn[0], uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[1], regIn[1], uint32Ctx, maskAllB32);

            // float 入/出同宽(4B->4B), 不能沿用 bf16/fp16 的 DeInterleave
            // 半宽合并(同宽对上是 2:1 抽取丢元素); 两个 64 lane 寄存器分别落回原位
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(outBuf + i * repeatSize32, regOut[0],
                                                                                maskAllB32f);
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(outBuf + i * repeatSize32 + 64,
                                                                                regOut[1], maskAllB32f);
        }
    }
}

__simd_callee__ inline void BroadcastLane(AscendC::Reg::RegTensor<float> &dst, __local_mem__ float *src,
                                          uint16_t laneIdx)
{
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(dst, src + laneIdx);
}

template <typename W_T>
__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint32_t> &out, // out    [S2Base]     [128   ] 2
                                             const LocalTensor<float> &qk,     // q*k^t  [G, S2Base]  [64 128] 2
                                             const LocalTensor<W_T> &weight,   // w      [G]          [64    ] 1
                                             const int gSize,                  // G 64
                                             const float scale)                // 1/sqrt(headDim)
{
    __local_mem__ W_T *weight_ = (__local_mem__ W_T *)weight.GetPhyAddr();

    constexpr uint32_t VL = 64; // vector length

    auto qk0 = (__local_mem__ float *)qk.GetPhyAddr();
    auto qk1 = qk0 + VL;
    auto out0 = (__local_mem__ uint32_t *)out.GetPhyAddr();
    auto out1 = out0 + VL;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint32_t> brcGatherIndex;
        AscendC::Reg::RegTensor<float> regQK[2];
        AscendC::Reg::RegTensor<float> regW;
        AscendC::Reg::RegTensor<float> regwBrc;
        AscendC::Reg::RegTensor<float> regQScale;
        AscendC::Reg::RegTensor<float> regKScale[2];
        AscendC::Reg::RegTensor<float> regSum[2];
        AscendC::Reg::RegTensor<float> regScale;
        AscendC::Reg::RegTensor<W_T> regWWT;

        AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskAll16 = AscendC::Reg::CreateMask<W_T, AscendC::Reg::MaskPattern::ALL>();

        FloatSortConstCtx<float> fp32Ctx;
        InitFloatSortConstCtx(fp32Ctx, maskAll);

        constexpr static Reg::CastTrait castTraitWTToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                             Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        AscendC::Reg::LoadAlign<W_T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regWWT, weight_);
        AscendC::Reg::Cast<float, W_T, castTraitWTToFP32>(regW, regWWT, maskAll16);

        AscendC::Reg::Duplicate(regSum[0], 0.0f, maskAll);
        AscendC::Reg::Duplicate(regSum[1], 0.0f, maskAll);

        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); ++i) {
            AscendC::Reg::Duplicate(brcGatherIndex, i);
            AscendC::Reg::LoadAlign<float>(regQK[0], qk0 + 128 * i);
            AscendC::Reg::LoadAlign<float>(regQK[1], qk1 + 128 * i);
            AscendC::Reg::Gather(regwBrc, regW, brcGatherIndex);

            AscendC::Reg::Relu(regQK[0], regQK[0], maskAll);
            AscendC::Reg::Relu(regQK[1], regQK[1], maskAll);

            AscendC::Reg::MulAddDst(regSum[0], regQK[0], regwBrc, maskAll);
            AscendC::Reg::MulAddDst(regSum[1], regQK[1], regwBrc, maskAll);
        }

        // 池级分数乘 1/sqrt(headDim)(文档公式 S = Q@K^T/sqrt(headDim));
        // 缩放为正数, 与 ReLU/加权求和可交换, 在聚合后统一应用
        AscendC::Reg::Duplicate(regScale, scale, maskAll);
        AscendC::Reg::Mul(regSum[0], regSum[0], regScale, maskAll);
        AscendC::Reg::Mul(regSum[1], regSum[1], regScale, maskAll);

        AscendC::Reg::RegTensor<uint32_t> regOut[2];
        FloatX2ToSortableKey<float>(regOut[0], regOut[1], regSum[0], regSum[1], fp32Ctx, maskAll);

        AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out0, regOut[0], maskAll);
        AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out1, regOut[1], maskAll);
    }
}

// mode=0 (FP8 per-token-head) scale 融合版: scale 为正标量, 可从 D 维点积中
// 完全提出, out = (Σ_g relu(Σ_d q·k)·w_g·dq)·dk; weight 为已预乘 qScale 的 fp32 UB
__aicore__ inline void MulWeightAndReduceSumWithScale(
    const LocalTensor<uint32_t> &out, // out    [S2Base]     [128   ] 2
    const LocalTensor<float> &qk,     // q*k^t  [G, S2Base]  [64 128] 2
    const LocalTensor<float> &weight, // w*dq   [G]          [64    ] 1
    const LocalTensor<float> &kScale, // kScale [S2Base]     [128   ] 2
    const int gSize,                  // G 64
    const float scale)                // 1/sqrt(headDim)
{
    __local_mem__ float *weight_ = (__local_mem__ float *)weight.GetPhyAddr();
    __local_mem__ float *kScale_ = (__local_mem__ float *)kScale.GetPhyAddr();

    constexpr uint32_t VL = 64; // vector length

    auto qk0 = (__local_mem__ float *)qk.GetPhyAddr();
    auto qk1 = qk0 + VL;
    auto out0 = (__local_mem__ uint32_t *)out.GetPhyAddr();
    auto out1 = out0 + VL;
    auto kScale0 = kScale_;
    auto kScale1 = kScale_ + VL;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> regQK[2];
        AscendC::Reg::RegTensor<float> regwBrc;
        AscendC::Reg::RegTensor<float> regKScale[2];
        AscendC::Reg::RegTensor<float> regSum[2];
        AscendC::Reg::RegTensor<float> regScale;

        AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

        FloatSortConstCtx<float> fp32Ctx;
        InitFloatSortConstCtx(fp32Ctx, maskAll);

        AscendC::Reg::Duplicate(regSum[0], 0.0f, maskAll);
        AscendC::Reg::Duplicate(regSum[1], 0.0f, maskAll);

        AscendC::Reg::LoadAlign<float>(regKScale[0], kScale0);
        AscendC::Reg::LoadAlign<float>(regKScale[1], kScale1);

        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); ++i) {
            // BroadcastLane: 从 UB 直接广播第 i 个 weight
            // (Gather 与 LoadAlign fp32 产物的 lane 布局不兼容)
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(regwBrc, weight_ + i);
            AscendC::Reg::LoadAlign<float>(regQK[0], qk0 + 128 * i);
            AscendC::Reg::LoadAlign<float>(regQK[1], qk1 + 128 * i);

            AscendC::Reg::Relu(regQK[0], regQK[0], maskAll);
            AscendC::Reg::Relu(regQK[1], regQK[1], maskAll);

            AscendC::Reg::MulAddDst(regSum[0], regQK[0], regwBrc, maskAll);
            AscendC::Reg::MulAddDst(regSum[1], regQK[1], regwBrc, maskAll);
        }

        // kScale 乘入聚合和(逐 s2 元素), 随后统一乘 1/sqrt(headDim)
        AscendC::Reg::Mul(regSum[0], regSum[0], regKScale[0], maskAll);
        AscendC::Reg::Mul(regSum[1], regSum[1], regKScale[1], maskAll);

        // 池级分数乘 1/sqrt(headDim)(文档公式 S = Q@K^T/sqrt(headDim));
        // 缩放为正数, 与 ReLU/加权求和可交换, 在聚合后统一应用
        AscendC::Reg::Duplicate(regScale, scale, maskAll);
        AscendC::Reg::Mul(regSum[0], regSum[0], regScale, maskAll);
        AscendC::Reg::Mul(regSum[1], regSum[1], regScale, maskAll);

        AscendC::Reg::RegTensor<uint32_t> regOut[2];
        FloatX2ToSortableKey<float>(regOut[0], regOut[1], regSum[0], regSum[1], fp32Ctx, maskAll);

        AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out0, regOut[0], maskAll);
        AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out1, regOut[1], maskAll);
    }
}
// mode=0: weight×qScale SIMD 预乘(单 s1 行, 64 lanes): 半精度 weight -> fp32
// cast -> 乘 qScale -> 落 fp32 UB; 独立 __simd_vf__ 保证 bf16 Cast 硬件展开
template <typename W_T>
__simd_vf__ void MulWeightQScaleSIMD(__local_mem__ W_T *weight_, __local_mem__ float *qScale_,
                                     __local_mem__ float *dst_)
{
    Reg::MaskReg maskAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskAllB16 = Reg::CreateMask<W_T, Reg::MaskPattern::ALL>();

    Reg::RegTensor<W_T> regWHalf;
    Reg::RegTensor<float> regW;
    Reg::RegTensor<float> regQScale;

    constexpr static Reg::CastTrait castTraitWTToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    Reg::LoadAlign<W_T, Reg::LoadDist::DIST_UNPACK_B16>(regWHalf, weight_);
    Reg::Cast<float, W_T, castTraitWTToFP32>(regW, regWHalf, maskAllB16);
    Reg::LoadAlign<float>(regQScale, qScale_);
    Reg::Mul(regW, regW, regQScale, maskAllB32);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(dst_, regW, maskAllB32);
}
} // namespace vector1

#endif
