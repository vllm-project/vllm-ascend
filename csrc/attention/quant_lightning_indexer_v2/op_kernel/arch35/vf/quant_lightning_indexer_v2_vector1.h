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
 * \file quant_lightning_indexer_v2_vector1.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_VECTOR1_H
#define QUANT_LIGHTNING_INDEXER_V2_VECTOR1_H

#include "kernel_operator.h"
#if __has_include("../../../lightning_indexer_v2/arch35/vf/common/lightning_indexer_v2_vector1_base.h")
#include "../../../lightning_indexer_v2/arch35/vf/common/lightning_indexer_v2_vector1_base.h"
#else
#include "../../../../lightning_indexer_v2/op_kernel/arch35/vf/common/lightning_indexer_v2_vector1_base.h"
#endif

namespace vector1 {
__simd_vf__ void UIntToFloatReturnValueVF(__ubuf__ bfloat16_t *outBuf, __ubuf__ uint16_t *inBuf, uint16_t vfLoop)
{
    MicroAPI::RegTensor<uint16_t> regIn;
    MicroAPI::RegTensor<bfloat16_t> regOut;
    MicroAPI::MaskReg maskAllB16 = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint16_t>(regIn, inBuf + i * 128);

        liV2Vector1::UIntSortConstCtx<bfloat16_t> uint16Ctx;
        liV2Vector1::InitUIntSortConstCtx(uint16Ctx, maskAllB16);

        liV2Vector1::UIntToSortableKey<bfloat16_t>(regOut, regIn, uint16Ctx, maskAllB16);

        MicroAPI::StoreAlign<bfloat16_t, MicroAPI::StoreDist::DIST_NORM>(outBuf + i * 128, regOut, maskAllB16);
    }
}

__aicore__ inline void UIntToFloatReturnValue(const LocalTensor<bfloat16_t> &out_, const LocalTensor<uint16_t> &in,
                                              const uint32_t topK)
{
    __ubuf__ bfloat16_t *outBuf = (__ubuf__ bfloat16_t *)out_.GetPhyAddr();
    __ubuf__ uint16_t *inBuf = (__ubuf__ uint16_t *)in.GetPhyAddr();
    const uint16_t repeatSize16 = 128;
    uint16_t topkLoopNum = (topK + repeatSize16 - 1) / repeatSize16;
    UIntToFloatReturnValueVF(outBuf, inBuf, topkLoopNum);
}

// 可排序键还原为 bf16 返回值，并将无效位（score==0）刷为 -inf
__simd_vf__ void UIntToFloatReturnValueWithInfMaskVF(__ubuf__ bfloat16_t *valueOutBuf, __ubuf__ uint16_t *scoreOutBuf,
                                                     uint16_t vfLoop, uint16_t negInfBits)
{
    MicroAPI::RegTensor<uint16_t> regIn;
    MicroAPI::RegTensor<bfloat16_t> regOut;
    MicroAPI::RegTensor<uint16_t> regNegInf;
    MicroAPI::RegTensor<uint16_t> regZero;
    MicroAPI::MaskReg maskAllB16 = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskInvalid;

    // 常量寄存器初始化：-inf(0xFF80) 与 0（用于识别无效位）
    MicroAPI::Duplicate(regNegInf, negInfBits, maskAllB16);
    MicroAPI::Duplicate(regZero, (uint16_t)0, maskAllB16);

    liV2Vector1::UIntSortConstCtx<bfloat16_t> uint16Ctx;
    liV2Vector1::InitUIntSortConstCtx(uint16Ctx, maskAllB16);

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint16_t>(regIn, scoreOutBuf + i * 128);
        // 比较得无效位掩码：score==0 即无效位（可排序键性质，真实值永不为0）
        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(maskInvalid, regIn, regZero, maskAllB16);
        // 逆变换：可排序键 → bf16 值
        liV2Vector1::UIntToSortableKey<bfloat16_t>(regOut, regIn, uint16Ctx, maskAllB16);
        // 无效位覆盖为 -inf，有效位保留还原值
        MicroAPI::Select((MicroAPI::RegTensor<uint16_t> &)regOut, regNegInf, (MicroAPI::RegTensor<uint16_t> &)regOut,
                         maskInvalid);
        MicroAPI::StoreAlign<bfloat16_t, MicroAPI::StoreDist::DIST_NORM>(valueOutBuf + i * 128, regOut, maskAllB16);
    }
}

__aicore__ inline void UIntToFloatReturnValueWithInfMask(const LocalTensor<bfloat16_t> &valueOutLocal,
                                                         const LocalTensor<uint16_t> &scoreOutLocal,
                                                         const uint32_t topK, const uint16_t negInfBits)
{
    __ubuf__ bfloat16_t *valueOutBuf = (__ubuf__ bfloat16_t *)valueOutLocal.GetPhyAddr();
    __ubuf__ uint16_t *scoreOutBuf = (__ubuf__ uint16_t *)scoreOutLocal.GetPhyAddr();
    const uint16_t repeatSize16 = 128;
    uint16_t topkLoopNum = (topK + repeatSize16 - 1) / repeatSize16;
    UIntToFloatReturnValueWithInfMaskVF(valueOutBuf, scoreOutBuf, topkLoopNum, negInfBits);
}

__simd_callee__ inline void LoadKScaleFP16(AscendC::MicroAPI::RegTensor<half> (&regKScaleFP16)[2],
                                           AscendC::MicroAPI::RegTensor<float> (&regKScale)[2],
                                           AscendC::MicroAPI::MaskReg &maskAllB16,
                                           __ubuf__ half *kScale_)
{
    constexpr static MicroAPI::CastTrait castTraitFP16ToFP32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    AscendC::MicroAPI::LoadAlign<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(regKScaleFP16[0], kScale_);
    AscendC::MicroAPI::LoadAlign<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(regKScaleFP16[1], kScale_ + 64);
    AscendC::MicroAPI::Cast<float, half, castTraitFP16ToFP32>(regKScale[0], regKScaleFP16[0], maskAllB16);
    AscendC::MicroAPI::Cast<float, half, castTraitFP16ToFP32>(regKScale[1], regKScaleFP16[1], maskAllB16);
}

__simd_callee__ inline void CastFP32ToFP16ToFP32(AscendC::MicroAPI::RegTensor<float> (&regQK0)[2],
                                                 AscendC::MicroAPI::RegTensor<half> (&regQK0Half)[2],
                                                 AscendC::MicroAPI::MaskReg &maskAllB32)
{
    AscendC::MicroAPI::MaskReg maskAllB16 = AscendC::MicroAPI::CreateMask<bfloat16_t,
                                                                          AscendC::MicroAPI::MaskPattern::ALL>();
    constexpr static MicroAPI::CastTrait castTraitFP32ToFP16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr static MicroAPI::CastTrait castTraitFP16ToFP32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    float mulsScalar = 1.0 / 1024;

    MicroAPI::Muls(regQK0[0], regQK0[0], mulsScalar, maskAllB32);
    MicroAPI::Muls(regQK0[1], regQK0[1], mulsScalar, maskAllB32);

    MicroAPI::Cast<half, float, castTraitFP32ToFP16>(regQK0Half[0], regQK0[0], maskAllB32);
    MicroAPI::Cast<half, float, castTraitFP32ToFP16>(regQK0Half[1], regQK0[1], maskAllB32);

    MicroAPI::Cast<float, half, castTraitFP16ToFP32>(regQK0[0], regQK0Half[0], maskAllB16);
    MicroAPI::Cast<float, half, castTraitFP16ToFP32>(regQK0[1], regQK0Half[1], maskAllB16);
}

// int32 in uint16 out
__simd_vf__ void MulWeightAndReduceSumInt32GSizeOddVF(__ubuf__ uint16_t *out, __ubuf__ int32_t *qk,
                                                       uint32_t qkVLStride, __ubuf__ half *weight,
                                                       __ubuf__ half *kScale, __ubuf__ half *qScale,
                                                       uint16_t gSize)
{
    MicroAPI::RegTensor<float> regwBrc;
    MicroAPI::RegTensor<float> regQK[2];
    MicroAPI::RegTensor<half> regQKHalf[2];
    MicroAPI::RegTensor<int32_t> regQKInt32[2];
    MicroAPI::RegTensor<float> regW;
    MicroAPI::RegTensor<half> regWFP16;
    MicroAPI::RegTensor<half> regWFP16Temp;
    MicroAPI::RegTensor<float> regQScale;
    MicroAPI::RegTensor<half> regQScaleFP16;
    MicroAPI::RegTensor<float> regKScale[2];
    MicroAPI::RegTensor<half> regKScaleFP16[2];
    MicroAPI::RegTensor<float> regSum0[2];
    MicroAPI::RegTensor<float> regSum1[2];
    MicroAPI::MaskReg maskAllB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskAllB16 = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {MicroAPI::RegLayout::ZERO,
                                                                   MicroAPI::SatMode::NO_SAT,
                                                                   MicroAPI::MaskMergeMode::MERGING,
                                                                   RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {MicroAPI::RegLayout::ONE,
                                                                  MicroAPI::SatMode::NO_SAT,
                                                                  MicroAPI::MaskMergeMode::ZEROING,
                                                                  RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF16ToF32 = {MicroAPI::RegLayout::ZERO,
                                                              MicroAPI::SatMode::UNKNOWN,
                                                              MicroAPI::MaskMergeMode::ZEROING,
                                                              RoundMode::UNKNOWN};
    constexpr static MicroAPI::CastTrait castTraitInt32ToFP32 = {MicroAPI::RegLayout::UNKNOWN,
                                                                 MicroAPI::SatMode::NO_SAT,
                                                                 MicroAPI::MaskMergeMode::ZEROING,
                                                                 RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16 = {MicroAPI::RegLayout::ZERO,
                                                              MicroAPI::SatMode::NO_SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING,
                                                              RoundMode::CAST_RINT};
    MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(regWFP16, weight);
    MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(regQScaleFP16, qScale);
    MicroAPI::Cast<float, half, castTraitF16ToF32>(regW, regWFP16, maskAllB16);
    MicroAPI::Cast<float, half, castTraitF16ToF32>(regQScale, regQScaleFP16, maskAllB16);
    MicroAPI::Mul(regW, regW, regQScale, maskAllB32);
    MicroAPI::Cast<half, float, castTraitF32ToF16>(regWFP16Temp, regW, maskAllB32);
    MicroAPI::Cast<float, half, castTraitF16ToF32>(regW, regWFP16Temp, maskAllB16);
    liV2Vector1::DuplicateZero(regSum0, maskAllB32);
    liV2Vector1::DuplicateZero(regSum1, maskAllB32);

    LoadKScaleFP16(regKScaleFP16, regKScale, maskAllB16, kScale);
    // float mulsScalar = 1.0f / 1024;
    // unroll2
    for (uint16_t i = (uint16_t)(0); (uint16_t)(i + 1) < gSize; i += 2) {
        MicroAPI::LoadAlign<int32_t>(regQKInt32[0], qk + 128 * i);
        MicroAPI::LoadAlign<int32_t>(regQKInt32[1], qk + 128 * i + qkVLStride);
        MicroAPI::Cast<float, int32_t, castTraitInt32ToFP32>(regQK[0], regQKInt32[0], maskAllB32);
        MicroAPI::Cast<float, int32_t, castTraitInt32ToFP32>(regQK[1], regQKInt32[1], maskAllB32);

        CastFP32ToFP16ToFP32(regQK, regQKHalf, maskAllB32);

        liV2Vector1::BroadcastLane(regwBrc, regW, i);
        liV2Vector1::WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

        MicroAPI::LoadAlign<int32_t>(regQKInt32[0], qk + 128 * i + 128);
        MicroAPI::LoadAlign<int32_t>(regQKInt32[1], qk + 128 * i + 128 + qkVLStride);
        MicroAPI::Cast<float, int32_t, castTraitInt32ToFP32>(regQK[0], regQKInt32[0], maskAllB32);
        MicroAPI::Cast<float, int32_t, castTraitInt32ToFP32>(regQK[1], regQKInt32[1], maskAllB32);

        CastFP32ToFP16ToFP32(regQK, regQKHalf, maskAllB32);

        liV2Vector1::BroadcastLane(regwBrc, regW, i + 1);
        liV2Vector1::WeightedAccum(regSum1, regQK, regwBrc, maskAllB32);
    }

    MicroAPI::LoadAlign<int32_t>(regQKInt32[0], qk + 128 * (gSize - 1));
    MicroAPI::LoadAlign<int32_t>(regQKInt32[1], qk + 128 * (gSize - 1) + qkVLStride);
    MicroAPI::Cast<float, int32_t, castTraitInt32ToFP32>(regQK[0], regQKInt32[0], maskAllB32);
    MicroAPI::Cast<float, int32_t, castTraitInt32ToFP32>(regQK[1], regQKInt32[1], maskAllB32);

    CastFP32ToFP16ToFP32(regQK, regQKHalf, maskAllB32);

    liV2Vector1::BroadcastLane(regwBrc, regW, gSize - 1);
    liV2Vector1::WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

    MicroAPI::Add(regSum0[0], regSum0[0], regSum1[0], maskAllB32);
    MicroAPI::Add(regSum0[1], regSum0[1], regSum1[1], maskAllB32);

    MicroAPI::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
    MicroAPI::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);

    MicroAPI::RegTensor<bfloat16_t> regSumBF16;
    // interleave cast ==> regSum[1] high regSum[0] low
    MicroAPI::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum0[1], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum0[0], maskAllB32);

    MicroAPI::RegTensor<uint16_t> regOut;
    liV2Vector1::FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
    // normal store
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(out, regOut, maskAllB16);
}

// float in uint16 out
__simd_vf__ void MulWeightAndReduceSumF32GSizeOddVF(__ubuf__ uint16_t *out, __ubuf__ float *qk, uint32_t qkVLStride,
                                                     __ubuf__ float *weight, __ubuf__ float *kScale,
                                                     __ubuf__ float *qScale, uint16_t gSize)
{
    MicroAPI::RegTensor<float> regwBrc;
    MicroAPI::RegTensor<float> regQK[2];
    MicroAPI::RegTensor<float> regW;

    MicroAPI::RegTensor<float> regQScale;
    MicroAPI::RegTensor<float> regKScale[2];
    MicroAPI::RegTensor<float> regSum0[2];
    MicroAPI::RegTensor<float> regSum1[2];
    MicroAPI::MaskReg maskAllB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskAllB16 = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {
        MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    MicroAPI::LoadAlign<float>(regW, weight);
    MicroAPI::LoadAlign<float>(regQScale, qScale);
    MicroAPI::Mul(regW, regW, regQScale, maskAllB32);

    liV2Vector1::DuplicateZero(regSum0, maskAllB32);
    liV2Vector1::DuplicateZero(regSum1, maskAllB32);

    MicroAPI::LoadAlign<float>(regKScale[0], kScale);
    MicroAPI::LoadAlign<float>(regKScale[1], kScale + 64);

    // unroll2
    for (uint16_t i = (uint16_t)(0); (uint16_t)(i + 1) < gSize; i += 2) {
        MicroAPI::LoadAlign<float>(regQK[0], qk + 128 * i); // RowStride是128, 行都落在一个bank上
        MicroAPI::LoadAlign<float>(regQK[1], qk + 128 * i + qkVLStride);
        liV2Vector1::BroadcastLane(regwBrc, regW, i);
        liV2Vector1::WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

        MicroAPI::LoadAlign<float>(regQK[0], qk + 128 * i + 128);
        MicroAPI::LoadAlign<float>(regQK[1], qk + 128 * i + 128 + qkVLStride);
        liV2Vector1::BroadcastLane(regwBrc, regW, i + 1);
        liV2Vector1::WeightedAccum(regSum1, regQK, regwBrc, maskAllB32);
    }

    MicroAPI::LoadAlign<float>(regQK[0], qk + 128 * (gSize - 1)); // RowStride是128, 行都落在一个bank上
    MicroAPI::LoadAlign<float>(regQK[1], qk + 128 * (gSize - 1) + qkVLStride);
    liV2Vector1::BroadcastLane(regwBrc, regW, (gSize - 1));
    liV2Vector1::WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

    MicroAPI::Add(regSum0[0], regSum0[0], regSum1[0], maskAllB32);
    MicroAPI::Add(regSum0[1], regSum0[1], regSum1[1], maskAllB32);

    MicroAPI::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
    MicroAPI::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);

    MicroAPI::RegTensor<bfloat16_t> regSumBF16;
    // interleave cast ==> regSum[1] high regSum[0] low
    MicroAPI::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum0[1], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum0[0], maskAllB32);

    MicroAPI::RegTensor<uint16_t> regOut;
    liV2Vector1::FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
    // normal store
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(out, regOut, maskAllB16);
}

// int32 in uint16 out
__simd_vf__ void MulWeightAndReduceSumInt32GSizeEvenVF(__ubuf__ uint16_t *out, __ubuf__ int32_t *qk,
                                                       uint32_t qkVLStride, __ubuf__ half *weight,
                                                       __ubuf__ half *kScale, __ubuf__ half *qScale,
                                                       uint16_t gSize)
{
    MicroAPI::RegTensor<float> regwBrc;
    MicroAPI::RegTensor<float> regQK[2];
    MicroAPI::RegTensor<half> regQKHalf[2];
    MicroAPI::RegTensor<int32_t> regQKInt32[2];
    MicroAPI::RegTensor<float> regW;
    MicroAPI::RegTensor<half> regWFP16;
    MicroAPI::RegTensor<half> regWFP16Temp;
    MicroAPI::RegTensor<float> regQScale;
    MicroAPI::RegTensor<half> regQScaleFP16;
    MicroAPI::RegTensor<float> regKScale[2];
    MicroAPI::RegTensor<half> regKScaleFP16[2];
    MicroAPI::RegTensor<float> regSum0[2];
    MicroAPI::RegTensor<float> regSum1[2];
    MicroAPI::MaskReg maskAllB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskAllB16 = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {MicroAPI::RegLayout::ZERO,
                                                                   MicroAPI::SatMode::NO_SAT,
                                                                   MicroAPI::MaskMergeMode::MERGING,
                                                                   RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {MicroAPI::RegLayout::ONE,
                                                                  MicroAPI::SatMode::NO_SAT,
                                                                  MicroAPI::MaskMergeMode::ZEROING,
                                                                  RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF16ToF32 = {MicroAPI::RegLayout::ZERO,
                                                              MicroAPI::SatMode::UNKNOWN,
                                                              MicroAPI::MaskMergeMode::ZEROING,
                                                              RoundMode::UNKNOWN};
    constexpr static MicroAPI::CastTrait castTraitInt32ToFP32 = {MicroAPI::RegLayout::UNKNOWN,
                                                                 MicroAPI::SatMode::NO_SAT,
                                                                 MicroAPI::MaskMergeMode::ZEROING,
                                                                 RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16 = {MicroAPI::RegLayout::ZERO,
                                                              MicroAPI::SatMode::NO_SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING,
                                                              RoundMode::CAST_RINT};
    MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(regWFP16, weight);
    MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(regQScaleFP16, qScale);
    MicroAPI::Cast<float, half, castTraitF16ToF32>(regW, regWFP16, maskAllB16);
    MicroAPI::Cast<float, half, castTraitF16ToF32>(regQScale, regQScaleFP16, maskAllB16);
    MicroAPI::Mul(regW, regW, regQScale, maskAllB32);
    MicroAPI::Cast<half, float, castTraitF32ToF16>(regWFP16Temp, regW, maskAllB32);
    MicroAPI::Cast<float, half, castTraitF16ToF32>(regW, regWFP16Temp, maskAllB16);
    liV2Vector1::DuplicateZero(regSum0, maskAllB32);
    liV2Vector1::DuplicateZero(regSum1, maskAllB32);

    LoadKScaleFP16(regKScaleFP16, regKScale, maskAllB16, kScale);
    // float mulsScalar = 1.0f / 1024;
    // unroll2
    for (uint16_t i = (uint16_t)(0); i < gSize; i += 2) {
        MicroAPI::LoadAlign<int32_t>(regQKInt32[0], qk + 128 * i);
        MicroAPI::LoadAlign<int32_t>(regQKInt32[1], qk + 128 * i + qkVLStride);
        MicroAPI::Cast<float, int32_t, castTraitInt32ToFP32>(regQK[0], regQKInt32[0], maskAllB32);
        MicroAPI::Cast<float, int32_t, castTraitInt32ToFP32>(regQK[1], regQKInt32[1], maskAllB32);

        CastFP32ToFP16ToFP32(regQK, regQKHalf, maskAllB32);

        liV2Vector1::BroadcastLane(regwBrc, regW, i);
        liV2Vector1::WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

        MicroAPI::LoadAlign<int32_t>(regQKInt32[0], qk + 128 * i + 128);
        MicroAPI::LoadAlign<int32_t>(regQKInt32[1], qk + 128 * i + 128 + qkVLStride);
        MicroAPI::Cast<float, int32_t, castTraitInt32ToFP32>(regQK[0], regQKInt32[0], maskAllB32);
        MicroAPI::Cast<float, int32_t, castTraitInt32ToFP32>(regQK[1], regQKInt32[1], maskAllB32);

        CastFP32ToFP16ToFP32(regQK, regQKHalf, maskAllB32);

        liV2Vector1::BroadcastLane(regwBrc, regW, i + 1);
        liV2Vector1::WeightedAccum(regSum1, regQK, regwBrc, maskAllB32);
    }

    MicroAPI::Add(regSum0[0], regSum0[0], regSum1[0], maskAllB32);
    MicroAPI::Add(regSum0[1], regSum0[1], regSum1[1], maskAllB32);

    MicroAPI::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
    MicroAPI::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);

    MicroAPI::RegTensor<bfloat16_t> regSumBF16;
    // interleave cast ==> regSum[1] high regSum[0] low
    MicroAPI::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum0[1], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum0[0], maskAllB32);

    MicroAPI::RegTensor<uint16_t> regOut;
    liV2Vector1::FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
    // normal store
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(out, regOut, maskAllB16);
}

__simd_vf__ void MulWeightAndReduceSumF32GSizeEvenVF(__ubuf__ uint16_t *out, __ubuf__ float *qk, uint32_t qkVLStride,
                                                     __ubuf__ float *weight, __ubuf__ float *kScale,
                                                     __ubuf__ float *qScale, uint16_t gSize)
{
    MicroAPI::RegTensor<float> regwBrc;
    MicroAPI::RegTensor<float> regQK[2];
    MicroAPI::RegTensor<float> regW;

    MicroAPI::RegTensor<float> regQScale;
    MicroAPI::RegTensor<float> regKScale[2];
    MicroAPI::RegTensor<float> regSum0[2];
    MicroAPI::RegTensor<float> regSum1[2];
    MicroAPI::MaskReg maskAllB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskAllB16 = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {
        MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    MicroAPI::LoadAlign<float>(regW, weight);
    MicroAPI::LoadAlign<float>(regQScale, qScale);
    MicroAPI::Mul(regW, regW, regQScale, maskAllB32);

    liV2Vector1::DuplicateZero(regSum0, maskAllB32);
    liV2Vector1::DuplicateZero(regSum1, maskAllB32);

    MicroAPI::LoadAlign<float>(regKScale[0], kScale);
    MicroAPI::LoadAlign<float>(regKScale[1], kScale + 64);

    // unroll2
    for (uint16_t i = (uint16_t)(0); i < gSize; i += 2) {
        MicroAPI::LoadAlign<float>(regQK[0], qk + 128 * i); // RowStride是128, 行都落在一个bank上
        MicroAPI::LoadAlign<float>(regQK[1], qk + 128 * i + qkVLStride);
        liV2Vector1::BroadcastLane(regwBrc, regW, i);
        liV2Vector1::WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

        MicroAPI::LoadAlign<float>(regQK[0], qk + 128 * i + 128);
        MicroAPI::LoadAlign<float>(regQK[1], qk + 128 * i + 128 + qkVLStride);
        liV2Vector1::BroadcastLane(regwBrc, regW, i + 1);
        liV2Vector1::WeightedAccum(regSum1, regQK, regwBrc, maskAllB32);
    }

    MicroAPI::Add(regSum0[0], regSum0[0], regSum1[0], maskAllB32);
    MicroAPI::Add(regSum0[1], regSum0[1], regSum1[1], maskAllB32);

    MicroAPI::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
    MicroAPI::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);

    MicroAPI::RegTensor<bfloat16_t> regSumBF16;
    // interleave cast ==> regSum[1] high regSum[0] low
    MicroAPI::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum0[1], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum0[0], maskAllB32);

    MicroAPI::RegTensor<uint16_t> regOut;
    liV2Vector1::FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
    // normal store
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(out, regOut, maskAllB16);
}

__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint16_t> &out_, // out    [S2Base]     [128   ]
                                             const LocalTensor<float> &qk_,     // q*k^t  [G, S2Base]  [64 128]
                                             const uint32_t qkVLStride,
                                             const LocalTensor<float> &weight_, // w      [G]          [64    ]
                                             const LocalTensor<float> &kScale_, // kScale [S2Base]     [128   ]
                                             const LocalTensor<float> &qScale_, // qScale [G]          [64    ]
                                             const int gSize)                   // G 64
{
    __ubuf__ uint16_t *out = (__ubuf__ uint16_t *)out_.GetPhyAddr();
    __ubuf__ float *weight = (__ubuf__ float *)weight_.GetPhyAddr();
    __ubuf__ float *qScale = (__ubuf__ float *)qScale_.GetPhyAddr();
    __ubuf__ float *kScale = (__ubuf__ float *)kScale_.GetPhyAddr();
    __ubuf__ float *qk = (__ubuf__ float *)qk_.GetPhyAddr();
    if (gSize % 2 == 0) {
        MulWeightAndReduceSumF32GSizeEvenVF(out, qk, qkVLStride, weight, kScale, qScale, (uint16_t)gSize);
    } else {
        MulWeightAndReduceSumF32GSizeOddVF(out, qk, qkVLStride, weight, kScale, qScale, (uint16_t)gSize);
    }
}

__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint16_t> &out_, // out    [S2Base]     [128   ]
                                             const LocalTensor<int32_t> &qk_,   // q*k^t  [G, S2Base]  [64 128]
                                             const uint32_t qkVLStride,
                                             const LocalTensor<half> &weight_, // w      [G]          [64    ]
                                             const LocalTensor<half> &kScale_, // kScale [S2Base]     [128   ]
                                             const LocalTensor<half> &qScale_, // qScale [G]          [64    ]
                                             const int gSize)                  // G 64
{
    __ubuf__ uint16_t *out = (__ubuf__ uint16_t *)out_.GetPhyAddr();
    __ubuf__ half *weight = (__ubuf__ half *)weight_.GetPhyAddr();
    __ubuf__ half *qScale = (__ubuf__ half *)qScale_.GetPhyAddr();
    __ubuf__ half *kScale = (__ubuf__ half *)kScale_.GetPhyAddr();
    __ubuf__ int32_t *qk = (__ubuf__ int32_t *)qk_.GetPhyAddr();
    if (gSize % 2 != 0) {
        MulWeightAndReduceSumInt32GSizeOddVF(out, qk, qkVLStride, weight, kScale, qScale, (uint16_t)gSize);
    } else {
        MulWeightAndReduceSumInt32GSizeEvenVF(out, qk, qkVLStride, weight, kScale, qScale, (uint16_t)gSize);
    }
}

// bfloat16_t in uint16 out
__simd_vf__ void MulWeightAndReduceSumB16VF(__ubuf__ uint16_t *out, __ubuf__ bfloat16_t *qk, __ubuf__ float *weight,
                                            __ubuf__ float *kScale, __ubuf__ float *qScale, uint16_t gSize)
{
    MicroAPI::RegTensor<float> regQK[4];
    MicroAPI::RegTensor<bfloat16_t> regQKB16[2];
    MicroAPI::RegTensor<float> regW;
    MicroAPI::RegTensor<float> regwBrc[2];
    MicroAPI::RegTensor<float> regQScale;
    MicroAPI::RegTensor<float> regKScale[2];
    MicroAPI::RegTensor<float> regSum[2];

    MicroAPI::MaskReg maskAllB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskAllB16 = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::RegTensor<bfloat16_t> regSumBF16;

    liV2Vector1::FloatSortConstCtx<bfloat16_t> bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    using CastTrait = MicroAPI::CastTrait;
    static constexpr CastTrait castTraitB162B32_EVEN = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr CastTrait castTraitB162B32_ODD = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    constexpr static CastTrait castTraitF32ToF16_EVEN = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                         MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static CastTrait castTraitF32ToF16_ODD = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    MicroAPI::LoadAlign<float>(regW, weight);
    MicroAPI::LoadAlign<float>(regQScale, qScale);
    MicroAPI::Mul(regW, regW, regQScale, maskAllB32);
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(weight, regW, maskAllB32);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    liV2Vector1::DuplicateZero(regSum, maskAllB32);

    // interleave load
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(regKScale[0], regKScale[1], kScale);

    // Duplicate + Gather方法劣化
    // Relu在cube随路做
    for (uint16_t i = (uint16_t)(0); i < gSize; i++) {
        // RowStride是256, 行都落在一个bank上
        MicroAPI::LoadAlign<bfloat16_t>(regQKB16[0], qk + 256 * i);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(regwBrc[0], weight + i);
        // interleave cast
        MicroAPI::Cast<float, bfloat16_t, castTraitB162B32_EVEN>(regQK[0], regQKB16[0], maskAllB16);
        MicroAPI::Cast<float, bfloat16_t, castTraitB162B32_ODD>(regQK[1], regQKB16[0], maskAllB16);
        MicroAPI::MulAddDst(regSum[0], regQK[0], regwBrc[0], maskAllB32);
        MicroAPI::MulAddDst(regSum[1], regQK[1], regwBrc[0], maskAllB32);
    }

    MicroAPI::Mul(regSum[0], regSum[0], regKScale[0], maskAllB32);
    MicroAPI::Mul(regSum[1], regSum[1], regKScale[1], maskAllB32);
    // interleave cast back
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum[1], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum[0], maskAllB32);

    MicroAPI::RegTensor<uint16_t> regOut;
    liV2Vector1::FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
    // norm load
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(out, regOut, maskAllB16);
}

__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint16_t> &out_,  // out    [S2Base]     [128   ]
                                             const LocalTensor<bfloat16_t> &qk_, // q*k^t  [G, S2Base]  [64 128]
                                             const uint32_t qkVLStride,          // unused for bfloat16
                                             const LocalTensor<float> &weight_,  // w      [G]          [64    ]
                                             const LocalTensor<float> &kScale_,  // kScale [S2Base]     [128   ]
                                             const LocalTensor<float> &qScale_,  // qScale [G]          [64    ]
                                             const int gSize)                    // G 64
{
    __ubuf__ uint16_t *out = (__ubuf__ uint16_t *)out_.GetPhyAddr();
    __ubuf__ float *weight = (__ubuf__ float *)weight_.GetPhyAddr();
    __ubuf__ float *qScale = (__ubuf__ float *)qScale_.GetPhyAddr();
    __ubuf__ bfloat16_t *qk = (__ubuf__ bfloat16_t *)qk_.GetPhyAddr();
    __ubuf__ float *kScale = (__ubuf__ float *)kScale_.GetPhyAddr();
    MulWeightAndReduceSumB16VF(out, qk, weight, kScale, qScale, (uint16_t)gSize);
}

// 计算S1=2
// float in uint16 out
__simd_vf__ void MulWeightAndReduceSum2F32VF(__ubuf__ uint16_t *out0, __ubuf__ uint16_t *out1, __ubuf__ float *qk0,
                                             __ubuf__ float *qk1, uint32_t qkVLStride, __ubuf__ float *weight0,
                                             __ubuf__ float *weight1, __ubuf__ float *weightTemp,
                                             __ubuf__ float *qScale0, __ubuf__ float *qScale1, __ubuf__ float *kScale0,
                                             uint16_t gSize)
{
    MicroAPI::RegTensor<float> regwBrc[2];
    MicroAPI::RegTensor<float> regQK0[2];
    MicroAPI::RegTensor<float> regQK1[2];
    MicroAPI::RegTensor<float> regW[2];

    MicroAPI::RegTensor<float> regQScale[2];
    MicroAPI::RegTensor<float> regKScale[2];
    MicroAPI::RegTensor<float> regSum0[2];
    MicroAPI::RegTensor<float> regSum1[2];
    MicroAPI::MaskReg maskAllB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskAllB16 = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {
        MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    MicroAPI::LoadAlign<float>(regW[0], weight0);
    MicroAPI::LoadAlign<float>(regW[1], weight1);
    MicroAPI::LoadAlign<float>(regQScale[0], qScale0);
    MicroAPI::LoadAlign<float>(regQScale[1], qScale1);
    MicroAPI::Mul(regW[0], regW[0], regQScale[0], maskAllB32);
    MicroAPI::Mul(regW[1], regW[1], regQScale[1], maskAllB32);
    // regW[0]与weight1混合使用
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(weightTemp, regW[1], maskAllB32);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    liV2Vector1::DuplicateZero(regSum0, maskAllB32);
    liV2Vector1::DuplicateZero(regSum1, maskAllB32);

    MicroAPI::LoadAlign<float>(regKScale[0], kScale0);
    MicroAPI::LoadAlign<float>(regKScale[1], kScale0 + 64);

    for (uint16_t i = (uint16_t)(0); i < gSize; i++) {
        MicroAPI::LoadAlign<float>(regQK0[0], qk0 + 128 * i);
        MicroAPI::LoadAlign<float>(regQK0[1], qk0 + 128 * i + qkVLStride);
        MicroAPI::LoadAlign<float>(regQK1[0], qk1 + 128 * i);
        MicroAPI::LoadAlign<float>(regQK1[1], qk1 + 128 * i + qkVLStride);
        // 混合使用对整体性能更好
        liV2Vector1::BroadcastLane(regwBrc[0], regW[0], i);
        // Weight无bank冲突，用LoadAlign来提取weight标量
        // 地址空间处理：原 BroadcastLane(ptr) 内联为 LoadAlign BRC，避免 __ubuf__ 传给 __local_mem__ 参数
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(regwBrc[1], weightTemp + i);
        MicroAPI::Relu(regQK0[0], regQK0[0], maskAllB32);
        MicroAPI::Relu(regQK0[1], regQK0[1], maskAllB32);
        MicroAPI::Relu(regQK1[0], regQK1[0], maskAllB32);
        MicroAPI::Relu(regQK1[1], regQK1[1], maskAllB32);
        MicroAPI::MulAddDst(regSum0[0], regQK0[0], regwBrc[0], maskAllB32);
        MicroAPI::MulAddDst(regSum0[1], regQK0[1], regwBrc[0], maskAllB32);
        MicroAPI::MulAddDst(regSum1[0], regQK1[0], regwBrc[1], maskAllB32);
        MicroAPI::MulAddDst(regSum1[1], regQK1[1], regwBrc[1], maskAllB32);
    }

    // Apply kScale scaling
    MicroAPI::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
    MicroAPI::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);
    MicroAPI::Mul(regSum1[0], regSum1[0], regKScale[0], maskAllB32);
    MicroAPI::Mul(regSum1[1], regSum1[1], regKScale[1], maskAllB32);

    // Convert to bfloat16 and store output channel
    MicroAPI::RegTensor<bfloat16_t> regSumBF16[2];
    MicroAPI::RegTensor<uint16_t> regOut[2];
    MicroAPI::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
    MicroAPI::DeInterleave(regSum1[0], regSum1[1], regSum1[0], regSum1[1]);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[0], regSum0[1], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[1], regSum1[1], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[0], regSum0[0], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[1], regSum1[0], maskAllB32);

    liV2Vector1::FloatX2ToSortableKey<bfloat16_t>(regOut[0], regOut[1], regSumBF16[0], regSumBF16[1], bf16Ctx,
                                                  maskAllB16);
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(out0, regOut[0], maskAllB16);
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(out1, regOut[1], maskAllB16);
}

// 计算S1=2
// int32 in uint16 out
__simd_vf__ void MulWeightAndReduceSum2Int32VF(__ubuf__ uint16_t *out0, __ubuf__ uint16_t *out1,
                                               __ubuf__ int32_t *qk0, __ubuf__ int32_t *qk1,
                                               uint32_t qkVLStride, __ubuf__ half *weight0,
                                               __ubuf__ half *weight1, __ubuf__ float *weightTemp,
                                               __ubuf__ half *qScale0, __ubuf__ half *qScale1,
                                               __ubuf__ half *kScale0, uint16_t gSize)
{
    MicroAPI::RegTensor<float> regwBrc[2];
    MicroAPI::RegTensor<float> regQK0[2];
    MicroAPI::RegTensor<float> regQK1[2];
    MicroAPI::RegTensor<half> regQK0Half[2];
    MicroAPI::RegTensor<half> regQK1Half[2];
    MicroAPI::RegTensor<int32_t> regQK0Int32[2];
    MicroAPI::RegTensor<int32_t> regQK1Int32[2];
    MicroAPI::RegTensor<float> regW[2];
    MicroAPI::RegTensor<half> regWFP16[2];
    MicroAPI::RegTensor<half> regWFP16Temp[2];
    MicroAPI::RegTensor<float> regQScale[2];
    MicroAPI::RegTensor<half> regQScaleFP16[2];
    MicroAPI::RegTensor<float> regKScale[2];
    MicroAPI::RegTensor<half> regKScaleFP16[2];
    MicroAPI::RegTensor<float> regSum0[2];
    MicroAPI::RegTensor<float> regSum1[2];
    MicroAPI::MaskReg maskAllB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskAllB16 = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {MicroAPI::RegLayout::ZERO,
                                                                   MicroAPI::SatMode::NO_SAT,
                                                                   MicroAPI::MaskMergeMode::MERGING,
                                                                   RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {MicroAPI::RegLayout::ONE,
                                                                  MicroAPI::SatMode::NO_SAT,
                                                                  MicroAPI::MaskMergeMode::ZEROING,
                                                                  RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF16ToF32 = {MicroAPI::RegLayout::ZERO,
                                                              MicroAPI::SatMode::UNKNOWN,
                                                              MicroAPI::MaskMergeMode::ZEROING,
                                                              RoundMode::UNKNOWN};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16 = {MicroAPI::RegLayout::ZERO,
                                                              MicroAPI::SatMode::NO_SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING,
                                                              RoundMode::CAST_RINT};
    constexpr static MicroAPI::CastTrait castTraitInt32ToFP32 = {MicroAPI::RegLayout::UNKNOWN,
                                                                 MicroAPI::SatMode::NO_SAT,
                                                                 MicroAPI::MaskMergeMode::ZEROING,
                                                                 RoundMode::CAST_ROUND};
    MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(regWFP16[0], weight0);
    MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(regWFP16[1], weight1);
    MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(regQScaleFP16[0], qScale0);
    MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(regQScaleFP16[1], qScale1);
    MicroAPI::Cast<float, half, castTraitF16ToF32>(regW[0], regWFP16[0], maskAllB16);
    MicroAPI::Cast<float, half, castTraitF16ToF32>(regW[1], regWFP16[1], maskAllB16);
    MicroAPI::Cast<float, half, castTraitF16ToF32>(regQScale[0], regQScaleFP16[0], maskAllB16);
    MicroAPI::Cast<float, half, castTraitF16ToF32>(regQScale[1], regQScaleFP16[1], maskAllB16);

    MicroAPI::Mul(regW[0], regW[0], regQScale[0], maskAllB32);
    MicroAPI::Mul(regW[1], regW[1], regQScale[1], maskAllB32);

    MicroAPI::Cast<half, float, castTraitF32ToF16>(regWFP16Temp[0], regW[0], maskAllB32);
    MicroAPI::Cast<float, half, castTraitF16ToF32>(regW[0], regWFP16Temp[0], maskAllB16);
    MicroAPI::Cast<half, float, castTraitF32ToF16>(regWFP16Temp[1], regW[1], maskAllB32);
    MicroAPI::Cast<float, half, castTraitF16ToF32>(regW[1], regWFP16Temp[1], maskAllB16);
    // regW[0]与weight1混合使用
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(weightTemp, regW[1], maskAllB32);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    liV2Vector1::DuplicateZero(regSum0, maskAllB32);
    liV2Vector1::DuplicateZero(regSum1, maskAllB32);

    LoadKScaleFP16(regKScaleFP16, regKScale, maskAllB16, kScale0);
    // float mulsScalar = 1.0 / 1024;

    for (uint16_t i = (uint16_t)(0); i < gSize; i++) {
        MicroAPI::LoadAlign<int32_t>(regQK0Int32[0], qk0 + 128 * i);
        MicroAPI::Cast<float, int32_t, castTraitInt32ToFP32>(regQK0[0], regQK0Int32[0], maskAllB32);
        MicroAPI::LoadAlign<int32_t>(regQK0Int32[1], qk0 + 128 * i + qkVLStride);
        MicroAPI::Cast<float, int32_t, castTraitInt32ToFP32>(regQK0[1], regQK0Int32[1], maskAllB32);
        MicroAPI::LoadAlign<int32_t>(regQK1Int32[0], qk1 + 128 * i);
        MicroAPI::Cast<float, int32_t, castTraitInt32ToFP32>(regQK1[0], regQK1Int32[0], maskAllB32);
        MicroAPI::LoadAlign<int32_t>(regQK1Int32[1], qk1 + 128 * i + qkVLStride);
        MicroAPI::Cast<float, int32_t, castTraitInt32ToFP32>(regQK1[1], regQK1Int32[1], maskAllB32);
        // 混合使用对整体性能更好
        liV2Vector1::BroadcastLane(regwBrc[0], regW[0], i);
        // Weight无bank冲突，用LoadAlign来提取weight标量
        // 地址空间处理：原 BroadcastLane(ptr) 内联为 LoadAlign BRC，避免 __ubuf__ 传给 __local_mem__ 参数
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(regwBrc[1], weightTemp + i);
        MicroAPI::Relu(regQK0[0], regQK0[0], maskAllB32);
        MicroAPI::Relu(regQK0[1], regQK0[1], maskAllB32);
        MicroAPI::Relu(regQK1[0], regQK1[0], maskAllB32);
        MicroAPI::Relu(regQK1[1], regQK1[1], maskAllB32);

        CastFP32ToFP16ToFP32(regQK0, regQK0Half, maskAllB32);
        CastFP32ToFP16ToFP32(regQK1, regQK1Half, maskAllB32);

        MicroAPI::MulAddDst(regSum0[0], regQK0[0], regwBrc[0], maskAllB32);
        MicroAPI::MulAddDst(regSum0[1], regQK0[1], regwBrc[0], maskAllB32);
        MicroAPI::MulAddDst(regSum1[0], regQK1[0], regwBrc[1], maskAllB32);
        MicroAPI::MulAddDst(regSum1[1], regQK1[1], regwBrc[1], maskAllB32);
    }

    // Apply kScale scaling
    MicroAPI::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
    MicroAPI::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);
    MicroAPI::Mul(regSum1[0], regSum1[0], regKScale[0], maskAllB32);
    MicroAPI::Mul(regSum1[1], regSum1[1], regKScale[1], maskAllB32);

    // Convert to bfloat16 and store output channel
    MicroAPI::RegTensor<bfloat16_t> regSumBF16[2];
    MicroAPI::RegTensor<uint16_t> regOut[2];
    MicroAPI::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
    MicroAPI::DeInterleave(regSum1[0], regSum1[1], regSum1[0], regSum1[1]);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[0], regSum0[1], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[1], regSum1[1], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[0], regSum0[0], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[1], regSum1[0], maskAllB32);

    liV2Vector1::FloatX2ToSortableKey<bfloat16_t>(regOut[0], regOut[1],
                                                  regSumBF16[0], regSumBF16[1], bf16Ctx, maskAllB16);
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(out0, regOut[0], maskAllB16);
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(out1, regOut[1], maskAllB16);
}

__aicore__ inline void MulWeightAndReduceSum2(const LocalTensor<uint16_t> &out_, // out    [2, S2Base]     [128   ]
                                              uint32_t outStride,
                                              const LocalTensor<float> &qk_, // q*k^t  [2, G, S2Base]  [64 128]
                                              uint32_t qkVLStride, uint32_t qkStride,
                                              const LocalTensor<float> &weight_, // w      [2, G]          [64    ]
                                              uint32_t weightStride, const LocalTensor<float> &weightTemp_,
                                              const LocalTensor<float> &kScale_, // kScale [S2Base]        [128   ]
                                              uint32_t kScaleStride,
                                              const LocalTensor<float> &qScale_, // qScale [2, G]          [64    ]
                                              uint32_t qScaleStride,
                                              const int gSize) // G 64
{
    __ubuf__ float *weight0 = (__ubuf__ float *)weight_.GetPhyAddr();
    __ubuf__ float *weightTemp = (__ubuf__ float *)weightTemp_.GetPhyAddr();
    __ubuf__ float *qScale0 = (__ubuf__ float *)qScale_.GetPhyAddr();
    __ubuf__ float *kScale0 = (__ubuf__ float *)kScale_.GetPhyAddr();
    __ubuf__ float *qk0 = (__ubuf__ float *)qk_.GetPhyAddr();
    __ubuf__ uint16_t *out0 = (__ubuf__ uint16_t *)out_.GetPhyAddr();

    __ubuf__ float *weight1 = weight0 + weightStride;
    __ubuf__ float *qScale1 = qScale0 + qScaleStride;
    __ubuf__ float *qk1 = qk0 + qkStride;
    // kScaleStride is zero
    __ubuf__ uint16_t *out1 = out0 + outStride;

    MulWeightAndReduceSum2F32VF(out0, out1, qk0, qk1, qkVLStride, weight0, weight1, weightTemp, qScale0, qScale1,
                                kScale0, (uint16_t)gSize);
}

__aicore__ inline void MulWeightAndReduceSum2(const LocalTensor<uint16_t> &out_, // out    [2, S2Base]     [128   ]
                                              uint32_t outStride,
                                              const LocalTensor<int32_t> &qk_, // q*k^t  [2, G, S2Base]  [64 128]
                                              uint32_t qkVLStride,
                                              uint32_t qkStride,
                                              const LocalTensor<half> &weight_, // w      [2, G]          [64    ]
                                              uint32_t weightStride,
                                              const LocalTensor<float> &weightTemp_,
                                              const LocalTensor<half> &kScale_, // kScale [S2Base]        [128   ]
                                              uint32_t kScaleStride,
                                              const LocalTensor<half> &qScale_, // qScale [2, G]          [64    ]
                                              uint32_t qScaleStride,
                                              const int gSize) // G 64
{
    __ubuf__ half *weight0 = (__ubuf__ half *)weight_.GetPhyAddr();
    __ubuf__ float *weightTemp = (__ubuf__ float *)weightTemp_.GetPhyAddr();
    __ubuf__ half *qScale0 = (__ubuf__ half *)qScale_.GetPhyAddr();
    __ubuf__ half *kScale0 = (__ubuf__ half *)kScale_.GetPhyAddr();
    __ubuf__ int32_t *qk0 = (__ubuf__ int32_t *)qk_.GetPhyAddr();
    __ubuf__ uint16_t *out0 = (__ubuf__ uint16_t *)out_.GetPhyAddr();

    __ubuf__ half *weight1 = weight0 + weightStride;
    __ubuf__ half *qScale1 = qScale0 + qScaleStride;
    __ubuf__ int32_t *qk1 = qk0 + qkStride;
    // kScaleStride is zero
    __ubuf__ uint16_t *out1 = out0 + outStride;

    MulWeightAndReduceSum2Int32VF(out0, out1, qk0, qk1, qkVLStride, weight0, weight1, weightTemp,
                                  qScale0, qScale1, kScale0, (uint16_t)gSize);
}

// 计算S1=2
// bfloat16 in uint16 out
__simd_vf__ void MulWeightAndReduceSum2B16VF(__ubuf__ uint16_t *out0, __ubuf__ uint16_t *out1,
                                             __ubuf__ bfloat16_t *qk0,
                                             __ubuf__ bfloat16_t *qk1, __ubuf__ float *weight0,
                                             __ubuf__ float *weight1,
                                             __ubuf__ float *weightTemp0, __ubuf__ float *weightTemp1,
                                             __ubuf__ float *qScale0, __ubuf__ float *qScale1,
                                             __ubuf__ float *kScale0, uint16_t gSize)
{
    MicroAPI::RegTensor<float> regwBrc[2];
    MicroAPI::RegTensor<float> regQK0[2];
    MicroAPI::RegTensor<float> regQK1[2];
    MicroAPI::RegTensor<float> regW[2];
    MicroAPI::RegTensor<bfloat16_t> regQKB16[2];

    MicroAPI::RegTensor<float> regQScale[2];
    MicroAPI::RegTensor<float> regKScale[2];
    MicroAPI::RegTensor<float> regSum0[2];
    MicroAPI::RegTensor<float> regSum1[2];
    MicroAPI::MaskReg maskAllB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskAllB16 = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    using CastTrait = MicroAPI::CastTrait;
    static constexpr CastTrait castTraitB162B32_EVEN = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr CastTrait castTraitB162B32_ODD = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {
        MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    MicroAPI::LoadAlign<float>(regW[0], weight0);
    MicroAPI::LoadAlign<float>(regW[1], weight1);
    MicroAPI::LoadAlign<float>(regQScale[0], qScale0);
    MicroAPI::LoadAlign<float>(regQScale[1], qScale1);
    MicroAPI::Mul(regW[0], regW[0], regQScale[0], maskAllB32);
    MicroAPI::Mul(regW[1], regW[1], regQScale[1], maskAllB32);
    // 读写依赖，寄存器可以保序
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(weightTemp0, regW[0], maskAllB32);
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(weightTemp1, regW[1], maskAllB32);
    liV2Vector1::DuplicateZero(regSum0, maskAllB32);
    liV2Vector1::DuplicateZero(regSum1, maskAllB32);

    // interleave load
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(regKScale[0], regKScale[1], kScale0);

    for (uint16_t i = (uint16_t)(0); i < gSize; i++) {
        // RowStride是256, 行都落在一个bank上
        MicroAPI::LoadAlign<bfloat16_t>(regQKB16[0], qk0 + 256 * i);
        // RowStride是256, 行都落在一个bank上
        MicroAPI::LoadAlign<bfloat16_t>(regQKB16[1], qk1 + 256 * i);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(regwBrc[0], weightTemp0 + i);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(regwBrc[1], weightTemp1 + i);
        // interleave cast
        MicroAPI::Cast<float, bfloat16_t, castTraitB162B32_EVEN>(regQK0[0], regQKB16[0], maskAllB32);
        MicroAPI::Cast<float, bfloat16_t, castTraitB162B32_ODD>(regQK0[1], regQKB16[0], maskAllB32);
        MicroAPI::Cast<float, bfloat16_t, castTraitB162B32_EVEN>(regQK1[0], regQKB16[1], maskAllB32);
        MicroAPI::Cast<float, bfloat16_t, castTraitB162B32_ODD>(regQK1[1], regQKB16[1], maskAllB32);
        MicroAPI::MulAddDst(regSum0[0], regQK0[0], regwBrc[0], maskAllB32);
        MicroAPI::MulAddDst(regSum0[1], regQK0[1], regwBrc[0], maskAllB32);
        MicroAPI::MulAddDst(regSum1[0], regQK1[0], regwBrc[1], maskAllB32);
        MicroAPI::MulAddDst(regSum1[1], regQK1[1], regwBrc[1], maskAllB32);
    }

    // Apply kScale scaling
    MicroAPI::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
    MicroAPI::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);
    MicroAPI::Mul(regSum1[0], regSum1[0], regKScale[0], maskAllB32);
    MicroAPI::Mul(regSum1[1], regSum1[1], regKScale[1], maskAllB32);

    // Convert to bfloat16 and store output channel
    MicroAPI::RegTensor<bfloat16_t> regSumBF16[2];
    MicroAPI::RegTensor<uint16_t> regOut[2];
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[0], regSum0[1], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[1], regSum1[1], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[0], regSum0[0], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[1], regSum1[0], maskAllB32);

    liV2Vector1::FloatX2ToSortableKey<bfloat16_t>(regOut[0], regOut[1], regSumBF16[0], regSumBF16[1], bf16Ctx,
                                                  maskAllB16);
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(out0, regOut[0], maskAllB16);
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(out1, regOut[1], maskAllB16);
}

__aicore__ inline void MulWeightAndReduceSum2(const LocalTensor<uint16_t> &out_, // out    [2, S2Base]     [128   ]
                                              uint32_t outStride,
                                              const LocalTensor<bfloat16_t> &qk_, // q*k^t  [2, G, S2Base]  [64 128]
                                              uint32_t qkVLStride,
                                              uint32_t qkStride,                 // gSize * 256
                                              const LocalTensor<float> &weight_, // w      [2, G]          [64    ]
                                              uint32_t weightStride, const LocalTensor<float> &weightTemp_,
                                              const LocalTensor<float> &kScale_, // kScale [S2Base]        [128   ]
                                              uint32_t kScaleStride,
                                              const LocalTensor<float> &qScale_, // qScale [2, G]          [64    ]
                                              uint32_t qScaleStride,
                                              const int gSize) // G 64
{
    __ubuf__ float *weight0 = (__ubuf__ float *)weight_.GetPhyAddr();
    __ubuf__ float *weightTemp0 = (__ubuf__ float *)weightTemp_.GetPhyAddr();
    __ubuf__ float *qScale0 = (__ubuf__ float *)qScale_.GetPhyAddr();
    __ubuf__ float *kScale0 = (__ubuf__ float *)kScale_.GetPhyAddr();
    __ubuf__ bfloat16_t *qk0 = (__ubuf__ bfloat16_t *)qk_.GetPhyAddr();
    __ubuf__ uint16_t *out0 = (__ubuf__ uint16_t *)out_.GetPhyAddr();

    __ubuf__ float *weightTemp1 = weightTemp0 + weightStride;
    __ubuf__ float *weight1 = weight0 + weightStride;
    __ubuf__ float *qScale1 = qScale0 + qScaleStride;
    __ubuf__ bfloat16_t *qk1 = qk0 + qkStride;
    // kScaleStride is zero
    __ubuf__ uint16_t *out1 = out0 + outStride;

    MulWeightAndReduceSum2B16VF(out0, out1, qk0, qk1, weight0, weight1, weightTemp0, weightTemp1, qScale0, qScale1,
                                kScale0, (uint16_t)gSize);
}

template <typename QK_T, typename SCORE_T, typename WEIGHT_T, typename SCALE_T>
__aicore__ inline void BatchMulWeightAndReduceSum(const LocalTensor<SCORE_T> &out_, // out    [S2Base]     [128   ]
                                                  uint32_t outStride,
                                                  const LocalTensor<QK_T> &qk_, // q*k^t  [G, S2Base]  [64 128]
                                                  uint32_t qkVLStride, uint32_t qkStride,
                                                  const LocalTensor<WEIGHT_T> &weight_, // w      [G]      [64    ]
                                                  uint32_t weightStride, const LocalTensor<float> &weightTemp_,
                                                  const LocalTensor<SCALE_T> &kScale_, // kScale [S2Base]    [128   ]
                                                  uint32_t kScaleStride,
                                                  const LocalTensor<SCALE_T> &qScale_, // qScale [G]         [64    ]
                                                  uint32_t qScaleStride,
                                                  const int gSize, // G 64
                                                  const int batch)
{
    // 暂只支持这两种情况, 后续改成循环
    if (batch != 2 && batch != 1) {
        return;
    }
    if (batch == 2) {
        MulWeightAndReduceSum2(out_, outStride, qk_, qkVLStride, qkStride, weight_, weightStride, weightTemp_, kScale_,
                               kScaleStride, qScale_, qScaleStride, gSize);
    } else {
        MulWeightAndReduceSum(out_, qk_, qkVLStride, weight_, kScale_, qScale_, gSize);
    }
}

// per_tensor与MX共用的weight加权归约实现，WITH_SCALE控制是否额外应用scalar scale
// float in uint16 out
template <bool WITH_SCALE>
__simd_vf__ void MulWeightAndReduceSumOptionalScaleGSizeEvenVF(__ubuf__ uint16_t *out, __ubuf__ float *qk,
                                                               uint32_t qkVLStride, __ubuf__ float *weight,
                                                               float kScaleValue, float qScaleValue, uint16_t gSize)
{
    if constexpr (!WITH_SCALE) {
        (void)kScaleValue;
        (void)qScaleValue;
    }

    MicroAPI::RegTensor<float> regwBrc;
    MicroAPI::RegTensor<float> regQK[2];
    MicroAPI::RegTensor<float> regW;
    MicroAPI::RegTensor<float> regSum0[2];
    MicroAPI::RegTensor<float> regSum1[2];
    MicroAPI::MaskReg maskAllB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskAllB16 = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {
        MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    MicroAPI::LoadAlign<float>(regW, weight);
    if constexpr (WITH_SCALE) {
        MicroAPI::Muls(regW, regW, qScaleValue, maskAllB32);
    }

    liV2Vector1::DuplicateZero(regSum0, maskAllB32);
    liV2Vector1::DuplicateZero(regSum1, maskAllB32);

    // unroll2
    for (uint16_t i = (uint16_t)(0); i < gSize; i += 2) {
        MicroAPI::LoadAlign<float>(regQK[0], qk + 128 * i); // RowStride是128, 行都落在一个bank上
        MicroAPI::LoadAlign<float>(regQK[1], qk + 128 * i + qkVLStride);
        liV2Vector1::BroadcastLane(regwBrc, regW, i);
        liV2Vector1::WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

        MicroAPI::LoadAlign<float>(regQK[0], qk + 128 * i + 128);
        MicroAPI::LoadAlign<float>(regQK[1], qk + 128 * i + 128 + qkVLStride);
        liV2Vector1::BroadcastLane(regwBrc, regW, i + 1);
        liV2Vector1::WeightedAccum(regSum1, regQK, regwBrc, maskAllB32);
    }

    MicroAPI::Add(regSum0[0], regSum0[0], regSum1[0], maskAllB32);
    MicroAPI::Add(regSum0[1], regSum0[1], regSum1[1], maskAllB32);

    if constexpr (WITH_SCALE) {
        MicroAPI::Muls(regSum0[0], regSum0[0], kScaleValue, maskAllB32);
        MicroAPI::Muls(regSum0[1], regSum0[1], kScaleValue, maskAllB32);
    }

    MicroAPI::RegTensor<bfloat16_t> regSumBF16;
    // interleave cast ==> regSum[1] high regSum[0] low
    MicroAPI::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum0[1], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum0[0], maskAllB32);

    MicroAPI::RegTensor<uint16_t> regOut;
    liV2Vector1::FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
    // normal store
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(out, regOut, maskAllB16);
}

template <bool WITH_SCALE>
__simd_vf__ void MulWeightAndReduceSumOptionalScaleGSizeOddVF(__ubuf__ uint16_t *out, __ubuf__ float *qk,
                                                              uint32_t qkVLStride, __ubuf__ float *weight,
                                                              float kScaleValue, float qScaleValue, uint16_t gSize)
{
    if constexpr (!WITH_SCALE) {
        (void)kScaleValue;
        (void)qScaleValue;
    }

    MicroAPI::RegTensor<float> regwBrc;
    MicroAPI::RegTensor<float> regQK[2];
    MicroAPI::RegTensor<float> regW;
    MicroAPI::RegTensor<float> regSum0[2];
    MicroAPI::RegTensor<float> regSum1[2];
    MicroAPI::MaskReg maskAllB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskAllB16 = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {
        MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    MicroAPI::LoadAlign<float>(regW, weight);
    if constexpr (WITH_SCALE) {
        MicroAPI::Muls(regW, regW, qScaleValue, maskAllB32);
    }

    liV2Vector1::DuplicateZero(regSum0, maskAllB32);
    liV2Vector1::DuplicateZero(regSum1, maskAllB32);

    // unroll2
    for (uint16_t i = (uint16_t)(0); (uint16_t)(i + 1) < gSize; i += 2) {
        MicroAPI::LoadAlign<float>(regQK[0], qk + 128 * i); // RowStride是128, 行都落在一个bank上
        MicroAPI::LoadAlign<float>(regQK[1], qk + 128 * i + qkVLStride);
        liV2Vector1::BroadcastLane(regwBrc, regW, i);
        liV2Vector1::WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

        MicroAPI::LoadAlign<float>(regQK[0], qk + 128 * i + 128);
        MicroAPI::LoadAlign<float>(regQK[1], qk + 128 * i + 128 + qkVLStride);
        liV2Vector1::BroadcastLane(regwBrc, regW, i + 1);
        liV2Vector1::WeightedAccum(regSum1, regQK, regwBrc, maskAllB32);
    }

    MicroAPI::LoadAlign<float>(regQK[0], qk + 128 * (gSize - 1)); // RowStride是128, 行都落在一个bank上
    MicroAPI::LoadAlign<float>(regQK[1], qk + 128 * (gSize - 1) + qkVLStride);
    liV2Vector1::BroadcastLane(regwBrc, regW, gSize - 1);
    liV2Vector1::WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

    MicroAPI::Add(regSum0[0], regSum0[0], regSum1[0], maskAllB32);
    MicroAPI::Add(regSum0[1], regSum0[1], regSum1[1], maskAllB32);

    if constexpr (WITH_SCALE) {
        MicroAPI::Muls(regSum0[0], regSum0[0], kScaleValue, maskAllB32);
        MicroAPI::Muls(regSum0[1], regSum0[1], kScaleValue, maskAllB32);
    }

    MicroAPI::RegTensor<bfloat16_t> regSumBF16;
    // interleave cast ==> regSum[1] high regSum[0] low
    MicroAPI::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum0[1], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum0[0], maskAllB32);

    MicroAPI::RegTensor<uint16_t> regOut;
    liV2Vector1::FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
    // normal store
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(out, regOut, maskAllB16);
}

template <bool WITH_SCALE>
__aicore__ inline void MulWeightAndReduceSumOptionalScaleImpl(
    const LocalTensor<uint16_t> &out_, // out    [S2Base]     [128   ]
    const LocalTensor<float> &qk_,     // q*k^t  [G, S2Base]  [64 128]
    const uint32_t qkVLStride,
    const LocalTensor<float> &weight_, // w      [G]          [64    ]
    const float kScaleValue,           // kScale scalar
    const float qScaleValue,           // qScale scalar
    const int gSize)                   // G 64
{
    __ubuf__ uint16_t *out = (__ubuf__ uint16_t *)out_.GetPhyAddr();
    __ubuf__ float *weight = (__ubuf__ float *)weight_.GetPhyAddr();
    __ubuf__ float *qk = (__ubuf__ float *)qk_.GetPhyAddr();
    if (gSize % 2 == 0) {
        MulWeightAndReduceSumOptionalScaleGSizeEvenVF<WITH_SCALE>(out, qk, qkVLStride, weight, kScaleValue, qScaleValue,
                                                                  (uint16_t)gSize);
    } else {
        MulWeightAndReduceSumOptionalScaleGSizeOddVF<WITH_SCALE>(out, qk, qkVLStride, weight, kScaleValue, qScaleValue,
                                                                 (uint16_t)gSize);
    }
}

__aicore__ inline void MulWeightAndReduceSumPerTensor(const LocalTensor<uint16_t> &out_, // out    [S2Base]     [128   ]
                                                      const LocalTensor<float> &qk_,     // q*k^t  [G, S2Base]  [64 128]
                                                      const uint32_t qkVLStride,
                                                      const LocalTensor<float> &weight_, // w      [G]          [64    ]
                                                      const float kScaleValue,           // kScale scalar
                                                      const float qScaleValue,           // qScale scalar
                                                      const int gSize)                   // G 64
{
    MulWeightAndReduceSumOptionalScaleImpl<true>(out_, qk_, qkVLStride, weight_, kScaleValue, qScaleValue, gSize);
}

__aicore__ inline void MulWeightAndReduceSumMX(const LocalTensor<uint16_t> &out_, // out    [S2Base]     [128   ]
                                               const LocalTensor<float> &qk_,     // q*k^t  [G, S2Base]  [64 128]
                                               const uint32_t qkVLStride,
                                               const LocalTensor<float> &weight_, // w      [G]          [64    ]
                                               const int gSize)                   // G 64
{
    MulWeightAndReduceSumOptionalScaleImpl<false>(out_, qk_, qkVLStride, weight_, 1.0f, 1.0f, gSize);
}

// 计算S1=2
// float in uint16 out
template <bool WITH_SCALE>
__simd_vf__ void MulWeightAndReduceSumOptionalScale2VF(__ubuf__ uint16_t *out0, __ubuf__ uint16_t *out1,
                                                       __ubuf__ float *qk0, __ubuf__ float *qk1, uint32_t qkVLStride,
                                                       __ubuf__ float *weight0, __ubuf__ float *weight1,
                                                       __ubuf__ float *weightTemp, float kScaleValue, float qScaleValue,
                                                       uint16_t gSize)
{
    if constexpr (!WITH_SCALE) {
        (void)kScaleValue;
        (void)qScaleValue;
    }

    MicroAPI::RegTensor<float> regwBrc[2];
    MicroAPI::RegTensor<float> regQK0[2];
    MicroAPI::RegTensor<float> regQK1[2];
    MicroAPI::RegTensor<float> regW[2];

    MicroAPI::RegTensor<float> regSum0[2];
    MicroAPI::RegTensor<float> regSum1[2];
    MicroAPI::MaskReg maskAllB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskAllB16 = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {
        MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    MicroAPI::LoadAlign<float>(regW[0], weight0);
    MicroAPI::LoadAlign<float>(regW[1], weight1);
    if constexpr (WITH_SCALE) {
        MicroAPI::Muls(regW[0], regW[0], qScaleValue, maskAllB32);
        MicroAPI::Muls(regW[1], regW[1], qScaleValue, maskAllB32);
    }
    // regW[0]与weight1混合使用
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(weightTemp, regW[1], maskAllB32);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    liV2Vector1::DuplicateZero(regSum0, maskAllB32);
    liV2Vector1::DuplicateZero(regSum1, maskAllB32);

    for (uint16_t i = (uint16_t)(0); i < gSize; i++) {
        MicroAPI::LoadAlign<float>(regQK0[0], qk0 + 128 * i);
        MicroAPI::LoadAlign<float>(regQK0[1], qk0 + 128 * i + qkVLStride);
        MicroAPI::LoadAlign<float>(regQK1[0], qk1 + 128 * i);
        MicroAPI::LoadAlign<float>(regQK1[1], qk1 + 128 * i + qkVLStride);
        // 混合使用对整体性能更好
        liV2Vector1::BroadcastLane(regwBrc[0], regW[0], i);
        // Weight无bank冲突，用LoadAlign来提取weight标量
        // 地址空间处理：原 BroadcastLane(ptr) 内联为 LoadAlign BRC，避免 __ubuf__ 传给 __local_mem__ 参数
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(regwBrc[1], weightTemp + i);
        MicroAPI::Relu(regQK0[0], regQK0[0], maskAllB32);
        MicroAPI::Relu(regQK0[1], regQK0[1], maskAllB32);
        MicroAPI::Relu(regQK1[0], regQK1[0], maskAllB32);
        MicroAPI::Relu(regQK1[1], regQK1[1], maskAllB32);
        MicroAPI::MulAddDst(regSum0[0], regQK0[0], regwBrc[0], maskAllB32);
        MicroAPI::MulAddDst(regSum0[1], regQK0[1], regwBrc[0], maskAllB32);
        MicroAPI::MulAddDst(regSum1[0], regQK1[0], regwBrc[1], maskAllB32);
        MicroAPI::MulAddDst(regSum1[1], regQK1[1], regwBrc[1], maskAllB32);
    }

    if constexpr (WITH_SCALE) {
        MicroAPI::Muls(regSum0[0], regSum0[0], kScaleValue, maskAllB32);
        MicroAPI::Muls(regSum0[1], regSum0[1], kScaleValue, maskAllB32);
        MicroAPI::Muls(regSum1[0], regSum1[0], kScaleValue, maskAllB32);
        MicroAPI::Muls(regSum1[1], regSum1[1], kScaleValue, maskAllB32);
    }

    // Convert to bfloat16 and store output channel
    MicroAPI::RegTensor<bfloat16_t> regSumBF16[2];
    MicroAPI::RegTensor<uint16_t> regOut[2];
    MicroAPI::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
    MicroAPI::DeInterleave(regSum1[0], regSum1[1], regSum1[0], regSum1[1]);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[0], regSum0[1], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[1], regSum1[1], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[0], regSum0[0], maskAllB32);
    MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[1], regSum1[0], maskAllB32);

    liV2Vector1::FloatX2ToSortableKey<bfloat16_t>(regOut[0], regOut[1], regSumBF16[0], regSumBF16[1], bf16Ctx,
                                                  maskAllB16);
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(out0, regOut[0], maskAllB16);
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(out1, regOut[1], maskAllB16);
}

template <bool WITH_SCALE>
__aicore__ inline void MulWeightAndReduceSumOptionalScale2Impl(
    const LocalTensor<uint16_t> &out_, // out    [2, S2Base]     [128   ]
    uint32_t outStride,
    const LocalTensor<float> &qk_, // q*k^t  [2, G, S2Base]  [64 128]
    uint32_t qkVLStride, uint32_t qkStride,
    const LocalTensor<float> &weight_, // w      [2, G]          [64    ]
    uint32_t weightStride, const LocalTensor<float> &weightTemp_,
    const float kScaleValue, // kScale scalar
    const float qScaleValue, // qScale scalar for batch 0和1
    const int gSize)         // G 64
{
    __ubuf__ float *weight0 = (__ubuf__ float *)weight_.GetPhyAddr();
    __ubuf__ float *weightTemp = (__ubuf__ float *)weightTemp_.GetPhyAddr();
    __ubuf__ float *qk0 = (__ubuf__ float *)qk_.GetPhyAddr();
    __ubuf__ uint16_t *out0 = (__ubuf__ uint16_t *)out_.GetPhyAddr();

    __ubuf__ float *weight1 = weight0 + weightStride;
    __ubuf__ float *qk1 = qk0 + qkStride;
    __ubuf__ uint16_t *out1 = out0 + outStride;

    MulWeightAndReduceSumOptionalScale2VF<WITH_SCALE>(out0, out1, qk0, qk1, qkVLStride, weight0, weight1, weightTemp,
                                                      kScaleValue, qScaleValue, (uint16_t)gSize);
}

__aicore__ inline void MulWeightAndReduceSumPerTensor2(
    const LocalTensor<uint16_t> &out_, // out    [2, S2Base]     [128   ]
    uint32_t outStride,
    const LocalTensor<float> &qk_, // q*k^t  [2, G, S2Base]  [64 128]
    uint32_t qkVLStride, uint32_t qkStride,
    const LocalTensor<float> &weight_, // w      [2, G]          [64    ]
    uint32_t weightStride, const LocalTensor<float> &weightTemp_,
    const float kScaleValue, // kScale scalar
    const float qScaleValue, // qScale scalar for batch 0和1
    const int gSize)         // G 64
{
    MulWeightAndReduceSumOptionalScale2Impl<true>(out_, outStride, qk_, qkVLStride, qkStride, weight_, weightStride,
                                                  weightTemp_, kScaleValue, qScaleValue, gSize);
}

__aicore__ inline void MulWeightAndReduceSumMX2(const LocalTensor<uint16_t> &out_, // out    [2, S2Base]     [128   ]
                                                uint32_t outStride,
                                                const LocalTensor<float> &qk_, // q*k^t  [2, G, S2Base]  [64 128]
                                                uint32_t qkVLStride, uint32_t qkStride,
                                                const LocalTensor<float> &weight_, // w      [2, G]          [64    ]
                                                uint32_t weightStride, const LocalTensor<float> &weightTemp_,
                                                const int gSize) // G 64
{
    MulWeightAndReduceSumOptionalScale2Impl<false>(out_, outStride, qk_, qkVLStride, qkStride, weight_, weightStride,
                                                   weightTemp_, 1.0f, 1.0f, gSize);
}

__simd_callee__ inline void CastWeightToBf16(AscendC::MicroAPI::RegTensor<bfloat16_t> &dst, __ubuf__ float *src,
                                             AscendC::MicroAPI::MaskReg &maskAllB32)
{
    using CastTrait = AscendC::MicroAPI::CastTrait;
    static constexpr CastTrait castTraitF32ToBf16 = {AscendC::MicroAPI::RegLayout::ZERO,
                                                     AscendC::MicroAPI::SatMode::NO_SAT,
                                                     AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    AscendC::MicroAPI::RegTensor<float> regWeightF32;
    AscendC::MicroAPI::LoadAlign<float>(regWeightF32, src);
    AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitF32ToBf16>(dst, regWeightF32, maskAllB32);
    // Cast结果按B32 lane落位，在目标寄存器内压紧为连续BF16，供BroadcastLane按元素索引。
    AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
        (AscendC::MicroAPI::RegTensor<uint16_t> &)dst, (AscendC::MicroAPI::RegTensor<uint32_t> &)dst);
}

__simd_vf__ void MulWeightAndReduceSumMXFP4VF(__ubuf__ uint16_t *out, __ubuf__ bfloat16_t *qk, __ubuf__ float *weight,
                                              uint16_t gSize)
{
    constexpr uint32_t BF16_QK_ROW_STRIDE = UB_BANK_DEPTH_STRIDE / sizeof(bfloat16_t);
    AscendC::MicroAPI::RegTensor<bfloat16_t> regQK;
    AscendC::MicroAPI::RegTensor<bfloat16_t> regWeight;
    AscendC::MicroAPI::RegTensor<bfloat16_t> regWeightBrc;
    AscendC::MicroAPI::RegTensor<bfloat16_t> regSum;
    AscendC::MicroAPI::MaskReg maskAllB16 =
        AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg maskAllB32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    CastWeightToBf16(regWeight, weight, maskAllB32);
    AscendC::MicroAPI::Duplicate(regSum, bfloat16_t(0.0f), maskAllB16);

    for (uint16_t i = 0; i < gSize; i++) {
        AscendC::MicroAPI::LoadAlign<bfloat16_t>(regQK, qk + BF16_QK_ROW_STRIDE * i);
        liV2Vector1::BroadcastLane(regWeightBrc, regWeight, i);
        AscendC::MicroAPI::MulAddDst(regSum, regQK, regWeightBrc, maskAllB16);
    }

    AscendC::MicroAPI::RegTensor<uint16_t> regOut;
    liV2Vector1::FloatToSortableKey<bfloat16_t>(regOut, regSum, bf16Ctx, maskAllB16);
    AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out, regOut, maskAllB16);
}

__aicore__ inline void MulWeightAndReduceSumMXFP4(const LocalTensor<uint16_t> &out_, const LocalTensor<bfloat16_t> &qk_,
                                                  const uint32_t qkVLStride, const LocalTensor<float> &weight_,
                                                  const int gSize)
{
    (void)qkVLStride;
    __ubuf__ uint16_t *out = (__ubuf__ uint16_t *)out_.GetPhyAddr();
    __ubuf__ bfloat16_t *qk = (__ubuf__ bfloat16_t *)qk_.GetPhyAddr();
    __ubuf__ float *weight = (__ubuf__ float *)weight_.GetPhyAddr();
    MulWeightAndReduceSumMXFP4VF(out, qk, weight, static_cast<uint16_t>(gSize));
}

__simd_vf__ void MulWeightAndReduceSumMXFP4TwoRowsVF(__ubuf__ uint16_t *out0, __ubuf__ uint16_t *out1,
                                                     __ubuf__ bfloat16_t *qk0, __ubuf__ bfloat16_t *qk1,
                                                     __ubuf__ float *weight0, __ubuf__ float *weight1, uint16_t gSize)
{
    constexpr uint32_t BF16_QK_ROW_STRIDE = UB_BANK_DEPTH_STRIDE / sizeof(bfloat16_t);
    AscendC::MicroAPI::RegTensor<bfloat16_t> regQK0;
    AscendC::MicroAPI::RegTensor<bfloat16_t> regQK1;
    AscendC::MicroAPI::RegTensor<bfloat16_t> regWeight[2];
    AscendC::MicroAPI::RegTensor<bfloat16_t> regWeightBrc[2];
    AscendC::MicroAPI::RegTensor<bfloat16_t> regSum[2];
    AscendC::MicroAPI::MaskReg maskAllB16 =
        AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg maskAllB32 =
        AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    CastWeightToBf16(regWeight[0], weight0, maskAllB32);
    CastWeightToBf16(regWeight[1], weight1, maskAllB32);
    AscendC::MicroAPI::Duplicate(regSum[0], bfloat16_t(0.0f), maskAllB16);
    AscendC::MicroAPI::Duplicate(regSum[1], bfloat16_t(0.0f), maskAllB16);

    for (uint16_t i = 0; i < gSize; i++) {
        AscendC::MicroAPI::LoadAlign<bfloat16_t>(regQK0, qk0 + BF16_QK_ROW_STRIDE * i);
        AscendC::MicroAPI::LoadAlign<bfloat16_t>(regQK1, qk1 + BF16_QK_ROW_STRIDE * i);
        liV2Vector1::BroadcastLane(regWeightBrc[0], regWeight[0], i);
        liV2Vector1::BroadcastLane(regWeightBrc[1], regWeight[1], i);
        AscendC::MicroAPI::MulAddDst(regSum[0], regQK0, regWeightBrc[0], maskAllB16);
        AscendC::MicroAPI::MulAddDst(regSum[1], regQK1, regWeightBrc[1], maskAllB16);
    }

    AscendC::MicroAPI::RegTensor<uint16_t> regOut[2];
    liV2Vector1::FloatX2ToSortableKey<bfloat16_t>(regOut[0], regOut[1], regSum[0], regSum[1], bf16Ctx, maskAllB16);
    AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out0, regOut[0], maskAllB16);
    AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out1, regOut[1], maskAllB16);
}

__aicore__ inline void MulWeightAndReduceSumMXFP4TwoRows(const LocalTensor<uint16_t> &out_, uint32_t outStride,
                                                         const LocalTensor<bfloat16_t> &qk_, uint32_t qkVLStride,
                                                         uint32_t qkStride, const LocalTensor<float> &weight_,
                                                         uint32_t weightStride, const int gSize)
{
    (void)qkVLStride;
    __ubuf__ uint16_t *out0 = (__ubuf__ uint16_t *)out_.GetPhyAddr();
    __ubuf__ uint16_t *out1 = out0 + outStride;
    __ubuf__ bfloat16_t *qk0 = (__ubuf__ bfloat16_t *)qk_.GetPhyAddr();
    __ubuf__ bfloat16_t *qk1 = qk0 + qkStride;
    __ubuf__ float *weight0 = (__ubuf__ float *)weight_.GetPhyAddr();
    __ubuf__ float *weight1 = weight0 + weightStride;
    MulWeightAndReduceSumMXFP4TwoRowsVF(out0, out1, qk0, qk1, weight0, weight1, static_cast<uint16_t>(gSize));
}

template <typename QK_T, typename SCORE_T, typename WEIGHT_T>
__aicore__ inline void BatchMulWeightAndReduceSumMXFP4(const LocalTensor<SCORE_T> &out_, uint32_t outStride,
                                                       const LocalTensor<QK_T> &qk_, uint32_t qkVLStride,
                                                       uint32_t qkStride, const LocalTensor<WEIGHT_T> &weight_,
                                                       uint32_t weightStride, const int gSize, const int batch)
{
    static_assert(std::is_same_v<QK_T, bfloat16_t>);
    static_assert(std::is_same_v<SCORE_T, uint16_t>);
    static_assert(std::is_same_v<WEIGHT_T, float>);
    if (batch == 2) {
        MulWeightAndReduceSumMXFP4TwoRows(out_, outStride, qk_, qkVLStride, qkStride, weight_, weightStride, gSize);
    } else if (batch == 1) {
        MulWeightAndReduceSumMXFP4(out_, qk_, qkVLStride, weight_, gSize);
    }
}
template <typename QK_T, typename SCORE_T, typename WEIGHT_T>
__aicore__ inline void BatchMulWeightAndReduceSumMX(const LocalTensor<SCORE_T> &out_,
                                                    uint32_t outStride,           // out    [S2Base]     [128   ]
                                                    const LocalTensor<QK_T> &qk_, // q*k^t  [G, S2Base]  [64 128]
                                                    uint32_t qkVLStride, uint32_t qkStride,
                                                    const LocalTensor<WEIGHT_T> &weight_, // w      [G]      [64    ]
                                                    uint32_t weightStride, const LocalTensor<float> &weightTemp_,
                                                    const int gSize, const int batch)
{
    // 暂只支持这两种情况, 后续改成循环
    if (batch != 2 && batch != 1) {
        return;
    }
    if (batch == 2) {
        MulWeightAndReduceSumMX2(out_, outStride, qk_, qkVLStride, qkStride,
                                 weight_, weightStride, weightTemp_, gSize);
    } else {
        MulWeightAndReduceSumMX(out_, qk_, qkVLStride, weight_, gSize);
    }
}

template <typename QK_T, typename SCORE_T, typename WEIGHT_T>
__aicore__ inline void BatchMulWeightAndReduceSumPerTensor(
    const LocalTensor<SCORE_T> &out_,
    uint32_t outStride,           // out    [S2Base]     [128   ]
    const LocalTensor<QK_T> &qk_, // q*k^t  [G, S2Base]  [64 128]
    uint32_t qkVLStride, uint32_t qkStride,
    const LocalTensor<WEIGHT_T> &weight_, // w  [G]          [64    ]
    uint32_t weightStride, const LocalTensor<float> &weightTemp_, const float kScaleValue, const float qScaleValue,
    const int gSize, const int batch)
{
    // 暂只支持这两种情况, 后续改成循环
    if (batch != 2 && batch != 1) {
        return;
    }
    if (batch == 2) {
        MulWeightAndReduceSumPerTensor2(out_, outStride, qk_, qkVLStride, qkStride, weight_, weightStride, weightTemp_,
                                        kScaleValue, qScaleValue, gSize);
    } else {
        MulWeightAndReduceSumPerTensor(out_, qk_, qkVLStride, weight_, kScaleValue, qScaleValue, gSize);
    }
}

} // namespace vector1

#endif // QUANT_LIGHTNING_INDEXER_V2_VECTOR1_H
