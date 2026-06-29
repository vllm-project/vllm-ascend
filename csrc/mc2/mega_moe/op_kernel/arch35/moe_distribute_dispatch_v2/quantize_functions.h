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
 * \file quantize_functions.h
 * \brief
 */

#ifndef QUANTIZE_FUNCTIONS_H
#define QUANTIZE_FUNCTIONS_H

#include "common.h"

namespace quant {

constexpr int DIGIT_TWO = 2;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax右移7位(BF16E8M7)
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint16_t FP4_E1M2_BF16_MAX_EXP = 0x0000;
constexpr uint16_t SPECIAL_VALUE_E2M1 = 0x00ff;
constexpr uint16_t SPECIAL_VALUE_E1M2 = 0x007f;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3_MAX_VALUE = 448.0f;
constexpr float HIFP8_MAX_VALUE = 32768.0f;
constexpr float INT8_MAX_VALUE = 127.0f;
constexpr uint16_t INVALID_FLOAT16 = 0x7c00;
constexpr uint16_t NEW_MANTISSA = 0x0008;

using namespace AscendC;

__aicore__ inline constexpr uint32_t GetUbBlockSizeDispatch()
{
    return 32U;
}

__aicore__ inline constexpr uint32_t GetVRegSizeDispatch()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

template<typename T>
__aicore__ inline void ComputeMaxExp(__ubuf__ T* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint32_t totalCountInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(T); // 每个向量寄存器可以存储的元素个数
    uint16_t elementAfterReduce = GetVRegSizeDispatch() / GetUbBlockSizeDispatch(); // Reduce操作后搬出的元素个数
    uint16_t loopNum = Ceil(totalCountInUB, 2 * vlForHalfNumber);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vdExp0;
        MicroAPI::RegTensor<T> vdExp1;
        MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;
        MicroAPI::RegTensor<uint16_t> vdExpSelect0;
        MicroAPI::RegTensor<uint16_t> vdExpSelect1;
        MicroAPI::RegTensor<uint16_t> vdExpExtract0;
        MicroAPI::RegTensor<uint16_t> vdExpExtract1;

        MicroAPI::RegTensor<uint16_t> expMaskBF16;
        MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

        MicroAPI::RegTensor<uint16_t> invalidMaskFP16;
        MicroAPI::Duplicate(invalidMaskFP16, INVALID_FLOAT16);
        MicroAPI::RegTensor<uint16_t> vdMaxExp;
        MicroAPI::MaskReg scaleMask1;
        MicroAPI::MaskReg scaleMask2;
        MicroAPI::MaskReg invalidDataMask0;
        MicroAPI::MaskReg invalidDataMask1;
        MicroAPI::UnalignReg u1;
        static constexpr MicroAPI::CastTrait castTraitHalf2Bf16 = {
            MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
            MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = MicroAPI::UpdateMask<T>(totalCountInUB);
            scaleMask2 = MicroAPI::UpdateMask<T>(totalCountInUB);
            // 双搬，将数据交织搬运到两个向量寄存器上
            MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE,
            MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
            if constexpr (Std::IsSame<T, half>::value) {
                MicroAPI::And(vdExpSelect0, (MicroAPI::RegTensor<uint16_t>&)vdExp0, invalidMaskFP16,
                    scaleMask1); // 将FP16非正常指数位与每个元素的指数位进行与操作，消除尾数位
                MicroAPI::And(vdExpSelect1, (MicroAPI::RegTensor<uint16_t>&)vdExp1, invalidMaskFP16,
                    scaleMask1);
                // 将FP16非正常指数位与实际指数位作对比，生成非正常指数位掩码
                MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask0, vdExpSelect0, invalidMaskFP16, scaleMask1);
                MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask1, vdExpSelect1, invalidMaskFP16, scaleMask1);
                MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, scaleMask1);
                MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, scaleMask1);
                MicroAPI::And(vdExpExtract0, (MicroAPI::RegTensor<uint16_t>&)vdExp0BF16, expMaskBF16,
                    scaleMask1); // 与操作保留指数位
                MicroAPI::And(vdExpExtract1, (MicroAPI::RegTensor<uint16_t>&)vdExp1BF16, expMaskBF16,
                    scaleMask1);
                // 筛选正常指数位，非正常值则使用expMaskBF16替代
                MicroAPI::Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
                MicroAPI::Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
            } else {
                MicroAPI::And(vdExpExtract0, (MicroAPI::RegTensor<uint16_t>&)vdExp0, expMaskBF16,
                    scaleMask1);
                MicroAPI::And(vdExpExtract1, (MicroAPI::RegTensor<uint16_t>&)vdExp1, expMaskBF16,
                    scaleMask1);
            }
            // 两个向量寄存器上的元素一一对比输出最大指数位
            MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            // 按每个block输出一个最大指数位
            MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

            MicroAPI::DataCopyUnAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(maxExpAddr,
                vdMaxExp, u1, elementAfterReduce);
        }
        MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
}

template<typename T>
__aicore__ inline void ComputeScale(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
    __ubuf__ uint16_t* halfScaleLocalAddr, uint32_t totalScaleInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(uint16_t);
    uint16_t loopNumScale = Ceil(totalScaleInUB, vlForHalfNumber);
    uint16_t maxExponent;
    if constexpr (Std::IsSame<T, fp8_e4m3fn_t>::value) {
        maxExponent = FP8_E4M3_MAX_EXP;
    } else if constexpr (Std::IsSame<T, fp8_e5m2_t>::value) {
        maxExponent = FP8_E5M2_MAX_EXP;
    } else if constexpr (Std::IsSame<T, fp4x2_e2m1_t>::value) {
        maxExponent = FP4_E2M1_BF16_MAX_EXP;
    } else {
        maxExponent = FP4_E1M2_BF16_MAX_EXP;
    }

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint16_t> expMask, vdMaxExp;
        MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
        MicroAPI::MaskReg cmpResult, zeroMask, preMaskScale;
        MicroAPI::RegTensor<uint16_t> maxExpValue;
        MicroAPI::Duplicate(maxExpValue, maxExponent);
        MicroAPI::RegTensor<uint16_t> sharedExp, scaleValue, scaleBias;
        MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);
        MicroAPI::RegTensor<uint16_t> halfScale, fp8NanRegTensor;
        MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        MicroAPI::RegTensor<uint16_t> zeroRegTensor;
        MicroAPI::Duplicate(zeroRegTensor, 0);
        MicroAPI::RegTensor<uint16_t> nanRegTensor;
        MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        MicroAPI::MaskReg invalidDataMask, specialDataMask;
        MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vdMaxExp,
                maxExpAddr, vlForHalfNumber);
            // 检测非正常值
            MicroAPI::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
            // 检测零值
            MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            // 检测超出目标格式的最大值
            MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue,
                preMaskScale);
            // 限制最大指数不超过目标格式最大指数
            MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            // 计算相对指数差值
            MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            // 右移得到缩放值
            MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            // 特殊值处理
            MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue, vlForHalfNumber / DIGIT_TWO,
                preMaskScale);

            MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias,
                preMaskScale);
            MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(halfScaleLocalAddr,
                halfScale, vlForHalfNumber, preMaskScale);
        }
    }
}

template <typename T, typename U, RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void ComputeFp8Data(__ubuf__ T* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
    __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(T);
    uint16_t elementAfterReduce = GetVRegSizeDispatch() / GetUbBlockSizeDispatch();
    uint32_t totalCountInUB2 = totalCountInUB * DIGIT_TWO;
    uint16_t loopNum = Ceil(totalCountInUB, 2 * vlForHalfNumber);
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg dataMask1;
        MicroAPI::MaskReg dataMask2;
        MicroAPI::MaskReg dataMask3;
        MicroAPI::MaskReg dataMask4;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint16_t,
            MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        MicroAPI::RegTensor<float> floatScaleForMul;
        MicroAPI::RegTensor<T> vdExp0;
        MicroAPI::RegTensor<T> vdExp1;

        MicroAPI::RegTensor<float> vdExp0FP32Zero;
        MicroAPI::RegTensor<float> vdExp0FP32One;
        MicroAPI::RegTensor<float> vdExp1FP32Zero;
        MicroAPI::RegTensor<float> vdExp1FP32One;
        MicroAPI::RegTensor<U> vdExp0FP8Zero;
        MicroAPI::RegTensor<U> vdExp0FP8One;
        MicroAPI::RegTensor<U> vdExp1FP8Zero;
        MicroAPI::RegTensor<U> vdExp1FP8One;
        // 放到索引位置0
        static constexpr MicroAPI::CastTrait castTraitZero = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
            MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        // 放到索引位置1
        static constexpr MicroAPI::CastTrait castTraitOne = {
            MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
            MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr MicroAPI::CastTrait castTrait32to8 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
            MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = MicroAPI::UpdateMask<T>(totalCountInUB);
            dataMask2 = MicroAPI::UpdateMask<T>(totalCountInUB);
            dataMask3 = MicroAPI::UpdateMask<T>(totalCountInUB2);
            dataMask4 = MicroAPI::UpdateMask<T>(totalCountInUB2);
            MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
            // 这里DIST_E2B_B16是将每个16bit元素广播到一个DataBlock(32B)中
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr, elementAfterReduce);
            // 因为前面scale计算用的BF16类型，所以对于fp16需要先cast到float再计算
            if constexpr (Std::IsSame<T, half>::value) {
                // 取偶数索引的元素Cast到vdExp0FP32Zero
                MicroAPI::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                // 取奇数索引的元素Cast到vdExp0FP32One
                MicroAPI::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                // 由于前面搬入是一个广播操作，所以直接取偶数索引元素即可
                MicroAPI::Cast<float, bfloat16_t, castTraitZero>(floatScaleForMul,
                    (MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, maskAll);
                MicroAPI::Mul(vdExp0FP32Zero, vdExp0FP32Zero, floatScaleForMul, dataMask3);
                MicroAPI::Mul(vdExp0FP32One, vdExp0FP32One, floatScaleForMul, dataMask4);
                MicroAPI::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
                MicroAPI::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask1);
                MicroAPI::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask1);
                MicroAPI::Mul(vdExp1FP32Zero, vdExp1FP32Zero, floatScaleForMul, dataMask3);
                MicroAPI::Mul(vdExp1FP32One, vdExp1FP32One, floatScaleForMul, dataMask4);
                MicroAPI::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
                MicroAPI::Interleave(vdExp0FP32Zero, vdExp1FP32Zero, vdExp0FP32Zero, vdExp1FP32Zero);
                MicroAPI::Interleave(vdExp0FP32One, vdExp1FP32One, vdExp0FP32One, vdExp1FP32One);
                MicroAPI::Cast<U, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
                MicroAPI::Cast<U, float, castTrait32to8>(vdExp0FP8One, vdExp1FP32Zero, dataMask3);
                MicroAPI::Cast<U, float, castTrait32to8>(vdExp1FP8Zero, vdExp0FP32One, dataMask4);
                MicroAPI::Cast<U, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);
            } else {
                MicroAPI::Mul(vdExp0, vdExp0, (MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
                MicroAPI::Mul(vdExp1, vdExp1, (MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
                MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                MicroAPI::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                MicroAPI::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                MicroAPI::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
                MicroAPI::Cast<U, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
                MicroAPI::Cast<U, float, castTrait32to8>(vdExp0FP8One, vdExp0FP32One, dataMask3);
                MicroAPI::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
                MicroAPI::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
                MicroAPI::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
                MicroAPI::Cast<U, float, castTrait32to8>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
                MicroAPI::Cast<U, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);
            }
            MicroAPI::DataCopy<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
                (MicroAPI::RegTensor<int8_t>&)vdExp0FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask3);
            MicroAPI::DataCopy<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
                (MicroAPI::RegTensor<int8_t>&)vdExp0FP8One, OUT_ELE_NUM_ONE_BLK, dataMask3);
            MicroAPI::DataCopy<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
                (MicroAPI::RegTensor<int8_t>&)vdExp1FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask4);
            MicroAPI::DataCopy<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
                (MicroAPI::RegTensor<int8_t>&)vdExp1FP8One, OUT_ELE_NUM_ONE_BLK, dataMask4);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void FP16Convert(
    MicroAPI::RegTensor<half>& output, MicroAPI::RegTensor<half>& input, MicroAPI::MaskReg& mask)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint16_t> specialValueTensor;
        MicroAPI::RegTensor<uint16_t> newMantissa;
        MicroAPI::RegTensor<uint16_t> andResult;
        MicroAPI::RegTensor<uint16_t> newValue;
        MicroAPI::MaskReg specialMask;
        MicroAPI::MaskReg nonzeroMask;
        uint16_t specialValue = SPECIAL_VALUE_E1M2;
        if constexpr (Std::IsSame<U, fp4x2_e2m1_t>::value) {
            specialValue = SPECIAL_VALUE_E2M1;
        }
        MicroAPI::Duplicate(specialValueTensor, specialValue);
        MicroAPI::Duplicate(newMantissa, NEW_MANTISSA);
        MicroAPI::And(andResult, (MicroAPI::RegTensor<uint16_t>&)input, specialValueTensor, mask);
        MicroAPI::CompareScalar<uint16_t, CMPMODE::GT>(nonzeroMask, andResult, 0, mask);
        MicroAPI::CompareScalar<uint16_t, CMPMODE::LT>(specialMask, andResult, NEW_MANTISSA, mask);
        MicroAPI::MaskAnd(specialMask, specialMask, nonzeroMask, mask);
        MicroAPI::Or(newValue, (MicroAPI::RegTensor<uint16_t>&)input, newMantissa, mask);
        MicroAPI::Select<uint16_t>(
            (MicroAPI::RegTensor<uint16_t>&)output, newValue, (MicroAPI::RegTensor<uint16_t>&)input,
            specialMask);
    }
}

template <typename T, typename U, RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void ComputeFp4Data(__ubuf__ T* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
    __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(T);
    uint16_t elementAfterReduce = GetVRegSizeDispatch() / GetUbBlockSizeDispatch();
    uint16_t loopNum = Ceil(totalCountInUB, 2 * vlForHalfNumber);
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg dataMask1;
        MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        MicroAPI::RegTensor<T> vdExp0;
        MicroAPI::RegTensor<T> vdExp1;

        MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;

        MicroAPI::RegTensor<U> vdExp0FP4;
        MicroAPI::RegTensor<U> vdExp1FP4;

        static constexpr MicroAPI::CastTrait castTrait = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
            MicroAPI::MaskMergeMode::ZEROING, roundMode};
        static constexpr MicroAPI::CastTrait castTraitHalf2Bf16 = {
            MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
            MicroAPI::MaskMergeMode::ZEROING, toBf16RoundMode};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = MicroAPI::UpdateMask<T>(totalCountInUB);
            MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr, elementAfterReduce);
            if constexpr (Std::IsSame<T, half>::value) {
                if constexpr (roundMode == RoundMode::CAST_RINT) {
                    FP16Convert<T, U>(vdExp0, vdExp0, dataMask1);
                    FP16Convert<T, U>(vdExp1, vdExp1, dataMask1);
                }
                MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, dataMask1);
                MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, dataMask1);
                MicroAPI::Mul(
                    vdExp0BF16, vdExp0BF16, (MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
                MicroAPI::Mul(
                    vdExp1BF16, vdExp1BF16, (MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
                MicroAPI::Interleave(vdExp0BF16, vdExp1BF16, vdExp0BF16, vdExp1BF16);
                MicroAPI::Cast<U, bfloat16_t, castTrait>(vdExp0FP4, vdExp0BF16, dataMask1);
                MicroAPI::Cast<U, bfloat16_t, castTrait>(vdExp1FP4, vdExp1BF16, dataMask1);
            } else {
                MicroAPI::Mul(vdExp0, vdExp0, (MicroAPI::RegTensor<T>&)halfScaleForMul, dataMask1);
                MicroAPI::Mul(vdExp1, vdExp1, (MicroAPI::RegTensor<T>&)halfScaleForMul, dataMask1);
                MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                MicroAPI::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
                MicroAPI::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask1);
            }
            MicroAPI::DataCopy<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
                (MicroAPI::RegTensor<int8_t>&)vdExp0FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
            MicroAPI::DataCopy<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
                (MicroAPI::RegTensor<int8_t>&)vdExp1FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
        }
    }
}

template <typename T, typename U, RoundMode RMode, bool HasSmooth>
__aicore__ inline void ComputePerTileDynamic(__ubuf__ T* srcAddr, __ubuf__ float* smoothLocalAddr,
    __ubuf__ float* scaleOutLocalAddr, __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB)
{
    uint32_t vlB16 = GetVRegSizeDispatch() / sizeof(T);
    uint32_t vlB32 = GetVRegSizeDispatch() / sizeof(float);
    uint16_t loopNum = Ceil(totalCountInUB, vlB16);
    uint32_t totalCntForB32 = totalCountInUB;
    float maxVal = 0.0f;
    if constexpr (Std::IsSame<U, fp8_e5m2_t>::value) {
        maxVal = FP8_E5M2_MAX_VALUE;
    } else if constexpr (Std::IsSame<U, fp8_e4m3fn_t>::value) {
        maxVal = FP8_E4M3_MAX_VALUE;
    } else if constexpr (Std::IsSame<U, hifloat8_t>::value) {
        maxVal = HIFP8_MAX_VALUE;
    } else if constexpr (Std::IsSame<U, int8_t>::value) {
        maxVal = INT8_MAX_VALUE;
    }

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg dataMask1;
        MicroAPI::MaskReg dataMask2;
        MicroAPI::MaskReg dataMask3;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<float,
            MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<float,
            MicroAPI::MaskPattern::VL1>();

        MicroAPI::RegTensor<T> vInB16;

        MicroAPI::RegTensor<float> vInFP32Zero;
        MicroAPI::RegTensor<float> vInFP32One;
        MicroAPI::RegTensor<float> vSmooth0;
        MicroAPI::RegTensor<float> vSmooth1;
        MicroAPI::RegTensor<float> vTileMax;
        MicroAPI::RegTensor<float> vDynScale;
        MicroAPI::RegTensor<float> vMaxVal;
        MicroAPI::RegTensor<float> vOneVal;

        MicroAPI::RegTensor<U> vOut0;
        MicroAPI::RegTensor<U> vOut1;

        MicroAPI::Duplicate(vMaxVal, maxVal);
        MicroAPI::Duplicate(vOneVal, 1.0f);

        static constexpr MicroAPI::CastTrait castTraitZero = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
            MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr MicroAPI::CastTrait castTraitOne = {
            MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
            MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        constexpr static MicroAPI::CastTrait castTrait32tof8 = {
            MicroAPI::RegLayout::ZERO,
            MicroAPI::SatMode::NO_SAT,
            MicroAPI::MaskMergeMode::ZEROING,
            RMode};

        constexpr static MicroAPI::CastTrait castTrait32tof16 = {
            MicroAPI::RegLayout::ZERO,
            MicroAPI::SatMode::SAT,
            MicroAPI::MaskMergeMode::ZEROING,
            RoundMode::CAST_RINT};

        constexpr static MicroAPI::CastTrait castTrait16toi8 = {
            MicroAPI::RegLayout::ZERO,
            MicroAPI::SatMode::SAT,
            MicroAPI::MaskMergeMode::ZEROING,
            RoundMode::CAST_TRUNC};

        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = MicroAPI::UpdateMask<T>(totalCountInUB);
            dataMask2 = MicroAPI::UpdateMask<float>(totalCntForB32);
            dataMask3 = MicroAPI::UpdateMask<float>(totalCntForB32);

            MicroAPI::DataCopy(vInB16, srcAddr + i * vlB16);
            MicroAPI::Cast<float, T, castTraitZero>(vInFP32Zero, vInB16, dataMask1);
            MicroAPI::Cast<float, T, castTraitOne>(vInFP32One, vInB16, dataMask1);
            if constexpr (HasSmooth) {
                MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    MicroAPI::LoadDist::DIST_DINTLV_B32>(vSmooth0, vSmooth1, smoothLocalAddr,
                    vlB32 * DIGIT_TWO);
                MicroAPI::Mul(vInFP32Zero, vInFP32Zero, vSmooth0, maskAll);
                MicroAPI::Mul(vInFP32One, vInFP32One, vSmooth1, maskAll);
            }
            MicroAPI::Interleave(vSmooth0, vSmooth1, vInFP32Zero, vInFP32One);
            MicroAPI::Abs(vInFP32Zero, vSmooth0, maskAll);
            MicroAPI::Abs(vInFP32One, vSmooth1, maskAll);
            MicroAPI::Max(vTileMax, vInFP32Zero, vInFP32One, maskAll);
            MicroAPI::ReduceMax(vTileMax, vTileMax, dataMask2);
            MicroAPI::Duplicate(vTileMax, vTileMax, maskAll);
            MicroAPI::Div(vDynScale, vMaxVal, vTileMax, maskAll);
            MicroAPI::Mul(vSmooth0, vSmooth0, vDynScale, maskAll);
            MicroAPI::Mul(vSmooth1, vSmooth1, vDynScale, maskAll);

            if constexpr (Std::IsSame<U, int8_t>::value) {
                MicroAPI::RegTensor<half> vHalf0;
                MicroAPI::RegTensor<half> vHalf1;
                MicroAPI::Cast<half, float, castTrait32tof16>(vHalf0, vSmooth0, maskAll);
                MicroAPI::Cast<half, float, castTrait32tof16>(vHalf1, vSmooth1, maskAll);
                MicroAPI::Cast<U, half, castTrait16toi8>(vOut0, vHalf0, maskAll);
                MicroAPI::Cast<U, half, castTrait16toi8>(vOut1, vHalf1, maskAll);
            } else {
                MicroAPI::Cast<U, float, castTrait32tof8>(vOut0, vSmooth0, dataMask2);
                MicroAPI::Cast<U, float, castTrait32tof8>(vOut1, vSmooth1, dataMask3);
            }

            MicroAPI::DataCopy<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
                (MicroAPI::RegTensor<int8_t>&)vOut0, OUT_ELE_NUM_ONE_BLK, dataMask2);
            MicroAPI::DataCopy<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
                (MicroAPI::RegTensor<int8_t>&)vOut1, OUT_ELE_NUM_ONE_BLK, dataMask3);

            MicroAPI::Div(vDynScale, vOneVal, vDynScale, maskAll);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                scaleOutLocalAddr + i, vDynScale, maskOne);
        }
    }
}

}
#endif