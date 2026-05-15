/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file swiglu_block_quant_base.h
 * \brief
 */

#ifndef SWIGLU_BLOCK_QUANT_BASE_H
#define SWIGLU_BLOCK_QUANT_BASE_H

#include "kernel_operator.h"

namespace IndexerCompressEpilog {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UnalignReg;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t VL_FP32 = 64;
constexpr int32_t PER_BLOCK_FP16 = 128;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;
constexpr float FP8_E5M2_MIN_VALUE = -57344.0f;
constexpr float FP8_E4M3FN_MIN_VALUE = -448.0f;
constexpr uint32_t FAST_LOG_SHIFT_BITS = 23U;
constexpr uint32_t FAST_LOG_AND_VALUE1 = 0xFF;
constexpr uint32_t FAST_LOG_AND_VALUE2 = (((uint32_t)1 << (uint32_t)23) - (uint32_t)1);

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif
constexpr float POS_INFINITY = INFINITY;
constexpr float NEG_INFINITY = -INFINITY;

__aicore__ inline int32_t CeilDiv(int32_t a, int b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int32_t CeilAlign(int32_t a, int b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num)
{
    int32_t elemNum = BLOCK_SIZE / sizeof(T);
    return CeilAlign(num, elemNum);
}

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num, int32_t elemNum)
{
    return CeilAlign(num, elemNum);
}

constexpr AscendC::MicroAPI::CastTrait castTraitB162B32Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitB322B16Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitF32toFp8Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitU32toU8Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};


template <typename T>
__aicore__ inline void LoadInputData(RegTensor<float>& dst, __local_mem__ T* src, MaskReg pregLoop, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dst, src + srcOffset);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void StoreOutputData(
    __local_mem__ T* dst, RegTensor<float>& src, MaskReg pregLoop, uint32_t dstOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dst + dstOffset, src, pregLoop);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitB322B16Even>(tmp, src, pregLoop);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + dstOffset, tmp, pregLoop);
    } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, fp8_e5m2_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitF32toFp8Even>(tmp, src, pregLoop);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(dst + dstOffset, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void StoreMxFp8Scale(
    __local_mem__ T* dst, RegTensor<float>& src, MaskReg pregLoop, uint32_t dstOffset)
{
    RegTensor<uint32_t> tmp;
    RegTensor<uint8_t> tmp1;
    ShiftRights(tmp, (RegTensor<uint32_t> &)src, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), pregLoop);
    Cast<uint8_t, uint32_t, castTraitU32toU8Even>(tmp1, tmp, pregLoop);
    DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B8>(dst + dstOffset, (RegTensor<T> &)tmp1, pregLoop);
}


template <typename T0, typename T1, typename T2, bool roundScale = true>
__aicore__ inline void VFProcessDynamicMxFp8Quant(
    const LocalTensor<T0>& yLocal, const LocalTensor<T1>& scaleLocal, const LocalTensor<T2>& xLocal, 
    float coeff, float fp8Min, float fp8Max, const uint16_t curRowNum, const uint32_t curColNum) 
{
    __local_mem__ T0* yLocalAddr = (__local_mem__ T0*)yLocal.GetPhyAddr();
    __local_mem__ T1* scaleLocalAddr = (__local_mem__ T1*)scaleLocal.GetPhyAddr();
    __local_mem__ T2* xLocalAddr = (__local_mem__ T2*)xLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<T2>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T0>(curColNum);
    uint16_t loopCountFoldTwo = loopCount / 2;
    uint16_t loopCountReminder = loopCount % 2;
    uint32_t tailReminder = curColNum - (loopCount - 1) * VL_FP32;
    uint32_t scaleColNumAlign = RoundUp<T1>((curColNum + 128 - 1) / 128);
    uint32_t sregNum = loopCountReminder == 0 ? curColNum - loopCountFoldTwo * VL_FP32 : loopCountFoldTwo * VL_FP32;
    __VEC_SCOPE__
    {
        RegTensor<float> x0;
        RegTensor<float> x0Abs;
        RegTensor<float> x1;
        RegTensor<float> x1Abs;
        RegTensor<float> max0;
        RegTensor<float> max1;
        RegTensor<float> max2;
        RegTensor<uint32_t> tmp0;
        RegTensor<uint32_t> tmp1;
        RegTensor<uint32_t> vreg0;
        RegTensor<uint32_t> vreg1;
        RegTensor<uint32_t> vreg2;
        RegTensor<uint32_t> vreg3;
        RegTensor<uint32_t> vreg4;
        RegTensor<int32_t> vreg5;
        RegTensor<uint32_t> zero;
        RegTensor<uint32_t> one;
        RegTensor<uint32_t> tmp3;
        RegTensor<float> dupScale;
        MaskReg pregLoop;
        MaskReg preg1 = CreateMask<T1, AscendC::MicroAPI::MaskPattern::VL1>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        MaskReg pregMain = CreateMask<float>();
        MaskReg cmpMask;
        Duplicate(tmp0, FAST_LOG_AND_VALUE1, pregMerge);
        Duplicate(tmp1, FAST_LOG_AND_VALUE2, pregMerge);
        Duplicate(zero, static_cast<uint32_t>(0), pregMerge);
        Duplicate(one, static_cast<uint32_t>(1), pregMerge);
        Duplicate(tmp3, static_cast<uint32_t>(127), pregMerge);
        for (uint16_t i = 0; i < curRowNum; i++) {
            uint32_t sreg = sregNum;
            for (uint16_t j = 0; j < loopCountFoldTwo; j++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<T2>(x0, xLocalAddr, pregMain, 2 * j * VL_FP32 + i * curColNumAlign);
                LoadInputData<T2>(x1, xLocalAddr, pregLoop, (2 * j + 1) * VL_FP32 + i * curColNumAlign);
                Abs(x0Abs, x0, pregMain);
                Abs(x1Abs, x1, pregLoop);
                ReduceMax(max0, x0Abs, pregMain);
                ReduceMax(max1, x1Abs, pregLoop);
                Max(max2, max0, max1, pregMerge);
                Maxs(max2, max2, static_cast<float>(1e-4), pregMerge);
                Muls(max2, max2, coeff, pregMerge);
                if constexpr (roundScale) {
                    ShiftRights(vreg0, (RegTensor<uint32_t> &)max2, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), pregMerge);
                    And(vreg1, vreg0, tmp0, pregMerge);
                    And(vreg2, vreg1, tmp1, pregMerge);
                    Compare<uint32_t, AscendC::CMPMODE::NE>(cmpMask, vreg2, zero, pregMerge);
                    Select(vreg4, one, zero, cmpMask);
                    Sub(vreg1, vreg1, tmp3, pregMerge);
                    Add(vreg1, vreg1, vreg4, pregMerge);
                    Adds(vreg5, (RegTensor<int32_t> &)vreg1, static_cast<int32_t>(127), pregMerge);
                    ShiftLefts((RegTensor<int32_t> &)max2, vreg5, static_cast<int16_t>(23), pregMerge);
                }
                Duplicate(dupScale, max2, pregMain);
                Div(x0, x0, dupScale, pregMain);
                Div(x1, x1, dupScale, pregLoop);
                Maxs(x0, x0, fp8Min, pregMain);
                Mins(x0, x0, fp8Max, pregMain);
                Maxs(x1, x1, fp8Min, pregLoop);
                Mins(x1, x1, fp8Max, pregLoop);
                StoreOutputData<T0>(yLocalAddr, x0, pregMain, 2 * j * VL_FP32 + i * dstCurColNumAlign);
                StoreOutputData<T0>(yLocalAddr, x1, pregLoop, (2 * j + 1) * VL_FP32 + i * dstCurColNumAlign);
                StoreMxFp8Scale<T1>(scaleLocalAddr, max2, preg1, j + i * scaleColNumAlign);
            }
            // 处理尾块, 这里只有一个for循环
            pregLoop = UpdateMask<float>(tailReminder);
            for (uint16_t j = 0; j < loopCountReminder; j++) {
                LoadInputData<T2>(x0, xLocalAddr, pregLoop, 2 * loopCountFoldTwo * VL_FP32 + i * curColNumAlign);
                Abs(x0Abs, x0, pregLoop);
                ReduceMax(max0, x0Abs, pregLoop);
                Maxs(max2, max0, static_cast<float>(1e-4), pregMerge);
                Muls(max2, max2, coeff, pregMerge);
                if constexpr (roundScale) {
                    ShiftRights(vreg0, (RegTensor<uint32_t> &)max2, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), pregMerge);
                    And(vreg1, vreg0, tmp0, pregMerge);
                    And(vreg2, vreg1, tmp1, pregMerge);
                    Compare<uint32_t, AscendC::CMPMODE::NE>(cmpMask, vreg2, zero, pregMerge);
                    Select(vreg4, one, zero, cmpMask);
                    Sub(vreg1, vreg1, tmp3, pregMerge);
                    Add(vreg1, vreg1, vreg4, pregMerge);
                    Adds(vreg5, (RegTensor<int32_t> &)vreg1, static_cast<int32_t>(127), pregMerge);
                    ShiftLefts((RegTensor<int32_t> &)max2, vreg5, static_cast<int16_t>(23), pregMerge);
                }
                Duplicate(dupScale, max2, pregLoop);
                Div(x0, x0, dupScale, pregLoop);
                Maxs(x0, x0, fp8Min, pregLoop);
                Mins(x0, x0, fp8Max, pregLoop);
                StoreOutputData<T0>(yLocalAddr, x0, pregLoop, 2 * loopCountFoldTwo * VL_FP32 + i * dstCurColNumAlign);
                StoreMxFp8Scale<T1>(scaleLocalAddr, max2, preg1, loopCountFoldTwo + i * scaleColNumAlign);
            }
        }
    }
}

template <typename T0, typename T1>
__aicore__ inline void VFProcessDynamicBlockQuant(
    const LocalTensor<T0>& yLocal, const LocalTensor<float>& scaleLocal, const LocalTensor<T1>& xLocal, 
    float coeff, const uint16_t curRowNum, const uint32_t curColNum)
{
    __local_mem__ T0* yLocalAddr = (__local_mem__ T0*)yLocal.GetPhyAddr();
    __local_mem__ float* scaleLocalAddr = (__local_mem__ float*)scaleLocal.GetPhyAddr();
    __local_mem__ T1* xLocalAddr = (__local_mem__ T1*)xLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<T1>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T0>(curColNum);
    uint16_t loopCountFoldTwo = loopCount / 2;
    uint16_t loopCountReminder = loopCount % 2;
    uint32_t tailReminder = curColNum - (loopCount - 1) * VL_FP32;
    uint32_t scaleColNumAlign = RoundUp<float>((curColNum + 128 - 1) / 128);
    uint32_t sregNum = loopCountReminder == 0 ? curColNum - loopCountFoldTwo * VL_FP32 : loopCountFoldTwo * VL_FP32;

    __VEC_SCOPE__
    {
        RegTensor<float> xLeft;
        RegTensor<float> xRight;
        RegTensor<float> x0Left;
        RegTensor<float> x0Right;
        RegTensor<float> x1Left;
        RegTensor<float> x1Right;
        RegTensor<float> xAbsLeft;
        RegTensor<float> xAbsRight;
        RegTensor<float> xMax;
        RegTensor<float> tmp;
        RegTensor<float> dupScale;
        RegTensor<float> scale;
        RegTensor<float> scale0;
        RegTensor<float> scale1;
        RegTensor<float> inf;
        RegTensor<float> one;
        MaskReg pregLoop = CreateMask<float>();
        Duplicate(one, static_cast<float>(1.0f), pregLoop);
        Duplicate(inf, POS_INFINITY, pregLoop);
        MaskReg pregMain = CreateMask<float>();
        MaskReg preg1 = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        for (uint16_t i = 0; i < curRowNum; i++) {
            uint32_t sreg = sregNum;
            for (uint16_t j = 0; j < loopCountFoldTwo; j++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<T1>(xLeft, xLocalAddr, pregMain, 2 * j * VL_FP32 + i * curColNumAlign);
                LoadInputData<T1>(xRight, xLocalAddr, pregLoop, (2 * j + 1) * VL_FP32 + i * curColNumAlign);
                Abs(xAbsLeft, xLeft, pregMain);
                ReduceMax(scale0, xAbsLeft, pregMain);
                Abs(xAbsRight, xRight, pregLoop);
                ReduceMax(scale1, xAbsRight, pregLoop);
                Max(scale, scale0, scale1, preg1);
                Muls(scale, scale, coeff, preg1);
                Min(scale, scale, inf, preg1);
                Duplicate(dupScale, scale, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleLocalAddr + j + i * scaleColNumAlign, scale, preg1);
                Div(xLeft, xLeft, dupScale, pregMain);
                Div(xRight, xRight, dupScale, pregLoop);
                StoreOutputData<T0>(yLocalAddr, xLeft, pregMain, 2 * j * VL_FP32 + i * dstCurColNumAlign);
                StoreOutputData<T0>(yLocalAddr, xRight, pregLoop, (2 * j + 1) * VL_FP32 + i * dstCurColNumAlign);
            }
            // 处理尾块, 这里只有一个for循环
            pregLoop = UpdateMask<float>(tailReminder);
            for (uint16_t j = 0; j < loopCountReminder; j++) {
                LoadInputData<T1>(xLeft, xLocalAddr, pregLoop, loopCountFoldTwo * 2 * VL_FP32 + i * curColNumAlign);
                Abs(xAbsLeft, xLeft, pregLoop);
                ReduceMax(scale, xAbsLeft, pregLoop);
                Muls(scale, scale, coeff, preg1);
                Min(scale, scale, inf, preg1);
                Duplicate(dupScale, scale, pregLoop);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleLocalAddr + loopCountFoldTwo + i * scaleColNumAlign, scale, preg1);
                Div(xLeft, xLeft, dupScale, pregLoop);
                StoreOutputData(yLocalAddr, xLeft, pregLoop, loopCountFoldTwo * 2 * VL_FP32 + i * dstCurColNumAlign);
            }
        }
    }
}


template <typename T>
__aicore__ inline void CopyIn(
    const GlobalTensor<T>& inputGm, const LocalTensor<T>& inputTensor, const uint16_t nBurst, const uint32_t copyLen,
    uint32_t srcStride = 0)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = srcStride * sizeof(T);
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(inputTensor, inputGm, dataCoptExtParams, dataCopyPadExtParams);
}


template <typename T, AscendC::PaddingMode mode = AscendC::PaddingMode::Normal>
__aicore__ inline void CopyOut(
    const LocalTensor<T>& outputTensor, const GlobalTensor<T>& outputGm, const uint16_t nBurst, const uint32_t copyLen,
    uint32_t dstStride = 0)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = dstStride * sizeof(T);
    DataCopyPad<T, mode>(outputGm, outputTensor, dataCopyParams);
}

} // namespace SwigluBlockQuant

#endif