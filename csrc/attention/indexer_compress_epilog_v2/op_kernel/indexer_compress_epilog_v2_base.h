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
 * \file indexer_compress_epilog_v2_base.h
 * \brief
 */

#ifndef INDEXER_COMPRESS_EPILOG_V2_BASE_H
#define INDEXER_COMPRESS_EPILOG_V2_BASE_H

#include "kernel_operator.h"

namespace IndexerCompressEpilogV2 {
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
constexpr uint32_t INV_FP8_E5M2_MAX_VALUE = 0x37924925;
constexpr uint32_t INV_FP8_E4M3_MAX_VALUE = 0x3b124925;
constexpr int64_t B32_INTERPRE_TO_B8_RATIO = 4;

#define FLOAT_OVERFLOW_MODE_CTRL 60
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
    static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, false};
    uint32_t maxValueInt = 0;
    if constexpr (IsSameType<T0, fp8_e5m2_t>::value) {
        maxValueInt = INV_FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<T0, fp8_e4m3fn_t>::value) {
        maxValueInt = INV_FP8_E4M3_MAX_VALUE;
    }

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
        RegTensor<float> zero;
        RegTensor<uint32_t> coeffReg;
        MaskReg pregLoop = CreateMask<float>();
        Duplicate(one, static_cast<float>(1.0f), pregLoop);
        Duplicate(coeffReg, maxValueInt, pregLoop);
        Duplicate(zero, 0.0f);
        Duplicate(inf, 1.0f);
        Div<float, &mode>(inf, inf, zero, pregLoop);
        MaskReg pregMain = CreateMask<float>();
        MaskReg preg1 = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        MaskReg compareLeft;
         MaskReg compareRight;
        MaskReg compareScalar;
        for (uint16_t i = 0; i < curRowNum; i++) {
            uint32_t sreg = sregNum;
            for (uint16_t j = 0; j < loopCountFoldTwo; j++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<T1>(xLeft, xLocalAddr, pregMain, 2 * j * VL_FP32 + i * curColNumAlign);
                LoadInputData<T1>(xRight, xLocalAddr, pregLoop, (2 * j + 1) * VL_FP32 + i * curColNumAlign);
                Muls(xAbsLeft, xLeft, 0.0f, pregMain);
                 Compare<float, CMPMODE::NE>(compareLeft, xAbsLeft, xAbsLeft, pregMain);
                 MaskNot(compareLeft, compareLeft, pregMain);
                 Abs(xAbsLeft, xLeft, compareLeft);
                ReduceMax(scale0, xAbsLeft, pregMain);
                Muls(xAbsRight, xRight, 0.0f, pregLoop);
                 Compare<float, CMPMODE::NE>(compareRight, xAbsRight, xAbsRight, pregLoop);
                 MaskNot(compareRight, compareRight, pregLoop);
                 Abs(xAbsRight, xRight, compareRight);
                ReduceMax(scale1, xAbsRight, pregLoop);
                Max(scale, scale0, scale1, preg1);
                CompareScalar<float, CMPMODE::NE>(compareScalar, scale, (float)0.0, preg1);
                Mul(scale, scale, (RegTensor<float>&)coeffReg, compareScalar);
                Min(scale, scale, inf, preg1);
                Duplicate(dupScale, scale, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleLocalAddr + j + i * scaleColNumAlign, scale, preg1);
                Div<float, &mode>(x0Left, xLeft, dupScale, pregMain);
                Muls(x1Left, x0Left, 0.0f, pregMain);
                Compare<float, CMPMODE::NE>(compareLeft, x1Left, x1Left, pregMain);
                Select(xLeft, xLeft, x0Left, compareLeft);
                Div<float, &mode>(x0Right, xRight, dupScale, pregLoop);
                Muls(x1Right, x0Right, 0.0f, pregLoop);
                Compare<float, CMPMODE::NE>(compareRight, x1Right, x1Right, pregLoop);
                Select(xRight, xRight, x0Right, compareRight);
                StoreOutputData<T0>(yLocalAddr, xLeft, pregMain, 2 * j * VL_FP32 + i * dstCurColNumAlign);
                StoreOutputData<T0>(yLocalAddr, xRight, pregLoop, (2 * j + 1) * VL_FP32 + i * dstCurColNumAlign);
            }
            // 处理尾块, 这里只有一个for循环
            uint32_t sregTail = tailReminder;
            pregLoop = UpdateMask<float>(sregTail);
            for (uint16_t j = 0; j < loopCountReminder; j++) {
                LoadInputData<T1>(xLeft, xLocalAddr, pregLoop, loopCountFoldTwo * 2 * VL_FP32 + i * curColNumAlign);
                Muls(xAbsLeft, xLeft, 0.0f, pregLoop);
                 Compare<float, CMPMODE::NE>(compareLeft, xAbsLeft, xAbsLeft, pregLoop);
                 MaskNot(compareLeft, compareLeft, pregLoop);
                Abs(xAbsLeft, xLeft, compareLeft);
                ReduceMax(scale, xAbsLeft, pregLoop);
                CompareScalar<float, CMPMODE::NE>(compareScalar, scale, (float)0.0, preg1);
                Mul(scale, scale, (RegTensor<float>&)coeffReg, compareScalar);
                Min(scale, scale, inf, preg1);
                Duplicate(dupScale, scale, pregLoop);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleLocalAddr + loopCountFoldTwo + i * scaleColNumAlign, scale, preg1);
                Div<float, &mode>(x0Left, xLeft, dupScale, pregLoop);
                Muls(x1Left, x0Left, 0.0f, pregLoop);
                Compare<float, CMPMODE::NE>(compareLeft, x1Left, x1Left, pregLoop);
                Select(xLeft, xLeft, x0Left, compareLeft);
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

} // namespace IndexerCompressEpilogV2

#endif