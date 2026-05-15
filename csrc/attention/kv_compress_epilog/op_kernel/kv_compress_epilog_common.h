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

#ifndef KV_COMPRESS_EPILOG_COMMON_H
#define KV_COMPRESS_EPILOG_COMMON_H

#include "kernel_operator.h"

namespace KvCompressEpilogOps {
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
constexpr int64_t QUANT_MDOE_GROUP_FP8 = 1;
constexpr int64_t QUANT_MDOE_GROUP_MXFP8 = 2;
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
    const LocalTensor<T1>& yLocal, const LocalTensor<T0>& xLocal, 
    float coeff, const uint16_t curRowNum, const uint32_t curColNum, const uint32_t concatColNum, const uint32_t padColNum)
{
    __local_mem__ T1* yLocalAddr = (__local_mem__ T1*)yLocal.GetPhyAddr();
    __local_mem__ T1* scaleLocalAddr = (__local_mem__ T1*)yLocal.GetPhyAddr();
    __local_mem__ T0* xLocalAddr = (__local_mem__ T0*)xLocal.GetPhyAddr();
    __local_mem__ T0* ropeXLocalAddr = (__local_mem__ T0*)xLocal.GetPhyAddr();
    __local_mem__ T0* ropeYLocalAddr = (__local_mem__ T0*)yLocal.GetPhyAddr();
    __local_mem__ T1* padLocalAddr = (__local_mem__ T1*)yLocal.GetPhyAddr();

    uint32_t quantColNum = curColNum - 64;
    uint16_t scaleColNum = CeilDiv(quantColNum, 128);
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<T0>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T1>(concatColNum+padColNum);
    uint16_t loopCountFoldTwo = loopCount / 2;
    uint16_t loopCountReminder = loopCount % 2;
    uint32_t tailReminder = curColNum - (loopCount - 1) * VL_FP32;
    uint32_t scaleColNumAlign = RoundUp<T0>(scaleColNum);
    uint32_t sregNum = loopCountReminder == 0 ? quantColNum - loopCountFoldTwo * VL_FP32 : loopCountFoldTwo * VL_FP32;
    __VEC_SCOPE__
    {
        RegTensor<float> xLeft;
        RegTensor<float> xRight;
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
        RegTensor<float> zeros;
        RegTensor<T0> ropeReg;
        UnalignRegForStore ureg0;
        UnalignRegForLoad ureg1;
        UnalignRegForStore ureg2;
        MaskReg pregLoop = CreateMask<float>();
        Duplicate(one, static_cast<float>(1.0f), pregLoop);
        Duplicate(zeros, static_cast<float>(0.0f), pregLoop);
        Duplicate(inf, POS_INFINITY, pregLoop);
        MaskReg pregMain = CreateMask<float>();
        MaskReg preg1 = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        for (uint16_t i = 0; i < curRowNum; i++) {
            // cat scale
            scaleLocalAddr = scaleLocalAddr + quantColNum + 128; // quantColNum个B8 + 64个B16元素
            uint32_t sreg = sregNum;
            for (uint16_t j = 0; j < loopCountFoldTwo; j++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<T0>(xLeft, xLocalAddr, pregMain, 2 * j * VL_FP32 + i * curColNum);
                LoadInputData<T0>(xRight, xLocalAddr, pregLoop, (2 * j + 1) * VL_FP32 + i * curColNum);
                Abs(xAbsLeft, xLeft, pregMain);
                ReduceMax(scale0, xAbsLeft, pregMain);
                Abs(xAbsRight, xRight, pregLoop);
                ReduceMax(scale1, xAbsRight, pregLoop);
                Max(scale, scale0, scale1, preg1);
                Muls(scale, scale, coeff, preg1);
                Min(scale, scale, inf, preg1);
                Duplicate(dupScale, scale, pregMain);
                // cat scale
                StoreUnAlign(scaleLocalAddr, (RegTensor<T1>&)scale, ureg2, 4);
                StoreUnAlignPost(scaleLocalAddr, ureg2, 0);
                Div(xLeft, xLeft, dupScale, pregMain);
                Div(xRight, xRight, dupScale, pregLoop);
                StoreOutputData<T1>(yLocalAddr, xLeft, pregMain, 2 * j * VL_FP32 + i * dstCurColNumAlign);
                StoreOutputData<T1>(yLocalAddr, xRight, pregLoop, (2 * j + 1) * VL_FP32 + i * dstCurColNumAlign);
            }
            // 处理尾块, 这里只有一个for循环
            pregLoop = UpdateMask<float>(tailReminder);
            for (uint16_t j = 0; j < loopCountReminder; j++) {
                LoadInputData<T0>(xLeft, xLocalAddr, pregLoop, loopCountFoldTwo * 2 * VL_FP32 + i * curColNum);
                Abs(xAbsLeft, xLeft, pregLoop);
                ReduceMax(scale, xAbsLeft, pregLoop);
                Muls(scale, scale, coeff, preg1);
                Min(scale, scale, inf, preg1);
                Duplicate(dupScale, scale, pregLoop);
                // cat scale
                StoreUnAlign(scaleLocalAddr, (RegTensor<T1>&)scale, ureg2, 4);
                StoreUnAlignPost(scaleLocalAddr, ureg2, 0);
                Div(xLeft, xLeft, dupScale, pregLoop);
                StoreOutputData(yLocalAddr, xLeft, pregLoop, loopCountFoldTwo * 2 * VL_FP32 + i * dstCurColNumAlign);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_STORE>();
            // cat rope
            ropeXLocalAddr = ropeXLocalAddr + quantColNum;
            ropeYLocalAddr = ropeYLocalAddr + quantColNum / 2;

            LoadUnAlignPre(ureg0, ropeXLocalAddr);
            LoadUnAlign(ropeReg, ureg0, ropeXLocalAddr, 64);
            StoreUnAlign(ropeYLocalAddr, ropeReg, ureg1, 64);
            StoreUnAlignPost(ropeYLocalAddr, ureg1, 0);

            // pad zero
            padLocalAddr = padLocalAddr + concatColNum;
            StoreUnAlign(padLocalAddr, (RegTensor<T1>&)zeros, ureg1, padColNum);
            StoreUnAlignPost(padLocalAddr, ureg1, 0);
            
            ropeYLocalAddr = ropeYLocalAddr + scaleColNum * 2 + padColNum / 2;
            scaleLocalAddr = scaleLocalAddr + padColNum;
        }
    }
}

template <typename T0, typename T1, bool roundScale = true>
__aicore__ inline void VFProcessDynamicMxFP8Quant(
    const LocalTensor<T1>& yLocal, const LocalTensor<T0>& xLocal, 
    float coeff, float fp8Min, float fp8Max, const uint16_t curRowNum, const uint32_t curColNum, const uint32_t concatColNum, const uint32_t padColNum) 
{
    __local_mem__ T1* yLocalAddr = (__local_mem__ T1*)yLocal.GetPhyAddr();
    __local_mem__ T0* xLocalAddr = (__local_mem__ T0*)xLocal.GetPhyAddr();
    __local_mem__ T1* scaleLocalAddr = (__local_mem__ T1*)yLocal.GetPhyAddr();
    __local_mem__ T0* ropeXLocalAddr = (__local_mem__ T0*)xLocal.GetPhyAddr();
    __local_mem__ T0* ropeYLocalAddr = (__local_mem__ T0*)yLocal.GetPhyAddr();
    __local_mem__ T1* padLocalAddr = (__local_mem__ T1*)yLocal.GetPhyAddr();

    uint32_t quantColNum = curColNum - 64;
    uint16_t scaleColNum = CeilDiv(quantColNum, 128);
    uint16_t ropeNum = 128;
    uint16_t loopCount = CeilDiv(quantColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<T0>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T1>(concatColNum+padColNum);
    uint16_t loopCountFoldTwo = loopCount / 2;
    uint16_t loopCountReminder = loopCount % 2;
    uint32_t tailReminder = quantColNum - (loopCount - 1) * VL_FP32;
    uint32_t scaleColNumAlign = RoundUp<T0>(scaleColNum);
    uint32_t sregNum = quantColNum;
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
        RegTensor<T0> ropeReg;
        RegTensor<uint32_t> scaleTmp0;
        RegTensor<uint8_t> scaleTmp1;
        UnalignRegForStore ureg0;
        UnalignRegForLoad ureg1;
        UnalignRegForStore ureg2;
        MaskReg pregLoop;
        MaskReg preg1 = CreateMask<T0, AscendC::MicroAPI::MaskPattern::VL1>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        MaskReg pregMain = CreateMask<float>();
        MaskReg pregRope = CreateMask<T0, AscendC::MicroAPI::MaskPattern::VL64>();
        MaskReg cmpMask;
        Duplicate(tmp0, FAST_LOG_AND_VALUE1, pregMerge);
        Duplicate(tmp1, FAST_LOG_AND_VALUE2, pregMerge);
        Duplicate(zero, static_cast<uint32_t>(0), pregMerge);
        Duplicate(one, static_cast<uint32_t>(1), pregMerge);
        Duplicate(tmp3, static_cast<uint32_t>(127), pregMerge);
        for (uint16_t i = 0; i < curRowNum; i++) {
            // cat rope
            ropeXLocalAddr = ropeXLocalAddr + quantColNum;
            
            LoadUnAlignPre(ureg0, ropeXLocalAddr);
            LoadUnAlign(ropeReg, ureg0, ropeXLocalAddr, 64);
            DataCopy(ropeYLocalAddr + i * dstCurColNumAlign / 2, ropeReg, pregRope);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_STORE>();
            uint32_t sreg = sregNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<T0>(x0, xLocalAddr, pregLoop, j * VL_FP32 + i * curColNumAlign);
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
                Duplicate(dupScale, max2, pregMain);
                Div(x0, x0, dupScale, pregLoop);
                Maxs(x0, x0, fp8Min, pregLoop);
                Mins(x0, x0, fp8Max, pregLoop);
                StoreOutputData<T1>(yLocalAddr, x0, pregLoop, j * VL_FP32 + i * dstCurColNumAlign + ropeNum);
                
                // cat scale
                ShiftRights(scaleTmp0, (RegTensor<uint32_t> &)max2, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), preg1);
                Cast<uint8_t, uint32_t, castTraitU32toU8Even>(scaleTmp1, scaleTmp0, preg1);
                DataCopy<T1, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B8>(scaleLocalAddr + quantColNum + ropeNum + j + i * dstCurColNumAlign, (RegTensor<T1>&)scaleTmp1, preg1);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_STORE>();

            // pad zero
            padLocalAddr = padLocalAddr + concatColNum;
            StoreUnAlign(padLocalAddr, (RegTensor<T1>&)zero, ureg1, padColNum);
            StoreUnAlignPost(padLocalAddr, ureg1, 0);
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

} // namespace KvCompressEpilogOps

#endif