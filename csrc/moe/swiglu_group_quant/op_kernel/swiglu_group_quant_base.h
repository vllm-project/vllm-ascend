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
 * \file swiglu_group_quant_base.h
 * \brief
 */

#ifndef SWIGLU_GROUP_QUANT_BASE_H
#define SWIGLU_GROUP_QUANT_BASE_H

#include "kernel_operator.h"

namespace SwigluGroupQuant {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UnalignReg;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t VL_FP32 = 64;
constexpr int32_t PER_BLOCK_FP16 = 128;
constexpr int32_t PER_MX_FP16 = 32;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;
constexpr float TOPK_WEIGHT_DEFAULT = 1.0f;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64LL;
constexpr uint16_t FP16_EMASK_AND_INF_VAL = 0x7c00;
constexpr uint16_t BF16_EMASK_AND_INF_VAL = 0x7f80;
constexpr uint16_t BF16_NAN_VAL = 0x7f81;
constexpr uint16_t LOWER_BOUND_OF_MAX_EXP_FOR_E5M2 = 0x0780;
constexpr uint16_t LOWER_BOUND_OF_MAX_EXP_FOR_E4M3 = 0x0400;
constexpr uint16_t FP8_E8M0_NAN_VAL = 0x00ff;
constexpr uint16_t FP8_E8M0_SPECIAL_MIN = 0x0040;
constexpr int16_t BF16_EXP_SHR_BITS = 7;
constexpr uint16_t BF16_EXP_INVSUB = 0x7f00;
constexpr uint32_t INV_FP8_E5M2_MAX_VALUE = 0x37924925;
constexpr uint32_t INV_FP8_E4M3_MAX_VALUE = 0x3b124925;
constexpr uint32_t FAST_LOG_SHIFT_BITS = 23U;
constexpr uint32_t FAST_LOG_AND_VALUE1 = 0xFF;
constexpr uint32_t FAST_LOG_AND_VALUE2 = (((uint32_t)1 << (uint32_t)23) - (uint32_t)1);
constexpr uint32_t REPEAT_SIZE = 256;
constexpr uint16_t FOUR_UNFOLD = 4;
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
__aicore__ inline void StoreOuputDataUnalign(
    RegTensor<float>& src, __local_mem__ T*& dst, UnalignReg& uDst, MaskReg pregLoop, uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopyUnAlign(dst, src, uDst, postUpdateStride);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        RegTensor<T> tmpPack;
        Cast<T, float, castTraitB322B16Even>(tmp, src, pregLoop);
        Pack((RegTensor<uint16_t>&)tmpPack, (RegTensor<uint32_t>&)tmp);
        DataCopyUnAlign(dst, tmpPack, uDst, postUpdateStride);
    }
}

template <typename T>
__aicore__ inline void StoreMxFp8Scale(
    __local_mem__ T* dst, RegTensor<int32_t>& src, MaskReg pregLoop, uint32_t dstOffset)
{
    RegTensor<uint8_t> tmp1;
    Cast<uint8_t, int32_t, castTraitU32toU8Even>(tmp1, src, pregLoop);
    DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B8>(dst + dstOffset, (RegTensor<T> &)tmp1, pregLoop);
}

__aicore__ inline void VFSwiGlu(
    RegTensor<float>& y, RegTensor<float>& x0, RegTensor<float>& x1, RegTensor<float>& one, RegTensor<float>& vreg, MaskReg pregLoop)
{
    Muls(vreg, x0, static_cast<float>(-1.0f), pregLoop);
    Exp(vreg, vreg, pregLoop);
    Adds(vreg, vreg, static_cast<float>(1.0f), pregLoop);
    Div(vreg, x0, vreg, pregLoop);
    Mul(y, vreg, x1, pregLoop);
}

template <typename T0, typename T1, typename T2, bool hasTopkWeight = false, bool hasClampValue = false>
__aicore__ inline void VFProcessSwigluGroupQuant(
    const LocalTensor<T0>& yLocal, const LocalTensor<T2>& scaleLocal, const LocalTensor<T1>& x0Local,
    const LocalTensor<T1>& x1Local, const LocalTensor<float> &topkWeightLocal, float coeff, const uint16_t curRowNum, const uint32_t curColNum, float clampValue)
{
    __local_mem__ T0* yLocalAddr = (__local_mem__ T0*)yLocal.GetPhyAddr();
    __local_mem__ T2* scaleLocalAddr = (__local_mem__ T2*)scaleLocal.GetPhyAddr();
    __local_mem__ T1* x0LocalAddr = (__local_mem__ T1*)x0Local.GetPhyAddr();
    __local_mem__ T1* x1LocalAddr = (__local_mem__ T1*)x1Local.GetPhyAddr();
    __local_mem__ float* topkWeightLocalAddr = hasTopkWeight ? (__local_mem__ float*)topkWeightLocal.GetPhyAddr() : nullptr;
    static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, false};
    uint32_t maxValueInt = 0;
    if constexpr (IsSameType<T0, fp8_e5m2_t>::value) {
        maxValueInt = INV_FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<T0, fp8_e4m3fn_t>::value) {
        maxValueInt = INV_FP8_E4M3_MAX_VALUE;
    }
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<T1>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T0>(curColNum);
    uint16_t loopCountFoldTwo = loopCount / 2;
    uint16_t loopCountReminder = loopCount % 2;
    uint32_t tailRemider = curColNum - (loopCount - 1) * VL_FP32;
    uint32_t scaleColNum = (curColNum + 128 - 1) / 128;
    uint32_t sregNum = loopCountReminder == 0 ? curColNum - loopCountFoldTwo * VL_FP32 : loopCountFoldTwo * VL_FP32;
    __VEC_SCOPE__
    {
        RegTensor<float> weight;
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
        UnalignReg uScale;
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
            if constexpr (hasTopkWeight) {
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(weight, topkWeightLocalAddr + i);
            }
            uint32_t sreg = sregNum;
            for (uint16_t j = 0; j < loopCountFoldTwo; j++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<T1>(x0Left, x0LocalAddr, pregMain, 2 * j * VL_FP32 + i * curColNumAlign);
                LoadInputData<T1>(x0Right, x0LocalAddr, pregLoop, (2 * j + 1) * VL_FP32 + i * curColNumAlign);
                LoadInputData<T1>(x1Left, x1LocalAddr, pregMain, 2 * j * VL_FP32 + i * curColNumAlign);
                LoadInputData<T1>(x1Right, x1LocalAddr, pregLoop, (2 * j + 1) * VL_FP32 + i * curColNumAlign);
                if constexpr (hasClampValue) {
                    Mins(x0Left, x0Left, clampValue, pregMain);
                    Mins(x0Right, x0Right, clampValue, pregLoop);
                    Maxs(x1Left, x1Left, -clampValue, pregMain);
                    Mins(x1Left, x1Left, clampValue, pregMain);
                    Maxs(x1Right, x1Right, -clampValue, pregLoop);
                    Mins(x1Right, x1Right, clampValue, pregLoop);
                }
                VFSwiGlu(xLeft, x0Left, x1Left, one, tmp, pregMain);
                VFSwiGlu(xRight, x0Right, x1Right, one, tmp, pregLoop);
                if constexpr (hasTopkWeight) {
                    Mul(xLeft, xLeft, weight, pregMain);
                    Mul(xRight, xRight, weight, pregLoop);
                }
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
                StoreOuputDataUnalign(scale, scaleLocalAddr, uScale, preg1, 1);
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
            pregLoop = UpdateMask<float>(tailRemider);
            for (uint16_t j = 0; j < loopCountReminder; j++) {
                LoadInputData<T1>(x0Left, x0LocalAddr, pregLoop, loopCountFoldTwo * 2 * VL_FP32 + i * curColNumAlign);
                LoadInputData<T1>(x1Left, x1LocalAddr, pregLoop, loopCountFoldTwo * 2 * VL_FP32 + i * curColNumAlign);
                if constexpr (hasClampValue) {
                    Mins(x0Left, x0Left, clampValue, pregLoop);
                    Maxs(x1Left, x1Left, -clampValue, pregLoop);
                    Mins(x1Left, x1Left, clampValue, pregLoop);
                }
                VFSwiGlu(xLeft, x0Left, x1Left, one, tmp, pregLoop);
                if constexpr (hasTopkWeight) {
                    Mul(xLeft, xLeft, weight, pregLoop);
                }
                Abs(xAbsLeft, xLeft, pregLoop);
                ReduceMax(scale, xAbsLeft, pregLoop);
                CompareScalar<float, CMPMODE::NE>(compareScalar, scale, (float)0.0, preg1);
                Mul(scale, scale, (RegTensor<float>&)coeffReg, compareScalar);
                Min(scale, scale, inf, preg1);
                Duplicate(dupScale, scale, pregLoop);
                StoreOuputDataUnalign(scale, scaleLocalAddr, uScale, preg1, 1);
                Div<float, &mode>(x0Left, xLeft, dupScale, pregLoop);
                Muls(x1Left, x0Left, 0.0f, pregLoop);
                Compare<float, CMPMODE::NE>(compareLeft, x1Left, x1Left, pregLoop);
                Select(xLeft, xLeft, x0Left, compareLeft);
                StoreOutputData(yLocalAddr, xLeft, pregLoop, loopCountFoldTwo * 2 * VL_FP32 + i * dstCurColNumAlign);
            }
        }
        DataCopyUnAlignPost(scaleLocalAddr, uScale, 0);
    }
}


template <typename T, bool hasTopkWeight = false, bool hasClampValue = false>
__aicore__ inline void VFProcessSwigluGroupQuant(
    const LocalTensor<T>& yLocal, const LocalTensor<T>& x0Local, const LocalTensor<T>& x1Local, const LocalTensor<float> &topkWeightLocal,
    const uint16_t curRowNum, const uint32_t curColNum, float clampValue)
{
    __local_mem__ T* yLocalAddr = (__local_mem__ T*)yLocal.GetPhyAddr();
    __local_mem__ T* x0LocalAddr = (__local_mem__ T*)x0Local.GetPhyAddr();
    __local_mem__ T* x1LocalAddr = (__local_mem__ T*)x1Local.GetPhyAddr();
    __local_mem__ float* topkWeightLocalAddr = hasTopkWeight ? (__local_mem__ float*)topkWeightLocal.GetPhyAddr() : nullptr;
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t sregNum = curColNum;
    uint32_t curColNumAlign = RoundUp<T>(curColNum);
    __VEC_SCOPE__
    {
        RegTensor<float> weight;
        RegTensor<float> x0;
        RegTensor<float> x1;
        RegTensor<float> y;
        RegTensor<float> one;
        RegTensor<float> tmp;
        MaskReg pregLoop = CreateMask<float>();
        Duplicate(one, static_cast<float>(1.0f), pregLoop);
        for (uint16_t i = 0; i < curRowNum; i++) {
            if constexpr (hasTopkWeight) {
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(weight, topkWeightLocalAddr + i);
            }
            uint32_t sreg = sregNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<T>(x0, x0LocalAddr, pregLoop, j * VL_FP32 + i * curColNumAlign);
                LoadInputData<T>(x1, x1LocalAddr, pregLoop, j * VL_FP32 + i * curColNumAlign);
                if constexpr (hasClampValue) {
                    Mins(x0, x0, clampValue, pregLoop);
                    Maxs(x1, x1, -clampValue, pregLoop);
                    Mins(x1, x1, clampValue, pregLoop);
                }
                VFSwiGlu(y, x0, x1, one, tmp, pregLoop);
                if constexpr (hasTopkWeight) {
                    Mul(y, y, weight, pregLoop);
                }
                StoreOutputData<T>(yLocalAddr, y, pregLoop, j * VL_FP32 + i * curColNumAlign);
            }
        }
    }
}


template <typename T>
__aicore__ inline void VFComputeMaxExp(const LocalTensor<uint16_t>& maxExpLocal, const LocalTensor<T>& xLocal, uint16_t curRowNum,
                                        uint32_t curColNum)
{
    __local_mem__ uint16_t* maxExpOriginLocalAddr = (__local_mem__ uint16_t*)maxExpLocal.GetPhyAddr();
    __local_mem__ T* xOriginLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ uint16_t* maxExpLocalAddr = maxExpOriginLocalAddr;
    __local_mem__ T* xLocalAddr = xOriginLocalAddr;
    uint16_t vlForT = 256 / sizeof(T);
    uint16_t loopCount = CeilDiv(curColNum, vlForT * 2);
    uint32_t numVRegBlocks = 8;
    uint32_t yCurColNumAlign = RoundUp<uint16_t>(CeilDiv(curColNum, 32));
    uint32_t curColNumAlign = RoundUp<T>(curColNum);
    __VEC_SCOPE__
    {
        // 用于把float16转为bfloat16，不涉及宽度变化
        static constexpr CastTrait traitFP16ToBF16 = {RegLayout::UNKNOWN, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                  RoundMode::CAST_TRUNC};
        // 0存奇数位元素，1存偶数位元素
        RegTensor<T> x0, x1;
        RegTensor<bfloat16_t> x0BF16, x1BF16;
        RegTensor<uint16_t> exp0, exp1, exp0FP16, exp1FP16, maxExp;
        // 存储FP16/BF16的指数位为1的mask
        RegTensor<uint16_t> emaskFP16, emaskBF16;
        Duplicate(emaskFP16, FP16_EMASK_AND_INF_VAL);
        Duplicate(emaskBF16, BF16_EMASK_AND_INF_VAL);
        // 2字节Reg的MaskALL
        MaskReg maskAllB16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg mask0, mask1, mask0FP16NanInf, mask1FP16NanInf;
        // 非对齐搬出至UB用
        UnalignReg uReg;
        for (uint16_t i = 0; i < curRowNum; i++) {
            uint32_t sreg0 = curColNum;
            uint32_t sreg1 = curColNum;
            maxExpLocalAddr = maxExpOriginLocalAddr + i * yCurColNumAlign;
            xLocalAddr = xOriginLocalAddr + i * curColNumAlign;
            for (uint16_t j = 0; j < loopCount; j++) {
                mask0 = UpdateMask<T>(sreg0);
                mask1 = UpdateMask<T>(sreg1);
                MaskDeInterleave<T>(mask0, mask1, mask0, mask1);
                DataCopy<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(x0, x1, xLocalAddr, vlForT * 2);
                if constexpr (IsSameType<T, half>::value) {
                    And(exp0FP16, (RegTensor<uint16_t> &)x0, emaskFP16, mask0);
                    And(exp1FP16, (RegTensor<uint16_t> &)x1, emaskFP16, mask1);
                    Compare<uint16_t, CMPMODE::EQ>(mask0FP16NanInf, exp0FP16, emaskFP16, mask0);
                    Compare<uint16_t, CMPMODE::EQ>(mask1FP16NanInf, exp1FP16, emaskFP16, mask1);
                    Cast<bfloat16_t, T, traitFP16ToBF16>(x0BF16, x0, mask0);
                    Cast<bfloat16_t, T, traitFP16ToBF16>(x1BF16, x1, mask1);
                    And(exp0, (RegTensor<uint16_t> &)x0BF16, emaskBF16, mask0);
                    And(exp1, (RegTensor<uint16_t> &)x1BF16, emaskBF16, mask1);
                    Select(exp0, emaskBF16, exp0, mask0FP16NanInf);
                    Select(exp1, emaskBF16, exp1, mask1FP16NanInf);
                } else {
                    And(exp0, (RegTensor<uint16_t> &)x0, emaskBF16, mask0);
                    And(exp1, (RegTensor<uint16_t> &)x1, emaskBF16, mask1);
                }
                Max(maxExp, exp0, exp1, mask0);
                ReduceMaxWithDataBlock(maxExp, maxExp, maskAllB16);
                DataCopyUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(maxExpLocalAddr, maxExp, uReg, numVRegBlocks);
            }
            DataCopyUnAlignPost(maxExpLocalAddr, uReg, 0);
        }
    }
}


__aicore__ inline void VFComputeScale(const LocalTensor<uint16_t>& mxScaleLocal, const LocalTensor<uint16_t>& invScaleLocal,
                                      const LocalTensor<uint16_t>& maxExpLocal, uint16_t curRowNum, uint16_t curColNum,
                                      uint32_t validCurColNum, uint16_t expLowerBoundValue)
{
    __local_mem__ uint16_t* mxScaleOriginLocalAddr = (__local_mem__ uint16_t*)mxScaleLocal.GetPhyAddr();
    __local_mem__ uint16_t* invScaleOriginLocalAddr = (__local_mem__ uint16_t*)invScaleLocal.GetPhyAddr();
    __local_mem__ uint16_t* maxExpOriginLocalAddr = (__local_mem__ uint16_t*)maxExpLocal.GetPhyAddr();
    __local_mem__ uint16_t* mxScaleLocalAddr = mxScaleOriginLocalAddr;
    __local_mem__ uint16_t* invScaleLocalAddr = invScaleOriginLocalAddr;
    __local_mem__ uint16_t* maxExpLocalLocalAddr = maxExpOriginLocalAddr;
    uint16_t vlForT = 256 / sizeof(uint16_t);
    uint16_t loopCount = CeilDiv(curColNum, vlForT);
    uint32_t numVRegBlocks = 8;
    uint32_t scaleCurColNumAlign = RoundUp<uint8_t>(validCurColNum) / 2;
    uint32_t invCurColNumAlign = RoundUp<uint16_t>(validCurColNum);
    __VEC_SCOPE__
    {
        RegTensor<uint16_t> maxExp, sharedExp, mxScale, invScale;
        RegTensor<uint16_t> infBF16;
        Duplicate(infBF16, BF16_EMASK_AND_INF_VAL);
        RegTensor<uint16_t> zeroB16;
        Duplicate(zeroB16, 0);
        RegTensor<uint16_t> expLowerBound;
        Duplicate(expLowerBound, expLowerBoundValue);
        RegTensor<uint16_t> nanForE8M0;
        Duplicate(nanForE8M0, FP8_E8M0_NAN_VAL);
        RegTensor<uint16_t> invSub;
        Duplicate(invSub, BF16_EXP_INVSUB);
        RegTensor<uint16_t> nanBF16;
        Duplicate(nanBF16, BF16_NAN_VAL);
        RegTensor<uint16_t> specialMinE8M0;
        Duplicate(specialMinE8M0, FP8_E8M0_SPECIAL_MIN);

        MaskReg maskLoop, maskValid;
        MaskReg maskInfBF16, maskZero, maskLowExp, maskSpecialMin;
        for (uint16_t i = 0; i < curRowNum; i++) {
            uint32_t sreg0 = curColNum;
            uint32_t sreg1 = validCurColNum;
            mxScaleLocalAddr = mxScaleOriginLocalAddr + i * scaleCurColNumAlign;
            invScaleLocalAddr = invScaleOriginLocalAddr + i * invCurColNumAlign;
            maxExpLocalLocalAddr = maxExpOriginLocalAddr + i * invCurColNumAlign;
            for (uint16_t j = 0; j < loopCount; j++) {
                maskLoop = UpdateMask<uint16_t>(sreg0);
                maskValid = UpdateMask<uint16_t>(sreg1);
                DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(maxExp, maxExpLocalLocalAddr, vlForT);
                Compare<uint16_t, CMPMODE::LT>(maskLowExp, maxExp, expLowerBound, maskValid);
                Select<uint16_t>(maxExp, expLowerBound, maxExp, maskLowExp);
                Sub(sharedExp, maxExp, expLowerBound, maskValid);
                ShiftRights(mxScale, sharedExp, BF16_EXP_SHR_BITS, maskValid);
                Compare<uint16_t, CMPMODE::EQ>(maskInfBF16, maxExp, infBF16, maskValid);
                Select<uint16_t>(mxScale, nanForE8M0, mxScale, maskInfBF16);
                Compare<uint16_t, CMPMODE::EQ>(maskZero, maxExp, zeroB16, maskValid);
                Select<uint16_t>(mxScale, zeroB16, mxScale, maskZero);
                DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, mxScale, vlForT / 2,
                                                                                            maskLoop);
                Sub<uint16_t>(invScale, invSub, sharedExp, maskValid);
                Select<uint16_t>(invScale, nanBF16, invScale, maskInfBF16);
                Select<uint16_t>(invScale, zeroB16, invScale, maskZero);
                Compare<uint16_t, CMPMODE::EQ>(maskSpecialMin, invSub, sharedExp, maskValid);
                Select<uint16_t>(invScale, specialMinE8M0, invScale, maskSpecialMin);
                DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(invScaleLocalAddr, invScale, vlForT, maskLoop);
            }

        }
    }
}

template <typename T, typename U>
__aicore__ inline void VFComputeData(const LocalTensor<U>& xQuantLocal,  const LocalTensor<T>& xLocal, const LocalTensor<uint16_t>& invScaleLocal,
                                      uint16_t curRowNum, uint32_t curColNum)
{
    __local_mem__ U* xQuantOriginLocalAddr = (__local_mem__ U*)xQuantLocal.GetPhyAddr();
    __local_mem__ T* xOriginLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ uint16_t* invScaleOriginLocalAddr = (__local_mem__ uint16_t*)invScaleLocal.GetPhyAddr();
    __local_mem__ U* xQuantLocalAddr = xQuantOriginLocalAddr;
    __local_mem__ T* xLocalAddr = xOriginLocalAddr;
    __local_mem__ uint16_t* invScaleLocalAddr = invScaleOriginLocalAddr;
    uint16_t vlForT = 256 / sizeof(T);
    uint16_t loopCount = CeilDiv(curColNum, vlForT * 2);
    uint32_t numVRegBlocks = 8;
    uint32_t xCurColNumAlign = RoundUp<T>(curColNum);
    uint32_t scaleCurColNumAlign = RoundUp<uint16_t>(CeilDiv(curColNum, 32));
    uint32_t xQuantCurColNumAlign = RoundUp<U>(curColNum);
    __VEC_SCOPE__
    {
        static constexpr CastTrait traitFP16ToBF16 = {RegLayout::UNKNOWN, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                    RoundMode::CAST_TRUNC};
        static constexpr CastTrait traitB16ToB32Layout0 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                        RoundMode::UNKNOWN};
        static constexpr CastTrait traitB16ToB32Layout1 = {RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                        RoundMode::UNKNOWN};
        static constexpr CastTrait traitB32ToB8Layout0 = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                        RoundMode::CAST_RINT};

        RegTensor<uint16_t> invScale;
        RegTensor<float> invScaleFP32;
        RegTensor<T> x0, x1;
        RegTensor<float> x0FP32Layout0, x0FP32Layout1, x1FP32Layout0, x1FP32Layout1;
        RegTensor<U> xQuant0, xQuant1, xQuant2, xQuant3;

        MaskReg maskXQuant0B32, maskXQuant1B32, maskXQuant2B32, maskXQuant3B32;
        MaskReg maskAllB16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg maskAllB32 = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t i = 0; i < curRowNum; i++) {
            uint32_t sreg0 = curColNum;
            uint32_t sreg1 = curColNum;
            uint32_t sreg2 = curColNum;
            uint32_t sreg3 = curColNum;
            xLocalAddr = xOriginLocalAddr + i * xCurColNumAlign;
            invScaleLocalAddr = invScaleOriginLocalAddr + i * scaleCurColNumAlign;
            xQuantLocalAddr = xQuantOriginLocalAddr + i * xQuantCurColNumAlign;
            for (uint16_t i = 0; i < loopCount; i++) {
                maskXQuant0B32 = UpdateMask<float>(sreg0);
                maskXQuant1B32 = UpdateMask<float>(sreg1);
                maskXQuant2B32 = UpdateMask<float>(sreg2);
                maskXQuant3B32 = UpdateMask<float>(sreg3);

                DataCopy<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(x0, x1, xLocalAddr, vlForT * 2);
                DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(invScale, invScaleLocalAddr,
                                                                                          numVRegBlocks);
                if constexpr (IsSameType<T, half>::value) {
                    Cast<float, bfloat16_t, traitB16ToB32Layout0>(invScaleFP32, (RegTensor<bfloat16_t> &)invScale, maskAllB16);
                    Cast<float, T, traitB16ToB32Layout0>(x0FP32Layout0, x0, maskAllB16);
                    Cast<float, T, traitB16ToB32Layout1>(x0FP32Layout1, x0, maskAllB16);
                    Mul(x0FP32Layout0, x0FP32Layout0, invScaleFP32, maskAllB32);
                    Mul(x0FP32Layout1, x0FP32Layout1, invScaleFP32, maskAllB32);
                    Interleave(x0FP32Layout0, x0FP32Layout1, x0FP32Layout0, x0FP32Layout1);

                    // 2.对偶数位元素x1进行量化，与x0的做法一致：
                    Cast<float, T, traitB16ToB32Layout0>(x1FP32Layout0, x1, maskAllB16);
                    Cast<float, T, traitB16ToB32Layout1>(x1FP32Layout1, x1, maskAllB16);
                    Mul(x1FP32Layout0, x1FP32Layout0, invScaleFP32, maskAllB32);
                    Mul(x1FP32Layout1, x1FP32Layout1, invScaleFP32, maskAllB32);
                    Interleave(x1FP32Layout0, x1FP32Layout1, x1FP32Layout0, x1FP32Layout1);

                    Interleave(x0FP32Layout0, x1FP32Layout0, x0FP32Layout0, x1FP32Layout0);
                    Interleave(x0FP32Layout1, x1FP32Layout1, x0FP32Layout1, x1FP32Layout1);

                    Cast<U, float, traitB32ToB8Layout0>(xQuant0, x0FP32Layout0, maskXQuant0B32);
                    Cast<U, float, traitB32ToB8Layout0>(xQuant1, x1FP32Layout0, maskXQuant1B32);
                    Cast<U, float, traitB32ToB8Layout0>(xQuant2, x0FP32Layout1, maskXQuant2B32);
                    Cast<U, float, traitB32ToB8Layout0>(xQuant3, x1FP32Layout1, maskXQuant3B32);
                } else {
                    Mul(x0, x0, (RegTensor<T> &)invScale, maskAllB16);
                    Mul(x1, x1, (RegTensor<T> &)invScale, maskAllB16);
                    Interleave(x0, x1, x0, x1);
                    Cast<float, T, traitB16ToB32Layout0>(x0FP32Layout0, x0, maskAllB16);
                    Cast<float, T, traitB16ToB32Layout1>(x0FP32Layout1, x0, maskAllB16);
                    Interleave(x0FP32Layout0, x0FP32Layout1, x0FP32Layout0, x0FP32Layout1);
                    Cast<U, float, traitB32ToB8Layout0>(xQuant0, x0FP32Layout0, maskXQuant0B32);
                    Cast<U, float, traitB32ToB8Layout0>(xQuant1, x0FP32Layout1, maskXQuant1B32);
                    Cast<float, T, traitB16ToB32Layout0>(x1FP32Layout0, x1, maskAllB16);
                    Cast<float, T, traitB16ToB32Layout1>(x1FP32Layout1, x1, maskAllB16);
                    Interleave(x1FP32Layout0, x1FP32Layout1, x1FP32Layout0, x1FP32Layout1);
                    Cast<U, float, traitB32ToB8Layout0>(xQuant2, x1FP32Layout0, maskXQuant2B32);
                    Cast<U, float, traitB32ToB8Layout0>(xQuant3, x1FP32Layout1, maskXQuant3B32);
                }

                DataCopy<U, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(xQuantLocalAddr, xQuant0, OUT_ELE_NUM_ONE_BLK, maskXQuant0B32);
                DataCopy<U, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(xQuantLocalAddr, xQuant1, OUT_ELE_NUM_ONE_BLK, maskXQuant1B32);
                DataCopy<U, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(xQuantLocalAddr, xQuant2, OUT_ELE_NUM_ONE_BLK, maskXQuant2B32);
                DataCopy<U, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(xQuantLocalAddr, xQuant3, OUT_ELE_NUM_ONE_BLK, maskXQuant3B32);
            }
        }
    }
}

template <typename T, bool withUbReduce = false>
__aicore__ inline void VFProcessGroupIndex(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal, uint16_t curColNum)
{
    __local_mem__ T* yLocalAddr = (__local_mem__ T*)yLocal.GetPhyAddr();
    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    uint16_t vlLen = REPEAT_SIZE / sizeof(T);
    uint16_t loopCount = CeilDiv(curColNum, vlLen);
    uint16_t fourLoopCount = loopCount / FOUR_UNFOLD;
    uint16_t tailLoopNum = loopCount % FOUR_UNFOLD;
    uint32_t tailReminder = curColNum - fourLoopCount * vlLen * FOUR_UNFOLD;
    if (loopCount < FOUR_UNFOLD) {
        __VEC_SCOPE__
        {
            RegTensor<T> x;
            RegTensor<T> sum;
            MaskReg pregMain = CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<T, AscendC::MicroAPI::MaskPattern::VL1>();
            Duplicate(sum, static_cast<T>(0), pregMain);
            uint32_t sreg = curColNum;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = UpdateMask<T>(sreg);
                DataCopy(x, xLocalAddr + i * vlLen);
                Adds(x, x, static_cast<T>(0), pregLoop);
                Add(sum, sum, x, pregMain);
            }
            ReduceSum(sum, sum, pregMain);
            if (withUbReduce) {
                RegTensor<T> origin;
                DataCopy(origin, yLocalAddr);
                Add(sum, sum, origin, pregMerge);
            }
            DataCopy(yLocalAddr, sum, pregMerge);
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<T> x0;
            RegTensor<T> x1;
            RegTensor<T> x2;
            RegTensor<T> x3;
            RegTensor<T> sum0;
            RegTensor<T> sum1;
            RegTensor<T> sum2;
            RegTensor<T> sum3;
            MaskReg pregMain = CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<T, AscendC::MicroAPI::MaskPattern::VL1>();
            Duplicate(sum0, static_cast<T>(0), pregMain);
            Duplicate(sum1, static_cast<T>(0), pregMain);
            Duplicate(sum2, static_cast<T>(0), pregMain);
            Duplicate(sum3, static_cast<T>(0), pregMain);
            MaskReg pregLoop;
            for (uint16_t i = 0; i < fourLoopCount; i++) {
                DataCopy(x0, xLocalAddr + i * FOUR_UNFOLD * vlLen);
                Add(sum0, sum0, x0, pregMain);
                DataCopy(x1, xLocalAddr + (i * FOUR_UNFOLD + 1) * vlLen);
                Add(sum1, sum1, x1, pregMain);
                DataCopy(x2, xLocalAddr + (i * FOUR_UNFOLD + 2) * vlLen);
                Add(sum2, sum2, x2, pregMain);
                DataCopy(x3, xLocalAddr + (i * FOUR_UNFOLD + 3) * vlLen);
                Add(sum3, sum3, x3, pregMain);
            }
            uint32_t sreg = tailReminder;
            for (uint16_t i = 0; i < tailLoopNum; i++) {
                pregLoop = UpdateMask<T>(sreg);
                DataCopy(x0, xLocalAddr + (fourLoopCount * FOUR_UNFOLD + i) * vlLen);
                Adds(x0, x0, static_cast<T>(0), pregLoop);
                Add(sum0, sum0, x0, pregMain);
            }
            Add(sum0, sum0, sum1, pregMain);
            Add(sum2, sum2, sum3, pregMain);
            Add(sum0, sum0, sum2, pregMain);
            ReduceSum(sum0, sum0, pregMain);
            if (withUbReduce) {
                RegTensor<T> origin;
                DataCopy(origin, yLocalAddr);
                Add(sum0, sum0, origin, pregMerge);
            }
            DataCopy(yLocalAddr, sum0, pregMerge);
        }
    }
}

// curColNum一定能被128整除
template <typename T0, typename T1, typename T2, bool hasRoundScale = false, bool hasClampValue = false, bool hasOutput = false, bool hasTopkWeight = false>
__aicore__ inline void VFProcessSwigluFp8QuantPerToken(const LocalTensor<T0> &yLocal, const LocalTensor<T1>& yOriginLocal,
                                                      const LocalTensor<T2> &scaleLocal, const LocalTensor<T1> &x0Local,
                                                      const LocalTensor<T1> &x1Local, const LocalTensor<float> &topkWeightLocal,
                                                      float clampValue, const uint16_t curRowNum, const uint32_t curColNum)
{
    __local_mem__ T0* yLocalAddr = (__local_mem__ T0*)yLocal.GetPhyAddr();
    __local_mem__ T1* yOriginLocalAddr = hasOutput ? (__local_mem__ T1*)yOriginLocal.GetPhyAddr() : nullptr;
    __local_mem__ T2* scaleLocalAddr = (__local_mem__ T2*)scaleLocal.GetPhyAddr();
    __local_mem__ T1* x0LocalAddr = (__local_mem__ T1*)x0Local.GetPhyAddr();
    __local_mem__ T1* x1LocalAddr = (__local_mem__ T1*)x1Local.GetPhyAddr();
    __local_mem__ float* topkWeightLocalAddr = hasTopkWeight ? (__local_mem__ float*)topkWeightLocal.GetPhyAddr() : nullptr;

    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<T1>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T0>(curColNum);
    uint32_t scaleColNum = CeilDiv(curColNum, PER_BLOCK_FP16);
    uint16_t loopCountFoldTwo = loopCount / 2;
    uint16_t loopCountReminder = loopCount % 2;
    uint32_t tailRemider = curColNum - (loopCount - 1) * VL_FP32;

    __VEC_SCOPE__
    {
        RegTensor<float> weight;
        RegTensor<float> one;
        RegTensor<uint32_t> oneUint32;
        RegTensor<float> zero;
        RegTensor<uint32_t> zeroUint32;
        RegTensor<float> xLeft;
        RegTensor<float> xRight;
        RegTensor<float> x0Left;
        RegTensor<float> x0Right;
        RegTensor<float> x1Left;
        RegTensor<float> x1Right;
        RegTensor<float> xAbsLeft;
        RegTensor<float> xAbsRight;
        RegTensor<float> tmp;
        RegTensor<uint32_t> tmp0;
        RegTensor<uint32_t> tmp1;
        RegTensor<uint32_t> tmp2;
        RegTensor<uint32_t> tmp3;
        RegTensor<int32_t> tmp4;
        RegTensor<float> dupScale;
        RegTensor<float> scale;
        RegTensor<float> clampScale;
        RegTensor<float> invScale;
        RegTensor<float> dupInvScale;
        RegTensor<float> scale0;
        RegTensor<float> scale1;
        RegTensor<uint32_t> scale2;
        RegTensor<uint32_t> scale3;
        RegTensor<int32_t> scale4;
        RegTensor<int32_t> scale5;
        RegTensor<uint32_t> coeff;
        RegTensor<float> invCoeff;
        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        MaskReg compareLeft;
        MaskReg compareRight;
        MaskReg compareMask0;
        Duplicate(zero, 0.0f, pregMain);
        Duplicate(zeroUint32, static_cast<uint32_t>(0), pregMain);
        Duplicate(one, 1.0f, pregMain);
        Duplicate(oneUint32, static_cast<uint32_t>(1), pregMain);
        Duplicate(coeff, INV_FP8_E4M3_MAX_VALUE, pregMerge);
        Duplicate(invCoeff, 448.0f, pregMerge);
        Duplicate(tmp0, FAST_LOG_AND_VALUE1, pregMerge);
        Duplicate(tmp1, FAST_LOG_AND_VALUE2, pregMerge);
        Duplicate(tmp3, static_cast<uint32_t>(127), pregMerge);
        Duplicate(tmp4, static_cast<int32_t>(127), pregMerge);
        for (uint16_t i = 0; i < curRowNum; i++) {
            if constexpr (hasTopkWeight) {
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(weight, topkWeightLocalAddr + i);
            }
            for (uint16_t j = 0; j < loopCountFoldTwo; j++) {
                LoadInputData<T1>(x0Left, x0LocalAddr, pregMain, 2 * j * VL_FP32 + i * curColNumAlign);
                LoadInputData<T1>(x0Right, x0LocalAddr, pregMain, (2 * j + 1) * VL_FP32 + i * curColNumAlign);
                LoadInputData<T1>(x1Left, x1LocalAddr, pregMain, 2 * j * VL_FP32 + i * curColNumAlign);
                LoadInputData<T1>(x1Right, x1LocalAddr, pregMain, (2 * j + 1) * VL_FP32 + i * curColNumAlign);
                if constexpr (hasClampValue) {
                    Mins(x0Left, x0Left, clampValue, pregMain);
                    Mins(x0Right, x0Right, clampValue, pregMain);
                    Maxs(x1Left, x1Left, -clampValue, pregMain);
                    Mins(x1Left, x1Left, clampValue, pregMain);
                    Maxs(x1Right, x1Right, -clampValue, pregMain);
                    Mins(x1Right, x1Right, clampValue, pregMain);
                }
                VFSwiGlu(xLeft, x0Left, x1Left, one, tmp, pregMain);
                VFSwiGlu(xRight, x0Right, x1Right, one, tmp, pregMain);
                if constexpr (hasTopkWeight) {
                    Mul(xLeft, xLeft, weight, pregMain);
                }
                Add(xLeft, xLeft, zero, pregMain);
                if constexpr (hasTopkWeight) {
                    Mul(xRight, xRight, weight, pregMain);
                }
                Add(xRight, xRight, zero, pregMain);
                if constexpr (hasOutput) {
                    StoreOutputData<T1>(yOriginLocalAddr, xLeft, pregMain, 2 * j * VL_FP32 + i * curColNumAlign);
                    StoreOutputData<T1>(yOriginLocalAddr, xRight, pregMain, (2 * j + 1) * VL_FP32 + i * curColNumAlign);
                }
                Muls(xAbsLeft, xLeft, 0.0f, pregMain);
                Compare<float, CMPMODE::NE>(compareLeft, xAbsLeft, xAbsLeft, pregMain);
                MaskNot(compareLeft, compareLeft, pregMain);
                Abs(xAbsLeft, xLeft, compareLeft);
                ReduceMax(scale0, xAbsLeft, pregMain);

                Muls(xAbsRight, xRight, 0.0f, pregMain);
                Compare<float, CMPMODE::NE>(compareRight, xAbsRight, xAbsRight, pregMain);
                MaskNot(compareRight, compareRight, pregMain);
                Abs(xAbsRight, xRight, compareRight);
                ReduceMax(scale1, xAbsRight, pregMain);
                Max(scale, scale0, scale1, pregMerge);
                Maxs(clampScale, scale, 0.0001f, pregMerge); // amax

                Mul(scale, clampScale, (RegTensor<float>&)coeff, pregMerge); // sf = amax / 448.0
                if constexpr (!hasRoundScale) {
                    Div(invScale, invCoeff, clampScale, pregMerge);
                    DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((__local_mem__ float*)scaleLocalAddr + j + i * scaleColNum, scale, pregMerge); // copy out scale
                } else {
                    // bits == (RegTensor<uint32_t>&)scale
                    ShiftRights(scale2, (RegTensor<uint32_t>&)scale, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), pregMerge);
                    And(scale2, scale2, tmp0, pregMerge); //exp
                    And(scale3, (RegTensor<uint32_t>&)scale, tmp1, pregMerge); // man_bits
                    Compare<uint32_t, AscendC::CMPMODE::NE>(compareMask0, scale3, zeroUint32, pregMerge);
                    Select(tmp2, oneUint32, zeroUint32, compareMask0); // man_bits != 0
                    Sub(scale3, scale2, tmp3, pregMerge); // exp - 127
                    Add((RegTensor<uint32_t>&)scale4, scale3, tmp2, pregMerge); // exp_scale-uint32 = exp - 127 + (man_bits != 0)
                    if constexpr (IsSameType<T2, fp8_e8m0_t>::value) {
                        Adds(scale5, scale4, 127, pregMerge); // sf_uint32
                        StoreMxFp8Scale<T2>(scaleLocalAddr, scale5, pregMerge, i * scaleColNum + j); // copy out scale
                    } else {
                        Adds(scale5, scale4, 127, pregMerge);
                        ShiftLefts((RegTensor<int32_t>&)scale, scale5, static_cast<int16_t>(23), pregMerge);
                        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((__local_mem__ float*)scaleLocalAddr + j + i * scaleColNum, scale, pregMerge); // copy out scale
                    }
                    Sub(scale5, tmp4, scale4, pregMerge); // 127 - exp_scale
                    ShiftLefts((RegTensor<int32_t>&)invScale, scale5, static_cast<int16_t>(23), pregMerge); // ((127 - exp_scale) << 23).view(float32)
                }
                Duplicate(dupInvScale, invScale, pregMain);

                Mul(x0Left, xLeft, dupInvScale, pregMain);
                Muls(x1Left, x0Left, 0.0f, pregMain);
                Compare<float, CMPMODE::NE>(compareLeft, x1Left, x1Left, pregMain);
                Select(xLeft, xLeft, x0Left, compareLeft);
                Mul(x0Right, xRight, dupInvScale, pregMain);
                Muls(x1Right, x0Right, 0.0f, pregMain);
                Compare<float, CMPMODE::NE>(compareRight, x1Right, x1Right, pregMain);
                Select(xRight, xRight, x0Right, compareRight);
                StoreOutputData<T0>(yLocalAddr, xLeft, pregMain, 2 * j * VL_FP32 + i * dstCurColNumAlign);
                StoreOutputData<T0>(yLocalAddr, xRight, pregMain, (2 * j + 1) * VL_FP32 + i * dstCurColNumAlign);
            }
        }
    }

}

template <typename T0, typename T1, typename T2>
__aicore__ inline void Fp8QuantPerTokenDispatcher(const LocalTensor<T0> &yLocal, const LocalTensor<T1>& yOriginLocal,
                                                      const LocalTensor<T2> &scaleLocal, const LocalTensor<T1> &x0Local,
                                                      const LocalTensor<T1> &x1Local, const LocalTensor<float> &topkWeightLocal,
                                                      float clampValue, const uint16_t curRowNum, const uint32_t curColNum, int32_t maskBit)
{
    if (maskBit == 0b000) {
        VFProcessSwigluFp8QuantPerToken<T0, T1, T2, false, false, false, false>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b001) {
        VFProcessSwigluFp8QuantPerToken<T0, T1, T2, false, false, false, true>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b010) {
        VFProcessSwigluFp8QuantPerToken<T0, T1, T2, false, true, false, false>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b011) {
        VFProcessSwigluFp8QuantPerToken<T0, T1, T2, false, true, false, true>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b100) {
        VFProcessSwigluFp8QuantPerToken<T0, T1, T2, true, false, false, false>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b101) {
        VFProcessSwigluFp8QuantPerToken<T0, T1, T2, true, false, false, true>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b110) {
        VFProcessSwigluFp8QuantPerToken<T0, T1, T2, true, true, false, false>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b111) {
        VFProcessSwigluFp8QuantPerToken<T0, T1, T2, true, true, false, true>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    }
}

template <typename T0, typename T1, typename T2>
__aicore__ inline void Fp8QuantPerTokenDispatcherYOrigin(const LocalTensor<T0> &yLocal, const LocalTensor<T1>& yOriginLocal,
                                                      const LocalTensor<T2> &scaleLocal, const LocalTensor<T1> &x0Local,
                                                      const LocalTensor<T1> &x1Local, const LocalTensor<float> &topkWeightLocal,
                                                      float clampValue, const uint16_t curRowNum, const uint32_t curColNum, int32_t maskBit)
{
    if (maskBit == 0b000) {
        VFProcessSwigluFp8QuantPerToken<T0, T1, T2, false, false, true, false>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b001) {
        VFProcessSwigluFp8QuantPerToken<T0, T1, T2, false, false, true, true>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b010) {
        VFProcessSwigluFp8QuantPerToken<T0, T1, T2, false, true, true, false>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b011) {
        VFProcessSwigluFp8QuantPerToken<T0, T1, T2, false, true, true, true>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b100) {
        VFProcessSwigluFp8QuantPerToken<T0, T1, T2, true, false, true, false>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b101) {
        VFProcessSwigluFp8QuantPerToken<T0, T1, T2, true, false, true, true>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b110) {
        VFProcessSwigluFp8QuantPerToken<T0, T1, T2, true, true, true, false>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
    } else if (maskBit == 0b111) {
        VFProcessSwigluFp8QuantPerToken<T0, T1, T2, true, true, true, true>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local, topkWeightLocal, clampValue, curRowNum, curColNum);
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

} // namespace SwigluGroupQuant

#endif