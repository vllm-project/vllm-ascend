/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HC_PRE_SINKHORN_RGEBASE_BASE_H
#define HC_PRE_SINKHORN_RGEBASE_BASE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace HcPreNs {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UnalignReg;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t VL_FP32 = 64;
constexpr int32_t C0_SIZE = 8;
constexpr int32_t FOUR_UNFOLD = 4;
constexpr int32_t DOUBLE_BUFFER = 2;
constexpr MatmulConfig MM_CFG = GetMDLConfig();

__aicore__ inline uint64_t Align(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t CeilAlign(uint64_t a, uint64_t b)
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

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

template <typename T>
__aicore__ inline void LoadInputData(RegTensor<float> &dst, __local_mem__ T *src, MaskReg pregLoop, uint32_t srcOffset)
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
__aicore__ inline void StoreOutputData(__local_mem__ T *dst, RegTensor<float> &src, MaskReg pregLoop,
                                       uint32_t dstOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dst + dstOffset, src, pregLoop);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitB322B16Even>(tmp, src, pregLoop);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + dstOffset, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void LoadInputDataWithBrc(RegTensor<float> &dst, __local_mem__ T *src, MaskReg pregLoop,
                                            uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(dst, src + srcOffset);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void LoadInputDataUnalign(
    RegTensor<float>& dst, __local_mem__ T*& src, UnalignReg& uSrc, MaskReg pregLoop, uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopyUnAlign(dst, uSrc, src, postUpdateStride);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        RegTensor<T> tmpUnPack;
        DataCopyUnAlign(tmp, uSrc, src, postUpdateStride);
        UnPack((RegTensor<uint32_t>&)tmpUnPack, (RegTensor<uint16_t>&)tmp);
        Cast<float, T, castTraitB162B32Even>(dst, tmpUnPack, pregLoop);
    }
}


__aicore__ inline void VFSigmoid(RegTensor<float> &y, RegTensor<float> &x, RegTensor<float> &one, MaskReg pregLoop)
{
    Muls(x, x, static_cast<float>(-1), pregLoop);
    Exp(x, x, pregLoop);
    Adds(x, x, static_cast<float>(1), pregLoop);
    Div(y, one, x, pregLoop);
}

__aicore__ inline void VFTransND2NZ(const LocalTensor<float> &yLocal, const LocalTensor<float> &xLocal,
                                    const uint16_t curRowNum, const uint16_t curColNum)
{
    __local_mem__ float *yLocalAddr = (__local_mem__ float *)yLocal.GetPhyAddr();
    __local_mem__ float *xLocalAddr = (__local_mem__ float *)xLocal.GetPhyAddr();
    // NZ分享要求M,N方向按照16对齐
    uint16_t curRowNumAlign = CeilAlign(curRowNum, C0_SIZE);
    uint16_t c1Size = BLOCK_SIZE / sizeof(float);
    uint16_t curColNumAlign = RoundUp<float>(curColNum);
    uint16_t curRowMainCount = curRowNum / C0_SIZE;
    uint16_t curRowReminder = curRowNum % C0_SIZE;
    uint16_t tailBaseOffset = curRowReminder * c1Size;
    uint32_t dataBlockStride = curColNumAlign / c1Size + 1;
    uint16_t loopCount = curColNumAlign / c1Size;
    if (curRowReminder == 0) {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            MaskReg pregMain = CreateMask<float>();
            for (uint16_t i = 0; i < curRowMainCount; i++) {
                for (uint16_t j = 0; j < loopCount; j++) {
                    DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        x, xLocalAddr + i * C0_SIZE * (curColNumAlign + BLOCK_SIZE / sizeof(float)) + j * c1Size, dataBlockStride, pregMain);
                    DataCopy(yLocalAddr + i * C0_SIZE * c1Size + j * curRowNumAlign * c1Size, x, pregMain);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            MaskReg pregMain = CreateMask<float>();
            uint32_t sreg = curRowReminder * C0_SIZE;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            for (uint16_t i = 0; i < curRowMainCount; i++) {
                for (uint16_t j = 0; j < loopCount; j++) {
                    DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        x, xLocalAddr + i * C0_SIZE * (curColNumAlign + BLOCK_SIZE / sizeof(float)) + j * c1Size, dataBlockStride, pregMain);
                    DataCopy(yLocalAddr + i * C0_SIZE * c1Size + j * curRowNumAlign * c1Size, x, pregMain);
                }
            }
            xLocalAddr = xLocalAddr + curRowMainCount * C0_SIZE * (curColNumAlign + BLOCK_SIZE / sizeof(float));
            yLocalAddr = yLocalAddr + curRowMainCount * C0_SIZE * c1Size;
            for (uint16_t i = 0; i < loopCount; i++) {
                DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(x, xLocalAddr + i * c1Size,
                                                                                  dataBlockStride, pregLoop);
                DataCopy(yLocalAddr + i * curRowNumAlign * c1Size, x, pregLoop);
            }
        }
    }
}

template <typename T>
__aicore__ inline void VFProcessCast(const LocalTensor<float> &yLocal, const LocalTensor<T> &xLocal,
                                      const uint16_t curRowNum, const uint16_t curColNum)
{
    __local_mem__ float *yLocalAddr = (__local_mem__ float *)yLocal.GetPhyAddr();
    __local_mem__ T *xLocalAddr = (__local_mem__ T *)xLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint16_t curColNumAlign = RoundUp<T>(curColNum);
    uint16_t dstCurColNumAlign = RoundUp<float>(curColNum) + BLOCK_SIZE / sizeof(float);
    if (loopCount > 1) {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            MaskReg pregLoop;
            uint32_t sreg;
            for (uint16_t i = 0; i < curRowNum; i++) {
                sreg = curColNum;
                for (uint16_t j = 0; j < loopCount; j++) {
                    pregLoop = UpdateMask<float>(sreg);
                    LoadInputData(x, xLocalAddr, pregLoop, i * curColNumAlign + j * VL_FP32);
                    StoreOutputData(yLocalAddr, x, pregLoop, i * dstCurColNumAlign + j * VL_FP32);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            MaskReg pregLoop = CreateMask<float>();
            for (uint16_t i = 0; i < curRowNum; i++) {
                LoadInputData(x, xLocalAddr, pregLoop, i * curColNumAlign);
                StoreOutputData(yLocalAddr, x, pregLoop, i * dstCurColNumAlign);
            }
        }
    }

}

template <typename T, bool WithUbReduce = false>
__aicore__ inline void VFProcessCastAndInvRmsPart1(const LocalTensor<float> &rmsNormLocal, const LocalTensor<float> &xCastLocal,
                                                   const LocalTensor<T> &xLocal, float coeff,
                                                   const uint16_t curRowNum, const uint16_t curColNum)
{
    __local_mem__ float *rmsNormLocalAddr = (__local_mem__ float *)rmsNormLocal.GetPhyAddr();
    __local_mem__ float *xCastLocalAddr = (__local_mem__ float *)xCastLocal.GetPhyAddr();
    __local_mem__ T *xLocalAddr = (__local_mem__ T *)xLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint16_t curColNumAlign = RoundUp<T>(curColNum);
    uint16_t dstCurColNumAlign = RoundUp<float>(curColNum) + BLOCK_SIZE / sizeof(float);
    if (loopCount > 1) {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> x1;
            RegTensor<float> sum;
            RegTensor<float> one;
            RegTensor<float> y;
            MaskReg pregLoop;
            MaskReg pregMain = CreateMask<float>();
            MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            uint32_t sreg;
            for (uint16_t i = 0; i < curRowNum; i++) {
                Duplicate(sum, 0.0f);
                if constexpr (WithUbReduce) {
                    LoadInputDataWithBrc<float>(y, rmsNormLocalAddr, pregMerge, i);
                }
                sreg = curColNum;
                for (uint16_t j = 0; j < loopCount; j++) {
                    pregLoop = UpdateMask<float>(sreg);
                    LoadInputData(x, xLocalAddr, pregLoop, i * curColNumAlign + j * VL_FP32);
                    Mul(x1, x, x, pregLoop);
                    Add(sum, sum, x1, pregMain);
                    StoreOutputData(xCastLocalAddr, x, pregLoop, i * dstCurColNumAlign + j * VL_FP32);
                }
                Muls(sum, sum, coeff, pregMain);
                ReduceSum(sum, sum, pregMain);
                if constexpr (WithUbReduce) {
                    Add(y, y, sum, pregMerge);
                    DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rmsNormLocalAddr + i, y, pregMerge);
                } else {
                    DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rmsNormLocalAddr + i, sum, pregMerge);
                }
            }

        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> x1;
            RegTensor<float> sum;
            RegTensor<float> one;
            RegTensor<float> y;
            MaskReg pregMain = CreateMask<float>();
            MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            uint32_t sreg = curColNum;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            for (uint16_t i = 0; i < curRowNum; i++) {
                Duplicate(sum, 0.0f);
                if constexpr (WithUbReduce) {
                    LoadInputDataWithBrc<float>(y, rmsNormLocalAddr, pregMerge, i);
                }
                LoadInputData(x, xLocalAddr, pregLoop, i * curColNumAlign);
                StoreOutputData(xCastLocalAddr, x, pregLoop, i * dstCurColNumAlign);
                Mul(x1, x, x, pregLoop);
                Add(sum, sum, x1, pregLoop);
                Muls(sum, sum, coeff, pregLoop);
                ReduceSum(sum, sum, pregLoop);
                if constexpr (WithUbReduce) {
                    Add(y, y, sum, pregMerge);
                    DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rmsNormLocalAddr + i, y, pregMerge);
                } else {
                    DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rmsNormLocalAddr + i, sum, pregMerge);
                }
            }
        }
    }
}


template <bool WithUbReduce = false>
__aicore__ inline void VFProcessInvRmsPart1(const LocalTensor<float> &yLocal, const LocalTensor<float> &xLocal,
                                            float coeff, uint16_t curRowNum, uint32_t curColNum)
{
    __local_mem__ float *yLocalAddr = (__local_mem__ float *)yLocal.GetPhyAddr();
    __local_mem__ float *xLocalAddr = (__local_mem__ float *)xLocal.GetPhyAddr();

    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint16_t curColNumAlign = RoundUp<float>(curColNum) + BLOCK_SIZE / sizeof(float);
    if (loopCount > 1) {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> sum;
            RegTensor<float> one;
            RegTensor<float> y;
            MaskReg pregLoop;
            MaskReg pregMain = CreateMask<float>();
            MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            uint32_t sreg;
            for (uint16_t i = 0; i < curRowNum; i++) {
                Duplicate(sum, 0.0f);
                if constexpr (WithUbReduce) {
                    LoadInputDataWithBrc<float>(y, yLocalAddr, pregMerge, i);
                }
                sreg = curColNum;
                for (uint16_t j = 0; j < loopCount; j++) {
                    pregLoop = UpdateMask<float>(sreg);
                    LoadInputData(x, xLocalAddr, pregLoop, i * curColNumAlign + j * VL_FP32);
                    Mul(x, x, x, pregLoop);
                    Add(sum, sum, x, pregMain);
                }
                Muls(sum, sum, coeff, pregMain);
                ReduceSum(sum, sum, pregMain);
                if constexpr (WithUbReduce) {
                    Add(y, y, sum, pregMerge);
                    DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(yLocalAddr + i, y, pregMerge);
                } else {
                    DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(yLocalAddr + i, sum, pregMerge);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> sum;
            RegTensor<float> one;
            RegTensor<float> y;
            MaskReg pregMain = CreateMask<float>();
            MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            uint32_t sreg = curColNum;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            for (uint16_t i = 0; i < curRowNum; i++) {
                Duplicate(sum, 0.0f);
                if constexpr (WithUbReduce) {
                    LoadInputDataWithBrc<float>(y, yLocalAddr, pregMerge, i);
                }
                LoadInputData(x, xLocalAddr, pregLoop, i * curColNumAlign);
                Mul(x, x, x, pregLoop);
                Add(sum, sum, x, pregLoop);
                Muls(sum, sum, coeff, pregLoop);
                ReduceSum(sum, sum, pregLoop);
                if constexpr (WithUbReduce) {
                    Add(y, y, sum, pregMerge);
                    DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(yLocalAddr + i, y, pregMerge);
                } else {
                    DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(yLocalAddr + i, sum, pregMerge);
                }
            }
        }
    }
}

// (bs, k) --> (bs, 1)
// k 为Matmul K轴切分时的分核数，必然小于64
__aicore__ inline void VFProcessInvRmsPart2(const LocalTensor<float> &yLocal, const LocalTensor<float> &xLocal,
                                            float eps, uint16_t curRowNum, uint32_t curColNum)
{
    __local_mem__ float *yLocalAddr = (__local_mem__ float *)yLocal.GetPhyAddr();
    __local_mem__ float *xLocalAddr = (__local_mem__ float *)xLocal.GetPhyAddr();
    uint16_t curColNumAlign = RoundUp<float>(curColNum);
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> sum;
        RegTensor<float> one;
        RegTensor<float> y;
        uint32_t sreg = curColNum;
        MaskReg pregLoop = UpdateMask<float>(sreg);
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        Duplicate(one, static_cast<float>(1.0), pregMerge);
        for (uint16_t i = 0; i < curRowNum; i++) {
            LoadInputData(x, xLocalAddr, pregLoop, i * curColNumAlign);
            ReduceSum(sum, x, pregLoop);
            Adds(sum, sum, eps, pregMerge);
            Sqrt(sum, sum, pregMerge);
            Div(y, one, sum, pregMerge);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(yLocalAddr + i, y, pregMerge);
        }
    }
}


// (k, bs, hc_mix) * (k, bs, 1) = (bs, hc_mix)
// for循环组织形式如下:
/*
    for (i, 0, bs)

        for (j, 0, k)
            for (h, 0, hc_mix)
*/
// hcMix小于64，因此直接去掉内层for循环
__aicore__ inline void VFProcessInvRmsPart3WithGroupReduce(const LocalTensor<float> &yLocal,
                                                           const LocalTensor<float> &mmLocal,
                                                           const LocalTensor<float> &xLocal, float eps, uint16_t groupK,
                                                           uint16_t bs, uint32_t hcMix)
{
    __local_mem__ float *yLocalAddr = (__local_mem__ float *)yLocal.GetPhyAddr();
    __local_mem__ float *mmLocalAddr = (__local_mem__ float *)mmLocal.GetPhyAddr();
    __local_mem__ float *xLocalAddr = (__local_mem__ float *)xLocal.GetPhyAddr();
    uint32_t hcMixAlign = RoundUp<float>(hcMix);
    uint32_t bsAlign = RoundUp<float>(bs);
    uint16_t fourLoopNum = groupK / FOUR_UNFOLD;
    uint16_t tailLoopNum = groupK % FOUR_UNFOLD;
    if (groupK < 4) {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> sum1;
            RegTensor<float> sum2;
            RegTensor<float> one;
            RegTensor<float> rsqrt;
            RegTensor<float> y;
            RegTensor<float> mm;
            uint32_t sreg = hcMix;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            Duplicate(one, static_cast<float>(1.0), pregMerge);
            for (uint16_t i = 0; i < bs; i++) {
                Duplicate(sum1, static_cast<float>(0.0f), pregMerge);
                Duplicate(sum2, static_cast<float>(0.0f), pregLoop);
                for (uint16_t j = 0; j < groupK; j++) {
                    LoadInputDataWithBrc<float>(x, xLocalAddr, pregMerge, i + j * bsAlign);
                    Add(sum1, sum1, x, pregMerge);
                    LoadInputData<float>(mm, mmLocalAddr, pregLoop, i * hcMixAlign + j * bs * hcMixAlign);
                    Add(sum2, sum2, mm, pregLoop);
                }
                Adds(sum1, sum1, eps, pregMerge);
                Sqrt(sum1, sum1, pregMerge);
                Div(rsqrt, one, sum1, pregMerge);
                Duplicate(rsqrt, rsqrt, pregLoop);
                Mul(y, sum2, rsqrt, pregLoop);
                StoreOutputData(yLocalAddr, y, pregLoop, i * hcMixAlign);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> x3;
            RegTensor<float> x4;
            RegTensor<float> mm1;
            RegTensor<float> mm2;
            RegTensor<float> mm3;
            RegTensor<float> mm4;
            RegTensor<float> sumX1;
            RegTensor<float> sumX2;
            RegTensor<float> sumX3;
            RegTensor<float> sumX4;
            RegTensor<float> sumM1;
            RegTensor<float> sumM2;
            RegTensor<float> sumM3;
            RegTensor<float> sumM4;
            RegTensor<float> one;
            RegTensor<float> rsqrt;
            RegTensor<float> y;
            uint32_t sreg = hcMix;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            Duplicate(one, static_cast<float>(1.0), pregMerge);
            for (uint16_t i = 0; i < bs; i++) {
                Duplicate(sumX1, static_cast<float>(0.0f), pregMerge);
                Duplicate(sumX2, static_cast<float>(0.0f), pregMerge);
                Duplicate(sumX3, static_cast<float>(0.0f), pregMerge);
                Duplicate(sumX4, static_cast<float>(0.0f), pregMerge);
                Duplicate(sumM1, static_cast<float>(0.0f), pregLoop);
                Duplicate(sumM2, static_cast<float>(0.0f), pregLoop);
                Duplicate(sumM3, static_cast<float>(0.0f), pregLoop);
                Duplicate(sumM4, static_cast<float>(0.0f), pregLoop);
                for (uint16_t j = 0; j < fourLoopNum; j++) {
                    LoadInputDataWithBrc<float>(x1, xLocalAddr, pregMerge, i + 4 * j * bsAlign);
                    Add(sumX1, sumX1, x1, pregMerge);
                    LoadInputData<float>(mm1, mmLocalAddr, pregLoop, i * hcMixAlign + 4 * j * bs * hcMixAlign);
                    Add(sumM1, sumM1, mm1, pregLoop);

                    LoadInputDataWithBrc<float>(x2, xLocalAddr, pregMerge, i + (4 * j + 1) * bsAlign);
                    Add(sumX2, sumX2, x2, pregMerge);
                    LoadInputData<float>(mm2, mmLocalAddr, pregLoop, i * hcMixAlign + (4 * j + 1) * bs * hcMixAlign);
                    Add(sumM2, sumM2, mm2, pregLoop);

                    LoadInputDataWithBrc<float>(x3, xLocalAddr, pregMerge, i + (4 * j + 2) * bsAlign);
                    Add(sumX3, sumX3, x3, pregMerge);
                    LoadInputData<float>(mm3, mmLocalAddr, pregLoop, i * hcMixAlign + (4 * j + 2) * bs * hcMixAlign);
                    Add(sumM3, sumM3, mm3, pregLoop);

                    LoadInputDataWithBrc<float>(x4, xLocalAddr, pregMerge, i + (4 * j + 3) * bsAlign);
                    Add(sumX4, sumX4, x4, pregMerge);
                    LoadInputData<float>(mm4, mmLocalAddr, pregLoop, i * hcMixAlign + (4 * j + 3) * bs * hcMixAlign);
                    Add(sumM4, sumM4, mm4, pregLoop);
                }
                for (uint16_t j = 0; j < tailLoopNum; j++) {
                    LoadInputDataWithBrc<float>(x1, xLocalAddr, pregMerge, i + (fourLoopNum * FOUR_UNFOLD + j) * bsAlign);
                    Add(sumX1, sumX1, x1, pregMerge);
                    LoadInputData<float>(mm1, mmLocalAddr, pregLoop, i * hcMixAlign + (fourLoopNum * FOUR_UNFOLD + j) * bs * hcMixAlign);
                    Add(sumM1, sumM1, mm1, pregLoop);
                }
                Add(sumX1, sumX1, sumX4, pregMerge);
                Add(sumX2, sumX2, sumX3, pregMerge);
                Add(sumX1, sumX1, sumX2, pregMerge);
                Add(sumM1, sumM1, sumM4, pregLoop);
                Add(sumM2, sumM2, sumM3, pregLoop);
                Add(sumM1, sumM1, sumM2, pregLoop);

                Adds(sumX1, sumX1, eps, pregMerge);
                Sqrt(sumX1, sumX1, pregMerge);
                Div(rsqrt, one, sumX1, pregMerge);
                Duplicate(rsqrt, rsqrt, pregLoop);
                Mul(y, sumM1, rsqrt, pregLoop);
                StoreOutputData(yLocalAddr, y, pregLoop, i * hcMixAlign);
            }
        }
    }
}


__aicore__ inline void VFProcessInvRmsPart3(const LocalTensor<float> &yLocal,
                                            const LocalTensor<float> &mmLocal,
                                            const LocalTensor<float> &xLocal, float eps,
                                            uint16_t bs, uint32_t hcMix)
{
    __local_mem__ float *yLocalAddr = (__local_mem__ float *)yLocal.GetPhyAddr();
    __local_mem__ float *mmLocalAddr = (__local_mem__ float *)mmLocal.GetPhyAddr();
    __local_mem__ float *xLocalAddr = (__local_mem__ float *)xLocal.GetPhyAddr();
    uint32_t hcMixAlign = RoundUp<float>(hcMix);
    uint32_t bsAlign = RoundUp<float>(bs);
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> sum1;
        RegTensor<float> sum2;
        RegTensor<float> one;
        RegTensor<float> rsqrt;
        RegTensor<float> y;
        RegTensor<float> mm;
        uint32_t sreg = hcMix;
        MaskReg pregLoop = UpdateMask<float>(sreg);
        Duplicate(one, static_cast<float>(1.0), pregLoop);
        for (uint16_t i = 0; i < bs; i++) {
            LoadInputDataWithBrc<float>(x, xLocalAddr, pregLoop, i);
            LoadInputData<float>(mm, mmLocalAddr, pregLoop, i * hcMixAlign);
            Adds(x, x, eps, pregLoop);
            Sqrt(x, x, pregLoop);
            Div(rsqrt, one, x, pregLoop);
            Mul(y, mm, rsqrt, pregLoop);
            StoreOutputData(yLocalAddr, y, pregLoop, i * hcMixAlign);
        }
    }
}


// Matmul的结果会直接FixPipe到UB上，不会在搬运时完成Split动作，因此需要在UB内完成Split动作
__aicore__ inline void VFProcessPre(const LocalTensor<float> &preLocal, const LocalTensor<float> &mixLocal,
                                    const LocalTensor<float> &hcBaseLocal,
                                    float scale, float eps, uint16_t curRowNum, uint16_t curColNum, uint16_t hcMix)
{
    __local_mem__ float *preLocalAddr = (__local_mem__ float *)preLocal.GetPhyAddr();
    __local_mem__ float *mixLocalAddr = (__local_mem__ float *)mixLocal.GetPhyAddr();
    __local_mem__ float *hcBaseLocalAddr = (__local_mem__ float *)hcBaseLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<float>(curColNum);
    uint32_t hcMixAlign = RoundUp<float>(hcMix);
    if (loopCount > 1) {
        __VEC_SCOPE__
        {
            RegTensor<float> mix;
            RegTensor<float> base;
            RegTensor<float> one;
            MaskReg pregLoop = CreateMask<float>();
            uint32_t sreg = curColNum;
            Duplicate(one, static_cast<float>(1), pregLoop);
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, i * VL_FP32);
                for (uint16_t j = 0; j < curRowNum; j++) {
                    LoadInputData<float>(mix, mixLocalAddr, pregLoop, i * VL_FP32 + j * hcMixAlign);
                    Muls(mix, mix, scale, pregLoop);
                    Add(mix, mix, base, pregLoop);
                    VFSigmoid(mix, mix, one, pregLoop);
                    Adds(mix, mix, eps, pregLoop);
                    StoreOutputData(preLocalAddr, mix, pregLoop, i * VL_FP32 + j * hcMixAlign);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> mix;
            RegTensor<float> base;
            RegTensor<float> one;
            uint32_t sreg = curColNum;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            Duplicate(one, static_cast<float>(1), pregLoop);
            LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, 0);
            for (uint16_t i = 0; i < curRowNum; i++) {
                LoadInputData<float>(mix, mixLocalAddr, pregLoop, i * hcMixAlign);
                Muls(mix, mix, scale, pregLoop);
                Add(mix, mix, base, pregLoop);
                VFSigmoid(mix, mix, one, pregLoop);
                Adds(mix, mix, eps, pregLoop);
                StoreOutputData(preLocalAddr, mix, pregLoop, i * hcMixAlign);
            }
        }
    }
}

__aicore__ inline void VFProcessPost(const LocalTensor<float> &postLocal, const LocalTensor<float> &mixLocal,
                                     const LocalTensor<float> &hcBaseLocal,
                                     float scale, float eps, uint16_t curRowNum, uint16_t curColNum, uint16_t hcMix)
{
    __local_mem__ float *postLocalAddr = (__local_mem__ float *)postLocal.GetPhyAddr();
    __local_mem__ float *mixOriginLocalAddr = (__local_mem__ float *)mixLocal.GetPhyAddr();
    __local_mem__ float *hcBaseLocalAddr = (__local_mem__ float *)hcBaseLocal.GetPhyAddr();
    __local_mem__ float *mixLocalAddr = mixOriginLocalAddr;
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<float>(curColNum);
    uint32_t hcMixAlign = RoundUp<float>(hcMix);
    if (loopCount > 1) {
        __VEC_SCOPE__
        {
            RegTensor<float> mix;
            RegTensor<float> base;
            RegTensor<float> one;
            UnalignReg uMix;
            MaskReg pregLoop = CreateMask<float>();
            uint32_t sreg = curColNum;
            Duplicate(one, static_cast<float>(1), pregLoop);
            DataCopyUnAlignPre<float>(uMix, mixLocalAddr);
            for (uint16_t i = 0; i < loopCount; i++) {
                mixLocalAddr = mixOriginLocalAddr + i * VL_FP32;
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, i * VL_FP32);
                for (uint16_t j = 0; j < curRowNum; j++) {
                    LoadInputDataUnalign(mix, mixLocalAddr, uMix, pregLoop, hcMixAlign);
                    Muls(mix, mix, scale, pregLoop);
                    Add(mix, mix, base, pregLoop);
                    VFSigmoid(mix, mix, one, pregLoop);
                    Muls(mix, mix, static_cast<float>(2.0), pregLoop);
                    StoreOutputData(postLocalAddr, mix, pregLoop, i * VL_FP32 + j * curColNumAlign);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> mix;
            RegTensor<float> base;
            RegTensor<float> one;
            UnalignReg uMix;
            uint32_t sreg = curColNum;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            Duplicate(one, static_cast<float>(1), pregLoop);
            DataCopyUnAlignPre<float>(uMix, mixLocalAddr);
            LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, 0);
            for (uint16_t i = 0; i < curRowNum; i++) {
                LoadInputDataUnalign(mix, mixLocalAddr, uMix, pregLoop, hcMixAlign);
                Muls(mix, mix, scale, pregLoop);
                Add(mix, mix, base, pregLoop);
                VFSigmoid(mix, mix, one, pregLoop);
                Muls(mix, mix, static_cast<float>(2.0), pregLoop);
                StoreOutputData(postLocalAddr, mix, pregLoop, i * curColNumAlign);
            }
        }
    }
}

// dim2是R轴，R轴小于64, 不需要回写UB
__aicore__ inline void VFProcessCombFragRLessVL(const LocalTensor<float> &combFragLocal,
                                                const LocalTensor<float> &mixLocal,
                                                const LocalTensor<float> &hcBaseLocal,
                                                float scale, float eps,
                                                uint16_t iters, uint16_t dim0, uint16_t dim1, uint16_t dim2, uint16_t hcMix)
{
    __local_mem__ float *combFragLocalAddr = (__local_mem__ float *)combFragLocal.GetPhyAddr();
    __local_mem__ float *mixLocalOriginAddr = (__local_mem__ float *)mixLocal.GetPhyAddr();
    __local_mem__ float *hcBaseLocalAddr = (__local_mem__ float *)hcBaseLocal.GetPhyAddr();
    __local_mem__ float *mixLocalAddr = mixLocalOriginAddr;
    uint32_t dim2Align = RoundUp<float>(dim2);
    uint32_t hcMixAlign = RoundUp<float>(hcMix);
    __VEC_SCOPE__
    {
        RegTensor<float> base;
        RegTensor<float> mix;
        RegTensor<float> rsqrt;
        RegTensor<float> max;
        RegTensor<float> sum;
        RegTensor<float> sum1;
        UnalignReg uMix;
        uint32_t sreg = dim2;
        MaskReg pregLoop = UpdateMask<float>(sreg);
        DataCopyUnAlignPre<float>(uMix, mixLocalAddr);
        for (uint16_t i = 0; i < dim0; i++) {
            Duplicate(sum1, static_cast<float>(0), pregLoop);
            for (uint16_t j = 0; j < dim1; j++) {
                mixLocalAddr = mixLocalOriginAddr + i * hcMixAlign + j * dim2;
                LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, j * dim2Align);
                LoadInputDataUnalign<float>(mix, mixLocalAddr, uMix, pregLoop, VL_FP32);
                Muls(mix, mix, scale, pregLoop);
                Add(mix, mix, base, pregLoop);
                ReduceMax(max, mix, pregLoop);
                Duplicate(max, max, pregLoop);
                Sub(mix, mix, max, pregLoop);
                Exp(mix, mix, pregLoop);
                ReduceSum(sum, mix, pregLoop);
                Duplicate(sum, sum, pregLoop);
                Div(mix, mix, sum, pregLoop);
                Adds(mix, mix, eps, pregLoop);
                Add(sum1, sum1, mix, pregLoop);
                StoreOutputData(combFragLocalAddr, mix, pregLoop, i * dim1 * dim2Align + j * dim2Align);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            Adds(sum1, sum1, eps, pregLoop);
            for (uint16_t j = 0; j < dim1; j++) {
                LoadInputData<float>(mix, combFragLocalAddr, pregLoop, i * dim1 * dim2Align + j * dim2Align);
                Div(mix, mix, sum1, pregLoop);
                StoreOutputData(combFragLocalAddr, mix, pregLoop, i * dim1 * dim2Align + j * dim2Align);
            }
        }
        for (uint16_t i = 0; i < iters; i++) {
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            for (uint16_t j = 0; j < dim0; j++) {
                Duplicate(sum1, static_cast<float>(0), pregLoop);
                for (uint16_t k = 0; k < dim1; k++) {
                    LoadInputData<float>(mix, combFragLocalAddr, pregLoop, j * dim1 * dim2Align + k * dim2Align);
                    ReduceSum(sum, mix, pregLoop);
                    Duplicate(sum, sum, pregLoop);
                    Adds(sum, sum, eps, pregLoop);
                    Div(mix, mix, sum, pregLoop);
                    Add(sum1, sum1, mix, pregLoop);
                    StoreOutputData(combFragLocalAddr, mix, pregLoop, j * dim1 * dim2Align + k * dim2Align);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                Adds(sum1, sum1, eps, pregLoop);
                for (uint16_t k = 0; k < dim1; k++) {
                    LoadInputData<float>(mix, combFragLocalAddr, pregLoop, j * dim1 * dim2Align + k * dim2Align);
                    Div(mix, mix, sum1, pregLoop);
                    StoreOutputData(combFragLocalAddr, mix, pregLoop, j * dim1 * dim2Align + k * dim2Align);
                }
            }
        }
    }
}

__aicore__ inline void VFProcessIteration(RegTensor<float> &sum0, RegTensor<float> &sum1, RegTensor<float> &mix,
                                          float eps, MaskReg pregLoop)
{
    ReduceSum(sum1, mix, pregLoop);
    Duplicate(sum1, sum1, pregLoop);
    Adds(sum1, sum1, eps, pregLoop);
    Div(mix, mix, sum1, pregLoop);
    Add(sum0, sum0, mix, pregLoop);
}

__aicore__ inline void VFProcessCombFragRLessVLUseFourUnfold(const LocalTensor<float> &combFragLocal,
                                                             const LocalTensor<float> &mixLocal,
                                                             const LocalTensor<float> &hcBaseLocal,
                                                              float scale,
                                                             float eps, uint16_t iters, uint16_t dim0, uint16_t dim1,
                                                             uint16_t dim2, uint16_t hcMix)
{
    __local_mem__ float *combFragLocalAddr = (__local_mem__ float *)combFragLocal.GetPhyAddr();
    __local_mem__ float *mixLocalOriginAddr = (__local_mem__ float *)mixLocal.GetPhyAddr();
    __local_mem__ float *hcBaseLocalAddr = (__local_mem__ float *)hcBaseLocal.GetPhyAddr();
    __local_mem__ float *mixLocalAddr = mixLocalOriginAddr;
    uint32_t dim2Align = RoundUp<float>(dim2);
    uint32_t hcMixAlign = RoundUp<float>(hcMix);
    __VEC_SCOPE__
    {
        RegTensor<float> base;
        RegTensor<float> mix;
        RegTensor<float> mix1;
        RegTensor<float> mix2;
        RegTensor<float> mix3;
        RegTensor<float> mix4;
        RegTensor<float> max;
        RegTensor<float> sum;
        RegTensor<float> sum1;
        RegTensor<float> sum2;
        RegTensor<float> sum3;
        RegTensor<float> sum4;
        UnalignReg uMix;
        uint32_t sreg = dim2;
        MaskReg pregLoop = UpdateMask<float>(sreg);
        DataCopyUnAlignPre<float>(uMix, mixLocalAddr);
        for (uint16_t i = 0; i < dim0; i++) {
            Duplicate(sum1, static_cast<float>(0), pregLoop);
            for (uint16_t j = 0; j < dim1; j++) {
                mixLocalAddr = mixLocalOriginAddr + i * hcMixAlign + j * dim2;
                LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, j * dim2Align);
                LoadInputDataUnalign<float>(mix, mixLocalAddr, uMix, pregLoop, VL_FP32);
                Muls(mix, mix, scale, pregLoop);
                Add(mix, mix, base, pregLoop);
                ReduceMax(max, mix, pregLoop);
                Duplicate(max, max, pregLoop);
                Sub(mix, mix, max, pregLoop);
                Exp(mix, mix, pregLoop);
                ReduceSum(sum, mix, pregLoop);
                Duplicate(sum, sum, pregLoop);
                Div(mix, mix, sum, pregLoop);
                Adds(mix, mix, eps, pregLoop);
                Add(sum1, sum1, mix, pregLoop);
                StoreOutputData(combFragLocalAddr, mix, pregLoop, i * dim1 * dim2Align + j * dim2Align);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            Adds(sum1, sum1, eps, pregLoop);
            for (uint16_t j = 0; j < dim1; j++) {
                LoadInputData<float>(mix, combFragLocalAddr, pregLoop, i * dim1 * dim2Align + j * dim2Align);
                Div(mix, mix, sum1, pregLoop);
                StoreOutputData(combFragLocalAddr, mix, pregLoop, i * dim1 * dim2Align + j * dim2Align);
            }
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < dim0; i++) {
            LoadInputData<float>(mix1, combFragLocalAddr, pregLoop, i * dim1 * dim2Align);
            LoadInputData<float>(mix2, combFragLocalAddr, pregLoop, i * dim1 * dim2Align + 1 * dim2Align);
            LoadInputData<float>(mix3, combFragLocalAddr, pregLoop, i * dim1 * dim2Align + 2 * dim2Align);
            LoadInputData<float>(mix4, combFragLocalAddr, pregLoop, i * dim1 * dim2Align + 3 * dim2Align);
            for (uint16_t j = 0; j < iters; j++) {
                Duplicate(sum, static_cast<float>(0), pregLoop);
                VFProcessIteration(sum, sum1, mix1, eps, pregLoop);
                VFProcessIteration(sum, sum2, mix2, eps, pregLoop);
                VFProcessIteration(sum, sum3, mix3, eps, pregLoop);
                VFProcessIteration(sum, sum4, mix4, eps, pregLoop);
                Adds(sum, sum, eps, pregLoop);
                Div(mix1, mix1, sum, pregLoop);
                Div(mix2, mix2, sum, pregLoop);
                Div(mix3, mix3, sum, pregLoop);
                Div(mix4, mix4, sum, pregLoop);
            }
            StoreOutputData(combFragLocalAddr, mix1, pregLoop, i * dim1 * dim2Align);
            StoreOutputData(combFragLocalAddr, mix2, pregLoop, i * dim1 * dim2Align + 1 * dim2Align);
            StoreOutputData(combFragLocalAddr, mix3, pregLoop, i * dim1 * dim2Align + 2 * dim2Align);
            StoreOutputData(combFragLocalAddr, mix4, pregLoop, i * dim1 * dim2Align + 3 * dim2Align);
        }
    }
}

__aicore__ inline void RowGroupMaxBcast(RegTensor<float> &out, RegTensor<float> &in, RegTensor<float> &t0,
                                        RegTensor<float> &t1, RegTensor<float> &t2, RegTensor<float> &t3, MaskReg preg)
{
    DeInterleave(t0, t1, in, in);        // t0 has even lanes, t1 has odd lanes, paired inside each group.
    Max(t0, t0, t1, preg);               // Pairwise max for c0/c1 and c2/c3 inside each group.
    DeInterleave(t2, t3, t0, t0);
    Max(t2, t2, t3, preg);               // lane0..3 holds the max for each row group.
    Interleave(t0, t1, t2, t2);          // [m0,m0,m1,m1,m2,m2,m3,m3]
    Interleave(out, t1, t0, t0);         // Broadcast each row max to 4 lanes.
}

// Reduce the row sum over C in contiguous groups of 4 and broadcast it: out[4r+c] = sum_c M[r][c].
__aicore__ inline void RowGroupSumBcast(RegTensor<float> &out, RegTensor<float> &in, RegTensor<float> &t0,
                                        RegTensor<float> &t1, RegTensor<float> &t2, RegTensor<float> &t3, MaskReg preg)
{
    DeInterleave(t0, t1, in, in);
    Add(t0, t0, t1, preg);
    DeInterleave(t2, t3, t0, t0);
    Add(t2, t2, t3, preg);
    Interleave(t0, t1, t2, t2);
    Interleave(out, t1, t0, t0);
}

// In-place softmax over C: subtract row max, exp, divide by row sum, then add eps.
__aicore__ inline void SoftmaxRowBcastInplace(RegTensor<float> &m, RegTensor<float> &red,
                                              RegTensor<float> &t0, RegTensor<float> &t1,
                                              RegTensor<float> &t2, RegTensor<float> &t3,
                                              float eps, MaskReg preg)
{
    RowGroupMaxBcast(red, m, t0, t1, t2, t3, preg);
    Sub(m, m, red, preg);
    Exp(m, m, preg);
    RowGroupSumBcast(red, m, t0, t1, t2, t3, preg);
    Div(m, m, red, preg);
    Adds(m, m, eps, preg);
}

// In-place divide by row sum plus eps, and accumulate the result into csum for Sinkhorn iterations.
__aicore__ inline void RowNormAccum(RegTensor<float> &m, RegTensor<float> &red, RegTensor<float> &csum,
                                    RegTensor<float> &t0, RegTensor<float> &t1,
                                    RegTensor<float> &t2, RegTensor<float> &t3,
                                    float eps, MaskReg preg)
{
    RowGroupSumBcast(red, m, t0, t1, t2, t3, preg);
    Adds(red, red, eps, preg);
    Div(m, m, red, preg);
    Add(csum, csum, m, preg);
}

__aicore__ inline void VFProcessCombFragPacked(const LocalTensor<float> &combFragLocal,
                                               const LocalTensor<float> &mixLocal,
                                               const LocalTensor<float> &hcBaseLocal,
                                               float scale, float eps, uint16_t iters,
                                               uint16_t dim0, uint16_t hcMult, uint16_t hcMix)
{
    __local_mem__ float *combOutAddr = (__local_mem__ float *)combFragLocal.GetPhyAddr();
    __local_mem__ float *mixAddr = (__local_mem__ float *)mixLocal.GetPhyAddr();
    __local_mem__ float *hcBaseAddr = (__local_mem__ float *)hcBaseLocal.GetPhyAddr();

    const uint16_t R = hcMult;
    const uint16_t pack = VL_FP32 / R;
    const uint16_t combLen = R * R;
    const uint16_t nChunks = (dim0 + pack - 1) / pack;
    __local_mem__ float *mBase = mixAddr;
    __local_mem__ float *oBase = combOutAddr;

    __VEC_SCOPE__
    {
        RegTensor<float> mix0, mix1, mix2, mix3;
        RegTensor<float> base0, base1, base2, base3;
        RegTensor<float> red, csum;
        RegTensor<float> t0, t1, t2, t3;
        RegTensor<int32_t> vL, vbs, vc, gIdx, sIdx, cShift, cMask;

        uint32_t uFull = VL_FP32;
        MaskReg pIdx = UpdateMask<int32_t>(uFull);

        Arange(vL, 0);
        Duplicate(cShift, (int32_t)2, pIdx);
        Duplicate(cMask, (int32_t)(R - 1), pIdx);
        ShiftRight(vbs, vL, cShift, pIdx);
        And(vc, vL, cMask, pIdx);
        Muls(gIdx, vbs, (int32_t)hcMix, pIdx);
        Add(gIdx, gIdx, vc, pIdx);
        Muls(sIdx, vbs, (int32_t)combLen, pIdx);
        Add(sIdx, sIdx, vc, pIdx);
        DataCopyGather(base0, hcBaseAddr + 0 * R, (RegTensor<uint32_t> &)vc, pIdx);
        DataCopyGather(base1, hcBaseAddr + 1 * R, (RegTensor<uint32_t> &)vc, pIdx);
        DataCopyGather(base2, hcBaseAddr + 2 * R, (RegTensor<uint32_t> &)vc, pIdx);
        DataCopyGather(base3, hcBaseAddr + 3 * R, (RegTensor<uint32_t> &)vc, pIdx);

        uint32_t act = (uint32_t)dim0 * R;
        for (uint16_t cIdx = 0; cIdx < nChunks; cIdx++) {
            uint16_t i = cIdx * pack;
            MaskReg preg = UpdateMask<float>(act);
            mBase = mixAddr + (uint32_t)i * hcMix;
            oBase = combOutAddr + (uint32_t)i * combLen;

            DataCopyGather(mix0, mBase + 0 * R, (RegTensor<uint32_t> &)gIdx, preg);
            DataCopyGather(mix1, mBase + 1 * R, (RegTensor<uint32_t> &)gIdx, preg);
            DataCopyGather(mix2, mBase + 2 * R, (RegTensor<uint32_t> &)gIdx, preg);
            DataCopyGather(mix3, mBase + 3 * R, (RegTensor<uint32_t> &)gIdx, preg);
            Muls(mix0, mix0, scale, preg); Add(mix0, mix0, base0, preg);
            Muls(mix1, mix1, scale, preg); Add(mix1, mix1, base1, preg);
            Muls(mix2, mix2, scale, preg); Add(mix2, mix2, base2, preg);
            Muls(mix3, mix3, scale, preg); Add(mix3, mix3, base3, preg);

            SoftmaxRowBcastInplace(mix0, red, t0, t1, t2, t3, eps, preg);
            SoftmaxRowBcastInplace(mix1, red, t0, t1, t2, t3, eps, preg);
            SoftmaxRowBcastInplace(mix2, red, t0, t1, t2, t3, eps, preg);
            SoftmaxRowBcastInplace(mix3, red, t0, t1, t2, t3, eps, preg);

            Add(csum, mix0, mix1, preg); Add(csum, csum, mix2, preg); Add(csum, csum, mix3, preg);
            Adds(csum, csum, eps, preg);
            Div(mix0, mix0, csum, preg); Div(mix1, mix1, csum, preg);
            Div(mix2, mix2, csum, preg); Div(mix3, mix3, csum, preg);

            for (uint16_t j = 0; j < iters; j++) {
                Duplicate(csum, static_cast<float>(0), preg);
                RowNormAccum(mix0, red, csum, t0, t1, t2, t3, eps, preg);
                RowNormAccum(mix1, red, csum, t0, t1, t2, t3, eps, preg);
                RowNormAccum(mix2, red, csum, t0, t1, t2, t3, eps, preg);
                RowNormAccum(mix3, red, csum, t0, t1, t2, t3, eps, preg);
                Adds(csum, csum, eps, preg);
                Div(mix0, mix0, csum, preg); Div(mix1, mix1, csum, preg);
                Div(mix2, mix2, csum, preg); Div(mix3, mix3, csum, preg);
            }
            DataCopyScatter(oBase + 0 * R, mix0, (RegTensor<uint32_t> &)sIdx, preg);
            DataCopyScatter(oBase + 1 * R, mix1, (RegTensor<uint32_t> &)sIdx, preg);
            DataCopyScatter(oBase + 2 * R, mix2, (RegTensor<uint32_t> &)sIdx, preg);
            DataCopyScatter(oBase + 3 * R, mix3, (RegTensor<uint32_t> &)sIdx, preg);
        }
    }
}


template <typename T>
__aicore__ inline void VFProcessY(const LocalTensor<T> &yLocal, const LocalTensor<float> &mixLocal,
                                  const LocalTensor<T> &xLocal, uint16_t bs, uint16_t hcMult, uint16_t d, uint16_t hcMix)
{
    __local_mem__ T *yLocalAddr = (__local_mem__ T *)yLocal.GetPhyAddr();
    __local_mem__ float *mixLocalAddr = (__local_mem__ float *)mixLocal.GetPhyAddr();
    __local_mem__ T *xLocalAddr = (__local_mem__ T *)xLocal.GetPhyAddr();
    uint32_t dAlign = RoundUp<T>(d);
    uint16_t loopCount = CeilDiv(d, VL_FP32);
    uint32_t hcMixAlign = RoundUp<float>(hcMix);
    if (loopCount > 1) {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> mix;
            RegTensor<float> sum;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < bs; i++) {
                uint32_t sreg = d;
                for (uint16_t j = 0; j < loopCount; j++) {
                    pregLoop = UpdateMask<float>(sreg);
                    Duplicate(sum, static_cast<float>(0), pregLoop);
                    for (uint16_t k = 0; k < hcMult; k++) {
                        LoadInputDataWithBrc<float>(mix, mixLocalAddr, pregLoop, i * hcMixAlign + k);
                        LoadInputData<T>(x, xLocalAddr, pregLoop, i * hcMult * dAlign + j * VL_FP32 + k * dAlign);
                        Mul(x, mix, x, pregLoop);
                        Add(sum, sum, x, pregLoop);
                    }
                    StoreOutputData(yLocalAddr, sum, pregLoop, i * dAlign + j * VL_FP32);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> mix;
            RegTensor<float> sum;
            uint32_t sreg = d;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            for (uint16_t i = 0; i < bs; i++) {
                Duplicate(sum, static_cast<float>(0), pregLoop);
                for (uint16_t j = 0; j < hcMult; j++) {
                    LoadInputDataWithBrc<float>(mix, mixLocalAddr, pregLoop, i * hcMixAlign + j);
                    LoadInputData<T>(x, xLocalAddr, pregLoop, i * hcMult * dAlign + j * dAlign);
                    Mul(x, mix, x, pregLoop);
                    Add(sum, sum, x, pregLoop);
                }
                StoreOutputData(yLocalAddr, sum, pregLoop, i * dAlign);
            }
        }
    }
}

template <typename T>
__aicore__ inline void CopyIn(const GlobalTensor<T> &inputGm, const LocalTensor<T> &inputTensor, const uint16_t nBurst,
                              const uint32_t copyLen, uint32_t srcStride = 0)
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

template <typename T>
__aicore__ inline void CopyToL1(const LocalTensor<T> &srcTensor, const LocalTensor<T> &dstTensor, const DataCopyParams dataCopyXParams)
{
    DataCopy(dstTensor, srcTensor, dataCopyXParams);
}

template <typename T>
__aicore__ inline void CopyInWithLoopMode(const GlobalTensor<T> &inputGm, const LocalTensor<T> &inputTensor,
                                          const uint16_t outerLoop, const uint16_t nBurst, const uint32_t copyLen,
                                          const uint32_t gmLastDim, uint32_t srcStride = 0)
{
    uint16_t copyLenAlign = RoundUp<T>(copyLen);
    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop1Size = outerLoop;
    loopParams.loop2SrcStride = 0;
    loopParams.loop1SrcStride = gmLastDim * sizeof(T);
    loopParams.loop2DstStride = 0;
    loopParams.loop1DstStride = nBurst * copyLenAlign * sizeof(T);

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
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPad(inputTensor, inputGm, dataCoptExtParams, dataCopyPadExtParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
}

template <typename T>
__aicore__ inline void CopyOut(const LocalTensor<T> &outputTensor, const GlobalTensor<T> &outputGm,
                               const uint16_t nBurst, const uint32_t copyLen, uint32_t dstStride = 0)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = dstStride * sizeof(T);
    DataCopyPad(outputGm, outputTensor, dataCopyParams);
}

template <typename T>
__aicore__ inline void CopyOut(const LocalTensor<T> &outputTensor, const LocalTensor<T> &outputGm,
                               const uint16_t nBurst, const uint32_t copyLen)
{
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = copyLen * sizeof(T) / BLOCK_SIZE;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopy(outputGm, outputTensor, dataCopyParams);
}


} // namespace HcPreSinkhorn

#endif
