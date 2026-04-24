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
 * \file hc_pre_sinkhorn_regbase_base.h
 * \brief
 */

#ifndef HC_PRE_SINKHORN_RGEBASE_BASE_H
#define HC_PRE_SINKHORN_RGEBASE_BASE_H

#include "kernel_operator.h"

namespace HcPreSinkhorn {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UnalignReg;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t VL_FP32 = 64;

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
    }
}

template <typename T>
__aicore__ inline void LoadInputDataWithBrc(
    RegTensor<float>& dst, __local_mem__ T* src, MaskReg pregLoop, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(dst, src + srcOffset);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

__aicore__ inline void VFSigmoid(
     RegTensor<float>& y, RegTensor<float>& x, RegTensor<float>& one, MaskReg pregLoop)
{
    Muls(x, x, static_cast<float>(-1), pregLoop);
    Exp(x, x, pregLoop);
    Adds(x, x, static_cast<float>(1), pregLoop);
    Div(y, one, x, pregLoop);
}

__aicore__ inline void VFProcessPre(
    const LocalTensor<float>& preLocal, const LocalTensor<float>& mixLocal, const LocalTensor<float>& hcBaseLocal,
    const LocalTensor<float>& rsqrtLocal, float scale, float eps, uint16_t curRowNum, uint16_t curColNum)
{
    __local_mem__ float* preLocalAddr = (__local_mem__ float*)preLocal.GetPhyAddr();
    __local_mem__ float* mixLocalAddr = (__local_mem__ float*)mixLocal.GetPhyAddr();
    __local_mem__ float* hcBaseLocalAddr = (__local_mem__ float*)hcBaseLocal.GetPhyAddr();
    __local_mem__ float* rsqrtLocalAddr = (__local_mem__ float*)rsqrtLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<float>(curColNum);
    if (loopCount > 1) {
        __VEC_SCOPE__
        {
            RegTensor<float> mix;
            RegTensor<float> base;
            RegTensor<float> rsqrt;
            RegTensor<float> one;
            MaskReg pregLoop = CreateMask<float>();
            uint32_t sreg = curColNum;
            Duplicate(one, static_cast<float>(1), pregLoop);
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, i * VL_FP32);
                for (uint16_t j = 0; j < curRowNum; j++) {
                    LoadInputDataWithBrc<float>(rsqrt, rsqrtLocalAddr, pregLoop, j);
                    LoadInputData<float>(mix, mixLocalAddr, pregLoop, i * VL_FP32 + j * curColNumAlign);
                    Mul(mix, mix, rsqrt, pregLoop);
                    Muls(mix, mix, scale, pregLoop);
                    Add(mix, mix, base, pregLoop);
                    VFSigmoid(mix, mix, one, pregLoop);
                    Adds(mix, mix, eps, pregLoop);
                    StoreOutputData(preLocalAddr, mix, pregLoop, i * VL_FP32 + j * curColNumAlign);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> mix;
            RegTensor<float> base;
            RegTensor<float> rsqrt;
            RegTensor<float> one;
            uint32_t sreg = curColNum;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            Duplicate(one, static_cast<float>(1), pregLoop);
            LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, 0);
            for (uint16_t i = 0; i < curRowNum; i++) {
                LoadInputData<float>(mix, mixLocalAddr, pregLoop, i * curColNumAlign);
                LoadInputDataWithBrc<float>(rsqrt, rsqrtLocalAddr, pregLoop, i);
                Mul(mix, mix, rsqrt, pregLoop);
                Muls(mix, mix, scale, pregLoop);
                Add(mix, mix, base, pregLoop);
                VFSigmoid(mix, mix, one, pregLoop);
                Adds(mix, mix, eps, pregLoop);
                StoreOutputData(preLocalAddr, mix, pregLoop, i * curColNumAlign);
            }
        }
    }
}

__aicore__ inline void VFProcessPost(
    const LocalTensor<float>& postLocal, const LocalTensor<float>& mixLocal, const LocalTensor<float>& hcBaseLocal,
    const LocalTensor<float>& rsqrtLocal, float scale, float eps, uint16_t curRowNum, uint16_t curColNum)
{
    __local_mem__ float* postLocalAddr = (__local_mem__ float*)postLocal.GetPhyAddr();
    __local_mem__ float* mixLocalAddr = (__local_mem__ float*)mixLocal.GetPhyAddr();
    __local_mem__ float* hcBaseLocalAddr = (__local_mem__ float*)hcBaseLocal.GetPhyAddr();
    __local_mem__ float* rsqrtLocalAddr = (__local_mem__ float*)rsqrtLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<float>(curColNum);
    if (loopCount > 1) {
        __VEC_SCOPE__
        {
            RegTensor<float> mix;
            RegTensor<float> base;
            RegTensor<float> rsqrt;
            RegTensor<float> one;
            MaskReg pregLoop = CreateMask<float>();
            uint32_t sreg = curColNum;
            Duplicate(one, static_cast<float>(1), pregLoop);
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, i * VL_FP32);
                for (uint16_t j = 0; j < curRowNum; j++) {
                    LoadInputData<float>(mix, mixLocalAddr, pregLoop, i * VL_FP32 + j * curColNumAlign);
                    LoadInputDataWithBrc<float>(rsqrt, rsqrtLocalAddr, pregLoop, i);
                    Mul(mix, mix, rsqrt, pregLoop);
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
            RegTensor<float> rsqrt;
            RegTensor<float> one;
            uint32_t sreg = curColNum;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            Duplicate(one, static_cast<float>(1), pregLoop);
            LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, 0);
            for (uint16_t i = 0; i < curRowNum; i++) {
                LoadInputData<float>(mix, mixLocalAddr, pregLoop, i * curColNumAlign);
                LoadInputDataWithBrc<float>(rsqrt, rsqrtLocalAddr, pregLoop, i);
                Mul(mix, mix, rsqrt, pregLoop);
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
__aicore__ inline void VFProcessCombFragRLessVL(
    const LocalTensor<float>& combFragLocal, const LocalTensor<float>& mixLocal, const LocalTensor<float>& hcBaseLocal,
    const LocalTensor<float>& rsqrtLocal, float scale, float eps, uint16_t iters, uint16_t dim0, uint16_t dim1,
    uint16_t dim2)
{
    __local_mem__ float* combFragLocalAddr = (__local_mem__ float*)combFragLocal.GetPhyAddr();
    __local_mem__ float* mixLocalAddr = (__local_mem__ float*)mixLocal.GetPhyAddr();
    __local_mem__ float* hcBaseLocalAddr = (__local_mem__ float*)hcBaseLocal.GetPhyAddr();
    __local_mem__ float* rsqrtLocalAddr = (__local_mem__ float*)rsqrtLocal.GetPhyAddr();
    uint32_t dim2Align = RoundUp<float>(dim2);
    __VEC_SCOPE__
    {
        RegTensor<float> base;
        RegTensor<float> mix;
        RegTensor<float> rsqrt;
        RegTensor<float> max;
        RegTensor<float> sum;
        RegTensor<float> sum1;
        uint32_t sreg = dim2;
        MaskReg pregLoop = UpdateMask<float>(sreg);
        for (uint16_t i = 0; i < dim0; i++) {
            Duplicate(sum1, static_cast<float>(0), pregLoop);
            LoadInputDataWithBrc<float>(rsqrt, rsqrtLocalAddr, pregLoop, i);
            for (uint16_t j = 0; j < dim1; j++) {
                LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, j * dim2Align);
                LoadInputData<float>(mix, mixLocalAddr, pregLoop, i * dim1 * dim2Align + j * dim2Align);
                Mul(mix, mix, rsqrt, pregLoop);
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

__aicore__ inline void VFProcessIteration(RegTensor<float>& sum0, RegTensor<float>& sum1, RegTensor<float>& mix, float eps, MaskReg pregLoop) 
{
    ReduceSum(sum1, mix, pregLoop);
    Duplicate(sum1, sum1, pregLoop);
    Adds(sum1, sum1, eps, pregLoop);
    Div(mix, mix, sum1, pregLoop);
    Add(sum0, sum0, mix, pregLoop);
}

__aicore__ inline void VFProcessCombFragRLessVLUseFourUnfold(
    const LocalTensor<float>& combFragLocal, const LocalTensor<float>& mixLocal, const LocalTensor<float>& hcBaseLocal,
    const LocalTensor<float>& rsqrtLocal, float scale, float eps, uint16_t iters, uint16_t dim0, uint16_t dim1,
    uint16_t dim2)
{
    __local_mem__ float* combFragLocalAddr = (__local_mem__ float*)combFragLocal.GetPhyAddr();
    __local_mem__ float* mixLocalAddr = (__local_mem__ float*)mixLocal.GetPhyAddr();
    __local_mem__ float* hcBaseLocalAddr = (__local_mem__ float*)hcBaseLocal.GetPhyAddr();
    __local_mem__ float* rsqrtLocalAddr = (__local_mem__ float*)rsqrtLocal.GetPhyAddr();
    uint32_t dim2Align = RoundUp<float>(dim2);
    __VEC_SCOPE__
    {
        RegTensor<float> base;
        RegTensor<float> mix;
        RegTensor<float> mix1;
        RegTensor<float> mix2;
        RegTensor<float> mix3;
        RegTensor<float> mix4;
        RegTensor<float> rsqrt;
        RegTensor<float> max;
        RegTensor<float> sum;
        RegTensor<float> sum1;
        RegTensor<float> sum2;
        RegTensor<float> sum3;
        RegTensor<float> sum4;
        uint32_t sreg = dim2;
        MaskReg pregLoop = UpdateMask<float>(sreg);
        for (uint16_t i = 0; i < dim0; i++) {
            Duplicate(sum1, static_cast<float>(0), pregLoop);
            LoadInputDataWithBrc<float>(rsqrt, rsqrtLocalAddr, pregLoop, i);
            for (uint16_t j = 0; j < dim1; j++) {
                LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, j * dim2Align);
                LoadInputData<float>(mix, mixLocalAddr, pregLoop, i * dim1 * dim2Align + j * dim2Align);
                Mul(mix, mix, rsqrt, pregLoop);
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

template <typename T>
__aicore__ inline void VFProcessY(
    const LocalTensor<T>& yLocal, const LocalTensor<float>& mixLocal, const LocalTensor<T>& xLocal, uint16_t bs,
    uint16_t hcMult, uint16_t d)
{
    __local_mem__ T* yLocalAddr = (__local_mem__ T*)yLocal.GetPhyAddr();
    __local_mem__ float* mixLocalAddr = (__local_mem__ float*)mixLocal.GetPhyAddr();
    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    uint32_t dAlign = RoundUp<T>(d);
    uint16_t loopCount = CeilDiv(d, VL_FP32);
    uint32_t hcMultAlign = RoundUp<float>(hcMult);
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
                        LoadInputDataWithBrc<float>(mix, mixLocalAddr, pregLoop, i * hcMultAlign + k);
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
                    LoadInputDataWithBrc<float>(mix, mixLocalAddr, pregLoop, i * hcMultAlign + j);
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
__aicore__ inline void CopyIn(
    const GlobalTensor<T>& inputGm, const LocalTensor<T>& inputTensor, const uint16_t nBurst, const uint32_t copyLen, uint32_t srcStride = 0)
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
__aicore__ inline void CopyInWithLoopMode(
    const GlobalTensor<T>& inputGm, const LocalTensor<T>& inputTensor, const uint16_t outerLoop, const uint16_t nBurst, const uint32_t copyLen, const uint32_t gmLastDim,  uint32_t srcStride = 0)
{
    uint16_t copyLenAlign = RoundUp<T>(copyLen);
    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop1Size = outerLoop;
    loopParams.loop2SrcStride = 0;
    loopParams.loop1SrcStride =  gmLastDim * sizeof(T);
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
__aicore__ inline void CopyOut(
    const LocalTensor<T>& outputTensor, const GlobalTensor<T>& outputGm, const uint16_t nBurst, const uint32_t copyLen, uint32_t dstStride = 0)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = dstStride * sizeof(T);
    DataCopyPad(outputGm, outputTensor, dataCopyParams);
}

} // namespace HCPreSinkhorn

#endif