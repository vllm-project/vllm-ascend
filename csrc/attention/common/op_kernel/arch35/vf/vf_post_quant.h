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
 * \file vf_post_quant.h
 * \brief
 */

#ifndef VF_POST_QUANT_H
#define VF_POST_QUANT_H

#include "vf_post_quant_common.h"

namespace FaVectorApi {

// fp32/fp16->int8/fp8
template <typename T, typename OUTPUT_T, typename POSTQUANT_PARAMS_T>
__simd_vf__ void PostQuantPerChnlOffsetImplVF(__ubuf__ OUTPUT_T *dstUb, __ubuf__ T *srcUb,
    __ubuf__ POSTQUANT_PARAMS_T *scaleUb, __ubuf__ POSTQUANT_PARAMS_T *offsetUb,
    const uint16_t floatRepSize, uint16_t dLoops, uint16_t dTailLoop, uint32_t pltTailD,
    const uint16_t gRowCount, const uint16_t s1RowCount, const uint16_t srcD)
{
    RegTensor<T> vregInput;
    RegTensor<T> vregMul;
    RegTensor<T> vreg_add;
    RegTensor<half> vregCastB16;
    RegTensor<OUTPUT_T> vregCast;
    RegTensor<T> vScale;
    RegTensor<T> vOffset;
    RegTensor<POSTQUANT_PARAMS_T> vScaleTmp;
    RegTensor<POSTQUANT_PARAMS_T> vOffsetTmp;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_tail = UpdateMask<float>(pltTailD);
    for (uint16_t m = 0; m < s1RowCount; ++m) {
        for (uint16_t j = 0; j < gRowCount; ++j) {
            for (uint16_t i = 0; i < dLoops; ++i) {
                LoadScaleParam<T, POSTQUANT_PARAMS_T>(vScale, vScaleTmp, scaleUb,
                    i * floatRepSize + j * srcD, preg_all);
                LoadOffsetParam<T, POSTQUANT_PARAMS_T>(vOffset, vOffsetTmp, offsetUb,
                    i * floatRepSize + j * srcD, preg_all);
                LoadAlign<T, LoadDist::DIST_NORM>(vregInput, srcUb + i * floatRepSize + j * srcD + m * srcD);
                Mul<T, MaskMergeMode::ZEROING>(vregMul, vregInput, vScale, preg_all);
                Add<T, MaskMergeMode::ZEROING>(vreg_add, vregMul, vOffset, preg_all);
                CastToOutputFromAdd<T, OUTPUT_T>(vregCastB16, vregCast, vreg_add, preg_all);
                StoreAlign<OUTPUT_T, StoreDist::DIST_PACK4_B32>(
                    dstUb + i * floatRepSize + j * srcD + m * srcD, vregCast, preg_all);
            }
            for (uint16_t k = 0; k < dTailLoop; ++k) {
                LoadScaleParam<T, POSTQUANT_PARAMS_T>(vScale, vScaleTmp, scaleUb,
                    dLoops * floatRepSize + j * srcD, preg_tail);
                LoadOffsetParam<T, POSTQUANT_PARAMS_T>(vOffset, vOffsetTmp, offsetUb,
                    dLoops * floatRepSize + j * srcD, preg_tail);
                LoadAlign<T, LoadDist::DIST_NORM>(vregInput,
                    srcUb + dLoops * floatRepSize + j * srcD + m * srcD);
                Mul<T, MaskMergeMode::ZEROING>(vregMul, vregInput, vScale, preg_tail);
                Add<T, MaskMergeMode::ZEROING>(vreg_add, vregMul, vOffset, preg_tail);
                CastToOutputFromAdd<T, OUTPUT_T>(vregCastB16, vregCast, vreg_add, preg_tail);
                StoreAlign<OUTPUT_T, StoreDist::DIST_PACK4_B32>(
                    dstUb + dLoops * floatRepSize + j * srcD + m * srcD, vregCast, preg_tail);
            }
        }
    }
}

template <typename T, typename OUTPUT_T, typename POSTQUANT_PARAMS_T>
__aicore__ inline void PostQuantPerChnlImpl(const LocalTensor<OUTPUT_T> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<POSTQUANT_PARAMS_T> &scaleTensor,
    const LocalTensor<POSTQUANT_PARAMS_T> &offsetTensor, const uint16_t gRowCount,
    const uint16_t s1RowCount, const uint32_t dSizeV, const uint16_t srcD)
{
    __ubuf__ OUTPUT_T *dstUb = (__ubuf__ OUTPUT_T *)dstTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ POSTQUANT_PARAMS_T *scaleUb = (__ubuf__ POSTQUANT_PARAMS_T *)scaleTensor.GetPhyAddr();
    __ubuf__ POSTQUANT_PARAMS_T *offsetUb = (__ubuf__ POSTQUANT_PARAMS_T *)offsetTensor.GetPhyAddr();
    const uint16_t floatRepSize = 64;
    uint16_t dLoops = dSizeV / floatRepSize;
    uint16_t dTail = dSizeV % floatRepSize;
    uint16_t dTailLoop = dTail > 0 ? 1 : 0;
    uint32_t pltTailD = dTail;
    PostQuantPerChnlOffsetImplVF<T, OUTPUT_T, POSTQUANT_PARAMS_T>(dstUb, srcUb, scaleUb, offsetUb,
        floatRepSize, dLoops, dTailLoop, pltTailD, gRowCount, s1RowCount, srcD);
}

// without offset
template <typename T, typename OUTPUT_T, typename POSTQUANT_PARAMS_T>
__simd_vf__ void PostQuantPerChnlNoOffsetImplVF(__ubuf__ OUTPUT_T *dstUb, __ubuf__ T *srcUb,
    __ubuf__ POSTQUANT_PARAMS_T *scaleUb,
    const uint16_t floatRepSize, uint16_t dLoops, uint16_t dTailLoop, uint32_t pltTailD,
    const uint16_t gRowCount, const uint16_t s1RowCount, const uint16_t srcD)
{
    RegTensor<T> vregInput;
    RegTensor<T> vregMul;
    RegTensor<half> vregCastB16;
    RegTensor<OUTPUT_T> vregCast;
    RegTensor<T> vScale;
    RegTensor<POSTQUANT_PARAMS_T> vScaleTmp;
    RegTensor<POSTQUANT_PARAMS_T> vOffsetTmp;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_tail = UpdateMask<float>(pltTailD);
    for (uint16_t m = 0; m < s1RowCount; ++m) {
        for (uint16_t j = 0; j < gRowCount; ++j) {
            for (uint16_t i = 0; i < dLoops; ++i) {
                LoadScaleParam<T, POSTQUANT_PARAMS_T>(vScale, vScaleTmp, scaleUb,
                    i * floatRepSize + j * srcD, preg_all);
                LoadAlign<T, LoadDist::DIST_NORM>(vregInput, srcUb + i * floatRepSize + j * srcD + m * srcD);
                Mul<T, MaskMergeMode::ZEROING>(vregMul, vregInput, vScale, preg_all);
                CastToOutputFromMul<T, OUTPUT_T>(vregCastB16, vregCast, vregMul, preg_all);
                StoreAlign<OUTPUT_T, StoreDist::DIST_PACK4_B32>(
                    dstUb + i * floatRepSize + j * srcD + m * srcD, vregCast, preg_all);
            }
            for (uint16_t k = 0; k < dTailLoop; ++k) {
                LoadScaleParam<T, POSTQUANT_PARAMS_T>(vScale, vScaleTmp, scaleUb,
                    dLoops * floatRepSize + j * srcD, preg_tail);
                LoadAlign<T, LoadDist::DIST_NORM>(vregInput,
                    srcUb + dLoops * floatRepSize + j * srcD + m * srcD);
                Mul<T, MaskMergeMode::ZEROING>(vregMul, vregInput, vScale, preg_tail);
                CastToOutputFromMul<T, OUTPUT_T>(vregCastB16, vregCast, vregMul, preg_tail);
                StoreAlign<OUTPUT_T, StoreDist::DIST_PACK4_B32>(
                    dstUb + dLoops * floatRepSize + j * srcD + m * srcD, vregCast, preg_tail);
            }
        }
    }
}

template <typename T, typename OUTPUT_T, typename POSTQUANT_PARAMS_T>
__aicore__ inline void PostQuantPerChnlImpl(const LocalTensor<OUTPUT_T> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<POSTQUANT_PARAMS_T> &scaleTensor, const uint16_t gRowCount,
    const uint16_t s1RowCount, const uint32_t dSizeV, const uint16_t srcD)
{
    __ubuf__ OUTPUT_T *dstUb = (__ubuf__ OUTPUT_T *)dstTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ POSTQUANT_PARAMS_T *scaleUb = (__ubuf__ POSTQUANT_PARAMS_T *)scaleTensor.GetPhyAddr();
    const uint16_t floatRepSize = 64;
    uint16_t dLoops = dSizeV / floatRepSize;
    uint16_t dTail = dSizeV % floatRepSize;
    uint16_t dTailLoop = dTail > 0 ? 1 : 0;
    uint32_t pltTailD = dTail;
    PostQuantPerChnlNoOffsetImplVF<T, OUTPUT_T, POSTQUANT_PARAMS_T>(dstUb, srcUb, scaleUb,
        floatRepSize, dLoops, dTailLoop, pltTailD, gRowCount, s1RowCount, srcD);
}

template <typename T, typename OUTPUT_T, bool hasOffset>
__simd_vf__ void PostQuantPerTensorImplVF(__ubuf__ OUTPUT_T *dstUb, __ubuf__ float *srcUb,
    const uint16_t floatRepSize, uint16_t dLoops, uint16_t dTailLoop, uint32_t pltTailD,
    const float postQuantScaleValue, const float postQuantOffsetValue,
    const uint16_t dealRowCount, const uint16_t srcD)
{
    RegTensor<T> vregInput;
    RegTensor<T> vregMul;
    RegTensor<T> vreg_add;
    RegTensor<half> vregCastB16;
    RegTensor<OUTPUT_T> vregCast;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_tail = UpdateMask<float>(pltTailD);
    for (uint16_t j = 0; j < dealRowCount; ++j) {
        for (uint16_t i = 0; i < dLoops; ++i) {
            LoadAlign<T, LoadDist::DIST_NORM>(vregInput, srcUb + i * floatRepSize + j * srcD);
            Muls<T, float, MaskMergeMode::ZEROING>(vregMul, vregInput, postQuantScaleValue, preg_all);
            if constexpr (IsSameType<OUTPUT_T, int8_t>::value) {
                if constexpr (hasOffset) {
                    Adds<T, float, MaskMergeMode::ZEROING>(vreg_add, vregMul, postQuantOffsetValue, preg_all);
                    Cast<half, T, castTraitP0>(vregCastB16, vreg_add, preg_all);
                } else {
                    Cast<half, T, castTraitP0>(vregCastB16, vregMul, preg_all);
                }
                Cast<OUTPUT_T, half, castTraitP0>(vregCast, vregCastB16, preg_all);
            } else {
                if constexpr (IsSameType<OUTPUT_T, hifloat8_t>::value) {
                    if constexpr (hasOffset) {
                        Adds<T, float, MaskMergeMode::ZEROING>(vreg_add, vregMul, postQuantOffsetValue, preg_all);
                        Cast<OUTPUT_T, T, castTraitP1>(vregCast, vreg_add, preg_all);
                    } else {
                        Cast<OUTPUT_T, T, castTraitP1>(vregCast, vregMul, preg_all);
                    }
                } else {
                    if constexpr (hasOffset) {
                        Adds<T, float, MaskMergeMode::ZEROING>(vreg_add, vregMul, postQuantOffsetValue, preg_all);
                        Cast<OUTPUT_T, T, castTraitP0>(vregCast, vreg_add, preg_all);
                    } else {
                        Cast<OUTPUT_T, T, castTraitP0>(vregCast, vregMul, preg_all);
                    }
                }
            }
            StoreAlign<OUTPUT_T, StoreDist::DIST_PACK4_B32>(dstUb + i * floatRepSize + j * srcD, vregCast, preg_all);
        }
        for (uint16_t k = 0; k < dTailLoop; ++k) {
            LoadAlign<T, LoadDist::DIST_NORM>(vregInput, srcUb + dLoops * floatRepSize + j * srcD);
            Muls<T, float, MaskMergeMode::ZEROING>(vregMul, vregInput, postQuantScaleValue, preg_tail);
            if constexpr (IsSameType<OUTPUT_T, int8_t>::value) {
                if constexpr (hasOffset) {
                    Adds<T, float, MaskMergeMode::ZEROING>(vreg_add, vregMul, postQuantOffsetValue, preg_tail);
                    Cast<half, T, castTraitP0>(vregCastB16, vreg_add, preg_tail);
                } else {
                    Cast<half, T, castTraitP0>(vregCastB16, vregMul, preg_tail);
                }
                Cast<OUTPUT_T, half, castTraitP0>(vregCast, vregCastB16, preg_tail);
            } else {
                if constexpr (IsSameType<OUTPUT_T, hifloat8_t>::value) {
                    if constexpr (hasOffset) {
                        Adds<T, float, MaskMergeMode::ZEROING>(vreg_add, vregMul, postQuantOffsetValue, preg_tail);
                        Cast<OUTPUT_T, T, castTraitP1>(vregCast, vreg_add, preg_tail);
                    } else {
                        Cast<OUTPUT_T, T, castTraitP1>(vregCast, vregMul, preg_tail);
                    }
                } else {
                    if constexpr (hasOffset) {
                        Adds<T, float, MaskMergeMode::ZEROING>(vreg_add, vregMul, postQuantOffsetValue, preg_tail);
                        Cast<OUTPUT_T, T, castTraitP0>(vregCast, vreg_add, preg_tail);
                    } else {
                        Cast<OUTPUT_T, T, castTraitP0>(vregCast, vregMul, preg_tail);
                    }
                }
            }
            StoreAlign<OUTPUT_T, StoreDist::DIST_PACK4_B32>(
                dstUb + dLoops * floatRepSize + j * srcD, vregCast, preg_tail);
        }
    }
}

template <typename T, typename OUTPUT_T, bool hasOffset>
__aicore__ inline void PostQuantPerTensorImpl(const LocalTensor<OUTPUT_T> &dstTensor, const LocalTensor<T> &srcTensor,
    const float postQuantScaleValue, const float postQuantOffsetValue,
    const uint16_t dealRowCount, const uint32_t dSizeV, const uint16_t srcD)
{
    __ubuf__ OUTPUT_T *dstUb = (__ubuf__ OUTPUT_T *)dstTensor.GetPhyAddr();
    __ubuf__ float *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    const uint16_t floatRepSize = 64;
    uint16_t dLoops = dSizeV / floatRepSize;
    uint16_t dTail = dSizeV % floatRepSize;
    uint16_t dTailLoop = dTail > 0 ? 1 : 0;
    uint32_t pltTailD = dTail;
    PostQuantPerTensorImplVF<T, OUTPUT_T, hasOffset>(dstUb, srcUb, floatRepSize, dLoops, dTailLoop,
        pltTailD, postQuantScaleValue, postQuantOffsetValue, dealRowCount, srcD);
}

} // namespace FaVectorApi

#endif // VF_POST_QUANT_H
