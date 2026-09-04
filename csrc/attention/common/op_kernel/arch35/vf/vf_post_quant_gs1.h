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
 * \file vf_post_quant_gs1.h
 * \brief Post-quantization VF implementation for both GS1 and S1G formats (arch35)
 */
#ifndef VF_POST_QUANT_V2_H
#define VF_POST_QUANT_V2_H
#include "vf_post_quant_common.h"
namespace FaVectorApi {

__simd_vf__ inline void CastBf16ToFp32VF(__ubuf__ float *dstFp32Ub, __ubuf__ bfloat16_t *srcBf16Ub,
                                         uint32_t elementCount)
{
    RegTensor<bfloat16_t> vregSrc;
    RegTensor<float> vregDst;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    const uint16_t floatRepSize = 64;
    uint16_t dLoops = elementCount / floatRepSize;
    uint16_t dTail = elementCount % floatRepSize;
    uint16_t dTailLoop = dTail > 0 ? 1 : 0;
    uint32_t pltTailD = dTail;
    for (uint16_t i = 0; i < dLoops; ++i) {
        LoadAlign<bfloat16_t, LoadDist::DIST_NORM>(vregSrc, srcBf16Ub + i * floatRepSize);
        Cast<float, bfloat16_t, castTraitP0>(vregDst, vregSrc, preg_all);
        StoreAlign<float, StoreDist::DIST_NORM>(dstFp32Ub + i * floatRepSize, vregDst, preg_all);
    }
    if (dTailLoop > 0) {
        MaskReg preg_tail = UpdateMask<float>(pltTailD);
        LoadAlign<bfloat16_t, LoadDist::DIST_NORM>(vregSrc, srcBf16Ub + dLoops * floatRepSize);
        Cast<float, bfloat16_t, castTraitP0>(vregDst, vregSrc, preg_tail);
        StoreAlign<float, StoreDist::DIST_NORM>(dstFp32Ub + dLoops * floatRepSize, vregDst, preg_tail);
    }
}

template <typename T, typename OUTPUT_T, typename POSTQUANT_PARAMS_T>
__simd_vf__ inline void
PostQuantPerChnlNoOffsetImplVF(__ubuf__ OUTPUT_T *dstUb, __ubuf__ T *srcUb, __ubuf__ POSTQUANT_PARAMS_T *scaleUb,
                               uint32_t src1Offset, uint32_t src0Offset, uint32_t floatRepSize, uint32_t mask_all,
                               uint32_t dLoops, uint32_t dTailLoop, uint32_t maskTail, uint32_t loopsOffset)
{
    RegTensor<T> vregInput;
    RegTensor<T> vregMul;
    RegTensor<half> vregCastB16;
    RegTensor<OUTPUT_T> vregCast;
    RegTensor<T> vScale;
    RegTensor<POSTQUANT_PARAMS_T> vScaleTmp;
    RegTensor<POSTQUANT_PARAMS_T> vOffsetTmp;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_tail = UpdateMask<float>(maskTail);
    for (uint16_t k = 0; k < dLoops; ++k) {
        LoadScaleParam<T, POSTQUANT_PARAMS_T>(vScale, vScaleTmp, scaleUb,
            src1Offset + k * floatRepSize, preg_all);
        LoadAlign<T, LoadDist::DIST_NORM>(vregInput, srcUb + src0Offset + loopsOffset + k * floatRepSize);
        Mul<T, MaskMergeMode::ZEROING>(vregMul, vregInput, vScale, preg_all);
        CastToOutputFromMul<T, OUTPUT_T>(vregCastB16, vregCast, vregMul, preg_all);
        StoreAlign<OUTPUT_T, StoreDist::DIST_PACK4_B32>(
            dstUb + src0Offset + loopsOffset + k * floatRepSize, vregCast, preg_all);
    }
    for (uint16_t k = 0; k < dTailLoop; ++k) {
        LoadScaleParam<T, POSTQUANT_PARAMS_T>(vScale, vScaleTmp, scaleUb,
            src1Offset + dLoops * floatRepSize, preg_tail);
        LoadAlign<T, LoadDist::DIST_NORM>(vregInput, srcUb + src0Offset + loopsOffset + dLoops * floatRepSize);
        Mul<T, MaskMergeMode::ZEROING>(vregMul, vregInput, vScale, preg_tail);
        CastToOutputFromMul<T, OUTPUT_T>(vregCastB16, vregCast, vregMul, preg_tail);
        StoreAlign<OUTPUT_T, StoreDist::DIST_PACK4_B32>(
            dstUb + src0Offset + loopsOffset + dLoops * floatRepSize, vregCast, preg_tail);
    }
}

template <typename T, typename OUTPUT_T, typename POSTQUANT_PARAMS_T>
__simd_vf__ inline void
PostQuantPerChnlOffsetImplVF(__ubuf__ OUTPUT_T *dstUb, __ubuf__ T *srcUb, __ubuf__ POSTQUANT_PARAMS_T *scaleUb,
                             __ubuf__ POSTQUANT_PARAMS_T *offsetUb, uint32_t src1Offset, uint32_t src0Offset,
                             uint32_t floatRepSize, uint32_t mask_all, uint32_t dLoops, uint32_t dTailLoop,
                             uint32_t maskTail, uint32_t loopsOffset)
{
    RegTensor<T> vregInput;
    RegTensor<T> vregMul;
    RegTensor<T> vreg_add;
    RegTensor<half> vregCastB16;
    RegTensor<T> vScale;
    RegTensor<T> vOffset;
    RegTensor<OUTPUT_T> vregCast;
    RegTensor<POSTQUANT_PARAMS_T> vScaleTmp;
    RegTensor<POSTQUANT_PARAMS_T> vOffsetTmp;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_tail = UpdateMask<float>(maskTail);
    for (uint16_t k = 0; k < dLoops; ++k) {
        LoadScaleParam<T, POSTQUANT_PARAMS_T>(vScale, vScaleTmp, scaleUb,
            src1Offset + k * floatRepSize, preg_all);
        LoadOffsetParam<T, POSTQUANT_PARAMS_T>(vOffset, vOffsetTmp, offsetUb,
            src1Offset + k * floatRepSize, preg_all);
        LoadAlign<T, LoadDist::DIST_NORM>(vregInput, srcUb + src0Offset + loopsOffset + k * floatRepSize);
        Mul<T, MaskMergeMode::ZEROING>(vregMul, vregInput, vScale, preg_all);
        Add<T, MaskMergeMode::ZEROING>(vreg_add, vregMul, vOffset, preg_all);
        CastToOutputFromAdd<T, OUTPUT_T>(vregCastB16, vregCast, vreg_add, preg_all);
        StoreAlign<OUTPUT_T, StoreDist::DIST_PACK4_B32>(
            dstUb + src0Offset + loopsOffset + k * floatRepSize, vregCast, preg_all);
    }
    for (uint16_t k = 0; k < dTailLoop; ++k) {
        LoadScaleParam<T, POSTQUANT_PARAMS_T>(vScale, vScaleTmp, scaleUb,
            src1Offset + dLoops * floatRepSize, preg_tail);
        LoadOffsetParam<T, POSTQUANT_PARAMS_T>(vOffset, vOffsetTmp, offsetUb,
            src1Offset + dLoops * floatRepSize, preg_tail);
        LoadAlign<T, LoadDist::DIST_NORM>(vregInput, srcUb + src0Offset + loopsOffset + dLoops * floatRepSize);
        Mul<T, MaskMergeMode::ZEROING>(vregMul, vregInput, vScale, preg_tail);
        Add<T, MaskMergeMode::ZEROING>(vreg_add, vregMul, vOffset, preg_tail);
        CastToOutputFromAdd<T, OUTPUT_T>(vregCastB16, vregCast, vreg_add, preg_tail);
        StoreAlign<OUTPUT_T, StoreDist::DIST_PACK4_B32>(
            dstUb + src0Offset + loopsOffset + dLoops * floatRepSize, vregCast, preg_tail);
    }
}

template <typename T, typename OUTPUT_T, typename POSTQUANT_PARAMS_T>
__aicore__ inline void
PostQuantPerChnlNoOffsetS1GImpl(const LocalTensor<OUTPUT_T> &dstTensor, const LocalTensor<T> &srcTensor,
                                const LocalTensor<POSTQUANT_PARAMS_T> &scaleTensor, uint32_t gSize, uint32_t s1Size,
                                uint32_t dSize, uint32_t gS1Idx, uint32_t gS1DealSize, uint32_t colCount)

{
    __ubuf__ OUTPUT_T *dstUb = (__ubuf__ OUTPUT_T *)dstTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ POSTQUANT_PARAMS_T *scaleUb = (__ubuf__ POSTQUANT_PARAMS_T *)scaleTensor.GetPhyAddr();
    const uint16_t floatRepSize = 64;
    uint16_t preg_all = 64;
    uint16_t preg_tail = colCount % 64;
    uint16_t dLoops = dSize / floatRepSize;
    uint16_t dTail = dSize % floatRepSize;
    uint16_t dTailLoop = dTail > 0 ? 1 : 0;
    uint32_t s1IdxStart = gS1Idx / gSize;
    uint32_t gIdxStart = gS1Idx % gSize;
    uint32_t s1IdxEnd = (gS1Idx + gS1DealSize) / gSize;
    uint32_t gIdxEnd = (gS1Idx + gS1DealSize) % gSize;
    if (s1IdxEnd - s1IdxStart > 1) {
        uint32_t headSize = gSize - gIdxStart;
        uint32_t src1Offset = gIdxStart * colCount;
        uint32_t src0Offset = 0;
        uint32_t dealSize = headSize * colCount;
        for (uint32_t i = 0; i < dealSize; i += colCount) {
            PostQuantPerChnlNoOffsetImplVF(dstUb, srcUb, scaleUb, src1Offset, src0Offset, floatRepSize, preg_all,
                                           dLoops, dTailLoop, preg_tail, i);
            src1Offset += colCount;
        }
        for (uint32_t i = s1IdxStart + 1; i < s1IdxEnd; i++) {
            src0Offset += dealSize;
            dealSize = gSize * colCount;
            src1Offset = 0;
            for (uint32_t j = 0; j < dealSize; j += colCount) {
                PostQuantPerChnlNoOffsetImplVF(dstUb, srcUb, scaleUb, src1Offset, src0Offset, floatRepSize, preg_all,
                                               dLoops, dTailLoop, preg_tail, j);
                src1Offset += colCount;
            }
        }
        if (gIdxEnd > 0) {
            src0Offset += dealSize;
            dealSize = gIdxEnd * colCount;
            src1Offset = 0;
            for (uint32_t j = 0; j < dealSize; j += colCount) {
                PostQuantPerChnlNoOffsetImplVF(dstUb, srcUb, scaleUb, src1Offset, src0Offset, floatRepSize, preg_all,
                                               dLoops, dTailLoop, preg_tail, j);
                src1Offset += colCount;
            }
        }
    } else {
        uint32_t dealSize = gS1DealSize * colCount;
        for (uint32_t offset = 0; offset < dealSize; offset += colCount) {
            PostQuantPerChnlNoOffsetImplVF(dstUb, srcUb, scaleUb, 0, 0, floatRepSize, preg_all, dLoops, dTailLoop,
                                           preg_tail, offset);
        }
    }
}

template <typename T, typename OUTPUT_T, typename POSTQUANT_PARAMS_T>
__aicore__ inline void
PostQuantPerChnlNoOffsetGS1Impl(const LocalTensor<OUTPUT_T> &dstTensor, const LocalTensor<T> &srcTensor,
                                const LocalTensor<POSTQUANT_PARAMS_T> &scaleTensor, uint32_t gSize, uint32_t s1Size,
                                uint32_t dSize, uint32_t gS1Idx, uint32_t gS1DealSize, uint32_t colCount)
{
    __ubuf__ OUTPUT_T *dstUb = (__ubuf__ OUTPUT_T *)dstTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ POSTQUANT_PARAMS_T *scaleUb = (__ubuf__ POSTQUANT_PARAMS_T *)scaleTensor.GetPhyAddr();
    uint16_t preg_all = 64;
    uint16_t preg_tail = colCount % 64;
    const uint16_t floatRepSize = 64;
    uint16_t dLoops = dSize / floatRepSize;
    uint16_t dTail = dSize % floatRepSize;
    uint16_t dTailLoop = dTail > 0 ? 1 : 0;
    uint32_t gIdxStart = gS1Idx / s1Size;
    uint32_t s1IdxStart = gS1Idx % s1Size;
    uint32_t gIdxEnd = (gS1Idx + gS1DealSize) / s1Size;
    uint32_t s1IdxEnd = (gS1Idx + gS1DealSize) % s1Size;
    uint32_t headS1 = 0;
    if (gIdxStart == gIdxEnd) {
        headS1 = s1IdxEnd - s1IdxStart;
    } else {
        headS1 = s1Size - s1IdxStart;
    }
    uint32_t src0Offset = 0;
    uint32_t src1Offset = 0;
    for (uint32_t m = s1IdxStart; m < s1IdxStart + headS1; ++m) {
        PostQuantPerChnlNoOffsetImplVF(dstUb, srcUb, scaleUb, src1Offset, src0Offset, floatRepSize, preg_all, dLoops,
                                       dTailLoop, preg_tail, 0);
        src0Offset += colCount;
    }
    if (gIdxEnd - gIdxStart >= 1) {
        src1Offset += colCount;
        for (uint32_t i = gIdxStart + 1; i < gIdxEnd; i++) {
            for (uint32_t m = 0; m < s1Size; ++m) {
                PostQuantPerChnlNoOffsetImplVF(dstUb, srcUb, scaleUb, src1Offset, src0Offset, floatRepSize, preg_all,
                                               dLoops, dTailLoop, preg_tail, 0);
                src0Offset += colCount;
            }
            src1Offset += colCount;
        }
        if (s1IdxEnd > 0) {
            for (uint32_t m = 0; m < s1IdxEnd; ++m) {
                PostQuantPerChnlNoOffsetImplVF(dstUb, srcUb, scaleUb, src1Offset, src0Offset, floatRepSize, preg_all,
                                               dLoops, dTailLoop, preg_tail, 0);
                src0Offset += colCount;
            }
        }
    }
}

template <typename T, typename OUTPUT_T, typename POSTQUANT_PARAMS_T>
__aicore__ inline void
PostQuantPerChnlWithOffsetS1GImpl(const LocalTensor<OUTPUT_T> &dstTensor, const LocalTensor<T> &srcTensor,
                                  const LocalTensor<POSTQUANT_PARAMS_T> &scaleTensor,
                                  const LocalTensor<POSTQUANT_PARAMS_T> &offsetTensor, uint32_t gSize, uint32_t s1Size,
                                  uint32_t dSize, uint32_t gS1Idx, uint32_t gS1DealSize, uint32_t colCount)
{
    __ubuf__ OUTPUT_T *dstUb = (__ubuf__ OUTPUT_T *)dstTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ POSTQUANT_PARAMS_T *scaleUb = (__ubuf__ POSTQUANT_PARAMS_T *)scaleTensor.GetPhyAddr();
    __ubuf__ POSTQUANT_PARAMS_T *offsetUb = (__ubuf__ POSTQUANT_PARAMS_T *)offsetTensor.GetPhyAddr();
    const uint16_t floatRepSize = 64;
    uint16_t preg_all = 64;
    uint16_t preg_tail = colCount % 64;
    uint16_t dLoops = dSize / floatRepSize;
    uint16_t dTail = dSize % floatRepSize;
    uint16_t dTailLoop = dTail > 0 ? 1 : 0;
    uint32_t s1IdxStart = gS1Idx / gSize;
    uint32_t gIdxStart = gS1Idx % gSize;
    uint32_t s1IdxEnd = (gS1Idx + gS1DealSize) / gSize;
    uint32_t gIdxEnd = (gS1Idx + gS1DealSize) % gSize;
    if (s1IdxEnd - s1IdxStart > 1) {
        uint32_t headSize = gSize - gIdxStart;
        uint32_t src0Offset = 0;
        uint32_t src1Offset = gIdxStart * colCount;
        uint32_t dealSize = headSize * colCount;
        for (uint32_t i = 0; i < dealSize; i += colCount) {
            PostQuantPerChnlOffsetImplVF(dstUb, srcUb, scaleUb, offsetUb, src1Offset, src0Offset, floatRepSize,
                                         preg_all, dLoops, dTailLoop, preg_tail, i);
            src1Offset += colCount;
        }
        for (uint32_t i = s1IdxStart + 1; i < s1IdxEnd; i++) {
            src0Offset += dealSize;
            dealSize = gSize * colCount;
            src1Offset = 0;
            for (uint32_t j = 0; j < dealSize; j += colCount) {
                PostQuantPerChnlOffsetImplVF(dstUb, srcUb, scaleUb, offsetUb, src1Offset, src0Offset, floatRepSize,
                                             preg_all, dLoops, dTailLoop, preg_tail, j);
                src1Offset += colCount;
            }
        }
        if (gIdxEnd > 0) {
            src0Offset += dealSize;
            dealSize = gIdxEnd * colCount;
            src1Offset = 0;
            for (uint32_t j = 0; j < dealSize; j += colCount) {
                PostQuantPerChnlOffsetImplVF(dstUb, srcUb, scaleUb, offsetUb, src1Offset, src0Offset, floatRepSize,
                                             preg_all, dLoops, dTailLoop, preg_tail, j);
                src1Offset += colCount;
            }
        }
    } else {
        uint32_t dealSize = gS1DealSize * colCount;
        for (uint32_t offset = 0; offset < dealSize; offset += colCount) {
            PostQuantPerChnlOffsetImplVF(dstUb, srcUb, scaleUb, offsetUb, 0, 0, floatRepSize, preg_all, dLoops,
                                         dTailLoop, preg_tail, offset);
        }
    }
}

template <typename T, typename OUTPUT_T, typename POSTQUANT_PARAMS_T>
__aicore__ inline void
PostQuantPerChnlWithOffsetGS1Impl(const LocalTensor<OUTPUT_T> &dstTensor, const LocalTensor<T> &srcTensor,
                                  const LocalTensor<POSTQUANT_PARAMS_T> &scaleTensor,
                                  const LocalTensor<POSTQUANT_PARAMS_T> &offsetTensor, uint32_t gSize, uint32_t s1Size,
                                  uint32_t dSize, uint32_t gS1Idx, uint32_t gS1DealSize, uint32_t colCount)
{
    __ubuf__ OUTPUT_T *dstUb = (__ubuf__ OUTPUT_T *)dstTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ POSTQUANT_PARAMS_T *offsetUb = (__ubuf__ POSTQUANT_PARAMS_T *)offsetTensor.GetPhyAddr();
    __ubuf__ POSTQUANT_PARAMS_T *scaleUb = (__ubuf__ POSTQUANT_PARAMS_T *)scaleTensor.GetPhyAddr();
    const uint16_t floatRepSize = 64;
    uint16_t preg_all = 64;
    uint16_t preg_tail = colCount % 64;
    uint16_t dLoops = dSize / floatRepSize;
    uint16_t dTail = dSize % floatRepSize;
    uint16_t dTailLoop = dTail > 0 ? 1 : 0;
    uint32_t s1IdxStart = gS1Idx % s1Size;
    uint32_t gIdxStart = gS1Idx / s1Size;
    uint32_t gIdxEnd = (gS1Idx + gS1DealSize) / s1Size;
    uint32_t s1IdxEnd = (gS1Idx + gS1DealSize) % s1Size;
    uint32_t headS1 = 0;
    if (gIdxStart == gIdxEnd) {
        headS1 = s1IdxEnd - s1IdxStart;
    } else {
        headS1 = s1Size - s1IdxStart;
    }
    uint32_t src1Offset = 0;
    uint32_t src0Offset = 0;
    for (uint32_t m = s1IdxStart; m < s1IdxStart + headS1; ++m) {
        PostQuantPerChnlOffsetImplVF(dstUb, srcUb, scaleUb, offsetUb, src1Offset, src0Offset, floatRepSize, preg_all,
                                     dLoops, dTailLoop, preg_tail, 0);
        src0Offset += colCount;
    }
    if (gIdxEnd - gIdxStart >= 1) {
        src1Offset += colCount;
        for (uint32_t i = gIdxStart + 1; i < gIdxEnd; i++) {
            for (uint32_t m = 0; m < s1Size; ++m) {
                PostQuantPerChnlOffsetImplVF(dstUb, srcUb, scaleUb, offsetUb, src1Offset, src0Offset, floatRepSize,
                                             preg_all, dLoops, dTailLoop, preg_tail, 0);
                src0Offset += colCount;
            }
            src1Offset += colCount;
        }
        if (s1IdxEnd > 0) {
            for (uint32_t m = 0; m < s1IdxEnd; ++m) {
                PostQuantPerChnlOffsetImplVF(dstUb, srcUb, scaleUb, offsetUb, src1Offset, src0Offset, floatRepSize,
                                             preg_all, dLoops, dTailLoop, preg_tail, 0);
                src0Offset += colCount;
            }
        }
    }
}

template <typename T, typename OUTPUT_T, typename POSTQUANT_PARAMS_T, bool isS1G>
__aicore__ inline void
PostQuantPerChnlOffsetImpl(const LocalTensor<OUTPUT_T> &dstTensor, const LocalTensor<T> &srcTensor,
                           const LocalTensor<POSTQUANT_PARAMS_T> &scaleTensor,
                           const LocalTensor<POSTQUANT_PARAMS_T> &offsetTensor, PostQuantInfo_V2 &postQuantInfo)
{
    uint32_t gS1Idx = postQuantInfo.gS1Idx;
    uint32_t colCount = postQuantInfo.colCount;
    uint32_t s1Size = postQuantInfo.s1Size;
    uint32_t gS1DealSize = postQuantInfo.gS1DealSize;
    uint32_t gSize = postQuantInfo.gSize;
    uint32_t dSize = postQuantInfo.dSize;
    if constexpr (isS1G) {
        PostQuantPerChnlWithOffsetS1GImpl(dstTensor, srcTensor, scaleTensor, offsetTensor, gS1Idx, s1Size, dSize,
                                          gS1Idx, gS1DealSize, colCount);
    } else {
        PostQuantPerChnlWithOffsetGS1Impl(dstTensor, srcTensor, scaleTensor, offsetTensor, gS1Idx, s1Size, dSize,
                                          gS1Idx, gS1DealSize, colCount);
    }
}

template <typename T, typename OUTPUT_T, typename POSTQUANT_PARAMS_T, bool isS1G>
__aicore__ inline void
PostQuantPerChnlNoOffsetImpl(const LocalTensor<OUTPUT_T> &dstTensor, const LocalTensor<T> &srcTensor,
                             const LocalTensor<POSTQUANT_PARAMS_T> &scaleTensor, PostQuantInfo_V2 &postQuantInfo)
{
    uint32_t gS1Idx = postQuantInfo.gS1Idx;
    uint32_t colCount = postQuantInfo.colCount;
    uint32_t s1Size = postQuantInfo.s1Size;
    uint32_t gS1DealSize = postQuantInfo.gS1DealSize;
    uint32_t gSize = postQuantInfo.gSize;
    uint32_t dSize = postQuantInfo.dSize;
    if constexpr (isS1G) {
        PostQuantPerChnlNoOffsetS1GImpl(dstTensor, srcTensor, scaleTensor, gS1Idx, s1Size, dSize, gS1Idx, gS1DealSize,
                                        colCount);
    } else {
        PostQuantPerChnlNoOffsetGS1Impl(dstTensor, srcTensor, scaleTensor, gS1Idx, s1Size, dSize, gS1Idx, gS1DealSize,
                                        colCount);
    }
}

} // namespace FaVectorApi
#endif // VF_POST_QUANT_V2_H
