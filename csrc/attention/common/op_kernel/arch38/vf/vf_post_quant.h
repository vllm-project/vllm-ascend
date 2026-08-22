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

#include "kernel_tensor.h"
namespace FaVectorApi {

// fp32/fp16->int8/fp8
static constexpr MicroAPI::CastTrait castTraitP0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp32->hifp8
static constexpr MicroAPI::CastTrait castTraitP1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};

// with offset
template <typename T, typename OUTPUT_T, typename POSTQUANT_PARAMS_T>
__aicore__ inline void PostQuantPerChnlVF(const LocalTensor<OUTPUT_T> &dstTensor, const LocalTensor<T> &srcTensor,
                                      const LocalTensor<POSTQUANT_PARAMS_T> &scaleTensor,
                                      const LocalTensor<POSTQUANT_PARAMS_T> &offsetTensor, const uint16_t gRowCount,
                                      const uint16_t s1RowCount, const uint32_t dSizeV, const uint16_t srcD)
{
    __ubuf__ OUTPUT_T *dstUb = (__ubuf__ OUTPUT_T *)dstTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ POSTQUANT_PARAMS_T *scaleUb = (__ubuf__ POSTQUANT_PARAMS_T *)scaleTensor.GetPhyAddr();
    __ubuf__ POSTQUANT_PARAMS_T *offsetUb = (__ubuf__ POSTQUANT_PARAMS_T *)offsetTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    uint16_t dLoops = dSizeV / floatRepSize;
    uint16_t dTail = dSizeV % floatRepSize;
    uint16_t dTailLoop = dTail > 0 ? 1 : 0;
    uint32_t pltTailD = dTail;

    __VEC_SCOPE__
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
                    if constexpr (IsSameType<POSTQUANT_PARAMS_T, T>::value) {
                        DataCopy<POSTQUANT_PARAMS_T, LoadDist::DIST_NORM>(vScale,
                                                                          scaleUb + i * floatRepSize + j * srcD);
                        DataCopy<POSTQUANT_PARAMS_T, LoadDist::DIST_NORM>(vOffset,
                                                                          offsetUb + i * floatRepSize + j * srcD);
                    } else {
                        DataCopy<POSTQUANT_PARAMS_T, LoadDist::DIST_UNPACK_B16>(vScaleTmp,
                                                                                scaleUb + i * floatRepSize + j * srcD);
                        DataCopy<POSTQUANT_PARAMS_T, LoadDist::DIST_UNPACK_B16>(vOffsetTmp,
                                                                                offsetUb + i * floatRepSize + j * srcD);
                        Cast<T, POSTQUANT_PARAMS_T, castTraitP0>(vScale, vScaleTmp, preg_all);
                        Cast<T, POSTQUANT_PARAMS_T, castTraitP0>(vOffset, vOffsetTmp, preg_all);
                    }
                    DataCopy<T, LoadDist::DIST_NORM>(vregInput, srcUb + i * floatRepSize + j * srcD + m * srcD);
                    Mul<T, MaskMergeMode::ZEROING>(vregMul, vregInput, vScale, preg_all);
                    if constexpr (!IsSameType<OUTPUT_T, int8_t>::value) {
                        if constexpr (IsSameType<OUTPUT_T, hifloat8_t>::value) {
                            Add<T, MaskMergeMode::ZEROING>(vreg_add, vregMul, vOffset, preg_all);
                            Cast<OUTPUT_T, T, castTraitP1>(vregCast, vreg_add, preg_all);
                        } else {
                            Add<T, MaskMergeMode::ZEROING>(vreg_add, vregMul, vOffset, preg_all);
                            Cast<OUTPUT_T, T, castTraitP0>(vregCast, vreg_add, preg_all);
                        }
                    } else {
                        Add<T, MaskMergeMode::ZEROING>(vreg_add, vregMul, vOffset, preg_all);
                        Cast<half, T, castTraitP0>(vregCastB16, vreg_add, preg_all);
                        Cast<OUTPUT_T, half, castTraitP0>(vregCast, vregCastB16, preg_all);
                    }
                    DataCopy<OUTPUT_T, StoreDist::DIST_PACK4_B32>(dstUb + i * floatRepSize + j * srcD + m * srcD,
                                                                  vregCast, preg_all);
                }
                // Process tail
                for (uint16_t k = 0; k < dTailLoop; ++k) {
                    if constexpr (IsSameType<POSTQUANT_PARAMS_T, T>::value) {
                        DataCopy<POSTQUANT_PARAMS_T, LoadDist::DIST_NORM>(vScale,
                                                                          scaleUb + dLoops * floatRepSize + j * srcD);
                        DataCopy<POSTQUANT_PARAMS_T, LoadDist::DIST_NORM>(vOffset,
                                                                          offsetUb + dLoops * floatRepSize + j * srcD);
                    } else {
                        DataCopy<POSTQUANT_PARAMS_T, LoadDist::DIST_UNPACK_B16>(
                            vScaleTmp, scaleUb + dLoops * floatRepSize + j * srcD);
                        DataCopy<POSTQUANT_PARAMS_T, LoadDist::DIST_UNPACK_B16>(
                            vOffsetTmp, offsetUb + dLoops * floatRepSize + j * srcD);
                        Cast<T, POSTQUANT_PARAMS_T, castTraitP0>(vScale, vScaleTmp, preg_tail);
                        Cast<T, POSTQUANT_PARAMS_T, castTraitP0>(vOffset, vOffsetTmp, preg_tail);
                    }
                    DataCopy<T, LoadDist::DIST_NORM>(vregInput, srcUb + dLoops * floatRepSize + j * srcD + m * srcD);
                    Mul<T, MaskMergeMode::ZEROING>(vregMul, vregInput, vScale, preg_tail);
                    if constexpr (!IsSameType<OUTPUT_T, int8_t>::value) {
                        if constexpr (IsSameType<OUTPUT_T, hifloat8_t>::value) {
                            Add<T, MaskMergeMode::ZEROING>(vreg_add, vregMul, vOffset, preg_tail);
                            Cast<OUTPUT_T, T, castTraitP1>(vregCast, vreg_add, preg_tail);
                        } else {
                            Add<T, MaskMergeMode::ZEROING>(vreg_add, vregMul, vOffset, preg_tail);
                            Cast<OUTPUT_T, T, castTraitP0>(vregCast, vreg_add, preg_tail);
                        }
                    } else {
                        Add<T, MaskMergeMode::ZEROING>(vreg_add, vregMul, vOffset, preg_tail);
                        Cast<half, T, castTraitP0>(vregCastB16, vreg_add, preg_tail);
                        Cast<OUTPUT_T, half, castTraitP0>(vregCast, vregCastB16, preg_tail);
                    }
                    DataCopy<OUTPUT_T, StoreDist::DIST_PACK4_B32>(dstUb + dLoops * floatRepSize + j * srcD + m * srcD,
                                                                  vregCast, preg_tail);
                }
            }
        }
    }
}

// without offset
template <typename T, typename OUTPUT_T, typename POSTQUANT_PARAMS_T>
__aicore__ inline void PostQuantPerChnlVF(const LocalTensor<OUTPUT_T> &dstTensor, const LocalTensor<T> &srcTensor,
                                      const LocalTensor<POSTQUANT_PARAMS_T> &scaleTensor, const uint16_t gRowCount,
                                      const uint16_t s1RowCount, const uint32_t dSizeV, const uint16_t srcD)
{
    __ubuf__ OUTPUT_T *dstUb = (__ubuf__ OUTPUT_T *)dstTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ POSTQUANT_PARAMS_T *scaleUb = (__ubuf__ POSTQUANT_PARAMS_T *)scaleTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    uint16_t dLoops = dSizeV / floatRepSize;
    uint16_t dTail = dSizeV % floatRepSize;
    uint16_t dTailLoop = dTail > 0 ? 1 : 0;
    uint32_t pltTailD = dTail;

    __VEC_SCOPE__
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
                    if constexpr (IsSameType<POSTQUANT_PARAMS_T, T>::value) {
                        DataCopy<POSTQUANT_PARAMS_T, LoadDist::DIST_NORM>(vScale,
                                                                          scaleUb + i * floatRepSize + j * srcD);
                    } else {
                        DataCopy<POSTQUANT_PARAMS_T, LoadDist::DIST_UNPACK_B16>(vScaleTmp,
                                                                                scaleUb + i * floatRepSize + j * srcD);
                        Cast<T, POSTQUANT_PARAMS_T, castTraitP0>(vScale, vScaleTmp, preg_all);
                    }
                    DataCopy<T, LoadDist::DIST_NORM>(vregInput, srcUb + i * floatRepSize + j * srcD + m * srcD);
                    Mul<T, MaskMergeMode::ZEROING>(vregMul, vregInput, vScale, preg_all);
                    // if constexpr (IsSameType<OUTPUT_T, fp8_e4m3fn_t>::value || IsSameType<OUTPUT_T,
                    // fp8_e5m2_t>::value) {
                    if constexpr (!IsSameType<OUTPUT_T, int8_t>::value) {
                        if constexpr (IsSameType<OUTPUT_T, hifloat8_t>::value) {
                            Cast<OUTPUT_T, T, castTraitP1>(vregCast, vregMul, preg_all);
                        } else {
                            Cast<OUTPUT_T, T, castTraitP0>(vregCast, vregMul, preg_all);
                        }
                    } else {
                        Cast<half, T, castTraitP0>(vregCastB16, vregMul, preg_all);
                        Cast<OUTPUT_T, half, castTraitP0>(vregCast, vregCastB16, preg_all);
                    }
                    DataCopy<OUTPUT_T, StoreDist::DIST_PACK4_B32>(dstUb + i * floatRepSize + j * srcD + m * srcD,
                                                                  vregCast, preg_all);
                }
                // Process tail
                for (uint16_t k = 0; k < dTailLoop; ++k) {
                    if constexpr (IsSameType<POSTQUANT_PARAMS_T, T>::value) {
                        DataCopy<POSTQUANT_PARAMS_T, LoadDist::DIST_NORM>(vScale,
                                                                          scaleUb + dLoops * floatRepSize + j * srcD);
                    } else {
                        DataCopy<POSTQUANT_PARAMS_T, LoadDist::DIST_UNPACK_B16>(
                            vScaleTmp, scaleUb + dLoops * floatRepSize + j * srcD);
                        Cast<T, POSTQUANT_PARAMS_T, castTraitP0>(vScale, vScaleTmp, preg_tail);
                    }
                    DataCopy<T, LoadDist::DIST_NORM>(vregInput, srcUb + dLoops * floatRepSize + j * srcD + m * srcD);
                    Mul<T, MaskMergeMode::ZEROING>(vregMul, vregInput, vScale, preg_tail);
                    // if constexpr (IsSameType<OUTPUT_T, fp8_e4m3fn_t>::value || IsSameType<OUTPUT_T,
                    // fp8_e5m2_t>::value) {
                    if constexpr (!IsSameType<OUTPUT_T, int8_t>::value) {
                        if constexpr (IsSameType<OUTPUT_T, hifloat8_t>::value) {
                            Cast<OUTPUT_T, T, castTraitP1>(vregCast, vregMul, preg_tail);
                        } else {
                            Cast<OUTPUT_T, T, castTraitP0>(vregCast, vregMul, preg_tail);
                        }
                    } else {
                        Cast<half, T, castTraitP0>(vregCastB16, vregMul, preg_tail);
                        Cast<OUTPUT_T, half, castTraitP0>(vregCast, vregCastB16, preg_tail);
                    }
                    DataCopy<OUTPUT_T, StoreDist::DIST_PACK4_B32>(dstUb + dLoops * floatRepSize + j * srcD + m * srcD,
                                                                  vregCast, preg_tail);
                }
            }
        }
    }
}

template <typename T, typename OUTPUT_T, bool hasOffset>
__aicore__ inline void PostQuantPerTensorVF(const LocalTensor<OUTPUT_T> &dstTensor, const LocalTensor<T> &srcTensor,
                                        const float postQuantScaleValue, const float postQuantOffsetValue,
                                        const uint16_t dealRowCount, const uint32_t dSizeV, const uint16_t srcD)
{
    __ubuf__ OUTPUT_T *dstUb = (__ubuf__ OUTPUT_T *)dstTensor.GetPhyAddr();
    __ubuf__ float *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    uint16_t dLoops = dSizeV / floatRepSize;
    uint16_t dTail = dSizeV % floatRepSize;
    uint16_t dTailLoop = dTail > 0 ? 1 : 0;
    uint32_t pltTailD = dTail;
    __VEC_SCOPE__
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
                DataCopy<T, LoadDist::DIST_NORM>(vregInput, srcUb + i * floatRepSize + j * srcD);
                Muls<T, float, MaskMergeMode::ZEROING>(vregMul, vregInput, postQuantScaleValue, preg_all);
                if constexpr (!IsSameType<OUTPUT_T, int8_t>::value) {
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
                } else {
                    if constexpr (hasOffset) {
                        Adds<T, float, MaskMergeMode::ZEROING>(vreg_add, vregMul, postQuantOffsetValue, preg_all);
                        Cast<half, T, castTraitP0>(vregCastB16, vreg_add, preg_all);
                    } else {
                        Cast<half, T, castTraitP0>(vregCastB16, vregMul, preg_all);
                    }
                    Cast<OUTPUT_T, half, castTraitP0>(vregCast, vregCastB16, preg_all);
                }
                DataCopy<OUTPUT_T, StoreDist::DIST_PACK4_B32>(dstUb + i * floatRepSize + j * srcD, vregCast, preg_all);
            }

            // Process tail
            for (uint16_t k = 0; k < dTailLoop; ++k) {
                DataCopy<T, LoadDist::DIST_NORM>(vregInput, srcUb + dLoops * floatRepSize + j * srcD);
                Muls<T, float, MaskMergeMode::ZEROING>(vregMul, vregInput, postQuantScaleValue, preg_tail);
                if constexpr (!IsSameType<OUTPUT_T, int8_t>::value) {
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
                } else {
                    if constexpr (hasOffset) {
                        Adds<T, float, MaskMergeMode::ZEROING>(vreg_add, vregMul, postQuantOffsetValue, preg_tail);
                        Cast<half, T, castTraitP0>(vregCastB16, vreg_add, preg_tail);
                    } else {
                        Cast<half, T, castTraitP0>(vregCastB16, vregMul, preg_tail);
                    }
                    Cast<OUTPUT_T, half, castTraitP0>(vregCast, vregCastB16, preg_tail);
                }
                DataCopy<OUTPUT_T, StoreDist::DIST_PACK4_B32>(dstUb + dLoops * floatRepSize + j * srcD, vregCast,
                                                              preg_tail);
            }
        }
    }
}

} // namespace FaVectorApi

#endif // VF_POST_QUANT_H