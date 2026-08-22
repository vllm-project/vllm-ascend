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
 * \file vf_post_quant_common.h
 * \brief shared __simd_callee__ helpers for post-quantization VF kernels
 */

#ifndef VF_POST_QUANT_COMMON_H
#define VF_POST_QUANT_COMMON_H

#include "kernel_tensor.h"

namespace FaVectorApi {

static constexpr MicroAPI::CastTrait castTraitP0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
static constexpr MicroAPI::CastTrait castTraitP1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};

template<typename T, typename OUTPUT_T>
__simd_callee__ inline void CastToOutputFromMul(RegTensor<half> &vregCastB16, RegTensor<OUTPUT_T> &vregCast,
    RegTensor<T> &vregMul, MaskReg &mask)
{
    if constexpr (!IsSameType<OUTPUT_T, int8_t>::value) {
        if constexpr (IsSameType<OUTPUT_T, hifloat8_t>::value) {
            Cast<OUTPUT_T, T, castTraitP1>(vregCast, vregMul, mask);
        } else {
            Cast<OUTPUT_T, T, castTraitP0>(vregCast, vregMul, mask);
        }
    } else {
        Cast<half, T, castTraitP0>(vregCastB16, vregMul, mask);
        Cast<OUTPUT_T, half, castTraitP0>(vregCast, vregCastB16, mask);
    }
}

template<typename T, typename OUTPUT_T>
__simd_callee__ inline void CastToOutputFromAdd(RegTensor<half> &vregCastB16, RegTensor<OUTPUT_T> &vregCast,
    RegTensor<T> &vregAdd, MaskReg &mask)
{
    if constexpr (!IsSameType<OUTPUT_T, int8_t>::value) {
        if constexpr (IsSameType<OUTPUT_T, hifloat8_t>::value) {
            Cast<OUTPUT_T, T, castTraitP1>(vregCast, vregAdd, mask);
        } else {
            Cast<OUTPUT_T, T, castTraitP0>(vregCast, vregAdd, mask);
        }
    } else {
        Cast<half, T, castTraitP0>(vregCastB16, vregAdd, mask);
        Cast<OUTPUT_T, half, castTraitP0>(vregCast, vregCastB16, mask);
    }
}

template<typename T, typename POSTQUANT_PARAMS_T>
__simd_callee__ inline void LoadScaleParam(RegTensor<T> &vScale, RegTensor<POSTQUANT_PARAMS_T> &vScaleTmp,
    __ubuf__ POSTQUANT_PARAMS_T *scaleUb, uint32_t offset, MaskReg &mask)
{
    if constexpr (IsSameType<POSTQUANT_PARAMS_T, T>::value) {
        LoadAlign<POSTQUANT_PARAMS_T, LoadDist::DIST_NORM>(vScale, scaleUb + offset);
    } else {
        LoadAlign<POSTQUANT_PARAMS_T, LoadDist::DIST_UNPACK_B16>(vScaleTmp, scaleUb + offset);
        Cast<T, POSTQUANT_PARAMS_T, castTraitP0>(vScale, vScaleTmp, mask);
    }
}

template<typename T, typename POSTQUANT_PARAMS_T>
__simd_callee__ inline void LoadOffsetParam(RegTensor<T> &vOffset, RegTensor<POSTQUANT_PARAMS_T> &vOffsetTmp,
    __ubuf__ POSTQUANT_PARAMS_T *offsetUb, uint32_t offset, MaskReg &mask)
{
    if constexpr (IsSameType<POSTQUANT_PARAMS_T, T>::value) {
        LoadAlign<POSTQUANT_PARAMS_T, LoadDist::DIST_NORM>(vOffset, offsetUb + offset);
    } else {
        LoadAlign<POSTQUANT_PARAMS_T, LoadDist::DIST_UNPACK_B16>(vOffsetTmp, offsetUb + offset);
        Cast<T, POSTQUANT_PARAMS_T, castTraitP0>(vOffset, vOffsetTmp, mask);
    }
}

} // namespace FaVectorApi

#endif // VF_POST_QUANT_COMMON_H
