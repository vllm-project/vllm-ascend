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
 * \file vf_rope.h
 * \brief
 */

#ifndef VF_ROPE_H
#define VF_ROPE_H

#include "kernel_operator.h"
#include "../compressor_comm.h"

using namespace AscendC;

constexpr MicroAPI::CastTrait castTraitB162B32 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr MicroAPI::CastTrait castTraitB322B16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};


template <typename T, typename ROPET>
__simd_vf__ void HalfModeRopeVF(__ubuf__ T *sinUb, __ubuf__ T *cosUb, __ubuf__ T *inUb, __ubuf__ ROPET *outUb,
                                uint32_t row, uint32_t col, uint32_t actualCol, uint64_t baseAddr)
{
    MicroAPI::RegTensor<T> vregCos;
    MicroAPI::RegTensor<T> vregHalfCos;
    MicroAPI::RegTensor<T> vregSin;
    MicroAPI::RegTensor<T> vregHalfSin;
    MicroAPI::RegTensor<T> vregIn;
    MicroAPI::RegTensor<T> vregHalfIn;
    MicroAPI::RegTensor<T> vregOut;
    MicroAPI::RegTensor<T> vregHalfOut;
    MicroAPI::RegTensor<T> vregCastIn;
    MicroAPI::RegTensor<ROPET> vregOutBf16;
    MicroAPI::RegTensor<ROPET> vregOutHalfBf16;
    MicroAPI::RegTensor<ROPET> vregCastOut;
    uint32_t maskValue = col / 2;
    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
    uint32_t halfCol = col / 2;


    for (uint32_t rIdx = 0; rIdx < row; rIdx++) {
        __ubuf__ T *curSinUb = sinUb + rIdx * col;
        __ubuf__ T *curCosUb = cosUb + rIdx * col;
        __ubuf__ T *curInUb = inUb + rIdx * actualCol;
        __ubuf__ ROPET *curOutUb = outUb + rIdx * actualCol;

        MicroAPI::DataCopy(vregIn, curInUb + baseAddr);
        MicroAPI::DataCopy(vregHalfIn, curInUb + baseAddr + halfCol);
        MicroAPI::DataCopy(vregCos, curCosUb);
        MicroAPI::DataCopy(vregHalfCos, curCosUb + halfCol);
        MicroAPI::DataCopy(vregSin, curSinUb);
        MicroAPI::DataCopy(vregHalfSin, curSinUb + halfCol);
        MicroAPI::Mul(vregSin, vregSin, vregHalfIn, mask);
        MicroAPI::Mul(vregHalfSin, vregHalfSin, vregIn, mask);
        MicroAPI::Mul(vregCos, vregCos, vregIn, mask);
        MicroAPI::Sub(vregOut, vregCos, vregSin, mask);
        MicroAPI::Mul(vregHalfCos, vregHalfCos, vregHalfIn, mask);
        MicroAPI::Add(vregHalfOut, vregHalfSin, vregHalfCos, mask);
        MicroAPI::Cast<ROPET, T, castTraitB322B16>(vregOutBf16, vregOut, mask);
        MicroAPI::DataCopy<ROPET, MicroAPI::StoreDist::DIST_PACK_B32>(curOutUb + baseAddr, vregOutBf16, mask);
        MicroAPI::Cast<ROPET, T, castTraitB322B16>(vregOutHalfBf16, vregHalfOut, mask);
        MicroAPI::DataCopy<ROPET, MicroAPI::StoreDist::DIST_PACK_B32>(curOutUb + baseAddr + halfCol, vregOutHalfBf16,
                                                                      mask);

        for (uint64_t dOffset = 0; dOffset < baseAddr; dOffset += 64) {
            uint32_t castMaskValue = min(baseAddr - dOffset, static_cast<uint64_t>(64));
            MicroAPI::MaskReg castMask = MicroAPI::UpdateMask<T>(castMaskValue);
            MicroAPI::DataCopy(vregCastIn, curInUb + dOffset);
            MicroAPI::Cast<ROPET, T, castTraitB322B16>(vregCastOut, vregCastIn, castMask);
            MicroAPI::DataCopy<ROPET, MicroAPI::StoreDist::DIST_PACK_B32>(curOutUb + dOffset, vregCastOut, castMask);
        }
    }
}


template <typename T, typename ROPET>
__simd_vf__ void InterleaveModeRopeVF(__ubuf__ T *sinUb, __ubuf__ T *cosUb, __ubuf__ T *inUb, __ubuf__ ROPET *outUb,
                                      uint32_t row, uint32_t col, uint32_t actualCol, uint64_t baseAddr)
{
    MicroAPI::RegTensor<T> vregCos;
    MicroAPI::RegTensor<T> vregSin;
    MicroAPI::RegTensor<T> vregIn;
    MicroAPI::RegTensor<T> vregOdd;
    MicroAPI::RegTensor<T> vregEven;
    MicroAPI::RegTensor<T> vregOut;
    MicroAPI::RegTensor<T> vregTemp;
    MicroAPI::RegTensor<T> vregCastIn;
    MicroAPI::RegTensor<ROPET> vregOutBf16;
    MicroAPI::RegTensor<ROPET> vregCastOut;
    uint32_t maskValue = col;
    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);


    for (uint32_t rIdx = 0; rIdx < row; rIdx++) {
        __ubuf__ T *curSinUb = sinUb + rIdx * col;
        __ubuf__ T *curCosUb = cosUb + rIdx * col;
        __ubuf__ T *curInUb = inUb + rIdx * actualCol;
        __ubuf__ ROPET *curOutUb = outUb + rIdx * actualCol;

        MicroAPI::DataCopy(vregIn, curInUb + baseAddr);
        MicroAPI::DataCopy(vregCos, curCosUb);
        MicroAPI::DataCopy(vregSin, curSinUb);
        MicroAPI::Mul(vregCos, vregCos, vregIn, mask);
        MicroAPI::DeInterleave<T>(vregEven, vregOdd, vregIn, vregTemp);
        MicroAPI::Muls(vregOdd, vregOdd, static_cast<T>(-1.0), mask);
        MicroAPI::Interleave<T>(vregIn, vregTemp, vregOdd, vregEven);
        MicroAPI::Mul(vregSin, vregSin, vregIn, mask);
        MicroAPI::Add(vregOut, vregCos, vregSin, mask);
        MicroAPI::Cast<ROPET, T, castTraitB322B16>(vregOutBf16, vregOut, mask);
        MicroAPI::DataCopy<ROPET, MicroAPI::StoreDist::DIST_PACK_B32>(curOutUb + baseAddr, vregOutBf16, mask);
        for (uint64_t dOffset = 0; dOffset < baseAddr; dOffset += 64) {
            uint32_t castMaskValue = min(baseAddr - dOffset, static_cast<uint64_t>(64));
            MicroAPI::MaskReg castMask = MicroAPI::UpdateMask<T>(castMaskValue);
            MicroAPI::DataCopy(vregCastIn, curInUb + dOffset);
            MicroAPI::Cast<ROPET, T, castTraitB322B16>(vregCastOut, vregCastIn, castMask);
            MicroAPI::DataCopy<ROPET, MicroAPI::StoreDist::DIST_PACK_B32>(curOutUb + dOffset, vregCastOut, castMask);
        }
    }
}


template <Compressor::ROTARY_MODE MODE, typename T, typename ROPET>
__aicore__ inline void RopeVF(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                              const LocalTensor<T> &inTensor, const LocalTensor<ROPET> &outTensor, uint32_t row,
                              uint32_t col, uint32_t actualCol, uint64_t baseAddr)
{
    __ubuf__ T *sinUb = (__ubuf__ T *)sinTensor.GetPhyAddr();
    __ubuf__ T *cosUb = (__ubuf__ T *)cosTensor.GetPhyAddr();
    __ubuf__ T *inUb = (__ubuf__ T *)inTensor.GetPhyAddr();
    __ubuf__ ROPET *outUb = (__ubuf__ ROPET *)outTensor.GetPhyAddr();

    if constexpr (MODE == Compressor::ROTARY_MODE::HALF) {
        HalfModeRopeVF(sinUb, cosUb, inUb, outUb, row, col, actualCol, baseAddr);
    } else {
        InterleaveModeRopeVF(sinUb, cosUb, inUb, outUb, row, col, actualCol, baseAddr);
    }
}

#endif
