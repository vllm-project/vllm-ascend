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
 * \file vf_mul.h
 * \brief
 */

#ifndef VF_MUL_H
#define VF_MUL_H
#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t FLOATBYTE = 4;
constexpr uint32_t baseD32 = 32;
constexpr uint32_t baseD64 = 64;
constexpr uint32_t baseD128= 128;

template<typename T>
__simd_vf__ void MulReduceSumbase32VFImpl(__ubuf__ T* kvAddr, __ubuf__ T* scoreAddr, __ubuf__ T* outputAddr,
    const uint16_t coff, const uint16_t r, const uint16_t scLoopCnt, const uint16_t baseD) 
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::MaskReg maskHalf = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL32>();
    uint32_t count = baseD;
    for(uint16_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum, 0, maskHalf);
        for(uint16_t rLoop = 0; rLoop < coff*r; rLoop++) {
            MicroAPI::LoadAlign(vreg0, kvAddr + rLoop*baseD);
            MicroAPI::LoadAlign(vreg1, scoreAddr + rLoop*baseD);
            MicroAPI::Mul(vreg0, vreg0, vreg1, maskHalf);
            MicroAPI::Add(vregSum, vregSum, vreg0, maskHalf);
        }
        MicroAPI::StoreAlign(outputAddr + scLoop*baseD, vregSum, maskHalf);
    }
}

template<typename T>
__simd_vf__ void MulReduceSumbase64VFImpl(__ubuf__ T* kvAddr, __ubuf__ T* scoreAddr, __ubuf__ T* outputAddr,
    const uint16_t coff, const uint16_t r, const uint16_t scLoopCnt, const uint16_t baseD) 
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vregMul;
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t offset = 0;
    for(uint16_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum, 0, mask);
        for(uint16_t rLoop = 0; rLoop < coff*r; rLoop++) {
            MicroAPI::LoadAlign(vreg0, kvAddr + offset);
            MicroAPI::LoadAlign(vreg1, scoreAddr + offset);
            MicroAPI::Mul(vregMul, vreg0, vreg1, mask);
            MicroAPI::Add(vregSum, vregSum, vregMul, mask);
            offset += baseD;
        }
        MicroAPI::StoreAlign(outputAddr + scLoop*baseD, vregSum, mask);
    }
}

template<typename T>
__simd_vf__ void MulReduceSumbase128VFImpl(__ubuf__ T* kvAddr, __ubuf__ T* scoreAddr, __ubuf__ T* outputAddr,
    const uint16_t coff, const uint16_t r, const uint16_t scLoopCnt, const uint16_t baseD) 
{
    MicroAPI::RegTensor<T> vreg00;
    MicroAPI::RegTensor<T> vreg01;
    MicroAPI::RegTensor<T> vreg10;
    MicroAPI::RegTensor<T> vreg11;
    MicroAPI::RegTensor<T> vregMul0;
    MicroAPI::RegTensor<T> vregMul1;
    MicroAPI::RegTensor<T> vregSum0;
    MicroAPI::RegTensor<T> vregSum1;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t offset = 0;
    for(uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum0, 0, mask);
        MicroAPI::Duplicate(vregSum1, 0, mask);
        for(uint32_t rLoop = 0; rLoop < coff * r; rLoop++) {
            MicroAPI::LoadAlign(vreg00, kvAddr + offset);
            MicroAPI::LoadAlign(vreg01, kvAddr + offset + baseD64);
            MicroAPI::LoadAlign(vreg10, scoreAddr + offset);
            MicroAPI::LoadAlign(vreg11, scoreAddr + offset + baseD64);
            MicroAPI::Mul(vregMul0, vreg00, vreg10, mask);
            MicroAPI::Mul(vregMul1, vreg01, vreg11, mask);
            MicroAPI::Add(vregSum0, vregSum0, vregMul0, mask);
            MicroAPI::Add(vregSum1, vregSum1, vregMul1, mask);
            offset += baseD;
        }
        MicroAPI::StoreAlign(outputAddr + scLoop*baseD, vregSum0, mask);
        MicroAPI::StoreAlign(outputAddr + baseD64 + scLoop*baseD, vregSum1, mask);
    }
}

/**
 * @brief MulReduceSumbaseVF 包含mul和reducesum
 * @param outputLocal 输出tensor []
 * @param r s方向的最小块
 * @param sc r个sc
 * @param baseD  核内d轴切分大小
 * @param baseS  行数,
 */


template<typename T>
__aicore__ inline void MulReduceSumbaseVF(LocalTensor<T> &kvLocal, LocalTensor<T> &scoreLocal, LocalTensor<T> &outputLocal,
    const uint16_t coff, const uint16_t r, uint32_t outIdx, const uint16_t baseD, const uint32_t baseS) 
{
    uint32_t scLoopCnt = baseS / r;

    __ubuf__ T * kvAddr = (__ubuf__ T *)kvLocal.GetPhyAddr();
    __ubuf__ T * scoreAddr = (__ubuf__ T *)scoreLocal.GetPhyAddr();
    __ubuf__ T * outputAddr = (__ubuf__ T *)outputLocal.GetPhyAddr()+outIdx;
    if(baseD == baseD32) {
        MulReduceSumbase32VFImpl(kvAddr, scoreAddr, outputAddr, coff, r, scLoopCnt, baseD);
    } else if(baseD == baseD64) {
        MulReduceSumbase64VFImpl(kvAddr, scoreAddr, outputAddr, coff, r, scLoopCnt, baseD);
    } else if(baseD == baseD128) {
        MulReduceSumbase128VFImpl(kvAddr, scoreAddr, outputAddr, coff, r, scLoopCnt, baseD);
    }
}


#endif