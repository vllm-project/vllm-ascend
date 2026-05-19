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
 * \file vf_mul.h
 * \brief
 */

#ifndef VF_MUL_H
#define VF_MUL_H
#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t FLOATBYTE = 4;
constexpr uint32_t baseD8 = 8;
constexpr uint32_t baseD16 = 16;
constexpr uint32_t baseD32 = 32;
constexpr uint32_t baseD64 = 64;
constexpr uint32_t baseD128 = 128;
constexpr uint32_t baseD256 = 256;
constexpr uint32_t baseD512 = 512;

template <typename T>
__simd_vf__ void MulReduceSumbase8VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                         const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                         const uint32_t baseD)
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::RegTensor<T> vregSum0;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskL32 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL32>();
    MicroAPI::MaskReg maskL16 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL16>();
    MicroAPI::MaskReg maskL8 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL8>();
    MicroAPI::MaskReg maskH32;
    MicroAPI::MaskReg maskH48;
    MicroAPI::MaskReg maskH56;
    MicroAPI::Not(maskH48, maskL16, mask);
    MicroAPI::Not(maskH32, maskL32, mask);
    MicroAPI::Not(maskH56, maskL8, mask);
    uint32_t offset = 0;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum, 0, mask);
        for (uint32_t rLoop = 0; rLoop < (coff * cmpRatio + 7) / 8; rLoop++) {
            MicroAPI::LoadAlign(vreg0, kvAddr + offset);
            MicroAPI::LoadAlign(vreg1, scoreAddr + offset);
            MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
            MicroAPI::Add(vregSum, vregSum, vreg0, mask);
            offset += min((coff * cmpRatio - rLoop) * baseD, baseD64);
        }
        if (coff * cmpRatio >= 8) {
            MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum0, vregSum, maskH32);
            MicroAPI::Add(vregSum, vregSum, vregSum0, maskL32);
        }
        if (coff * cmpRatio >= 4) {
            MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum0, vregSum, maskH48);
            MicroAPI::Add(vregSum, vregSum, vregSum0, maskL16);
        }
        if (coff * cmpRatio >= 2) {
            MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum0, vregSum, maskH56);
            MicroAPI::Add(vregSum, vregSum, vregSum0, maskL8);
        }
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, vregSum, maskL8);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase16VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                          const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                          const uint32_t baseD)
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::RegTensor<T> vregSum0;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskL32 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL32>();
    MicroAPI::MaskReg maskL16 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL16>();
    MicroAPI::MaskReg maskH32;
    MicroAPI::MaskReg maskH48;
    MicroAPI::Not(maskH48, maskL16, mask);
    MicroAPI::Not(maskH32, maskL32, mask);
    uint32_t offset = 0;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum, 0, mask);
        for (uint32_t rLoop = 0; rLoop < (coff * cmpRatio + 3) / 4; rLoop++) {
            MicroAPI::LoadAlign(vreg0, kvAddr + offset);
            MicroAPI::LoadAlign(vreg1, scoreAddr + offset);
            MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
            MicroAPI::Add(vregSum, vregSum, vreg0, mask);
            offset += min((coff * cmpRatio - rLoop) * baseD, baseD64);
        }
        if (coff * cmpRatio >= 4) {
            MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum0, vregSum, maskH32);
            MicroAPI::Add(vregSum, vregSum, vregSum0, maskL32);
        }
        if (coff * cmpRatio >= 2) {
            MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum0, vregSum, maskH48);
            MicroAPI::Add(vregSum, vregSum, vregSum0, maskL16);
        }
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, vregSum, maskL16);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase32VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                          const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                          const uint32_t baseD)
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::RegTensor<T> vregSum0;
    MicroAPI::RegTensor<T> vregSum1;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskL32 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL32>();
    MicroAPI::MaskReg maskH32;
    MicroAPI::Not(maskH32, maskL32, mask);
    uint32_t offset = 0;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum, 0, mask);
        for (uint32_t rLoop = 0; rLoop < (coff * cmpRatio + 1) / 2; rLoop++) {
            MicroAPI::LoadAlign(vreg0, kvAddr + offset);
            MicroAPI::LoadAlign(vreg1, scoreAddr + offset);
            MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
            MicroAPI::Add(vregSum, vregSum, vreg0, mask);
            offset += min((coff * cmpRatio - rLoop) * baseD, baseD64);
        }
        MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum0, vregSum, maskL32);
        if (coff * cmpRatio >= 2) {
            MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum1, vregSum, maskH32);
            MicroAPI::Add(vregSum0, vregSum0, vregSum1, maskL32);
        }
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, vregSum0, maskL32);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase64VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                          const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                          const uint32_t baseD)
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vregMul;
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t offset = 0;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum, 0, mask);
        for (uint32_t rLoop = 0; rLoop < coff * cmpRatio; rLoop++) {
            MicroAPI::LoadAlign(vreg0, kvAddr + offset);
            MicroAPI::LoadAlign(vreg1, scoreAddr + offset);
            MicroAPI::Mul(vregMul, vreg0, vreg1, mask);
            MicroAPI::Add(vregSum, vregSum, vregMul, mask);
            offset += baseD;
        }
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, vregSum, mask);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase128VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                           const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                           const uint32_t baseD)
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
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum0, 0, mask);
        MicroAPI::Duplicate(vregSum1, 0, mask);
        for (uint32_t rLoop = 0; rLoop < coff * cmpRatio; rLoop++) {
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
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, vregSum0, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + baseD64, vregSum1, mask);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase256VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                           const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                           const uint32_t baseD)
{
    MicroAPI::RegTensor<T> vreg00;
    MicroAPI::RegTensor<T> vreg01;
    MicroAPI::RegTensor<T> vreg02;
    MicroAPI::RegTensor<T> vreg03;
    MicroAPI::RegTensor<T> vreg10;
    MicroAPI::RegTensor<T> vreg11;
    MicroAPI::RegTensor<T> vreg12;
    MicroAPI::RegTensor<T> vreg13;
    MicroAPI::RegTensor<T> vregMul0;
    MicroAPI::RegTensor<T> vregMul1;
    MicroAPI::RegTensor<T> vregMul2;
    MicroAPI::RegTensor<T> vregMul3;
    MicroAPI::RegTensor<T> vregSum0;
    MicroAPI::RegTensor<T> vregSum1;
    MicroAPI::RegTensor<T> vregSum2;
    MicroAPI::RegTensor<T> vregSum3;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t offset = 0;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum0, 0, mask);
        MicroAPI::Duplicate(vregSum1, 0, mask);
        MicroAPI::Duplicate(vregSum2, 0, mask);
        MicroAPI::Duplicate(vregSum3, 0, mask);
        for (uint32_t rLoop = 0; rLoop < coff * cmpRatio; rLoop++) {
            MicroAPI::LoadAlign(vreg00, kvAddr + offset);
            MicroAPI::LoadAlign(vreg01, kvAddr + offset + baseD64);
            MicroAPI::LoadAlign(vreg02, kvAddr + offset + 2 * baseD64);
            MicroAPI::LoadAlign(vreg03, kvAddr + offset + 3 * baseD64);
            MicroAPI::LoadAlign(vreg10, scoreAddr + offset);
            MicroAPI::LoadAlign(vreg11, scoreAddr + offset + baseD64);
            MicroAPI::LoadAlign(vreg12, scoreAddr + offset + 2 * baseD64);
            MicroAPI::LoadAlign(vreg13, scoreAddr + offset + 3 * baseD64);
            MicroAPI::Mul(vregMul0, vreg00, vreg10, mask);
            MicroAPI::Mul(vregMul1, vreg01, vreg11, mask);
            MicroAPI::Mul(vregMul2, vreg02, vreg12, mask);
            MicroAPI::Mul(vregMul3, vreg03, vreg13, mask);
            MicroAPI::Add(vregSum0, vregSum0, vregMul0, mask);
            MicroAPI::Add(vregSum1, vregSum1, vregMul1, mask);
            MicroAPI::Add(vregSum2, vregSum2, vregMul2, mask);
            MicroAPI::Add(vregSum3, vregSum3, vregMul3, mask);
            offset += baseD;
        }
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, vregSum0, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + baseD64, vregSum1, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + 2 * baseD64, vregSum2, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + 3 * baseD64, vregSum3, mask);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase512VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                           const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                           const uint32_t baseD)
{
    MicroAPI::RegTensor<T> vreg00;
    MicroAPI::RegTensor<T> vreg01;
    MicroAPI::RegTensor<T> vreg02;
    MicroAPI::RegTensor<T> vreg03;
    MicroAPI::RegTensor<T> vreg04;
    MicroAPI::RegTensor<T> vreg05;
    MicroAPI::RegTensor<T> vreg06;
    MicroAPI::RegTensor<T> vreg07;
    MicroAPI::RegTensor<T> vreg10;
    MicroAPI::RegTensor<T> vreg11;
    MicroAPI::RegTensor<T> vreg12;
    MicroAPI::RegTensor<T> vreg13;
    MicroAPI::RegTensor<T> vreg14;
    MicroAPI::RegTensor<T> vreg15;
    MicroAPI::RegTensor<T> vreg16;
    MicroAPI::RegTensor<T> vreg17;
    MicroAPI::RegTensor<T> vregMul0;
    MicroAPI::RegTensor<T> vregMul1;
    MicroAPI::RegTensor<T> vregMul2;
    MicroAPI::RegTensor<T> vregMul3;
    MicroAPI::RegTensor<T> vregMul4;
    MicroAPI::RegTensor<T> vregMul5;
    MicroAPI::RegTensor<T> vregMul6;
    MicroAPI::RegTensor<T> vregMul7;
    MicroAPI::RegTensor<T> vregSum0;
    MicroAPI::RegTensor<T> vregSum1;
    MicroAPI::RegTensor<T> vregSum2;
    MicroAPI::RegTensor<T> vregSum3;
    MicroAPI::RegTensor<T> vregSum4;
    MicroAPI::RegTensor<T> vregSum5;
    MicroAPI::RegTensor<T> vregSum6;
    MicroAPI::RegTensor<T> vregSum7;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t offset = 0;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum0, 0, mask);
        MicroAPI::Duplicate(vregSum1, 0, mask);
        MicroAPI::Duplicate(vregSum2, 0, mask);
        MicroAPI::Duplicate(vregSum3, 0, mask);
        MicroAPI::Duplicate(vregSum4, 0, mask);
        MicroAPI::Duplicate(vregSum5, 0, mask);
        MicroAPI::Duplicate(vregSum6, 0, mask);
        MicroAPI::Duplicate(vregSum7, 0, mask);
        for (uint32_t rLoop = 0; rLoop < coff * cmpRatio; rLoop++) {
            MicroAPI::LoadAlign(vreg00, kvAddr + offset);
            MicroAPI::LoadAlign(vreg01, kvAddr + offset + baseD64);
            MicroAPI::LoadAlign(vreg02, kvAddr + offset + 2 * baseD64);
            MicroAPI::LoadAlign(vreg03, kvAddr + offset + 3 * baseD64);
            MicroAPI::LoadAlign(vreg04, kvAddr + offset + 4 * baseD64);
            MicroAPI::LoadAlign(vreg05, kvAddr + offset + 5 * baseD64);
            MicroAPI::LoadAlign(vreg06, kvAddr + offset + 6 * baseD64);
            MicroAPI::LoadAlign(vreg07, kvAddr + offset + 7 * baseD64);
            MicroAPI::LoadAlign(vreg10, scoreAddr + offset);
            MicroAPI::LoadAlign(vreg11, scoreAddr + offset + baseD64);
            MicroAPI::LoadAlign(vreg12, scoreAddr + offset + 2 * baseD64);
            MicroAPI::LoadAlign(vreg13, scoreAddr + offset + 3 * baseD64);
            MicroAPI::LoadAlign(vreg14, scoreAddr + offset + 4 * baseD64);
            MicroAPI::LoadAlign(vreg15, scoreAddr + offset + 5 * baseD64);
            MicroAPI::LoadAlign(vreg16, scoreAddr + offset + 6 * baseD64);
            MicroAPI::LoadAlign(vreg17, scoreAddr + offset + 7 * baseD64);
            MicroAPI::Mul(vregMul0, vreg00, vreg10, mask);
            MicroAPI::Mul(vregMul1, vreg01, vreg11, mask);
            MicroAPI::Mul(vregMul2, vreg02, vreg12, mask);
            MicroAPI::Mul(vregMul3, vreg03, vreg13, mask);
            MicroAPI::Mul(vregMul4, vreg04, vreg14, mask);
            MicroAPI::Mul(vregMul5, vreg05, vreg15, mask);
            MicroAPI::Mul(vregMul6, vreg06, vreg16, mask);
            MicroAPI::Mul(vregMul7, vreg07, vreg17, mask);
            MicroAPI::Add(vregSum0, vregSum0, vregMul0, mask);
            MicroAPI::Add(vregSum1, vregSum1, vregMul1, mask);
            MicroAPI::Add(vregSum2, vregSum2, vregMul2, mask);
            MicroAPI::Add(vregSum3, vregSum3, vregMul3, mask);
            MicroAPI::Add(vregSum4, vregSum4, vregMul4, mask);
            MicroAPI::Add(vregSum5, vregSum5, vregMul5, mask);
            MicroAPI::Add(vregSum6, vregSum6, vregMul6, mask);
            MicroAPI::Add(vregSum7, vregSum7, vregMul7, mask);
            offset += baseD;
        }
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, vregSum0, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + baseD64, vregSum1, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + 2 * baseD64, vregSum2, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + 3 * baseD64, vregSum3, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + 4 * baseD64, vregSum4, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + 5 * baseD64, vregSum5, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + 6 * baseD64, vregSum6, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + 7 * baseD64, vregSum7, mask);
    }
}

/**
 * @brief MulReduceSumbaseVF 包含mul和reducesum
 * @param outputLocal 输出tensor []
 * @param coff
 * @param cmpRatio 压缩块大小
 * @param baseD  核内d轴切分大小
 * @param scLoopCnt  sc数,
 */

//当前仅支持coff * cmpRatio为2的幂的情况
template <typename T>
__aicore__ inline void MulReduceSumbaseVF(const LocalTensor<T> &kvLocal, const LocalTensor<T> &scoreLocal,
                                          const LocalTensor<T> &outputLocal, const uint32_t coff, const uint32_t cmpRatio,
                                          const uint32_t baseD, const uint32_t scLoopCnt)
{

    __ubuf__ T *kvAddr = (__ubuf__ T *)kvLocal.GetPhyAddr();
    __ubuf__ T *scoreAddr = (__ubuf__ T *)scoreLocal.GetPhyAddr();
    __ubuf__ T *outputAddr = (__ubuf__ T *)outputLocal.GetPhyAddr();
    if (baseD == baseD8) {
        MulReduceSumbase8VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD16) {
        MulReduceSumbase16VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD32) {
        MulReduceSumbase32VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD64) {
        MulReduceSumbase64VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD128) {
        MulReduceSumbase128VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD256) {
        MulReduceSumbase256VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD512) {
        MulReduceSumbase512VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    }
}


#endif