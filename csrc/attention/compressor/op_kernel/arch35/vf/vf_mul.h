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
#include <cstdint>
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
__simd_callee__ inline T SimdCeilDivT(T num1, T num2)
{
    if (num2 == 0) {
        return static_cast<T>(0);
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
struct ReduceMulRegList {
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vregMul;
    MicroAPI::RegTensor<T> vregSum;
};


template <typename T>
__simd_callee__ void LoadMulAddVFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, ReduceMulRegList<T> &regList, uint64_t offset, uint32_t maskValue)
{
    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
    MicroAPI::LoadAlign(regList.vreg0, kvAddr + offset);
    MicroAPI::LoadAlign(regList.vreg1, scoreAddr + offset);
    MicroAPI::Mul(regList.vregMul, regList.vreg0, regList.vreg1, mask);
    MicroAPI::Add(regList.vregSum, regList.vregSum, regList.vregMul, mask);
}



template <typename T>
__simd_vf__ void MulReduceSumbase8VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                         const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                         const uint32_t baseD)
{
    ReduceMulRegList<T> regList;
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
    uint32_t rCnt = coff * cmpRatio;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(regList.vregSum, 0, mask);
        // 当前仅支持coff * cmpRatio为2的幂的情况
        for (uint32_t rLoop = 0; rLoop < SimdCeilDivT(rCnt, 8U); rLoop++) {
            uint32_t dealLen = min((rCnt - rLoop * 8) * baseD, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList, offset, dealLen);
            offset += dealLen;
        }
        // 64 -> 32
        MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum0, regList.vregSum, maskH32);
        MicroAPI::Add(regList.vregSum, regList.vregSum, vregSum0, maskL32);

        // 32 -> 16
        MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum0, regList.vregSum, maskH48);
        MicroAPI::Add(regList.vregSum, regList.vregSum, vregSum0, maskL16);

        // 16 -> 8
        MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum0, regList.vregSum, maskH56);
        MicroAPI::Add(regList.vregSum, regList.vregSum, vregSum0, maskL8);

        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, regList.vregSum, maskL8);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase16VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                          const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                          const uint32_t baseD)
{
    ReduceMulRegList<T> regList;
    MicroAPI::RegTensor<T> vregSum0;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskL32 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL32>();
    MicroAPI::MaskReg maskL16 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL16>();
    MicroAPI::MaskReg maskH32;
    MicroAPI::MaskReg maskH48;
    MicroAPI::Not(maskH48, maskL16, mask);
    MicroAPI::Not(maskH32, maskL32, mask);
    uint32_t offset = 0;
    uint32_t rCnt = coff * cmpRatio;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(regList.vregSum, 0, mask);
        // 当前仅支持coff * cmpRatio为2的幂的情况
        for (uint32_t rLoop = 0; rLoop < SimdCeilDivT(rCnt, 4U); rLoop++) {
            uint32_t dealLen = min((rCnt - rLoop * 4) * baseD, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList, offset, dealLen);
            offset += dealLen;
        }
        // 64 -> 32
        MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum0, regList.vregSum, maskH32);
        MicroAPI::Add(regList.vregSum, regList.vregSum, vregSum0, maskL32);

        // 32 -> 16
        MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum0, regList.vregSum, maskH48);
        MicroAPI::Add(regList.vregSum, regList.vregSum, vregSum0, maskL16);

        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, regList.vregSum, maskL16);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase32VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                          const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                          const uint32_t baseD)
{
    ReduceMulRegList<T> regList;
    MicroAPI::RegTensor<T> vregSum0;
    MicroAPI::RegTensor<T> vregSum1;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskL32 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL32>();
    MicroAPI::MaskReg maskH32;
    MicroAPI::Not(maskH32, maskL32, mask);
    uint32_t offset = 0;
    uint32_t rCnt = coff * cmpRatio;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(regList.vregSum, 0, mask);
        // 当前仅支持coff * cmpRatio为2的幂的情况
        for (uint32_t rLoop = 0; rLoop < SimdCeilDivT(rCnt, 2U); rLoop++) {
            uint32_t dealLen = min((rCnt - rLoop * 2) * baseD, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList, offset, dealLen);
            offset += dealLen;
        }
        // 64 -> 32
        MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum0, regList.vregSum, maskH32);
        MicroAPI::Add(regList.vregSum, regList.vregSum, vregSum0, maskL32);

        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, regList.vregSum, maskL32);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase64VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                          const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                          const uint32_t baseD)
{
    ReduceMulRegList<T> regList;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t offset = 0;
    uint32_t rCnt = coff * cmpRatio;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(regList.vregSum, 0, mask);
        for (uint32_t rLoop = 0; rLoop < rCnt; rLoop++) {
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList, offset, baseD64);
            offset += baseD;
        }
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, regList.vregSum, mask);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase128VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                           const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                           const uint32_t baseD)
{
    ReduceMulRegList<T> regList[2];
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t offset = 0;
    uint32_t rCnt = coff * cmpRatio;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(regList[0].vregSum, 0, mask);
        MicroAPI::Duplicate(regList[1].vregSum, 0, mask);
        for (uint32_t rLoop = 0; rLoop < rCnt; rLoop++) {
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[0], offset, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[1], offset + baseD64, baseD64);
            offset += baseD;
        }
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, regList[0].vregSum, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + baseD64, regList[1].vregSum, mask);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase256VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                           const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                           const uint32_t baseD)
{
    ReduceMulRegList<T> regList[4];
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t offset = 0;
    uint32_t rCnt = coff * cmpRatio;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(regList[0].vregSum, 0, mask);
        MicroAPI::Duplicate(regList[1].vregSum, 0, mask);
        MicroAPI::Duplicate(regList[2].vregSum, 0, mask);
        MicroAPI::Duplicate(regList[3].vregSum, 0, mask);
        for (uint32_t rLoop = 0; rLoop < rCnt; rLoop++) {
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[0], offset, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[1], offset + baseD64, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[2], offset + 2 * baseD64, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[3], offset + 3 * baseD64, baseD64);
            offset += baseD;
        }
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, regList[0].vregSum, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + baseD64, regList[1].vregSum, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + 2 * baseD64, regList[2].vregSum, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + 3 * baseD64, regList[3].vregSum, mask);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase512VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                           const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                           const uint32_t baseD)
{
    ReduceMulRegList<T> regList[8];
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t offset = 0;
    uint32_t rCnt = coff * cmpRatio;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(regList[0].vregSum, 0, mask);
        MicroAPI::Duplicate(regList[1].vregSum, 0, mask);
        MicroAPI::Duplicate(regList[2].vregSum, 0, mask);
        MicroAPI::Duplicate(regList[3].vregSum, 0, mask);
        MicroAPI::Duplicate(regList[4].vregSum, 0, mask);
        MicroAPI::Duplicate(regList[5].vregSum, 0, mask);
        MicroAPI::Duplicate(regList[6].vregSum, 0, mask);
        MicroAPI::Duplicate(regList[7].vregSum, 0, mask);
        for (uint32_t rLoop = 0; rLoop < rCnt; rLoop++) {
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[0], offset, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[1], offset + baseD64, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[2], offset + 2 * baseD64, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[3], offset + 3 * baseD64, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[4], offset + 4 * baseD64, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[5], offset + 5 * baseD64, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[6], offset + 6 * baseD64, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[7], offset + 7 * baseD64, baseD64);
            offset += baseD;
        }
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, regList[0].vregSum, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + baseD64, regList[1].vregSum, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + 2 * baseD64, regList[2].vregSum, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + 3 * baseD64, regList[3].vregSum, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + 4 * baseD64, regList[4].vregSum, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + 5 * baseD64, regList[5].vregSum, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + 6 * baseD64, regList[6].vregSum, mask);
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD + 7 * baseD64, regList[7].vregSum, mask);
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

// 当前仅支持coff * cmpRatio为2的幂的情况
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