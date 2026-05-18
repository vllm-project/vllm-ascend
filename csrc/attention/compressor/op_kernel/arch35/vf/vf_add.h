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
 * \file vf_add.h
 * \brief
 */

#ifndef VF_ADD_H
#define VF_ADD_H

#include "kernel_operator.h"
#include <cstdint>
using namespace AscendC;
constexpr uint32_t FLOAT_REP_SIZE = 64;
constexpr uint32_t BTYEALIGNSIZE = 32;
constexpr uint32_t REGSIZE = 256;
constexpr uint32_t HALFCORED = 128;

template <typename T>
__simd_vf__ void AddVFImpl(__ubuf__ T *inputAddr, __ubuf__ T *apeAddr, uint32_t row, uint32_t col, uint32_t actualCol)
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vregape0;
    uint32_t maskValue = col;
    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
    for (uint64_t offset = 0; offset < row * actualCol; offset += actualCol) {
        MicroAPI::LoadAlign(vreg0, inputAddr + offset);
        MicroAPI::LoadAlign(vregape0, apeAddr + offset);
        MicroAPI::Add(vreg0, vreg0, vregape0, mask);
        MicroAPI::StoreAlign(inputAddr + offset, vreg0, mask);
    }
}

template <typename T>
__simd_vf__ void Add128VFImpl(__ubuf__ T *inputAddr, __ubuf__ T *apeAddr, uint32_t row, uint32_t actualCol)
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vregape0;
    MicroAPI::RegTensor<T> vregape1;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    for (uint64_t offset = 0; offset < row * actualCol; offset += actualCol) {
        MicroAPI::LoadAlign(vreg0, inputAddr + offset);
        MicroAPI::LoadAlign(vreg1, inputAddr + offset + FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vregape0, apeAddr + offset);
        MicroAPI::LoadAlign(vregape1, apeAddr + offset + FLOAT_REP_SIZE);
        MicroAPI::Add(vreg0, vreg0, vregape0, mask);
        MicroAPI::Add(vreg1, vreg1, vregape1, mask);
        MicroAPI::StoreAlign(inputAddr + offset, vreg0, mask);
        MicroAPI::StoreAlign(inputAddr + offset + FLOAT_REP_SIZE, vreg1, mask);
    }
}

template <typename T>
__simd_vf__ void Add256VFImpl(__ubuf__ T *inputAddr, __ubuf__ T *apeAddr, uint32_t row, uint32_t actualCol)
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vreg2;
    MicroAPI::RegTensor<T> vreg3;
    MicroAPI::RegTensor<T> vregape0;
    MicroAPI::RegTensor<T> vregape1;
    MicroAPI::RegTensor<T> vregape2;
    MicroAPI::RegTensor<T> vregape3;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    for (uint64_t offset = 0; offset < row * actualCol; offset += actualCol) {
        MicroAPI::LoadAlign(vreg0, inputAddr + offset);
        MicroAPI::LoadAlign(vreg1, inputAddr + offset + FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vreg2, inputAddr + offset + 2 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vreg3, inputAddr + offset + 3 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vregape0, apeAddr + offset);
        MicroAPI::LoadAlign(vregape1, apeAddr + offset + FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vregape2, apeAddr + offset + 2 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vregape3, apeAddr + offset + 3 * FLOAT_REP_SIZE);
        MicroAPI::Add(vreg0, vreg0, vregape0, mask);
        MicroAPI::Add(vreg1, vreg1, vregape1, mask);
        MicroAPI::Add(vreg2, vreg2, vregape2, mask);
        MicroAPI::Add(vreg3, vreg3, vregape3, mask);
        MicroAPI::StoreAlign(inputAddr + offset, vreg0, mask);
        MicroAPI::StoreAlign(inputAddr + offset + FLOAT_REP_SIZE, vreg1, mask);
        MicroAPI::StoreAlign(inputAddr + offset + 2 * FLOAT_REP_SIZE, vreg2, mask);
        MicroAPI::StoreAlign(inputAddr + offset + 3 * FLOAT_REP_SIZE, vreg3, mask);
    }
}

template <typename T>
__simd_vf__ void Add512VFImpl(__ubuf__ T *inputAddr, __ubuf__ T *apeAddr, uint32_t row, uint32_t actualCol)
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vreg2;
    MicroAPI::RegTensor<T> vreg3;
    MicroAPI::RegTensor<T> vreg4;
    MicroAPI::RegTensor<T> vreg5;
    MicroAPI::RegTensor<T> vreg6;
    MicroAPI::RegTensor<T> vreg7;
    MicroAPI::RegTensor<T> vregape0;
    MicroAPI::RegTensor<T> vregape1;
    MicroAPI::RegTensor<T> vregape2;
    MicroAPI::RegTensor<T> vregape3;
    MicroAPI::RegTensor<T> vregape4;
    MicroAPI::RegTensor<T> vregape5;
    MicroAPI::RegTensor<T> vregape6;
    MicroAPI::RegTensor<T> vregape7;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    for (uint64_t offset = 0; offset < row * actualCol; offset += actualCol) {
        MicroAPI::LoadAlign(vreg0, inputAddr + offset);
        MicroAPI::LoadAlign(vreg1, inputAddr + offset + FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vreg2, inputAddr + offset + 2 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vreg3, inputAddr + offset + 3 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vreg4, inputAddr + offset + 4 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vreg5, inputAddr + offset + 5 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vreg6, inputAddr + offset + 6 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vreg7, inputAddr + offset + 7 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vregape0, apeAddr + offset);
        MicroAPI::LoadAlign(vregape1, apeAddr + offset + FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vregape2, apeAddr + offset + 2 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vregape3, apeAddr + offset + 3 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vregape4, apeAddr + offset + 4 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vregape5, apeAddr + offset + 5 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vregape6, apeAddr + offset + 6 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vregape7, apeAddr + offset + 7 * FLOAT_REP_SIZE);
        MicroAPI::Add(vreg0, vreg0, vregape0, mask);
        MicroAPI::Add(vreg1, vreg1, vregape1, mask);
        MicroAPI::Add(vreg2, vreg2, vregape2, mask);
        MicroAPI::Add(vreg3, vreg3, vregape3, mask);
        MicroAPI::Add(vreg4, vreg4, vregape4, mask);
        MicroAPI::Add(vreg5, vreg5, vregape5, mask);
        MicroAPI::Add(vreg6, vreg6, vregape6, mask);
        MicroAPI::Add(vreg7, vreg7, vregape7, mask);
        MicroAPI::StoreAlign(inputAddr + offset, vreg0, mask);
        MicroAPI::StoreAlign(inputAddr + offset + FLOAT_REP_SIZE, vreg1, mask);
        MicroAPI::StoreAlign(inputAddr + offset + 2 * FLOAT_REP_SIZE, vreg2, mask);
        MicroAPI::StoreAlign(inputAddr + offset + 3 * FLOAT_REP_SIZE, vreg3, mask);
        MicroAPI::StoreAlign(inputAddr + offset + 4 * FLOAT_REP_SIZE, vreg4, mask);
        MicroAPI::StoreAlign(inputAddr + offset + 5 * FLOAT_REP_SIZE, vreg5, mask);
        MicroAPI::StoreAlign(inputAddr + offset + 6 * FLOAT_REP_SIZE, vreg6, mask);
        MicroAPI::StoreAlign(inputAddr + offset + 7 * FLOAT_REP_SIZE, vreg7, mask);
    }
}

/**
 * @brief AddVF 输入与apt相加
 * @param rightLocal 输出tensor []
 * @param leftLocal 输入tensor [row, col]
 * @param aptLocal apt输入tensor [r]
 * @param apeIdx ape起始位置
 * @param d  coff*d为ape的D轴大小
 * @param coreSplitD scoreleft大小，coff*coreSplitD为总大小
 * @param coreSplitS 核间d轴切分大小
 */
template <typename T>
__aicore__ inline void AddVF(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeLocal,
                             uint32_t row, uint32_t col, uint32_t actualCol)
{
    __ubuf__ T *scoreAddr = (__ubuf__ T *)scoreLocal.GetPhyAddr();
    __ubuf__ T *apeAddr = (__ubuf__ T *)apeLocal.GetPhyAddr();

    if (col <= 64) {
        AddVFImpl<T>(scoreAddr, apeAddr, row, col, actualCol);
    } else if (col == 128) {
        Add128VFImpl<T>(scoreAddr, apeAddr, row, actualCol);
    } else if (col == 256) {
        Add256VFImpl<T>(scoreAddr, apeAddr, row, actualCol);
    } else if (col == 512) {
        Add512VFImpl<T>(scoreAddr, apeAddr, row, actualCol);
    }
}

#endif
