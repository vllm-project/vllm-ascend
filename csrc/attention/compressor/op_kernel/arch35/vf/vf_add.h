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
struct AddRegList {
    MicroAPI::RegTensor<T> vreg;
    MicroAPI::RegTensor<T> vregape;
};


template <typename T>
__simd_callee__ void AddVFImpl(__ubuf__ T *inputAddr, __ubuf__ T *apeAddr, AddRegList<T> &regList, uint32_t row,
                               uint32_t col, uint64_t offset0, uint64_t offset1)
{
    uint32_t maskValue = col;
    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
    MicroAPI::LoadAlign(regList.vreg, inputAddr + offset0);
    MicroAPI::LoadAlign(regList.vregape, apeAddr + offset1);
    MicroAPI::Add(regList.vreg, regList.vreg, regList.vregape, mask);
    MicroAPI::StoreAlign(inputAddr + offset0, regList.vreg, mask);
}

template <bool IS_FIRST, typename T>
__simd_callee__ void MultiAddVFImpl(__ubuf__ T *outputAddr, __ubuf__ T *inputAddr, AddRegList<T> &regList, uint32_t row,
                                    uint32_t col, uint64_t offset, uint32_t repeatNum, uint64_t repeatOffset)
{
    uint32_t maskValue = col;
    uint32_t initialRepeatIdx = IS_FIRST ? 1 : 0;
    __ubuf__ T *initialAddr = IS_FIRST ? inputAddr : outputAddr;
    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
    MicroAPI::LoadAlign(regList.vreg, initialAddr + offset);
    for (uint32_t repeatIdx = initialRepeatIdx; repeatIdx < repeatNum; repeatIdx++) {
        uint64_t addOffset = offset + repeatIdx * repeatOffset;
        MicroAPI::LoadAlign(regList.vregape, inputAddr + addOffset);
        MicroAPI::Add(regList.vreg, regList.vreg, regList.vregape, mask);
    }
    MicroAPI::StoreAlign(outputAddr + offset, regList.vreg, mask);
}

template <typename T>
__simd_vf__ void Add64VFImpl(__ubuf__ T *inputAddr, __ubuf__ T *apeAddr, uint32_t row, uint32_t col, uint32_t actualCol0, uint32_t actualCol1)
{
    AddRegList<T> regList[4];
    uint32_t loopTimes = row / 4;
    for (uint32_t idx = 0; idx < loopTimes; idx++) {
        uint64_t offset0 = idx * 4 * actualCol0;
        uint64_t offset1 = idx * 4 * actualCol1;
        AddVFImpl(inputAddr, apeAddr, regList[0], row, col, offset0, offset1);
        AddVFImpl(inputAddr, apeAddr, regList[1], row, col, offset0 + actualCol0, offset1 + actualCol1);
        AddVFImpl(inputAddr, apeAddr, regList[2], row, col, offset0 + 2 * actualCol0, offset1 + 2 * actualCol1);
        AddVFImpl(inputAddr, apeAddr, regList[3], row, col, offset0 + 3 * actualCol0, offset1 + 3 * actualCol1);
    }

    if (row % 4 > 0) {
        AddVFImpl(inputAddr, apeAddr, regList[0], row, col, loopTimes * 4 * actualCol0, loopTimes * 4 * actualCol1);
    }

    if (row % 4 > 1) {
        AddVFImpl(inputAddr, apeAddr, regList[1], row, col, (loopTimes * 4 + 1) * actualCol0, (loopTimes * 4 + 1) * actualCol1);
    }

    if (row % 4 > 2) {
        AddVFImpl(inputAddr, apeAddr, regList[2], row, col, (loopTimes * 4 + 2) * actualCol0, (loopTimes * 4 + 2) * actualCol1);
    }
}

template <typename T>
__simd_vf__ void Add128VFImpl(__ubuf__ T *inputAddr, __ubuf__ T *apeAddr, uint32_t row, uint32_t actualCol0, uint32_t actualCol1)
{
    AddRegList<T> regList[4];
    uint32_t loopTimes = row / 2;
    for (uint32_t idx = 0; idx < loopTimes; idx++) {
        uint64_t offset0 = idx * 2 * actualCol0;
        uint64_t offset1 = idx * 2 * actualCol1;
        AddVFImpl(inputAddr, apeAddr, regList[0], row, FLOAT_REP_SIZE, offset0, offset1);
        AddVFImpl(inputAddr, apeAddr, regList[1], row, FLOAT_REP_SIZE, offset0 + FLOAT_REP_SIZE, offset1 + FLOAT_REP_SIZE);
        AddVFImpl(inputAddr, apeAddr, regList[2], row, FLOAT_REP_SIZE, offset0 + actualCol0, offset1 + actualCol1);
        AddVFImpl(inputAddr, apeAddr, regList[3], row, FLOAT_REP_SIZE, offset0 + actualCol0 + FLOAT_REP_SIZE, offset1 + actualCol1 + FLOAT_REP_SIZE);
    }

    if (row % 2 > 0) {
        AddVFImpl(inputAddr, apeAddr, regList[0], row, FLOAT_REP_SIZE, loopTimes * 2 * actualCol0, loopTimes * 2 * actualCol1);
        AddVFImpl(inputAddr, apeAddr, regList[1], row, FLOAT_REP_SIZE, loopTimes * 2 * actualCol0 + FLOAT_REP_SIZE, loopTimes * 2 * actualCol1 + FLOAT_REP_SIZE);
    }
}

template <typename T>
__simd_vf__ void Add256VFImpl(__ubuf__ T *inputAddr, __ubuf__ T *apeAddr, uint32_t row, uint32_t actualCol0, uint32_t actualCol1)
{
    AddRegList<T> regList[4];
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    for (uint32_t idx = 0; idx < row; idx++) {
        uint64_t offset0 = idx * actualCol0;
        uint64_t offset1 = idx * actualCol1;
        AddVFImpl(inputAddr, apeAddr, regList[0], row, FLOAT_REP_SIZE, offset0, offset1);
        AddVFImpl(inputAddr, apeAddr, regList[1], row, FLOAT_REP_SIZE, offset0 + FLOAT_REP_SIZE, offset1 + FLOAT_REP_SIZE);
        AddVFImpl(inputAddr, apeAddr, regList[2], row, FLOAT_REP_SIZE, offset0 + 2 * FLOAT_REP_SIZE, offset1 + 2 * FLOAT_REP_SIZE);
        AddVFImpl(inputAddr, apeAddr, regList[3], row, FLOAT_REP_SIZE, offset0 + 3 * FLOAT_REP_SIZE, offset1 + 3 * FLOAT_REP_SIZE);
    }
}

template <typename T>
__simd_vf__ void Add512VFImpl(__ubuf__ T *inputAddr, __ubuf__ T *apeAddr, uint32_t row, uint32_t actualCol0, uint32_t actualCol1)
{
    AddRegList<T> regList[8];
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    for (uint32_t idx = 0; idx < row; idx++) {
        uint64_t offset0 = idx * actualCol0;
        uint64_t offset1 = idx * actualCol1;
        AddVFImpl(inputAddr, apeAddr, regList[0], row, FLOAT_REP_SIZE, offset0, offset1);
        AddVFImpl(inputAddr, apeAddr, regList[1], row, FLOAT_REP_SIZE, offset0 + FLOAT_REP_SIZE, offset1 + FLOAT_REP_SIZE);
        AddVFImpl(inputAddr, apeAddr, regList[2], row, FLOAT_REP_SIZE, offset0 + 2 * FLOAT_REP_SIZE, offset1 + 2 * FLOAT_REP_SIZE);
        AddVFImpl(inputAddr, apeAddr, regList[3], row, FLOAT_REP_SIZE, offset0 + 3 * FLOAT_REP_SIZE, offset1 + 3 * FLOAT_REP_SIZE);
        AddVFImpl(inputAddr, apeAddr, regList[4], row, FLOAT_REP_SIZE, offset0 + 4 * FLOAT_REP_SIZE, offset1 + 4 * FLOAT_REP_SIZE);
        AddVFImpl(inputAddr, apeAddr, regList[5], row, FLOAT_REP_SIZE, offset0 + 5 * FLOAT_REP_SIZE, offset1 + 5 * FLOAT_REP_SIZE);
        AddVFImpl(inputAddr, apeAddr, regList[6], row, FLOAT_REP_SIZE, offset0 + 6 * FLOAT_REP_SIZE, offset1 + 6 * FLOAT_REP_SIZE);
        AddVFImpl(inputAddr, apeAddr, regList[7], row, FLOAT_REP_SIZE, offset0 + 7 * FLOAT_REP_SIZE, offset1 + 7 * FLOAT_REP_SIZE);
    }
}

template <bool IS_FIRST, typename T>
__simd_vf__ void MultiAdd64VFImpl(__ubuf__ T *outputAddr, __ubuf__ T *inputAddr, uint32_t row, uint32_t col,
                                  uint32_t actualCol, uint32_t repeatNum, uint64_t repeatOffset)
{
    AddRegList<T> regList[4];
    uint32_t loopTimes = row / 4;
    uint32_t maskValue = col;
    uint32_t initialRepeatIdx = IS_FIRST ? 1 : 0;
    __ubuf__ T *initialAddr = IS_FIRST ? inputAddr : outputAddr;
    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
    for (uint32_t idx = 0; idx < loopTimes; idx++) {
        uint64_t offset = idx * 4 * actualCol;
        MicroAPI::LoadAlign(regList[0].vreg, initialAddr + offset);
        MicroAPI::LoadAlign(regList[1].vreg, initialAddr + offset + actualCol);
        MicroAPI::LoadAlign(regList[2].vreg, initialAddr + offset + 2 * actualCol);
        MicroAPI::LoadAlign(regList[3].vreg, initialAddr + offset + 3 * actualCol);
        for (uint32_t repeatIdx = initialRepeatIdx; repeatIdx < repeatNum; repeatIdx++) {
            uint64_t addOffset = offset + repeatIdx * repeatOffset;
            MicroAPI::LoadAlign(regList[0].vregape, inputAddr + addOffset);
            MicroAPI::LoadAlign(regList[1].vregape, inputAddr + addOffset + actualCol);
            MicroAPI::LoadAlign(regList[2].vregape, inputAddr + addOffset + 2 * actualCol);
            MicroAPI::LoadAlign(regList[3].vregape, inputAddr + addOffset + 3 * actualCol);
            MicroAPI::Add(regList[0].vreg, regList[0].vreg, regList[0].vregape, mask);
            MicroAPI::Add(regList[1].vreg, regList[1].vreg, regList[1].vregape, mask);
            MicroAPI::Add(regList[2].vreg, regList[2].vreg, regList[2].vregape, mask);
            MicroAPI::Add(regList[3].vreg, regList[3].vreg, regList[3].vregape, mask);
        }
        MicroAPI::StoreAlign(outputAddr + offset, regList[0].vreg, mask);
        MicroAPI::StoreAlign(outputAddr + offset + actualCol, regList[1].vreg, mask);
        MicroAPI::StoreAlign(outputAddr + offset + 2 * actualCol, regList[2].vreg, mask);
        MicroAPI::StoreAlign(outputAddr + offset + 3 * actualCol, regList[3].vreg, mask);
    }

    if (row % 4 > 0) {
        MultiAddVFImpl<IS_FIRST, T>(outputAddr, inputAddr, regList[0], row, col, loopTimes * 4 * actualCol, repeatNum,
                                 repeatOffset);
    }

    if (row % 4 > 1) {
        MultiAddVFImpl<IS_FIRST, T>(outputAddr, inputAddr, regList[1], row, col, (loopTimes * 4 + 1) * actualCol,
                                 repeatNum, repeatOffset);
    }

    if (row % 4 > 2) {
        MultiAddVFImpl<IS_FIRST, T>(outputAddr, inputAddr, regList[2], row, col, (loopTimes * 4 + 2) * actualCol,
                                 repeatNum, repeatOffset);
    }
}

template <bool IS_FIRST, typename T>
__simd_vf__ void MultiAdd128VFImpl(__ubuf__ T *outputAddr, __ubuf__ T *inputAddr, uint32_t row, uint32_t col,
                                   uint32_t actualCol, uint32_t repeatNum, uint64_t repeatOffset)
{
    AddRegList<T> regList[4];
    uint32_t loopTimes = row / 2;
    uint32_t initialRepeatIdx = IS_FIRST ? 1 : 0;
    __ubuf__ T *initialAddr = IS_FIRST ? inputAddr : outputAddr;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    for (uint32_t idx = 0; idx < loopTimes; idx++) {
        uint64_t offset = idx * actualCol * 2;
        MicroAPI::LoadAlign(regList[0].vreg, initialAddr + offset);
        MicroAPI::LoadAlign(regList[1].vreg, initialAddr + offset + FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(regList[2].vreg, initialAddr + offset + actualCol);
        MicroAPI::LoadAlign(regList[3].vreg, initialAddr + offset + actualCol + FLOAT_REP_SIZE);
        for (uint32_t repeatIdx = initialRepeatIdx; repeatIdx < repeatNum; repeatIdx++) {
            uint64_t addOffset = offset + repeatIdx * repeatOffset;
            MicroAPI::LoadAlign(regList[0].vregape, inputAddr + addOffset);
            MicroAPI::LoadAlign(regList[1].vregape, inputAddr + addOffset + FLOAT_REP_SIZE);
            MicroAPI::LoadAlign(regList[2].vregape, inputAddr + addOffset + actualCol);
            MicroAPI::LoadAlign(regList[3].vregape, inputAddr + addOffset + actualCol + FLOAT_REP_SIZE);
            MicroAPI::Add(regList[0].vreg, regList[0].vreg, regList[0].vregape, mask);
            MicroAPI::Add(regList[1].vreg, regList[1].vreg, regList[1].vregape, mask);
            MicroAPI::Add(regList[2].vreg, regList[2].vreg, regList[2].vregape, mask);
            MicroAPI::Add(regList[3].vreg, regList[3].vreg, regList[3].vregape, mask);
        }
        MicroAPI::StoreAlign(outputAddr + offset, regList[0].vreg, mask);
        MicroAPI::StoreAlign(outputAddr + offset + FLOAT_REP_SIZE, regList[1].vreg, mask);
        MicroAPI::StoreAlign(outputAddr + offset + actualCol, regList[2].vreg, mask);
        MicroAPI::StoreAlign(outputAddr + offset + actualCol + FLOAT_REP_SIZE, regList[3].vreg, mask);
    }

    if (row % 2 > 0) {
        MultiAddVFImpl<IS_FIRST, T>(outputAddr, inputAddr, regList[0], row, col, loopTimes * 2 * actualCol, repeatNum,
                                 repeatOffset);
        MultiAddVFImpl<IS_FIRST, T>(outputAddr, inputAddr, regList[1], row, col,
                                 loopTimes * 2 * actualCol + FLOAT_REP_SIZE, repeatNum, repeatOffset);
    }
}

template <bool IS_FIRST, typename T>
__simd_vf__ void MultiAdd256VFImpl(__ubuf__ T *outputAddr, __ubuf__ T *inputAddr, uint32_t row,
                                   uint32_t actualCol, uint32_t repeatNum, uint64_t repeatOffset)
{
    AddRegList<T> regList[4];
    uint32_t loopTimes = row;
    uint32_t initialRepeatIdx = IS_FIRST ? 1 : 0;
    __ubuf__ T *initialAddr = IS_FIRST ? inputAddr : outputAddr;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    for (uint32_t idx = 0; idx < loopTimes; idx++) {
        uint64_t offset = idx * actualCol;
        MicroAPI::LoadAlign(regList[0].vreg, initialAddr + offset);
        MicroAPI::LoadAlign(regList[1].vreg, initialAddr + offset + FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(regList[2].vreg, initialAddr + offset + 2 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(regList[3].vreg, initialAddr + offset + 3 * FLOAT_REP_SIZE);
        for (uint32_t repeatIdx = initialRepeatIdx; repeatIdx < repeatNum; repeatIdx++) {
            uint64_t addOffset = offset + repeatIdx * repeatOffset;
            MicroAPI::LoadAlign(regList[0].vregape, inputAddr + addOffset);
            MicroAPI::LoadAlign(regList[1].vregape, inputAddr + addOffset + FLOAT_REP_SIZE);
            MicroAPI::LoadAlign(regList[2].vregape, inputAddr + addOffset + 2 * FLOAT_REP_SIZE);
            MicroAPI::LoadAlign(regList[3].vregape, inputAddr + addOffset + 3 * FLOAT_REP_SIZE);
            MicroAPI::Add(regList[0].vreg, regList[0].vreg, regList[0].vregape, mask);
            MicroAPI::Add(regList[1].vreg, regList[1].vreg, regList[1].vregape, mask);
            MicroAPI::Add(regList[2].vreg, regList[2].vreg, regList[2].vregape, mask);
            MicroAPI::Add(regList[3].vreg, regList[3].vreg, regList[3].vregape, mask);
        }
        MicroAPI::StoreAlign(outputAddr + offset, regList[0].vreg, mask);
        MicroAPI::StoreAlign(outputAddr + offset + FLOAT_REP_SIZE, regList[1].vreg, mask);
        MicroAPI::StoreAlign(outputAddr + offset + 2 * FLOAT_REP_SIZE, regList[2].vreg, mask);
        MicroAPI::StoreAlign(outputAddr + offset + 3 * FLOAT_REP_SIZE, regList[3].vreg, mask);
    }
}

template <bool IS_FIRST, typename T>
__simd_vf__ void MultiAdd512VFImpl(__ubuf__ T *outputAddr, __ubuf__ T *inputAddr, uint32_t row,
                                   uint32_t actualCol, uint32_t repeatNum, uint64_t repeatOffset)
{
    AddRegList<T> regList[8];
    uint32_t loopTimes = row;
    uint32_t initialRepeatIdx = IS_FIRST ? 1 : 0;
    __ubuf__ T *initialAddr = IS_FIRST ? inputAddr : outputAddr;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    for (uint32_t idx = 0; idx < loopTimes; idx++) {
        uint64_t offset = idx * actualCol;
        MicroAPI::LoadAlign(regList[0].vreg, initialAddr + offset);
        MicroAPI::LoadAlign(regList[1].vreg, initialAddr + offset + FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(regList[2].vreg, initialAddr + offset + 2 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(regList[3].vreg, initialAddr + offset + 3 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(regList[4].vreg, initialAddr + offset + 4 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(regList[5].vreg, initialAddr + offset + 5 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(regList[6].vreg, initialAddr + offset + 6 * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(regList[7].vreg, initialAddr + offset + 7 * FLOAT_REP_SIZE);
        for (uint32_t repeatIdx = initialRepeatIdx; repeatIdx < repeatNum; repeatIdx++) {
            uint64_t addOffset = offset + repeatIdx * row * actualCol;
            MicroAPI::LoadAlign(regList[0].vregape, inputAddr + addOffset);
            MicroAPI::LoadAlign(regList[1].vregape, inputAddr + addOffset + FLOAT_REP_SIZE);
            MicroAPI::LoadAlign(regList[2].vregape, inputAddr + addOffset + 2 * FLOAT_REP_SIZE);
            MicroAPI::LoadAlign(regList[3].vregape, inputAddr + addOffset + 3 * FLOAT_REP_SIZE);
            MicroAPI::LoadAlign(regList[4].vregape, inputAddr + addOffset + 4 * FLOAT_REP_SIZE);
            MicroAPI::LoadAlign(regList[5].vregape, inputAddr + addOffset + 5 * FLOAT_REP_SIZE);
            MicroAPI::LoadAlign(regList[6].vregape, inputAddr + addOffset + 6 * FLOAT_REP_SIZE);
            MicroAPI::LoadAlign(regList[7].vregape, inputAddr + addOffset + 7 * FLOAT_REP_SIZE);
            MicroAPI::Add(regList[0].vreg, regList[0].vreg, regList[0].vregape, mask);
            MicroAPI::Add(regList[1].vreg, regList[1].vreg, regList[1].vregape, mask);
            MicroAPI::Add(regList[2].vreg, regList[2].vreg, regList[2].vregape, mask);
            MicroAPI::Add(regList[3].vreg, regList[3].vreg, regList[3].vregape, mask);
            MicroAPI::Add(regList[4].vreg, regList[4].vreg, regList[4].vregape, mask);
            MicroAPI::Add(regList[5].vreg, regList[5].vreg, regList[5].vregape, mask);
            MicroAPI::Add(regList[6].vreg, regList[6].vreg, regList[6].vregape, mask);
            MicroAPI::Add(regList[7].vreg, regList[7].vreg, regList[7].vregape, mask);
        }
        MicroAPI::StoreAlign(outputAddr + offset, regList[0].vreg, mask);
        MicroAPI::StoreAlign(outputAddr + offset + FLOAT_REP_SIZE, regList[1].vreg, mask);
        MicroAPI::StoreAlign(outputAddr + offset + 2 * FLOAT_REP_SIZE, regList[2].vreg, mask);
        MicroAPI::StoreAlign(outputAddr + offset + 3 * FLOAT_REP_SIZE, regList[3].vreg, mask);
        MicroAPI::StoreAlign(outputAddr + offset + 4 * FLOAT_REP_SIZE, regList[4].vreg, mask);
        MicroAPI::StoreAlign(outputAddr + offset + 5 * FLOAT_REP_SIZE, regList[5].vreg, mask);
        MicroAPI::StoreAlign(outputAddr + offset + 6 * FLOAT_REP_SIZE, regList[6].vreg, mask);
        MicroAPI::StoreAlign(outputAddr + offset + 7 * FLOAT_REP_SIZE, regList[7].vreg, mask);
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
__aicore__ inline void AddVF(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeLocal, uint32_t row,
                             uint32_t col, uint32_t actualCol0, uint32_t actualCol1)
{
    __ubuf__ T *scoreAddr = (__ubuf__ T *)scoreLocal.GetPhyAddr();
    __ubuf__ T *apeAddr = (__ubuf__ T *)apeLocal.GetPhyAddr();

    if (col <= 64) {
        Add64VFImpl<T>(scoreAddr, apeAddr, row, col, actualCol0, actualCol1);
    } else if (col == 128) {
        Add128VFImpl<T>(scoreAddr, apeAddr, row, actualCol0, actualCol1);
    } else if (col == 256) {
        Add256VFImpl<T>(scoreAddr, apeAddr, row, actualCol0, actualCol1);
    } else if (col == 512) {
        Add512VFImpl<T>(scoreAddr, apeAddr, row, actualCol0, actualCol1);
    }
}

template <typename T>
__aicore__ inline void AddVF(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeLocal, uint32_t row,
                             uint32_t col, uint32_t actualCol)
{
    __ubuf__ T *scoreAddr = (__ubuf__ T *)scoreLocal.GetPhyAddr();
    __ubuf__ T *apeAddr = (__ubuf__ T *)apeLocal.GetPhyAddr();

    if (col <= 64) {
        Add64VFImpl<T>(scoreAddr, apeAddr, row, col, actualCol, actualCol);
    } else if (col == 128) {
        Add128VFImpl<T>(scoreAddr, apeAddr, row, actualCol, actualCol);
    } else if (col == 256) {
        Add256VFImpl<T>(scoreAddr, apeAddr, row, actualCol, actualCol);
    } else if (col == 512) {
        Add512VFImpl<T>(scoreAddr, apeAddr, row, actualCol, actualCol);
    }
}

template <bool IS_FIRST, typename T>
__aicore__ inline void MultiAddVF(const LocalTensor<T> &outputLocal, const LocalTensor<T> &inputLocal, uint32_t row,
                                  uint32_t col, uint32_t actualCol, uint32_t repeatNum, uint64_t repeatOffset)
{
    __ubuf__ T *outputAddr = (__ubuf__ T *)outputLocal.GetPhyAddr();
    __ubuf__ T *inputAddr = (__ubuf__ T *)inputLocal.GetPhyAddr();
    if (col <= 64) {
        MultiAdd64VFImpl<IS_FIRST, T>(outputAddr, inputAddr, row, col, actualCol, repeatNum, repeatOffset);
    } else if (col == 128) {
        MultiAdd128VFImpl<IS_FIRST, T>(outputAddr, inputAddr, row, col, actualCol, repeatNum, repeatOffset);
    } else if (col == 256) {
        MultiAdd256VFImpl<IS_FIRST, T>(outputAddr, inputAddr, row, actualCol, repeatNum, repeatOffset);
    } else if (col == 512) {
        MultiAdd512VFImpl<IS_FIRST, T>(outputAddr, inputAddr, row, actualCol, repeatNum, repeatOffset);
    }
}

#endif
