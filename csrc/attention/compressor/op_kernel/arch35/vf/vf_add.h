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
 * \file vf_add.h
 * \brief
 */

#ifndef VF_ADD_H
#define VF_ADD_H

#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t FLOAT_REP_SIZE = 64;
constexpr uint32_t BTYEALIGNSIZE = 32;
constexpr uint32_t REGSIZE = 256;
constexpr uint32_t HALFCORED = 128;

/*apeOffset —— ape的偏移
  loopCnt —— r行循环次数
  count —— regtensor上的元素数


*/
template<typename T>
__simd_vf__ void AddVFImpl(__ubuf__ T* inputAddr, __ubuf__ T* apeAddr, const uint16_t coreSplitD, const uint16_t loopCnt)
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vregape0;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    for(uint16_t loop = 0; loop < loopCnt; loop++) {
        MicroAPI::LoadAlign(vreg0, inputAddr + loop * FLOAT_REP_SIZE);
        MicroAPI::LoadAlign(vregape0, apeAddr + loop* FLOAT_REP_SIZE);
        MicroAPI::Add(vreg0, vreg0, vregape0, mask);
        MicroAPI::StoreAlign(inputAddr + loop*FLOAT_REP_SIZE, vreg0, mask);
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
__aicore__ inline void AddVF(LocalTensor<T> &scoreLocal, LocalTensor<T> &apeLocal,
    uint16_t apeIdx, const uint16_t coff, const uint16_t coreSplitD, uint32_t coreSplitS) 
{
    uint16_t loopCnt = coff *coreSplitD*coreSplitS/FLOAT_REP_SIZE;

    __ubuf__ T * scoreAddr = (__ubuf__ T *)scoreLocal.GetPhyAddr();
    __ubuf__ T * apeAddr = (__ubuf__ T *)apeLocal.GetPhyAddr() + apeIdx;

    AddVFImpl<T>(scoreAddr, apeAddr, coreSplitD, loopCnt);
}

#endif
