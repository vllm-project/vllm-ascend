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
 * \file kernel_common.h
 * \brief
 */
#ifndef KERNEL_COMMON_H
#define KERNEL_COMMON_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

using namespace AscendC;
using AscendC::GlobalTensor;

namespace fa_base_kernel {

__aicore__ inline uint64_t GetActualQSeqLength(GlobalTensor<uint64_t> &actualSeqLengths, uint32_t bIdx, const ConstInfo &constInfo)
{
    if (constInfo.actualLenQDims == 0) {
        return constInfo.qSeqSize;
    } 
    if (constInfo.actualLenQDims == 1 || bIdx == 0) {
        return actualSeqLengths.GetValue(0);
    }
    if (constInfo.accumQSeqFlag) {
        return actualSeqLengths.GetValue(bIdx) - actualSeqLengths.GetValue(bIdx - 1);
    } else {
        return actualSeqLengths.GetValue(bIdx);
    }
    return 0;
}

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline uint64_t SeqLenFromTensorList(__gm__ uint8_t *keyPtr, uint32_t bIndex)
{
    uint64_t dimInfo[4]; // this mem is used to set shapeinfo, BSH(3) or BNSD(4)
    AscendC::TensorDesc<__gm__ uint8_t> keyTensorDesc;
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    keyTensorDesc.SetShapeAddr(&dimInfo[0]);
    keyListTensorDesc.GetDesc(keyTensorDesc, bIndex);
    if constexpr (LAYOUT_T == FIA_LAYOUT::BSH || LAYOUT_T == FIA_LAYOUT::BSND) {
        return keyTensorDesc.GetShape(1); // BSH, idx of s is 1
    } else {
        return keyTensorDesc.GetShape(2); // BNSD, idx of s is 2
    }
}

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline uint64_t GetActualKVSeqLength(GlobalTensor<uint64_t> &actualSeqLengths, uint32_t bIdx, const ConstInfo &constInfo, __gm__ uint8_t *keyPtr)
{
    if (constInfo.actualLenDims == 0) {
        if (!constInfo.batchContinuous) {
            return SeqLenFromTensorList<LAYOUT_T>(keyPtr, bIdx);
        }
        return constInfo.kvSeqSize;
    } 
    if (constInfo.actualLenDims == 1 || bIdx == 0) {
        return actualSeqLengths.GetValue(0);
    }
    if (constInfo.accumKVSeqFlag) {
        return actualSeqLengths.GetValue(bIdx) - actualSeqLengths.GetValue(bIdx - 1);
    } else {
        return actualSeqLengths.GetValue(bIdx);
    }
    return 0;
}

} // namespace fa_base_kernel
#endif