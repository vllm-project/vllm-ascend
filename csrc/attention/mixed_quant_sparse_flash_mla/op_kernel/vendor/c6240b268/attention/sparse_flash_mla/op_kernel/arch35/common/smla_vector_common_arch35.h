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
 * \file smla_vector_common_arch35.h
 * \brief sparse_flash_mla / mixed_quant_sparse_flash_mla / quant_sparse_flash_mla 三个算子共用的
 *        KV 物理地址块搬入/搬出辅助函数。
 */

#ifndef SMLA_VECTOR_COMMON_ARCH35_H
#define SMLA_VECTOR_COMMON_ARCH35_H

#include <stdint.h>
#include "static_buffer.h"

namespace AttentionCommon {

// 将物理地址块从 UB 拷出到 GM
__aicore__ inline void CopyPhyAddrToGm(LocalTensor<uint32_t> kvPhyAddrUb, int64_t bS1Idx, int64_t s1Idx,
                                       int64_t validS2, int64_t alignNum, GlobalTensor<uint32_t> &phyAddrGm,
                                       uint32_t alignedSparseBlockCount)
{
    constexpr int64_t numPerBlock = 32;
    DataCopyParams params;
    params.blockCount = 1U;
    params.blockLen = ((validS2 + alignNum - 1) / alignNum * alignNum) * sizeof(int64_t) / numPerBlock;
    params.srcGap = 0U;
    params.dstGap = 0U;
    DataCopy(phyAddrGm[(bS1Idx + s1Idx) * alignedSparseBlockCount * 2], kvPhyAddrUb, params);
}

// 将 paged attention block table 从 GM 搬入 UB
__aicore__ inline void CopyPaTableToUb(LocalTensor<int32_t> blkTableUb, int64_t bIdx,
                                       GlobalTensor<int32_t> &blockTableGm, uint32_t maxBlockNumPerBatch)
{
    DataCopyExtParams params;
    params.blockCount = 1U;
    params.blockLen = maxBlockNumPerBatch * sizeof(int32_t);
    params.srcStride = 0U;
    params.dstStride = 0U;
    DataCopyPadExtParams<int32_t> padParams;
    DataCopyPad(blkTableUb, blockTableGm[bIdx * maxBlockNumPerBatch], params, padParams);
}

// 将 sparse block 索引从 GM 搬入 UB
__aicore__ inline void CopySparseIdxToUb(LocalTensor<int32_t> sparseIdxUb, int64_t bS1Idx, int64_t s1Idx,
                                         int64_t validS2, GlobalTensor<int32_t> &sparseIndicesGm,
                                         uint32_t sparseBlockCount)
{
    DataCopyExtParams params;
    params.blockCount = 1U;
    params.blockLen = static_cast<uint32_t>(validS2) * sizeof(int32_t);
    params.srcStride = 0U;
    params.dstStride = 0U;
    DataCopyPadExtParams<int32_t> padParams;
    DataCopyPad(sparseIdxUb, sparseIndicesGm[(bS1Idx + s1Idx) * sparseBlockCount], params, padParams);
}

} // namespace AttentionCommon

#endif // SMLA_VECTOR_COMMON_ARCH35_H
