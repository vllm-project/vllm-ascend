/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sparse_flash_mla_tiling.h
 * \brief Kernel UT helpers for SparseFlashMla tiling data on CPU simulator.
 */

#ifndef TEST_SPARSE_FLASH_MLA_TILING_H
#define TEST_SPARSE_FLASH_MLA_TILING_H

#include <cstdint>
#include <cstring>

#include "../../../../../sparse_flash_mla/op_host/sparse_flash_mla_tiling.h"
#include "../../../../../sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_arch22_metadata.h"

namespace smla_ut {

constexpr uint32_t kTndPaBnbdSwaTilingKey = 66U;
constexpr uint32_t kTndPaBnbdHcaTilingKey = 578U;
constexpr uint32_t kTndPaBnbdCsaTilingKey = 1090U;

inline void InitMetadataGm(int32_t *metadata, uint32_t batchSize, uint32_t kvHeadNum)
{
    optiling::detail::SasMetadata *meta = reinterpret_cast<optiling::detail::SasMetadata *>(metadata);
    std::memset(meta, 0, optiling::SMLA_META_SIZE * sizeof(int32_t));
    const uint32_t bn2End = batchSize * kvHeadNum;
    meta->faMetadata[0][optiling::FA_CORE_ENABLE_INDEX] = 1U;
    meta->faMetadata[0][optiling::FA_BN2_END_INDEX] = bn2End;
    meta->faMetadata[0][optiling::FA_M_END_INDEX] = 1U;
    meta->faMetadata[0][optiling::FA_S2_END_INDEX] = 1U;
}

inline void InitSwaTilingData(optiling::SparseFlashMlaTilingData &tiling, uint32_t tokenNum, uint32_t kvSeqLen,
                              uint32_t oriBlockNum, uint32_t blockSize)
{
    std::memset(&tiling, 0, sizeof(tiling));
    tiling.baseParams.set_batchSize(1U);
    tiling.baseParams.set_qSeqSize(tokenNum);
    tiling.baseParams.set_kvSeqSize(kvSeqLen);
    tiling.baseParams.set_nNumOfQInOneGroup(64U);
    tiling.baseParams.set_paBlockSize(blockSize);
    tiling.baseParams.set_oriBlockSize(blockSize);
    tiling.baseParams.set_cmpBlockSize(blockSize);
    tiling.baseParams.set_oriMaxBlockNumPerBatch(oriBlockNum);
    tiling.baseParams.set_actualLenDimsQ(1U);
    tiling.baseParams.set_actualLenDimsKV(1U);
    tiling.baseParams.set_softmaxScale(0.04419417381615906f);
    tiling.baseParams.set_outputLayout(static_cast<uint32_t>(optiling::SMLALayout::TND));
    tiling.baseParams.set_oriMaskMode(4U);
    tiling.baseParams.set_oriKvStride0(0);
    tiling.baseParams.set_oriWinLeft(127);
    tiling.baseParams.set_oriWinRight(0);
    tiling.baseParams.set_sparseBlockSize(1);
    tiling.baseParams.set_returnSoftmaxLse(false);
    tiling.baseParams.set_usedCoreNum(24U);
    tiling.baseParams.set_mmResUbSize(8192U);
    tiling.baseParams.set_bmm2ResUbSize(8192U);
    tiling.baseParams.set_mBaseSize(64U);
    tiling.baseParams.set_s2BaseSize(512U);
}

inline void InitCsaTilingData(optiling::SparseFlashMlaTilingData &tiling, uint32_t tokenNum, uint32_t kvSeqLen,
                              uint32_t oriBlockNum, uint32_t cmpBlockNum, uint32_t blockSize)
{
    InitSwaTilingData(tiling, tokenNum, kvSeqLen, oriBlockNum, blockSize);
    tiling.cmpParams.set_cmpMaxBlockNumPerBatch(cmpBlockNum);
    tiling.cmpParams.set_sparseBlockCount(512U);
    tiling.cmpParams.set_cmpRatio(4);
    tiling.cmpParams.set_cmpMaskMode(3U);
    tiling.cmpParams.set_cmpKvStride0(0);
}

} // namespace smla_ut

#endif // TEST_SPARSE_FLASH_MLA_TILING_H
