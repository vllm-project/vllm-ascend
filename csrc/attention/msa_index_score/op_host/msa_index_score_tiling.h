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
 * \file msa_index_score_tiling.h
 * \brief MsaIndexScore TilingData 定义与 host 侧解析类声明。
 */

#ifndef MSA_INDEX_SCORE_TILING_H
#define MSA_INDEX_SCORE_TILING_H

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "err/ops_err.h"

namespace optiling {

// 与 op_def 入参顺序一致
constexpr uint32_t MSA_IDX_QUERY = 0;
constexpr uint32_t MSA_IDX_KEY = 1;
constexpr uint32_t MSA_IDX_BLOCK_TABLE = 2;
constexpr uint32_t MSA_IDX_SCALE = 3;
constexpr uint32_t MSA_IDX_ATTEN_MASK = 4;
constexpr uint32_t MSA_IDX_ACTUAL_SEQ_QLEN = 5;
constexpr uint32_t MSA_IDX_ACTUAL_SEQ_KLEN = 6;
constexpr uint32_t MSA_IDX_START_LOC = 7;

constexpr uint32_t MSA_ATTR_LAYOUT_KEY = 0;
constexpr uint32_t MSA_ATTR_SPARSE_MODE = 1;
constexpr uint32_t MSA_ATTR_INIT_BLOCKS = 2;
constexpr uint32_t MSA_ATTR_LOCAL_BLOCKS = 3;

BEGIN_TILING_DATA_DEF(MsaIndexScoreTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch)
TILING_DATA_FIELD_DEF(uint32_t, totalQ)
TILING_DATA_FIELD_DEF(uint32_t, numQHeads)
TILING_DATA_FIELD_DEF(uint32_t, numKvHeads)
TILING_DATA_FIELD_DEF(uint32_t, gqaGroup)
TILING_DATA_FIELD_DEF(uint32_t, headDim)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlocksPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, scoreBlockStride)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, isQuant)    // 1: key 为 int8
TILING_DATA_FIELD_DEF(uint32_t, sparseMode) // 0 / 3
TILING_DATA_FIELD_DEF(uint32_t, initBlocks)
TILING_DATA_FIELD_DEF(uint32_t, localBlocks)
TILING_DATA_FIELD_DEF(uint32_t, numPages)
TILING_DATA_FIELD_DEF(uint32_t, strideQt)
TILING_DATA_FIELD_DEF(uint32_t, strideQn)
TILING_DATA_FIELD_DEF(uint32_t, strideKvBlock)
TILING_DATA_FIELD_DEF(uint32_t, strideKvToken)
TILING_DATA_FIELD_DEF(uint32_t, strideScalePage)
TILING_DATA_FIELD_DEF(uint32_t, strideScaleHead)
TILING_DATA_FIELD_DEF(uint32_t, strideOutHead)
TILING_DATA_FIELD_DEF(uint32_t, strideOutToken)
TILING_DATA_FIELD_DEF(uint32_t, kScratchOffsetElems)
TILING_DATA_FIELD_DEF(uint32_t, keyLayout) // MSA_KEY_LAYOUT_BBND / BNBD / TND
TILING_DATA_FIELD_DEF(uint32_t, totalK)    // TND：key 第 0 维 T2；PA：0
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(MsaIndexScore, MsaIndexScoreTilingData)

struct MsaIndexScoreCompileInfo {};

struct MsaIndexScoreInfo {
    fe::PlatFormInfos *platformInfo = nullptr;
    uint32_t batch = 0;
    uint32_t totalQ = 0;
    uint32_t numQHeads = 0;
    uint32_t numKvHeads = 0;
    uint32_t headDim = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlocksPerBatch = 0;
    uint32_t numPages = 0;
    uint32_t sparseMode = 3;
    uint32_t initBlocks = 0;
    uint32_t localBlocks = 1;
    uint32_t keyLayout = 0;
    uint32_t totalK = 0;
    bool isQuant = false;
    ge::DataType queryDtype = ge::DT_BF16;
    ge::DataType keyDtype = ge::DT_BF16;
};

} // namespace optiling

#endif // MSA_INDEX_SCORE_TILING_H
