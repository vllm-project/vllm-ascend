/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file blasst_attention_score_tiling.h
 * \brief Tiling data structures and FAInferTiling declaration for
 *        VllmBlasstAttentionScore. Implementations live in
 *        blasst_attention_score_tiling.cpp.
 */

#ifndef BLASST_ATTENTION_SCORE_TILING_H
#define BLASST_ATTENTION_SCORE_TILING_H

#include <cstdint>
#include <string>
#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/tiling_context.h>
#include <tiling/platform/platform_ascendc.h>
#include "register/tilingdata_base.h"

namespace optiling {
constexpr int32_t MAX_CORE_NUM_FD = 26;
// host-list-only: seq lens are embedded in TilingData (framework-managed
// upload, task-update safe) instead of a device input tensor. Decode
// cudagraph buckets top out at 96; 256 covers eager mixed batches.
constexpr int32_t BLASST_MAX_HOST_SEQ_LIST = 256;

BEGIN_TILING_DATA_DEF(splitNode)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, batchIdx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, headStartIdx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, headEndIdx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, qStartIdx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, qEndIdx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, splitNum)
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_NUM_FD, lseTaskOffset)
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_NUM_FD, oTaskOffset)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(splitNodeOp, splitNode)

BEGIN_TILING_DATA_DEF(coreNode)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, startBIdx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, startN1Idx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, startS1Idx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, startS2Idx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, endBIdx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, endN1Idx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, endS1Idx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, endS2Idx)
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_NUM_FD, firstSplitKVTaskLseOffset)
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_NUM_FD, firstSplitKVTaskOOffset)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(coreNodeOp, coreNode)

BEGIN_TILING_DATA_DEF(FAInferTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numHeads)
TILING_DATA_FIELD_DEF(uint32_t, embeddingSize)
TILING_DATA_FIELD_DEF(uint32_t, embeddingSizeV)
TILING_DATA_FIELD_DEF(uint32_t, numBlocks)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxQSeqlen)
TILING_DATA_FIELD_DEF(uint32_t, maxKvSeqlen)
TILING_DATA_FIELD_DEF(uint32_t, kvHeads)
TILING_DATA_FIELD_DEF(uint32_t, batch)
TILING_DATA_FIELD_DEF(uint32_t, maxNumBlocksPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, firstBatchTaskNum)
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum)
TILING_DATA_FIELD_DEF(uint32_t, maskType)
TILING_DATA_FIELD_DEF(uint64_t, mm1OutSize)
TILING_DATA_FIELD_DEF(uint64_t, smOnlineOutSize)
TILING_DATA_FIELD_DEF(uint64_t, mm2OutSize)
TILING_DATA_FIELD_DEF(uint64_t, UpdateSize)
TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(float, sparseLamda)
TILING_DATA_FIELD_DEF(int64_t, preToken)
TILING_DATA_FIELD_DEF(int64_t, nextToken)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(uint32_t, mainLoopTaskNum)
TILING_DATA_FIELD_DEF(uint32_t, tailLoopTaskNum)
TILING_DATA_FIELD_DEF(uint32_t, tailStartBatch)
TILING_DATA_FIELD_DEF(uint32_t, tailStartN2)
TILING_DATA_FIELD_DEF(uint32_t, tailKvNBlockTile)
TILING_DATA_FIELD_DEF(uint32_t, sparseStatsFlag)
TILING_DATA_FIELD_DEF(uint64_t, sparseStatsSize)
TILING_DATA_FIELD_DEF(uint32_t, totalSplitNodeNum)
TILING_DATA_FIELD_DEF(uint64_t, splitLseTotalSize)
TILING_DATA_FIELD_DEF(uint64_t, splitOTotalSize)
TILING_DATA_FIELD_DEF_STRUCT(coreNode, coreInfo)
TILING_DATA_FIELD_DEF_STRUCT(splitNode, splitInfo)
TILING_DATA_FIELD_DEF_ARR(int64_t, BLASST_MAX_HOST_SEQ_LIST, actualQSeq)
TILING_DATA_FIELD_DEF_ARR(int64_t, BLASST_MAX_HOST_SEQ_LIST, actualKvSeq)
END_TILING_DATA_DEF

const uint32_t SIZE_OF_16BIT = 2;
const uint32_t SIZE_OF_32BIT = 4;
const uint32_t N_SPLIT_HELPER = 2;
const uint32_t MAX_KV_STACK_LEN = 512;
const uint32_t Q_TILE_CEIL = 128;
const uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;
const uint32_t BASE_KV_SIZE = 128;
const uint32_t PRELANCH_NUM = 3;
const int64_t SPARSE_MODE_INT_MAX = 2147483647;

enum class MaskType : uint32_t {
    NO_MASK = 0,
    MASK_SPEC = 1
};

enum class DataType : uint32_t {
    FP16 = 0,
    BF16 = 1
};

struct FAInferContext {
    int32_t numTokens = 0;
    int32_t numHeads = 0;
    int32_t embeddingSize = 0;
    int32_t embeddingSizeV = 0;
    int32_t numBlocks = 0;
    int32_t blockSize = 0;
    int32_t kvHeads = 0;
    int32_t batch = 0;
    int32_t innerPrecise = 0;
    int64_t maxQSeqlen = 0;
    int64_t maxKvSeqlen = 0;
    int64_t preToken = 0;
    int64_t nextToken = 0;
    int32_t sparseMode = 0;
    uint32_t maxNumBlocksPerBatch = 0;
    const int64_t *qSeqlenList{nullptr};
    const int64_t *kvSeqlenList{nullptr};
    float scaleValue = 0.0;
    float sparseLamda = -99.0f;
    MaskType maskType = MaskType::MASK_SPEC;
    DataType dataType = DataType::FP16;
    bool pagedCacheFlag = false;
    bool lseFlag = false;
    bool flashDecodeFlag = false;
    bool sparseStatsFlag = false;
    std::string layout;
};

class FAInferTiling {
public:
    FAInferTiling() = default;
    explicit FAInferTiling(const FAInferContext &faInfo) : faInfo_(faInfo) {}
    ge::graphStatus DoTiling(FAInferTilingData &tilingdata);
    void SetCoreNum(uint32_t blockNum)
    {
        this->blockNum_ = blockNum;
    }
    uint32_t GetCoreNum()
    {
        return this->blockNum_;
    }
    uint64_t GetTilingKey();

private:
    void FillSplitCoreTilingData(FAInferTilingData &tilingdata);
    void FillWorkSpaceTilingData(FAInferTilingData &faTilingData);
    uint32_t GetQSBlockTile();
    uint32_t GetKvNBlockTile(uint32_t rowNumPerQSGTile, uint32_t kvHead);
    uint32_t GetKSBlockTile();
    uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize);
    void FillBasicTilingData(FAInferTilingData &faTilingData);
    struct BatchParams {
        uint32_t qSeqlen;
        uint32_t kvSeqlen;
        uint32_t curQNBlockTile;
        uint32_t qNBlockNumPerGroup;
        uint32_t curQNBlockNum;
        uint32_t curQSBlockTile;
        uint32_t curQSBlockNum;
        uint32_t curKSBlockTile;
        uint32_t curKSBlockNum;
    };
    BatchParams getBatchParams(uint32_t bIdx, uint32_t groupSize);
    void fillCoreInfoForFlashDecode(FAInferTilingData &faTilingData, uint32_t groupSize,
                                    uint64_t perCoreTaskNum);
    void fillSplitInfoForFlashDecode(FAInferTilingData &faTilingData, uint32_t groupSize);
    void splitBN2S1GS2(FAInferTilingData &faTilingData);

private:
    FAInferContext faInfo_;
    uint32_t blockNum_;
};

// Inputs Index
constexpr uint32_t BLASST_QUERY_INPUT_INDEX = 0;
constexpr uint32_t BLASST_KEY_INPUT_INDEX = 1;
constexpr uint32_t BLASST_VALUE_INPUT_INDEX = 2;
constexpr uint32_t BLASST_PSE_SHIFT_INPUT_INDEX = 3;
constexpr uint32_t BLASST_ATTEN_MASK_INPUT_INDEX = 4;
constexpr uint32_t BLASST_ACTUAL_SEQ_LENGTHS_INPUT_INDEX = 5;
constexpr uint32_t BLASST_ACTUAL_SEQ_LENGTHS_KV_INPUT_INDEX = 6;
constexpr uint32_t BLASST_BLOCK_TABLE_INPUT_INDEX = 7;

// Outputs Index
constexpr uint32_t BLASST_ATTENTION_OUT_INDEX = 0;
constexpr uint32_t BLASST_SOFTMAX_LSE_INDEX = 1;

// Attributes Index
constexpr uint32_t BLASST_NUM_HEADS_ATTR_INDEX = 0;
constexpr uint32_t BLASST_SCALE_ATTR_INDEX = 1;
constexpr uint32_t BLASST_PRE_TOKENS_ATTR_INDEX = 2;
constexpr uint32_t BLASST_NEXT_TOKENS_ATTR_INDEX = 3;
constexpr uint32_t BLASST_INPUT_LAYOUT_ATTR_INDEX = 4;
constexpr uint32_t BLASST_NUM_KEY_VALUE_HEADS_ATTR_INDEX = 5;
constexpr uint32_t BLASST_SPARSE_MODE_ATTR_INDEX = 6;
constexpr uint32_t BLASST_INNER_PRECISE_ATTR_INDEX = 7;
constexpr uint32_t BLASST_BLOCK_SIZE_ATTR_INDEX = 8;
constexpr uint32_t BLASST_ANTIQUANT_MODE_ATTR_INDEX = 9;
constexpr uint32_t BLASST_SPARSE_LAMBDA_ATTR_INDEX = 10;
constexpr uint32_t BLASST_SOFTMAX_LSE_FLAG_ATTR_INDEX = 11;
constexpr uint32_t BLASST_ACTUAL_SEQ_LENGTHS_Q_HOST_ATTR_INDEX = 12;
constexpr uint32_t BLASST_ACTUAL_SEQ_LENGTHS_KV_HOST_ATTR_INDEX = 13;
constexpr uint32_t BLASST_SPARSE_STATS_FLAG_ATTR_INDEX = 14;
constexpr uint32_t BLASST_FLASH_DECODE_ATTR_INDEX = 15;

constexpr uint32_t BLASST_SPARSE_STATS_INDEX = 2;

ge::graphStatus TilingVllmBlasstAttentionScore(gert::TilingContext *context);
} // namespace optiling

#endif // BLASST_ATTENTION_SCORE_TILING_H
