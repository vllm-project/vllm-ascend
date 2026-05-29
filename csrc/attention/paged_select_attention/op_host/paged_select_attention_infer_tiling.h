/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PAGED_SELECT_ATTENTION_INFER_TILING_H
#define PAGED_SELECT_ATTENTION_INFER_TILING_H

#include <algorithm>
#include "exe_graph/runtime/tiling_context.h"
#include "register/tilingdata_base.h"
#include "paged_select_attention_tiling.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PagedSelectAttentionTilingData)
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
TILING_DATA_FIELD_DEF(uint32_t, sparseLaunchEnabled)
TILING_DATA_FIELD_DEF(uint32_t, sparseKMax)
TILING_DATA_FIELD_DEF(uint32_t, firstBatchTaskNum)
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum)
TILING_DATA_FIELD_DEF(uint32_t, maskType)
TILING_DATA_FIELD_DEF(uint64_t, mm1OutSize)
TILING_DATA_FIELD_DEF(uint64_t, smOnlineOutSize)
TILING_DATA_FIELD_DEF(uint64_t, mm2OutSize)
TILING_DATA_FIELD_DEF(uint64_t, UpdateSize)
TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint64_t, padding1)
TILING_DATA_FIELD_DEF(uint64_t, padding2)
TILING_DATA_FIELD_DEF(uint32_t, padding3)
END_TILING_DATA_DEF

const uint32_t SIZE_OF_16BIT = 2;
const uint32_t SIZE_OF_32BIT = 4;
const uint32_t N_SPLIT_HELPER = 2;
const uint32_t MAX_KV_STACK_LEN = 512;
const uint32_t Q_TILE_CEIL = 128;
const uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;
const uint32_t PRELANCH_NUM = 3;

enum class MaskType : uint32_t {
    NO_MASK = 0,
};

enum class DataType : uint32_t {
    FP16 = 0,
    BF16 = 1
};

struct PagedSelectAttentionContext {
    int32_t numTokens = 0;
    int32_t numHeads = 0;
    int32_t embeddingSize = 0;
    int32_t embeddingSizeV = 0;
    int32_t numBlocks = 0;
    int32_t blockSize = 0;
    int32_t kvHeads = 0;
    int32_t batch = 0;
    int64_t maxQSeqlen = 0;
    int64_t maxKvSeqlen = 0;
    uint32_t maxNumBlocksPerBatch = 0;
    const int64_t *qSeqlenList{nullptr};
    const int64_t *kvSeqlenList{nullptr};
    float scaleValue = 0.0F;
    size_t *workspaces{nullptr};
    MaskType maskType = MaskType::NO_MASK;
    DataType dataType = DataType::FP16;
    bool sparseLaunchEnabled = true;
    uint32_t sparseKMax = 0;
    bool isTilingSink = false;
};

class PagedSelectAttentionTiling {
public:
    explicit PagedSelectAttentionTiling(const PagedSelectAttentionContext &faInfo) : faInfo_(faInfo) {}

    ge::graphStatus DoTiling(PagedSelectAttentionTilingData &tilingData)
    {
        FillBasicTilingData(tilingData);
        if (!faInfo_.isTilingSink) {
            FillSplitCoreTilingData(tilingData);
        }
        FillWorkSpaceTilingData(tilingData);
        return ge::GRAPH_SUCCESS;
    }

    void SetCoreNum(uint32_t blockNum)
    {
        blockNum_ = blockNum;
    }

    uint32_t GetCoreNum() const
    {
        return blockNum_;
    }

    uint64_t GetTilingKey() const
    {
        return (faInfo_.dataType == DataType::BF16) ? 1ULL : 0ULL;
    }

private:
    static uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
    {
        uint32_t qNBlockTile = (qSeqlen != 0U) ?
            (Q_TILE_CEIL / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
        qNBlockTile = std::min(qNBlockTile, groupSize);
        qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
        return qNBlockTile;
    }

    static uint32_t GetQSBlockTile(int64_t /* kvSeqlen */)
    {
        return Q_TILE_CEIL;
    }

    void FillBasicTilingData(PagedSelectAttentionTilingData &tilingData)
    {
        tilingData.set_batch(static_cast<uint32_t>(faInfo_.batch));
        tilingData.set_numHeads(static_cast<uint32_t>(faInfo_.numHeads));
        tilingData.set_kvHeads(static_cast<uint32_t>(faInfo_.kvHeads));
        tilingData.set_embeddingSize(static_cast<uint32_t>(faInfo_.embeddingSize));
        tilingData.set_embeddingSizeV(static_cast<uint32_t>(faInfo_.embeddingSizeV));
        tilingData.set_numBlocks(static_cast<uint32_t>(faInfo_.numBlocks));
        tilingData.set_blockSize(static_cast<uint32_t>(faInfo_.blockSize));
        tilingData.set_maxQSeqlen(static_cast<uint32_t>(faInfo_.maxQSeqlen));
        tilingData.set_maxKvSeqlen(static_cast<uint32_t>(faInfo_.maxKvSeqlen));
        tilingData.set_maskType(static_cast<uint32_t>(faInfo_.maskType));
        tilingData.set_scaleValue(faInfo_.scaleValue);
        tilingData.set_maxNumBlocksPerBatch(faInfo_.maxNumBlocksPerBatch);
        tilingData.set_sparseLaunchEnabled(static_cast<uint32_t>(faInfo_.sparseLaunchEnabled));
        tilingData.set_sparseKMax(faInfo_.sparseKMax);
    }

    void FillWorkSpaceTilingData(PagedSelectAttentionTilingData &tilingData)
    {
        uint64_t mm1OutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_32BIT * PRELANCH_NUM;
        uint64_t smOnlineOutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_16BIT * PRELANCH_NUM;
        uint64_t mm2OutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_32BIT * PRELANCH_NUM;
        uint64_t updateSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_32BIT * PRELANCH_NUM;
        tilingData.set_mm1OutSize(mm1OutSize);
        tilingData.set_smOnlineOutSize(smOnlineOutSize);
        tilingData.set_mm2OutSize(mm2OutSize);
        tilingData.set_UpdateSize(updateSize);
        tilingData.set_workSpaceSize(mm1OutSize + smOnlineOutSize + mm2OutSize + updateSize);
    }

    void FillSplitCoreTilingData(PagedSelectAttentionTilingData &tilingData)
    {
        uint32_t totalTaskNum = 0;
        uint32_t groupSize = static_cast<uint32_t>(faInfo_.numHeads / faInfo_.kvHeads);
        for (int32_t batchIdx = 0; batchIdx < faInfo_.batch; ++batchIdx) {
            uint32_t qSeqlen = static_cast<uint32_t>(*(faInfo_.qSeqlenList + batchIdx));
            uint32_t kvSeqlen = static_cast<uint32_t>(*(faInfo_.kvSeqlenList + batchIdx));
            if (batchIdx > 0) {
                uint64_t prevQSeqlenSum = *(faInfo_.qSeqlenList + batchIdx - 1);
                qSeqlen = qSeqlen - static_cast<uint32_t>(prevQSeqlenSum);
            }
            uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
            uint32_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1U) / curQNBlockTile;
            uint32_t curQNBlockNum = qNBlockNumPerGroup * static_cast<uint32_t>(faInfo_.kvHeads);
            uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
            uint32_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1U) / curQSBlockTile;
            uint32_t curTaskNum = curQNBlockNum * curQSBlockNum;
            if (batchIdx == 0) {
                tilingData.set_firstBatchTaskNum(curTaskNum);
            }
            totalTaskNum += curTaskNum;
        }
        tilingData.set_totalTaskNum(totalTaskNum);
    }

    PagedSelectAttentionContext faInfo_;
    uint32_t blockNum_ = 0;
};
} // namespace optiling

#endif // PAGED_SELECT_ATTENTION_INFER_TILING_H
