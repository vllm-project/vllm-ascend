/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sparse_attention_score_tiling.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <limits>
#include <string>
#include "log/log.h"
#include "err/ops_err.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_base.h"

using namespace ge;
using namespace std;

constexpr int QUERY_INDEX = 0;
constexpr int KEY_INDEX = 1;
constexpr int VALUE_INDEX = 2;
constexpr int SELECT_IDX_INDEX = 3;
constexpr int BLOCK_TABLE_INDEX = 4;
constexpr int SELECT_NUM_IDX_INDEX = 5;
constexpr int ACTUAL_SEQ_LENGTHS_INDEX = 6;
constexpr int ACTUAL_SEQ_LENGTHS_KV_INDEX = 7;

constexpr int ATTENTION_OUT_INDEX = 0;

constexpr int TND_DIM_T = 0;
constexpr int TND_DIM_N = 1;
constexpr int TND_DIM_D = 2;

constexpr int BLOCKED_KV_DIM_BLOCK_NUM = 0;
constexpr int BLOCKED_KV_DIM_BLOCK_SIZE = 1;
constexpr int BLOCKED_KV_DIM_KV_HEAD = 2;
constexpr int BLOCKED_KV_DIM_D = 3;

constexpr int SELECT_IDX_DIM_KV_HEAD = 0;
constexpr int SELECT_IDX_DIM_SEQ = 1;
constexpr int SELECT_IDX_DIM_TOPK = 2;

constexpr int BLOCK_TABLE_DIM_BATCH = 0;
constexpr int BLOCK_TABLE_DIM_MAX_BLOCKS = 1;

constexpr int ATTR_NUM_KV_HEADS_INDEX = 0;
constexpr int ATTR_SCALE_VALUE_INDEX = 1;
constexpr int ATTR_BLOCK_SIZE_INDEX = 2;
constexpr int ATTR_TOP_K_INDEX = 3;
constexpr int ATTR_INNER_PRECISE_INDEX = 4;

constexpr uint32_t SOC_VER_950_CODE = 4;

namespace optiling {

// The fitted costs use 0.001 us as one integer cost unit. The global
// 5.081045 us term is independent of the core count and is therefore omitted
// from the FD core-count selection objective.
constexpr uint64_t FD_COST_M16 = 125U;
constexpr uint64_t FD_COST_N = 740U;
constexpr uint64_t FD_COST_M16_N = 35U;
constexpr uint64_t FD_LAUNCH_COST = 278U;

static inline uint32_t CeilDiv(uint32_t n1, uint32_t n2)
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? ((n1 + n2 - 1) / n2) : n1;
}

static inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

static inline uint32_t CalcFdBestCore(uint64_t totalCost, uint64_t launchCost, uint32_t maxCore)
{
    if (totalCost == 0U || launchCost == 0U || maxCore == 0U) {
        return 1U;
    }

    const double ratio = static_cast<double>(totalCost) / static_cast<double>(launchCost);
    const double root = (std::sqrt(1.0 + 4.0 * ratio) - 1.0) / 2.0;
    const uint32_t bestCore = static_cast<uint32_t>(std::ceil(root));
    return std::max(1U, std::min(bestCore, maxCore));
}

ge::graphStatus SASATiling::GetNpuInfo(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    blockDim_ = aicNum_;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    socVer_ = static_cast<uint32_t>(ascendcPlatform.GetSocVersion());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASATiling::ParseAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "GetAttrs returned nullptr."), return ge::GRAPH_FAILED);

    const int64_t *numKvHeadsPtr = attrs->GetInt(ATTR_NUM_KV_HEADS_INDEX);
    if (numKvHeadsPtr != nullptr) {
        kvHeads_ = static_cast<uint32_t>(*numKvHeadsPtr);
    }

    const float *scalePtr = attrs->GetFloat(ATTR_SCALE_VALUE_INDEX);
    if (scalePtr != nullptr) {
        scaleValue_ = *scalePtr;
    }

    const int64_t *blockSizePtr = attrs->GetInt(ATTR_BLOCK_SIZE_INDEX);
    if (blockSizePtr != nullptr) {
        blockSize_ = static_cast<uint32_t>(*blockSizePtr);
    }

    const int64_t *topKPtr = attrs->GetInt(ATTR_TOP_K_INDEX);
    if (topKPtr != nullptr) {
        topK_ = static_cast<uint32_t>(*topKPtr);
    }

    const int64_t *innerPrecPtr = attrs->GetInt(ATTR_INNER_PRECISE_INDEX);
    if (innerPrecPtr != nullptr) {
        innerPrecise_ = static_cast<uint32_t>(*innerPrecPtr);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASATiling::CheckAttentionOutDtype(gert::TilingContext *sasContext)
{
    if (dataType_ == ge::DT_FLOAT8_E4M3FN) {
        attentionOutDtype_ = sasContext->GetOutputDesc(ATTENTION_OUT_INDEX)->GetDataType();
        if (attentionOutDtype_ != ge::DT_FLOAT16 && attentionOutDtype_ != ge::DT_BF16) {
            OP_LOGE(sasContext->GetNodeName(),
                    "The supported dtype of attentionOut is float16 or bfloat16 when the dtype of query/key/value is "
                    "all float8_e4m3fn, but now it is %d.",
                    attentionOutDtype_);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASATiling::ParseInputTensors(gert::TilingContext *context)
{
    const gert::StorageShape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_IF(queryShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "Query shape is nullptr."), return ge::GRAPH_FAILED);

    totalQTokens_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_T));
    numHeads_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_N));
    embeddingSize_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_D));

    const gert::StorageShape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_IF(keyShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "Key shape is nullptr."), return ge::GRAPH_FAILED);

    if (kvHeads_ == 0) {
        kvHeads_ = static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(BLOCKED_KV_DIM_KV_HEAD));
    }

    const gert::StorageShape *blockTableShape = context->GetInputShape(BLOCK_TABLE_INDEX);
    OP_CHECK_IF(blockTableShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "BlockTable shape is nullptr."), return ge::GRAPH_FAILED);

    batch_ = static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(BLOCK_TABLE_DIM_BATCH));
    maxBlocksPerBatch_ = static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(BLOCK_TABLE_DIM_MAX_BLOCKS));

    const gert::StorageShape *selectIdxShape = context->GetInputShape(SELECT_IDX_INDEX);
    OP_CHECK_IF(selectIdxShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "SelectIdx shape is nullptr."), return ge::GRAPH_FAILED);

    kvHeads_ = static_cast<uint32_t>(selectIdxShape->GetStorageShape().GetDim(SELECT_IDX_DIM_KV_HEAD));
    maxQSeqlen_ = static_cast<uint32_t>(selectIdxShape->GetStorageShape().GetDim(SELECT_IDX_DIM_SEQ));
    topK_ = static_cast<uint32_t>(selectIdxShape->GetStorageShape().GetDim(SELECT_IDX_DIM_TOPK));

    auto queryDesc = context->GetInputDesc(QUERY_INDEX);
    if (queryDesc != nullptr) {
        dataType_ = queryDesc->GetDataType();
    }

    if (scaleValue_ < 1e-9f && scaleValue_ > -1e-9f && embeddingSize_ > 0) {
        scaleValue_ = 1.0f / std::sqrt(static_cast<float>(embeddingSize_));
    }
    if (socVer_ == SOC_VER_950_CODE) {
        if (CheckAttentionOutDtype(context) != GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASATiling::ParseSeqlens(gert::TilingContext *context)
{
    const gert::Tensor *seqLensTensor = context->GetInputTensor(ACTUAL_SEQ_LENGTHS_INDEX);
    if (seqLensTensor != nullptr) {
        qSeqLenList_ = reinterpret_cast<const int32_t *>(seqLensTensor->GetAddr());
    }

    const gert::Tensor *seqLensKvTensor = context->GetInputTensor(ACTUAL_SEQ_LENGTHS_KV_INDEX);
    if (seqLensKvTensor != nullptr) {
        kvSeqLenList_ = reinterpret_cast<const int32_t *>(seqLensKvTensor->GetAddr());
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASATiling::ParseSelectNumIdx(gert::TilingContext *context)
{
    selectNumIdxList_ = nullptr;
    const gert::Tensor *selectNumIdxTensor = context->GetOptionalInputTensor(SELECT_NUM_IDX_INDEX);
    if (selectNumIdxTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (!gert::TensorPlacementUtils::IsOnHost(selectNumIdxTensor->GetPlacement())) {
        OP_LOGW(context->GetNodeName(),
            "selectNumIdx tiling data is not on Host; use topK=%u for FD cost estimation.", topK_);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t selectNumIdxSize = selectNumIdxTensor->GetShapeSize();
    const uint64_t requiredSize = static_cast<uint64_t>(kvHeads_) * maxQSeqlen_;
    if (selectNumIdxSize < 0 || static_cast<uint64_t>(selectNumIdxSize) < requiredSize) {
        OP_LOGW(context->GetNodeName(),
            "selectNumIdx contains %ld elements, fewer than required %lu; "
            "use topK=%u for FD cost estimation.",
            selectNumIdxSize, requiredSize, topK_);
        return ge::GRAPH_SUCCESS;
    }

    selectNumIdxList_ = selectNumIdxTensor->GetData<int32_t>();
    if (selectNumIdxList_ == nullptr) {
        OP_LOGW(context->GetNodeName(),
            "selectNumIdx Host data is nullptr; use topK=%u for FD cost estimation.", topK_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASATiling::CalculateTaskSplit(gert::TilingContext *context)
{
    totalTaskNum_ = totalQTokens_ * kvHeads_;
    blockDim_ = std::min(totalTaskNum_, aicNum_);
    if (blockDim_ == 0) {
        blockDim_ = 1;
    }

    enableFd_ = false;
    fdCoreRange_.perCoreTaskNum = 0;
    fdCombineRange_.combineTaskNum = 0;
    fdCombineRange_.partialTaskNum = 0;
    fdCoreRange_.taskStart.fill(0);
    fdCoreRange_.taskEnd.fill(0);
    fdCombineRange_.baseTask.fill(0);
    fdCombineRange_.partialStartByBase.fill(0);
    fdCombineRange_.partialCountByBase.fill(0);

    const uint32_t groupSize = kvHeads_ == 0 ? 0 : numHeads_ / kvHeads_;
    const bool fdDtypeSupported = dataType_ == ge::DT_FLOAT16 || dataType_ == ge::DT_BF16 ||
        (socVer_ == SOC_VER_950_CODE && dataType_ == ge::DT_FLOAT8_E4M3FN);
    const bool fdShapeSupported = fdDtypeSupported && innerPrecise_ == 4 &&
        embeddingSize_ == 128 && blockSize_ == 128 && topK_ >= 12 && topK_ <= 16 && kvHeads_ > 0 &&
        numHeads_ % kvHeads_ == 0 && groupSize > 0 && groupSize <= 128 && maxQSeqlen_ >= totalQTokens_ &&
        totalTaskNum_ > 0 && static_cast<uint64_t>(totalTaskNum_) * 10 < static_cast<uint64_t>(aicNum_) * 3;
    if (!fdShapeSupported) {
        return ge::GRAPH_SUCCESS;
    }

    // Match splitBN2S1GS2's load-balancing model without its B/N/S1 axis
    // merging: flatten every selectable block of every base task onto one
    // continuous S2 axis, then give each selected AIC a contiguous range.
    const uint32_t totalSplitTaskNum = totalTaskNum_ * topK_;
    if (totalSplitTaskNum <= blockDim_ || aicNum_ == 0) {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t totalCost = 0U;
    uint64_t totalValidKvBlockNum = 0U;
    uint32_t validTaskNum = 0U;
    const uint32_t m16 = CeilDiv(groupSize, 16U);
    for (uint32_t qToken = 0U; qToken < totalQTokens_; ++qToken) {
        for (uint32_t kvHead = 0U; kvHead < kvHeads_; ++kvHead) {
            uint32_t validBlockNum = topK_;
            if (selectNumIdxList_ != nullptr) {
                const uint64_t offset = static_cast<uint64_t>(kvHead) * maxQSeqlen_ + qToken;
                const int32_t selectNum = selectNumIdxList_[offset];
                validBlockNum = selectNum <= 0 ?
                    0U : std::min(static_cast<uint32_t>(selectNum), topK_);
            }
            if (validBlockNum == 0U) {
                continue;
            }

            totalCost += FD_COST_M16 * static_cast<uint64_t>(m16) +
                FD_COST_N * static_cast<uint64_t>(validBlockNum) +
                FD_COST_M16_N * static_cast<uint64_t>(m16) * static_cast<uint64_t>(validBlockNum);
            totalValidKvBlockNum += validBlockNum;
            ++validTaskNum;
        }
    }
    if (totalCost == 0U || totalValidKvBlockNum == 0U) {
        return ge::GRAPH_SUCCESS;
    }

    const uint64_t maxCoreLimit = std::min(static_cast<uint64_t>(aicNum_),
        static_cast<uint64_t>(SASA_FD_MAX_AIC));
    const uint32_t maxCore = static_cast<uint32_t>(std::min(maxCoreLimit, totalValidKvBlockNum));
    const uint32_t bestCoreNum = CalcFdBestCore(totalCost, FD_LAUNCH_COST, maxCore);
    fdCoreRange_.perCoreTaskNum = CeilDiv(totalSplitTaskNum, bestCoreNum);
    const uint32_t usedCoreNum = bestCoreNum;
    const uint32_t activeCoreNum = CeilDiv(totalSplitTaskNum, fdCoreRange_.perCoreTaskNum);
    if (usedCoreNum == 0 || usedCoreNum > SASA_FD_MAX_AIC) {
        return ge::GRAPH_SUCCESS;
    }
    for (uint32_t core = 0; core < usedCoreNum; ++core) {
        fdCoreRange_.taskStart[core] =
            std::min(core * fdCoreRange_.perCoreTaskNum, totalSplitTaskNum);
        fdCoreRange_.taskEnd[core] =
            std::min(fdCoreRange_.taskStart[core] + fdCoreRange_.perCoreTaskNum, totalSplitTaskNum);
    }

    // A base task needs combine only when a core boundary cuts its topK
    // interval. Partial workspace ids stay contiguous for each base task.
    for (uint32_t task = 0; task < totalTaskNum_; ++task) {
        const uint32_t taskStart = task * topK_;
        const uint32_t firstCore = taskStart / fdCoreRange_.perCoreTaskNum;
        const uint32_t lastCore = (taskStart + topK_ - 1) / fdCoreRange_.perCoreTaskNum;
        const uint32_t splitCount = lastCore - firstCore + 1;
        if (splitCount <= 1) {
            continue;
        }
        fdCombineRange_.baseTask[fdCombineRange_.combineTaskNum++] = task;
        fdCombineRange_.partialStartByBase[task] = fdCombineRange_.partialTaskNum;
        fdCombineRange_.partialCountByBase[task] = splitCount;
        fdCombineRange_.partialTaskNum += splitCount;
    }
    fdLseSubStride_ = CeilDiv(CeilDiv(groupSize, 2), 8) * 8;
    blockDim_ = usedCoreNum;
    enableFd_ = true;
    OP_LOGI(context->GetNodeName(),
        "Enable %s FlashDecoding: baseTasks=%u, validTasks=%u, splitTasks=%u, validKvBlocks=%lu, "
        "totalCost=%lu, bestCores=%u, perCoreTasks=%u, usedCores=%u, activeCores=%u, "
        "combineTasks=%u, partialTasks=%u.",
        socVer_ == SOC_VER_950_CODE ? "Arch35" : "Arch22", totalTaskNum_, validTaskNum, totalSplitTaskNum,
        totalValidKvBlockNum, totalCost, bestCoreNum, fdCoreRange_.perCoreTaskNum, usedCoreNum, activeCoreNum,
        fdCombineRange_.combineTaskNum, fdCombineRange_.partialTaskNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASATiling::CalculateWorkSpace(gert::TilingContext *context)
{
    if (socVer_ != SOC_VER_950_CODE) {
        constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = 131072;
        constexpr uint32_t NUM3 = 3;
        mm1OutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        smOnlineOutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(uint16_t) * NUM3;
        mm2OutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        updateSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        const uint64_t identityIdxSize = static_cast<uint64_t>(topK_) * sizeof(int32_t);
        const uint64_t pipelineWorkspaceSize =
            identityIdxSize + mm1OutSize_ + smOnlineOutSize_ + mm2OutSize_ + updateSize_;
        if (enableFd_) {
            constexpr uint64_t WORKSPACE_ALIGNMENT = 512;
            fdIdentityOffset_ = 0;
            fdPartialLseOffset_ = AlignUp(pipelineWorkspaceSize, WORKSPACE_ALIGNMENT);
            fdPartialLseSize_ =
                static_cast<uint64_t>(fdCombineRange_.partialTaskNum) * 2 * fdLseSubStride_ * sizeof(float);
            fdPartialOOffset_ = AlignUp(fdPartialLseOffset_ + fdPartialLseSize_, WORKSPACE_ALIGNMENT);
            fdPartialOSize_ = static_cast<uint64_t>(fdCombineRange_.partialTaskNum) *
                (numHeads_ / kvHeads_) * embeddingSize_ * sizeof(float);
            const uint64_t userWorkspaceSize = fdPartialOOffset_ + fdPartialOSize_;
            if (userWorkspaceSize > std::numeric_limits<size_t>::max() - libapiSize_) {
                OP_LOGE(context->GetNodeName(), "FlashDecoding workspace size overflow.");
                return ge::GRAPH_FAILED;
            }
            workSpaceSize_ = libapiSize_ + userWorkspaceSize;
        } else {
            workSpaceSize_ = libapiSize_ + pipelineWorkspaceSize;
        }
    } else {
        if (enableFd_) {
            constexpr uint64_t WORKSPACE_ALIGNMENT = 512;
            const uint64_t identityIdxSize = static_cast<uint64_t>(topK_) * sizeof(int32_t);
            fdIdentityOffset_ = 0;
            fdPartialLseOffset_ = AlignUp(identityIdxSize, WORKSPACE_ALIGNMENT);
            fdPartialLseSize_ =
                static_cast<uint64_t>(fdCombineRange_.partialTaskNum) * 2 * fdLseSubStride_ * sizeof(float);
            fdPartialOOffset_ = AlignUp(fdPartialLseOffset_ + fdPartialLseSize_, WORKSPACE_ALIGNMENT);
            fdPartialOSize_ = static_cast<uint64_t>(fdCombineRange_.partialTaskNum) *
                (numHeads_ / kvHeads_) * embeddingSize_ * sizeof(float);
            const uint64_t userWorkspaceSize = fdPartialOOffset_ + fdPartialOSize_;
            if (userWorkspaceSize > std::numeric_limits<size_t>::max() - libapiSize_) {
                OP_LOGE(context->GetNodeName(), "FlashDecoding workspace size overflow.");
                return ge::GRAPH_FAILED;
            }
            workSpaceSize_ = libapiSize_ + userWorkspaceSize;
        } else {
            uint32_t dtypeSize = (dataType_ == ge::DT_FLOAT8_E4M3FN) ? 1 : 2;
            uint64_t perTaskWorkspace = static_cast<uint64_t>(topK_) * blockSize_ * embeddingSize_ * dtypeSize * 2;
            uint64_t identityIdxSize = static_cast<uint64_t>(topK_) * sizeof(int32_t);
            workSpaceSize_ = libapiSize_ + identityIdxSize + static_cast<uint64_t>(blockDim_) * perTaskWorkspace;
        }
    }

    context->SetBlockDim(blockDim_);
    size_t *workspaceArray = context->GetWorkspaceSizes(1);
    if (workspaceArray != nullptr) {
        workspaceArray[0] = static_cast<size_t>(workSpaceSize_);
    }

    return ge::GRAPH_SUCCESS;
}

uint64_t SASATiling::GenerateTilingKey()
{
    if (socVer_ != SOC_VER_950_CODE) {
        if (dataType_ == ge::DT_BF16 && embeddingSize_ == 128 && blockSize_ == 128) {
            if (enableFd_) {
                return SASA_BF16_D128_ARCH22_FD_TILING;
            }
            return SASA_BF16_D128_ARCH22_TILING;
        }
        if (dataType_ == ge::DT_FLOAT16 && embeddingSize_ == 128 && blockSize_ == 128) {
            if (enableFd_) {
                return SASA_FP16_D128_ARCH22_FD_TILING;
            }
            return SASA_FP16_D128_ARCH22_TILING;
        }
        return SASA_FP16_D128_ARCH22_TILING;
    }
    if (dataType_ == ge::DT_FLOAT8_E4M3FN && embeddingSize_ == 128 && blockSize_ == 128) {
        if (attentionOutDtype_ == ge::DT_BF16) {
            if (enableFd_) {
                return SASA_FP8_D128_BF16_ARCH35_FD_TILING;
            }
            return SASA_FP8_D128_BF16_TILING;
        }
        if (enableFd_) {
            return SASA_FP8_D128_ARCH35_FD_TILING;
        }
        return SASA_FP8_D128_TILING;
    }
    if (dataType_ == ge::DT_BF16 && embeddingSize_ == 128 && blockSize_ == 128) {
        if (enableFd_) {
            return SASA_BF16_D128_ARCH35_FD_TILING;
        }
        return SASA_BF16_D128_TILING;
    }
    if (dataType_ == ge::DT_FLOAT16 && embeddingSize_ == 128 && blockSize_ == 128) {
        if (enableFd_) {
            return SASA_FP16_D128_ARCH35_FD_TILING;
        }
        return SASA_FP16_D128_TILING;
    }
    return SASA_FP16_D128_TILING;
}

ge::graphStatus SASATiling::FillTilingData(gert::TilingContext *context)
{
    tilingData_->set_batch(batch_);
    tilingData_->set_numHeads(numHeads_);
    tilingData_->set_kvHeads(kvHeads_);
    tilingData_->set_embeddingSize(embeddingSize_);
    tilingData_->set_blockSize(blockSize_);
    tilingData_->set_topK(topK_);
    tilingData_->set_maxBlocksPerBatch(maxBlocksPerBatch_);
    tilingData_->set_totalQTokens(totalQTokens_);
    tilingData_->set_totalTaskNum(totalTaskNum_);
    tilingData_->set_firstBatchTaskNum(kvHeads_);
    tilingData_->set_scaleValue(scaleValue_);
    tilingData_->set_innerPrecise(innerPrecise_);
    tilingData_->set_maxQSeqlen(maxQSeqlen_);
    tilingData_->set_mm1OutSize(mm1OutSize_);
    tilingData_->set_smOnlineOutSize(smOnlineOutSize_);
    tilingData_->set_mm2OutSize(mm2OutSize_);
    tilingData_->set_updateSize(updateSize_);
    tilingData_->set_workSpaceSize(workSpaceSize_);
    uint32_t groupSize = (kvHeads_ > 0) ? (numHeads_ / kvHeads_) : 1;
    tilingData_->set_groupSize(groupSize);
    uint64_t tilingKey = GenerateTilingKey();
    tilingData_->set_tilingKey(tilingKey);
    context->SetTilingKey(tilingKey);

    // BaseTileInfo
    uint32_t qBaseTile = (embeddingSize_ <= 128) ? 128 : 64;
    uint32_t kvBaseTile = blockSize_;
    tilingData_->set_qBaseTile(qBaseTile);
    tilingData_->set_kvBaseTile(kvBaseTile);

    // MmPhaseL1TileInfo: QK matmul L1 tile = [qBaseTile, kvBaseTile, embed]
    tilingData_->set_mm1L1TileM(qBaseTile);
    tilingData_->set_mm1L1TileN(kvBaseTile);
    tilingData_->set_mm1L1TileKLeft(embeddingSize_);
    tilingData_->set_mm1L1TileKRight(embeddingSize_);
    // PV matmul L1 tile = [qBaseTile, embed, kvBaseTile]
    tilingData_->set_mm2L1TileM(qBaseTile);
    tilingData_->set_mm2L1TileN(embeddingSize_);
    tilingData_->set_mm2L1TileKLeft(kvBaseTile);
    tilingData_->set_mm2L1TileKRight(kvBaseTile);
    // Buffer counts
    tilingData_->set_qL1BufNum(1);
    tilingData_->set_kL1BufNum(1);
    tilingData_->set_vL1BufNum(1);
    tilingData_->set_pL1BufNum(3);  // PRE_LAUNCH + 1
    tilingData_->set_fdLseSubStride(enableFd_ ? fdLseSubStride_ : 0);
    tilingData_->set_fdCorePerCoreTaskNum(fdCoreRange_.perCoreTaskNum);
    tilingData_->set_fdCoreTaskStart(fdCoreRange_.taskStart.data());
    tilingData_->set_fdCoreTaskEnd(fdCoreRange_.taskEnd.data());
    tilingData_->set_fdCombineTaskNum(fdCombineRange_.combineTaskNum);
    tilingData_->set_fdCombineBaseTask(fdCombineRange_.baseTask.data());
    tilingData_->set_fdPartialStartByBase(fdCombineRange_.partialStartByBase.data());
    tilingData_->set_fdPartialCountByBase(fdCombineRange_.partialCountByBase.data());
    tilingData_->set_fdIdentityOffset(fdIdentityOffset_);
    tilingData_->set_fdPartialLseOffset(fdPartialLseOffset_);
    tilingData_->set_fdPartialOOffset(fdPartialOOffset_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASATiling::GetTiling(gert::TilingContext *context,
    SparseAttentionScoreTilingData &tilingData)
{
    tilingData_ = &tilingData;

    ge::graphStatus ret = GetNpuInfo(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ParseAttrs(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ParseInputTensors(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ParseSeqlens(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ParseSelectNumIdx(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CalculateTaskSplit(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CalculateWorkSpace(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = FillTilingData(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASATiling::SetTilingData(gert::TilingContext *context,
    SparseAttentionScoreTilingData &tilingData)
{
    OP_CHECK_IF(context->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "RawTilingData got from GE context is nullptr."), return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingSparseAttentionScore(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "Context is nullptr."), return ge::GRAPH_FAILED);
    SparseAttentionScoreTilingData tilingData;
    SASATiling tiling;
    if (tiling.GetTiling(context, tilingData) == ge::GRAPH_SUCCESS) {
        tiling.SetTilingData(context, tilingData);
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE(context->GetNodeName(), "GetTiling failed");
        return ge::GRAPH_FAILED;
    }
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForSparseAttentionScore(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseAttentionScore)
    .Tiling(TilingSparseAttentionScore)
    .TilingInputsDataDependency({5, 6, 7},
        {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_AICPU})
    .TilingParse<SparseAttentionScoreCompileInfo>(TilingPrepareForSparseAttentionScore);

}  // namespace optiling
