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
constexpr int ATTR_INPUT_LAYOUT_INDEX = 5;
constexpr int ATTR_IS_DENSE_INDEX = 6;

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

    const char *inputLayoutPtr = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
    OP_CHECK_IF(inputLayoutPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "input_layout is nullptr."), return ge::GRAPH_FAILED);
    const std::string inputLayout(inputLayoutPtr);
    if (inputLayout == "TND") {
        isQNtd_ = false;
        isKvNtd_ = false;
    } else if (inputLayout == "NTD") {
        isQNtd_ = true;
        isKvNtd_ = true;
    } else if (inputLayout == "TND_BNSD") {
        isQNtd_ = false;
        isKvNtd_ = true;
    } else {
        OP_LOGE(context->GetNodeName(),
            "input_layout only supports TND, NTD or TND_BNSD, got %s.", inputLayoutPtr);
        return ge::GRAPH_FAILED;
    }

    const bool *isDensePtr = attrs->GetAttrPointer<bool>(ATTR_IS_DENSE_INDEX);
    denseMode_ = isDensePtr != nullptr && *isDensePtr;

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

    const auto &qShape = queryShape->GetStorageShape();
    if (qShape.GetDimNum() != 3U) {
        OP_LOGE(context->GetNodeName(), "Query must be rank 3, got %zu.", qShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    const int64_t qTokenDim = qShape.GetDim(isQNtd_ ? TND_DIM_N : TND_DIM_T);
    const int64_t qHeadDim = qShape.GetDim(isQNtd_ ? TND_DIM_T : TND_DIM_N);
    const int64_t qEmbedDim = qShape.GetDim(TND_DIM_D);
    if (qTokenDim <= 0 || qHeadDim <= 0 || qEmbedDim <= 0) {
        OP_LOGE(context->GetNodeName(), "Query dimensions must be positive.");
        return ge::GRAPH_FAILED;
    }
    totalQTokens_ = static_cast<uint32_t>(qTokenDim);
    numHeads_ = static_cast<uint32_t>(qHeadDim);
    embeddingSize_ = static_cast<uint32_t>(qEmbedDim);

    const gert::StorageShape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_IF(keyShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "Key shape is nullptr."), return ge::GRAPH_FAILED);
    const gert::StorageShape *valueShape = context->GetInputShape(VALUE_INDEX);
    OP_CHECK_IF(valueShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "Value shape is nullptr."), return ge::GRAPH_FAILED);
    const auto &kShape = keyShape->GetStorageShape();
    const auto &vShape = valueShape->GetStorageShape();
    if (kShape.GetDimNum() != 4U || vShape.GetDimNum() != 4U) {
        OP_LOGE(context->GetNodeName(), "Key and value must both be rank 4.");
        return ge::GRAPH_FAILED;
    }
    for (size_t dim = 0; dim < 4U; ++dim) {
        if (kShape.GetDim(dim) != vShape.GetDim(dim)) {
            OP_LOGE(context->GetNodeName(), "Key and value shapes must be identical.");
            return ge::GRAPH_FAILED;
        }
    }
    const int kvHeadDimIdx = isKvNtd_ ? BLOCKED_KV_DIM_BLOCK_SIZE : BLOCKED_KV_DIM_KV_HEAD;
    const int blockSizeDimIdx = isKvNtd_ ? BLOCKED_KV_DIM_KV_HEAD : BLOCKED_KV_DIM_BLOCK_SIZE;
    const int64_t kvHeadsFromShape = kShape.GetDim(kvHeadDimIdx);
    const int64_t blockSizeFromShape = kShape.GetDim(blockSizeDimIdx);
    const int64_t kvEmbedDim = kShape.GetDim(BLOCKED_KV_DIM_D);
    if (kvHeadsFromShape <= 0 || blockSizeFromShape <= 0 || kvEmbedDim <= 0) {
        OP_LOGE(context->GetNodeName(), "Key/value dimensions must be positive.");
        return ge::GRAPH_FAILED;
    }
    if (kvHeads_ == 0U) {
        kvHeads_ = static_cast<uint32_t>(kvHeadsFromShape);
    }
    if (static_cast<uint32_t>(kvHeadsFromShape) != kvHeads_ ||
        static_cast<uint32_t>(blockSizeFromShape) != blockSize_ ||
        static_cast<uint32_t>(kvEmbedDim) != embeddingSize_) {
        OP_LOGE(context->GetNodeName(),
            "Q/K/V shape does not match attrs: layout=%s, qHeads=%u, kvHeads(attr/shape)=%u/%ld, "
            "headDim(Q/KV)=%u/%ld, blockSize(attr/shape)=%u/%ld.",
            isQNtd_ ? "NTD" : "TND", numHeads_, kvHeads_, kvHeadsFromShape,
            embeddingSize_, kvEmbedDim, blockSize_, blockSizeFromShape);
        return ge::GRAPH_FAILED;
    }
    if (numHeads_ % kvHeads_ != 0U) {
        OP_LOGE(context->GetNodeName(), "qHeads=%u must be divisible by kvHeads=%u.", numHeads_, kvHeads_);
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *blockTableShape = context->GetInputShape(BLOCK_TABLE_INDEX);
    OP_CHECK_IF(blockTableShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
        "BlockTable shape is nullptr."), return ge::GRAPH_FAILED);
    if (blockTableShape->GetStorageShape().GetDimNum() != 2U) {
        OP_LOGE(context->GetNodeName(), "BlockTable must be rank 2.");
        return ge::GRAPH_FAILED;
    }

    batch_ = static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(BLOCK_TABLE_DIM_BATCH));
    maxBlocksPerBatch_ = static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(BLOCK_TABLE_DIM_MAX_BLOCKS));

    if (denseMode_) {
        maxQSeqlen_ = totalQTokens_;
    } else {
        const gert::StorageShape *selectIdxShape = context->GetOptionalInputShape(SELECT_IDX_INDEX);
        OP_CHECK_IF(selectIdxShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttentionScore",
            "SelectIdx is required in sparse mode."), return ge::GRAPH_FAILED);
        const auto &selectShape = selectIdxShape->GetStorageShape();
        if (selectShape.GetDimNum() != 3U) {
            OP_LOGE(context->GetNodeName(), "SelectIdx must be rank 3 in sparse mode.");
            return ge::GRAPH_FAILED;
        }
        const uint32_t selectKvHeads = static_cast<uint32_t>(selectShape.GetDim(SELECT_IDX_DIM_KV_HEAD));
        maxQSeqlen_ = static_cast<uint32_t>(selectShape.GetDim(SELECT_IDX_DIM_SEQ));
        const uint32_t selectTopK = static_cast<uint32_t>(selectShape.GetDim(SELECT_IDX_DIM_TOPK));
        if (selectKvHeads != kvHeads_ || selectTopK != topK_ || maxQSeqlen_ < totalQTokens_) {
            OP_LOGE(context->GetNodeName(),
                "SelectIdx shape must match [kvHeads, >=totalQTokens, topK], got [%u,%u,%u], "
                "expected [%u,>=%u,%u].", selectKvHeads, maxQSeqlen_, selectTopK,
                kvHeads_, totalQTokens_, topK_);
            return ge::GRAPH_FAILED;
        }
    }

    auto queryDesc = context->GetInputDesc(QUERY_INDEX);
    if (queryDesc != nullptr) {
        dataType_ = queryDesc->GetDataType();
    }
    if (denseMode_ &&
        (socVer_ != SOC_VER_950_CODE || dataType_ != ge::DT_FLOAT8_E4M3FN)) {
        OP_LOGE(context->GetNodeName(),
            "DenseAttentionScore only supports FP8 input on Arch35, got dtype=%d, soc=%u.",
            static_cast<int32_t>(dataType_), socVer_);
        return ge::GRAPH_FAILED;
    }
    if (!denseMode_ && (isQNtd_ || isKvNtd_)) {
        OP_LOGE(context->GetNodeName(),
            "SparseAttentionScore only supports the legacy TND layout.");
        return ge::GRAPH_FAILED;
    }
    if (isQNtd_ && dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16 &&
        !(socVer_ == SOC_VER_950_CODE && dataType_ == ge::DT_FLOAT8_E4M3FN)) {
        OP_LOGE(context->GetNodeName(),
            "NTD supports FP16/BF16 on Arch22/Arch35 and FP8 on Arch35, got dtype=%d, soc=%u.",
            static_cast<int32_t>(dataType_), socVer_);
        return ge::GRAPH_FAILED;
    }

    const uint64_t qTokenStride = isQNtd_ ? embeddingSize_ :
        static_cast<uint64_t>(numHeads_) * embeddingSize_;
    const uint64_t qHeadStride = isQNtd_ ?
        static_cast<uint64_t>(totalQTokens_) * embeddingSize_ : embeddingSize_;
    const uint64_t kvHeadStride = isKvNtd_ ?
        static_cast<uint64_t>(blockSize_) * embeddingSize_ : embeddingSize_;
    const uint64_t kvTokenStride = isKvNtd_ ? embeddingSize_ :
        static_cast<uint64_t>(kvHeads_) * embeddingSize_;
    if (qTokenStride > std::numeric_limits<uint32_t>::max() ||
        qHeadStride > std::numeric_limits<uint32_t>::max() ||
        kvHeadStride > std::numeric_limits<uint32_t>::max() ||
        kvTokenStride > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(context->GetNodeName(), "Q/K/V layout stride exceeds uint32 range.");
        return ge::GRAPH_FAILED;
    }
    qTokenStride_ = static_cast<uint32_t>(qTokenStride);
    qHeadStride_ = static_cast<uint32_t>(qHeadStride);
    kvHeadStride_ = static_cast<uint32_t>(kvHeadStride);
    kvTokenStride_ = static_cast<uint32_t>(kvTokenStride);

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
    qSeqLenList_ = nullptr;
    kvSeqLenList_ = nullptr;
    qSeqLenCount_ = 0;
    kvSeqLenCount_ = 0;
    const gert::Tensor *seqLensTensor = context->GetInputTensor(ACTUAL_SEQ_LENGTHS_INDEX);
    if (seqLensTensor != nullptr) {
        if (gert::TensorPlacementUtils::IsOnHost(seqLensTensor->GetPlacement())) {
            qSeqLenList_ = seqLensTensor->GetData<int32_t>();
            const int64_t shapeSize = seqLensTensor->GetShapeSize();
            qSeqLenCount_ = shapeSize > 0 ? static_cast<uint64_t>(shapeSize) : 0;
        }
    }

    const gert::Tensor *seqLensKvTensor = context->GetInputTensor(ACTUAL_SEQ_LENGTHS_KV_INDEX);
    if (seqLensKvTensor != nullptr) {
        if (gert::TensorPlacementUtils::IsOnHost(seqLensKvTensor->GetPlacement())) {
            kvSeqLenList_ = seqLensKvTensor->GetData<int32_t>();
            const int64_t shapeSize = seqLensKvTensor->GetShapeSize();
            kvSeqLenCount_ = shapeSize > 0 ? static_cast<uint64_t>(shapeSize) : 0;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASATiling::ParseSelectNumIdx(gert::TilingContext *context)
{
    selectNumIdxList_ = nullptr;
    if (denseMode_) {
        return ge::GRAPH_SUCCESS;
    }
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

ge::graphStatus SASATiling::CalculateDenseTaskSplit(gert::TilingContext *context)
{
    if (kvHeads_ == 0 || numHeads_ % kvHeads_ != 0 || blockSize_ == 0 ||
        embeddingSize_ != 128 || blockSize_ != 128 || innerPrecise_ != 4) {
        OP_LOGE(context->GetNodeName(),
            "Dense FP8 only supports headDim=128, blockSize=128, innerPrecise=4 and valid GQA heads; "
            "got numHeads=%u, kvHeads=%u, headDim=%u, blockSize=%u, innerPrecise=%u.",
            numHeads_, kvHeads_, embeddingSize_, blockSize_, innerPrecise_);
        return ge::GRAPH_FAILED;
    }

    denseValidBlockCount_.assign(totalTaskNum_, 0U);
    denseBlockPrefix_.assign(static_cast<size_t>(totalTaskNum_) + 1U, 0U);
    uint32_t maxValidBlockNum = 0U;
    const bool hasHostSeqLens = qSeqLenList_ != nullptr && kvSeqLenList_ != nullptr &&
        qSeqLenCount_ >= batch_ && kvSeqLenCount_ >= batch_;
    if (hasHostSeqLens) {
        uint64_t qTokenOffset = 0U;
        for (uint32_t batchIdx = 0; batchIdx < batch_; ++batchIdx) {
            const int32_t qSeqLenValue = qSeqLenList_[batchIdx];
            const int32_t kvSeqLenValue = kvSeqLenList_[batchIdx];
            if (qSeqLenValue <= 0 || kvSeqLenValue < qSeqLenValue) {
                OP_LOGE(context->GetNodeName(),
                    "Dense FP8 requires 0 < qSeqLen <= kvSeqLen for every batch; "
                    "batch=%u has qSeqLen=%d, kvSeqLen=%d.",
                    batchIdx, qSeqLenValue, kvSeqLenValue);
                return ge::GRAPH_FAILED;
            }
            const uint32_t qSeqLen = static_cast<uint32_t>(qSeqLenValue);
            const uint32_t kvSeqLen = static_cast<uint32_t>(kvSeqLenValue);
            if (CeilDiv(kvSeqLen, blockSize_) > maxBlocksPerBatch_) {
                OP_LOGE(context->GetNodeName(),
                    "Dense FP8 blockTable cannot cover actual KV length: batch=%u, kvSeqLen=%u, "
                    "blockSize=%u, maxBlocksPerBatch=%u.",
                    batchIdx, kvSeqLen, blockSize_, maxBlocksPerBatch_);
                return ge::GRAPH_FAILED;
            }

            const uint32_t historyLen = kvSeqLen - qSeqLen;
            for (uint32_t localQToken = 0; localQToken < qSeqLen; ++localQToken) {
                if (qTokenOffset >= totalQTokens_) {
                    OP_LOGE(context->GetNodeName(),
                        "Sum(actualSeqLengths) exceeds totalQTokens=%u.", totalQTokens_);
                    return ge::GRAPH_FAILED;
                }
                const uint32_t visibleKvLen = historyLen + localQToken + 1U;
                const uint32_t validBlockNum = CeilDiv(visibleKvLen, blockSize_);
                maxValidBlockNum = std::max(maxValidBlockNum, validBlockNum);
                for (uint32_t kvHead = 0; kvHead < kvHeads_; ++kvHead) {
                    const uint32_t baseTask = static_cast<uint32_t>(qTokenOffset) * kvHeads_ + kvHead;
                    denseValidBlockCount_[baseTask] = validBlockNum;
                    const uint64_t nextPrefix = denseBlockPrefix_[baseTask] + validBlockNum;
                    if (nextPrefix > std::numeric_limits<uint32_t>::max()) {
                        OP_LOGE(context->GetNodeName(), "Dense FP8 flattened block count exceeds uint32 range.");
                        return ge::GRAPH_FAILED;
                    }
                    denseBlockPrefix_[baseTask + 1U] = nextPrefix;
                }
                ++qTokenOffset;
            }
        }
        if (qTokenOffset != totalQTokens_) {
            OP_LOGE(context->GetNodeName(),
                "Sum(actualSeqLengths)=%lu does not match totalQTokens=%u.", qTokenOffset, totalQTokens_);
            return ge::GRAPH_FAILED;
        }
    } else {
        // Eager ACLNN keeps sequence-length tensors on NPU.  Host tiling uses
        // the block-table width as a rectangular upper bound; the kernel
        // clamps every range with the real causal visible length and emits a
        // neutral partial for an upper-bound-only shard.
        if (maxBlocksPerBatch_ == 0U) {
            OP_LOGE(context->GetNodeName(), "Dense FP8 requires a non-empty blockTable.");
            return ge::GRAPH_FAILED;
        }
        maxValidBlockNum = maxBlocksPerBatch_;
        for (uint32_t baseTask = 0; baseTask < totalTaskNum_; ++baseTask) {
            denseValidBlockCount_[baseTask] = maxBlocksPerBatch_;
            const uint64_t nextPrefix = denseBlockPrefix_[baseTask] + maxBlocksPerBatch_;
            if (nextPrefix > std::numeric_limits<uint32_t>::max()) {
                OP_LOGE(context->GetNodeName(), "Dense FP8 flattened block upper bound exceeds uint32 range.");
                return ge::GRAPH_FAILED;
            }
            denseBlockPrefix_[baseTask + 1U] = nextPrefix;
        }
        OP_LOGI(context->GetNodeName(),
            "Dense FP8 uses blockTable-width upper-bound tiling because actual sequence lengths are on NPU: "
            "baseTasks=%u, blocksPerTask=%u.", totalTaskNum_, maxBlocksPerBatch_);
    }

    fdIdentityCount_ = std::max(1U, maxValidBlockNum);
    const uint64_t totalValidKvBlockNum = denseBlockPrefix_.back();
    if (totalValidKvBlockNum <= blockDim_ || aicNum_ == 0U ||
        totalTaskNum_ > SASA_FD_MAX_BASE_TASK) {
        OP_LOGI(context->GetNodeName(),
            "Use Arch35 dense FP8 without FD: baseTasks=%u, validKvBlocks=%lu, normalBlockDim=%u.",
            totalTaskNum_, totalValidKvBlockNum, blockDim_);
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t groupSize = numHeads_ / kvHeads_;
    const uint32_t m16 = CeilDiv(groupSize, 16U);
    uint64_t totalCost = 0U;
    for (uint32_t baseTask = 0; baseTask < totalTaskNum_; ++baseTask) {
        const uint64_t validBlockNum = denseValidBlockCount_[baseTask];
        totalCost += FD_COST_M16 * static_cast<uint64_t>(m16) +
            FD_COST_N * validBlockNum +
            FD_COST_M16_N * static_cast<uint64_t>(m16) * validBlockNum;
    }
    const uint64_t maxCoreLimit = std::min(static_cast<uint64_t>(aicNum_),
        static_cast<uint64_t>(SASA_FD_MAX_AIC));
    const uint32_t maxCore = static_cast<uint32_t>(std::min(maxCoreLimit, totalValidKvBlockNum));
    const uint32_t bestCoreNum = CalcFdBestCore(totalCost, FD_LAUNCH_COST, maxCore);
    if (bestCoreNum <= blockDim_) {
        OP_LOGI(context->GetNodeName(),
            "Use Arch35 dense FP8 without FD: cost-model bestCores=%u does not exceed normalBlockDim=%u.",
            bestCoreNum, blockDim_);
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t usedCoreNum = bestCoreNum;
    fdCoreRange_.usedCoreNum = usedCoreNum;
    fdCoreRange_.perCoreTaskNum = CeilDiv(static_cast<uint32_t>(totalValidKvBlockNum), usedCoreNum);
    std::array<uint64_t, SASA_FD_MAX_AIC> coreFlatStart{};
    std::array<uint64_t, SASA_FD_MAX_AIC> coreFlatEnd{};
    auto decodePosition = [this, totalValidKvBlockNum](uint64_t flatPos,
        uint32_t &baseTask, uint32_t &blockIdx) {
        if (flatPos >= totalValidKvBlockNum) {
            baseTask = totalTaskNum_;
            blockIdx = 0U;
            return;
        }
        const auto upper = std::upper_bound(denseBlockPrefix_.begin(), denseBlockPrefix_.end(), flatPos);
        baseTask = static_cast<uint32_t>(std::distance(denseBlockPrefix_.begin(), upper) - 1);
        blockIdx = static_cast<uint32_t>(flatPos - denseBlockPrefix_[baseTask]);
    };

    fdIdentityCount_ = 1U;
    for (uint32_t core = 0; core < usedCoreNum; ++core) {
        const uint64_t flatStart = totalValidKvBlockNum * core / usedCoreNum;
        const uint64_t flatEnd = totalValidKvBlockNum * (core + 1U) / usedCoreNum;
        coreFlatStart[core] = flatStart;
        coreFlatEnd[core] = flatEnd;
        fdCoreRange_.taskStart[core] = static_cast<uint32_t>(flatStart);
        fdCoreRange_.taskEnd[core] = static_cast<uint32_t>(flatEnd);
        decodePosition(flatStart, fdCoreRange_.startBaseTask[core], fdCoreRange_.startBlockIdx[core]);
        decodePosition(flatEnd, fdCoreRange_.endBaseTask[core], fdCoreRange_.endBlockIdx[core]);
        fdIdentityCount_ = std::max(fdIdentityCount_, static_cast<uint32_t>(flatEnd - flatStart));
    }

    for (uint32_t baseTask = 0; baseTask < totalTaskNum_; ++baseTask) {
        const uint64_t taskStart = denseBlockPrefix_[baseTask];
        const uint64_t taskEnd = denseBlockPrefix_[baseTask + 1U];
        uint32_t splitCount = 0U;
        for (uint32_t core = 0; core < usedCoreNum; ++core) {
            if (coreFlatStart[core] < taskEnd && coreFlatEnd[core] > taskStart) {
                ++splitCount;
            }
        }
        if (splitCount <= 1U) {
            continue;
        }
        fdCombineRange_.baseTask[fdCombineRange_.combineTaskNum++] = baseTask;
        fdCombineRange_.partialStartByBase[baseTask] = fdCombineRange_.partialTaskNum;
        fdCombineRange_.partialCountByBase[baseTask] = splitCount;
        fdCombineRange_.partialTaskNum += splitCount;
    }

    fdLseSubStride_ = CeilDiv(CeilDiv(groupSize, 2U), 8U) * 8U;
    blockDim_ = usedCoreNum;
    enableFd_ = true;
    OP_LOGI(context->GetNodeName(),
        "Enable Arch35 dense FP8 FlashDecoding: baseTasks=%u, splitBlocks=%lu, totalCost=%lu, "
        "bestCores=%u, perCoreBlocks=%u, combineTasks=%u, partialTasks=%u, identityCount=%u.",
        totalTaskNum_, totalValidKvBlockNum, totalCost, bestCoreNum, fdCoreRange_.perCoreTaskNum,
        fdCombineRange_.combineTaskNum, fdCombineRange_.partialTaskNum, fdIdentityCount_);
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
    fdIdentityCount_ = topK_;
    fdCoreRange_.perCoreTaskNum = 0;
    fdCoreRange_.usedCoreNum = 0;
    fdCombineRange_.combineTaskNum = 0;
    fdCombineRange_.partialTaskNum = 0;
    fdCoreRange_.taskStart.fill(0);
    fdCoreRange_.taskEnd.fill(0);
    fdCoreRange_.startBaseTask.fill(0);
    fdCoreRange_.startBlockIdx.fill(0);
    fdCoreRange_.endBaseTask.fill(0);
    fdCoreRange_.endBlockIdx.fill(0);
    fdCombineRange_.baseTask.fill(0);
    fdCombineRange_.partialStartByBase.fill(0);
    fdCombineRange_.partialCountByBase.fill(0);

    if (denseMode_) {
        return CalculateDenseTaskSplit(context);
    }

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
    fdCoreRange_.usedCoreNum = usedCoreNum;
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
            const uint64_t identityIdxSize = static_cast<uint64_t>(fdIdentityCount_) * sizeof(int32_t);
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
        } else if (denseMode_) {
            const uint64_t identityIdxSize = static_cast<uint64_t>(fdIdentityCount_) * sizeof(int32_t);
            if (identityIdxSize > std::numeric_limits<size_t>::max() - libapiSize_) {
                OP_LOGE(context->GetNodeName(), "Dense FP8 workspace size overflow.");
                return ge::GRAPH_FAILED;
            }
            fdIdentityOffset_ = 0;
            workSpaceSize_ = libapiSize_ + identityIdxSize;
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
            if (isQNtd_) {
                return enableFd_ ? SASA_BF16_D128_ARCH22_NTD_FD_TILING :
                    SASA_BF16_D128_ARCH22_NTD_TILING;
            }
            if (enableFd_) {
                return SASA_BF16_D128_ARCH22_FD_TILING;
            }
            return SASA_BF16_D128_ARCH22_TILING;
        }
        if (dataType_ == ge::DT_FLOAT16 && embeddingSize_ == 128 && blockSize_ == 128) {
            if (isQNtd_) {
                return enableFd_ ? SASA_FP16_D128_ARCH22_NTD_FD_TILING :
                    SASA_FP16_D128_ARCH22_NTD_TILING;
            }
            if (enableFd_) {
                return SASA_FP16_D128_ARCH22_FD_TILING;
            }
            return SASA_FP16_D128_ARCH22_TILING;
        }
        return SASA_FP16_D128_ARCH22_TILING;
    }
    if (dataType_ == ge::DT_FLOAT8_E4M3FN && embeddingSize_ == 128 && blockSize_ == 128) {
        if (attentionOutDtype_ == ge::DT_BF16) {
            if (denseMode_) {
                if (!isQNtd_ && isKvNtd_) {
                    return enableFd_ ? SASA_FP8_D128_BF16_ARCH35_DENSE_KVNTD_FD_TILING :
                        SASA_FP8_D128_BF16_ARCH35_DENSE_KVNTD_TILING;
                }
                if (isQNtd_) {
                    return enableFd_ ? SASA_FP8_D128_BF16_ARCH35_DENSE_NTD_FD_TILING :
                        SASA_FP8_D128_BF16_ARCH35_DENSE_NTD_TILING;
                }
                return enableFd_ ? SASA_FP8_D128_BF16_ARCH35_DENSE_FD_TILING :
                    SASA_FP8_D128_BF16_ARCH35_DENSE_TILING;
            }
            if (enableFd_) {
                return SASA_FP8_D128_BF16_ARCH35_FD_TILING;
            }
            return SASA_FP8_D128_BF16_TILING;
        }
        if (denseMode_) {
            if (!isQNtd_ && isKvNtd_) {
                return enableFd_ ? SASA_FP8_D128_ARCH35_DENSE_KVNTD_FD_TILING :
                    SASA_FP8_D128_ARCH35_DENSE_KVNTD_TILING;
            }
            if (isQNtd_) {
                return enableFd_ ? SASA_FP8_D128_ARCH35_DENSE_NTD_FD_TILING :
                    SASA_FP8_D128_ARCH35_DENSE_NTD_TILING;
            }
            return enableFd_ ? SASA_FP8_D128_ARCH35_DENSE_FD_TILING :
                SASA_FP8_D128_ARCH35_DENSE_TILING;
        }
        if (enableFd_) {
            return SASA_FP8_D128_ARCH35_FD_TILING;
        }
        return SASA_FP8_D128_TILING;
    }
    if (dataType_ == ge::DT_BF16 && embeddingSize_ == 128 && blockSize_ == 128) {
        if (isQNtd_) {
            return enableFd_ ? SASA_BF16_D128_ARCH35_NTD_FD_TILING :
                SASA_BF16_D128_ARCH35_NTD_TILING;
        }
        if (enableFd_) {
            return SASA_BF16_D128_ARCH35_FD_TILING;
        }
        return SASA_BF16_D128_TILING;
    }
    if (dataType_ == ge::DT_FLOAT16 && embeddingSize_ == 128 && blockSize_ == 128) {
        if (isQNtd_) {
            return enableFd_ ? SASA_FP16_D128_ARCH35_NTD_FD_TILING :
                SASA_FP16_D128_ARCH35_NTD_TILING;
        }
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
    tilingData_->set_denseMode(denseMode_ ? 1U : 0U);
    tilingData_->set_layoutMode(isQNtd_ ? 1U : 0U);
    tilingData_->set_qTokenStride(qTokenStride_);
    tilingData_->set_qHeadStride(qHeadStride_);
    tilingData_->set_kvHeadStride(kvHeadStride_);
    tilingData_->set_kvTokenStride(kvTokenStride_);
    tilingData_->set_fdUsedCoreNum(fdCoreRange_.usedCoreNum);
    tilingData_->set_fdIdentityCount(fdIdentityCount_);
    tilingData_->set_fdCorePerCoreTaskNum(fdCoreRange_.perCoreTaskNum);
    tilingData_->set_fdCoreTaskStart(fdCoreRange_.taskStart.data());
    tilingData_->set_fdCoreTaskEnd(fdCoreRange_.taskEnd.data());
    tilingData_->set_fdCoreStartBaseTask(fdCoreRange_.startBaseTask.data());
    tilingData_->set_fdCoreStartBlockIdx(fdCoreRange_.startBlockIdx.data());
    tilingData_->set_fdCoreEndBaseTask(fdCoreRange_.endBaseTask.data());
    tilingData_->set_fdCoreEndBlockIdx(fdCoreRange_.endBlockIdx.data());
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

IMPL_OP_OPTILING(SparseAttentionScore_950)
    .Tiling(TilingSparseAttentionScore)
    .TilingInputsDataDependency({5, 6, 7},
        {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_AICPU})
    .TilingParse<SparseAttentionScoreCompileInfo>(TilingPrepareForSparseAttentionScore);

}  // namespace optiling
