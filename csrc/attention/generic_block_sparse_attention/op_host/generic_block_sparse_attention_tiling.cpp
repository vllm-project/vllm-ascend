/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "generic_block_sparse_attention_tiling.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <string>
#include "log/log.h"
#include "err/ops_err.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_base.h"

constexpr int QUERY_INDEX = 0;
constexpr int KEY_INDEX = 1;
constexpr int VALUE_INDEX = 2;
constexpr int SPARSE_BLOCK_IDX_INDEX = 3;
constexpr int SPARSE_BLOCK_COUNT_INDEX = 4;
constexpr int METADATA_INDEX = 5;
constexpr int BLOCK_TABLE_INDEX = 15;

// Keep in sync with sparse_attention_score_metadata.h METADATA_TOTAL_SIZE.
constexpr uint32_t GSA_METADATA_TOTAL_SIZE = 1024U;

constexpr int ATTENTION_OUT_INDEX = 0;

constexpr int TND_DIM_T = 0;
constexpr int TND_DIM_N = 1;
constexpr int TND_DIM_D = 2;

constexpr int BLOCKED_KV_DIM_BLOCK_NUM = 0;
constexpr int BLOCKED_KV_DIM_BLOCK_SIZE = 1;
constexpr int BLOCKED_KV_DIM_KV_HEAD = 2;
constexpr int BLOCKED_KV_DIM_D = 3;

// TND + isPackedGQA=1 sparseBlockIdx 3D: [N_kv, totalQBlocks, topK]
constexpr int SPARSE_IDX_DIM_KV_HEAD = 0;
constexpr int SPARSE_IDX_DIM_Q_BLOCK = 1;
constexpr int SPARSE_IDX_DIM_KV_BLOCK = 2;
constexpr int SPARSE_IDX_DIM_NUM = 3;

// TND + isPackedGQA=1 sparseBlockCount 2D: [N_kv, totalQBlocks]
constexpr int SPARSE_COUNT_DIM_KV_HEAD = 0;
constexpr int SPARSE_COUNT_DIM_Q_BLOCK = 1;
constexpr int SPARSE_COUNT_DIM_NUM = 2;

constexpr int BLOCK_TABLE_DIM_BATCH = 0;
constexpr int BLOCK_TABLE_DIM_MAX_BLOCKS = 1;

constexpr int ATTR_BLOCK_SHAPE_INDEX = 0;
constexpr int ATTR_IS_PACKED_GQA_INDEX = 1;
constexpr int ATTR_Q_INPUT_LAYOUT_INDEX = 2;
constexpr int ATTR_KV_INPUT_LAYOUT_INDEX = 3;
constexpr int ATTR_SCALE_VALUE_INDEX = 4;
constexpr int ATTR_MASK_TYPE_INDEX = 5;
constexpr int ATTR_QUANT_TYPE_INDEX = 6;
constexpr int ATTR_SOFTMAX_PRECISION_INDEX = 8;
constexpr int ATTR_SOFTMAX_LSE_FLAG_INDEX = 11;

constexpr uint32_t SOC_VER_950_CODE = 4;
constexpr uint32_t GSA_MAX_SPARSE_BLOCK_CAPACITY = 16U;
constexpr uint32_t GSA_FD_MAX_ACTIVE_CORE_NUM = 32U;
constexpr uint32_t GSA_FD_MAX_COMBINE_TASK_NUM = 32U;
constexpr uint32_t GSA_FD_BASE_TASK_GATE_NUMERATOR = 3U;
constexpr uint32_t GSA_FD_BASE_TASK_GATE_DENOMINATOR = 10U;
constexpr uint64_t GSA_FD_WORKSPACE_ALIGNMENT = 512U;

namespace {
uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}
} // namespace

namespace optiling {

ge::graphStatus GSATiling::GetNpuInfo(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    // Task schedule is owned by AICPU metadata (saTotalTaskNum). Host only launches
    // all AIC cores; idle cores exit when taskIdx >= metadata saTotalTaskNum.
    blockDim_ = (aicNum_ == 0) ? 1U : aicNum_;
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    socVer_ = static_cast<uint32_t>(ascendcPlatform.GetSocVersion());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GSATiling::ParseAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention",
        "GetAttrs returned nullptr."), return ge::GRAPH_FAILED);

    const float *scalePtr = attrs->GetFloat(ATTR_SCALE_VALUE_INDEX);
    if (scalePtr != nullptr) {
        scaleValue_ = *scalePtr;
    }

    const gert::TypedContinuousVector<int64_t> *blockShapeArr = attrs->GetListInt(ATTR_BLOCK_SHAPE_INDEX);
    if (blockShapeArr != nullptr && blockShapeArr->GetSize() >= 2) {
        blockShapeX_ = static_cast<uint32_t>(blockShapeArr->GetData()[0]);
        blockShapeY_ = static_cast<uint32_t>(blockShapeArr->GetData()[1]);
    }

    const int64_t *softmaxPrecPtr = attrs->GetInt(ATTR_SOFTMAX_PRECISION_INDEX);
    if (softmaxPrecPtr != nullptr) {
        softmaxPrecision_ = static_cast<uint32_t>(*softmaxPrecPtr);
    }

    const char *layoutQPtr = attrs->GetStr(ATTR_Q_INPUT_LAYOUT_INDEX);
    if (layoutQPtr != nullptr) {
        layoutQ_ = std::string(layoutQPtr);
    }

    const char *layoutKvPtr = attrs->GetStr(ATTR_KV_INPUT_LAYOUT_INDEX);
    if (layoutKvPtr != nullptr) {
        layoutKv_ = std::string(layoutKvPtr);
    }

    const int64_t *maskTypePtr = attrs->GetInt(ATTR_MASK_TYPE_INDEX);
    if (maskTypePtr != nullptr) {
        maskType_ = *maskTypePtr;
    }

    const int64_t *quantTypePtr = attrs->GetInt(ATTR_QUANT_TYPE_INDEX);
    if (quantTypePtr != nullptr) {
        quantType_ = *quantTypePtr;
    }

    // Kernel task decode and sparse layouts are packed-GQA only (task = T * Nkv).
    const int64_t *isPackedGqaPtr = attrs->GetInt(ATTR_IS_PACKED_GQA_INDEX);
    const int64_t isPackedGQA = (isPackedGqaPtr != nullptr) ? *isPackedGqaPtr : 1;
    if (isPackedGQA != 1) {
        OP_LOGE(context->GetNodeName(),
                "Unsupported isPackedGQA=%ld, only 1 (packed GQA) is supported.",
                isPackedGQA);
        return ge::GRAPH_FAILED;
    }

    const int64_t *lseFlagPtr = attrs->GetInt(ATTR_SOFTMAX_LSE_FLAG_INDEX);
    if (lseFlagPtr != nullptr) {
        if (*lseFlagPtr != 0 && *lseFlagPtr != 1) {
            OP_LOGE(context->GetNodeName(),
                    "Unsupported returnSoftmaxlse=%ld, only 0 or 1 are supported.",
                    *lseFlagPtr);
            return ge::GRAPH_FAILED;
        }
        returnSoftmaxlse_ = (*lseFlagPtr == 1);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GSATiling::CheckAttentionOutDtype(gert::TilingContext *context)
{
    if (dataType_ == ge::DT_FLOAT8_E4M3FN) {
        attentionOutDtype_ = context->GetOutputDesc(ATTENTION_OUT_INDEX)->GetDataType();
        if (attentionOutDtype_ != ge::DT_FLOAT16 && attentionOutDtype_ != ge::DT_BF16) {
            OP_LOGE(context->GetNodeName(),
                    "The supported dtype of attentionOut is float16 or bfloat16 when the dtype of query/key/value is "
                    "all float8_e4m3fn, but now it is %d.",
                    attentionOutDtype_);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// FIA PR 6399: GetDynamicInputStride (TensorList). GBSA key/value are REQUIRED.
static ge::graphStatus ValidatePagedBbndDim0OnlyNonContig(gert::TilingContext *context, uint64_t inputIndex,
                                                          const gert::Shape &shape, const char *tensorName)
{
    auto *stride = context->GetRequiredInputStride(inputIndex);
    if (stride == nullptr || stride->GetDimNum() != shape.GetDimNum()) {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t expectedStride = 1;
    for (size_t i = shape.GetDimNum() - 1; i >= 1; --i) {
        const uint64_t actualStride = static_cast<uint64_t>(stride->GetStride(i));
        if (actualStride != expectedStride) {
            OP_LOGE(context->GetNodeName(),
                    "Tensor %s dim%zu is non-contiguous: actual stride=%llu, expected=%llu. "
                    "Only the first axis (dim0) may be non-contiguous for PAGED_BBND.",
                    tensorName, i,
                    static_cast<unsigned long long>(actualStride),
                    static_cast<unsigned long long>(expectedStride));
            return ge::GRAPH_FAILED;
        }
        expectedStride *= static_cast<uint64_t>(shape.GetDim(i));
        if (i == 1) {
            break;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GSATiling::ParseKvCacheStride0(gert::TilingContext *context)
{
    const uint64_t pageElems =
        static_cast<uint64_t>(blockSize_) * static_cast<uint64_t>(kvHeads_) *
        static_cast<uint64_t>(embeddingSize_);

    const gert::StorageShape *keyShape = context->GetInputShape(KEY_INDEX);
    const gert::StorageShape *valueShape = context->GetInputShape(VALUE_INDEX);
    OP_CHECK_IF(keyShape == nullptr || valueShape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention",
            "key/value shape is nullptr when parsing KV stride0."),
        return ge::GRAPH_FAILED);

    if (ValidatePagedBbndDim0OnlyNonContig(context, KEY_INDEX, keyShape->GetOriginShape(), "key") !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidatePagedBbndDim0OnlyNonContig(context, VALUE_INDEX, valueShape->GetOriginShape(), "value") !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto *keyStrides = context->GetRequiredInputStride(KEY_INDEX);
    kStride0_ = (keyStrides != nullptr && keyStrides->GetDimNum() > 0 && keyStrides->GetStride(0) > 0) ?
        static_cast<uint64_t>(keyStrides->GetStride(0)) : pageElems;
    auto *valueStrides = context->GetRequiredInputStride(VALUE_INDEX);
    vStride0_ = (valueStrides != nullptr && valueStrides->GetDimNum() > 0 && valueStrides->GetStride(0) > 0) ?
        static_cast<uint64_t>(valueStrides->GetStride(0)) : pageElems;

    const uint64_t rowElems =
        static_cast<uint64_t>(kvHeads_) * static_cast<uint64_t>(embeddingSize_);
    if (kStride0_ < pageElems || (rowElems > 0 && (kStride0_ % rowElems) != 0)) {
        OP_LOGE(context->GetNodeName(),
                "key dim0 stride (%llu) invalid for PAGED_BBND: expect >= pageElems=%llu and "
                "aligned to Nkv*D=%llu.",
                static_cast<unsigned long long>(kStride0_),
                static_cast<unsigned long long>(pageElems),
                static_cast<unsigned long long>(rowElems));
        return ge::GRAPH_FAILED;
    }
    if (vStride0_ < pageElems || (rowElems > 0 && (vStride0_ % rowElems) != 0)) {
        OP_LOGE(context->GetNodeName(),
                "value dim0 stride (%llu) invalid for PAGED_BBND: expect >= pageElems=%llu and "
                "aligned to Nkv*D=%llu.",
                static_cast<unsigned long long>(vStride0_),
                static_cast<unsigned long long>(pageElems),
                static_cast<unsigned long long>(rowElems));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GSATiling::ParseInputTensors(gert::TilingContext *context)
{
    const gert::StorageShape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_IF(queryShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention",
        "Query shape is nullptr."), return ge::GRAPH_FAILED);

    totalQTokens_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_T));
    numHeads_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_N));
    embeddingSize_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_D));

    const gert::StorageShape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_IF(keyShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention",
        "Key shape is nullptr."), return ge::GRAPH_FAILED);

    // ND PA cache: origin [blockNum, blockSize, Nkv, D]. Do not use storage shape
    // (dim0-strided view can collapse storage, same as scatter_pa / mla_preprocess).
    const gert::Shape &keyOrigin = keyShape->GetOriginShape();
    blockSize_ = static_cast<uint32_t>(keyOrigin.GetDim(BLOCKED_KV_DIM_BLOCK_SIZE));

    // TND + isPackedGQA=1: sparseBlockIdx 3D [N_kv, totalQBlocks, topK]
    const gert::StorageShape *sparseIdxShape = context->GetInputShape(SPARSE_BLOCK_IDX_INDEX);
    OP_CHECK_IF(sparseIdxShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention",
        "sparseBlockIdx shape is nullptr."), return ge::GRAPH_FAILED);

    if (sparseIdxShape->GetStorageShape().GetDimNum() != SPARSE_IDX_DIM_NUM) {
        OP_LOGE(context->GetNodeName(),
                "sparseBlockIdx must be 3D [N_kv, totalQBlocks, topK] for TND, but got %zu dims.",
                sparseIdxShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }

    kvHeads_ = static_cast<uint32_t>(sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_KV_HEAD));
    qBlockNum_ = static_cast<uint32_t>(sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_Q_BLOCK)); // totalQBlocks
    const int64_t sparseBlockCapacity =
        sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_KV_BLOCK);
    if (sparseBlockCapacity <= 0 ||
        static_cast<uint64_t>(sparseBlockCapacity) > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(context->GetNodeName(),
                "sparseBlockIdx.shape[-1]=%ld must fit a positive uint32 capacity.",
                sparseBlockCapacity);
        return ge::GRAPH_FAILED;
    }
    topK_ = static_cast<uint32_t>(sparseBlockCapacity);

    // sparseBlockCount 2D: [N_kv, totalQBlocks]
    const gert::StorageShape *sparseCountShape = context->GetInputShape(SPARSE_BLOCK_COUNT_INDEX);
    OP_CHECK_IF(sparseCountShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention",
        "sparseBlockCount shape is nullptr."), return ge::GRAPH_FAILED);

    if (sparseCountShape->GetStorageShape().GetDimNum() != SPARSE_COUNT_DIM_NUM) {
        OP_LOGE(context->GetNodeName(),
                "sparseBlockCount must be 2D [N_kv, totalQBlocks] for TND, but got %zu dims.",
                sparseCountShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }

    const uint32_t sparseCountKvHeads =
        static_cast<uint32_t>(sparseCountShape->GetStorageShape().GetDim(SPARSE_COUNT_DIM_KV_HEAD));
    const uint32_t sparseCountQBlocks =
        static_cast<uint32_t>(sparseCountShape->GetStorageShape().GetDim(SPARSE_COUNT_DIM_Q_BLOCK));
    if (sparseCountKvHeads != kvHeads_ || sparseCountQBlocks != qBlockNum_) {
        OP_LOGE(context->GetNodeName(),
                "sparseBlockCount shape [%u,%u] must match sparseBlockIdx [N_kv,totalQBlocks]=[%u,%u].",
                sparseCountKvHeads, sparseCountQBlocks, kvHeads_, qBlockNum_);
        return ge::GRAPH_FAILED;
    }

    // blockTable is OPTIONAL in OpDef — must use GetOptionalInputShape (GetInputShape always nullptr).
    const gert::StorageShape *blockTableShape = context->GetOptionalInputShape(BLOCK_TABLE_INDEX);
    if (blockTableShape != nullptr) {
        blockTablePresent_ = true;
        batch_ = static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(BLOCK_TABLE_DIM_BATCH));
        maxBlocksPerBatch_ = static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(BLOCK_TABLE_DIM_MAX_BLOCKS));
    } else {
        blockTablePresent_ = false;
        OP_LOGE(context->GetNodeName(),
                "Stage 1 requires blockTable for PAGED_BBND layout, but blockTableOptional is nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto queryDesc = context->GetInputDesc(QUERY_INDEX);
    if (queryDesc != nullptr) {
        dataType_ = queryDesc->GetDataType();
    }

    if (scaleValue_ < 1e-9f && scaleValue_ > -1e-9f && embeddingSize_ > 0) {
        scaleValue_ = 1.0f / std::sqrt(static_cast<float>(embeddingSize_));
    }
    if (socVer_ == SOC_VER_950_CODE) {
        if (CheckAttentionOutDtype(context) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    // maxQSeqlen: upper bound from totalQBlocks (packed across batch)
    maxQSeqlen_ = qBlockNum_ * blockShapeX_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GSATiling::CalculateWorkSpace(gert::TilingContext *context)
{
    uint64_t pipelineWorkspaceSize = 0;
    if (socVer_ != SOC_VER_950_CODE) {
        constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = 131072;
        constexpr uint32_t NUM3 = 3;
        // Identity reserved after S/P/O buffers (must match kernel layout).
        mm1OutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        smOnlineOutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(uint16_t) * NUM3;
        mm2OutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        updateSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        uint64_t identityIdxSize = static_cast<uint64_t>(topK_) * sizeof(int32_t);
        pipelineWorkspaceSize = mm1OutSize_ + smOnlineOutSize_ + mm2OutSize_ + updateSize_ + identityIdxSize;
    } else {
        uint32_t dtypeSize = (dataType_ == ge::DT_FLOAT8_E4M3FN) ? 1 : 2;
        uint64_t perTaskWorkspace = static_cast<uint64_t>(topK_) * blockShapeY_ * embeddingSize_ * dtypeSize * 2;
        uint64_t identityIdxSize = static_cast<uint64_t>(topK_) * sizeof(int32_t);
        pipelineWorkspaceSize = identityIdxSize + static_cast<uint64_t>(blockDim_) * perTaskWorkspace;
    }

    uint64_t userWorkspaceSize = pipelineWorkspaceSize;
    if (fdStaticEnabled_) {
        // Keep this Bmax formula in sync with the Metadata FD gate:
        // saTotalTaskNum * 10 < physicalAicNum * 3.
        const uint32_t maxNonEmptyBaseTaskNum = aicNum_ == 0U ? 0U :
            std::min(GSA_FD_MAX_COMBINE_TASK_NUM,
                (aicNum_ * GSA_FD_BASE_TASK_GATE_NUMERATOR - 1U) / GSA_FD_BASE_TASK_GATE_DENOMINATOR);
        const uint32_t maxActiveCoreNum = std::min(aicNum_, GSA_FD_MAX_ACTIVE_CORE_NUM);
        // The base-task intervals and active-core intervals are two continuous
        // partitions of the same flat task range, so their non-empty intersection
        // count is bounded by Bmax + Cmax - 1 instead of Bmax * Cmax.
        fdPartialCapacity_ = maxNonEmptyBaseTaskNum == 0U || maxActiveCoreNum == 0U ?
            0U : maxNonEmptyBaseTaskNum + maxActiveCoreNum - 1U;
        fdLseSubStride_ = ((groupSize_ + 1U) / 2U + 7U) / 8U * 8U;
        fdPartialLseOffset_ = AlignUp(pipelineWorkspaceSize, GSA_FD_WORKSPACE_ALIGNMENT);
        const uint64_t partialLseSize = static_cast<uint64_t>(fdPartialCapacity_) * 2U *
            fdLseSubStride_ * sizeof(float);
        fdPartialOOffset_ = AlignUp(fdPartialLseOffset_ + partialLseSize, GSA_FD_WORKSPACE_ALIGNMENT);
        const uint64_t partialOSize = static_cast<uint64_t>(fdPartialCapacity_) * groupSize_ *
            embeddingSize_ * sizeof(float);
        if (fdPartialOOffset_ > std::numeric_limits<uint64_t>::max() - partialOSize) {
            OP_LOGE(context->GetNodeName(), "Flash Decoding workspace size overflow.");
            return ge::GRAPH_FAILED;
        }
        userWorkspaceSize = fdPartialOOffset_ + partialOSize;
    }
    if (userWorkspaceSize > std::numeric_limits<size_t>::max() - libapiSize_) {
        OP_LOGE(context->GetNodeName(), "GenericBlockSparseAttention workspace size overflow.");
        return ge::GRAPH_FAILED;
    }
    workSpaceSize_ = libapiSize_ + userWorkspaceSize;

    context->SetBlockDim(blockDim_);
    size_t *workspaceArray = context->GetWorkspaceSizes(1);
    if (workspaceArray != nullptr) {
        workspaceArray[0] = static_cast<size_t>(workSpaceSize_);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GSATiling::CheckMetadata(gert::TilingContext *context)
{
    // Align SMLA / FlashAttn: metadata is required shell — INT32, 1D, fixed size.
    // Content (magic / schedule tables) is produced by AICPU and not re-validated here.
    const gert::StorageShape *metadataShape = context->GetOptionalInputShape(METADATA_INDEX);
    if (metadataShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "metadata must be provided.");
        return ge::GRAPH_FAILED;
    }
    if (metadataShape->GetStorageShape().GetDimNum() != 1) {
        OP_LOGE(context->GetNodeName(), "metadata dim num must be 1, but got %zu.",
                metadataShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    const int64_t metadataSize = metadataShape->GetStorageShape().GetDim(0);
    if (metadataSize != static_cast<int64_t>(GSA_METADATA_TOTAL_SIZE)) {
        OP_LOGE(context->GetNodeName(), "metadata dim 0 must be %u, but got %ld.",
                GSA_METADATA_TOTAL_SIZE, metadataSize);
        return ge::GRAPH_FAILED;
    }
    auto metadataDesc = context->GetOptionalInputDesc(METADATA_INDEX);
    if (metadataDesc == nullptr) {
        OP_LOGE(context->GetNodeName(), "metadata desc is nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (metadataDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE(context->GetNodeName(), "metadata dtype must be DT_INT32, but got %d.",
                static_cast<int32_t>(metadataDesc->GetDataType()));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GSATiling::ValidateSupportedConfig(gert::TilingContext *context)
{

    // Regular path: query=TND, kv=PAGED_BBND, maskType=1, blockShapeX=1, isPackedGQA=1
    // IMPORTANT: must fail hard — previously returned a FP16 tiling key on mismatch,
    // which can launch the wrong dtype kernel on bf16 inputs and destroy accuracy.
    if (layoutQ_ != "TND" || layoutKv_ != "PAGED_BBND" || maskType_ != 1 || blockShapeX_ != 1) {
        OP_LOGE(context->GetNodeName(),
                "Unsupported config: layoutQ=%s, layoutKv=%s, maskType=%ld, blockShapeX=%u. "
                "Regular path requires query=TND, kv=PAGED_BBND, maskType=1, blockShapeX=1.",
                layoutQ_.c_str(), layoutKv_.c_str(), maskType_, blockShapeX_);
        return ge::GRAPH_FAILED;
    }
    if (blockShapeY_ != 128 || embeddingSize_ != 128) {
        OP_LOGE(context->GetNodeName(),
                "Unsupported blockShapeY=%u or embeddingSize=%u, currently only D=128, blockShapeY=128.",
                blockShapeY_, embeddingSize_);
        return ge::GRAPH_FAILED;
    }
    if (softmaxPrecision_ != 0 && softmaxPrecision_ != 1) {
        OP_LOGE(context->GetNodeName(),
                "Unsupported softmaxPrecision=%u, only 0 (fp32 SM) or 1 (half/low SM) are supported.",
                softmaxPrecision_);
        return ge::GRAPH_FAILED;
    }
    if (socVer_ == SOC_VER_950_CODE) {
        // Align BSA 950: only low/mixed precision path is supported.
        if (softmaxPrecision_ != 1) {
            OP_LOGE(context->GetNodeName(),
                    "On chip 950, only softmaxPrecision=1 is supported, but got %u.",
                    softmaxPrecision_);
            return ge::GRAPH_FAILED;
        }
    } else if (dataType_ == ge::DT_BF16 && softmaxPrecision_ == 1) {
        // Align BSA arch22: bf16 + low-precision SM is unsupported.
        OP_LOGE(context->GetNodeName(),
                "On chip 910 & 910_93, when query dtype is bfloat16, "
                "only softmaxPrecision=0 is supported, but got %u.",
                softmaxPrecision_);
        return ge::GRAPH_FAILED;
    }
    // Full-quant contract: quantType=5 iff Q/K/V dtype is FLOAT8_E4M3FN.
    const bool isFp8 = (dataType_ == ge::DT_FLOAT8_E4M3FN);
    const bool isQuant5 = (quantType_ == 5);
    if (isFp8 != isQuant5) {
        OP_LOGE(context->GetNodeName(),
                "FP8 full-quant requires quantType=5 with FLOAT8_E4M3FN Q/K/V, "
                "got quantType=%ld dtype=%d.",
                quantType_, static_cast<int32_t>(dataType_));
        return ge::GRAPH_FAILED;
    }
    if (returnSoftmaxlse_ && isQuant5) {
        OP_LOGE(context->GetNodeName(),
                "returnSoftmaxlse=1 is not supported for FP8 full-quant path.");
        return ge::GRAPH_FAILED;
    }
    if (kvHeads_ == 0 || numHeads_ % kvHeads_ != 0) {
        OP_LOGE(context->GetNodeName(),
                "numHeads=%u must be divisible by kvHeads=%u (and kvHeads > 0).",
                numHeads_, kvHeads_);
        return ge::GRAPH_FAILED;
    }
    groupSize_ = numHeads_ / kvHeads_;
    if (groupSize_ == 0 || groupSize_ > 128) {
        OP_LOGE(context->GetNodeName(), "Unsupported GQA group size %u, expect [1, 128].", groupSize_);
        return ge::GRAPH_FAILED;
    }
    if (topK_ == 0U || topK_ > GSA_MAX_SPARSE_BLOCK_CAPACITY) {
        OP_LOGE(context->GetNodeName(), "Unsupported sparse block capacity=%u, expect [1, %u].",
                topK_, GSA_MAX_SPARSE_BLOCK_CAPACITY);
        return ge::GRAPH_FAILED;
    }
    // Runtime FD is selected by metadata. LSE output keeps using the normal path
    // until FD combine writes the public LSE tensor as part of its contract.
    fdStaticEnabled_ = topK_ >= 12U && !returnSoftmaxlse_;

    return ge::GRAPH_SUCCESS;
}

uint64_t GSATiling::GenerateTilingKey()
{
    // Axes: quant | arch(910B/950) | dtype | softmaxPrecision | LSE.
    // Unsupported combos (bf16+halfSM on 910B, FP8+LSE, 950+prec=0, quant/dtype mismatch)
    // are rejected in Validate; here quantType=5 implies FLOAT8_E4M3FN.
    const bool isFullQuant = (quantType_ == 5);
    const bool isArch35 = (socVer_ == SOC_VER_950_CODE);

    uint64_t key = 0;
    if (isFullQuant) {
        key = (attentionOutDtype_ == ge::DT_BF16) ? GSA_FP8_D128_BF16_TILING
                                                  : GSA_FP8_D128_TILING;
        return key;  // no LSE key for FP8
    }

    if (!isArch35) {
        // arch22: bf16 always float-SM; fp16 selects halfSM by softmaxPrecision.
        if (dataType_ == ge::DT_BF16) {
            key = GSA_BF16_D128_ARCH22_TILING;
        } else {
            key = (softmaxPrecision_ == 1) ? GSA_FP16_D128_ARCH22_HALFSM_TILING
                                           : GSA_FP16_D128_ARCH22_TILING;
        }
    } else {
        // arch35 regular: only low-prec path; key by dtype.
        key = (dataType_ == ge::DT_BF16) ? GSA_BF16_D128_TILING : GSA_FP16_D128_TILING;
    }

    if (returnSoftmaxlse_) {
        key += GSA_LSE_OUT_OFFSET;
    }
    return key;
}

ge::graphStatus GSATiling::FillTilingData(gert::TilingContext *context)
{
    tilingData_->set_batch(batch_);
    tilingData_->set_numHeads(numHeads_);
    tilingData_->set_kvHeads(kvHeads_);
    tilingData_->set_embeddingSize(embeddingSize_);
    tilingData_->set_blockShapeX(blockShapeX_);
    tilingData_->set_blockShapeY(blockShapeY_);
    tilingData_->set_blockSize(blockSize_);
    tilingData_->set_topK(topK_);
    tilingData_->set_qBlockNum(qBlockNum_);
    tilingData_->set_maxBlocksPerBatch(maxBlocksPerBatch_);
    tilingData_->set_totalQTokens(totalQTokens_);
    tilingData_->set_scaleValue(scaleValue_);
    tilingData_->set_softmaxPrecision(softmaxPrecision_);
    tilingData_->set_maxQSeqlen(maxQSeqlen_);
    tilingData_->set_mm1OutSize(mm1OutSize_);
    tilingData_->set_smOnlineOutSize(smOnlineOutSize_);
    tilingData_->set_mm2OutSize(mm2OutSize_);
    tilingData_->set_updateSize(updateSize_);
    tilingData_->set_workSpaceSize(workSpaceSize_);
    tilingData_->set_groupSize(groupSize_);
    uint64_t tilingKey = GenerateTilingKey();
    tilingData_->set_tilingKey(tilingKey);
    context->SetTilingKey(tilingKey);

    // BaseTileInfo
    uint32_t qBaseTile = (embeddingSize_ <= 128) ? 128 : 64;
    uint32_t kvBaseTile = blockShapeY_;
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
    tilingData_->set_kStride0(kStride0_);
    tilingData_->set_vStride0(vStride0_);
    tilingData_->set_fdStaticEnabled(fdStaticEnabled_ ? 1U : 0U);
    tilingData_->set_fdLseSubStride(fdLseSubStride_);
    tilingData_->set_fdPartialCapacity(fdPartialCapacity_);
    tilingData_->set_fdPartialLseOffset(fdPartialLseOffset_);
    tilingData_->set_fdPartialOOffset(fdPartialOOffset_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GSATiling::GetTiling(gert::TilingContext *context,
    GenericBlockSparseAttentionTilingData &tilingData)
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

    ret = ParseKvCacheStride0(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckMetadata(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ValidateSupportedConfig(context);
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

ge::graphStatus GSATiling::SetTilingData(gert::TilingContext *context,
    GenericBlockSparseAttentionTilingData &tilingData)
{
    OP_CHECK_IF(context->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention",
        "RawTilingData got from GE context is nullptr."), return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingGenericBlockSparseAttention(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention",
        "Context is nullptr."), return ge::GRAPH_FAILED);
    GenericBlockSparseAttentionTilingData tilingData;
    GSATiling tiling;
    if (tiling.GetTiling(context, tilingData) == ge::GRAPH_SUCCESS) {
        tiling.SetTilingData(context, tilingData);
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE(context->GetNodeName(), "GetTiling failed");
        return ge::GRAPH_FAILED;
    }
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForGenericBlockSparseAttention(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GenericBlockSparseAttention)
    .Tiling(TilingGenericBlockSparseAttention)
    .TilingParse<GenericBlockSparseAttentionCompileInfo>(TilingPrepareForGenericBlockSparseAttention);

}  // namespace optiling
