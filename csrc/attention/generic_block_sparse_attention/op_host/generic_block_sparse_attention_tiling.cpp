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
#include <initializer_list>
#include <string>
#include "log/log.h"
#include "err/ops_err.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef ASCENDC_EXTERN_C
#ifdef ASCENDC_OP_TEST
#define ASCENDC_EXTERN_C extern "C"
#else
#define ASCENDC_EXTERN_C
#endif
#endif

constexpr int QUERY_INDEX = 0;
constexpr int KEY_INDEX = 1;
constexpr int VALUE_INDEX = 2;
constexpr int SPARSE_BLOCK_IDX_INDEX = 3;
constexpr int SPARSE_BLOCK_COUNT_INDEX = 4;
constexpr int METADATA_INDEX = 5;
constexpr int ATTEN_MASK_INDEX = 6;
constexpr int Q_DEQUANT_SCALE_INDEX = 7;
constexpr int K_DEQUANT_SCALE_INDEX = 8;
constexpr int V_DEQUANT_SCALE_INDEX = 9;
constexpr int P_QUANT_SCALE_INDEX = 10;
constexpr int CU_SEQ_LENGTHS_Q_INDEX = 11;
constexpr int CU_SEQ_LENGTHS_KV_INDEX = 12;
constexpr int SEQUSED_Q_INDEX = 13;
constexpr int SEQUSED_KV_INDEX = 14;
constexpr int BLOCK_TABLE_INDEX = 15;

// Must match METADATA_TOTAL_SIZE in the AICPU / kernel metadata protocol.
constexpr uint32_t GBSA_METADATA_TOTAL_SIZE = 1024U;

constexpr int ATTENTION_OUT_INDEX = 0;

constexpr int TND_DIM_T = 0;
constexpr int TND_DIM_N = 1;
constexpr int TND_DIM_D = 2;

constexpr int BLOCKED_KV_DIM_BLOCK_NUM = 0;
constexpr int BLOCKED_KV_DIM_BLOCK_SIZE = 1;
constexpr int BLOCKED_KV_DIM_KV_HEAD = 2;
constexpr int BLOCKED_KV_DIM_D = 3;

// layout_sparse_pattern=KVN_TotalQB_KB: sparseBlockIdx 3D [N_kv, totalQBlocks, topK]
constexpr int SPARSE_IDX_DIM_KV_HEAD = 0;
constexpr int SPARSE_IDX_DIM_Q_BLOCK = 1;
constexpr int SPARSE_IDX_DIM_KV_BLOCK = 2;
constexpr int SPARSE_IDX_DIM_NUM = 3;

// layout_sparse_pattern=KVN_TotalQB_KB: sparseBlockCount 2D [N_kv, totalQBlocks]
constexpr int SPARSE_COUNT_DIM_KV_HEAD = 0;
constexpr int SPARSE_COUNT_DIM_Q_BLOCK = 1;
constexpr int SPARSE_COUNT_DIM_NUM = 2;

constexpr int BLOCK_TABLE_DIM_BATCH = 0;
constexpr int BLOCK_TABLE_DIM_MAX_BLOCKS = 1;

constexpr int ATTR_BLOCK_SHAPE_INDEX = 0;
constexpr int ATTR_Q_INPUT_LAYOUT_INDEX = 1;
constexpr int ATTR_KV_INPUT_LAYOUT_INDEX = 2;
constexpr int ATTR_LAYOUT_SPARSE_PATTERN_INDEX = 3;
constexpr int ATTR_SCALE_VALUE_INDEX = 4;
constexpr int ATTR_MASK_TYPE_INDEX = 5;
constexpr int ATTR_QUANT_TYPE_INDEX = 6;
constexpr int ATTR_DST_TYPE_MAX_INDEX = 7;
constexpr int ATTR_SOFTMAX_PRECISION_INDEX = 8;
constexpr int ATTR_WIN_LEFT_INDEX = 9;
constexpr int ATTR_WIN_RIGHT_INDEX = 10;
constexpr int ATTR_SOFTMAX_LSE_FLAG_INDEX = 11;
constexpr int ATTR_RESIDUAL_BLOCK_MODE_INDEX = 12;
constexpr int ATTR_IS_CONSISTENT_TOPK_INDEX = 13;

constexpr int64_t GBSA_RESIDUAL_BLOCK_MODE_NONE = 0;
constexpr int64_t GBSA_RESIDUAL_BLOCK_MODE_FORCE = 1;
constexpr uint64_t GBSA_RESIDUAL_BLOCK_MODE_TILING_OFFSET = 1000000000ULL;
constexpr int64_t GBSA_LAYOUT_SPARSE_KVN_TOTALQB_KB = 4;

constexpr uint32_t SOC_VER_950_CODE = 4;
constexpr uint32_t GBSA_MAX_GROUP_SIZE = 128U;
constexpr uint32_t GBSA_TILE_SIZE_16 = 16U;
constexpr uint32_t GBSA_TILE_SIZE_32 = 32U;
constexpr uint32_t GBSA_TILE_SIZE_64 = 64U;
constexpr uint32_t GBSA_TILE_SIZE_128 = 128U;
constexpr uint32_t GBSA_TILE_SIZE_256 = 256U;
constexpr uint32_t GBSA_TILE_SIZE_384 = 384U;
constexpr uint32_t GBSA_TILE_SIZE_512 = 512U;
constexpr int64_t GBSA_QUANT_TYPE_NONE = 0;
constexpr int64_t GBSA_QUANT_TYPE_FULL = 5;
constexpr int64_t GBSA_WIN_DISABLED = -1;
constexpr float GBSA_DST_TYPE_MAX_DISABLED = 0.0f;
constexpr uint32_t GSA_FD_MAX_ACTIVE_CORE_NUM = 32U;        // Maximum AIC cores used by FD.
constexpr uint32_t GSA_FD_MAX_COMBINE_TASK_NUM = 32U;       // Maximum FD combine tasks.
constexpr uint32_t GSA_FD_BASE_TASK_GATE_NUMERATOR = 3U;    // Numerator of the FD base-task gate ratio.
constexpr uint32_t GSA_FD_BASE_TASK_GATE_DENOMINATOR = 10U; // Denominator of the FD base-task gate ratio.
constexpr uint64_t GSA_FD_WORKSPACE_ALIGNMENT = 512U;       // FD workspace alignment in bytes.

namespace {
uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}
} // namespace

namespace optiling {

ge::graphStatus GBSATiling::GetNpuInfo(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    if (aicNum_ == 0) {
        OP_LOGE(context->GetNodeName(), "GetCoreNumAic returned 0.");
        return ge::GRAPH_FAILED;
    }
    // Task schedule is owned by AICPU metadata (saTotalTaskNum). Host only launches
    // all AIC cores; idle cores exit when taskIdx >= metadata saTotalTaskNum.
    blockDim_ = aicNum_;
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    socVer_ = static_cast<uint32_t>(ascendcPlatform.GetSocVersion());
    return ge::GRAPH_SUCCESS;
}

namespace {

ge::graphStatus ParseBlockShapeAttr(gert::TilingContext *context,
                                    const gert::TypedContinuousVector<int64_t> *blockShapeArr, uint32_t &blockShapeX,
                                    uint32_t &blockShapeY)
{
    if (blockShapeArr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (blockShapeArr->GetSize() != 2) {
        OP_LOGE(context->GetNodeName(), "blockShape must contain two elements [x, y], got size %zu.",
                blockShapeArr->GetSize());
        return ge::GRAPH_FAILED;
    }
    const int64_t *data = blockShapeArr->GetData();
    if (data == nullptr) {
        OP_LOGE(context->GetNodeName(), "blockShape data is null.");
        return ge::GRAPH_FAILED;
    }
    if (data[0] != 1 || data[1] <= 0 || data[1] > 128) {
        OP_LOGE(context->GetNodeName(), "blockShape requires X=1 and 1<=Y<=128, got [%ld, %ld].", data[0], data[1]);
        return ge::GRAPH_FAILED;
    }
    blockShapeX = static_cast<uint32_t>(data[0]);
    blockShapeY = static_cast<uint32_t>(data[1]);
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus GBSATiling::ParseCapabilityAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetAttrs returned nullptr.");
        return ge::GRAPH_FAILED;
    }

    const int64_t *softmaxPrecPtr = attrs->GetInt(ATTR_SOFTMAX_PRECISION_INDEX);
    if (softmaxPrecPtr != nullptr) {
        softmaxPrecision_ = static_cast<uint32_t>(*softmaxPrecPtr);
    }
    if (softmaxPrecision_ != 0 && softmaxPrecision_ != 1) {
        OP_LOGE(context->GetNodeName(),
                "Unsupported softmaxPrecision=%u, only 0 (fp32 SM) or 1 (half/low SM) are supported.",
                softmaxPrecision_);
        return ge::GRAPH_FAILED;
    }

    const int64_t *maskTypePtr = attrs->GetInt(ATTR_MASK_TYPE_INDEX);
    if (maskTypePtr != nullptr) {
        maskType_ = *maskTypePtr;
    }
    if (maskType_ != 1) {
        OP_LOGE(context->GetNodeName(), "Unsupported maskType=%ld, only maskType=1 is supported.", maskType_);
        return ge::GRAPH_FAILED;
    }

    const int64_t *quantTypePtr = attrs->GetInt(ATTR_QUANT_TYPE_INDEX);
    if (quantTypePtr != nullptr) {
        quantType_ = *quantTypePtr;
    }
    if (quantType_ != GBSA_QUANT_TYPE_NONE && quantType_ != GBSA_QUANT_TYPE_FULL) {
        OP_LOGE(context->GetNodeName(), "Unsupported quantType=%ld, only 0 (none) or 5 (full-quant) are supported.",
                quantType_);
        return ge::GRAPH_FAILED;
    }

    const int64_t *layoutSparsePtr = attrs->GetInt(ATTR_LAYOUT_SPARSE_PATTERN_INDEX);
    if (layoutSparsePtr != nullptr) {
        layoutSparsePattern_ = *layoutSparsePtr;
    }
    if (layoutSparsePattern_ != GBSA_LAYOUT_SPARSE_KVN_TOTALQB_KB) {
        OP_LOGE(context->GetNodeName(), "Unsupported layout_sparse_pattern=%ld, only 4 (KVN_TotalQB_KB) is supported.",
                layoutSparsePattern_);
        return ge::GRAPH_FAILED;
    }

    const int64_t *residualModePtr = attrs->GetInt(ATTR_RESIDUAL_BLOCK_MODE_INDEX);
    if (residualModePtr != nullptr) {
        residualBlockMode_ = *residualModePtr;
    }
    const bool residualModeSupported = residualBlockMode_ == GBSA_RESIDUAL_BLOCK_MODE_NONE ||
        (residualBlockMode_ == GBSA_RESIDUAL_BLOCK_MODE_FORCE && socVer_ == SOC_VER_950_CODE &&
         quantType_ == GBSA_QUANT_TYPE_NONE);
    if (!residualModeSupported) {
        OP_LOGE(context->GetNodeName(),
                "Unsupported residual_block_mode=%ld; mode 1 only supports A5 non-quant currently.",
                residualBlockMode_);
        return ge::GRAPH_FAILED;
    }

    const bool *consistentTopkPtr = attrs->GetBool(ATTR_IS_CONSISTENT_TOPK_INDEX);
    if (consistentTopkPtr != nullptr) {
        isConsistentTopk_ = *consistentTopkPtr;
    }

    const int64_t *lseFlagPtr = attrs->GetInt(ATTR_SOFTMAX_LSE_FLAG_INDEX);
    if (lseFlagPtr != nullptr) {
        softmaxLseFlag_ = *lseFlagPtr;
    }
    if (softmaxLseFlag_ != 0) {
        OP_LOGE(context->GetNodeName(), "Unsupported returnSoftmaxlse=%ld, only 0 is supported.", softmaxLseFlag_);
        return ge::GRAPH_FAILED;
    }
    returnSoftmaxlse_ = false;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckReservedAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetAttrs returned nullptr.");
        return ge::GRAPH_FAILED;
    }

    const float *dstTypeMaxPtr = attrs->GetFloat(ATTR_DST_TYPE_MAX_INDEX);
    if (dstTypeMaxPtr != nullptr && *dstTypeMaxPtr != GBSA_DST_TYPE_MAX_DISABLED) {
        OP_LOGE(context->GetNodeName(), "Since dst_type_max is not yet supported, it must be 0, but got %f.",
                *dstTypeMaxPtr);
        return ge::GRAPH_FAILED;
    }
    const int64_t *winLeftPtr = attrs->GetInt(ATTR_WIN_LEFT_INDEX);
    const int64_t *winRightPtr = attrs->GetInt(ATTR_WIN_RIGHT_INDEX);
    const int64_t winLeft = (winLeftPtr != nullptr) ? *winLeftPtr : GBSA_WIN_DISABLED;
    const int64_t winRight = (winRightPtr != nullptr) ? *winRightPtr : GBSA_WIN_DISABLED;
    if (winLeft != GBSA_WIN_DISABLED || winRight != GBSA_WIN_DISABLED) {
        OP_LOGE(context->GetNodeName(),
                "Since windowed atten mask is not yet supported, "
                "win_left & win_right must be -1, but got %ld, %ld.",
                winLeft, winRight);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetAttrs returned nullptr.");
        return ge::GRAPH_FAILED;
    }

    const float *scalePtr = attrs->GetFloat(ATTR_SCALE_VALUE_INDEX);
    if (scalePtr != nullptr) {
        scaleValue_ = *scalePtr;
    }

    if (ParseBlockShapeAttr(context, attrs->GetListInt(ATTR_BLOCK_SHAPE_INDEX), blockShapeX_, blockShapeY_) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseCapabilityAttrs(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckReservedAttrs(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::GetInputLayout(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetAttrs returned nullptr.");
        return ge::GRAPH_FAILED;
    }

    const char *layoutQPtr = attrs->GetStr(ATTR_Q_INPUT_LAYOUT_INDEX);
    if (layoutQPtr != nullptr) {
        layoutQ_ = std::string(layoutQPtr);
    }
    const char *layoutKvPtr = attrs->GetStr(ATTR_KV_INPUT_LAYOUT_INDEX);
    if (layoutKvPtr != nullptr) {
        layoutKv_ = std::string(layoutKvPtr);
    }

    if (layoutQ_ != "TND") {
        OP_LOGE(context->GetNodeName(), "layoutQ only supports TND, got %s.", layoutQ_.c_str());
        return ge::GRAPH_FAILED;
    }
    if (layoutKv_ != "PA_BBND" && layoutKv_ != "TND") {
        OP_LOGE(context->GetNodeName(), "layoutKv only supports PA_BBND or TND, got %s.", layoutKv_.c_str());
        return ge::GRAPH_FAILED;
    }
    // Determine paging before parsing shapes, strides or sequence-input requirements.
    isPaged_ = context->GetOptionalInputShape(BLOCK_TABLE_INDEX) != nullptr;
    if ((isPaged_ && layoutKv_ != "PA_BBND") || (!isPaged_ && layoutKv_ != "TND")) {
        OP_LOGE(context->GetNodeName(),
                "KV layout must be PA_BBND with blockTable, or TND without blockTable.");
        return ge::GRAPH_FAILED;
    }
    if (!isPaged_ && socVer_ != SOC_VER_950_CODE) {
        OP_LOGE(context->GetNodeName(), "TND non-paged KV is only supported on chip 950.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckAttentionOutDtype(gert::TilingContext *context)
{
    if (dataType_ == ge::DT_FLOAT8_E4M3FN) {
        auto outDesc = context->GetOutputDesc(ATTENTION_OUT_INDEX);
        if (outDesc == nullptr) {
            OP_LOGE(context->GetNodeName(), "attentionOut desc is nullptr.");
            return ge::GRAPH_FAILED;
        }
        attentionOutDtype_ = outDesc->GetDataType();
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

// Validate PA_BBND key/value: only dim0 may be non-contiguous.
static ge::graphStatus ValidatePagedBbndDim0OnlyNonContig(gert::TilingContext *context, uint64_t inputIndex,
                                                          const gert::Shape &shape, const char *tensorName)
{
    auto *stride = context->GetRequiredInputStride(inputIndex);
    if (stride == nullptr || stride->GetDimNum() != shape.GetDimNum()) {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t expectedStride = 1;
    for (size_t i = shape.GetDimNum() - 1; i >= 1; --i) {
        const int64_t dimSize = shape.GetDim(i);
        // size<=1 axes are not used in addressing; PyTorch often leaves non-canonical strides
        // (e.g. Nkv=1). Skip so dim0-strided KV is not falsely rejected.
        if (dimSize <= 1) {
            continue;
        }
        const uint64_t actualStride = static_cast<uint64_t>(stride->GetStride(i));
        if (actualStride != expectedStride) {
            OP_LOGE(context->GetNodeName(),
                    "Tensor %s dim%zu is non-contiguous: actual stride=%llu, expected=%llu. "
                    "Only the first axis (dim0) may be non-contiguous for PA_BBND.",
                    tensorName, i, static_cast<unsigned long long>(actualStride),
                    static_cast<unsigned long long>(expectedStride));
            return ge::GRAPH_FAILED;
        }
        expectedStride *= static_cast<uint64_t>(dimSize);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseKvCacheStride0(gert::TilingContext *context)
{
    if (!isPaged_) {
        for (const int index : {QUERY_INDEX, KEY_INDEX, VALUE_INDEX}) {
            const auto &shape = context->GetInputShape(index)->GetStorageShape();
            if (shape.GetDimNum() != 3) {
                OP_LOGE(context->GetNodeName(), "Dense TND Q/K/V storage shape must be rank 3, inputIndex=%d, "
                                                "storageRank=%zu.",
                        index, shape.GetDimNum());
                return ge::GRAPH_FAILED;
            }
            const auto *stride = context->GetRequiredInputStride(index);
            if (stride == nullptr || stride->GetDimNum() == 0) {
                continue; // No explicit stride: infer a contiguous layout from storage shape.
            }
            if (stride->GetDimNum() != shape.GetDimNum()) {
                OP_LOGE(context->GetNodeName(),
                        "Dense TND Q/K/V explicit stride rank must equal storage shape rank, inputIndex=%d, "
                        "storageRank=%zu, strideRank=%zu.",
                        index, shape.GetDimNum(), stride->GetDimNum());
                return ge::GRAPH_FAILED;
            }
            int64_t expected = 1;
            for (int axis = 2; axis >= 0; --axis) {
                if (shape.GetDim(axis) > 1 && stride->GetStride(axis) != expected) {
                    OP_LOGE(context->GetNodeName(), "Dense TND Q/K/V must be contiguous.");
                    return ge::GRAPH_FAILED;
                }
                expected *= shape.GetDim(axis);
            }
        }
        kStride0_ = 0;
        vStride0_ = 0;
        return ge::GRAPH_SUCCESS;
    }

    const uint64_t pageElems =
        static_cast<uint64_t>(blockSize_) * static_cast<uint64_t>(kvHeads_) * static_cast<uint64_t>(embeddingSize_);

    const gert::StorageShape *keyShape = context->GetInputShape(KEY_INDEX);
    const gert::StorageShape *valueShape = context->GetInputShape(VALUE_INDEX);
    if (keyShape == nullptr || valueShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "key/value shape is nullptr when parsing KV stride0.");
        return ge::GRAPH_FAILED;
    }

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
                    static_cast<uint64_t>(keyStrides->GetStride(0)) :
                    pageElems;
    auto *valueStrides = context->GetRequiredInputStride(VALUE_INDEX);
    vStride0_ = (valueStrides != nullptr && valueStrides->GetDimNum() > 0 && valueStrides->GetStride(0) > 0) ?
                    static_cast<uint64_t>(valueStrides->GetStride(0)) :
                    pageElems;

    const uint64_t rowElems = static_cast<uint64_t>(kvHeads_) * static_cast<uint64_t>(embeddingSize_);
    if (kStride0_ < pageElems || (rowElems > 0 && (kStride0_ % rowElems) != 0)) {
        OP_LOGE(context->GetNodeName(),
                "key dim0 stride (%llu) invalid for PA_BBND: expect >= pageElems=%llu and "
                "aligned to Nkv*D=%llu.",
                static_cast<unsigned long long>(kStride0_), static_cast<unsigned long long>(pageElems),
                static_cast<unsigned long long>(rowElems));
        return ge::GRAPH_FAILED;
    }
    if (vStride0_ < pageElems || (rowElems > 0 && (vStride0_ % rowElems) != 0)) {
        OP_LOGE(context->GetNodeName(),
                "value dim0 stride (%llu) invalid for PA_BBND: expect >= pageElems=%llu and "
                "aligned to Nkv*D=%llu.",
                static_cast<unsigned long long>(vStride0_), static_cast<unsigned long long>(pageElems),
                static_cast<unsigned long long>(rowElems));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseQueryKeyShapes(gert::TilingContext *context)
{
    const auto *queryShape = context->GetInputShape(QUERY_INDEX);
    const auto *keyShape = context->GetInputShape(KEY_INDEX);
    const auto *valueShape = context->GetInputShape(VALUE_INDEX);
    if (queryShape == nullptr || keyShape == nullptr || valueShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "Q/K/V shapes are required.");
        return ge::GRAPH_FAILED;
    }
    const auto &q = queryShape->GetStorageShape();
    const auto &k = keyShape->GetOriginShape();
    const auto &v = valueShape->GetOriginShape();
    const bool isPaged = isPaged_;
    const size_t kvRank = isPaged ? 4U : 3U;
    const size_t headAxis = isPaged ? 2U : 1U;
    const size_t dAxis = isPaged ? 3U : 2U;
    if (q.GetDimNum() != 3U || k.GetDimNum() != kvRank || v.GetDimNum() != kvRank) {
        OP_LOGE(context->GetNodeName(), "Invalid Q/K/V ranks for the selected layouts.");
        return ge::GRAPH_FAILED;
    }
    if (q.GetDim(0) < 0 || q.GetDim(0) > std::numeric_limits<int32_t>::max() ||
        q.GetDim(1) <= 0 || q.GetDim(1) > std::numeric_limits<int32_t>::max() || q.GetDim(2) <= 0 ||
        q.GetDim(2) > GBSA_TILE_SIZE_512) {
        OP_LOGE(context->GetNodeName(),
                "TND Q requires nonnegative T within INT32, positive heads and 1<=D<=512.");
        return ge::GRAPH_FAILED;
    }
    for (size_t axis = 0; axis < kvRank; ++axis) {
        if (k.GetDim(axis) < 0 || k.GetDim(axis) > std::numeric_limits<int32_t>::max() ||
            k.GetDim(axis) != v.GetDim(axis)) {
            OP_LOGE(context->GetNodeName(), "K/V shapes must match and fit the supported index range.");
            return ge::GRAPH_FAILED;
        }
    }
    numHeads_ = static_cast<uint32_t>(q.GetDim(1));
    embeddingSize_ = static_cast<uint32_t>(q.GetDim(2));
    kvHeads_ = static_cast<uint32_t>(k.GetDim(headAxis));
    if (kvHeads_ == 0U || numHeads_ % kvHeads_ != 0U || k.GetDim(dAxis) != embeddingSize_) {
        OP_LOGE(context->GetNodeName(), "Invalid Q/K/V head counts or head dimensions.");
        return ge::GRAPH_FAILED;
    }
    totalQTokens_ = static_cast<uint64_t>(q.GetDim(0));
    totalKvTokens_ = static_cast<uint64_t>(k.GetDim(0));
    blockSize_ = 0U;
    if (isPaged) {
        blockSize_ = static_cast<uint32_t>(k.GetDim(1));
        if (blockSize_ != 128U) {
            OP_LOGE(context->GetNodeName(), "PA page size must be 128, independently of blockShapeY.");
            return ge::GRAPH_FAILED;
        }
        totalKvTokens_ *= blockSize_;
    }
    if (totalKvTokens_ > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / kvHeads_ / embeddingSize_) {
        OP_LOGE(context->GetNodeName(), "KV storage size exceeds the supported address range.");
        return ge::GRAPH_FAILED;
    }
    if (totalQTokens_ * kvHeads_ > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        OP_LOGE(context->GetNodeName(), "Q task capacity exceeds the metadata index range.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseSparseTensors(gert::TilingContext *context)
{
    // layout_sparse_pattern=KVN_TotalQB_KB: sparseBlockIdx 3D [N_kv, totalQBlocks, topK]
    const gert::StorageShape *sparseIdxShape = context->GetInputShape(SPARSE_BLOCK_IDX_INDEX);
    if (sparseIdxShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "sparseBlockIdx shape is nullptr.");
        return ge::GRAPH_FAILED;
    }

    if (sparseIdxShape->GetStorageShape().GetDimNum() != SPARSE_IDX_DIM_NUM) {
        OP_LOGE(context->GetNodeName(),
                "sparseBlockIdx must be 3D [N_kv, totalQBlocks, topK] for TND, but got %zu dims.",
                sparseIdxShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }

    if (sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_KV_HEAD) != kvHeads_) {
        OP_LOGE(context->GetNodeName(), "Sparse head count must equal the K/V head count.");
        return ge::GRAPH_FAILED;
    }
    const int64_t topK = sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_KV_BLOCK);
    qBlockNum_ =
        static_cast<uint32_t>(sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_Q_BLOCK)); // totalQBlocks
    if (kvHeads_ == 0 || numHeads_ % kvHeads_ != 0) {
        OP_LOGE(context->GetNodeName(), "numHeads=%u must be divisible by kvHeads=%u (and kvHeads > 0).", numHeads_,
                kvHeads_);
        return ge::GRAPH_FAILED;
    }
    groupSize_ = numHeads_ / kvHeads_;
    if (groupSize_ > GBSA_MAX_GROUP_SIZE) {
        OP_LOGE(context->GetNodeName(),
                "Unsupported groupSize=%u (numHeads=%u, kvHeads=%u), currently only groupSize <= %u is supported to "
                "avoid kernel L0C/UB overflow.",
                groupSize_, numHeads_, kvHeads_, GBSA_MAX_GROUP_SIZE);
        return ge::GRAPH_FAILED;
    }
    if (sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_Q_BLOCK) !=
            static_cast<int64_t>(totalQTokens_) ||
        topK <= 0) {
        OP_LOGE(context->GetNodeName(), "For blockShapeX=1, totalQBlocks must equal Tq and topK must be positive.");
        return ge::GRAPH_FAILED;
    }
    constexpr int64_t GBSA_LEGACY_MAX_TOPK = 256;
    const int64_t maxTopK = socVer_ == SOC_VER_950_CODE && quantType_ == GBSA_QUANT_TYPE_NONE ?
                                std::numeric_limits<int32_t>::max() : GBSA_LEGACY_MAX_TOPK;
    if (topK > maxTopK) {
        OP_LOGE(context->GetNodeName(), "Unsupported topK=%lld, maximum is %lld for this configuration.", topK,
                maxTopK);
        return ge::GRAPH_FAILED;
    }
    topK_ = static_cast<uint32_t>(topK);

    // sparseBlockCount 2D: [N_kv, totalQBlocks]
    const gert::StorageShape *sparseCountShape = context->GetInputShape(SPARSE_BLOCK_COUNT_INDEX);
    if (sparseCountShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "sparseBlockCount shape is nullptr.");
        return ge::GRAPH_FAILED;
    }

    if (sparseCountShape->GetStorageShape().GetDimNum() != SPARSE_COUNT_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "sparseBlockCount must be 2D [N_kv, totalQBlocks] for TND, but got %zu dims.",
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
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseBlockTable(gert::TilingContext *context)
{
    const auto *cuQ = context->GetOptionalInputShape(CU_SEQ_LENGTHS_Q_INDEX);
    if (cuQ == nullptr || cuQ->GetStorageShape().GetDimNum() != 1U ||
        cuQ->GetStorageShape().GetDim(0) < 1 ||
        cuQ->GetStorageShape().GetDim(0) - 1 > std::numeric_limits<int32_t>::max()) {
        OP_LOGE(context->GetNodeName(), "TND Q requires cuSeqLengthsQ[B+1].");
        return ge::GRAPH_FAILED;
    }
    batch_ = static_cast<uint32_t>(cuQ->GetStorageShape().GetDim(0) - 1);
    const auto *blockTableShape = context->GetOptionalInputShape(BLOCK_TABLE_INDEX);
    maxBlocksPerBatch_ = 0U;
    if (!isPaged_) {
        return ge::GRAPH_SUCCESS;
    }
    if (blockTableShape == nullptr || blockTableShape->GetStorageShape().GetDimNum() != 2U ||
        blockTableShape->GetStorageShape().GetDim(0) != batch_ ||
        blockTableShape->GetStorageShape().GetDim(1) <= 0 ||
        blockTableShape->GetStorageShape().GetDim(1) > std::numeric_limits<int32_t>::max()) {
        OP_LOGE(context->GetNodeName(), "PA_BBND requires blockTable[B,maxBlocksPerBatch].");
        return ge::GRAPH_FAILED;
    }
    const auto *desc = context->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
    if (desc == nullptr || desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE(context->GetNodeName(), "blockTable must be INT32.");
        return ge::GRAPH_FAILED;
    }
    maxBlocksPerBatch_ = static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(1));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseQkvDtype(gert::TilingContext *context)
{
    auto queryDesc = context->GetInputDesc(QUERY_INDEX);
    if (queryDesc != nullptr) {
        dataType_ = queryDesc->GetDataType();
    }
    auto keyDesc = context->GetInputDesc(KEY_INDEX);
    auto valueDesc = context->GetInputDesc(VALUE_INDEX);
    if (keyDesc == nullptr || valueDesc == nullptr) {
        OP_LOGE(context->GetNodeName(), "key/value desc is nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (keyDesc->GetDataType() != dataType_ || valueDesc->GetDataType() != dataType_) {
        OP_LOGE(context->GetNodeName(), "query/key/value dtypes must match, got query=%d key=%d value=%d.",
                static_cast<int32_t>(dataType_), static_cast<int32_t>(keyDesc->GetDataType()),
                static_cast<int32_t>(valueDesc->GetDataType()));
        return ge::GRAPH_FAILED;
    }
    if (dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16 && dataType_ != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE(context->GetNodeName(), "Unsupported query/key/value dtype=%d, only float16/bfloat16/float8_e4m3fn.",
                static_cast<int32_t>(dataType_));
        return ge::GRAPH_FAILED;
    }

    if (scaleValue_ == 0.0f && embeddingSize_ > 0) {
        scaleValue_ = 1.0f / std::sqrt(static_cast<float>(embeddingSize_));
    }
    if (CheckAttentionOutDtype(context) != ge::GRAPH_SUCCESS || CheckSoftmaxPrecision(context) != ge::GRAPH_SUCCESS ||
        CheckQuantConfig(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseInputTensors(gert::TilingContext *context)
{
    if (ParseQueryKeyShapes(context) != ge::GRAPH_SUCCESS || ParseSparseTensors(context) != ge::GRAPH_SUCCESS ||
        ParseBlockTable(context) != ge::GRAPH_SUCCESS || ParseQkvDtype(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CalculateWorkSpace(gert::TilingContext *context)
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
        pipelineWorkspaceSize = static_cast<uint64_t>(topK_) * sizeof(int32_t);
        if (quantType_ == 0U && embeddingSize_ > GBSA_TILE_SIZE_256 && qBaseTile_ > GBSA_TILE_SIZE_64) {
            // Two complete FP32 LO tiles per physical AIC; FD workspace follows this region.
            mm2OutSize_ = static_cast<uint64_t>(blockDim_) * 2U * qBaseTile_ *
                          AlignUp(embeddingSize_, 16U) * sizeof(float);
            pipelineWorkspaceSize = std::max(pipelineWorkspaceSize, mm2OutSize_);
        }
    }

    // Four per-AIC slots. Each L1-N group has a 64B header and
    // PA entries hold {K offset, V offset, rows}; dense entries hold
    // {offset, rows, ndNum, source stride}, paired by AIV before copying to GM.
    if (socVer_ == SOC_VER_950_CODE && quantType_ == 0U) {
        const uint64_t sparseBlocks = (mm1L1TileN_ + 2ULL * blockShapeY_ - 2U) / blockShapeY_;
        const uint64_t copyBlocks = isPaged_ ? std::min<uint64_t>(mm1L1TileN_, sparseBlocks * 2U) : sparseBlocks;
        const uint64_t recordWords = isPaged_ ? 3U : 4U;
        const uint64_t groupBytes = AlignUp(64U + copyBlocks * recordWords * sizeof(uint64_t), 64U);
        const uint64_t tileWords = ((kvBaseTile_ + mm1L1TileN_ - 1U) / mm1L1TileN_) *
                                   groupBytes / sizeof(uint64_t);
        const uint64_t offsetBytes = static_cast<uint64_t>(blockDim_) * 4U * tileWords * sizeof(uint64_t);
        const uint64_t offset = AlignUp(pipelineWorkspaceSize, GSA_FD_WORKSPACE_ALIGNMENT);
        if (tileWords > std::numeric_limits<uint32_t>::max() ||
            offset > std::numeric_limits<uint64_t>::max() - offsetBytes) {
            OP_LOGE(context->GetNodeName(), "KV offset workspace size overflow.");
            return ge::GRAPH_FAILED;
        }
        tilingData_->set_kvOffsetInfoOffset(offset);
        tilingData_->set_kvOffsetTileWords(static_cast<uint32_t>(tileWords));
        pipelineWorkspaceSize = offset + offsetBytes;
    }
    uint64_t userWorkspaceSize = pipelineWorkspaceSize;
    if (fdStaticEnabled_) {
        // saTotalTaskNum * 10 < physicalAicNum * 3.
        const uint32_t maxNonEmptyBaseTaskNum =
            aicNum_ == 0U ? 0U :
                            std::min(GSA_FD_MAX_COMBINE_TASK_NUM, (aicNum_ * GSA_FD_BASE_TASK_GATE_NUMERATOR - 1U) /
                                                                      GSA_FD_BASE_TASK_GATE_DENOMINATOR);
        const uint32_t maxActiveCoreNum = std::min(aicNum_, GSA_FD_MAX_ACTIVE_CORE_NUM);
        // The base-task intervals and active-core intervals are two continuous
        // partitions of the same flat task range, count is bounded by Bmax + Cmax - 1.
        fdPartialCapacity_ = maxNonEmptyBaseTaskNum == 0U || maxActiveCoreNum == 0U ?
                                 0U :
                                 maxNonEmptyBaseTaskNum + maxActiveCoreNum - 1U;
        // Each FD partial task is split across two Vector sub-blocks. Reserve the larger half of groupSize and
        // align each sub-block's FP32 LSE slice to 8 elements (32 bytes) for GM DataCopy alignment.
        fdLseSubStride_ = ((groupSize_ + 1U) / 2U + 7U) / 8U * 8U;
        fdPartialLseOffset_ = AlignUp(pipelineWorkspaceSize, GSA_FD_WORKSPACE_ALIGNMENT);
        const uint64_t partialLseSize =
            static_cast<uint64_t>(fdPartialCapacity_) * 2U * fdLseSubStride_ * sizeof(float);
        fdPartialOOffset_ = AlignUp(fdPartialLseOffset_ + partialLseSize, GSA_FD_WORKSPACE_ALIGNMENT);
        const uint64_t fdPartialOStride = AlignUp(embeddingSize_, 16U);
        const uint64_t partialOSize =
            static_cast<uint64_t>(fdPartialCapacity_) * groupSize_ * fdPartialOStride * sizeof(float);
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

ge::graphStatus GBSATiling::CheckMetadata(gert::TilingContext *context)
{
    // Metadata is required: INT32, 1D, fixed size. Content is filled by AICPU and not re-checked here.
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
    if (metadataSize != static_cast<int64_t>(GBSA_METADATA_TOTAL_SIZE)) {
        OP_LOGE(context->GetNodeName(), "metadata dim 0 must be %u, but got %ld.", GBSA_METADATA_TOTAL_SIZE,
                metadataSize);
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

ge::graphStatus GBSATiling::CheckReservedOptionalInputs(gert::TilingContext *context)
{
    // Reserved optional tensors must stay nullptr until the feature is wired.
    if (context->GetOptionalInputTensor(ATTEN_MASK_INDEX) != nullptr) {
        OP_LOGE(context->GetNodeName(), "atten_mask is NOT YET supported.");
        return ge::GRAPH_FAILED;
    }
    if (context->GetOptionalInputTensor(P_QUANT_SCALE_INDEX) != nullptr) {
        OP_LOGE(context->GetNodeName(), "p_quant_scale is NOT YET supported.");
        return ge::GRAPH_FAILED;
    }
    // Full-quant kernel does not consume dequant scales yet; reject non-null (tests pass null).
    if (context->GetOptionalInputTensor(Q_DEQUANT_SCALE_INDEX) != nullptr ||
        context->GetOptionalInputTensor(K_DEQUANT_SCALE_INDEX) != nullptr ||
        context->GetOptionalInputTensor(V_DEQUANT_SCALE_INDEX) != nullptr) {
        OP_LOGE(context->GetNodeName(), "q/k/v_dequant_scale are NOT YET supported and must be nullptr.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckCuSeqLengths(gert::TilingContext *context)
{
    const bool isPaged = isPaged_;
    const int indices[] = {CU_SEQ_LENGTHS_Q_INDEX, CU_SEQ_LENGTHS_KV_INDEX, SEQUSED_Q_INDEX, SEQUSED_KV_INDEX};
    for (const int index : indices) {
        const bool isCu = index == CU_SEQ_LENGTHS_Q_INDEX || index == CU_SEQ_LENGTHS_KV_INDEX;
        const bool required = index == CU_SEQ_LENGTHS_Q_INDEX ||
                              (index == CU_SEQ_LENGTHS_KV_INDEX && !isPaged) ||
                              (index == SEQUSED_KV_INDEX && isPaged);
        const auto *shape = context->GetOptionalInputShape(index);
        if (shape == nullptr) {
            if (required) {
                OP_LOGE(context->GetNodeName(), "Required sequence input %d is absent.", index);
                return ge::GRAPH_FAILED;
            }
            continue;
        }
        if (isPaged && index == CU_SEQ_LENGTHS_KV_INDEX) {
            OP_LOGE(context->GetNodeName(), "cuSeqLengthsKv must be absent for PA_BBND.");
            return ge::GRAPH_FAILED;
        }
        const auto *desc = context->GetOptionalInputDesc(index);
        if (desc == nullptr || desc->GetDataType() != (isCu ? ge::DT_INT64 : ge::DT_INT32) ||
            shape->GetStorageShape().GetDimNum() != 1U ||
            shape->GetStorageShape().GetDim(0) != static_cast<int64_t>(batch_) + (isCu ? 1 : 0)) {
            OP_LOGE(context->GetNodeName(), "Invalid sequence input %d dtype or shape.", index);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckSoftmaxPrecision(gert::TilingContext *context)
{
    if (softmaxPrecision_ != 0 && softmaxPrecision_ != 1) {
        OP_LOGE(context->GetNodeName(), "Unsupported softmaxPrecision=%u, only 0 or 1 is supported.",
                softmaxPrecision_);
        return ge::GRAPH_FAILED;
    }
    if (socVer_ == SOC_VER_950_CODE) {
        if (quantType_ == GBSA_QUANT_TYPE_FULL && softmaxPrecision_ != 1) {
            OP_LOGE(context->GetNodeName(), "On chip 950, FP8 full-quant only supports softmaxPrecision=1, but got %u.",
                    softmaxPrecision_);
            return ge::GRAPH_FAILED;
        }
    } else if (dataType_ == ge::DT_BF16 && softmaxPrecision_ == 1) {
        OP_LOGE(context->GetNodeName(),
                "On chip 910 & 910_93, when query dtype is bfloat16, "
                "only softmaxPrecision=0 is supported, but got %u.",
                softmaxPrecision_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckQuantConfig(gert::TilingContext *context)
{
    if (blockShapeY_ != 128U && (socVer_ != SOC_VER_950_CODE || quantType_ != 0 ||
        (dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16))) {
        OP_LOGE(context->GetNodeName(), "Generalized blockShapeY requires arch35 FP16/BF16 quantType=0.");
        return ge::GRAPH_FAILED;
    }
    // Full-quant contract: quantType=5 iff Q/K/V dtype is FLOAT8_E4M3FN; FP8 only on 950.
    if (!isPaged_ && (quantType_ != 0 ||
        (dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16))) {
        OP_LOGE(context->GetNodeName(), "TND non-paged KV currently supports FP16/BF16 quantType=0 only.");
        return ge::GRAPH_FAILED;
    }
    const bool isFp8 = (dataType_ == ge::DT_FLOAT8_E4M3FN);
    const bool isQuant5 = (quantType_ == GBSA_QUANT_TYPE_FULL);
    if (isFp8 != isQuant5) {
        OP_LOGE(context->GetNodeName(),
                "FP8 full-quant requires quantType=5 with FLOAT8_E4M3FN Q/K/V, "
                "got quantType=%ld dtype=%d.",
                quantType_, static_cast<int32_t>(dataType_));
        return ge::GRAPH_FAILED;
    }
    if (isQuant5 && socVer_ != SOC_VER_950_CODE) {
        OP_LOGE(context->GetNodeName(), "FP8 full-quant is only supported on chip 950, but socVer=%u.", socVer_);
        return ge::GRAPH_FAILED;
    }
    if (returnSoftmaxlse_ && isQuant5) {
        OP_LOGE(context->GetNodeName(), "returnSoftmaxlse=1 is not supported for FP8 full-quant path.");
        return ge::GRAPH_FAILED;
    }
    if (socVer_ == SOC_VER_950_CODE && !isQuant5 && embeddingSize_ > GBSA_TILE_SIZE_512) {
        OP_LOGE(context->GetNodeName(),
                "Arch35 FP16/BF16 non-quant supports headDim up to 512.");
        return ge::GRAPH_FAILED;
    }
    const bool fdHeadDimSupported = embeddingSize_ == GBSA_TILE_SIZE_128 ||
        (socVer_ == SOC_VER_950_CODE && !isQuant5 && embeddingSize_ <= GBSA_TILE_SIZE_512);
    fdStaticEnabled_ = topK_ >= 12U && fdHeadDimSupported;
    return ge::GRAPH_SUCCESS;
}

uint64_t GBSATiling::GenerateTilingKey()
{
    /**
     * BSA-style decimal bitfields (must match op_kernel tilingkey.h):
     * - [0-1]   Q Layout           2=TND
     * - [2-4]   Mask Type          maskType_ * 1000 (current path: 1)
     * - [5-7]   Softmax Precision  0=Float, 1=Half/BF16 (*100000)
     * - [8-10]  PagedCache         0=Dense, 1=WithCache (*1000000)
     * - [11-13] KV Layout          30=TND, 70=PA_BBND (*1000000)
     * - dtype                    FP16=+0, BF16=+22220, FP8 out FP16=+10 / BF16=+20
     * - LSE                      +100000000
     * - op+arch                  920=aicore220, 925=aicore310
     */
    const bool isArch35 = (socVer_ == SOC_VER_950_CODE);
    uint64_t tilingKey = isArch35 ? GBSA_OP_ARCH35_BASE : GBSA_OP_ARCH22_BASE;

    // dtype / FP8 out-dtype (same additives as BSA)
    if (dataType_ == ge::DT_FLOAT16) {
        // +0
    } else if (dataType_ == ge::DT_BF16) {
        tilingKey += 22220ULL;
    } else if (dataType_ == ge::DT_FLOAT8_E4M3FN) {
        if (attentionOutDtype_ == ge::DT_FLOAT16) {
            tilingKey += 10ULL;
        } else if (attentionOutDtype_ == ge::DT_BF16) {
            tilingKey += 20ULL;
        }
    }

    // Encode layout and storage mode independently. The paging flag is derived from blockTable.
    tilingKey += layoutKv_ == "PA_BBND" ? 70000000ULL : 30000000ULL;
    if (isPaged_) {
        tilingKey += 1000000ULL;
    }

    // Softmax precision: 0=FP32 softmax, 1=FP16/BF16 softmax.
    if (softmaxPrecision_ == 1) {
        tilingKey += 100000ULL;
    }

    // maskType (current support: 1)
    tilingKey += static_cast<uint64_t>(maskType_) * 1000ULL;

    // Q layout: TND = 2
    tilingKey += 2ULL;

    if (returnSoftmaxlse_) {
        tilingKey += GBSA_LSE_OUT_OFFSET;
    }
    if (residualBlockMode_ == GBSA_RESIDUAL_BLOCK_MODE_FORCE) {
        tilingKey += GBSA_RESIDUAL_BLOCK_MODE_TILING_OFFSET;
    }
    return tilingKey;
}

void GBSATiling::CalcBaseTileTilingParams950()
{
    if (groupSize_ <= GBSA_TILE_SIZE_16) {
        qBaseTile_ = GBSA_TILE_SIZE_16;
    } else if (groupSize_ <= GBSA_TILE_SIZE_32) {
        qBaseTile_ = GBSA_TILE_SIZE_32;
    } else if (groupSize_ <= GBSA_TILE_SIZE_64) {
        qBaseTile_ = GBSA_TILE_SIZE_64;
    } else {
        qBaseTile_ = GBSA_TILE_SIZE_128;
    }

    if (embeddingSize_ <= GBSA_TILE_SIZE_256) {
        kvBaseTile_ = qBaseTile_ <= GBSA_TILE_SIZE_32 ? GBSA_TILE_SIZE_512 :
                      qBaseTile_ <= GBSA_TILE_SIZE_64 ? GBSA_TILE_SIZE_256 : GBSA_TILE_SIZE_128;
    } else {
        kvBaseTile_ = qBaseTile_ <= GBSA_TILE_SIZE_16 ? GBSA_TILE_SIZE_384 :
                      qBaseTile_ <= GBSA_TILE_SIZE_32 ? GBSA_TILE_SIZE_256 : GBSA_TILE_SIZE_128;
    }

    // High-precision softmax supports up to 512 columns. Low precision still
    // supports at most 256, including the 384-column tile used by D > 256.
    if (softmaxPrecision_ == 1U && kvBaseTile_ > GBSA_TILE_SIZE_256) {
        kvBaseTile_ = GBSA_TILE_SIZE_256;
    }
}

void GBSATiling::CalcMatmulPhaseL1TileInfo950()
{
    const uint32_t embeddingSizeAligned = embeddingSize_ <= GBSA_TILE_SIZE_128 ? GBSA_TILE_SIZE_128 :
                                                  embeddingSize_ <= GBSA_TILE_SIZE_256 ? GBSA_TILE_SIZE_256 :
                                                                                       GBSA_TILE_SIZE_512;

    // Q is resident in L1. K is the only matrix further split in L1 and uses
    // double buffering. P and V hold a complete attention base tile.
    mm1L1TileM_ = qBaseTile_;
    mm1L1TileKLeft_ = embeddingSizeAligned;
    qL1BufNum_ = 1U;
    kL1BufNum_ = 2U;

    mm2L1TileM_ = qBaseTile_;
    mm2L1TileN_ = embeddingSizeAligned;
    mm2L1TileKLeft_ = kvBaseTile_;
    mm2L1TileKRight_ = kvBaseTile_;
    pL1BufNum_ = 3U;

    if (embeddingSize_ <= GBSA_TILE_SIZE_128) {
        mm1L1TileN_ = qBaseTile_ <= GBSA_TILE_SIZE_64 ? GBSA_TILE_SIZE_256 : GBSA_TILE_SIZE_128;
        mm1L1TileKRight_ = GBSA_TILE_SIZE_128;
        vL1BufNum_ = 2U;
    } else if (embeddingSize_ <= GBSA_TILE_SIZE_256) {
        mm1L1TileN_ = GBSA_TILE_SIZE_128;
        mm1L1TileKRight_ = GBSA_TILE_SIZE_256;
        vL1BufNum_ = qBaseTile_ <= GBSA_TILE_SIZE_32 ? 1U : 2U;
    } else {
        mm1L1TileN_ = GBSA_TILE_SIZE_128;
        mm1L1TileKRight_ = GBSA_TILE_SIZE_128;
        vL1BufNum_ = (qBaseTile_ == GBSA_TILE_SIZE_64) ? 2U : 1U;
    }
}

ge::graphStatus GBSATiling::FillTilingData(gert::TilingContext *context)
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
    tilingData_->set_scaleValue(scaleValue_);
    tilingData_->set_mm1OutSize(mm1OutSize_);
    tilingData_->set_smOnlineOutSize(smOnlineOutSize_);
    tilingData_->set_mm2OutSize(mm2OutSize_);
    tilingData_->set_updateSize(updateSize_);
    tilingData_->set_workSpaceSize(workSpaceSize_);
    tilingData_->set_groupSize(groupSize_);
    uint64_t tilingKey = GenerateTilingKey();
    tilingData_->set_tilingKey(tilingKey);
    context->SetTilingKey(tilingKey);

    tilingData_->baseTileInfo.set_qBaseTile(qBaseTile_);
    tilingData_->baseTileInfo.set_kvBaseTile(kvBaseTile_);
    tilingData_->mmPhaseL1TileInfo.set_mm1L1TileM(mm1L1TileM_);
    tilingData_->mmPhaseL1TileInfo.set_mm1L1TileN(mm1L1TileN_);
    tilingData_->mmPhaseL1TileInfo.set_mm1L1TileKLeft(mm1L1TileKLeft_);
    tilingData_->mmPhaseL1TileInfo.set_mm1L1TileKRight(mm1L1TileKRight_);
    tilingData_->mmPhaseL1TileInfo.set_mm2L1TileM(mm2L1TileM_);
    tilingData_->mmPhaseL1TileInfo.set_mm2L1TileN(mm2L1TileN_);
    tilingData_->mmPhaseL1TileInfo.set_mm2L1TileKLeft(mm2L1TileKLeft_);
    tilingData_->mmPhaseL1TileInfo.set_mm2L1TileKRight(mm2L1TileKRight_);
    tilingData_->mmPhaseL1TileInfo.set_qL1BufNum(qL1BufNum_);
    tilingData_->mmPhaseL1TileInfo.set_kL1BufNum(kL1BufNum_);
    tilingData_->mmPhaseL1TileInfo.set_vL1BufNum(vL1BufNum_);
    tilingData_->mmPhaseL1TileInfo.set_pL1BufNum(pL1BufNum_);
    tilingData_->set_kStride0(kStride0_);
    tilingData_->set_vStride0(vStride0_);
    tilingData_->set_fdStaticEnabled(fdStaticEnabled_ ? 1U : 0U);
    tilingData_->set_fdLseSubStride(fdLseSubStride_);
    tilingData_->set_fdPartialCapacity(fdPartialCapacity_);
    tilingData_->set_fdPartialLseOffset(fdPartialLseOffset_);
    tilingData_->set_fdPartialOOffset(fdPartialOOffset_);
    tilingData_->set_residualBlockMode(static_cast<uint32_t>(residualBlockMode_));
    tilingData_->set_isConsistentTopk(static_cast<uint32_t>(isConsistentTopk_ ? 1 : 0));
    tilingData_->set_totalQTokens(totalQTokens_);
    tilingData_->set_totalKvTokens(totalKvTokens_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::GetTiling(gert::TilingContext *context, GenericBlockSparseAttentionTilingData &tilingData)
{
    tilingData_ = &tilingData;
    ge::graphStatus ret = GetNpuInfo(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "GetNpuInfo failed");
        return ret;
    }
    if (GetInputLayout(context) != ge::GRAPH_SUCCESS || ParseAttrs(context) != ge::GRAPH_SUCCESS ||
        ParseInputTensors(context) != ge::GRAPH_SUCCESS || ParseKvCacheStride0(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Parse attrs/inputs failed");
        return ge::GRAPH_FAILED;
    }
    if (CheckMetadata(context) != ge::GRAPH_SUCCESS || CheckReservedOptionalInputs(context) != ge::GRAPH_SUCCESS ||
        CheckCuSeqLengths(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Validate config failed");
        return ge::GRAPH_FAILED;
    }
    if (socVer_ == SOC_VER_950_CODE && quantType_ == GBSA_QUANT_TYPE_NONE) {
        CalcBaseTileTilingParams950();
        CalcMatmulPhaseL1TileInfo950();
    }
    ret = CalculateWorkSpace(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CalculateWorkSpace failed");
        return ret;
    }
    ret = FillTilingData(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "FillTilingData failed");
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::SetTilingData(gert::TilingContext *context,
                                          GenericBlockSparseAttentionTilingData &tilingData)
{
    OP_CHECK_IF(
        context->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "RawTilingData got from GE context is nullptr."),
        return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingGenericBlockSparseAttention(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "Context is nullptr."),
                return ge::GRAPH_FAILED);
    GenericBlockSparseAttentionTilingData tilingData;
    GBSATiling tiling;
    if (tiling.GetTiling(context, tilingData) == ge::GRAPH_SUCCESS) {
        tiling.SetTilingData(context, tilingData);
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE(context->GetNodeName(), "GetTiling failed");
        return ge::GRAPH_FAILED;
    }
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForGenericBlockSparseAttention(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GenericBlockSparseAttention)
    .Tiling(TilingGenericBlockSparseAttention)
    .TilingParse<GenericBlockSparseAttentionCompileInfo>(TilingPrepareForGenericBlockSparseAttention);

} // namespace optiling
