/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file fused_gdn_gating_tiling.cpp
 * \brief Tiling implementation for FusedGdnGating.
 */

#include "fused_gdn_gating_tiling.h"
#include "fused_gdn_gating_tiling_utils.h"

#include "register/op_impl_registry.h"
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include "../op_kernel/fused_gdn_gating_tiling_data.h"

using namespace FusedGdnGating;

namespace optiling {

namespace {

constexpr uint64_t TILING_KEY_BF16 = 1;
constexpr uint64_t TILING_KEY_FP16 = 2;
constexpr uint64_t TILING_KEY_PARAM_BF16_OFFSET = 2;
constexpr uint64_t TILING_KEY_PARAM_FP16_OFFSET = 4;
constexpr size_t INPUT_INDEX_A_LOG = 0;
constexpr size_t INPUT_INDEX_A = 1;
constexpr size_t INPUT_INDEX_DT_BIAS = 3;

} // namespace

ge::graphStatus FusedGdnGatingTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aivNum == 0) {
        aivNum = 1;
    }

    auto *shapeA = context->GetInputShape(INPUT_INDEX_A);
    if (shapeA == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &storageShape = shapeA->GetStorageShape();
    if (storageShape.GetDimNum() < 2) {
        return ge::GRAPH_FAILED;
    }
    int64_t numBatches = storageShape.GetDim(0);
    int64_t numHeads   = storageShape.GetDim(1);
    if (numBatches <= 0 || numHeads <= 0) {
        return ge::GRAPH_FAILED;
    }

    float beta = 1.0f;
    float threshold = 20.0f;
    auto *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float *betaAttr = attrs->GetAttrPointer<float>(0);
        if (betaAttr != nullptr)      { beta      = *betaAttr; }
        const float *thresholdAttr = attrs->GetAttrPointer<float>(1);
        if (thresholdAttr != nullptr) { threshold = *thresholdAttr; }
    }

    auto *aDesc = context->GetInputDesc(INPUT_INDEX_A);
    auto *aLogDesc = context->GetInputDesc(INPUT_INDEX_A_LOG);
    auto *dtBiasDesc = context->GetInputDesc(INPUT_INDEX_DT_BIAS);
    if (aDesc == nullptr || aLogDesc == nullptr || dtBiasDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType aDtype = aDesc->GetDataType();
    ge::DataType aLogDtype = aLogDesc->GetDataType();
    ge::DataType dtBiasDtype = dtBiasDesc->GetDataType();
    if (aLogDtype != dtBiasDtype) {
        return ge::GRAPH_FAILED;
    }
    uint64_t tilingKey = TILING_KEY_BF16;
    if (aDtype == ge::DT_FLOAT16) {
        tilingKey = TILING_KEY_FP16;
    }
    if (aLogDtype == ge::DT_BF16) {
        tilingKey += TILING_KEY_PARAM_BF16_OFFSET;
    } else if (aLogDtype == ge::DT_FLOAT16) {
        tilingKey += TILING_KEY_PARAM_FP16_OFFSET;
    }

    uint32_t blockDim = static_cast<uint32_t>(numBatches);
    if (blockDim > aivNum) {
        blockDim = aivNum;
    }

    uint32_t numHeadsU32 = static_cast<uint32_t>(numHeads);
    uint32_t numBatchesU32 = static_cast<uint32_t>(numBatches);
    uint32_t rowsConservative = ComputeRowsPerIter(numHeadsU32, ubSize);
    uint32_t rowsPerIter = rowsConservative;

    // Block utilization: ensure enough chunks for all AIV cores.
    {
        uint32_t totalChunksForRPI = (numBatchesU32 + rowsPerIter - 1) / rowsPerIter;
        if (numBatchesU32 <= rowsPerIter || totalChunksForRPI < blockDim) {
            uint32_t maxRPI = numBatchesU32 / blockDim;
            if (maxRPI < 1) { maxRPI = 1; }
            if      (maxRPI >= 128) { rowsPerIter = 128; }
            else if (maxRPI >= 64)  { rowsPerIter = 64;  }
            else if (maxRPI >= 32)  { rowsPerIter = 32;  }
            else if (maxRPI >= 16)  { rowsPerIter = 16;  }
            else if (maxRPI >= 8)   { rowsPerIter = 8;   }
            else if (maxRPI >= 4)   { rowsPerIter = 4;   }
            else if (maxRPI >= 2)   { rowsPerIter = 2;   }
            else                    { rowsPerIter = 1;   }
            if (rowsPerIter > rowsConservative) { rowsPerIter = rowsConservative; }
        }
    }

    const bool bulkDmaBatchOk = (numBatchesU32 > blockDim * rowsPerIter);
    bool useBulkDma = bulkDmaBatchOk && CanUseBulkDma(numHeadsU32, rowsPerIter);

    FusedGdnGatingTilingData td{};
    td.numHeads      = numHeadsU32;
    td.numBatches    = numBatchesU32;
    td.rowsPerIter   = rowsPerIter;
    td.useBulkDma    = useBulkDma ? 1u : 0u;
    td.beta          = beta;
    td.threshold     = threshold;

    const size_t tilingSize = sizeof(FusedGdnGatingTilingData);
    auto *rawTilingData = context->GetRawTilingData();
    if (rawTilingData == nullptr || rawTilingData->GetCapacity() < tilingSize) {
        return ge::GRAPH_FAILED;
    }
    errno_t rc = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(),
                          &td, tilingSize);
    if (rc != EOK) {
        return ge::GRAPH_FAILED;
    }
    rawTilingData->SetDataSize(tilingSize);

    context->SetBlockDim(blockDim);
    context->SetTilingKey(tilingKey);

    // No GM workspace needed.
    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = 0;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForFusedGdnGating(gert::TilingParseContext *context)
{
    // Required by CANN tiling framework for "_pattern" registration.
    (void)context;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

IMPL_OP_OPTILING(FusedGdnGating)
    .Tiling(optiling::FusedGdnGatingTilingFunc)
    .TilingParse<optiling::FusedGdnGatingCompileInfo>(optiling::TilingPrepareForFusedGdnGating);
