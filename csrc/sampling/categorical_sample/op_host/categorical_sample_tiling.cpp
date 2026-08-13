/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "categorical_sample_tiling.h"

#include <algorithm>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {
namespace {
constexpr size_t PROCESSED_LOGITS_INDEX = 0;
constexpr size_t EXPANDED_IDX_MAPPING_INDEX = 1;
constexpr size_t TEMPERATURE_INDEX = 2;
constexpr size_t SEED_INDEX = 3;
constexpr size_t POS_INDEX = 4;
constexpr size_t OUTPUT_PROCESSED_LOGITS_INDEX = 5;
constexpr size_t OUTPUT_PROCESSED_LOGITS_COL_INDEX = 6;
constexpr size_t SAMPLED_TOKEN_IDS_INDEX = 0;
constexpr size_t LSE_INDEX = 1;

constexpr uint32_t MAX_VOCAB_SIZE = 1'048'576;
constexpr uint32_t MAX_TILE_ELEMENTS = 4096;
constexpr uint32_t TILE_ALIGNMENT = 256;

uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

uint32_t CeilDiv(uint32_t lhs, uint32_t rhs)
{
    return (lhs + rhs - 1) / rhs;
}

bool IsOneDimensional(const gert::StorageShape* shape)
{
    return shape != nullptr && shape->GetOriginShape().GetDimNum() == 1;
}
}  // namespace

static ge::graphStatus CategoricalSampleTilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* logitsShape = context->GetInputShape(PROCESSED_LOGITS_INDEX);
    const gert::StorageShape* mappingShape = context->GetInputShape(EXPANDED_IDX_MAPPING_INDEX);
    const gert::StorageShape* temperatureShape = context->GetInputShape(TEMPERATURE_INDEX);
    const gert::StorageShape* seedShape = context->GetInputShape(SEED_INDEX);
    const gert::StorageShape* posShape = context->GetInputShape(POS_INDEX);
    const gert::StorageShape* outputProcessedLogitsShape =
        context->GetOptionalInputShape(OUTPUT_PROCESSED_LOGITS_INDEX);
    const gert::StorageShape* outputProcessedLogitsColShape =
        context->GetOptionalInputShape(OUTPUT_PROCESSED_LOGITS_COL_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, logitsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, mappingShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, temperatureShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, seedShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, posShape);

    const gert::Shape& logitsOriginShape = logitsShape->GetOriginShape();
    if (logitsOriginShape.GetDimNum() != 2) {
        OP_LOGE(context->GetNodeName(), "processedLogits must be 2D.");
        return ge::GRAPH_FAILED;
    }
    const int64_t numRows = logitsOriginShape.GetDim(0);
    const int64_t vocabSize = logitsOriginShape.GetDim(1);
    if (numRows <= 0 || numRows > UINT32_MAX || vocabSize <= 0 || vocabSize > MAX_VOCAB_SIZE) {
        OP_LOGE(context->GetNodeName(), "processedLogits shape is outside the supported range.");
        return ge::GRAPH_FAILED;
    }
    if (!IsOneDimensional(mappingShape) || !IsOneDimensional(temperatureShape) || !IsOneDimensional(seedShape) ||
        !IsOneDimensional(posShape)) {
        OP_LOGE(context->GetNodeName(), "sampling metadata tensors must be 1D.");
        return ge::GRAPH_FAILED;
    }
    if (mappingShape->GetOriginShape().GetDim(0) != numRows || posShape->GetOriginShape().GetDim(0) != numRows ||
        temperatureShape->GetOriginShape().GetDim(0) <= 0 ||
        seedShape->GetOriginShape().GetDim(0) != temperatureShape->GetOriginShape().GetDim(0)) {
        OP_LOGE(context->GetNodeName(), "sampling metadata shapes are inconsistent.");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* rowStride = attrs->GetInt(0);
    const int64_t* outputProcessedLogitsStride = attrs->GetInt(1);
    const bool* applyTemperature = attrs->GetBool(2);
    const bool* returnLse = attrs->GetBool(3);
    const bool* useFp64 = attrs->GetBool(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowStride);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputProcessedLogitsStride);
    OP_CHECK_NULL_WITH_CONTEXT(context, applyTemperature);
    OP_CHECK_NULL_WITH_CONTEXT(context, returnLse);
    OP_CHECK_NULL_WITH_CONTEXT(context, useFp64);
    if ((*rowStride != 0 && *rowStride < vocabSize) || *rowStride > UINT32_MAX) {
        OP_LOGE(context->GetNodeName(), "processedLogits row stride is invalid.");
        return ge::GRAPH_FAILED;
    }

    uint32_t outputProcessedLogitsNumCols = 0;
    bool outputProcessedLogitsColPerToken = false;
    if (outputProcessedLogitsShape != nullptr) {
        const gert::Shape& cacheShape = outputProcessedLogitsShape->GetOriginShape();
        if ((cacheShape.GetDimNum() != 2 && cacheShape.GetDimNum() != 3) ||
            cacheShape.GetDim(0) != temperatureShape->GetOriginShape().GetDim(0) ||
            cacheShape.GetDim(cacheShape.GetDimNum() - 1) != vocabSize) {
            OP_LOGE(context->GetNodeName(), "outputProcessedLogits shape is inconsistent.");
            return ge::GRAPH_FAILED;
        }
        const int64_t numCols = cacheShape.GetDimNum() == 3 ? cacheShape.GetDim(1) : 1;
        if (numCols <= 0 || numCols > UINT32_MAX || *outputProcessedLogitsStride < vocabSize ||
            *outputProcessedLogitsStride > UINT32_MAX) {
            OP_LOGE(context->GetNodeName(), "outputProcessedLogits stride or column count is invalid.");
            return ge::GRAPH_FAILED;
        }
        outputProcessedLogitsNumCols = static_cast<uint32_t>(numCols);

        if (outputProcessedLogitsColShape != nullptr) {
            const gert::Shape& colShape = outputProcessedLogitsColShape->GetOriginShape();
            if (colShape.GetDimNum() == 0) {
                outputProcessedLogitsColPerToken = false;
            } else if (colShape.GetDimNum() == 1 && colShape.GetDim(0) == numRows) {
                outputProcessedLogitsColPerToken = true;
            } else {
                OP_LOGE(context->GetNodeName(), "outputProcessedLogitsCol must be scalar or one value per row.");
                return ge::GRAPH_FAILED;
            }
        }
    } else if (outputProcessedLogitsColShape != nullptr) {
        OP_LOGE(context->GetNodeName(), "outputProcessedLogitsCol requires outputProcessedLogits.");
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* sampledIdsShape = context->GetOutputShape(SAMPLED_TOKEN_IDS_INDEX);
    const gert::StorageShape* lseShape = context->GetOutputShape(LSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sampledIdsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, lseShape);
    if (!IsOneDimensional(sampledIdsShape) || sampledIdsShape->GetOriginShape().GetDim(0) != numRows ||
        !IsOneDimensional(lseShape) || lseShape->GetOriginShape().GetDim(0) != numRows) {
        OP_LOGE(context->GetNodeName(), "categorical sample output shapes are inconsistent.");
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum == 0) {
        OP_LOGE(context->GetNodeName(), "failed to get AIV core count.");
        return ge::GRAPH_FAILED;
    }

    auto logitsDesc = context->GetInputDesc(PROCESSED_LOGITS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, logitsDesc);
    uint64_t tilingKey = 0;
    switch (logitsDesc->GetDataType()) {
        case ge::DT_FLOAT:
            tilingKey = 1;
            break;
        case ge::DT_FLOAT16:
            tilingKey = 2;
            break;
        case ge::DT_BF16:
            tilingKey = 3;
            break;
        default:
            OP_LOGE(context->GetNodeName(), "unsupported processedLogits dtype.");
            return ge::GRAPH_FAILED;
    }

    const uint32_t vocab = static_cast<uint32_t>(vocabSize);
    const uint32_t tileElements = std::min(MAX_TILE_ELEMENTS, AlignUp(vocab, TILE_ALIGNMENT));
    CategoricalSampleTilingData tilingData;
    tilingData.set_numRows(static_cast<uint32_t>(numRows));
    tilingData.set_vocabSize(vocab);
    tilingData.set_numRequests(static_cast<uint32_t>(temperatureShape->GetOriginShape().GetDim(0)));
    tilingData.set_rowStride(static_cast<uint32_t>(*rowStride));
    tilingData.set_outputProcessedLogitsStride(static_cast<uint32_t>(*outputProcessedLogitsStride));
    tilingData.set_outputProcessedLogitsNumCols(outputProcessedLogitsNumCols);
    tilingData.set_tileElements(tileElements);
    tilingData.set_tileCount(CeilDiv(vocab, tileElements));
    tilingData.set_hasOutputProcessedLogits(outputProcessedLogitsShape != nullptr ? 1U : 0U);
    tilingData.set_hasOutputProcessedLogitsCol(outputProcessedLogitsColShape != nullptr ? 1U : 0U);
    tilingData.set_outputProcessedLogitsColPerToken(outputProcessedLogitsColPerToken ? 1U : 0U);
    tilingData.set_applyTemperature(*applyTemperature ? 1U : 0U);
    tilingData.set_returnLse(*returnLse ? 1U : 0U);
    tilingData.set_useFp64(*useFp64 ? 1U : 0U);

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSize);
    *workspaceSize = 0;
    context->SetBlockDim(std::min(coreNum, static_cast<uint32_t>(numRows)));
    context->SetTilingKey(tilingKey);

    auto rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCategoricalSample(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CategoricalSample)
    .Tiling(CategoricalSampleTilingFunc)
    .TilingParse<CategoricalSampleCompileInfo>(TilingParseForCategoricalSample);
}  // namespace optiling
