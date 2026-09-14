/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_prolog_tiling_check.cpp
 * \brief
 */

#include "mla_prolog_tiling_check.h"
#include <graph/utils/type_utils.h>
#include "log/log.h"

using namespace ge;

namespace optiling {

const std::unordered_map<ge::DataType, uint32_t> DTYPE_TO_SIZE{
    {ge::DT_BF16, 2},        {ge::DT_FLOAT16, 2},  {ge::DT_INT8, 1},  {ge::DT_FLOAT8_E4M3FN, 1},
    {ge::DT_FLOAT8_E8M0, 1}, {ge::DT_HIFLOAT8, 1}, {ge::DT_INT32, 4}, {ge::DT_FLOAT, 4}};

template <typename E>
std::string ElemToString(const E &elem)
{
    return std::to_string(elem);
}

std::string FormatToString(const ge::Format format)
{
    return std::string(ge::GetFormatName(format));
}

template <typename C, typename Func = std::string (*)(const typename C::value_type &)>
std::string ConvertContainerToString(const C &container, Func func = ElemToString<typename C::value_type>)
{
    if (container.empty() || func == nullptr) {
        return "[]";
    }
    std::stringstream ss;
    ss << "[";
    bool isFirst = true;
    for (const auto &elem : container) {
        if (!isFirst) {
            ss << ", ";
        }
        ss << func(elem);
        isFirst = false;
    }
    ss << "]";
    return ss.str();
}

template <typename C, typename Func = std::string (*)(const typename C::value_type &)>
std::string ConvertContainerToStringV3(const C &container, Func func = ElemToString<typename C::value_type>)
{
    if (container.empty() || func == nullptr) {
        return "[]";
    }
    std::stringstream ss;
    bool isFirst = true;
    for (const auto &elem : container) {
        if (!isFirst) {
            ss << ", ";
        }
        ss << func(elem);
        isFirst = false;
    }
    return ss.str();
}

std::string GetShapeStr(const gert::Shape &aShape)
{
    std::string shapeStr = "[";
    for (size_t i = 0; i < aShape.GetDimNum(); ++i) {
        shapeStr += std::to_string(aShape.GetDim(i)) + (i + 1 < aShape.GetDimNum() ? ", " : "");
    }
    return shapeStr + "]";
}

template <typename T>
inline auto CeilDiv(T a, T b) -> T
{
    if (b == 0) {
        return b;
    }
    return (a + b - 1) / b;
}

NpuArch MlaPrologTilingCheck::GetCurNpuArch() const
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_.platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    return npuArch;
}

// =================================全量参数校验=================================
bool MlaPrologTilingCheck::CheckAttrsRange() const
{
    if (std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        if (GetCurNpuArch() == NpuArch::DAV_3510) {
            // arch35: 非量化(wq=0) + MXFP8(wq=3)
            const std::unordered_set<uint32_t> supportedWeightQuantMode{0U, 3U};
            OP_CHECK_IF(supportedWeightQuantMode.find(*context_.weightQuantMode) == supportedWeightQuantMode.end(),
                        OP_LOGE_FOR_INVALID_VALUE(context_.opName, "WeightQuantMode",
                                                  std::to_string(*context_.weightQuantMode), "{0, 3}"),
                        return false);
            // MXFP8: kvq 仅 {0,1}，且必须与 queryQuantMode 一致
            if (*context_.weightQuantMode == static_cast<int64_t>(WEIGHT_QUANT_MODE::MXFP8_FULL_QUANT)) {
                const std::unordered_set<uint32_t> supportedKvQuantMode{0U, 1U};
                OP_CHECK_IF(supportedKvQuantMode.find(*context_.kvQuantMode) == supportedKvQuantMode.end(),
                            OP_LOGE_FOR_INVALID_VALUE(context_.opName, "KvQuantMode",
                                                      std::to_string(*context_.kvQuantMode), "{0, 1}"),
                            return false);
                OP_CHECK_IF(*context_.kvQuantMode != *context_.queryQuantMode,
                            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                                context_.opName, "kvQuantMode/queryQuantMode",
                                "kv=" + std::to_string(*context_.kvQuantMode) +
                                    ", query=" + std::to_string(*context_.queryQuantMode),
                                "On DAV3510 MXFP8, kvQuantMode must equal queryQuantMode"),
                            return false);
            } else {
                const std::unordered_set<uint32_t> supportedKvQuantMode{0U};
                OP_CHECK_IF(supportedKvQuantMode.find(*context_.kvQuantMode) == supportedKvQuantMode.end(),
                            OP_LOGE_FOR_INVALID_VALUE(context_.opName, "KvQuantMode",
                                                      std::to_string(*context_.kvQuantMode), "{0}"),
                            return false);
            }
        } else {
            // arch22: 非量化(wq=0) + c8 INT8(wq=1/2)
            const std::unordered_set<uint32_t> supportedWeightQuantMode{0U, 1U, 2U};
            OP_CHECK_IF(supportedWeightQuantMode.find(*context_.weightQuantMode) == supportedWeightQuantMode.end(),
                        OP_LOGE_FOR_INVALID_VALUE(context_.opName, "WeightQuantMode",
                                                  std::to_string(*context_.weightQuantMode), "{0, 1, 2}"),
                        return false);
            // c8: 半量化 kv∈{0,2}；全量化 kv∈{0,1}（不再支持 pertile=3）
            std::unordered_set<uint32_t> supportedKvQuantMode{0U};
            if (*context_.weightQuantMode == static_cast<int64_t>(WEIGHT_QUANT_MODE::PARTIAL_QUANT)) {
                supportedKvQuantMode = {0U, 2U};
            } else if (*context_.weightQuantMode == static_cast<int64_t>(WEIGHT_QUANT_MODE::FULL_QUANT)) {
                supportedKvQuantMode = {0U, 1U};
            }
            OP_CHECK_IF(supportedKvQuantMode.find(*context_.kvQuantMode) == supportedKvQuantMode.end(),
                        OP_LOGE_FOR_INVALID_VALUE(context_.opName, "KvQuantMode",
                                                  std::to_string(*context_.kvQuantMode),
                                                  ConvertContainerToStringV3(supportedKvQuantMode)),
                        return false);
        }

        const std::unordered_set<uint32_t> supportedQueryQuantMode{0U, 1U};
        OP_CHECK_IF(supportedQueryQuantMode.find(*context_.queryQuantMode) == supportedQueryQuantMode.end(),
                    OP_LOGE_FOR_INVALID_VALUE(context_.opName, "QueryQuantMode",
                                              std::to_string(*context_.queryQuantMode), "{0, 1}"),
                    return false);

        const std::unordered_set<uint32_t> supportedCkvkrRepoMode{0U, 1U};
        OP_CHECK_IF(supportedCkvkrRepoMode.find(*context_.ckvkrRepoMode) == supportedCkvkrRepoMode.end(),
                    OP_LOGE_FOR_INVALID_VALUE(context_.opName, "CkvkrRepoMode", std::to_string(*context_.ckvkrRepoMode),
                                              "{0, 1}"),
                    return false);

        const std::unordered_set<uint32_t> supportedQuantScaleRepoMode{0U, 1U};
        OP_CHECK_IF(supportedQuantScaleRepoMode.find(*context_.quantScaleRepoMode) == supportedQuantScaleRepoMode.end(),
                    OP_LOGE_FOR_INVALID_VALUE(context_.opName, "QuantScaleRepoMode",
                                              std::to_string(*context_.quantScaleRepoMode), "{0, 1}"),
                    return false);

        const std::unordered_set<uint32_t> supportedTileSize{128U};
        OP_CHECK_IF(supportedTileSize.find(*context_.tileSize) == supportedTileSize.end(),
                    OP_LOGE_FOR_INVALID_VALUE(context_.opName, "TileSize", std::to_string(*context_.tileSize), "{128}"),
                    return false);
    }
    return true;
}

bool MlaPrologTilingCheck::CheckAttrsNotNull() const
{
    OP_CHECK_IF(context_.rmsNormEspilonCq == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_.opName, "rmsNormEspilonCq"),
                return false);

    OP_CHECK_IF(context_.rmsNormEspilonCkv == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_.opName, "rmsNormEspilonCkv"),
                return false);

    OP_CHECK_IF(context_.cacheMode == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_.opName, "cacheMode"), return false);

    if (std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        OP_CHECK_IF(context_.queryNormFlag == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_.opName, "queryNormFlag"),
                    return false);

        OP_CHECK_IF(context_.weightQuantMode == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_.opName, "weightQuantMode"),
                    return false);

        OP_CHECK_IF(context_.kvQuantMode == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_.opName, "kvQuantMode"),
                    return false);

        OP_CHECK_IF(context_.queryQuantMode == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_.opName, "queryQuantMode"),
                    return false);

        OP_CHECK_IF(context_.ckvkrRepoMode == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_.opName, "ckvkrRepoMode"),
                    return false);

        OP_CHECK_IF(context_.quantScaleRepoMode == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context_.opName, "quantScaleRepoMode"), return false);

        OP_CHECK_IF(context_.tileSize == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_.opName, "tileSize"),
                    return false);

        OP_CHECK_IF(context_.qcQrScale == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_.opName, "qcQrScale"),
                    return false);

        OP_CHECK_IF(context_.kcScale == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_.opName, "kcScale"), return false);
    }
    return true;
}

ge::graphStatus MlaPrologTilingCheck::CheckAttrs() const
{
    if (!CheckAttrsNotNull() || !CheckAttrsRange()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTilingCheck::CheckDims() const
{
    OP_CHECK_IF(
        baseShapeInfo_.bSize > MAX_B_SIZE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_.opName, "B", std::to_string(baseShapeInfo_.bSize),
                                              "B size should not be greater than " + std::to_string(MAX_B_SIZE)),
        return ge::GRAPH_FAILED);
    const std::set<uint32_t> supportedHeSize{1024U, 2048U, 3072U, 4096U, 5120U, 6144U, 7168U, 7680U, 8192U};
    OP_CHECK_IF(supportedHeSize.find(baseShapeInfo_.heSize) == supportedHeSize.end(),
                OP_LOGE_FOR_INVALID_VALUE(context_.opName, "He", std::to_string(baseShapeInfo_.heSize),
                                          ConvertContainerToStringV3(supportedHeSize)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(baseShapeInfo_.nSize < 1U || baseShapeInfo_.nSize > 128U,
                OP_LOGE_FOR_INVALID_VALUE(context_.opName, "N", std::to_string(baseShapeInfo_.nSize),
                                          "N size should be within [1, 128]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(baseShapeInfo_.hckvSize != HCKV_SIZE,
                OP_LOGE_FOR_INVALID_VALUE(context_.opName, "Hckv", std::to_string(baseShapeInfo_.hckvSize),
                                          std::to_string(HCKV_SIZE)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(baseShapeInfo_.drSize != DR_SIZE,
                OP_LOGE_FOR_INVALID_VALUE(context_.opName, "Dr", std::to_string(baseShapeInfo_.drSize),
                                          std::to_string(DR_SIZE)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(baseShapeInfo_.nkvSize != NKV_SIZE,
                OP_LOGE_FOR_INVALID_VALUE(context_.opName, "Nkv", std::to_string(baseShapeInfo_.nkvSize),
                                          std::to_string(NKV_SIZE)),
                return ge::GRAPH_FAILED);
    if (scenarioInfo_.cacheMode_ != CACHE_MODE::BSND && scenarioInfo_.cacheMode_ != CACHE_MODE::TND) {
        OP_CHECK_IF(baseShapeInfo_.blockSize < MIN_BLOCK_SIZE || baseShapeInfo_.blockSize > MAX_BLOCK_SIZE ||
                        baseShapeInfo_.blockSize % ALIGN_BLOCK_SIZE != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_.opName, "blockSize", std::to_string(baseShapeInfo_.blockSize),
                        "BlockSize must be within [" + std::to_string(MIN_BLOCK_SIZE) + ", " +
                            std::to_string(MAX_BLOCK_SIZE) + "] and a multiple of " + std::to_string(ALIGN_BLOCK_SIZE)),
                    return ge::GRAPH_FAILED);
    }
    if (CheckHcqSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDtileSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTilingCheck::CheckQuantMode() const
{
    if (GetCurNpuArch() == NpuArch::DAV_3510) {
        // arch35: 非量化 + MXFP8(kvq/qq matched 0/1)
        const std::set<uint32_t> supportedQuantModes{
            static_cast<uint32_t>(QUANT_MODE::NO_QUANT),
            static_cast<uint32_t>(QUANT_MODE::MXFP8_FULL_QUANT_KV_NO_QUANT),
            static_cast<uint32_t>(QUANT_MODE::MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR)};
        OP_CHECK_IF(supportedQuantModes.find(static_cast<uint32_t>(scenarioInfo_.quantMode_)) ==
                        supportedQuantModes.end(),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_.opName, "quantMode", std::to_string(static_cast<uint32_t>(scenarioInfo_.quantMode_)),
                        "On DAV3510, quantMode allows only " + ConvertContainerToStringV3(supportedQuantModes)),
                    return ge::GRAPH_FAILED);
        // MXFP8 仅支持 cache_mode=PA_BSND
        if (scenarioInfo_.quantMode_ == QUANT_MODE::MXFP8_FULL_QUANT_KV_NO_QUANT ||
            scenarioInfo_.quantMode_ == QUANT_MODE::MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR) {
            OP_CHECK_IF(scenarioInfo_.cacheMode_ != CACHE_MODE::PA_BSND,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context_.opName, "cacheMode",
                            std::to_string(static_cast<uint32_t>(scenarioInfo_.cacheMode_)),
                            "On DAV3510 MXFP8, cacheMode allows only PA_BSND"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        // arch22: 非量化 + c8 INT8（QUANT_MODE 1~4）
        const std::set<uint32_t> supportedQuantModes{
            static_cast<uint32_t>(QUANT_MODE::NO_QUANT),
            static_cast<uint32_t>(QUANT_MODE::PARTIAL_QUANT_KV_NO_QUANT),
            static_cast<uint32_t>(QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_CHANNEL),
            static_cast<uint32_t>(QUANT_MODE::FULL_QUANT_KV_NO_QUANT),
            static_cast<uint32_t>(QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TENSOR)};
        OP_CHECK_IF(supportedQuantModes.find(static_cast<uint32_t>(scenarioInfo_.quantMode_)) ==
                        supportedQuantModes.end(),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_.opName, "quantMode", std::to_string(static_cast<uint32_t>(scenarioInfo_.quantMode_)),
                        "On arch22, quantMode allows only " + ConvertContainerToStringV3(supportedQuantModes)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTilingCheck::CheckHcqSize() const
{
    if (std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        const std::set<uint32_t> supportedHcqSize{1536U, 2048U};
        OP_CHECK_IF(supportedHcqSize.find(baseShapeInfo_.hcqSize) == supportedHcqSize.end(),
                    OP_LOGE_FOR_INVALID_VALUE(context_.opName, "Hcq", std::to_string(baseShapeInfo_.hcqSize),
                                              ConvertContainerToStringV3(supportedHcqSize)),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(baseShapeInfo_.hcqSize != HCQ_SIZE,
                    OP_LOGE(context_.opName, "Hcq allows only %u, got %u.", HCQ_SIZE, baseShapeInfo_.hcqSize),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTilingCheck::CheckDSize() const
{
    if (std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        const std::set<uint32_t> supportedDSize{128U, 192U};
        OP_CHECK_IF(supportedDSize.find(baseShapeInfo_.dSize) == supportedDSize.end(),
                    OP_LOGE_FOR_INVALID_VALUE(context_.opName, "D", std::to_string(baseShapeInfo_.dSize),
                                              ConvertContainerToStringV3(supportedDSize)),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(baseShapeInfo_.dSize != D_SIZE,
                    OP_LOGE(context_.opName, "D allows only %u, got %u.", D_SIZE, baseShapeInfo_.dSize),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTilingCheck::CheckDtileSize() const
{
    if (std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        uint32_t supportedDtileSize = baseShapeInfo_.hckvSize;
        if (*(context_.ckvkrRepoMode) == static_cast<int>(CKVKR_REPO_MODE::COMBINE)) {
            supportedDtileSize +=
                baseShapeInfo_.drSize * (DTYPE_TO_SIZE.at(ge::DT_BF16) / DTYPE_TO_SIZE.at(ge::DT_INT8));
        }
        if (*(context_.quantScaleRepoMode) == static_cast<int>(QUANT_SCALE_REPO_MODE::COMBINE)) {
            supportedDtileSize += baseShapeInfo_.hckvSize / static_cast<uint32_t>(*(context_.tileSize)) *
                                  (DTYPE_TO_SIZE.at(ge::DT_FLOAT) / DTYPE_TO_SIZE.at(ge::DT_INT8));
        }
        if (baseShapeInfo_.dtileSize != supportedDtileSize) {
            if (*(context_.kvQuantMode) == static_cast<int>(KV_QUANT_MODE::PER_TILE)) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_.opName, "dtileSize", std::to_string(baseShapeInfo_.dtileSize),
                    "when kvQuantMode is PER_TILE, dtileSize allows only " + std::to_string(supportedDtileSize));
            } else {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_.opName, "dtileSize", std::to_string(baseShapeInfo_.dtileSize),
                    "when kvQuantMode is in {NO_QUANT, PER_TENSOR, PER_CHANNEL}, dtileSize allows only " +
                        std::to_string(supportedDtileSize));
            }
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void MlaPrologTilingCheck::GenExpectedParamInfo()
{
    FillCommonParamInfo();
    FillScenarioParamInfo();
}

void MlaPrologTilingCheck::FillCommonParamInfo()
{
    FillRequiredParamShapeWithDims();
    FillOptionalOutputParamShapeWithDims();

    if (context_.weightDq.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_DIM_NUM_4) {
        expectedParamInfo_[WEIGHT_DQ_NAME].dimNum = MLA_PROLOG_DIM_NUM_4;
        int64_t weightAxisSize = 32L / ge::GetSizeByDataType(context_.weightDq.desc->GetDataType());
        expectedParamInfo_[WEIGHT_DQ_NAME].shape =
            std::vector<int64_t>{static_cast<int64_t>(baseShapeInfo_.hcqSize) / weightAxisSize,
                                 static_cast<int64_t>(baseShapeInfo_.heSize) / NZ_H0_SIZE, NZ_H0_SIZE, weightAxisSize};
    }
    if (context_.weightUqQr.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_DIM_NUM_4) {
        expectedParamInfo_[WEIGHT_UQ_QR_NAME].dimNum = MLA_PROLOG_DIM_NUM_4;
        int64_t weightAxisSize = 32L / ge::GetSizeByDataType(context_.weightUqQr.desc->GetDataType());
        expectedParamInfo_[WEIGHT_UQ_QR_NAME].shape =
            std::vector<int64_t>{static_cast<int64_t>(baseShapeInfo_.headSizeUqQr) / weightAxisSize,
                                 static_cast<int64_t>(baseShapeInfo_.hcqSize) / NZ_H0_SIZE, NZ_H0_SIZE, weightAxisSize};
    }
    if (context_.weightDkvKr.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_DIM_NUM_4) {
        expectedParamInfo_[WEIGHT_DKV_KR_NAME].dimNum = MLA_PROLOG_DIM_NUM_4;
        int64_t weightAxisSize = 32L / ge::GetSizeByDataType(context_.weightDkvKr.desc->GetDataType());
        expectedParamInfo_[WEIGHT_DKV_KR_NAME].shape =
            std::vector<int64_t>{static_cast<int64_t>(baseShapeInfo_.hckvSize + baseShapeInfo_.drSize) / weightAxisSize,
                                 static_cast<int64_t>(baseShapeInfo_.heSize) / NZ_H0_SIZE, NZ_H0_SIZE, weightAxisSize};
    }

    expectedParamInfo_[WEIGHT_DQ_NAME].format = ge::FORMAT_FRACTAL_NZ;
    expectedParamInfo_[WEIGHT_UQ_QR_NAME].format = ge::FORMAT_FRACTAL_NZ;
    expectedParamInfo_[WEIGHT_DKV_KR_NAME].format = ge::FORMAT_FRACTAL_NZ;

    if (scenarioInfo_.cacheMode_ == CACHE_MODE::PA_BLK_BSND || scenarioInfo_.cacheMode_ == CACHE_MODE::PA_BLK_NZ) {
        if (scenarioInfo_.batchSeqFusedFlag_) {
            expectedParamInfo_.emplace(ACTUAL_SEQ_LEN_NAME, std::vector<uint32_t>{baseShapeInfo_.bSize});
            expectedParamInfo_[ACTUAL_SEQ_LEN_NAME].dtype = ge::DT_INT32;
            expectedParamInfo_[ACTUAL_SEQ_LEN_NAME].format = ge::FORMAT_ND;
            expectedParamInfo_[CACHE_INDEX_NAME].shape = actualParamInfo_[CACHE_INDEX_NAME].shape;
        } else {
            expectedParamInfo_[CACHE_INDEX_NAME].shape =
                std::vector<int64_t>{baseShapeInfo_.bSize, CeilDiv(baseShapeInfo_.s1Size, baseShapeInfo_.blockSize)};
        }
    }
}

void MlaPrologTilingCheck::FillRequiredParamShapeWithDims()
{
    FillTokenAndQueryShapes();
    FillWeightAndNormShapes();
    FillCacheShapes();
    expectedParamInfo_.emplace(KV_CACHE_OUT_NAME, expectedParamInfo_[KV_CACHE_NAME]);
    expectedParamInfo_.emplace(KR_CACHE_OUT_NAME, expectedParamInfo_[KR_CACHE_NAME]);
}

void MlaPrologTilingCheck::FillTokenAndQueryShapes()
{
    const bool ropeEnabled =
        !(std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) == 0 && context_.doRope != nullptr &&
          !*(context_.doRope));
    if (scenarioInfo_.batchSeqFusedFlag_) {
        expectedParamInfo_.emplace(TOKEN_X_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.heSize});
        if (ropeEnabled) {
            expectedParamInfo_.emplace(ROPE_SIN_NAME,
                                       std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.drSize});
            expectedParamInfo_.emplace(ROPE_COS_NAME,
                                       std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.drSize});
        } else {
            expectedParamInfo_[ROPE_SIN_NAME].isValid = false;
            expectedParamInfo_[ROPE_COS_NAME].isValid = false;
        }
        if ((scenarioInfo_.cacheMode_ != CACHE_MODE::BSND) && (scenarioInfo_.cacheMode_ != CACHE_MODE::TND)) {
            expectedParamInfo_.emplace(CACHE_INDEX_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize});
            expectedParamInfo_[CACHE_INDEX_NAME].dtype = ge::DT_INT64;
        }
        expectedParamInfo_.emplace(
            QUERY_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.nSize, baseShapeInfo_.hckvSize});
        expectedParamInfo_.emplace(
            QUERY_ROPE_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.nSize, baseShapeInfo_.drSize});
    } else {
        expectedParamInfo_.emplace(
            TOKEN_X_NAME, std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size, baseShapeInfo_.heSize});
        if (ropeEnabled) {
            expectedParamInfo_.emplace(
                ROPE_SIN_NAME, std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size, baseShapeInfo_.drSize});
            expectedParamInfo_.emplace(
                ROPE_COS_NAME, std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size, baseShapeInfo_.drSize});
        } else {
            expectedParamInfo_[ROPE_SIN_NAME].isValid = false;
            expectedParamInfo_[ROPE_COS_NAME].isValid = false;
        }
        if ((scenarioInfo_.cacheMode_ != CACHE_MODE::BSND) && (scenarioInfo_.cacheMode_ != CACHE_MODE::TND)) {
            expectedParamInfo_.emplace(CACHE_INDEX_NAME,
                                       std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size});
            expectedParamInfo_[CACHE_INDEX_NAME].dtype = ge::DT_INT64;
        }
        expectedParamInfo_.emplace(QUERY_NAME, std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size,
                                                                     baseShapeInfo_.nSize, baseShapeInfo_.hckvSize});
        expectedParamInfo_.emplace(QUERY_ROPE_NAME, std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size,
                                                                          baseShapeInfo_.nSize, baseShapeInfo_.drSize});
    }
}

void MlaPrologTilingCheck::FillWeightAndNormShapes()
{
    expectedParamInfo_.emplace(WEIGHT_DQ_NAME, std::vector<uint32_t>{baseShapeInfo_.heSize, baseShapeInfo_.hcqSize});
    expectedParamInfo_.emplace(WEIGHT_UQ_QR_NAME,
                               std::vector<uint32_t>{baseShapeInfo_.hcqSize, baseShapeInfo_.headSizeUqQr});
    expectedParamInfo_.emplace(
        WEIGHT_UK_NAME, std::vector<uint32_t>{baseShapeInfo_.nSize, baseShapeInfo_.dSize, baseShapeInfo_.hckvSize});
    expectedParamInfo_.emplace(
        WEIGHT_DKV_KR_NAME,
        std::vector<uint32_t>{baseShapeInfo_.heSize, baseShapeInfo_.hckvSize + baseShapeInfo_.drSize});
    expectedParamInfo_.emplace(RMSNORM_GAMMA_CQ_NAME, std::vector<uint32_t>{baseShapeInfo_.hcqSize});
    expectedParamInfo_.emplace(RMSNORM_GAMMA_CKV_NAME, std::vector<uint32_t>{baseShapeInfo_.hckvSize});
}

void MlaPrologTilingCheck::FillCacheShapes()
{
    if (scenarioInfo_.cacheMode_ == CACHE_MODE::TND) {
        expectedParamInfo_.emplace(KV_CACHE_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.nkvSize,
                                                                        baseShapeInfo_.dtileSize});
        expectedParamInfo_.emplace(
            KR_CACHE_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.nkvSize, baseShapeInfo_.drSize});
    } else if (scenarioInfo_.cacheMode_ == CACHE_MODE::BSND) {
        expectedParamInfo_.emplace(KV_CACHE_NAME,
                                   std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size,
                                                         baseShapeInfo_.nkvSize, baseShapeInfo_.dtileSize});
        expectedParamInfo_.emplace(KR_CACHE_NAME, std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size,
                                                                        baseShapeInfo_.nkvSize, baseShapeInfo_.drSize});
    } else {
        expectedParamInfo_.emplace(KV_CACHE_NAME,
                                   std::vector<uint32_t>{baseShapeInfo_.blockNum, baseShapeInfo_.blockSize,
                                                         baseShapeInfo_.nkvSize, baseShapeInfo_.dtileSize});
        expectedParamInfo_.emplace(KR_CACHE_NAME,
                                   std::vector<uint32_t>{baseShapeInfo_.blockNum, baseShapeInfo_.blockSize,
                                                         baseShapeInfo_.nkvSize, baseShapeInfo_.drSize});
    }
}

void MlaPrologTilingCheck::FillOptionalOutputParamShapeWithDims()
{
    if (std::strncmp(context_.opType, V2_OP_NAME, OP_NAME_LEN) == 0) {
        FillOptionalOutputParamShapeWithDimsV2();
    }

    if (std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        FillOptionalOutputParamShapeWithDimsV3();
    }
}

void MlaPrologTilingCheck::FillOptionalOutputParamShapeWithDimsV2()
{
    if (scenarioInfo_.quantMode_ == QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TENSOR) {
        expectedParamInfo_.emplace(DEQUANT_SCALE_Q_NOPE_NAME,
                                   std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.nSize, 1});
        expectedParamInfo_[DEQUANT_SCALE_Q_NOPE_NAME].dtype = ge::DT_FLOAT;
    } else {
        // 仅校验dequantScaleQNope有传入
        expectedParamInfo_.emplace(DEQUANT_SCALE_Q_NOPE_NAME, context_.dequantScaleQNope);
    }
}

void MlaPrologTilingCheck::FillOptionalOutputParamShapeWithDimsV3()
{
    if (scenarioInfo_.kvQuantMode_ == KV_QUANT_MODE::PER_TENSOR) {
        expectedParamInfo_.emplace(DEQUANT_SCALE_Q_NOPE_NAME,
                                   std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.nSize, 1});
    } else {
        expectedParamInfo_.emplace(DEQUANT_SCALE_Q_NOPE_NAME, std::vector<uint32_t>{0});
    }
    expectedParamInfo_[DEQUANT_SCALE_Q_NOPE_NAME].dtype = ge::DT_FLOAT;

    if (*(context_.queryNormFlag)) {
        if (scenarioInfo_.batchSeqFusedFlag_) {
            expectedParamInfo_.emplace(QUERY_NORM_NAME,
                                       std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.hcqSize});
        } else {
            expectedParamInfo_.emplace(
                QUERY_NORM_NAME,
                std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size, baseShapeInfo_.hcqSize});
        }
        FillQueryNormScaleShape();
    } else {
        expectedParamInfo_.emplace(QUERY_NORM_NAME, std::vector<uint32_t>{0});
        expectedParamInfo_.emplace(DEQUANT_SCALE_Q_NORM_NAME, std::vector<uint32_t>{0});
    }
    FillQueryNormDtypes();
}

void MlaPrologTilingCheck::FillQueryNormScaleShape()
{
    if (scenarioInfo_.weightQuantMode_ == WEIGHT_QUANT_MODE::NO_QUANT) {
        expectedParamInfo_.emplace(DEQUANT_SCALE_Q_NORM_NAME, std::vector<uint32_t>{0});
    } else if (scenarioInfo_.weightQuantMode_ == WEIGHT_QUANT_MODE::MXFP8_FULL_QUANT) {
        expectedParamInfo_.emplace(
            DEQUANT_SCALE_Q_NORM_NAME,
            std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.hcqSize / MXFP8_BLOCK_SIZE});
    } else {
        expectedParamInfo_.emplace(DEQUANT_SCALE_Q_NORM_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize, 1});
    }
}

void MlaPrologTilingCheck::FillQueryNormDtypes()
{
    if (scenarioInfo_.weightQuantMode_ == WEIGHT_QUANT_MODE::NO_QUANT) {
        expectedParamInfo_[QUERY_NORM_NAME].dtype = ge::DT_BF16;
        expectedParamInfo_[DEQUANT_SCALE_Q_NORM_NAME].dtype = ge::DT_FLOAT;
    } else if (scenarioInfo_.weightQuantMode_ == WEIGHT_QUANT_MODE::MXFP8_FULL_QUANT) {
        expectedParamInfo_[QUERY_NORM_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
        expectedParamInfo_[DEQUANT_SCALE_Q_NORM_NAME].dtype = ge::DT_FLOAT8_E8M0;
    } else if (scenarioInfo_.weightQuantMode_ == WEIGHT_QUANT_MODE::FP8_FULL_QUANT) {
        expectedParamInfo_[QUERY_NORM_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
        expectedParamInfo_[DEQUANT_SCALE_Q_NORM_NAME].dtype = ge::DT_FLOAT;
    } else if (scenarioInfo_.weightQuantMode_ == WEIGHT_QUANT_MODE::HIF8_FULL_QUANT) {
        expectedParamInfo_[QUERY_NORM_NAME].dtype = ge::DT_HIFLOAT8;
        expectedParamInfo_[DEQUANT_SCALE_Q_NORM_NAME].dtype = ge::DT_FLOAT;
    } else {
        expectedParamInfo_[QUERY_NORM_NAME].dtype = ge::DT_INT8;
        expectedParamInfo_[DEQUANT_SCALE_Q_NORM_NAME].dtype = ge::DT_FLOAT;
    }
}

void MlaPrologTilingCheck::FillScenarioParamInfo()
{
    using FillFunc = void (MlaPrologTilingCheck::*)();
    static const std::unordered_map<QUANT_MODE, FillFunc> dispatchTable = {
        {QUANT_MODE::NO_QUANT,                              &MlaPrologTilingCheck::FillNonQuantParamInfo},
        {QUANT_MODE::PARTIAL_QUANT_KV_NO_QUANT,             &MlaPrologTilingCheck::FillPartialQuantParamInfo},
        {QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_CHANNEL,    &MlaPrologTilingCheck::FillPartialKVQuantParamInfo},
        {QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_TILE,       &MlaPrologTilingCheck::FillPartialKVPertileQuantParamInfo},
        {QUANT_MODE::FULL_QUANT_KV_NO_QUANT,                &MlaPrologTilingCheck::FillFullQuantParamInfo},
        {QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TENSOR,        &MlaPrologTilingCheck::FillFullKVQuantParamInfo},
        {QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TILE,          &MlaPrologTilingCheck::FillFullKVPertileQuantParamInfo},
        {QUANT_MODE::MXFP8_FULL_QUANT_KV_NO_QUANT,          &MlaPrologTilingCheck::FillMxfp8FullQuantParamInfo},
        {QUANT_MODE::MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR,  &MlaPrologTilingCheck::FillMxfp8FullKVQuantParamInfo},
        {QUANT_MODE::MXFP8_FULL_QUANT_KV_QUANT_PER_TILE,    &MlaPrologTilingCheck::FillMxfp8FullKVPertileParamInfo},
        {QUANT_MODE::FP8_FULL_QUANT_KV_NO_QUANT,            &MlaPrologTilingCheck::FillFP8FullQuantParamInfo},
        {QUANT_MODE::FP8_FULL_QUANT_KV_QUANT_PER_TENSOR,    &MlaPrologTilingCheck::FillFP8FullKVQuantParamInfo},
        {QUANT_MODE::HIF8_FULL_QUANT_KV_NO_QUANT,           &MlaPrologTilingCheck::FillHIF8FullQuantParamInfo},
        {QUANT_MODE::HIF8_FULL_QUANT_KV_QUANT_PER_TENSOR,   &MlaPrologTilingCheck::FillHIF8FullKVQuantParamInfo},
        {QUANT_MODE::FP8_FULL_QUANT_KV_QUANT_PER_TILE,      &MlaPrologTilingCheck::FillFP8FullKVPertileQuantParamInfo},
        {QUANT_MODE::HIF8_FULL_QUANT_KV_QUANT_PER_TILE,     &MlaPrologTilingCheck::FillHIF8FullKVPertileQuantParamInfo}
    };
    auto it = dispatchTable.find(scenarioInfo_.quantMode_);
    if (it != dispatchTable.end()) {
        (this->*(it->second))();
    }
}

void MlaPrologTilingCheck::FillNonQuantParamInfo()
{
    expectedParamInfo_[TOKEN_X_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[WEIGHT_DQ_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[WEIGHT_UQ_QR_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[WEIGHT_UK_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[WEIGHT_DKV_KR_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[RMSNORM_GAMMA_CQ_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[RMSNORM_GAMMA_CKV_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[ROPE_SIN_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[ROPE_COS_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[KR_CACHE_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[QUERY_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[QUERY_ROPE_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[KR_CACHE_OUT_NAME].dtype = ge::DT_BF16;
}

void MlaPrologTilingCheck::FillPartialQuantParamInfo()
{
    FillNonQuantParamInfo();

    expectedParamInfo_.emplace(DEQUANT_SCALE_W_UQ_QR_NAME, std::vector<uint32_t>{1, baseShapeInfo_.headSizeUqQr});
    expectedParamInfo_.emplace(SMOOTH_SCALES_CQ_NAME, std::vector<uint32_t>{1, baseShapeInfo_.hcqSize});

    expectedParamInfo_[WEIGHT_UQ_QR_NAME].dtype = ge::DT_INT8;

    expectedParamInfo_[DEQUANT_SCALE_W_UQ_QR_NAME].dtype = ge::DT_FLOAT;
    expectedParamInfo_[SMOOTH_SCALES_CQ_NAME].dtype = ge::DT_FLOAT;

    expectedParamInfo_[SMOOTH_SCALES_CQ_NAME].isValid = (context_.smoothScalesCq.desc != nullptr);
}

void MlaPrologTilingCheck::FillPartialKVQuantParamInfo()
{
    FillPartialQuantParamInfo();

    expectedParamInfo_.emplace(QUANT_SCALE_CKV_NAME, std::vector<uint32_t>{1, baseShapeInfo_.hckvSize});
    expectedParamInfo_.emplace(QUANT_SCALE_CKR_NAME, std::vector<uint32_t>{1, baseShapeInfo_.drSize});

    expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KR_CACHE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KR_CACHE_OUT_NAME].dtype = ge::DT_INT8;

    expectedParamInfo_[QUANT_SCALE_CKV_NAME].dtype = ge::DT_FLOAT;
    expectedParamInfo_[QUANT_SCALE_CKR_NAME].dtype = ge::DT_FLOAT;
}

void MlaPrologTilingCheck::FillPartialKVPertileQuantParamInfo()
{
    FillPartialQuantParamInfo();

    expectedParamInfo_.emplace(K_NOPE_CLIP_ALPHA_NAME, std::vector<uint32_t>{1});
    expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[K_NOPE_CLIP_ALPHA_NAME].dtype = ge::DT_FLOAT;
}

void MlaPrologTilingCheck::FillFullQuantParamInfo()
{
    FillPartialQuantParamInfo();

    expectedParamInfo_.emplace(DEQUANT_SCALE_X_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize, 1});
    expectedParamInfo_.emplace(DEQUANT_SCALE_W_DQ_NAME, std::vector<uint32_t>{1, baseShapeInfo_.hcqSize});
    expectedParamInfo_.emplace(DEQUANT_SCALE_W_DKV_KR_NAME,
                               std::vector<uint32_t>{1, baseShapeInfo_.hckvSize + baseShapeInfo_.drSize});

    expectedParamInfo_[TOKEN_X_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[WEIGHT_DQ_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[WEIGHT_DKV_KR_NAME].dtype = ge::DT_INT8;

    if (GetCurNpuArch() == NpuArch::DAV_3510 && std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        if (*(context_.weightQuantMode) == static_cast<int>(WEIGHT_QUANT_MODE::FP8_FULL_QUANT)) {
            expectedParamInfo_[TOKEN_X_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
            expectedParamInfo_[WEIGHT_DQ_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
            expectedParamInfo_[WEIGHT_UQ_QR_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
            expectedParamInfo_[WEIGHT_DKV_KR_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
        } else if (*(context_.weightQuantMode) == static_cast<int>(WEIGHT_QUANT_MODE::HIF8_FULL_QUANT)) {
            expectedParamInfo_[TOKEN_X_NAME].dtype = ge::DT_HIFLOAT8;
            expectedParamInfo_[WEIGHT_DQ_NAME].dtype = ge::DT_HIFLOAT8;
            expectedParamInfo_[WEIGHT_UQ_QR_NAME].dtype = ge::DT_HIFLOAT8;
            expectedParamInfo_[WEIGHT_DKV_KR_NAME].dtype = ge::DT_HIFLOAT8;
        }
    }

    expectedParamInfo_[DEQUANT_SCALE_X_NAME].dtype = ge::DT_FLOAT;
    expectedParamInfo_[DEQUANT_SCALE_W_DQ_NAME].dtype = ge::DT_FLOAT;
    expectedParamInfo_[DEQUANT_SCALE_W_DKV_KR_NAME].dtype = ge::DT_FLOAT;
}

void MlaPrologTilingCheck::FillFullKVQuantParamInfo()
{
    FillFullQuantParamInfo();

    expectedParamInfo_[QUERY_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_INT8;
    if (GetCurNpuArch() == NpuArch::DAV_3510 && std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        if (*(context_.weightQuantMode) == static_cast<int>(WEIGHT_QUANT_MODE::FP8_FULL_QUANT)) {
            expectedParamInfo_[QUERY_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
            expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
            expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
        } else if (*(context_.weightQuantMode) == static_cast<int>(WEIGHT_QUANT_MODE::HIF8_FULL_QUANT)) {
            expectedParamInfo_[QUERY_NAME].dtype = ge::DT_HIFLOAT8;
            expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_HIFLOAT8;
            expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_HIFLOAT8;
        }
    }
    if (std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        expectedParamInfo_.emplace(QUANT_SCALE_CKV_NAME, std::vector<uint32_t>{1});
    } else {
        expectedParamInfo_.emplace(QUANT_SCALE_CKV_NAME, std::vector<uint32_t>{1, baseShapeInfo_.hckvSize});
    }

    expectedParamInfo_[QUANT_SCALE_CKV_NAME].dtype = ge::DT_FLOAT;
}

void MlaPrologTilingCheck::FillFullKVPertileQuantParamInfo()
{
    FillFullQuantParamInfo();

    expectedParamInfo_.emplace(K_NOPE_CLIP_ALPHA_NAME, std::vector<uint32_t>{1});
    expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_INT8;
    if (GetCurNpuArch() == NpuArch::DAV_3510 && std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        if (*(context_.weightQuantMode) == static_cast<int>(WEIGHT_QUANT_MODE::FP8_FULL_QUANT)) {
            expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
            expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
        } else if (*(context_.weightQuantMode) == static_cast<int>(WEIGHT_QUANT_MODE::HIF8_FULL_QUANT)) {
            expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_HIFLOAT8;
            expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_HIFLOAT8;
        }
    }
    expectedParamInfo_[K_NOPE_CLIP_ALPHA_NAME].dtype = ge::DT_FLOAT;
}

void MlaPrologTilingCheck::FillMxfp8FullQuantParamInfo()
{
    FillPartialQuantParamInfo();
    // dequantScaleX: (M, K / 32) [BS, He / 32] || [T, He / 32]
    expectedParamInfo_.emplace(DEQUANT_SCALE_X_NAME,
                               std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.heSize / MXFP8_BLOCK_SIZE});
    // dequantScaleWDq: (N, K / 32) [Hcq, He / 32]
    expectedParamInfo_.emplace(DEQUANT_SCALE_W_DQ_NAME,
                               std::vector<uint32_t>{baseShapeInfo_.hcqSize, baseShapeInfo_.heSize / MXFP8_BLOCK_SIZE});
    // dequantScaleWDq: (N, K / 32) [Numhead * (D + Dr), Hcq / 32]
    expectedParamInfo_.erase(DEQUANT_SCALE_W_UQ_QR_NAME);
    expectedParamInfo_.emplace(
        DEQUANT_SCALE_W_UQ_QR_NAME,
        std::vector<uint32_t>{baseShapeInfo_.headSizeUqQr, baseShapeInfo_.hcqSize / MXFP8_BLOCK_SIZE});
    // dequantScaleWDkvKr: (N, K / 32) [Hckv + Dr, He / 32]
    expectedParamInfo_.emplace(DEQUANT_SCALE_W_DKV_KR_NAME,
                               std::vector<uint32_t>{baseShapeInfo_.hckvSize + baseShapeInfo_.drSize,
                                                     baseShapeInfo_.heSize / MXFP8_BLOCK_SIZE});

    expectedParamInfo_[TOKEN_X_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
    expectedParamInfo_[WEIGHT_DQ_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
    expectedParamInfo_[WEIGHT_UQ_QR_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
    expectedParamInfo_[WEIGHT_DKV_KR_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
    expectedParamInfo_[DEQUANT_SCALE_X_NAME].dtype = ge::DT_FLOAT8_E8M0;
    expectedParamInfo_[DEQUANT_SCALE_W_DQ_NAME].dtype = ge::DT_FLOAT8_E8M0;
    expectedParamInfo_[DEQUANT_SCALE_W_UQ_QR_NAME].dtype = ge::DT_FLOAT8_E8M0;
    expectedParamInfo_[DEQUANT_SCALE_W_DKV_KR_NAME].dtype = ge::DT_FLOAT8_E8M0;

    expectedParamInfo_[QUANT_SCALE_CKR_NAME].isValid = false;
    expectedParamInfo_[SMOOTH_SCALES_CQ_NAME].isValid = false;
    expectedParamInfo_[K_NOPE_CLIP_ALPHA_NAME].isValid = false;
}

void MlaPrologTilingCheck::FillMxfp8FullKVQuantParamInfo()
{
    FillMxfp8FullQuantParamInfo();

    expectedParamInfo_.emplace(QUANT_SCALE_CKV_NAME, std::vector<uint32_t>{1});

    expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
    expectedParamInfo_[QUANT_SCALE_CKV_NAME].dtype = ge::DT_FLOAT;
    expectedParamInfo_[QUERY_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
    expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
}

void MlaPrologTilingCheck::FillMxfp8FullKVPertileParamInfo()
{
    FillMxfp8FullQuantParamInfo();

    expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
    expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_FLOAT8_E4M3FN;
}

void MlaPrologTilingCheck::FillFP8FullQuantParamInfo()
{
    FillFullQuantParamInfo();
}

void MlaPrologTilingCheck::FillFP8FullKVQuantParamInfo()
{
    FillFullKVQuantParamInfo();
}

void MlaPrologTilingCheck::FillHIF8FullQuantParamInfo()
{
    FillFullQuantParamInfo();
}

void MlaPrologTilingCheck::FillHIF8FullKVQuantParamInfo()
{
    FillFullKVQuantParamInfo();
}

void MlaPrologTilingCheck::FillFP8FullKVPertileQuantParamInfo()
{
    FillFullKVPertileQuantParamInfo();
    expectedParamInfo_.erase(K_NOPE_CLIP_ALPHA_NAME);
    expectedParamInfo_[K_NOPE_CLIP_ALPHA_NAME].isValid = false;
}

void MlaPrologTilingCheck::FillHIF8FullKVPertileQuantParamInfo()
{
    FillFullKVPertileQuantParamInfo();
    expectedParamInfo_.erase(K_NOPE_CLIP_ALPHA_NAME);
    expectedParamInfo_[K_NOPE_CLIP_ALPHA_NAME].isValid = false;
}

void MlaPrologTilingCheck::GenActualParamInfo()
{
    actualParamInfo_.emplace(TOKEN_X_NAME, context_.tokenX);
    actualParamInfo_.emplace(WEIGHT_DQ_NAME, context_.weightDq);
    actualParamInfo_.emplace(WEIGHT_UQ_QR_NAME, context_.weightUqQr);
    actualParamInfo_.emplace(WEIGHT_UK_NAME, context_.weightUk);
    actualParamInfo_.emplace(WEIGHT_DKV_KR_NAME, context_.weightDkvKr);
    actualParamInfo_.emplace(RMSNORM_GAMMA_CQ_NAME, context_.rmsnormGammaCq);
    actualParamInfo_.emplace(RMSNORM_GAMMA_CKV_NAME, context_.rmsnormGammaCkv);
    actualParamInfo_.emplace(ROPE_SIN_NAME, context_.ropeSin);
    actualParamInfo_.emplace(ROPE_COS_NAME, context_.ropeCos);
    actualParamInfo_.emplace(CACHE_INDEX_NAME, context_.cacheIndex);
    actualParamInfo_.emplace(KV_CACHE_NAME, context_.kvCache);
    actualParamInfo_.emplace(KR_CACHE_NAME, context_.krCache);
    actualParamInfo_.emplace(DEQUANT_SCALE_X_NAME, context_.dequantScaleX);
    actualParamInfo_.emplace(DEQUANT_SCALE_W_DQ_NAME, context_.dequantScaleWDq);
    actualParamInfo_.emplace(DEQUANT_SCALE_W_UQ_QR_NAME, context_.dequantScaleWUqQr);
    actualParamInfo_.emplace(DEQUANT_SCALE_W_DKV_KR_NAME, context_.dequantScaleWDkvKr);
    actualParamInfo_.emplace(QUANT_SCALE_CKV_NAME, context_.quantScaleCkv);
    actualParamInfo_.emplace(QUANT_SCALE_CKR_NAME, context_.quantScaleCkr);
    actualParamInfo_.emplace(SMOOTH_SCALES_CQ_NAME, context_.smoothScalesCq);
    actualParamInfo_.emplace(ACTUAL_SEQ_LEN_NAME, context_.actualSeqLen);
    actualParamInfo_.emplace(K_NOPE_CLIP_ALPHA_NAME, context_.kNopeClipAlpha);
    actualParamInfo_.emplace(QUERY_NAME, context_.query);
    actualParamInfo_.emplace(QUERY_ROPE_NAME, context_.queryRope);
    actualParamInfo_.emplace(KV_CACHE_OUT_NAME, context_.kvCacheOut);
    actualParamInfo_.emplace(KR_CACHE_OUT_NAME, context_.krCacheOut);
    actualParamInfo_.emplace(DEQUANT_SCALE_Q_NOPE_NAME, context_.dequantScaleQNope);
    actualParamInfo_.emplace(QUERY_NORM_NAME, context_.queryNorm);
    actualParamInfo_.emplace(DEQUANT_SCALE_Q_NORM_NAME, context_.dequantScaleQNorm);
    if (std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) == 0 &&
        *(context_.ckvkrRepoMode) == static_cast<int>(CKVKR_REPO_MODE::COMBINE)) {
        actualParamInfo_.erase(KR_CACHE_NAME);
        actualParamInfo_.erase(KR_CACHE_OUT_NAME);
    }
}

ge::graphStatus MlaPrologTilingCheck::CheckCkvkrRepoMode()
{
    ge::graphStatus isCorrect{ge::GRAPH_SUCCESS};
    if (std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) != 0) {
        return isCorrect;
    }
    if (*(context_.ckvkrRepoMode) == static_cast<int>(CKVKR_REPO_MODE::COMBINE)) {
        // 校验所有维度的乘积是否为0
        if (context_.krCache.shape->GetStorageShape().GetShapeSize() != 0) {
            isCorrect = ge::GRAPH_FAILED;
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_.opName, "krCache",
                                                  GetShapeStr(context_.krCache.shape->GetStorageShape()),
                                                  "When ckvkrRepoMode is COMBINE, krCache should be empty tensor");
        }
        if (context_.krCacheOut.shape->GetStorageShape().GetShapeSize() != 0) {
            isCorrect = ge::GRAPH_FAILED;
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_.opName, "krCacheOut",
                                                  GetShapeStr(context_.krCacheOut.shape->GetStorageShape()),
                                                  "When ckvkrRepoMode is COMBINE, krCacheOut should be empty tensor");
        }
    }
    return isCorrect;
}

ge::graphStatus MlaPrologTilingCheck::CheckCacheIndexDim()
{
    if (std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) != 0) {
        return ge::GRAPH_SUCCESS;
    }
    if (!scenarioInfo_.batchSeqFusedFlag_) {
        return ge::GRAPH_SUCCESS;
    }
    if (scenarioInfo_.cacheMode_ != CACHE_MODE::PA_BLK_BSND && scenarioInfo_.cacheMode_ != CACHE_MODE::PA_BLK_NZ) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(context_.cacheIndex.shape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_.opName, "cacheIndex", "null",
                    "When cacheMode is in {PA_BLK_BSND, PA_BLK_NZ}, cacheIndex should not be null"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context_.cacheIndex.shape->GetStorageShape().GetDimNum() != MLA_PROLOG_DIM_NUM_1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_.opName, "cacheIndex",
                    std::to_string(context_.cacheIndex.shape->GetStorageShape().GetDimNum()) + "D",
                    "When cacheMode in {PA_BLK_BSND, PA_BLK_NZ} and tokenX dim is 2, cacheIndex dim should be 1"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTilingCheck::CheckSpecialScenarioParamShape()
{
    if (CheckCkvkrRepoMode() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CheckCacheIndexDim() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTilingCheck::CheckParamByScenario()
{
    GenActualParamInfo();
    GenExpectedParamInfo();
    ge::graphStatus isCorrect{ge::GRAPH_SUCCESS};
    for (const auto &it : actualParamInfo_) {
        const auto &expectedParam{expectedParamInfo_[it.first]};
        if (__builtin_expect((expectedParam != it.second), 0)) {
            isCorrect = ge::GRAPH_FAILED;
            if (expectedParam.isValid != it.second.isValid) {
                if (expectedParam.isValid && !it.second.isValid) {
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_.opName, it.first, "null",
                                                          "this parameter is required under current configuration");
                } else {
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_.opName, it.first, "not null",
                                                          "this parameter is not required under current configuration");
                }
                continue;
            }
            if (expectedParam.dtype != it.second.dtype) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_.opName, it.first, TypeUtils::DataTypeToSerialString(it.second.dtype),
                    "this parameter requires dtype " + TypeUtils::DataTypeToSerialString(expectedParam.dtype) +
                        " under current configuration");
            }
            if (expectedParam.format != it.second.format) {
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                    context_.opName, it.first, std::string(ge::GetFormatName(it.second.format)),
                    "this parameter requires format " + std::string(ge::GetFormatName(expectedParam.format)) +
                        " under current configuration");
            }
            if (expectedParam.shape != it.second.shape) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_.opName, it.first, ConvertContainerToStringV3(it.second.shape),
                    "this parameter requires shape " + ConvertContainerToString(expectedParam.shape) +
                        " under current configuration");
            }
        }
    }
    return isCorrect;
}

ge::graphStatus MlaPrologTilingCheck::CheckScenarParam()
{
    if (std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) != 0) {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus isCorrect{ge::GRAPH_SUCCESS};
    CheckRepoMode(scenarioInfo_.kvQuantMode_ == KV_QUANT_MODE::PER_TILE, isCorrect);
    CheckQueryQuantMode(scenarioInfo_.kvQuantMode_ == KV_QUANT_MODE::PER_TENSOR, isCorrect);
    return isCorrect;
}

void MlaPrologTilingCheck::CheckRepoMode(bool isPertile, ge::graphStatus &isCorrect)
{
    auto expectedCkvkr = isPertile ? CKVKR_REPO_MODE::COMBINE : CKVKR_REPO_MODE::DIVIDE;
    auto expectedQuantScale = isPertile ? QUANT_SCALE_REPO_MODE::COMBINE : QUANT_SCALE_REPO_MODE::DIVIDE;
    std::string desc = isPertile ? "pertile" : "non-pertile";
    std::string name = isPertile ? "COMBINE" : "DIVIDE";

    if (*(context_.ckvkrRepoMode) != static_cast<int>(expectedCkvkr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_.opName, "ckvkrRepoMode",
                                              std::to_string(*(context_.ckvkrRepoMode)),
                                              "When " + desc + " quant mode, must be " + name + "(" +
                                                  std::to_string(static_cast<int>(expectedCkvkr)) + ")");
        isCorrect = ge::GRAPH_FAILED;
    }
    if (*(context_.quantScaleRepoMode) != static_cast<int>(expectedQuantScale)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_.opName, "quantScaleRepoMode",
                                              std::to_string(*(context_.quantScaleRepoMode)),
                                              "When " + desc + " quant mode, must be " + name + "(" +
                                                  std::to_string(static_cast<int>(expectedQuantScale)) + ")");
        isCorrect = ge::GRAPH_FAILED;
    }
}

void MlaPrologTilingCheck::CheckQueryQuantMode(bool isPertensor, ge::graphStatus &isCorrect)
{
    auto expected = isPertensor ? QUERY_QUANT_MODE::PER_TOKEN_HEAD : QUERY_QUANT_MODE::NO_QUANT;
    std::string desc = isPertensor ? "pertensor" : "non-pertensor";
    std::string name = isPertensor ? "PER_TOKEN_HEAD" : "NO_QUANT";

    if (*(context_.queryQuantMode) != static_cast<int>(expected)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_.opName, "queryQuantMode",
                                              std::to_string(*(context_.queryQuantMode)),
                                              "When " + desc + " quant mode, must be " + name + "(" +
                                                  std::to_string(static_cast<int>(expected)) + ")");
        isCorrect = ge::GRAPH_FAILED;
    }
}
// =================================全量参数校验=================================

// ==================================单参数校验==================================
bool MlaPrologTilingCheck::IsSingleParamValid(const BaseParaInfo &param, const std::string &paramName,
                                              const std::set<ge::DataType> &expectedDtype,
                                              const std::set<ge::Format> &expectedFormat,
                                              const std::set<size_t> &expectedDimNum) const
{
    OP_CHECK_IF((param.shape == nullptr) || (param.desc == nullptr),
                OP_LOGE_WITH_INVALID_INPUT(context_.opName, paramName), return false);

    ge::DataType dtype = param.desc->GetDataType();
    OP_CHECK_IF((expectedDtype.find(dtype) == expectedDtype.end()),
                OP_LOGE_FOR_INVALID_DTYPE(context_.opName, paramName, TypeUtils::DataTypeToSerialString(dtype),
                                          ConvertContainerToStringV3(expectedDtype, TypeUtils::DataTypeToSerialString)),
                return false);

    ge::Format format = static_cast<ge::Format>(ge::GetPrimaryFormat(param.desc->GetStorageFormat()));
    OP_CHECK_IF((expectedFormat.find(format) == expectedFormat.end()),
                OP_LOGE_FOR_INVALID_FORMAT(context_.opName, paramName, std::string(ge::GetFormatName(format)),
                                           ConvertContainerToStringV3(expectedFormat, FormatToString)),
                return false);

    size_t dimNum = param.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((expectedDimNum.find(dimNum) == expectedDimNum.end()),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_.opName, paramName, std::to_string(dimNum),
                                             ConvertContainerToStringV3(expectedDimNum)),
                return false);
    return true;
}

ge::graphStatus MlaPrologTilingCheck::CheckSingleRequiredParam() const
{
    if (!CheckTokenX() || !CheckWDq() || !CheckWUqQr() || !CheckWUk() || !CheckWDkvKr() || !CheckRmsnormGammaCq() ||
        !CheckRmsnormGammaCkv() || !CheckRopeSin() || !CheckRopeCos() || !CheckCacheIndex() || !CheckKvCache() ||
        !CheckKrCache() || !CheckActSeqLen()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool MlaPrologTilingCheck::CheckTokenX() const
{
    if (GetCurNpuArch() == NpuArch::DAV_3510) {
        return IsSingleParamValid(context_.tokenX, TOKEN_X_NAME,
                                  {ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8},
                                  {ge::FORMAT_ND, ge::FORMAT_NCHW}, {2, 3});
    } else {
        return IsSingleParamValid(context_.tokenX, TOKEN_X_NAME, {ge::DT_BF16, ge::DT_INT8},
                                  {ge::FORMAT_ND, ge::FORMAT_NCHW}, {2, 3});
    }
}

bool MlaPrologTilingCheck::CheckWDq() const
{
    if (GetCurNpuArch() == NpuArch::DAV_3510) {
        return IsSingleParamValid(context_.weightDq, WEIGHT_DQ_NAME,
                                  {ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8},
                                  {ge::FORMAT_FRACTAL_NZ}, {2, 4});
    } else {
        return IsSingleParamValid(context_.weightDq, WEIGHT_DQ_NAME, {ge::DT_BF16, ge::DT_INT8},
                                  {ge::FORMAT_FRACTAL_NZ}, {2, 4});
    }
}

bool MlaPrologTilingCheck::CheckWUqQr() const
{
    if (GetCurNpuArch() == NpuArch::DAV_3510) {
        return IsSingleParamValid(context_.weightUqQr, WEIGHT_UQ_QR_NAME,
                                  {ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8},
                                  {ge::FORMAT_FRACTAL_NZ}, {2, 4});
    } else {
        return IsSingleParamValid(context_.weightUqQr, WEIGHT_UQ_QR_NAME, {ge::DT_BF16, ge::DT_INT8},
                                  {ge::FORMAT_FRACTAL_NZ}, {2, 4});
    }
}

bool MlaPrologTilingCheck::CheckWUk() const
{
    if (GetCurNpuArch() == NpuArch::DAV_3510) {
        return IsSingleParamValid(context_.weightUk, WEIGHT_UK_NAME, {ge::DT_BF16}, {ge::FORMAT_ND, ge::FORMAT_NCHW},
                                  {3});
    } else {
        return IsSingleParamValid(context_.weightUk, WEIGHT_UK_NAME, {ge::DT_BF16, ge::DT_INT8},
                                  {ge::FORMAT_ND, ge::FORMAT_NCHW}, {3});
    }
}

bool MlaPrologTilingCheck::CheckWDkvKr() const
{
    if (GetCurNpuArch() == NpuArch::DAV_3510) {
        return IsSingleParamValid(context_.weightDkvKr, WEIGHT_DKV_KR_NAME,
                                  {ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8},
                                  {ge::FORMAT_FRACTAL_NZ}, {2, 4});
    } else {
        return IsSingleParamValid(context_.weightDkvKr, WEIGHT_DKV_KR_NAME, {ge::DT_BF16, ge::DT_INT8},
                                  {ge::FORMAT_FRACTAL_NZ}, {2, 4});
    }
}

bool MlaPrologTilingCheck::CheckRmsnormGammaCq() const
{
    return IsSingleParamValid(context_.rmsnormGammaCq, RMSNORM_GAMMA_CQ_NAME, {ge::DT_BF16},
                              {ge::FORMAT_ND, ge::FORMAT_NCHW}, {1});
}

bool MlaPrologTilingCheck::CheckRmsnormGammaCkv() const
{
    return IsSingleParamValid(context_.rmsnormGammaCkv, RMSNORM_GAMMA_CKV_NAME, {ge::DT_BF16},
                              {ge::FORMAT_ND, ge::FORMAT_NCHW}, {1});
}

bool MlaPrologTilingCheck::CheckRopeSin() const
{
    // V3: rope may be null when do_rope=false.
    if (std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) == 0 && context_.doRope != nullptr &&
        !*(context_.doRope)) {
        OP_CHECK_IF(context_.ropeSin.shape != nullptr || context_.ropeSin.desc != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_.opName, "ropeSin", "non-null",
                                                         "ropeSin must be null when do_rope is false"),
                    return false);
        return true;
    }
    return IsSingleParamValid(context_.ropeSin, ROPE_SIN_NAME, {ge::DT_BF16}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {2, 3});
}

bool MlaPrologTilingCheck::CheckRopeCos() const
{
    if (std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) == 0 && context_.doRope != nullptr &&
        !*(context_.doRope)) {
        OP_CHECK_IF(context_.ropeCos.shape != nullptr || context_.ropeCos.desc != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_.opName, "ropeCos", "non-null",
                                                         "ropeCos must be null when do_rope is false"),
                    return false);
        return true;
    }
    return IsSingleParamValid(context_.ropeCos, ROPE_COS_NAME, {ge::DT_BF16}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {2, 3});
}

bool MlaPrologTilingCheck::CheckCacheIndex() const
{
    return std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) == 0 ||
           IsSingleParamValid(context_.cacheIndex, CACHE_INDEX_NAME, {ge::DT_INT64}, {ge::FORMAT_ND, ge::FORMAT_NCHW},
                              {1, 2});
}

bool MlaPrologTilingCheck::CheckKvCache() const
{
    if (GetCurNpuArch() == NpuArch::DAV_3510) {
        if (std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) != 0) {
            return IsSingleParamValid(context_.kvCache, KV_CACHE_NAME, {ge::DT_BF16, ge::DT_INT8},
                                      {ge::FORMAT_ND, ge::FORMAT_NCHW}, {4});
        } else {
            return IsSingleParamValid(context_.kvCache, KV_CACHE_NAME,
                                      {ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8},
                                      {ge::FORMAT_ND, ge::FORMAT_NCHW}, {3, 4});
        }
    } else {
        if (std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) != 0) {
            return IsSingleParamValid(context_.kvCache, KV_CACHE_NAME, {ge::DT_BF16, ge::DT_INT8},
                                      {ge::FORMAT_ND, ge::FORMAT_NCHW}, {4});
        } else {
            return IsSingleParamValid(context_.kvCache, KV_CACHE_NAME, {ge::DT_BF16, ge::DT_INT8},
                                      {ge::FORMAT_ND, ge::FORMAT_NCHW}, {3, 4});
        }
    }
}

bool MlaPrologTilingCheck::CheckKrCache() const
{
    if (GetCurNpuArch() == NpuArch::DAV_3510) {
        return IsSingleParamValid(context_.krCache, KR_CACHE_NAME, {ge::DT_BF16, ge::DT_INT8},
                                  {ge::FORMAT_ND, ge::FORMAT_NCHW}, {1, 3, 4});
    } else {
        return (std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) == 0 &&
                *(context_.ckvkrRepoMode) == static_cast<int>(CKVKR_REPO_MODE::COMBINE)) ||
               IsSingleParamValid(context_.krCache, KR_CACHE_NAME, {ge::DT_BF16, ge::DT_INT8},
                                  {ge::FORMAT_ND, ge::FORMAT_NCHW}, {1, 3, 4});
    }
}

bool MlaPrologTilingCheck::CheckActSeqLen() const
{
    if (context_.actualSeqLen.desc == nullptr) {
        return true;
    };
    ge::DataType dtype = context_.actualSeqLen.desc->GetDataType();
    OP_CHECK_IF((ge::DT_INT32 != dtype),
                OP_LOGE_FOR_INVALID_DTYPE(context_.opName, "actualSeqLen", TypeUtils::DataTypeToSerialString(dtype),
                                          TypeUtils::DataTypeToSerialString(ge::DT_INT32)),
                return false);
    return true;
}

bool MlaPrologTilingCheck::CheckCacheModeParamShape() const
{
    if (std::strncmp(context_.cacheMode, CACHE_MODE_TND, CACHE_MODE_LEN) == 0) {
        OP_CHECK_IF(context_.tokenX.shape->GetStorageShape().GetDimNum() != MLA_PROLOG_DIM_NUM_2,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context_.opName, "tokenX",
                        std::to_string(context_.tokenX.shape->GetStorageShape().GetDimNum()) + "D",
                        "When cacheMode is TND, tokenX dim must be 2"),
                    return false);
        OP_CHECK_IF(context_.kvCache.shape->GetStorageShape().GetDimNum() != MLA_PROLOG_DIM_NUM_3,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context_.opName, "kvCache",
                        std::to_string(context_.kvCache.shape->GetStorageShape().GetDimNum()) + "D",
                        "When cacheMode is TND, kvCache dim must be 3"),
                    return false);
    } else if (std::strncmp(context_.cacheMode, CACHE_MODE_BSND, CACHE_MODE_LEN) == 0) {
        OP_CHECK_IF(context_.tokenX.shape->GetStorageShape().GetDimNum() != MLA_PROLOG_DIM_NUM_3,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context_.opName, "tokenX",
                        std::to_string(context_.tokenX.shape->GetStorageShape().GetDimNum()) + "D",
                        "When cacheMode is BSND, tokenX dim must be 3"),
                    return false);
        OP_CHECK_IF(context_.kvCache.shape->GetStorageShape().GetDimNum() != MLA_PROLOG_DIM_NUM_4,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context_.opName, "kvCache",
                        std::to_string(context_.kvCache.shape->GetStorageShape().GetDimNum()) + "D",
                        "When cacheMode is BSND, kvCache dim must be 4"),
                    return false);
    } else {
        OP_CHECK_IF(context_.kvCache.shape->GetStorageShape().GetDimNum() != MLA_PROLOG_DIM_NUM_4,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context_.opName, "kvCache",
                        std::to_string(context_.kvCache.shape->GetStorageShape().GetDimNum()) + "D",
                        "When cacheMode in {PA_BSND, PA_NZ, PA_BLK_BSND, PA_BLK_NZ}, kvCache dim must be 4"),
                    return false);
    }
    return true;
}

ge::graphStatus MlaPrologTilingCheck::CheckCacheMode() const
{
    if (std::strncmp(context_.opType, V3_OP_NAME, OP_NAME_LEN) != 0) {
        if (std::strncmp(context_.cacheMode, CACHE_MODE_PA_BSND, CACHE_MODE_LEN) != 0 &&
            std::strncmp(context_.cacheMode, CACHE_MODE_PA_NZ, CACHE_MODE_LEN) != 0) {
            OP_LOGE_FOR_INVALID_VALUE(context_.opName, "cacheMode", std::string(context_.cacheMode),
                                      "{PA_BSND, PA_NZ}");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    if ((std::strncmp(context_.cacheMode, CACHE_MODE_BSND, CACHE_MODE_LEN) != 0) &&
        (std::strncmp(context_.cacheMode, CACHE_MODE_TND, CACHE_MODE_LEN) != 0) &&
        (std::strncmp(context_.cacheMode, CACHE_MODE_PA_BSND, CACHE_MODE_LEN) != 0) &&
        (std::strncmp(context_.cacheMode, CACHE_MODE_PA_NZ, CACHE_MODE_LEN) != 0) &&
        (std::strncmp(context_.cacheMode, CACHE_MODE_PA_BLK_BSND, CACHE_MODE_LEN) != 0) &&
        (std::strncmp(context_.cacheMode, CACHE_MODE_PA_BLK_NZ, CACHE_MODE_LEN) != 0)) {
        OP_LOGE_FOR_INVALID_VALUE(context_.opName, "cacheMode", std::string(context_.cacheMode),
                                  "{BSND, TND, PA_BSND, PA_NZ, PA_BLK_BSND, PA_BLK_NZ}");
        return ge::GRAPH_FAILED;
    }
    if (!CheckCacheModeParamShape()) {
        return ge::GRAPH_FAILED;
    }

    if (*(context_.kvQuantMode) != static_cast<int>(KV_QUANT_MODE::PER_TILE)) {
        return ge::GRAPH_SUCCESS;
    }

    if ((std::strncmp(context_.cacheMode, CACHE_MODE_PA_NZ, CACHE_MODE_LEN) == 0) ||
        (std::strncmp(context_.cacheMode, CACHE_MODE_PA_BLK_BSND, CACHE_MODE_LEN) == 0) ||
        (std::strncmp(context_.cacheMode, CACHE_MODE_PA_BLK_NZ, CACHE_MODE_LEN) == 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_.opName, "cacheMode", std::string(context_.cacheMode),
            "When pertile quant mode, cacheMode cannot be {PA_NZ, PA_BLK_BSND, PA_BLK_NZ}");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ==================================单参数校验==================================
} // namespace optiling