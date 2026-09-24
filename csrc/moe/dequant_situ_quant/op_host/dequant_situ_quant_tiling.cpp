/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dequant_situ_quant_tiling.cpp
 * \brief
 */

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <string>
#include "tiling/tiling_api.h"
#include "dequant_situ_quant_tiling.h"
#include "../../dequant_swiglu_quant/tiling_base/tiling_util.h"
#include "../../dequant_swiglu_quant/tiling_base/tiling_templates_registry.h"

#define CHECK_FAIL(cont, cond, ...)                      \
    do {                                                 \
        if (cond) {                                      \
            OP_LOGE(cont->GetNodeName(), ##__VA_ARGS__); \
            return ge::GRAPH_FAILED;                     \
        }                                                \
    } while (0)

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
using Ops::NN::Optiling::TilingRegistry;

constexpr uint32_t UB_RESERVED_BUFF = 1024;
constexpr uint32_t ALIGN_UINT_IN_CACHE_32B = 32;
constexpr uint32_t PACK_UINT_IN_CACHE_512B = 512;
constexpr uint32_t DEFAULT_BUFFER_NUM = 2;
constexpr uint32_t MAX_CORE_NUMBER = 64;
constexpr uint32_t PERFORMANCE_COL_LEN = 1536;
constexpr uint32_t PERFORMANCE_ROW_LEN = 128;
constexpr uint32_t MIN_CORE = 12;
constexpr uint64_t USER_WORKSPACE = 16777216; // 16 * 1024 * 1024
constexpr uint64_t BLOCK_BYTES = 32;

constexpr size_t INDEX_IN_X = 0;
constexpr size_t INDEX_IN_WEIGHT_SCALE = 1;
constexpr size_t INDEX_IN_ACTIVATION_SCALE = 2;
constexpr size_t INDEX_IN_BIAS = 3;
constexpr size_t INDEX_IN_QUANT_SCALE = 4;
constexpr size_t INDEX_IN_QUANT_OFFSET = 5;
constexpr size_t INDEX_IN_GROUP_INDEX = 6;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t QUANT_MODE_STATIC = 0;
constexpr int64_t QUANT_MODE_DYNAMIC = 1;

void DequantSituQuantTiling::Reset()
{
    opName = nullptr;
    totalCore = 0;
    totalUsedCoreNum = 0;
    inputDTypeLen = 1;
    ubMinBlockLen = 32;
    cacheLineLen = 512;
    maxTileLen = 0;
    optBaseRowLen = 0;
    optBaseColLen = 0;
    workspaceSize_ = 0;
    hasDequantBias = false;
    hasQuantScale = false;
    hasQuantOffset = false;
    quantIsOne = false;
    activateLeft = 0;
    quantMode = 0;
    beta = 4.0f;
    linearBeta = 25.0f;
    quantScaleShapeSize = 0;
    inDimx = 0;
    inDimy = 0;
    outDimy = 0;
    xDtype_ = ge::DT_INT8;
    isPreDequantized_ = false;
    hasWeightScale_ = false;
    hasActivationScale_ = false;
    hasGroupIndex_ = false;
    expertNum_ = 1;
}

bool DequantSituQuantTiling::IsCapable()
{
    if (Ops::NN::OpTiling::IsRegbaseSocVersion(context_)) {
        return false;
    }
    return true;
}

ge::graphStatus DequantSituQuantTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    totalCore = ascendcPlatform.GetCoreNumAiv();
    aicoreParams_.numBlocks = totalCore;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;
    socVersion = ascendcPlatform.GetSocVersion();
    return ge::GRAPH_SUCCESS;
}

bool DequantSituQuantTiling::SetAttrs(const gert::RuntimeAttrs* attrs)
{
    auto betaPtr = attrs->GetAttrPointer<float>(0);
    beta = (betaPtr == nullptr) ? 4.0f : *betaPtr;
    OP_CHECK_IF(beta == 0.0f, OP_LOGE(context_->GetNodeName(), "beta must not be 0"), return false);

    auto linearBetaPtr = attrs->GetAttrPointer<float>(1);
    linearBeta = (linearBetaPtr == nullptr) ? 25.0f : *linearBetaPtr;

    auto isActivateLeftPtr = attrs->GetBool(2);
    bool isActivateLeft = isActivateLeftPtr == nullptr ? true : *isActivateLeftPtr;
    activateLeft = (isActivateLeft ? 1 : 0);

    auto str = attrs->GetStr(3);
    OP_CHECK_IF(str == nullptr, OP_LOGE(context_->GetNodeName(), "quant_mode attr is null"), return false);
    std::string quantModeAttr{str};
    std::transform(quantModeAttr.begin(), quantModeAttr.end(), quantModeAttr.begin(), ::tolower);
    OP_CHECK_IF((quantModeAttr != "static") && (quantModeAttr != "dynamic"),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "quant_mode", quantModeAttr.c_str(),
                                                      "quant_mode should be static or dynamic with case insensitive"),
                return false);
    quantMode = ((quantModeAttr == "static") ? QUANT_MODE_STATIC : QUANT_MODE_DYNAMIC);

    tilingData.set_activateLeft(activateLeft);
    tilingData.set_beta(beta);
    tilingData.set_linearBeta(linearBeta);
    tilingData.set_quantMode(quantMode);
    return true;
}

ge::graphStatus DequantSituQuantTiling::CheckInputShapesInt8(int64_t dimNum, int64_t inDimy, int64_t outDimy)
{
    CHECK_FAIL(context_, dimNum <= 1, "The shape dim of x can not be less than 2");

    // weight_scale: required, shape [inDimy] or [1]
    auto weightScaleShapePtr = context_->GetOptionalInputShape(INDEX_IN_WEIGHT_SCALE);
    OP_CHECK_IF(weightScaleShapePtr == nullptr,
                OP_LOGE(context_->GetNodeName(), "weight_scale is required for int8 x"),
                return ge::GRAPH_FAILED);
    uint64_t weightScaleSize = weightScaleShapePtr->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(weightScaleSize != static_cast<uint64_t>(inDimy) && weightScaleSize != 1,
                OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "weight_scale",
                                              std::to_string(weightScaleSize).c_str(),
                                              (std::to_string(inDimy) + " or 1").c_str()),
                return ge::GRAPH_FAILED);

    // activation_scale: must be absent for int8
    OP_CHECK_IF(context_->GetOptionalInputShape(INDEX_IN_ACTIVATION_SCALE) != nullptr,
                OP_LOGE(context_->GetNodeName(), "activation_scale must be absent for int8 x"),
                return ge::GRAPH_FAILED);

    // bias: optional, shape [inDimy] or [1]
    auto biasShapePtr = context_->GetOptionalInputShape(INDEX_IN_BIAS);
    hasDequantBias = (biasShapePtr != nullptr);
    tilingData.set_dequantBiasIsEmpty(!hasDequantBias);
    if (hasDequantBias) {
        uint64_t biasSize = biasShapePtr->GetStorageShape().GetShapeSize();
        OP_CHECK_IF(biasSize != static_cast<uint64_t>(inDimy) && biasSize != 1,
                    OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "bias",
                                                  std::to_string(biasSize).c_str(),
                                                  (std::to_string(inDimy) + " or 1").c_str()),
                    return ge::GRAPH_FAILED);
    }

    // quant_scale: optional, shape [outDimy] or [1]
    auto quantScaleShapePtr = context_->GetOptionalInputShape(INDEX_IN_QUANT_SCALE);
    hasQuantScale = (quantScaleShapePtr != nullptr);
    tilingData.set_quantScaleIsEmpty(!hasQuantScale);
    if (hasQuantScale) {
        quantScaleShapeSize = quantScaleShapePtr->GetStorageShape().GetShapeSize();
        OP_CHECK_IF(quantScaleShapeSize != static_cast<uint64_t>(outDimy) && quantScaleShapeSize != 1,
                    OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "quant_scale",
                                                  std::to_string(quantScaleShapeSize).c_str(),
                                                  (std::to_string(outDimy) + " or 1").c_str()),
                    return ge::GRAPH_FAILED);
        quantIsOne = (quantScaleShapeSize == 1);
    } else {
        quantIsOne = false;
    }
    tilingData.set_quantIsOne(quantIsOne);

    // quant_offset: optional (only static), shape [outDimy] or [1]
    auto quantOffsetShapePtr = context_->GetOptionalInputShape(INDEX_IN_QUANT_OFFSET);
    hasQuantOffset = (quantOffsetShapePtr != nullptr);
    tilingData.set_quantOffsetIsEmpty(!hasQuantOffset);
    if (quantMode == QUANT_MODE_STATIC) {
        OP_CHECK_IF(!hasQuantScale,
                    OP_LOGE(context_->GetNodeName(), "quant_scale must be provided when quant_mode is static"),
                    return ge::GRAPH_FAILED);
        if (hasQuantOffset) {
            uint64_t quantOffsetSize = quantOffsetShapePtr->GetStorageShape().GetShapeSize();
            OP_CHECK_IF(quantOffsetSize != static_cast<uint64_t>(outDimy) && quantOffsetSize != 1,
                        OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "quant_offset",
                                                      std::to_string(quantOffsetSize).c_str(),
                                                      (std::to_string(outDimy) + " or 1").c_str()),
                        return ge::GRAPH_FAILED);
        }
    }

    // group_index: must be absent for int8
    OP_CHECK_IF(context_->GetOptionalInputShape(INDEX_IN_GROUP_INDEX) != nullptr,
                OP_LOGE(context_->GetNodeName(), "group_index must be absent for int8 x"),
                return ge::GRAPH_FAILED);

    // check output shapes
    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    const gert::Shape yShape = yShapePtr->GetStorageShape();
    OP_CHECK_IF(yShape.GetDimNum() != static_cast<size_t>(dimNum),
                OP_LOGE(context_->GetNodeName(), "y shape dim must equal to x shape dim"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(yShape.GetDim(dimNum - 1) != outDimy,
                OP_LOGE(context_->GetNodeName(), "y last dim must be x last dim / 2"), return ge::GRAPH_FAILED);

    auto scaleShapePtr = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleShapePtr);
    const gert::Shape scaleShape = scaleShapePtr->GetStorageShape();
    OP_CHECK_IF(static_cast<uint64_t>(scaleShape.GetShapeSize()) != static_cast<uint64_t>(inDimx),
                OP_LOGE(context_->GetNodeName(), "scale shape size must equal to rowLen"), return ge::GRAPH_FAILED);

    tilingData.set_expertNum(1);
    tilingData.set_hasGroupIndex(0);
    tilingData.set_hasActivationScale(0);
    tilingData.set_isPreDequantized(0);
    tilingData.set_inputWidth(0);
    tilingData.set_outputWidth(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSituQuantTiling::ValidateInt32Contract()
{
    // quant_scale/quant_offset must be absent
    OP_CHECK_IF(context_->GetOptionalInputShape(INDEX_IN_QUANT_SCALE) != nullptr,
                OP_LOGE(context_->GetNodeName(), "quant_scale is not supported for int32 x"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOptionalInputShape(INDEX_IN_QUANT_OFFSET) != nullptr,
                OP_LOGE(context_->GetNodeName(), "quant_offset is not supported for int32 x"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(quantMode != QUANT_MODE_DYNAMIC,
                OP_LOGE(context_->GetNodeName(), "quant_mode must be dynamic for int32 x"),
                return ge::GRAPH_FAILED);

    // weight_scale: required
    auto weightScaleShapePtr = context_->GetOptionalInputShape(INDEX_IN_WEIGHT_SCALE);
    OP_CHECK_IF(weightScaleShapePtr == nullptr,
                OP_LOGE(context_->GetNodeName(), "weight_scale is required for int32 x"),
                return ge::GRAPH_FAILED);

    // activation_scale: required
    auto activationScaleShapePtr = context_->GetOptionalInputShape(INDEX_IN_ACTIVATION_SCALE);
    OP_CHECK_IF(activationScaleShapePtr == nullptr,
                OP_LOGE(context_->GetNodeName(), "activation_scale is required for int32 x"),
                return ge::GRAPH_FAILED);

    // group_index: optional
    auto groupIndexShapePtr = context_->GetOptionalInputShape(INDEX_IN_GROUP_INDEX);
    hasGroupIndex_ = (groupIndexShapePtr != nullptr);

    expertNum_ = 1;
    if (hasGroupIndex_) {
        const gert::Shape groupIndexShape = groupIndexShapePtr->GetStorageShape();
        OP_CHECK_IF(groupIndexShape.GetDimNum() != 1 || groupIndexShape.GetDim(0) <= 0,
                    OP_LOGE(context_->GetNodeName(), "group_index must be a non-empty 1-D tensor"),
                    return ge::GRAPH_FAILED);
        expertNum_ = static_cast<uint32_t>(groupIndexShape.GetDim(0));
    }

    // weight_scale shape validation
    const gert::Shape weightScaleShape = weightScaleShapePtr->GetStorageShape();
    const bool weightShapeValid = hasGroupIndex_ ?
        (weightScaleShape.GetDimNum() == 2 &&
         weightScaleShape.GetDim(0) == static_cast<int64_t>(expertNum_) &&
         weightScaleShape.GetDim(1) == inDimy) :
        ((weightScaleShape.GetDimNum() == 1 && weightScaleShape.GetDim(0) == inDimy) ||
         (weightScaleShape.GetDimNum() == 2 && weightScaleShape.GetDim(0) == 1 &&
          weightScaleShape.GetDim(1) == inDimy));
    OP_CHECK_IF(!weightShapeValid,
                OP_LOGE(context_->GetNodeName(), "weight_scale shape does not match the group contract"),
                return ge::GRAPH_FAILED);

    // activation_scale shape: [rows] or [rows, 1]
    const gert::Shape activationScaleShape = activationScaleShapePtr->GetStorageShape();
    const bool actScaleValid = (activationScaleShape.GetDimNum() == 1 ||
                                (activationScaleShape.GetDimNum() == 2 && activationScaleShape.GetDim(1) == 1)) &&
                               activationScaleShape.GetShapeSize() == inDimx;
    OP_CHECK_IF(!actScaleValid,
                OP_LOGE(context_->GetNodeName(), "activation_scale must contain one FP32 value per row"),
                return ge::GRAPH_FAILED);

    // bias: optional, same shape as weight_scale
    auto biasShapePtr = context_->GetOptionalInputShape(INDEX_IN_BIAS);
    hasDequantBias = (biasShapePtr != nullptr);
    tilingData.set_dequantBiasIsEmpty(!hasDequantBias);
    if (hasDequantBias) {
        const gert::Shape biasShape = biasShapePtr->GetStorageShape();
        const bool biasShapeValid = hasGroupIndex_ ?
            (biasShape.GetDimNum() == 2 && biasShape.GetDim(0) == static_cast<int64_t>(expertNum_) &&
             biasShape.GetDim(1) == inDimy) :
            ((biasShape.GetDimNum() == 1 && biasShape.GetDim(0) == inDimy) ||
             (biasShape.GetDimNum() == 2 && biasShape.GetDim(0) == 1 && biasShape.GetDim(1) == inDimy));
        OP_CHECK_IF(!biasShapeValid,
                    OP_LOGE(context_->GetNodeName(), "bias shape must match weight_scale"),
                    return ge::GRAPH_FAILED);
    }

    // check output shapes
    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    const gert::Shape yShape = yShapePtr->GetStorageShape();
    OP_CHECK_IF(yShape.GetDimNum() != 2 || yShape.GetDim(0) != inDimx || yShape.GetDim(1) != outDimy,
                OP_LOGE(context_->GetNodeName(), "y shape must be [rows, input_width / 2]"),
                return ge::GRAPH_FAILED);

    auto scaleShapePtr = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleShapePtr);
    const gert::Shape scaleShape = scaleShapePtr->GetStorageShape();
    OP_CHECK_IF(scaleShape.GetDimNum() != 1 || scaleShape.GetDim(0) != inDimx,
                OP_LOGE(context_->GetNodeName(), "scale shape must be [rows]"),
                return ge::GRAPH_FAILED);

    // UB capacity check
    const uint64_t inputBytes = static_cast<uint64_t>(inDimy) * sizeof(int32_t);
    const uint64_t paramBytes = static_cast<uint64_t>(inDimy) * sizeof(float);
    const uint64_t tempBytes = static_cast<uint64_t>(inDimy) * sizeof(float);
    const uint64_t outputBytes = static_cast<uint64_t>(outDimy) * sizeof(int8_t) + BLOCK_BYTES;
    const uint64_t requiredUb = inputBytes + paramBytes + (hasDequantBias ? paramBytes : 0) +
                                tempBytes + outputBytes + UB_RESERVED_BUFF;
    OP_CHECK_IF(aicoreParams_.ubSize < requiredUb,
                OP_LOGE(context_->GetNodeName(), "UB size %lu is smaller than required %lu bytes",
                        aicoreParams_.ubSize, requiredUb),
                return ge::GRAPH_FAILED);

    tilingData.set_expertNum(expertNum_);
    tilingData.set_hasGroupIndex(hasGroupIndex_ ? 1 : 0);
    tilingData.set_hasActivationScale(1);
    tilingData.set_isPreDequantized(0);
    tilingData.set_inputWidth(static_cast<uint32_t>(inDimy));
    tilingData.set_outputWidth(static_cast<uint32_t>(outDimy));
    tilingData.set_quantScaleIsEmpty(1);
    tilingData.set_quantOffsetIsEmpty(1);
    tilingData.set_quantIsOne(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSituQuantTiling::CheckInputShapesBF16()
{
    OP_CHECK_IF(quantMode != QUANT_MODE_DYNAMIC,
                OP_LOGE(context_->GetNodeName(), "quant_mode must be dynamic for bfloat16 x"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context_->GetOptionalInputShape(INDEX_IN_WEIGHT_SCALE) != nullptr,
                OP_LOGE(context_->GetNodeName(), "weight_scale must be absent for bfloat16 x"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOptionalInputShape(INDEX_IN_ACTIVATION_SCALE) != nullptr,
                OP_LOGE(context_->GetNodeName(), "activation_scale must be absent for bfloat16 x"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOptionalInputShape(INDEX_IN_BIAS) != nullptr,
                OP_LOGE(context_->GetNodeName(), "bias must be absent for bfloat16 x"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOptionalInputShape(INDEX_IN_QUANT_SCALE) != nullptr,
                OP_LOGE(context_->GetNodeName(), "quant_scale must be absent for bfloat16 x"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOptionalInputShape(INDEX_IN_QUANT_OFFSET) != nullptr,
                OP_LOGE(context_->GetNodeName(), "quant_offset must be absent for bfloat16 x"),
                return ge::GRAPH_FAILED);
    auto groupIndexShapePtr = context_->GetOptionalInputShape(INDEX_IN_GROUP_INDEX);
    hasGroupIndex_ = (groupIndexShapePtr != nullptr);
    expertNum_ = 1;
    if (hasGroupIndex_) {
        const gert::Shape groupIndexShape = groupIndexShapePtr->GetStorageShape();
        OP_CHECK_IF(groupIndexShape.GetDimNum() != 1 || groupIndexShape.GetDim(0) <= 0,
                    OP_LOGE(context_->GetNodeName(), "group_index must be a non-empty 1-D tensor"),
                    return ge::GRAPH_FAILED);
        expertNum_ = static_cast<uint32_t>(groupIndexShape.GetDim(0));
    }

    // check output shapes
    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    const gert::Shape yShape = yShapePtr->GetStorageShape();
    OP_CHECK_IF(yShape.GetDimNum() != 2 || yShape.GetDim(0) != inDimx || yShape.GetDim(1) != outDimy,
                OP_LOGE(context_->GetNodeName(), "y shape must be [rows, input_width / 2]"),
                return ge::GRAPH_FAILED);

    auto scaleShapePtr = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleShapePtr);
    const gert::Shape scaleShape = scaleShapePtr->GetStorageShape();
    OP_CHECK_IF(scaleShape.GetDimNum() != 1 || scaleShape.GetDim(0) != inDimx,
                OP_LOGE(context_->GetNodeName(), "scale shape must be [rows]"),
                return ge::GRAPH_FAILED);

    // UB capacity check
    const uint64_t inputBytes = static_cast<uint64_t>(inDimy) * 2; // BF16 = 2 bytes
    const uint64_t tempBytes = static_cast<uint64_t>(inDimy) * sizeof(float);
    const uint64_t dequantBytes = tempBytes;
    const uint64_t outputBytes = static_cast<uint64_t>(outDimy) * sizeof(int8_t) + BLOCK_BYTES;
    const uint64_t requiredUb = inputBytes + tempBytes + dequantBytes + outputBytes + UB_RESERVED_BUFF;
    OP_CHECK_IF(aicoreParams_.ubSize < requiredUb,
                OP_LOGE(context_->GetNodeName(), "UB size %lu is smaller than required %lu bytes",
                        aicoreParams_.ubSize, requiredUb),
                return ge::GRAPH_FAILED);

    hasDequantBias = false;
    tilingData.set_dequantBiasIsEmpty(1);
    tilingData.set_quantScaleIsEmpty(1);
    tilingData.set_quantOffsetIsEmpty(1);
    tilingData.set_quantIsOne(0);
    tilingData.set_expertNum(expertNum_);
    tilingData.set_hasGroupIndex(hasGroupIndex_ ? 1 : 0);
    tilingData.set_hasActivationScale(0);
    tilingData.set_isPreDequantized(1);
    tilingData.set_inputWidth(static_cast<uint32_t>(inDimy));
    tilingData.set_outputWidth(static_cast<uint32_t>(outDimy));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSituQuantTiling::CheckInputShapes()
{
    auto xShapePtr = context_->GetInputShape(INDEX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    const gert::Shape xShape = xShapePtr->GetStorageShape();
    int64_t dimNum = xShape.GetDimNum();

    int64_t shapeBefore = 1;
    for (int64_t i = 0; i < dimNum - 1; i++) {
        shapeBefore *= xShape.GetDim(i);
    }
    inDimy = xShape.GetDim(dimNum - 1);
    CHECK_FAIL(context_, inDimy % NUM_TWO != 0, "The last dim of x must be even number");

    inDimx = shapeBefore;
    outDimy = inDimy / NUM_TWO;

    tilingData.set_rowLen(inDimx);
    tilingData.set_colLen(outDimy);

    if (xDtype_ == ge::DT_INT8) {
        return CheckInputShapesInt8(dimNum, inDimy, outDimy);
    } else if (xDtype_ == ge::DT_INT32) {
        CHECK_FAIL(context_, dimNum != 2, "x shape rank must be 2 for int32, but is %ld", dimNum);
        return ValidateInt32Contract();
    } else if (xDtype_ == ge::DT_BF16) {
        CHECK_FAIL(context_, dimNum != 2, "x shape rank must be 2 for bfloat16, but is %ld", dimNum);
        return CheckInputShapesBF16();
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus DequantSituQuantTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSituQuantTiling::GetShapeAttrsInfoInner()
{
    opName = context_->GetNodeName();

    auto inputDesc = context_->GetInputDesc(INDEX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    xDtype_ = inputDesc->GetDataType();
    OP_CHECK_IF(xDtype_ != ge::DT_INT8 && xDtype_ != ge::DT_INT32 && xDtype_ != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x",
                                          ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str(),
                                          "int8/int32/bfloat16"),
                return ge::GRAPH_FAILED);

    isPreDequantized_ = (xDtype_ == ge::DT_BF16);

    if (xDtype_ == ge::DT_INT8) {
        inputDTypeLen = 1;
    } else if (xDtype_ == ge::DT_INT32) {
        inputDTypeLen = 4;
    } else {
        inputDTypeLen = 2;
    }

    const gert::RuntimeAttrs* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    if (!SetAttrs(attrs)) {
        return ge::GRAPH_FAILED;
    }

    if (CheckInputShapes() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

bool DequantSituQuantTiling::CalcUbMaxTileLen(const uint64_t ubSize, uint32_t& maxTileLen)
{
    uint64_t bytesPerElement = 8 + 2 + 16 + 16 + 4; // 46 (weightScale + inQueueX + tmpBuf + castBuf + outQueue)
    if (hasDequantBias) {
        bytesPerElement += 8;
    }
    if (hasQuantScale && !quantIsOne) {
        bytesPerElement += (quantMode == QUANT_MODE_DYNAMIC) ? 4 : 8;
    }

    uint64_t availableUb = ubSize - UB_RESERVED_BUFF - ALIGN_UINT_IN_CACHE_32B;
    uint64_t maxElements = availableUb / bytesPerElement;
    maxTileLen = static_cast<uint32_t>(maxElements / ALIGN_UINT_IN_CACHE_32B * ALIGN_UINT_IN_CACHE_32B);
    OP_LOGI(opName, "CalcUbMaxTileLen ubSize:%lu, maxTileLen:%u, bytesPerElement:%lu", ubSize, maxTileLen,
            bytesPerElement);
    return true;
}

bool DequantSituQuantTiling::CalcOptBaseShape(uint32_t maxTileLen)
{
    uint32_t colLen = static_cast<uint32_t>(outDimy);
    uint32_t rowLen = static_cast<uint32_t>(inDimx);

    uint32_t baseColLen = std::min(colLen, maxTileLen);
    if (baseColLen < colLen && baseColLen > cacheLineLen) {
        baseColLen = baseColLen / cacheLineLen * cacheLineLen;
    }
    if (baseColLen == 0) {
        baseColLen = std::min(colLen, static_cast<uint32_t>(ALIGN_UINT_IN_CACHE_32B));
    }

    uint32_t baseRowLen = 1;

    optBaseRowLen = baseRowLen;
    optBaseColLen = baseColLen;

    totalUsedCoreNum = std::min(rowLen, static_cast<uint32_t>(totalCore));
    if (colLen < PERFORMANCE_COL_LEN && rowLen < PERFORMANCE_ROW_LEN) {
        totalUsedCoreNum = std::min(totalUsedCoreNum, static_cast<uint32_t>(MIN_CORE));
    }
    totalUsedCoreNum = std::min(totalUsedCoreNum, static_cast<uint32_t>(MAX_CORE_NUMBER));

    return true;
}

bool DequantSituQuantTiling::CalcTiling(const uint32_t totalCores, const uint64_t ubSize)
{
    ubMinBlockLen = ALIGN_UINT_IN_CACHE_32B / inputDTypeLen;
    cacheLineLen = PACK_UINT_IN_CACHE_512B / inputDTypeLen;

    if (xDtype_ == ge::DT_INT32 || xDtype_ == ge::DT_BF16) {
        // INT32/BF16 path: no column tiling, whole row in UB
        tilingData.set_is32BAligned(1);
        tilingData.set_isDoubleBuffer(1);
        tilingData.set_baseRowLen(1);
        tilingData.set_baseColLen(static_cast<uint32_t>(outDimy));
        totalUsedCoreNum = inDimx == 0 ? 0 : std::min(static_cast<uint32_t>(inDimx), totalCore);
        tilingData.set_usedCoreNum(totalUsedCoreNum);
        return true;
    }

    // INT8 path
    tilingData.set_is32BAligned(static_cast<uint32_t>(outDimy % ubMinBlockLen == 0));
    tilingData.set_isDoubleBuffer(1);

    if (!CalcUbMaxTileLen(ubSize, maxTileLen)) {
        return false;
    }
    if (!CalcOptBaseShape(maxTileLen)) {
        return false;
    }

    tilingData.set_baseRowLen(optBaseRowLen);
    tilingData.set_baseColLen(optBaseColLen);
    tilingData.set_usedCoreNum(totalUsedCoreNum);
    return true;
}

ge::graphStatus DequantSituQuantTiling::DoOpTiling()
{
    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (!CalcTiling(totalCore, aicoreParams_.ubSize)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSituQuantTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t DequantSituQuantTiling::GetTilingKey() const
{
    if (xDtype_ == ge::DT_INT32) {
        return DSQ_INT32_DYNAMIC;
    }
    if (xDtype_ == ge::DT_BF16) {
        return DSQ_BF16_DYNAMIC;
    }
    // INT8 path
    if (quantMode == QUANT_MODE_STATIC) {
        if (quantIsOne) {
            return hasDequantBias ? DSQ_STATIC_QUANT_ONE_BIAS : DSQ_STATIC_QUANT_ONE;
        } else {
            return hasDequantBias ? DSQ_STATIC_QUANT_VEC_BIAS : DSQ_STATIC_QUANT_VEC;
        }
    } else {
        if (hasQuantScale) {
            return hasDequantBias ? DSQ_DYNAMIC_QUANT_SMOOTH_BIAS : DSQ_DYNAMIC_QUANT_SMOOTH;
        } else {
            return hasDequantBias ? DSQ_DYNAMIC_QUANT_NO_SMOOTH_BIAS : DSQ_DYNAMIC_QUANT_NO_SMOOTH;
        }
    }
}

ge::graphStatus DequantSituQuantTiling::GetWorkspaceSize()
{
    if (xDtype_ == ge::DT_INT32 || xDtype_ == ge::DT_BF16) {
        workspaceSize_ = 0;
        return ge::GRAPH_SUCCESS;
    }
    workspaceSize_ = USER_WORKSPACE;
    if (quantMode == QUANT_MODE_DYNAMIC && (static_cast<uint64_t>(outDimy) > optBaseColLen)) {
        workspaceSize_ += (totalUsedCoreNum * static_cast<uint64_t>(outDimy) * sizeof(float));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSituQuantTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    if (xDtype_ == ge::DT_INT32 || xDtype_ == ge::DT_BF16) {
        context_->SetBlockDim(totalUsedCoreNum == 0 ? 1 : totalUsedCoreNum);
    } else {
        context_->SetBlockDim(totalUsedCoreNum);
    }
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void DequantSituQuantTiling::ShowTilingData()
{
    std::ostringstream info;
    info << "rowLen: " << tilingData.get_rowLen();
    info << ", colLen: " << tilingData.get_colLen();
    info << ", baseRowLen: " << tilingData.get_baseRowLen();
    info << ", baseColLen: " << tilingData.get_baseColLen();
    info << ", usedCoreNum: " << tilingData.get_usedCoreNum();
    info << ", quantMode: " << tilingData.get_quantMode();
    info << ", quantIsOne: " << tilingData.get_quantIsOne();
    info << ", beta: " << tilingData.get_beta();
    info << ", linearBeta: " << tilingData.get_linearBeta();
    info << ", dequantBiasIsEmpty: " << tilingData.get_dequantBiasIsEmpty();
    info << ", expertNum: " << tilingData.get_expertNum();
    info << ", hasGroupIndex: " << tilingData.get_hasGroupIndex();
    info << ", hasActivationScale: " << tilingData.get_hasActivationScale();
    info << ", isPreDequantized: " << tilingData.get_isPreDequantized();
    info << ", inputWidth: " << tilingData.get_inputWidth();
    info << ", outputWidth: " << tilingData.get_outputWidth();
    OP_LOGI(opName, "%s", info.str().c_str());
}

REGISTER_TILING_TEMPLATE("DequantSituQuant", DequantSituQuantTiling, 0);

ge::graphStatus TilingForDequantSituQuant(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForDequantSituQuant(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4DequantSituQuant enter.");
    auto compileInfo = context->GetCompiledInfo<DequantSituQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
                OP_LOGE(context->GetNodeName(), "Get core num failed, core num: %u",
                        static_cast<uint32_t>(compileInfo->coreNum)),
                return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF((compileInfo->ubSize <= 0),
                OP_LOGE(context->GetNodeName(), "Get ub size failed, ub size: %u",
                        static_cast<uint32_t>(compileInfo->ubSize)),
                return ge::GRAPH_FAILED);

    OP_LOGD(context, "TilingPrepare4DequantSituQuant exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DequantSituQuant)
    .Tiling(TilingForDequantSituQuant)
    .TilingParse<DequantSituQuantCompileInfo>(TilingPrepareForDequantSituQuant);

} // namespace optiling
