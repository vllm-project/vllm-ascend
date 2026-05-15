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
 * \file dequant_swiglu_quant_tiling.cpp
 * \brief
 */

#include "dequant_swiglu_quant_tiling.h"
#include "../tiling_base/tiling_util.h"
#include "swi_glu_tiling.h"
#include "../tiling_base/tiling_templates_registry.h"

using Ops::NN::Optiling::TilingRegistry;
// using namespace AscendC;
using namespace ge;
namespace optiling
{
constexpr int64_t ATTR_ACTIVATE_LEFT_INDEX = 0;
constexpr int64_t ATTR_QUANT_MODE_INDEX = 1;
constexpr int64_t X_INDEX = 0;
constexpr int64_t WEIGHT_SCALE_INDEX = 1;
constexpr int64_t ACTIVATION_SCALE_INDEX = 2;
constexpr int64_t BIAS_INDEX = 3;
constexpr int64_t QUANT_SCALE_INDEX = 4;
constexpr int64_t QUANT_OFFSET_INDEX = 5;
constexpr int64_t INPUT_GROUP_INDEX = 6;
constexpr int64_t Y_INDEX = 0;
// attr index for SwiGLU used by GPT-OSS
constexpr int64_t SWIGLU_MODE_INDEX = 5;
constexpr int64_t CLAMP_LIMIT_INDEX = 6;
constexpr int64_t GLU_ALPHA_INDEX = 7;
constexpr int64_t GLU_BIAS_INDEX = 8;

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t BLOCK_ELEM = BLOCK_SIZE / static_cast<int64_t>(sizeof(float));
constexpr uint64_t WORKSPACE_SIZE = 32;
// define tiling key offset
constexpr uint64_t TILING_KEY_HAS_GROUP = 100000000;
constexpr uint64_t TILING_KEY_NO_GROUP = 200000000;
// define cut by group
constexpr uint64_t TILING_KEY_CUT_GROUP = 10000000;
constexpr int64_t CUT_GROUP_LARGE_THAN_64 = 64;
constexpr int64_t CUT_GROUP_LARGE_THAN_32 = 32;
constexpr int64_t EACH_GROUP_TOKEN_LESS_THAN = 16;

// quant_scale tiling offset
constexpr uint64_t TILING_KEY_QS_DTYPE = 100;
// bias tiling offset
constexpr uint64_t TILING_KEY_BIAS_DTYPE = 1000;

constexpr int64_t UB_RESERVE = 1024;
constexpr int64_t SWI_FACTOR = 2;
constexpr int64_t QUANT_MODE_DYNAMIC = 1;
constexpr int64_t PERFORMANCE_H_2048 = 2048;
constexpr int64_t PERFORMANCE_H_4096 = 4096;
constexpr int64_t PERFORMANCE_CORE_NUM = 36;
constexpr int64_t PERFORMANCE_UB_FACTOR = static_cast<int64_t>(4096) * 4;

constexpr int QUANT_SCALE_DTYPE_BF16 = 2;
constexpr int QUANT_SCALE_DTYPE_FP32 = 0;
constexpr int QUANT_SCALE_DTYPE_FP16 = 1;

constexpr int BIAS_DTYPE_BF16 = 0;
constexpr int BIAS_DTYPE_FP16 = 1;
constexpr int BIAS_DTYPE_FP32 = 2;
constexpr int BIAS_DTYPE_INT32 = 3;

constexpr int DIM_SIZE_2 = 2;

constexpr float CLAMP_LIMIT_DEFAULT= 7.0;
constexpr float GLU_ALPHA_DEFAULT = 1.702;
constexpr float GLU_BIAS_DEFAULT = 1.0;

static const std::set<ge::DataType> SUPPORT_DTYPE = {ge::DT_INT32, ge::DT_BF16};
static const std::map<std::string, int64_t> SUPPORT_QUANT_MODE = {{"dynamic", 1},{"static", 0}};

bool DequantSwigluQuantDskTiling::CheckOptionalShapeExisting(const gert::StorageShape* storageShape){
  if(storageShape == nullptr){
    return false;
  }
  int64_t shapeSize = storageShape->GetOriginShape().GetShapeSize();
  if(shapeSize <= 0){
    return false;
  }
  return true;
}

ge::graphStatus DequantSwigluQuantDskTiling::GetPlatformInfo() {
  auto platformInfo = context_->GetPlatformInfo();
  if (platformInfo == nullptr) {
    auto compileInfoPtr = context_->GetCompileInfo<DequantSwigluQuantCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"),
                    return ge::GRAPH_FAILED);
    coreNum_ = compileInfoPtr->coreNum;
    ubSize_ = compileInfoPtr->ubSize;
  } else {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = ubSizePlatForm;
    socVersion = ascendcPlatform.GetSocVersion();
  }

  maxPreCore_ = static_cast<int64_t>(coreNum_);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::CheckXAndGroupIndexDtype() {
  auto xPtr = context_->GetInputDesc(X_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, xPtr);
  auto xDtype = xPtr->GetDataType();
  OP_CHECK_IF((SUPPORT_DTYPE.find(xDtype) == SUPPORT_DTYPE.end()),
                  OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x",
                      ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(), "int32 or bfloat16"),
                  return ge::GRAPH_FAILED);
  tilingData_.set_groupIndexDtype(-1);
  if (hasGroupIndex_) {
    auto groupIndexPtr = context_->GetOptionalInputDesc(INPUT_GROUP_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, groupIndexPtr);
    auto groupIndexDtype = groupIndexPtr->GetDataType();
    bool dtypeInValid = groupIndexDtype != ge::DT_INT64;
    OP_CHECK_IF(
        dtypeInValid,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "group_index",
            ge::TypeUtils::DataTypeToSerialString(groupIndexDtype).c_str(), "int64"),
        return ge::GRAPH_FAILED);
    tilingData_.set_groupIndexDtype(1);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::CheckBias() {
  auto biasShapePtr = context_->GetOptionalInputShape(BIAS_INDEX);
  if (biasShapePtr != nullptr) {
    hasBias_ = true;
    OP_CHECK_IF(CheckScaleShapeWithDim(BIAS_INDEX, inDimy_, "bias") != ge::GRAPH_SUCCESS,
              OP_LOGE(context_->GetNodeName(), "bias shape check failed."),
              return ge::GRAPH_FAILED);
  }
  else {
    hasBias_ = false;
  }

  auto biasPtr = context_->GetOptionalInputDesc(BIAS_INDEX);
  if (biasPtr != nullptr && hasBias_ == true) {
    auto biasDtype = biasPtr->GetDataType();
    bool dtypeInValid = (biasDtype != ge::DT_INT32 && biasDtype != ge::DT_FLOAT && biasDtype != ge::DT_FLOAT16 && biasDtype != ge::DT_BF16);
    OP_CHECK_IF(
      dtypeInValid,
      OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "bias",
          ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str(), "bf16, fp16, float or int32"),
      return ge::GRAPH_FAILED);
    if (biasDtype == ge::DT_BF16) {
      tilingData_.set_biasDtype(BIAS_DTYPE_BF16);
    } else if (biasDtype == ge::DT_FLOAT16) {
      tilingData_.set_biasDtype(BIAS_DTYPE_FP16);
    } else if (biasDtype == ge::DT_FLOAT) {
      tilingData_.set_biasDtype(BIAS_DTYPE_FP32);
    } else if (biasDtype == ge::DT_INT32) {
      tilingData_.set_biasDtype(BIAS_DTYPE_INT32);
    }
  }
  else {
    tilingData_.set_biasDtype(0);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::CheckWeightScale() {
  auto weightScalePtr = context_->GetOptionalInputDesc(WEIGHT_SCALE_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, weightScalePtr);
  auto weightScaleDtype = weightScalePtr->GetDataType();
  bool dtypeInValid = weightScaleDtype != ge::DT_FLOAT;
  OP_CHECK_IF(dtypeInValid,
                  OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "weight_scale",
                      ge::TypeUtils::DataTypeToSerialString(weightScaleDtype).c_str(), "float32"),
                  return ge::GRAPH_FAILED);
  OP_CHECK_IF(CheckScaleShapeWithDim(WEIGHT_SCALE_INDEX, inDimy_, "weight_scale") != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "weight scale shape check failed."),
                return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::CheckActivationScale() {
  auto activationScaleShapePtr = context_->GetOptionalInputShape(ACTIVATION_SCALE_INDEX);
  if(CheckOptionalShapeExisting(activationScaleShapePtr)) {
    auto activationScalePtr = context_->GetOptionalInputDesc(ACTIVATION_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, activationScalePtr);
    auto activationScaleDtype = activationScalePtr->GetDataType();
    bool dtypeInValid = activationScaleDtype != ge::DT_FLOAT;

    OP_CHECK_IF(dtypeInValid,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "activation_scale",
                        ge::TypeUtils::DataTypeToSerialString(activationScaleDtype).c_str(), "float32"),
                    return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, activationScaleShapePtr);
    auto activationScaleShape = activationScaleShapePtr->GetStorageShape();
    int64_t activationScaleNum = activationScaleShape.GetShapeSize();

    OP_CHECK_IF(
        activationScaleNum != inDimx_,
        OP_LOGE(
            context_->GetNodeName(),
            "activation_scale num(%ld) must be equal to the tokens num(%ld), please check.",
            activationScaleNum, inDimx_),
        return ge::GRAPH_FAILED);
    tilingData_.set_activationScaleIsEmpty(0);
  }
  else {
    tilingData_.set_activationScaleIsEmpty(1);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::CheckForDequant() {
  // check weight scale, activation scale and bias
  auto xPtr = context_->GetInputDesc(X_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, xPtr);
  auto xDtype = xPtr->GetDataType();
  if (xDtype == ge::DT_INT32) {
    OP_CHECK_IF(CheckWeightScale() != ge::GRAPH_SUCCESS,
          OP_LOGE(context_->GetNodeName(), "weight scale check failed."),
          return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckActivationScale() != ge::GRAPH_SUCCESS,
          OP_LOGE(context_->GetNodeName(), "activation scale check failed."),
          return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckBias() != ge::GRAPH_SUCCESS,
          OP_LOGE(context_->GetNodeName(), "bias check failed."),
          return ge::GRAPH_FAILED);
  }

  if (xDtype == ge::DT_BF16 && hasGroupIndex_) {
    auto shapeGroupIndex = context_->GetOptionalInputShape(INPUT_GROUP_INDEX);
    const gert::Shape& inputShapeGroupIndex = shapeGroupIndex->GetStorageShape();
    OP_CHECK_IF(inputShapeGroupIndex.GetDimNum() != 1,
                    OP_LOGE(context_->GetNodeName(),
                                                    "groupIndex only support 1D Tensor now, please check."),
                    return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::CheckForDynamicQuant() {
  auto offsetPtr = context_->GetOptionalInputShape(QUANT_OFFSET_INDEX);
  OP_CHECK_IF(offsetPtr != nullptr,
                OP_LOGE(context_->GetNodeName(),
                                                "quantOffSet only support None in dynamic quantization of group mode now, please check."),
                return ge::GRAPH_FAILED);

  OP_CHECK_IF(CheckScaleShapeWithDim(QUANT_SCALE_INDEX, outDimy_, "quant_scale") != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "quant scale shape check failed."),
            return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::CheckForStaticQuant() {
  // check quantOffset dtype
  auto quantOffsetDescPtr = context_->GetOptionalInputDesc(QUANT_OFFSET_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, quantOffsetDescPtr);
  auto quantScaleDescPtr = context_->GetOptionalInputDesc(QUANT_SCALE_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, quantScaleDescPtr);
  auto quantOffsetDtype = quantOffsetDescPtr->GetDataType();
  auto quantScaleDtype = quantScaleDescPtr->GetDataType();
  OP_CHECK_IF(quantOffsetDtype != quantScaleDtype,
                  OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                      context_->GetNodeName(), "quant_offset and quant_scale",
                      (ge::TypeUtils::DataTypeToSerialString(quantOffsetDtype) + " and " +
                       ge::TypeUtils::DataTypeToSerialString(quantScaleDtype)).c_str(),
                      "quantOffset dtype must be same as quantScale dtype"),
                  return ge::GRAPH_FAILED);

  int64_t quantScaleColLen = 0;
  int64_t quantOffsetColLen = 0;
  OP_CHECK_IF(CheckStaticQuantShape(QUANT_SCALE_INDEX, quantScaleColLen, "quant_scale") != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "quant scale shape check failed."),
        return ge::GRAPH_FAILED);
  OP_CHECK_IF(CheckStaticQuantShape(QUANT_OFFSET_INDEX, quantOffsetColLen, "quant_offset") != ge::GRAPH_SUCCESS,
    OP_LOGE(context_->GetNodeName(), "quant offset shape check failed."),
    return ge::GRAPH_FAILED);

  OP_CHECK_IF(quantScaleColLen != quantOffsetColLen,
          OP_LOGE(context_->GetNodeName(), "quant offset shape is different from quant scale."),
          return ge::GRAPH_FAILED);
  if(quantScaleColLen == 1){
    tilingData_.set_quantIsOne(1);
  }
  else {
    tilingData_.set_quantIsOne(0);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::CheckForQuant() {
  // check and set quant scale dtype
  OP_CHECK_IF(CheckQuantScaleDtype() != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "Check QuantScale Dtype failed."),
            return ge::GRAPH_FAILED);

  // check quant offset and quant scale shape in dynamic scenario
  if(quantMode_ == QUANT_MODE_DYNAMIC){
    OP_CHECK_IF(CheckForDynamicQuant() != ge::GRAPH_SUCCESS,
          OP_LOGE(context_->GetNodeName(), "Check For Dynamic Quant failed."),
          return ge::GRAPH_FAILED);
  }
  // // check quant offset and quant scale shape in static scenario
  else {
    OP_CHECK_IF(CheckForStaticQuant() != ge::GRAPH_SUCCESS,
          OP_LOGE(context_->GetNodeName(), "Check For Static Quant failed."),
          return ge::GRAPH_FAILED);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::CheckQuantScaleDtype() {
  bool dtypeInValid = false;

  auto quantScaleShapePtr = context_->GetOptionalInputShape(QUANT_SCALE_INDEX);
  if (quantScaleShapePtr == nullptr) {
    tilingData_.set_quantScaleDtype(0);
    tilingData_.set_needSmoothScale(0);
  } else {
    auto quantScalePtr = context_->GetOptionalInputDesc(QUANT_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, quantScalePtr);
    tilingData_.set_needSmoothScale(1);
    auto quantScaleDtype = quantScalePtr->GetDataType();
    dtypeInValid =
        quantScaleDtype != ge::DT_FLOAT && quantScaleDtype != ge::DT_FLOAT16 && quantScaleDtype != ge::DT_BF16;
    OP_CHECK_IF(
        dtypeInValid,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "quant_scale",
            ge::TypeUtils::DataTypeToSerialString(quantScaleDtype).c_str(), "float32, float16 or bfloat16"),
        return ge::GRAPH_FAILED);
    tilingData_.set_quantScaleDtype(QUANT_SCALE_DTYPE_BF16);
    if (quantScaleDtype == ge::DT_FLOAT) {
      tilingData_.set_quantScaleDtype(QUANT_SCALE_DTYPE_FP32);
    } else if (quantScaleDtype == ge::DT_FLOAT16) {
      tilingData_.set_quantScaleDtype(QUANT_SCALE_DTYPE_FP16);
    }
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::GetAttr() {
  auto* attrs = context_->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

  auto* attrActivateLeft = attrs->GetAttrPointer<bool>(ATTR_ACTIVATE_LEFT_INDEX);
  actRight_ = (attrActivateLeft == nullptr || *attrActivateLeft == false) ? 1 : 0;
  std::string quantMode = attrs->GetAttrPointer<char>(ATTR_QUANT_MODE_INDEX);
  auto it = SUPPORT_QUANT_MODE.find(quantMode);
  OP_CHECK_IF(it == SUPPORT_QUANT_MODE.end(),
                  OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                      context_->GetNodeName(), "quant_mode",
                      quantMode.c_str(), "quant_mode only support dynamic(1) and static(0) currently"),
                  return ge::GRAPH_FAILED);
  quantMode_ = it->second;

  auto* swigluMode = attrs->GetAttrPointer<int>(SWIGLU_MODE_INDEX);
  auto* clampLimit = attrs->GetAttrPointer<float>(CLAMP_LIMIT_INDEX);
  auto* gluAlpha = attrs->GetAttrPointer<float>(GLU_ALPHA_INDEX);
  auto* gluBias = attrs->GetAttrPointer<float>(GLU_BIAS_INDEX);

  swigluMode_ = swigluMode == nullptr ? 0 : *swigluMode;
  clampLimit_ = clampLimit == nullptr ? CLAMP_LIMIT_DEFAULT : *clampLimit;
  gluAlpha_ = gluAlpha == nullptr ? GLU_ALPHA_DEFAULT : *gluAlpha;
  gluBias_ = gluBias == nullptr ? GLU_BIAS_DEFAULT : *gluBias;

  OP_CHECK_IF(swigluMode_ != 0 && swigluMode_ != 1,
                  OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                      context_->GetNodeName(), "swigluMode",
                      std::to_string(swigluMode_).c_str(), "swigluMode only support 0 or 1"),
                  return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::CheckScaleShapeWithDim(const int64_t scaleInputIdx,
                                                                    const int64_t expectDim,
                                                                    const char* paramName) {
  auto scalePtr = context_->GetOptionalInputShape(scaleInputIdx);
  if (scalePtr == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  auto scaleShape = scalePtr->GetStorageShape();
  OP_CHECK_IF(scaleShape.GetDimNum() < 1,
                  OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), paramName,
                      std::to_string(scaleShape.GetDimNum()).c_str(), "greater than or equal to 1"),
                  return ge::GRAPH_FAILED);
  OP_CHECK_IF(scaleShape.GetDim(scaleShape.GetDimNum() - 1) != expectDim,
                  OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), paramName,
                      Ops::Base::ToString(scaleShape).c_str(),
                      std::to_string(expectDim).c_str()),
                  return ge::GRAPH_FAILED);
  if (groupNum_ > 1) {
    // check with group index
    OP_CHECK_IF(
        scaleShape.GetDimNum() != DIM_SIZE_2,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), paramName,
            std::to_string(scaleShape.GetDimNum()).c_str(), "2D"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        scaleShape.GetDim(0) != groupNum_,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), paramName, Ops::Base::ToString(scaleShape).c_str(),
            ("the first dimension of " + std::string(paramName) + " (" + std::to_string(scaleShape.GetDim(0)) +
             ") must be equal to the first dimension of group_index (" + std::to_string(groupNum_) + ")")
                .c_str()),
        return ge::GRAPH_FAILED);
  } else {
    OP_CHECK_IF(
        scaleShape.GetDimNum() > DIM_SIZE_2,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), paramName,
            std::to_string(scaleShape.GetDimNum()).c_str(), "less than or equal to 2"),
        return ge::GRAPH_FAILED);
    int64_t groupNumFromScale = scaleShape.GetDimNum() <= 1 ? 1 : scaleShape.GetDim(0);
    OP_CHECK_IF(
        groupNumFromScale != 1,
        OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), paramName,
            Ops::Base::ToString(scaleShape).c_str(), 
            ("[1," + std::to_string(expectDim) + "] or [" + std::to_string(expectDim) + "]").c_str()),
        return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::CheckStaticQuantShape(const int64_t quantInputIdx, int64_t& colLen, const char* paramName) {
  // check quant scale and quant offset shape
  auto quantPtr = context_->GetOptionalInputShape(quantInputIdx);
  if(quantPtr == nullptr){
    return ge::GRAPH_SUCCESS;
  }
  auto quantShape = quantPtr->GetStorageShape();
  OP_CHECK_IF(quantShape.GetDimNum() < 1,
                  OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), paramName,
                      std::to_string(quantShape.GetDimNum()).c_str(), "greater than or equal to 1"),
                  return ge::GRAPH_FAILED);
  colLen = quantShape.GetDim(quantShape.GetDimNum() - 1);
  if(quantShape.GetDimNum() == 1){
    OP_CHECK_IF(colLen != groupNum_,
              OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), paramName,
                  Ops::Base::ToString(quantShape).c_str(),
                  ("[" + std::to_string(groupNum_) + ", ] or [" +
                   std::to_string(groupNum_) + ", " + std::to_string(outDimy_) + "]").c_str()),
              return ge::GRAPH_FAILED);
    colLen = 1;
  }
  else {
    OP_CHECK_IF(colLen != outDimy_ || quantShape.GetDim(0) != groupNum_,
        OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), paramName,
            Ops::Base::ToString(quantShape).c_str(),
            ("[" + std::to_string(groupNum_) + ", ] or [" +
             std::to_string(groupNum_) + ", " + std::to_string(outDimy_) + "]").c_str()),
        return ge::GRAPH_FAILED);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::GetShapeAttrsInfo() {
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::CheckIllegalParam() {
  // if hasbias, speGroupType_ must be false
  if (hasBias_) {
      OP_CHECK_IF(speGroupType_ == true,
                      OP_LOGE(context_->GetNodeName(), "speGroupType_ only support false when using bias"),
                      return ge::GRAPH_FAILED);
  }

  // if swigluMode is 1, speGroupType_ must be false
  if (swigluMode_) {
      OP_CHECK_IF(speGroupType_ == true,
                  OP_LOGE(context_->GetNodeName(), "speGroupType_ only support false when swiglu mode is 1"),
                  return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::GetShapeAttrsInfoInner() {
  if (!IsPerformanceAndGroupIndexBrach()) {
    return ge::GRAPH_SUCCESS;
  }
  // get 2H from x, get H from y, check if 2H can be divided by 64
  auto shapeX = context_->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context_, shapeX);
  const gert::Shape& inputShapeX = shapeX->GetStorageShape();
  int64_t inputShapeXTotalNum = inputShapeX.GetShapeSize();
  int64_t inputShapeXRank = inputShapeX.GetDimNum();
  inDimy_ = inputShapeX.GetDim(inputShapeXRank - 1);
  inDimx_ = inputShapeXTotalNum / inDimy_;
  auto shapeY = context_->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context_, shapeY);
  const gert::Shape& outputShapeY = shapeY->GetStorageShape();
  outDimy_ = outputShapeY.GetDim(inputShapeXRank - 1);
  OP_CHECK_IF(inDimy_ % (BLOCK_SIZE * SWI_FACTOR) != 0,
                  OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
                      std::to_string(inDimy_).c_str(),
                      "lastdimSize of x must be divisible by 64"),
                  return ge::GRAPH_FAILED);

  // set the relevant param of group, hasGroupIndex_, groupNum_ and speGroupType_
  auto shapeGroupIndex = context_->GetOptionalInputShape(INPUT_GROUP_INDEX);
  hasGroupIndex_ = shapeGroupIndex != nullptr;
  groupNum_ = 0;
  speGroupType_ = false;
  if (hasGroupIndex_) {
    const gert::Shape& inputShapeGroupIndex = shapeGroupIndex->GetStorageShape();
    groupNum_ = inputShapeGroupIndex.GetDimNum() == 0 ? 1 : inputShapeGroupIndex.GetDim(0);
    speGroupType_ = inputShapeGroupIndex.GetDimNum() == DIM_SIZE_2;
  }

  OP_CHECK_IF(CheckXAndGroupIndexDtype() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "dtype check failed."),
                return ge::GRAPH_FAILED);

  OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "get attr failed."),
                return ge::GRAPH_FAILED);

  OP_CHECK_IF(CheckForDequant() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "check for dequant failed."),
                return ge::GRAPH_FAILED);

  OP_CHECK_IF(CheckForQuant() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "check for quant failed."),
                return ge::GRAPH_FAILED);

  OP_CHECK_IF(CheckIllegalParam() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "check illegal param failed."),
                return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

bool DequantSwigluQuantDskTiling::IsPerformanceAndGroupIndexBrach() {
  auto shapeGroupIndex = context_->GetOptionalInputShape(INPUT_GROUP_INDEX);
  if (shapeGroupIndex != nullptr) {
    return true;
  }
  return false;
}

bool DequantSwigluQuantDskTiling::IsCapable() {
  if (Ops::NN::OpTiling::IsRegbaseSocVersion(context_)) {
    return false;
  }
  return IsPerformanceAndGroupIndexBrach();
}

void DequantSwigluQuantDskTiling::CountTilingKey() {
  auto xPtr = context_->GetInputDesc(X_INDEX);
  auto xDtype = xPtr->GetDataType();
  tilingKey_ = hasGroupIndex_ ? TILING_KEY_HAS_GROUP : TILING_KEY_NO_GROUP;
  // add quant scale offet to tilingKey_
  tilingKey_ += TILING_KEY_QS_DTYPE * tilingData_.get_quantScaleDtype();
  // add bias offset to tilingKey_
  tilingKey_ += TILING_KEY_BIAS_DTYPE * tilingData_.get_biasDtype();
  // tiling based on groupnum, pre cut num by coreNum_ and total tokens
  bool cond1 = speGroupType_ &&
               (groupNum_ >= CUT_GROUP_LARGE_THAN_64) &&
               (inDimx_ / groupNum_ <= EACH_GROUP_TOKEN_LESS_THAN);
  bool cond2 = !speGroupType_ &&
               (groupNum_ >= CUT_GROUP_LARGE_THAN_32) &&
               (inDimx_ / groupNum_ <= EACH_GROUP_TOKEN_LESS_THAN) &&
               !tilingData_.get_biasDtype() && !tilingData_.get_quantScaleDtype() &&
               (xDtype == ge::DT_INT32);
  if (cond1 || cond2) {
    tilingKey_ += TILING_KEY_CUT_GROUP;
    maxPreCore_ = std::min(static_cast<int64_t>(coreNum_), static_cast<int64_t>(inDimx_));
  }
}

ge::graphStatus DequantSwigluQuantDskTiling::CountMaxDim(int64_t& ubFactorDimx) {
  /*
  x used mem: [UbFactorDimx, outDimy_ * 2] dtype: float
  activation_scale used mem: [UbFactorDimx, 8] dtype: float
  weight_scale used mem: [1, outDimy_ * 2] dtype: float
  quant_scale used mem: [1, outDimy_] dtype: float
  y used mem: [UbFactorDimx, outDimy_] dtype: int8_t
  scale used mem: [UbFactorDimx,] dtype: float
  tmp used mem: [UbFactorDimx, outDimy_ * 2] dtype: float
  x, activation_scale enable db
  ub reserve 1024B

  optional buffer：
  bias used mem: [1, outDimy_ * 2] dtype: float

  clamp tmp buffer: [UbFactorDimx, outDimy_] dtype: uint8

  gather offset buffer: [UbFactorDimx, outDimy_] dtype: uint32

  */
  int64_t db = 2;
  int64_t maxOutDimy = 0;
  int64_t biasBufferY = hasBias_ == false ? 0 : static_cast<int64_t>(SWI_FACTOR * sizeof(float));
  int64_t biasBufferX = hasBias_ == false ? 0 : outDimy_ * SWI_FACTOR * static_cast<int64_t>(sizeof(float));

  int64_t SweiGLUBufferY = swigluMode_ == 0 ? 0 : static_cast<int64_t>(sizeof(int8_t) + sizeof(int32_t));
  int64_t SweiGLUBufferX = swigluMode_ == 0 ? 0 : outDimy_ * static_cast<int64_t>(sizeof(int8_t)) + outDimy_ * static_cast<int64_t>(sizeof(int32_t));

  int64_t quantOffsetSpace = quantMode_ == QUANT_MODE_DYNAMIC ? 0 : static_cast<int64_t>(sizeof(float));

  // UbFactorDimx is 1,compute maxOutDimy
  int64_t numerator = static_cast<int64_t>(ubSize_) - UB_RESERVE - BLOCK_SIZE - db * BLOCK_SIZE - static_cast<int64_t>(sizeof(float));
  int64_t denominator =
      5 * static_cast<int64_t>(sizeof(float)) + db * SWI_FACTOR * static_cast<int64_t>(sizeof(float)) + static_cast<int64_t>(sizeof(int8_t)) + biasBufferY + SweiGLUBufferY + quantOffsetSpace;
  maxOutDimy = static_cast<int64_t>(numerator / denominator);
  maxOutDimy = maxOutDimy / BLOCK_SIZE * BLOCK_SIZE;
  int64_t maxInDimy = static_cast<int64_t>(maxOutDimy * SWI_FACTOR);
  OP_LOGI(context_->GetNodeName(), "Get maxInDimy[%ld]", maxInDimy);
  OP_CHECK_IF(inDimy_ > maxInDimy,
                  OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "x",
                      std::to_string(inDimy_).c_str(),
                      ("less than or equal to " + std::to_string(maxInDimy)).c_str()),
                  return ge::GRAPH_FAILED);

  // compute ubFactorDimx
  quantOffsetSpace = quantMode_ == QUANT_MODE_DYNAMIC ? 0 : outDimy_ * sizeof(float);
  numerator = static_cast<int64_t>(ubSize_) - UB_RESERVE - outDimy_ * static_cast<int64_t>(sizeof(float)) - BLOCK_SIZE - SWI_FACTOR * outDimy_ * static_cast<int64_t>(sizeof(float)) - biasBufferX - quantOffsetSpace;

  denominator = db * (outDimy_ * SWI_FACTOR + BLOCK_ELEM) * static_cast<int64_t>(sizeof(float)) + outDimy_ * static_cast<int64_t>(sizeof(int8_t)) + static_cast<int64_t>(sizeof(float)) +
                outDimy_ * SWI_FACTOR * static_cast<int64_t>(sizeof(float)) + SweiGLUBufferX;
  ubFactorDimx  = static_cast<int64_t>(numerator / denominator);
  ubFactorDimx = std::min(ubFactorDimx, inDimx_);
  OP_LOGI(context_->GetNodeName(), "Get ubFactorDimx[%ld]", ubFactorDimx);

  // special ub cut for 2048 4096
  if (swigluMode_ == 0 && hasBias_ == false) {
      ubFactorDimx =
      (inDimy_ == PERFORMANCE_H_2048 || inDimy_ == PERFORMANCE_H_4096) ? PERFORMANCE_UB_FACTOR / inDimy_ : ubFactorDimx;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::DoOpTiling() {
  if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
    return ge::GRAPH_FAILED;
  }
  auto inputShapeX = context_->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context_, inputShapeX);

  int64_t ubFactorDimx = 0;
  OP_CHECK_IF(CountMaxDim(ubFactorDimx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Count MaxDim failed."),
                return ge::GRAPH_FAILED);

  maxPreCore_ = (inDimx_ + ubFactorDimx - 1) / ubFactorDimx;
  maxPreCore_ = std::min(maxPreCore_, static_cast<int64_t>(PERFORMANCE_CORE_NUM));
  maxPreCore_ = std::min(maxPreCore_, static_cast<int64_t>(coreNum_));

  CountTilingKey();

  tilingData_.set_inDimx(inDimx_);
  tilingData_.set_inDimy(inDimy_);
  tilingData_.set_outDimy(outDimy_);
  tilingData_.set_UbFactorDimx(ubFactorDimx);
  tilingData_.set_UbFactorDimy(outDimy_);
  tilingData_.set_usedCoreNum(maxPreCore_);
  tilingData_.set_maxCoreNum(maxPreCore_);
  tilingData_.set_inGroupNum(groupNum_);
  tilingData_.set_quantMode(quantMode_);
  tilingData_.set_actRight(actRight_);
  tilingData_.set_speGroupType(static_cast<int64_t>(speGroupType_));
  tilingData_.set_hasBias(hasBias_);

  tilingData_.set_swigluMode(swigluMode_);
  tilingData_.set_clampLimit(clampLimit_);
  tilingData_.set_gluAlpha(gluAlpha_);
  tilingData_.set_gluBias(gluBias_);
  return ge::GRAPH_SUCCESS;
}

void DequantSwigluQuantDskTiling::DumpTilingInfo() {
  std::ostringstream info;
  info << "inDimx_: " << tilingData_.get_inDimx();
  info << ", inDimy_: " << tilingData_.get_inDimy();
  info << ", outDimy: " << tilingData_.get_outDimy();
  info << ", UbFactorDimx: " << tilingData_.get_UbFactorDimx();
  info << ", UbFactorDimy: " << tilingData_.get_UbFactorDimy();
  info << ", usedCoreNum: " << tilingData_.get_usedCoreNum();
  info << ", maxCoreNum: " << tilingData_.get_maxCoreNum();
  info << ", inGroupNum: " << tilingData_.get_inGroupNum();
  info << ", quantMode: " << tilingData_.get_quantMode();
  info << ", actRight: " << tilingData_.get_actRight();
  info << ", tilingKey: " << tilingKey_;
  info << ", hasBias: " << hasBias_;
  info << ", swigluMode: " << tilingData_.get_swigluMode();
  info << ", clampLimit: " << tilingData_.get_clampLimit();
  info << ", gluAlpha: " << tilingData_.get_gluAlpha();
  info << ", gluBias: " << tilingData_.get_gluBias();

  OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

ge::graphStatus DequantSwigluQuantDskTiling::DoLibApiTiling() {
  return ge::GRAPH_SUCCESS;
}

uint64_t DequantSwigluQuantDskTiling::GetTilingKey() const {
  return tilingKey_;
}

ge::graphStatus DequantSwigluQuantDskTiling::GetWorkspaceSize() {
  workspaceSize_ = WORKSPACE_SIZE;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantDskTiling::PostTiling() {
  context_->SetTilingKey(GetTilingKey());
  context_->SetBlockDim(maxPreCore_);
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  workspaces[0] = workspaceSize_;
  tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("DequantSwigluQuant", DequantSwigluQuantDskTiling, 0);

ge::graphStatus TilingForDequantSwigluQuant(gert::TilingContext* context) {
  return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForDequantSwigluQuant(gert::TilingParseContext* context) {
  OP_LOGD(context, "TilingPrepare4DequantSwigluQuant enter.");
  auto compileInfo = context->GetCompiledInfo<DequantSwigluQuantCompileInfo>();
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

  OP_LOGD(context, "TilingPrepare4DequantSwigluQuant exit.");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DequantSwigluQuant)
    .Tiling(TilingForDequantSwigluQuant)
    .TilingParse<DequantSwigluQuantCompileInfo>(TilingPrepareForDequantSwigluQuant);

}  // namespace optiling
