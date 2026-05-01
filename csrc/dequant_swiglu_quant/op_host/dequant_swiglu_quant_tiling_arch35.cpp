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
 * \file dequant_swiglu_quant_tiling_arch35.cpp
 * \brief
 */

#include <cmath>
#include "dequant_swiglu_quant_tiling.h"
#include "../tiling_base/tiling_templates_registry.h"
#include "dequant_swiglu_quant_tiling.h"
#include "../tiling_base/tiling_base.h"
#include "../tiling_base/tiling_util.h"

using namespace AscendC;
using namespace ge;
namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;

constexpr int64_t ATTR_ACTIVATE_LEFT_INDEX = 0;
constexpr int64_t ATTR_QUANT_MODE_INDEX = 1;
constexpr int64_t ATTR_DST_TYPE_INDEX = 2;
constexpr int64_t ATTR_ROUND_MODE_INDEX = 3;
constexpr int64_t ATTR_ACTIVATE_DIM_INDEX = 4;
constexpr int64_t ATTR_SWIGLU_MODE_INDEX = 5;
constexpr int64_t ATTR_CLAMP_LIMIT_INDEX = 6;
constexpr int64_t ATTR_GLU_ALPHA_INDEX = 7;
constexpr int64_t ATTR_GLU_BIAS_INDEX = 8;
constexpr int64_t X_INDEX = 0;
constexpr int64_t WEIGHT_SCALE_INDEX = 1;
constexpr int64_t ACTIVATION_SCALE_INDEX = 2;
constexpr int64_t BIAS_INDEX = 3;
constexpr int64_t QUANT_SCALE_INDEX = 4;
constexpr int64_t QUANT_OFFSET_INDEX = 5;
constexpr int64_t INPUT_GROUP_INDEX = 6;
constexpr int64_t Y_INDEX = 0;
constexpr int64_t SCALE_INDEX = 1;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t BLOCK_ELEM_B32 = BLOCK_SIZE / static_cast<int64_t>(sizeof(float));
constexpr int64_t BLOCK_ELEM_B16 = BLOCK_SIZE / static_cast<int64_t>(sizeof(int16_t));
constexpr int64_t BLOCK_ELEM_B8 = BLOCK_SIZE / static_cast<int64_t>(sizeof(int8_t));
constexpr size_t SYS_WORK_SPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);
constexpr uint64_t WORKSPACE_SIZE = 32;
constexpr int64_t UB_REVERSE = 1024;
constexpr int64_t SWI_FACTOR = 2;
constexpr int64_t Y_LAST_DIM_FULL_LOAD_MAX_VALUE = 5120; // 能够命中UB全载模板的输出尾轴最大值
constexpr int64_t QUANT_MODE_DYNAMIC = 1;
constexpr int64_t QUANT_MODE_INDEX = 1;
constexpr int64_t ACTIVATE_DIM_FACTOR = 100000;
constexpr int64_t INPUT_X_FACTOR = 10000;
constexpr int64_t BIAS_FACTOR = 1000;
constexpr int64_t ACT_SCALE_FACTOR = 100;
constexpr int64_t QUANT_SCALE_FACTOR = 10;
constexpr int64_t GROUP_INDEX_FACTOR = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t PLACEHOLDER = 1000000;
constexpr int64_t QUANT_MODE_FACTOR = 100000;
constexpr int64_t BIAS_FACTOR_FOR_NOT_FULL = 10000;
constexpr int64_t ACTIVATE_FACTOR_FOR_NOT_FULL = 1000;
constexpr int64_t QUANT_SCALE_FACTOR_FOR_NOT_FULL = 100;
constexpr int64_t QUANT_OFFSET_FACTOR_FOR_NOT_FULL = 10;
constexpr int64_t SPECIAL_GROUP_NUM_64 = 64; // groupIndex存在时走特殊分核的group条件
constexpr int64_t SPECIAL_GROUP_NUM_32 = 32; // groupIndex不存在时走特殊分核的group条件
constexpr int64_t SPECIAL_GROUP_NUM_16 = 16; // 走特殊分核场景的分组条件
constexpr float CLAMP_LIMIT_DEFAULT= 7.0;
constexpr float GLU_ALPHA_DEFAULT = 1.702;
constexpr float GLU_BIAS_DEFAULT = 1.0;
static const gert::Shape g_vec_1_shape = {1};

inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
  if (in_shape.IsScalar()) {
    return g_vec_1_shape;
  }
  return in_shape;
}

static const std::set<ge::DataType> SUPPORT_DTYPE = {ge::DT_INT32, ge::DT_BF16, ge::DT_FLOAT16};
static const std::map<std::string, int64_t> SUPPORT_QUANT_MODE = {{"dynamic", 1}, {"static", 0}};
// 定义bias支持的所有类型
static const std::set<ge::DataType> BIAS_SUPPORT_DTYPE = {ge::DT_INT32, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT};
static const std::map<ge::DataType, int64_t> SUPPORT_BIAS_MODE = {{ge::DT_INT32, 1}, {ge::DT_BF16, 2}, {ge::DT_FLOAT16, 3}, {ge::DT_FLOAT, 4}};
// 定义quant_scale支持的所有类型 
static const std::set<ge::DataType> QUANT_SCALE_SUPPORT_DTYPE = {ge::DT_FLOAT};
// 定义quant_offset支持的所有类型 
static const std::set<ge::DataType> QUANT_OFFSET_SUPPORT_DTYPE = {ge::DT_FLOAT};
// 定义输出y支持的所有类型:int8, hifloat8, float8的两种类型, float4的两种类型
static const std::set<ge::DataType> OUTPUT_SUPPORT_DTYPE = {ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
// 定义roundMode映射表。
static const std::map<std::string, uint32_t> SUPPORT_ROUND_MODE = {{"rint", 0}, {"round", 1}, {"floor", 2}, {"ceil", 3}, {"trunc", 4}};

ge::graphStatus DequantSwigluQuantV35DskTiling::GetPlatformInfo()
{
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

ge::graphStatus DequantSwigluQuantV35DskTiling::GetInputX()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    ge::DataType xDType = xDesc->GetDataType();
    // 校验x的数据类型是否合法
    OP_CHECK_IF((SUPPORT_DTYPE.find(xDType) == SUPPORT_DTYPE.end()),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x",
            ge::TypeUtils::DataTypeToSerialString(xDType).c_str(), "int32, float16 or bf16"),
        return ge::GRAPH_FAILED);

    auto xStorageShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorageShape);
    xShape_ = EnsureNotScalar(xStorageShape->GetStorageShape());
    xDimNum_ = xShape_.GetDimNum();
    OP_CHECK_IF(xDimNum_ < 2,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x",
            std::to_string(xDimNum_).c_str(), "greater than or equal to 2"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDimNum_ > 8,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x",
            std::to_string(xDimNum_).c_str(), "less than or equal to 8"),
        return ge::GRAPH_FAILED);
    for (size_t i = 0; i < xDimNum_; i++) {
        OP_CHECK_IF(xShape_.GetDim(i) <= 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
                Ops::Base::ToString(xShape_).c_str(),
                ("the dim[" + std::to_string(i) + "] of x must be positive").c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::GetInputGroupIndex()
{
    auto groupIndexDesc = context_->GetOptionalInputDesc(INPUT_GROUP_INDEX);
    if (groupIndexDesc != nullptr) {
        ge::DataType groupIndexDType = groupIndexDesc->GetDataType();
        OP_CHECK_IF(groupIndexDType != ge::DT_INT64 && groupIndexDType != ge::DT_INT32,
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "group_index",
                ge::TypeUtils::DataTypeToSerialString(groupIndexDType).c_str(), "int32 or int64"),
            return ge::GRAPH_FAILED);

        auto groupIndexStorageShape = context_->GetOptionalInputShape(INPUT_GROUP_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, groupIndexStorageShape);
        groupIndexShape_ = EnsureNotScalar(groupIndexStorageShape->GetStorageShape());
        auto groupIndexDimNum = groupIndexShape_.GetDimNum();
        OP_CHECK_IF((groupIndexDimNum != 1) && (groupIndexDimNum != 2),
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "group_index",
                std::to_string(groupIndexDimNum).c_str(), "1 or 2"),
            return ge::GRAPH_FAILED);
        
        groupNum_ = groupIndexShape_.GetDim(0);
        OP_CHECK_IF(groupNum_ < 1,
            OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "group_index",
                std::to_string(groupNum_).c_str(), "group_index[0] must be greater than or equal to 1"),
            return ge::GRAPH_FAILED);

        hasGroupIndex_ = true;
        groupIndexMode_ = groupIndexDType == ge::DT_INT32 ? 1 : 2; // groupIndex：int32类型时mode设为1；int64类型时设为2；不存在则保持默认值0
        if (groupIndexDimNum == 2) {
          speGroupType_ = 1;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::GetAttrActivateDim()
{
  auto* attrs = context_->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
  // 校验activate_dim
  auto* attrActivateDim = attrs->GetAttrPointer<int>(ATTR_ACTIVATE_DIM_INDEX);
  // 类型校验
  activateDim_ = (attrActivateDim != nullptr) ? *attrActivateDim : -1;
  // 指定切分轴维度转换为正数
  activateDim_ = activateDim_ < 0 ? activateDim_ + static_cast<int64_t>(xDimNum_) : activateDim_;
  
  // 判断切分轴维度合法性
  OP_CHECK_IF(activateDim_ < 0 || activateDim_ >= static_cast<int64_t>(xDimNum_),
                  OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "activate_dim",
                    std::to_string(activateDim_).c_str(),
                    ("activate_dim must be in [-" + std::to_string(xDimNum_) + ", " +
                     std::to_string(xDimNum_ - 1) + "]").c_str()),
                  return ge::GRAPH_FAILED);
  // 校验切分轴对应的shape是不是偶数
  OP_CHECK_IF(xShape_.GetDim(activateDim_) % 2 != 0,
                  OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
                      Ops::Base::ToString(xShape_).c_str(),
                      ("the split dim(" + std::to_string(activateDim_)+ "dimension) must be even").c_str()),
                  return ge::GRAPH_FAILED);

  //如果activateDim不是尾轴，则不允许输入group
  if (activateDim_ != static_cast<int64_t>(xDimNum_ - 1)) {
    OP_CHECK_IF(hasGroupIndex_ == true,
                  OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "group_index and activate_dim",
                      ("group_index is not None, and activate_dim is " + std::to_string(activateDim_)).c_str(), "group_index must be None when activate_dim is not the last dim of x"),
                  return ge::GRAPH_FAILED);
  }

  // activate_dim对应在x的轴需要是偶数
  OP_CHECK_IF((xShape_.GetDim(activateDim_) % 2) != 0,
      OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
          Ops::Base::ToString(xShape_).c_str(),
          "the x dimension of activateDim must be even"),
      return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::CheckOutputY()
{
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    ge::DataType yDType = yDesc->GetDataType();
    OP_CHECK_IF(OUTPUT_SUPPORT_DTYPE.find(yDType) == OUTPUT_SUPPORT_DTYPE.end(),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y",
            ge::TypeUtils::DataTypeToSerialString(yDType).c_str(),
            "int8, hifloat8, float8e4m3, float8e5m2, float4e2m1 or floate1m2"),
        return ge::GRAPH_FAILED);
    auto yStorageShape = context_->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yStorageShape);
    auto& yShape = EnsureNotScalar(yStorageShape->GetStorageShape());
    const size_t yDimNum = yShape.GetDimNum();
    // 输出y是fp4类型时，y的尾轴对应的shape需要是偶数
    if (yDType == ge::DT_FLOAT4_E2M1 || yDType == ge::DT_FLOAT4_E1M2) {
      OP_CHECK_IF((yShape.GetDim(xDimNum_ - 1) % 2) != 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y",
            Ops::Base::ToString(yShape).c_str(),
            "The last dim of y must be even when the type of y is FP4X2_E2M1 or FP4X2_E1M2"),
        return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(yDimNum != xDimNum_,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "y",
            std::to_string(yDimNum).c_str(),
            ("equal to x dimension " + std::to_string(xDimNum_)).c_str()),
        return ge::GRAPH_FAILED);
    for (size_t i = 0; i < yDimNum; i++) {
      if (static_cast<int>(i) != activateDim_){
        OP_CHECK_IF(yShape.GetDim(i) != xShape_.GetDim(i),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x and y",
                (Ops::Base::ToString(xShape_) + "and" + Ops::Base::ToString(yShape)).c_str(),
                ("dim[" + std::to_string(i) + "] of y must be equal to dim[" + std::to_string(i) + "] of x").c_str()),
            return ge::GRAPH_FAILED);
      } else {
        OP_CHECK_IF(yShape.GetDim(i) != xShape_.GetDim(i) / SWI_FACTOR,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x and y",
                Ops::Base::ToString(yShape).c_str(),
                ("dim[" + std::to_string(i) + "] of y must be equal to half of dim[" + std::to_string(i) + "] of x").c_str()),
            return ge::GRAPH_FAILED);
      }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::CheckInputWeightScale()
{
    auto wScaleDesc = context_->GetOptionalInputDesc(WEIGHT_SCALE_INDEX);
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    ge::DataType xDType = xDesc->GetDataType();
    // 如果输入x是bf16 or float16，则weight_scale需要为空，非法性校验
    if (wScaleDesc != nullptr) {
      OP_CHECK_IF(xDType == ge::DT_FLOAT16 || xDType == ge::DT_BF16,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "weight_scale",
            "not None", "weight_scale must be None when x's datatype is in [bfloat16, float16]"),
        return ge::GRAPH_FAILED);
    }
    
    // 如果输入x是int32，则weight_scale必须有值，合法性校验
    OP_CHECK_IF((xDType == ge::DT_INT32) && (wScaleDesc == nullptr),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "weight_scale",
            "None", "weight_scale must be not None when x's datatype is int32"),
        return ge::GRAPH_FAILED);

    // weight_scale不为空，进行判断
    if (wScaleDesc != nullptr) {
      OP_CHECK_NULL_WITH_CONTEXT(context_, wScaleDesc);
      ge::DataType wScaleDType = wScaleDesc->GetDataType();
      OP_CHECK_IF(wScaleDType != ge::DT_FLOAT,
          OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "weight_scale",
              ge::TypeUtils::DataTypeToSerialString(wScaleDType).c_str(), "float32"),
          return ge::GRAPH_FAILED);

      auto wScaleStorageShape = context_->GetOptionalInputShape(WEIGHT_SCALE_INDEX);
      OP_CHECK_NULL_WITH_CONTEXT(context_, wScaleStorageShape);
      auto& wScaleShape = EnsureNotScalar(wScaleStorageShape->GetStorageShape());
      const size_t wScaleDimNum = wScaleShape.GetDimNum();
      OP_CHECK_IF(wScaleDimNum > 2,
          OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "weight_scale",
              std::to_string(wScaleDimNum).c_str(), "less than or equal to 2"),
          return ge::GRAPH_FAILED);
      
      if (wScaleDimNum == static_cast<size_t>(1)) {
        OP_CHECK_IF(hasGroupIndex_ == true,
          OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "group_index",
              "not None", "group_index should be none when weight_scale dimension is 1"),
          return ge::GRAPH_FAILED);
        OP_CHECK_IF(wScaleShape.GetDim(0) != xShape_.GetDim(xDimNum_ - 1),
          OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "weight_scale",
              Ops::Base::ToString(wScaleShape).c_str(),
              ("The first dim of weight_scale must be the same as the last dim of x: " + std::to_string(xShape_.GetDim(xDimNum_ - 1))).c_str()),
          return ge::GRAPH_FAILED);
      }
      if (wScaleDimNum > static_cast<size_t>(1)) {
        if (hasGroupIndex_) {
          OP_CHECK_IF(!(wScaleShape.GetDim(0) == groupIndexShape_.GetDim(0) && wScaleShape[wScaleDimNum - 1] == xShape_.GetDim(xDimNum_ - 1)),
            OP_LOGE(context_->GetNodeName(),
                                            "weight_scale shape[0] must be equal to group_index shape[0], and shape[-1] must be equal to x shape[-1] "
                                            "when group_index exists, please check."),
            return ge::GRAPH_FAILED);
        } else {
          OP_CHECK_IF(!(wScaleShape.GetDim(0) == 1 && wScaleShape.GetDim(wScaleDimNum - 1) == xShape_.GetDim(xDimNum_ - 1)) &&
                      !(wScaleShape.GetDim(0) == xShape_.GetDim(xDimNum_ - 1) && wScaleShape.GetDim(wScaleDimNum - 1) == 1),
                      OP_LOGE(context_->GetNodeName(), "weight_scale shape must be in {[1, %ld], [%ld, 1]} when weight_scale dimension == 2 and group_index not exists,"
                      "please check.", xShape_.GetDim(xDimNum_ - 1), xShape_.GetDim(xDimNum_ - 1)),
                      return ge::GRAPH_FAILED);
        }
      }
      hasWeightScale_ = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::CheckInputActScale()
{
    auto aScaleDesc = context_->GetOptionalInputDesc(ACTIVATION_SCALE_INDEX);
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    ge::DataType xDType = xDesc->GetDataType();
    // 当x：bfloat16 or float16时，activate_scale需要为空
    if (aScaleDesc != nullptr) {
        OP_CHECK_IF(xDType == ge::DT_FLOAT16 || xDType == ge::DT_BF16,
          OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "activation_scale",
              "not None", "activate_scale must be None when x's datatype is in [bfloat16, float16]"),
          return ge::GRAPH_FAILED);

        ge::DataType aScaleDType = aScaleDesc->GetDataType();
        OP_CHECK_IF(aScaleDType != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "activation_scale",
                ge::TypeUtils::DataTypeToSerialString(aScaleDType).c_str(), "float32"),
            return ge::GRAPH_FAILED);

        auto aScaleStorageShape = context_->GetOptionalInputShape(ACTIVATION_SCALE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, aScaleStorageShape);
        auto& aScaleShape = EnsureNotScalar(aScaleStorageShape->GetStorageShape());
        const size_t aScaleDimNum = aScaleShape.GetDimNum();

        OP_CHECK_IF(aScaleDimNum <= 0,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "activation_scale",
                std::to_string(aScaleDimNum).c_str(), "greater than 0"),
            return ge::GRAPH_FAILED);
        
        // shape校验
        // activation_scale的shape size与x除尾轴外的shape size一致
        int64_t aScaleSize = aScaleStorageShape->GetStorageShape().GetShapeSize();
        int64_t xSizeWithoutLastDim = xShape_.GetShapeSize() / xShape_.GetDim(xDimNum_ - 1);
        OP_CHECK_IF(aScaleSize != xSizeWithoutLastDim,
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "activation_scale",
                        std::to_string(aScaleSize).c_str(),
                        ("The shape size of activation_scale should be the same as x's shape size without last dim " + std::to_string(xSizeWithoutLastDim)).c_str()),
                    return ge::GRAPH_FAILED);
        hasActivationScale_ = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::CheckInputBias()
{
    auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    ge::DataType xDType = xDesc->GetDataType();
    // 首先判断bias不为空时，其数据类型是不是满足计算要求
    if (biasDesc != nullptr) {
      ge::DataType biasDtype = biasDesc->GetDataType();
      auto it = SUPPORT_BIAS_MODE.find(biasDtype);
      OP_CHECK_IF(it == SUPPORT_BIAS_MODE.end(),
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "bias",
                ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str(), "float16, float, bf16 or int32"),
            return ge::GRAPH_FAILED);
      biasMode_ = it->second;
      // 当前bias支持四种数据类型，但是有些bias类型仅支持x的特定类型
      // x：bf16, float16，bias不支持输入
      if (xDType == ge::DT_BF16 or xDType == ge::DT_FLOAT16) {
        OP_CHECK_IF(BIAS_SUPPORT_DTYPE.find(biasDtype) != BIAS_SUPPORT_DTYPE.end(),
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "bias",
                ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str(),
                "bias not support when the type of x is bf16 or float16"),
            return ge::GRAPH_FAILED);
      }

      // 然后判断bias的维度是不是满足要求
      auto biasStorageShape = context_->GetOptionalInputShape(BIAS_INDEX);
      OP_CHECK_NULL_WITH_CONTEXT(context_, biasStorageShape);
      auto& biasShape = EnsureNotScalar(biasStorageShape->GetStorageShape());
      const size_t biasDimNum = biasShape.GetDimNum();
      OP_CHECK_IF(biasDimNum > static_cast<size_t>(2) || biasDimNum == static_cast<size_t>(0),
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "bias",
                std::to_string(biasDimNum).c_str(), "1D or 2D"),
            return ge::GRAPH_FAILED);
      // 当biasDimNum=1时
      if (biasDimNum == static_cast<size_t>(1)) {
        OP_CHECK_IF(hasGroupIndex_ == true,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "group_index",
                        "not None", "group_index should be none when bias dimension is 1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(biasShape.GetDim(0) != xShape_.GetDim(xDimNum_ - 1),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "bias",
                Ops::Base::ToString(biasShape).c_str(),
                ("The last dimension of bias should be the same as the last dimension of x " +
                 std::to_string(xShape_.GetDim(xDimNum_ - 1))).c_str()),
            return ge::GRAPH_FAILED);
      }

      // 当biasDimNum=2时
      if (biasDimNum == static_cast<size_t>(2)) {
        OP_CHECK_IF(biasShape.GetDim(1) != xShape_.GetDim(xDimNum_ - 1),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "bias",
                Ops::Base::ToString(biasShape).c_str(),
                ("The last dimension of bias should be the same as the last dimension of x " +
                 std::to_string(xShape_.GetDim(xDimNum_ - 1))).c_str()),
            return ge::GRAPH_FAILED);
          if (hasGroupIndex_) {
              if (biasShape.GetDim(0) != groupNum_) {
                  std::string reason =
                      "when the dimension of bias is 2 and group_index exists, the first dimension of bias (" +
                      std::to_string(biasShape.GetDim(0)) +
                      ") should be equal to the first dimension of group_index (" + std::to_string(groupNum_) +
                      "), please check";
                  OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                      context_->GetNodeName(), "bias", Ops::Base::ToString(biasShape).c_str(), reason.c_str());
                  return ge::GRAPH_FAILED;
              }
          } else {
            OP_CHECK_IF(biasShape.GetDim(0) != 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "bias",
                std::to_string(biasShape.GetDim(0)).c_str(),
                "The first dimension of bias should be 1 when the dimension of bias is 2 and group_index does not exist"),
                return ge::GRAPH_FAILED);
          }
      }
      hasBias_ = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::CheckInputQuantScale()
{
    auto qScaleDesc = context_->GetOptionalInputDesc(QUANT_SCALE_INDEX);
    if (qScaleDesc != nullptr) {
        ge::DataType qScaleDType = qScaleDesc->GetDataType();
        OP_CHECK_IF(QUANT_SCALE_SUPPORT_DTYPE.find(qScaleDType) == QUANT_SCALE_SUPPORT_DTYPE.end(),
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "quant_scale",
                ge::TypeUtils::DataTypeToSerialString(qScaleDType).c_str(), "float32"),
            return ge::GRAPH_FAILED);

        auto qScaleStorageShape = context_->GetOptionalInputShape(QUANT_SCALE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, qScaleStorageShape);
        auto& qScaleShape = EnsureNotScalar(qScaleStorageShape->GetStorageShape());
        const size_t qScaleDimNum = qScaleShape.GetDimNum();
        OP_CHECK_IF(qScaleDimNum > 2,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "quant_scale",
                std::to_string(qScaleDimNum).c_str(), "less than or equal to 2"),
            return ge::GRAPH_FAILED);
        
        // 获取y的shape
        auto yStorageShape = context_->GetOutputShape(Y_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, yStorageShape);
        auto& yShape = EnsureNotScalar(yStorageShape->GetStorageShape());
        if (hasGroupIndex_) {
          if (quantMode_ == 0) {
            OP_CHECK_IF(qScaleShape.GetDim(0) != (groupIndexShape_.GetDim(0)),
                        OP_LOGE(context_->GetNodeName(),
                        "quant_scale shape[0] must be equal to group_index shape[0] when static_quant and group_index exists, please check."),
                        return ge::GRAPH_FAILED);
            if (qScaleDimNum == DIM_TWO) {
              OP_CHECK_IF(!((qScaleShape.GetDim(qScaleDimNum - 1) == yShape.GetDim(xDimNum_ - 1)) || (qScaleShape.GetDim(qScaleDimNum - 1) == 1)),
                          OP_LOGE(context_->GetNodeName(),
                          "quant_scale shape[-1] must be equal to or can be broadcast to y shape[-1] when static_quant and group_index exists, please check."),
                          return ge::GRAPH_FAILED);
            }
          } else if (quantMode_ == 1) {
            OP_CHECK_IF(qScaleShape.GetDim(0) != groupIndexShape_.GetDim(0) || qScaleShape.GetDim(qScaleDimNum - 1) != yShape.GetDim(xDimNum_ - 1),
                        OP_LOGE(context_->GetNodeName(),
                        "quant_scale shape must be [ group_index_shape[0], y_shape[-1] ] when dynamic quant and group_index exists, please check."),
                        return ge::GRAPH_FAILED);
          }
          quantIsOne_ = (qScaleDimNum == DIM_TWO && qScaleShape.GetDim(qScaleDimNum - 1) == yShape.GetDim(xDimNum_ - 1)) ? 0 : 1;
        } else {
          if (qScaleDimNum == DIM_TWO) {
            OP_CHECK_IF(qScaleShape.GetDim(0) != 1,
                        OP_LOGE(context_->GetNodeName(),
                        "if dim of quant_scale is 2, shape[0] must be [1] when group_index not exists, please check."),
                        return ge::GRAPH_FAILED);
          }
          if (quantMode_ == 0) {
            OP_CHECK_IF(qScaleShape.GetDim(0) != 1 && qScaleShape.GetDim(0) != yShape.GetDim(xDimNum_ - 1),
                        OP_LOGE(context_->GetNodeName(),
                        "quant_scale shape[0] must be or can be broadcast to y shape[-1] when static_quant and group_index not exists, please check."),
                        return ge::GRAPH_FAILED);
          } else if (quantMode_ == 1) {
            OP_CHECK_IF(qScaleShape.GetDim(qScaleDimNum - 1) != yShape.GetDim(xDimNum_ - 1),
                        OP_LOGE(context_->GetNodeName(),
                        "quant_scale shape[-1] must be equal to y shape[-1] when dynamic_quant and group_index not exists, please check."),
                        return ge::GRAPH_FAILED);
          }
          quantIsOne_ = qScaleShape.GetDim(qScaleDimNum - 1) == yShape.GetDim(xDimNum_ - 1) ? 0 : 1;
        }
        hasQuantScale_ = true;
    }
    if (quantMode_ == 0) {
      OP_CHECK_IF(!hasQuantScale_,
                  OP_LOGE(context_->GetNodeName(), "quant_scale must exist when static_quant, please check."),
                  return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::CheckInputQuantOffset()
{
    auto qOffsetDesc = context_->GetOptionalInputDesc(QUANT_OFFSET_INDEX);
    /*
    校验：
    1、是静态量化
    2、activate_dim=-1
    3、有group_index时shape维度≤2
    3.1、1维时：shape为[G]
    3.2、2维时：shape为[G,1]/[G,H]
    4、无group_index时shape为[1]/[H]/[1,H]
    */
    if (qOffsetDesc != nullptr) {
      ge::DataType qOffsetDType = qOffsetDesc->GetDataType();
      OP_CHECK_IF(QUANT_OFFSET_SUPPORT_DTYPE.find(qOffsetDType) == QUANT_OFFSET_SUPPORT_DTYPE.end(),
                  OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "quant_offset",
                      ge::TypeUtils::DataTypeToSerialString(qOffsetDType).c_str(), "float32"),
            return ge::GRAPH_FAILED);
      OP_CHECK_IF(quantMode_ != 0,
                  OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "quant_offset",
                      "not None", "quant_offset only be supported when static quant, but current quant mode is dynamic quant, quant_offset should be None."),
                  return ge::GRAPH_FAILED);
      auto qOffsetStorageShape = context_->GetOptionalInputShape(QUANT_OFFSET_INDEX);
      OP_CHECK_NULL_WITH_CONTEXT(context_, qOffsetStorageShape);
      auto& qOffsetShape = EnsureNotScalar(qOffsetStorageShape->GetStorageShape());
      const size_t qOffsetDimNum = qOffsetShape.GetDimNum();
      OP_CHECK_IF(qOffsetDimNum > 2,
                  OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "quant_offset",
                      std::to_string(qOffsetDimNum).c_str(), "less than or equal to 2"),
                  return ge::GRAPH_FAILED);
      auto yStorageShape = context_->GetOutputShape(Y_INDEX);
      OP_CHECK_NULL_WITH_CONTEXT(context_, yStorageShape);
      auto& yShape = EnsureNotScalar(yStorageShape->GetStorageShape());
      if (hasGroupIndex_) {
        OP_CHECK_IF(qOffsetShape.GetDim(0) != (groupIndexShape_.GetDim(0)),
                    OP_LOGE(context_->GetNodeName(),
                    "quant_offset shape[0] must be equal to group_index shape[0] when group_index exists, please check."),
                    return ge::GRAPH_FAILED);
        if (qOffsetDimNum == DIM_TWO) {
              OP_CHECK_IF(qOffsetShape.GetDim(qOffsetDimNum - 1) != 1 && qOffsetShape.GetDim(qOffsetDimNum - 1) != yShape.GetDim(xDimNum_ - 1),
                          OP_LOGE(context_->GetNodeName(),
                          "quant_offset shape[-1] must be equal to or can be broadcast to y shape[-1] when group_index exists, please check."),
                          return ge::GRAPH_FAILED);
        }
      } else {
        if (qOffsetDimNum == DIM_TWO) {
            OP_CHECK_IF(qOffsetShape.GetDim(0) != 1,
                        OP_LOGE(context_->GetNodeName(),
                        "if dim of quant_offset is 2, shape[0] must be [1] when group_index not exists, please check."),
                        return ge::GRAPH_FAILED);
          }
          OP_CHECK_IF(qOffsetShape.GetDim(0) != 1 && qOffsetShape.GetDim(0) != yShape.GetDim(xDimNum_ - 1),
                      OP_LOGE(context_->GetNodeName(),
                      "quant_offset shape[0] must be or can be broadcast to y shape[-1] when static_quant and group_index not exists, please check."),
                      return ge::GRAPH_FAILED);
      }
      hasQuantOffset_ = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::CheckForStaticQuant() // 静态量化quant_scale和quant_offset的shape size需要一致
{
    if (!hasQuantScale_ || !hasQuantOffset_) {
      return ge::GRAPH_SUCCESS;
    }
    auto qScaleStorageShape = context_->GetOptionalInputShape(QUANT_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qScaleStorageShape);
    int64_t qScaleSize = qScaleStorageShape->GetStorageShape().GetShapeSize();
    auto qOffsetStorageShape = context_->GetOptionalInputShape(QUANT_OFFSET_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qOffsetStorageShape);
    int64_t qOffsetSize = qOffsetStorageShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(qScaleSize != qOffsetSize,
                OP_LOGE(context_->GetNodeName(), "quant_scale size should be equal to quant_offset size, please check."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::CheckOutputScale()
{
    auto scaleDesc = context_->GetOutputDesc(SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleDesc);
    ge::DataType scaleDType = scaleDesc->GetDataType();
    OP_CHECK_IF(scaleDType != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "scale",
            ge::TypeUtils::DataTypeToSerialString(scaleDType).c_str(), "float32"),
        return ge::GRAPH_FAILED);
    auto scaleStorageShape = context_->GetOutputShape(SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleStorageShape);
    auto& scaleShape = EnsureNotScalar(scaleStorageShape->GetStorageShape());
    const size_t scaleDimNum = scaleShape.GetDimNum();

    auto yStorageShape = context_->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yStorageShape);
    auto& yShape = EnsureNotScalar(yStorageShape->GetStorageShape());
    const size_t yDimNum = yShape.GetDimNum();

    OP_CHECK_IF(scaleDimNum != (yDimNum - 1),
        OP_LOGE(context_->GetNodeName(),
                                        "scale dimension should be only 1 less than y dimension, please check."),
        return ge::GRAPH_FAILED);

    for (size_t i = 0; i < scaleDimNum; i++) {
        OP_CHECK_IF(scaleShape[i] != yShape[i],
            OP_LOGE(context_->GetNodeName(),
                                            "scale shape[%zu] must be equal to y shape[%zu], please check.", i, i),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::GetAttr()
{
  auto* attrs = context_->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

  auto* attrActivateLeft = attrs->GetAttrPointer<bool>(ATTR_ACTIVATE_LEFT_INDEX);
  actRight_ = (attrActivateLeft == nullptr || *attrActivateLeft == false) ? 1 : 0;
  const char* attrQuantMode = attrs->GetAttrPointer<char>(ATTR_QUANT_MODE_INDEX);
  std::string quantMode = attrQuantMode == nullptr ? "static" : attrQuantMode;
  auto it = SUPPORT_QUANT_MODE.find(quantMode);
  OP_CHECK_IF(it == SUPPORT_QUANT_MODE.end(),
                  OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                      context_->GetNodeName(), "quant_mode",
                      quantMode.c_str(), "quant_mode only support [dynamic] or [static]"),
                  return ge::GRAPH_FAILED);
  quantMode_ = it->second;
  // 校验dst_type
  auto* attrDstType = attrs->GetAttrPointer<int>(ATTR_DST_TYPE_INDEX);
  // 类型校验,防止空指针
  dstType_ = (attrDstType != nullptr) ? *attrDstType : 2; // 默认是2，也即对应输出类型为int8
  OP_CHECK_IF(dstType_ != 2 && dstType_ != 34 && dstType_ != 35 && dstType_ != 36 && dstType_ != 40 && dstType_ != 41,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "dst_type",
                    std::to_string(dstType_).c_str(), "dst_type only support [2, 34, 35, 36, 40, 41]"),
                return ge::GRAPH_FAILED);
  // 校验round_mode
  const char* attrRoundMode = attrs->GetAttrPointer<char>(ATTR_ROUND_MODE_INDEX);
  std::string roundMode = attrRoundMode == nullptr ? "rint" : attrRoundMode;
  auto roundModeIt = SUPPORT_ROUND_MODE.find(roundMode);
  OP_CHECK_IF(roundModeIt == SUPPORT_ROUND_MODE.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "round_mode",
                    roundMode.c_str(), "round_mode only support [rint, round, floor, ceil, trunc]"),
                return ge::GRAPH_FAILED);
  roundMode_ = roundModeIt->second;
  // y:[int8, float8]，仅支持rint，y:[float4]，五种类型都支持
  auto yDesc = context_->GetOutputDesc(Y_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
  ge::DataType yDType = yDesc->GetDataType();
  // 校验y属于int8和float8时，roundMode是不是rint
  if (yDType != ge::DT_HIFLOAT8) {
    OP_CHECK_IF((yDType == ge::DT_INT8 || yDType == ge::DT_FLOAT8_E5M2 || yDType == ge::DT_FLOAT8_E4M3FN) && roundMode_ != 0,
                 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                     context_->GetNodeName(), "round_mode",
                     roundMode.c_str(), "round_mode only support [rint] when the type of y in [int8, float8]"),
                 return ge::GRAPH_FAILED);
  } else {
    // 校验y属于hifloat8时，roundMode是不是round
    OP_CHECK_IF(roundMode_ != 1,
                 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                     context_->GetNodeName(), "round_mode",
                     roundMode.c_str(), "round_mode only support [round] when the type of y in [hifloat8]"),
                 return ge::GRAPH_FAILED);
  }
  auto* attrSwigluMode = attrs->GetAttrPointer<int>(ATTR_SWIGLU_MODE_INDEX);
  swigluMode_ = (attrSwigluMode == nullptr) ? 0 : *attrSwigluMode;
  OP_CHECK_IF(swigluMode_ != 0 && swigluMode_ != 1,
              OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                  context_->GetNodeName(), "swigluMode",
                  std::to_string(swigluMode_).c_str(), "swigluMode only support [0, 1]"),
              return ge::GRAPH_FAILED);
  auto* attrClampLimit = attrs->GetAttrPointer<float>(ATTR_CLAMP_LIMIT_INDEX);
  clampLimit_ = (attrClampLimit == nullptr) ? CLAMP_LIMIT_DEFAULT : *attrClampLimit;
  OP_CHECK_IF(!(std::isfinite(clampLimit_) && clampLimit_ > 0.0),
              OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                  context_->GetNodeName(), "clamp_limit",
                  std::to_string(clampLimit_).c_str(), "clamp_limit should be positive finite"),
              return ge::GRAPH_FAILED);
  auto* attrGluAlpha = attrs->GetAttrPointer<float>(ATTR_GLU_ALPHA_INDEX);
  gluAlpha_ = (attrGluAlpha == nullptr) ? GLU_ALPHA_DEFAULT : *attrGluAlpha;
  auto* attrGluBias = attrs->GetAttrPointer<float>(ATTR_GLU_BIAS_INDEX);
  gluBias_ = (attrGluBias == nullptr) ? GLU_BIAS_DEFAULT : *attrGluBias;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::GetShapeAttrsInfo() {
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("DequantSwigluQuant", "context is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputX() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "get input x failed."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputGroupIndex() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "get input group_index failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAttrActivateDim() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "get attr activate_dim failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "get attr failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutputY() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "check output y failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputWeightScale() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "check input weight_scale failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputActScale() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "check input activation_scale failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputBias() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "check input bias failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputQuantScale() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "check input quant_scale failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputQuantOffset() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "check input quant_offset failed."), return ge::GRAPH_FAILED);
    if (quantMode_ == 0) {
        OP_CHECK_IF(CheckForStaticQuant() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "check input quant_scale and quant_offset size failed."),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(CheckOutputScale() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "check output scale failed."), return ge::GRAPH_FAILED);

    int64_t xTotalNum = xShape_.GetShapeSize();
    inDimy_ = xShape_.GetDim(xDimNum_ - 1);
    inDimx_ = xTotalNum / inDimy_;
    if ((speGroupType_ == 1) && (groupNum_ >= SPECIAL_GROUP_NUM_64) &&
        (inDimx_ / groupNum_ <= SPECIAL_GROUP_NUM_16) &&
            (activateDim_ == static_cast<int64_t>(xDimNum_-1))) {
      isSpecialCoreCut_ = 1;
    }
    if ((speGroupType_ == 0) && (groupNum_ >= SPECIAL_GROUP_NUM_32) &&
        (inDimx_ / groupNum_ <= SPECIAL_GROUP_NUM_16) &&
        (activateDim_ == static_cast<int64_t>(xDimNum_-1)) && (!hasBias_) && (!hasQuantScale_)) {
      isSpecialCoreCut_ = 1;
    }
    auto shapeY = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeY);
    const gert::Shape& outputShapeY = shapeY->GetStorageShape();
    outDimy_ = outputShapeY.GetDim(xDimNum_ - 1); // 输出y的-1轴对应的shape
    return ge::GRAPH_SUCCESS;
}

bool DequantSwigluQuantV35DskTiling::IsCapable() {
  if (!Ops::NN::OpTiling::IsRegbaseSocVersion(context_)) {
    return false;
  }
  if (static_cast<size_t>(activateDim_) != xDimNum_ - static_cast<size_t>(1)) {
    OP_LOGI(context_->GetNodeName(), "transform tiling template 2!");
    return false;
  }
  return true;
}

void DequantSwigluQuantV35DskTiling::CalcTilingKeyForNotFull() {
  // tilingkey含义：占位\quantMode_\bias\activate_scale\quant_scale\quant_offset\group
  if (quantMode_ == 0) {
    tilingKey_ = PLACEHOLDER + quantMode_ * QUANT_MODE_FACTOR  + hasBias_ * BIAS_FACTOR_FOR_NOT_FULL +
                 hasActivationScale_ * ACTIVATE_FACTOR_FOR_NOT_FULL + hasQuantScale_ * QUANT_SCALE_FACTOR_FOR_NOT_FULL +
                 hasQuantOffset_ * QUANT_OFFSET_FACTOR_FOR_NOT_FULL + hasGroupIndex_;
  } else {
    auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    int8_t biasDtypeValue = 0;
    int8_t value_int32 = 1;
    int8_t value_float = 2;
    int8_t vlaue_float16 = 3;
    int8_t value_bf16 = 4;
    if (biasDesc != nullptr) {
      ge::DataType biasDtype = biasDesc->GetDataType();
      if (biasDesc != nullptr) {
          if (biasDtype == ge::DT_INT32) {
              biasDtypeValue = value_int32;
          }
          if (biasDtype == ge::DT_BF16) {
              biasDtypeValue = value_bf16;
          }
          if (biasDtype == ge::DT_FLOAT16) {
              biasDtypeValue = vlaue_float16;
          }
          if (biasDtype == ge::DT_FLOAT) {
              biasDtypeValue = value_float;
          }
      }
    }
    // tilingkey含义：占位\quantMode_\bias\activate_scale\quant_scale\quant\offset
    tilingKey_ = PLACEHOLDER + quantMode_ * QUANT_MODE_FACTOR  + biasDtypeValue * BIAS_FACTOR_FOR_NOT_FULL +
                 hasActivationScale_ * ACTIVATE_FACTOR_FOR_NOT_FULL + hasQuantScale_ * QUANT_SCALE_FACTOR_FOR_NOT_FULL +
                 hasQuantOffset_ * QUANT_OFFSET_FACTOR_FOR_NOT_FULL + hasGroupIndex_;
  }
}

ge::graphStatus DequantSwigluQuantV35DskTiling::DoOpTilingNotFull() {
  /*
  参数             行尾轴大小            类型占用(bit)     可选/必选
  x                   2H                    32/16           必选
  weight_scale        2H                    32              可选(x为int32——必须，x为16——必须不存在)
  activation_scale    1                     32              可选(x为int32——可选，x为16——必须不存在)
  bias                2H                    32/16           可选(x为int32——可选，x为16——必须不存在)
  quant_scale         1/H                   32              可选(静态量化必选)
  quant_offset        1/H                   32              可选
  group_index         1                     64/32           可选(控制参数不占用UB)
  y                   H                     8/4             必选
  scale               1                     32              必选(静态量化不计算)
  tmp_buffer
  */
  int64_t ubFactorDimy = 1;
  int64_t ubAvailable = ubSize_ - UB_REVERSE - 1 * BLOCK_SIZE; // 预留1个block给activation_scale（无论activation_scale输入是否存在）
  auto yDesc = context_->GetOutputDesc(Y_INDEX);
  ge::DataType yDtype = yDesc->GetDataType();
  if (yDtype == ge::DT_FLOAT4_E2M1 or yDtype == ge::DT_FLOAT4_E1M2) {
    int64_t ubSplitNum = 1; // 以4bit类型的输出y为一份
    auto xDesc = context_->GetInputDesc(X_INDEX);
    ge::DataType xDtype = xDesc->GetDataType();
    ubSplitNum = xDtype == ge::DT_INT32 ? ubSplitNum + 16 : ubSplitNum + 8; // x尾轴2H，32bit类型占16倍空间，否则占8倍
    ubSplitNum = hasWeightScale_ ? ubSplitNum + 16 : ubSplitNum; // weight_scale尾轴2H，输入存在32bit类型占16倍空间，否则不额外占空间
    ubSplitNum = ubSplitNum + 8; // 静态量化场景quant_scale为必选输入，尾轴可能为1或H，按照H预留8倍空间
    ubSplitNum = hasQuantOffset_ ? ubSplitNum + 8 : ubSplitNum; // quant_offset尾轴可能为1或H，输入存在32bit类型占8倍空间，否则不占空间
    if (hasBias_) {
      auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
      ge::DataType biasDtype = biasDesc->GetDataType();
      ubSplitNum = (biasDtype == ge::DT_INT32 or biasDtype == ge::DT_FLOAT) ? ubSplitNum + 16 : ubSplitNum + 8; // bias尾轴2H，32bit类型占16倍空间，否则占8倍
    }
    ubSplitNum += 1; // y为4bit类型时，ub分配内存实际按照1B/num分配，多加一倍空间
    int64_t doubleBuffer = 2;
    ubSplitNum *= doubleBuffer; // x、weight_scale、bias、y doublebuffer
    ubSplitNum += 16; // tmp buffer尾轴2H，固定为32bit，占16倍空间

    int64_t ySize = (ubAvailable / ubSplitNum) / BLOCK_SIZE * BLOCK_SIZE; // 一份4bit类型y占用空间，此处32B对齐保证y的空间32B对齐，偶数

    ubFactorDimy = ySize / 0.5; // 非全载模板，单核内一次循环处理4bit类型y的元素个数（单个元素0.5Byte）
  } else {
    int64_t ubSplitNum = 1; // 以uint8类型的输出y为一份
    auto xDesc = context_->GetInputDesc(X_INDEX);
    ge::DataType xDtype = xDesc->GetDataType();
    ubSplitNum = xDtype == ge::DT_INT32 ? ubSplitNum + 8 : ubSplitNum + 4; // x尾轴2H，32bit类型占8倍空间，否则占4倍
    ubSplitNum = hasWeightScale_ ? ubSplitNum + 8 : ubSplitNum; // weight_scale尾轴2H，输入存在32bit类型占8倍空间，否则不额外占空间
    ubSplitNum = ubSplitNum + 4; // 静态量化场景quant_scale为必选输入，尾轴可能为1或H，按照H预留4倍空间
    ubSplitNum = hasQuantOffset_ ? ubSplitNum + 4 : ubSplitNum; // quant_offset尾轴可能为1或H，输入存在32bit类型占4倍空间，否则不占空间
    if (hasBias_) {
      auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
      ge::DataType biasDtype = biasDesc->GetDataType();
      ubSplitNum = (biasDtype == ge::DT_INT32 or biasDtype == ge::DT_FLOAT) ? ubSplitNum + 8 : ubSplitNum + 4; // bias尾轴2H，32bit类型占8倍空间，否则占4倍
    }
    int64_t doubleBuffer = 2;
    ubSplitNum *= doubleBuffer; // x、weight_scale、bias、y doublebuffer
    ubSplitNum += 8; // tmp buffer尾轴2H，固定为32bit，占8倍空间

    int64_t ySize = (ubAvailable / ubSplitNum) / BLOCK_SIZE * BLOCK_SIZE; // 一份8bit类型y占用空间，此处32B对齐保证y的空间32B对齐，偶数

    ubFactorDimy = ySize; // 非全载模板，单核内一次循环处理8bit类型y的元素个数（单个元素1Byte）
  }
  // 当实际输出尾轴长度小于搬运长度时的适配
  if (ubFactorDimy > outDimy_) {
    //获取一个block内y的元素个数，当y为4bit类型时(0.5Byte)，一个block内元素个数等于BLOCK_SIZE * 2, 否则y为8bit类型，一个block内元素个数等于BLOCK_SIZE
    int64_t numPerBlock = (yDtype == ge::DT_FLOAT4_E2M1 or yDtype == ge::DT_FLOAT4_E1M2) ? BLOCK_SIZE * 2 : BLOCK_SIZE;
    ubFactorDimy = Ops::Base::CeilDiv(outDimy_, numPerBlock) * numPerBlock;
  }

  int64_t loopTimesPerRow = (outDimy_ + ubFactorDimy - 1) / ubFactorDimy; // 非全载模板，单核处理一行需要的循环次数
  int64_t tailPerRow = outDimy_ - (loopTimesPerRow - 1) * ubFactorDimy; // 非全载模板，单核处理一行的尾块大小

  maxPreCore_ = std::min(static_cast<int64_t>(coreNum_), inDimx_); // 非全载模板，按照行数分核，单核内将一行分段处理

  CalcTilingKeyForNotFull();
  tilingData_.set_inDimx(inDimx_);
  tilingData_.set_inDimy(inDimy_);
  tilingData_.set_outDimy(outDimy_);
  tilingData_.set_UbFactorDimx(1); // 非全载模板固定1
  tilingData_.set_UbFactorDimy(ubFactorDimy); // 非全载模板ub内一次循环处理的尾轴元素个数
  tilingData_.set_usedCoreNum(maxPreCore_);
  tilingData_.set_maxCoreNum(coreNum_);
  tilingData_.set_inGroupNum(groupNum_);
  tilingData_.set_quantMode(quantMode_); // quantMode,0:静态量化,1:动态量化
  tilingData_.set_speGroupType(speGroupType_); 
  tilingData_.set_isSpecialCoreCut(isSpecialCoreCut_); 
  tilingData_.set_actRight(actRight_);
  tilingData_.set_dstType(dstType_);
  tilingData_.set_roundMode(roundMode_);
  tilingData_.set_activateDim(activateDim_);
  tilingData_.set_loopTimesPerRow(loopTimesPerRow); // 非全载模板处理一行需要的UB循环次数
  tilingData_.set_tailPerRow(tailPerRow); // 非全载模板UB循环最后一次的元素个数
  tilingData_.set_swiGluMode(swigluMode_); // swiGluMode
  tilingData_.set_biasMode(biasMode_); // bias类型，0：不存在；1：int32；2：bf16；3：fp16；4：fp32
  tilingData_.set_groupIndexMode(groupIndexMode_); // group_index类型，0：不存在；1：int32；2：int64
  tilingData_.set_quantIsOne(quantIsOne_);
  tilingData_.set_clampLimit(clampLimit_);
  tilingData_.set_gluAlpha(gluAlpha_);
  tilingData_.set_gluBias(gluBias_);

  OP_LOGI(context_->GetNodeName(), "inDimx is %ld, inDimy is %ld, outDimy is %ld, UbFactorDimx is %ld, UbFactorDimy is %ld, usedCoreNum is %ld, maxCoreNum is %ld, \
 	     inGroupNum is %ld, quantMode is %ld, actRight is %ld, dstType is %ld, roundMode is %ld, activateDim is %ld, loopTimesPerRow is %ld, \
 	     tailPerRow is %ld, swiGluMode is %ld, biasMode is %ld, groupIndexMode is %ld, quantIsOne is %ld, clampLimit is %f, gluAlpha is %f, gluBias is %f", \
       tilingData_.get_inDimx(), tilingData_.get_inDimy(), tilingData_.get_outDimy(), tilingData_.get_UbFactorDimx(), tilingData_.get_UbFactorDimy(), \
 	     tilingData_.get_usedCoreNum(), tilingData_.get_maxCoreNum(), tilingData_.get_inGroupNum(), tilingData_.get_quantMode(), tilingData_.get_actRight(), tilingData_.get_dstType(), \
 	     tilingData_.get_roundMode(), tilingData_.get_activateDim(), tilingData_.get_loopTimesPerRow(), tilingData_.get_tailPerRow(), tilingData_.get_swiGluMode(), \
       tilingData_.get_biasMode(), tilingData_.get_groupIndexMode(), tilingData_.get_quantIsOne(), tilingData_.get_clampLimit(), tilingData_.get_gluAlpha(), tilingData_.get_gluBias());
  OP_LOGI(context_->GetNodeName(), "tilingKey_ is %ld, speGroupType is %ld, isSpecialCoreCut is %ld", tilingKey_, speGroupType_, isSpecialCoreCut_);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::DoOpTiling() {
  if (outDimy_ > Y_LAST_DIM_FULL_LOAD_MAX_VALUE) {
    return DequantSwigluQuantV35DskTiling::DoOpTilingNotFull();
  }
  auto xDesc = context_->GetInputDesc(X_INDEX);
  ge::DataType xDtype = xDesc->GetDataType();
  size_t xBits = xDtype == ge::DT_INT32 ? sizeof(int32_t) : sizeof(int16_t);
  int64_t xUbAlign32B = ((outDimy_ + BLOCK_ELEM_B32 - 1) / BLOCK_ELEM_B32) * BLOCK_ELEM_B32;
  int64_t xUbAlign = xDtype == ge::DT_INT32 ? xUbAlign32B :
                     ((outDimy_ + BLOCK_ELEM_B16 - 1) / BLOCK_ELEM_B16) * BLOCK_ELEM_B16;
  int64_t aScaleAlign32B_ = BLOCK_ELEM_B32;
  int64_t yAlign8B = ((outDimy_ + BLOCK_ELEM_B8 - 1) / BLOCK_ELEM_B8) * BLOCK_ELEM_B8;
  int64_t doubleBuffer = 2;
  int64_t ubAvailable = ubSize_ - UB_REVERSE - (xUbAlign32B * SWI_FACTOR + xUbAlign32B) * sizeof(float);
  int64_t denominator = doubleBuffer * (xUbAlign * SWI_FACTOR * xBits + aScaleAlign32B_ * sizeof(float)) +
                        doubleBuffer * yAlign8B* sizeof(int8_t) + aScaleAlign32B_ * sizeof(float) +
                        xUbAlign32B * sizeof(float);
  
  // swiglu_mode=1时，增加x weight_scale的尾轴128向上对齐ub
  if (swigluMode_ == 1) {
    int64_t tailSupply = (inDimy_ + 128 - 1) / 128 * 128 - inDimy_;
    denominator += tailSupply * xBits + tailSupply * sizeof(int32_t);
  }

  // 判断bias, bias=nullptr：biasDtypeValue = 0, 否则，biasDtypeValue = 1
  auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
  int64_t biasDtypeValue = 0;
  int64_t value_int32 = 2;
  int64_t value_bf16 = 4;
  int64_t vlaue_float16 = 3;
  int64_t value_float = 2;
  // 判断bias是否合法存在，如果存在的话，则需要考虑bias在ub里面占用的内存
  if (biasDesc != nullptr) {
    ge::DataType biasDtype = biasDesc->GetDataType();
    int64_t biasMemory = xUbAlign32B * sizeof(int16_t); // base为4B类型
    // swigluV2场景下，增加bias的尾块128B对齐，防止越界
    int64_t biasTailSupply = (inDimy_ + 128 - 1) / 128 * 128 - inDimy_;
    if (biasDesc != nullptr) {
      if (biasDtype == ge::DT_INT32) {
        denominator += biasMemory * value_int32;
        biasDtypeValue = 1;
        if (swigluMode_ == 1) {
          denominator += biasTailSupply * sizeof(int32_t);
        }
      }
      if (biasDtype == ge::DT_BF16) {
        denominator += biasMemory;
        biasDtypeValue = value_bf16;
        if (swigluMode_ == 1) {
          denominator += biasTailSupply * sizeof(int16_t);
        }
      }
      if (biasDtype == ge::DT_FLOAT16) {
        denominator += biasMemory;
        biasDtypeValue = vlaue_float16;
        if (swigluMode_ == 1) {
          denominator += biasTailSupply * sizeof(int16_t);
        }
      }
      if (biasDtype == ge::DT_FLOAT) {
        denominator += biasMemory * SWI_FACTOR;
        biasDtypeValue = value_float;
        if (swigluMode_ == 1) {
          denominator += biasTailSupply * sizeof(int32_t);
        }
      }
    }
  }

  // 静态量化下，增加quant_offset的UB
  if (hasQuantOffset_) {
    denominator += xUbAlign32B * sizeof(float);
  }

  // ubFactorDimX: ub最多可以处理多少行数据
  int64_t ubFactorDimx = ubAvailable / denominator;
  ubFactorDimx = std::min(ubFactorDimx, inDimx_);
  OP_CHECK_IF(ubFactorDimx < 1,
        OP_LOGE(context_->GetNodeName(), "x last dim:%ld is too large to full load", inDimy_),
        return ge::GRAPH_FAILED);
  maxPreCore_ = std::min(maxPreCore_, (inDimx_ + ubFactorDimx - 1) / ubFactorDimx);
  OP_LOGI(context_->GetNodeName(), "start maxPreCore_ is %ld ", maxPreCore_);
  if (isSpecialCoreCut_ == static_cast<int64_t>(1)) {
      maxPreCore_ = std::min(static_cast<int64_t>(coreNum_), static_cast<int64_t>(inDimx_));
      OP_LOGI(context_->GetNodeName(), "after maxPreCore_ is %ld ", maxPreCore_);
  }
  
  auto quantScaleDesc = context_->GetOptionalInputDesc(QUANT_SCALE_INDEX);
  auto actScaleDesc = context_->GetOptionalInputDesc(ACTIVATION_SCALE_INDEX);
  auto groupIndexDesc = context_->GetOptionalInputDesc(INPUT_GROUP_INDEX);
  
  // 输入x的tiling key计算位
  int64_t hasXInt = quantMode_ == 1 ? 0: 1;
  int64_t hasAScale = actScaleDesc != nullptr;
  // hasQScale=0:无quant_scale，hasQScale=1:float
  int64_t hasQScale = quantScaleDesc != nullptr;
  int64_t hasGIndex = groupIndexDesc != nullptr;
  // activateDim=-1时，hasActivateDim=0，否则为1; 当activateDim=xDim-1时，hashActivateDim=1
  int64_t hasActivateDim = static_cast<size_t>(activateDim_) != xDimNum_ - static_cast<size_t>(1);
  
  // 增加十万分位的tiling_key，hasActivateDim;增加万分位的tilling_key hasXInt:判断输入x是不是int32;千分位的tiling_key biasDtypeValue
  tilingKey_ = hasActivateDim * ACTIVATE_DIM_FACTOR + hasXInt * INPUT_X_FACTOR + biasDtypeValue * BIAS_FACTOR + hasAScale * ACT_SCALE_FACTOR + hasQScale * QUANT_SCALE_FACTOR + hasGIndex * GROUP_INDEX_FACTOR;
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
  tilingData_.set_dstType(dstType_);
  tilingData_.set_roundMode(roundMode_);
  tilingData_.set_activateDim(activateDim_);
  tilingData_.set_swiGluMode(swigluMode_); // swiGluMode
  tilingData_.set_biasMode(biasMode_); // bias类型，0：不存在；1：int32；2：bf16；3：fp16；4：fp32
  tilingData_.set_groupIndexMode(groupIndexMode_); // group_index类型，0：不存在；1：int32；2：int64
  tilingData_.set_quantIsOne(quantIsOne_);
  tilingData_.set_clampLimit(clampLimit_);
  tilingData_.set_gluAlpha(gluAlpha_);
  tilingData_.set_gluBias(gluBias_);
  tilingData_.set_speGroupType(speGroupType_); 
  tilingData_.set_isSpecialCoreCut(isSpecialCoreCut_);
  OP_LOGI(context_->GetNodeName(), "inDimx is %ld, inDimy is %ld, outDimy is %ld, UbFactorDimx is %ld, UbFactorDimy is %ld, usedCoreNum is %ld, maxCoreNum is %ld, \
 	     inGroupNum is %ld, quantMode is %ld, actRight is %ld, dstType is %ld, roundMode is %ld, activateDim is %ld, swiGluMode is %ld, \
 	     biasMode is %ld, groupIndexMode is %ld, biasMode is %ld, groupIndexMode is %ld, quantIsOne is %ld, clampLimit is %f, gluAlpha is %f, gluBias is %f", \
       tilingData_.get_inDimx(), tilingData_.get_inDimy(), tilingData_.get_outDimy(), tilingData_.get_UbFactorDimx(), tilingData_.get_UbFactorDimy(), \
 	     tilingData_.get_usedCoreNum(), tilingData_.get_maxCoreNum(), tilingData_.get_inGroupNum(), tilingData_.get_quantMode(), tilingData_.get_actRight(), tilingData_.get_dstType(), \
 	     tilingData_.get_roundMode(), tilingData_.get_activateDim(), tilingData_.get_swiGluMode(), tilingData_.get_biasMode(), tilingData_.get_groupIndexMode(), \
       tilingData_.get_biasMode(), tilingData_.get_groupIndexMode(), tilingData_.get_quantIsOne(), tilingData_.get_clampLimit(), tilingData_.get_gluAlpha(), tilingData_.get_gluBias());
  OP_LOGI(context_->GetNodeName(), "tilingKey_ is %ld, speGroupType is %ld, isSpecialCoreCut is %ld", tilingKey_, speGroupType_, isSpecialCoreCut_);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::DoLibApiTiling() {
  return ge::GRAPH_SUCCESS;
}

uint64_t DequantSwigluQuantV35DskTiling::GetTilingKey() const {
  return tilingKey_;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::GetWorkspaceSize() {
   // 如果是动态非全载，需要使用workspace存Dequant Swiglu的计算结果
  size_t usrSize = 0;
  if (outDimy_ > Y_LAST_DIM_FULL_LOAD_MAX_VALUE && quantMode_ == 1) {
    usrSize = maxPreCore_ * outDimy_ * sizeof(float);
  }
  OP_LOGI(context_->GetNodeName(), "usrSize is %u", usrSize);
  workspaceSize_ = SYS_WORK_SPACE_SIZE + usrSize;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35DskTiling::PostTiling() {
  context_->SetTilingKey(GetTilingKey());
  context_->SetBlockDim(maxPreCore_);
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
  workspaces[0] = workspaceSize_;
  OP_LOGI(context_->GetNodeName(), "workspace is %lu, SetBlockDim is %ld", workspaceSize_, maxPreCore_);
  tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35NlastTiling::GetPlatformInfo()
{
  auto platformInfo = context_->GetPlatformInfo();
  if (platformInfo == nullptr) {
    auto compileInfoPtr = static_cast<const DequantSwigluQuantCompileInfo*>(context_->GetCompileInfo());
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

  return ge::GRAPH_SUCCESS;
}

void DequantSwigluQuantV35NlastTiling::FusedShape()
{
  inDim0_ = 1;
  inDim1_ = 1;
  inDim2_ = 1;
  for (size_t i = 0; i < xShape_.GetDimNum(); i++) {
    if (i < static_cast<size_t>(actDimIndex_)) {
      inDim0_ *= xShape_.GetDim(i);
    } else if (i == xShape_.GetDimNum() - 1) {
      inDim2_ *= xShape_.GetDim(i);
    } else {
      inDim1_ *= xShape_.GetDim(i);
    }
  }
  return;
}

ge::graphStatus DequantSwigluQuantV35NlastTiling::GetShapeAttrsInfo()
{
  auto xStorageShape = context_->GetInputShape(X_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, xStorageShape);
  xShape_ = EnsureNotScalar(xStorageShape->GetStorageShape());

  auto* attrs = context_->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
  auto* attrActivateLeft = attrs->GetAttrPointer<bool>(ATTR_ACTIVATE_LEFT_INDEX);
  actRight_ = (attrActivateLeft == nullptr || *attrActivateLeft == false) ? 1 : 0;

  const char* attrRoundMode = attrs->GetAttrPointer<char>(ATTR_ROUND_MODE_INDEX);
  std::string roundMode = attrRoundMode == nullptr ? "rint" : attrRoundMode;
  // has checked
  roundMode_ = SUPPORT_ROUND_MODE.find(roundMode)->second;

  auto* attrActivateDim = attrs->GetAttrPointer<int>(ATTR_ACTIVATE_DIM_INDEX);
  actDimIndex_ = (attrActivateDim != nullptr) ? *attrActivateDim : -1;
  actDimIndex_ = actDimIndex_ < 0 ? actDimIndex_ + static_cast<int32_t>(xShape_.GetDimNum()) : actDimIndex_;

  FusedShape();

  outDim1_ = inDim1_ / SWI_FACTOR;

  return ge::GRAPH_SUCCESS;
}

bool DequantSwigluQuantV35NlastTiling::IsCapable()
{
  if (!Ops::NN::OpTiling::IsRegbaseSocVersion(context_)) {
    return false;
  }
  if (static_cast<size_t>(actDimIndex_) == xShape_.GetDimNum() - 1) {
    return false;
  }

  return true;
}

void DequantSwigluQuantV35NlastTiling::DoBlockSplit()
{
  int64_t maxCoreNum = static_cast<int64_t>(coreNum_);
  blockFormer0_ = (inDim0_ + maxCoreNum - 1) / maxCoreNum;
  blockNum0_ = (inDim0_ + blockFormer0_ - 1) / blockFormer0_;
  int64_t needBlockNum1 = maxCoreNum / blockNum0_;
  blockFormer1_ = (outDim1_ + needBlockNum1 - 1) / needBlockNum1;
  blockNum1_ = (outDim1_ + blockFormer1_ - 1) / blockFormer1_;
}

bool DequantSwigluQuantV35NlastTiling::DoUbSplit()
{
  int64_t xUbAlign32B = ((inDim2_ + BLOCK_ELEM_B32 - 1) / BLOCK_ELEM_B32) * BLOCK_ELEM_B32;
  int64_t ubAvailable = ubSize_ - UB_REVERSE - (xUbAlign32B + xUbAlign32B) * sizeof(float);
  int64_t doubleBuffer = 2;
  int64_t aScaleAlign32B = BLOCK_ELEM_B32;
  int64_t yAlign8B = ((inDim2_ + BLOCK_ELEM_B8 - 1) / BLOCK_ELEM_B8) * BLOCK_ELEM_B8;
  auto xDesc = context_->GetInputDesc(X_INDEX);
  size_t xBits = xDesc->GetDataType() == ge::DT_INT32 ? sizeof(int32_t) : sizeof(int16_t);
  int64_t xUbAlign = xDesc->GetDataType() == ge::DT_INT32 ? xUbAlign32B :
                     ((inDim2_ + BLOCK_ELEM_B16 - 1) / BLOCK_ELEM_B16) * BLOCK_ELEM_B16;
  int64_t denominator = doubleBuffer * (xUbAlign * SWI_FACTOR * xBits +
                                        aScaleAlign32B * SWI_FACTOR * sizeof(float)) +
                        doubleBuffer * yAlign8B * sizeof(int8_t) + aScaleAlign32B * sizeof(float) +
                        xUbAlign32B * sizeof(float);
  auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
  biasDtypeValue_ = 0;
  int64_t value_int32 = 2;
  int64_t value_bf16 = 4;
  int64_t vlaue_float16 = 3;
  int64_t value_float = 2;
  if (biasDesc != nullptr) {
    ge::DataType biasDtype = biasDesc->GetDataType();
    int64_t biasMemory = xUbAlign32B * sizeof(int16_t);
    if (biasDtype == ge::DT_INT32) {
      denominator += biasMemory * value_int32;
      biasDtypeValue_ = 1;
    }
    if (biasDtype == ge::DT_BF16) {
      denominator += biasMemory;
      biasDtypeValue_ = value_bf16;
    }
    if (biasDtype == ge::DT_FLOAT16) {
      denominator += biasMemory;
      biasDtypeValue_ = vlaue_float16;
    }
    if (biasDtype == ge::DT_FLOAT) {
      denominator += biasMemory * SWI_FACTOR;
      biasDtypeValue_ = value_float;
    }
  }
  ubFormer1_ = ubAvailable / denominator;
  if (ubFormer1_ < 1) {
    return false;
  }
  ubFormer0_ = 1;
  if (ubFormer1_ > blockFormer1_) {
    ubFormer0_ = ubFormer1_ / blockFormer1_;
    ubFormer1_ = blockFormer1_;
  }

  ubFormer0_ = std::min(ubFormer0_, blockFormer0_);

  return true;
}

ge::graphStatus DequantSwigluQuantV35NlastTiling::DoOpTiling()
{
  DoBlockSplit();
  if (!DoUbSplit()) {
    OP_LOGE(context_->GetNodeName(), "UB size cannot load last dim of input x, return failed.");
    return ge::GRAPH_FAILED;
  }
  int64_t ubLoopOfFormerBlock0 = (blockFormer0_ + ubFormer0_ - 1) / ubFormer0_;
  int64_t blockTail0 = inDim0_ - blockFormer0_ * (blockNum0_ - 1);
  int64_t ubLoopOfTailBlock0 = (blockTail0 + ubFormer0_ - 1) / ubFormer0_;
  int64_t ubTailOfFormerBlock0 = blockFormer0_ - (ubLoopOfFormerBlock0 - 1) * ubFormer0_;
  int64_t ubTailOfTailBlock0 = blockTail0 - (ubLoopOfTailBlock0 - 1) * ubFormer0_;

  int64_t ubLoopOfFormerBlock1 = (blockFormer1_ + ubFormer1_ - 1) / ubFormer1_;
  int64_t blockTail1 = outDim1_ - blockFormer1_ * (blockNum1_ - 1);
  int64_t ubLoopOfTailBlock1 = (blockTail1 + ubFormer1_ - 1) / ubFormer1_;
  int64_t ubTailOfFormerBlock1 = blockFormer1_ - (ubLoopOfFormerBlock1 - 1) * ubFormer1_;
  int64_t ubTailOfTailBlock1 = blockTail1 - (ubLoopOfTailBlock1 - 1) * ubFormer1_;

  blockNum_ = blockNum0_ * blockNum1_;
  tilingData_.set_inDim0(inDim0_);
  tilingData_.set_inDim1(inDim1_);
  tilingData_.set_inDim2(inDim2_);
  tilingData_.set_outDim1(inDim1_ / SWI_FACTOR);
  tilingData_.set_blockNum0(blockNum0_);
  tilingData_.set_blockNum1(blockNum1_);
  tilingData_.set_blockFormer0(blockFormer0_);
  tilingData_.set_blockFormer1(blockFormer1_);
  tilingData_.set_ubFormer0(ubFormer0_);
  tilingData_.set_ubFormer1(ubFormer1_);
  tilingData_.set_ubLoopOfFormerBlock0(ubLoopOfFormerBlock0);
  tilingData_.set_ubLoopOfFormerBlock1(ubLoopOfFormerBlock1);
  tilingData_.set_ubLoopOfTailBlock0(ubLoopOfTailBlock0);
  tilingData_.set_ubLoopOfTailBlock1(ubLoopOfTailBlock1);
  tilingData_.set_ubTailOfFormerBlock0(ubTailOfFormerBlock0);
  tilingData_.set_ubTailOfFormerBlock1(ubTailOfFormerBlock1);
  tilingData_.set_ubTailOfTailBlock0(ubTailOfTailBlock0);
  tilingData_.set_ubTailOfTailBlock1(ubTailOfTailBlock1);
  tilingData_.set_actRight(actRight_);
  tilingData_.set_roundMode(roundMode_);

  auto quantScaleDesc = context_->GetOptionalInputDesc(QUANT_SCALE_INDEX);
  auto actScaleDesc = context_->GetOptionalInputDesc(ACTIVATION_SCALE_INDEX);

  int64_t hasAScale = actScaleDesc != nullptr;
  int64_t hasQScale = quantScaleDesc != nullptr;

  tilingKey_ = 1 * ACTIVATE_DIM_FACTOR + biasDtypeValue_ * BIAS_FACTOR + hasAScale * ACT_SCALE_FACTOR + 
               hasQScale * QUANT_SCALE_FACTOR + 0 * GROUP_INDEX_FACTOR;

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35NlastTiling::DoLibApiTiling() {
  return ge::GRAPH_SUCCESS;
}

uint64_t DequantSwigluQuantV35NlastTiling::GetTilingKey() const {
  return tilingKey_;
}

ge::graphStatus DequantSwigluQuantV35NlastTiling::GetWorkspaceSize() {
  workspaceSize_ = WORKSPACE_SIZE;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantV35NlastTiling::PostTiling() {
  context_->SetTilingKey(GetTilingKey());
  context_->SetBlockDim(blockNum_);
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
  workspaces[0] = workspaceSize_;
  OP_LOGI(context_->GetNodeName(), "SetBlockDim is %ld", blockNum_);
  tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("DequantSwigluQuant", DequantSwigluQuantV35DskTiling, 1000);
REGISTER_TILING_TEMPLATE("DequantSwigluQuant", DequantSwigluQuantV35NlastTiling, 2000);

REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_0, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_10, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_11, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_2111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_3111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_4111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_2110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_3110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_4110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_2101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_3101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_4101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_2100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_3100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_4100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1011, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_2011, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_3011, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_4011, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1010, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_2010, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_3010, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_4010, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_2001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_3001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_4001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_2000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_3000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_4000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100110, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100100, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100010, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100000, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_101110, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_102110, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_103110, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_104110, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_101100, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_102100, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_103100, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_104100, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_101010, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_102010, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_103010, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_104010, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_101000, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_102000, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_103000, DequantSwigluQuantV35NlastTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_104000, DequantSwigluQuantV35NlastTilingData)
}  // namespace optiling
