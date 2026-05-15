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
 * \file dequant_swiglu_quant_tiling.h
 * \brief
 */

#ifndef DEQUANT_SWIGLU_QUANT_TILING_H
#define DEQUANT_SWIGLU_QUANT_TILING_H


#include <vector>
#include <iostream>
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "dequant_swiglu_quant_proto.h"
#include "../tiling_base/tiling_base.h"
#include "../tiling_base/tiling_templates_registry.h"

using namespace Ops::NN::Optiling;
namespace optiling
{
using Ops::NN::Optiling::TilingBaseClass;
BEGIN_TILING_DATA_DEF(DequantSwigluQuantBaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, inDimx);
TILING_DATA_FIELD_DEF(int64_t, inDimy);
TILING_DATA_FIELD_DEF(int64_t, outDimy);
TILING_DATA_FIELD_DEF(int64_t, UbFactorDimx);
TILING_DATA_FIELD_DEF(int64_t, UbFactorDimy);  // cut for output dim
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, maxCoreNum);
TILING_DATA_FIELD_DEF(int64_t, inGroupNum);
TILING_DATA_FIELD_DEF(int64_t, hasBias);
TILING_DATA_FIELD_DEF(int64_t, quantMode);
TILING_DATA_FIELD_DEF(int64_t, actRight);
TILING_DATA_FIELD_DEF(int64_t, quantScaleDtype);
TILING_DATA_FIELD_DEF(int64_t, groupIndexDtype);
TILING_DATA_FIELD_DEF(int64_t, needSmoothScale);
TILING_DATA_FIELD_DEF(int64_t, biasDtype);
TILING_DATA_FIELD_DEF(int64_t, speGroupType);
TILING_DATA_FIELD_DEF(int64_t, activationScaleIsEmpty);
TILING_DATA_FIELD_DEF(int64_t, quantIsOne);
// data field for SwiGLU used by GPT-OSS
TILING_DATA_FIELD_DEF(int64_t, swigluMode);
TILING_DATA_FIELD_DEF(float, clampLimit);
TILING_DATA_FIELD_DEF(float, gluAlpha);
TILING_DATA_FIELD_DEF(float, gluBias);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100000000, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100001000, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100002000, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100003000, DequantSwigluQuantBaseTilingData)

REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100000100, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100001100, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100002100, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100003100, DequantSwigluQuantBaseTilingData)

REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100000200, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100001200, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100002200, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_100003200, DequantSwigluQuantBaseTilingData)

REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_200000000, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_200000100, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_200000200, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_110000000, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_110000100, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_110000200, DequantSwigluQuantBaseTilingData)

BEGIN_TILING_DATA_DEF(DequantSwigluQuantV35BaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, inDimx);
TILING_DATA_FIELD_DEF(int64_t, inDimy);
TILING_DATA_FIELD_DEF(int64_t, outDimy);
TILING_DATA_FIELD_DEF(int64_t, UbFactorDimx);
TILING_DATA_FIELD_DEF(int64_t, UbFactorDimy);  // cut for output dim
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, maxCoreNum);
TILING_DATA_FIELD_DEF(int64_t, inGroupNum);
TILING_DATA_FIELD_DEF(int64_t, quantMode);
TILING_DATA_FIELD_DEF(int64_t, actRight); // swish的激活与门控左右排布情况下生效，1表示右半部为激活
TILING_DATA_FIELD_DEF(int64_t, dstType);
TILING_DATA_FIELD_DEF(int64_t, roundMode);
TILING_DATA_FIELD_DEF(int64_t, activateDim);
TILING_DATA_FIELD_DEF(int64_t, loopTimesPerRow); // 非全载模板下处理一行需要的UB循环次数
TILING_DATA_FIELD_DEF(int64_t, tailPerRow); // 非全载模板UB循环最后一次的元素个数
TILING_DATA_FIELD_DEF(int64_t, swiGluMode); // 0表示swish的激活与门控左右排布，1表示奇偶排布
TILING_DATA_FIELD_DEF(int64_t, biasMode); // bias类型，0：不存在；1：int32；2：int64
TILING_DATA_FIELD_DEF(int64_t, groupIndexMode); // group_index类型，0：不存在；1：int32；2：int64
TILING_DATA_FIELD_DEF(int64_t, quantIsOne); // kernel侧计算时quant尾轴是否为单个元素
TILING_DATA_FIELD_DEF(int64_t, speGroupType); //groupidx是否2维
TILING_DATA_FIELD_DEF(int64_t, isSpecialCoreCut);  // 是否多专家少token场景
TILING_DATA_FIELD_DEF(float, clampLimit);
TILING_DATA_FIELD_DEF(float, gluAlpha);
TILING_DATA_FIELD_DEF(float, gluBias);
END_TILING_DATA_DEF;

// static quant full
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_10000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_10001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_10010, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_10011, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_10100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_10101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_10110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_10111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_11111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_12111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_13111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_14111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_11110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_12110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_13110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_14110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_11101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_12101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_13101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_14101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_11100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_12100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_13100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_14100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_11011, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_12011, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_13011, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_14011, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_11010, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_12010, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_13010, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_14010, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_11001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_12001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_13001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_14001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_11000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_12000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_13000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_14000, DequantSwigluQuantV35BaseTilingData)
// static quant not full
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1000000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1000010, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1000100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1000110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1001000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1001010, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1001100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1001110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1010000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1010010, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1010100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1010110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1011000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1011010, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1011100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1011110, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1000001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1000011, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1000101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1000111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1001001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1001011, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1001101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1001111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1010001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1010011, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1010101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1010111, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1011001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1011011, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1011101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1011111, DequantSwigluQuantV35BaseTilingData)
// ## dynamic
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1100000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1100001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1100100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1100101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1101000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1101001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1101100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1101101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1110000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1110001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1110100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1110101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1111000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1111001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1111100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1111101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1120000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1120001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1120100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1120101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1121000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1121001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1121100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1121101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1130000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1130001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1130100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1130101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1131000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1131001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1131100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1131101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1140000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1140001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1140100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1140101, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1141000, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1141001, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1141100, DequantSwigluQuantV35BaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1141101, DequantSwigluQuantV35BaseTilingData)

BEGIN_TILING_DATA_DEF(DequantSwigluQuantV35NlastTilingData)
TILING_DATA_FIELD_DEF(int64_t, inDim0);
TILING_DATA_FIELD_DEF(int64_t, inDim1);
TILING_DATA_FIELD_DEF(int64_t, inDim2);
TILING_DATA_FIELD_DEF(int64_t, outDim1);
TILING_DATA_FIELD_DEF(int64_t, blockNum0);
TILING_DATA_FIELD_DEF(int64_t, blockNum1);
TILING_DATA_FIELD_DEF(int64_t, blockFormer0);
TILING_DATA_FIELD_DEF(int64_t, blockFormer1);
TILING_DATA_FIELD_DEF(int64_t, ubFormer0);
TILING_DATA_FIELD_DEF(int64_t, ubFormer1);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfFormerBlock0);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfFormerBlock1);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfTailBlock0);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfTailBlock1);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfFormerBlock0);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfFormerBlock1);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfTailBlock0);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfTailBlock1);
TILING_DATA_FIELD_DEF(int64_t, actRight);
TILING_DATA_FIELD_DEF(int64_t, roundMode);
END_TILING_DATA_DEF;

struct DequantSwigluQuantCompileInfo {
  uint64_t coreNum = 0;
  uint64_t ubSize = 0;
};

class DequantSwigluQuantDskTiling : public TilingBaseClass
{
  public:
  explicit DequantSwigluQuantDskTiling(gert::TilingContext* tilingContext) : TilingBaseClass(tilingContext)
  {
  }
  ~DequantSwigluQuantDskTiling() override
  {
  }
  uint64_t coreNum_ = 0;
  uint64_t ubSize_ = 0;
  int64_t groupNum_ = 0;
  int64_t actRight_ = 0;
  int64_t quantMode_ = 0;
  uint64_t workspaceSize_ = 0;
  int64_t maxPreCore_ = 0;
  bool hasWeightScale_ = false;
  bool hasActivationScale_ = false;
  bool hasBias_ = false;
  bool hasQuantScale_ = false;
  bool hasQuantOffset_ = false;
  bool hasGroupIndex_ = false;
  bool speGroupType_ = false;

  // variable for SwiGLU used by GPT-OSS
  int64_t swigluMode_ = 0;
  float clampLimit_ = 0.0;
  float gluAlpha_ = 0.0;
  float gluBias_ = 0.0;
  
  protected:
  bool IsCapable() override;
  ge::graphStatus GetPlatformInfo() override;
  ge::graphStatus GetShapeAttrsInfo() override;
  ge::graphStatus DoOpTiling() override;
  ge::graphStatus DoLibApiTiling() override;
  uint64_t GetTilingKey() const override;
  ge::graphStatus GetWorkspaceSize() override;
  ge::graphStatus PostTiling() override;
  void DumpTilingInfo() override;
  ge::graphStatus GetAttr();
  ge::graphStatus CheckBias();
  ge::graphStatus CheckWeightScale();
  ge::graphStatus CheckActivationScale();
  ge::graphStatus CheckXAndGroupIndexDtype();
  ge::graphStatus CheckForDequant();
  ge::graphStatus CheckForQuant();
  ge::graphStatus CheckForDynamicQuant();
  ge::graphStatus CheckForStaticQuant();
  ge::graphStatus CheckQuantScaleDtype();
  ge::graphStatus CheckStaticQuantShape(const int64_t quantInputIdx, int64_t& colLen, const char* paramName);
  ge::graphStatus CheckIllegalParam();
  void CountTilingKey();
  ge::graphStatus CountMaxDim(int64_t& ubFactorDimx);
  ge::graphStatus CheckScaleShapeWithDim(const int64_t scaleInputIdx, const int64_t expectDim, const char* paramName);
  bool IsPerformanceAndGroupIndexBrach();
  ge::graphStatus GetShapeAttrsInfoInner();
  static bool CheckOptionalShapeExisting(const gert::StorageShape* storageShape);

  private:
  uint64_t tilingKey_ = 0;
  DequantSwigluQuantBaseTilingData tilingData_;
  int64_t inDimx_ = 0;
  int64_t inDimy_ = 0;
  int64_t outDimy_ = 0;
  platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

template <typename T>
inline auto AlignUp(T num, T rnd) -> decltype(num)
{
  return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}
// align num to multiples of rnd, round down
template <typename T>
inline auto AlignDown(T num, T rnd) -> decltype(num)
{
  return ((((rnd) == 0) || ((num) < (rnd))) ? 0 : ((num) / (rnd) * (rnd)));
}

template <typename T>
inline auto DivCeil(T num, T div) -> decltype(num)
{
  return (((div) == 0) ? 0 : (((num) + (div)-1) / (div)));
}

inline bool GetLengthByType(int32_t dtype, uint32_t& dsize)
{
  switch (dtype) {
    case ge::DT_FLOAT16:
    case ge::DT_INT16:
    case ge::DT_UINT16:
    case ge::DT_BF16:
      dsize = sizeof(int16_t);
      return true;
    case ge::DT_FLOAT:
    case ge::DT_INT32:
    case ge::DT_UINT32:
      dsize = sizeof(int32_t);
      return true;
    case ge::DT_DOUBLE:
    case ge::DT_INT64:
    case ge::DT_UINT64:
      dsize = sizeof(int64_t);
      return true;
    default:
      return false;
  }
}

class DequantSwigluQuantV35DskTiling : public TilingBaseClass {
  public:
    explicit DequantSwigluQuantV35DskTiling(gert::TilingContext* tilingContext) : TilingBaseClass(tilingContext) {
    }
    ~DequantSwigluQuantV35DskTiling() override {
    }
    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    int64_t actRight_ = 0;
    int64_t quantMode_ = 0;
    uint64_t workspaceSize_ = 0;
    int64_t maxPreCore_ = 0;
    int64_t groupNum_ = 0;
    int64_t biasMode_ = 0;
    int64_t groupIndexMode_ = 0;
    int64_t swigluMode_ = 0;
    int64_t speGroupType_ = 0;
    int64_t isSpecialCoreCut_ = 0;
    float clampLimit_ = 0;
    float gluAlpha_ = 1.702;
    float gluBias_ = 1.0;
    bool hasWeightScale_ = false;
    bool hasActivationScale_ = false;
    bool hasBias_ = false;
    bool hasQuantScale_ = false;
    bool hasQuantOffset_ = false;
    bool quantIsOne_ = true;
    bool hasGroupIndex_ = false;
    gert::Shape xShape_ = gert::Shape();
    size_t xDimNum_ = 0;
    gert::Shape groupIndexShape_ = gert::Shape();
    int64_t dstType_ = 2;
    int64_t roundMode_ = 0;
    int64_t activateDim_ = -1UL;
  
    protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetAttr();
    ge::graphStatus GetInputX();
    ge::graphStatus GetAttrActivateDim();
    ge::graphStatus CheckInputWeightScale();
    ge::graphStatus CheckInputActScale();
    ge::graphStatus CheckInputBias();
    ge::graphStatus CheckInputQuantScale();
    ge::graphStatus CheckInputQuantOffset();
    ge::graphStatus CheckForStaticQuant();
    ge::graphStatus GetInputGroupIndex();
    ge::graphStatus CheckOutputY();
    ge::graphStatus CheckOutputScale();
    ge::graphStatus DoOpTilingNotFull();
    void CalcTilingKeyForNotFull();
  
    private:
    uint64_t tilingKey_ = 0;
    DequantSwigluQuantV35BaseTilingData tilingData_;
    int64_t inDimx_ = 0;
    int64_t inDimy_ = 0;
    int64_t outDimy_ = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
 };

class DequantSwigluQuantV35NlastTiling : public TilingBaseClass {
  public:
    explicit DequantSwigluQuantV35NlastTiling(gert::TilingContext* tilingContext) : TilingBaseClass(tilingContext) {
    }
    ~DequantSwigluQuantV35NlastTiling() override {
    }
    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
  protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void FusedShape();
    void DoBlockSplit();
    bool DoUbSplit();
  
  private:
    uint64_t tilingKey_ = 0;
    uint64_t workspaceSize_ = 0;
    int32_t actDimIndex_ = 0;
    int64_t actRight_ = 0;
    int64_t roundMode_ = 0;
    gert::Shape xShape_ = gert::Shape();
    int64_t inDim0_ = 1;
    int64_t inDim1_ = 1;
    int64_t inDim2_ = 1;
    int64_t outDim1_ = 1;
    int64_t blockFormer0_ = 0;
    int64_t blockNum0_ = 0;
    int64_t blockFormer1_ = 0;
    int64_t blockNum1_ = 0;
    int64_t blockNum_ = 0;
    int64_t ubFormer0_ = 0;
    int64_t ubFormer1_ = 0;
    int64_t biasDtypeValue_ = 0;

    DequantSwigluQuantV35NlastTilingData tilingData_;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

}  // namespace optiling
#endif  // DEQUANT_SWIGLU_QUANT_TILING_H