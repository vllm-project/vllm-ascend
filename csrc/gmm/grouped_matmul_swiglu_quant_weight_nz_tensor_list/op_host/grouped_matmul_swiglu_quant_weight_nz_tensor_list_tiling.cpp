/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_weight_nz_tensor_list_tiling.cpp
 * \brief
 */
#include <climits>
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "log/ops_log.h"
#include "error/ops_error.h"
#include "tiling/tiling_base.h"
#include "grouped_matmul_swiglu_quant_weight_nz_tensor_list_tiling.h"
using namespace ge;
using namespace AscendC;
using namespace GroupedMatmulSwigluQuantWeightNzTensorListTiling;

template <typename T1, typename T2>
static T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

namespace optiling {

struct GMMSwigluCompileInfo {
  uint64_t ubSize_ = 0;
  uint32_t aicNum_ = 0;
  uint32_t baseM_ = 128;
  uint32_t baseN_ = 256;
};

static uint64_t CalcMaxTmpSize(const uint32_t row, const uint64_t n) {
  std::vector<int64_t> shape_vec = {static_cast<int64_t>(row * n)};
  Shape shape(shape_vec);
  uint32_t max;
  uint32_t min;
  GetSwiGLUMaxMinTmpSize(shape, 4, max, min, false);
  uint32_t averageTmp = (max + min) >> 1;
  GetAscendQuantMaxMinTmpSize(shape, 4, max, min);
  uint32_t average = (max + min) >> 1;
  average = average > averageTmp ? average : averageTmp;
  GetAscendDequantMaxMinTmpSize(shape, 4, max, min);
  averageTmp = (max + min) >> 1;
  return average > averageTmp ? average : averageTmp;
}

static uint64_t CalRows(const uint64_t ubSize, const uint64_t n) {
  uint64_t tokenSize = n << 2;
  uint64_t expectSize = ubSize - tokenSize;
  uint64_t rows = expectSize / (8 + tokenSize);
  uint64_t realSize = (8 + tokenSize) * rows + CalcMaxTmpSize(rows, n);
  while (expectSize < realSize) {
    rows -= CeilDiv(realSize - expectSize, (8 + tokenSize) << 2);
    realSize = (8 + tokenSize) * rows + CalcMaxTmpSize(rows, n);
  }
  return rows;
}

static void SetTilingKey(gert::TilingContext* context, bool isSplitWorkSpace) {
  if(isSplitWorkSpace){
    context->SetTilingKey(1);
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
  } else {
    context->SetTilingKey(0);
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
  }
}

static bool IsPreFill(GMMSwigluQuantTilingData &tilingData) {
  int64_t k = tilingData.gmmSwigluBaseParams.get_K();
  int64_t n = tilingData.gmmSwigluBaseParams.get_N();
  int64_t m = tilingData.gmmSwigluBaseParams.get_M();
  int64_t groupNum = tilingData.gmmSwigluBaseParams.get_groupNum();
  if (groupNum == 128 && m >= PREFILL_M_MIN_SIZE) { // 128:prefiling groupNum
    std::array<int64_t, 2> kNList = {k, n}; // 2: kNList size
    if (PREFILL_WHITE_LIST.count(kNList)) {
      return true;
    }
  }
  return false;
}

ASCENDC_EXTERN_C graphStatus TilingGMMSwigluQuant(gert::TilingContext* context) {
  // set info
  OPS_LOG_I(context->GetNodeName(), "Begin Run GMM Swiglu Tiling .");
  
  auto compileInfoPtr = context->GetCompileInfo<GMMSwigluCompileInfo>();
  auto xTensor = context->GetInputTensor(X_INDEX);
  OPS_LOG_E_IF_NULL(context, xTensor, return GRAPH_FAILED);
  const int64_t m = xTensor->GetStorageShape().GetDim(0);
  const int64_t k = xTensor->GetStorageShape().GetDim(1);
  auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, 0);
  OPS_LOG_E_IF_NULL(context, wTensor, return GRAPH_FAILED);
  const int64_t n = wTensor->GetStorageShape().GetDim(0) * wTensor->GetStorageShape().GetDim(3);
  auto groupListTensor = context->GetDynamicInputTensor(GROUPLIST_INDEX, 0);
  OPS_LOG_E_IF_NULL(context, groupListTensor, return GRAPH_FAILED);
  const int64_t groupNum = groupListTensor->GetStorageShape().GetDim(0);
  GMMSwigluQuantTilingData tilingData;
  const int64_t row = CalRows(compileInfoPtr->ubSize_, n);
  tilingData.gmmSwigluBaseParams.set_groupNum(groupNum);
  tilingData.gmmSwigluBaseParams.set_coreNum(compileInfoPtr->aicNum_);
  tilingData.gmmSwigluBaseParams.set_K(k);
  tilingData.gmmSwigluBaseParams.set_N(n);
  tilingData.gmmSwigluBaseParams.set_M(m);
  tilingData.gmmSwiglu.set_maxProcessRowNum(row);
  tilingData.gmmSwiglu.set_groupListLen(groupNum);
  tilingData.gmmSwiglu.set_tokenLen(n);
  
  OPS_LOG_D(context->GetNodeName(),"grouped_matmul_swiglu_quant_weight_nz_tensor_list_tiling.");
  OPS_LOG_D(context->GetNodeName(),"gmmSwigluBaseParams.groupNum:  %ld", groupNum);
  OPS_LOG_D(context->GetNodeName(),"gmmSwigluBaseParams.coreNum:   %u ", compileInfoPtr->aicNum_);
  OPS_LOG_D(context->GetNodeName(),"gmmSwigluBaseParams.M:         %ld", m);
  OPS_LOG_D(context->GetNodeName(),"gmmSwigluBaseParams.K:         %ld", k);
  OPS_LOG_D(context->GetNodeName(),"gmmSwigluBaseParams.N:         %ld", n);
  OPS_LOG_D(context->GetNodeName(),"gmmSwiglu.maxProcessRowNum:    %ld", row);
  OPS_LOG_D(context->GetNodeName(),"gmmSwiglu.groupListLen:        %ld", groupNum);
  OPS_LOG_D(context->GetNodeName(),"gmmSwiglu.tokenLen:            %ld", n);
  
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  using namespace matmul_tiling;
  MatmulApiTiling tiling(ascendcPlatform);
  tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT8);
  tiling.SetBType(TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_INT8);
  tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
  tiling.SetBias(false);
  tiling.SetShape(compileInfoPtr->baseM_, compileInfoPtr->baseN_, k);
  tiling.SetOrgShape(m, n, k);
  tiling.SetBufferSpace(-1, -1, -1);
  OPS_ERR_IF(tiling.GetTiling(tilingData.mmTilingData) == -1,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "grouped_matmul_swiglu_quant_weight_nz_tensor_list_tiling, get tiling failed"),
             return GRAPH_FAILED);
  auto workspaceSizes = context->GetWorkspaceSizes(1);
  bool isPreFill = IsPreFill(tilingData);
  tilingData.gmmSwigluBaseParams.set_isPreFill(isPreFill);
  int64_t usrWorkspaceLimut = isPreFill ? PREFILL_USER_WORKSPACE_LIMIT : USER_WORKSPACE_LIMIT;
  int64_t mLimit = ((usrWorkspaceLimut / DOUBLE_WORKSPACE_SPLIT) / INT32_DTYPE_SIZE) / n;
  OPS_ERR_IF(mLimit <= 0, 
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),"mLimit is %ld must over then 0.", mLimit),
             return GRAPH_FAILED);
  tilingData.gmmSwigluBaseParams.set_mLimit(mLimit);
  workspaceSizes[0] = SYS_WORKSPACE_SIZE + ((mLimit * DOUBLE_WORKSPACE_SPLIT > m \
                      ? m \
                      : mLimit * DOUBLE_WORKSPACE_SPLIT) * n * sizeof(int32_t));
  bool isSplitWorkSpace = m > mLimit * DOUBLE_WORKSPACE_SPLIT;
  OPS_LOG_D(context->GetNodeName(), "USER_WORKSPACE_LIMIT:         %ld", usrWorkspaceLimut);
  OPS_LOG_D(context->GetNodeName(), "mLimit:                       %ld", mLimit);
  OPS_LOG_D(context->GetNodeName(), "workspaceSizes:               %lu", workspaceSizes[0]);
  OPS_LOG_D(context->GetNodeName(), "isSplitWorkSpace:             %s", isSplitWorkSpace ? "true" : "false");
  OPS_LOG_D(context->GetNodeName(), "isPreFill:                    %s", isPreFill ? "true" : "false");
  SetTilingKey(context, isSplitWorkSpace);
  tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->SetBlockDim(compileInfoPtr->aicNum_); // block dim is the number of aicube
  context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  
  OPS_LOG_D(context->GetNodeName(), "End Run GMM Swiglu Tiling.");
  return GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C graphStatus TilingPrepareForGMMSwigluQuant(gert::TilingParseContext* context) {
  // get info
  fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
  OPS_LOG_E_IF_NULL(context, platformInfoPtr, return GRAPH_FAILED);
  auto compileInfoPtr = context->GetCompiledInfo<GMMSwigluCompileInfo>();
  OPS_LOG_E_IF_NULL(context, compileInfoPtr, return GRAPH_FAILED);

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  compileInfoPtr->aicNum_ = ascendcPlatform.GetCoreNumAic();
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize_);
  OPS_LOG_D(context->GetNodeName(), "ubSize is %lu, aicNum is %u.", compileInfoPtr->ubSize_, compileInfoPtr->aicNum_);
  return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatmulSwigluQuantWeightNzTensorList)
.Tiling(TilingGMMSwigluQuant)
.TilingParse<GMMSwigluCompileInfo>(TilingPrepareForGMMSwigluQuant); 
}  // namespace optiling
