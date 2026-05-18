/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file grouped_matmul_swiglu_quant_v2_fusion_tiling.cpp
 * \brief
 */
#include "grouped_matmul_swiglu_quant_v2_fusion_tiling.h"
#include "util/math_util.h"
#include "err/ops_err.h"

namespace optiling {
namespace GroupedMatmulSwigluQuantV2Tiling {

constexpr int64_t BASE_M = 128;
constexpr int64_t BASE_K = 128;
constexpr int64_t BASE_N = 256;
constexpr int64_t UB_Y_FACTOR = 2;
constexpr int64_t EXTEND_WORKSPACE_SIZE = (20 * 1024 * 1024);
constexpr int64_t NZ_WEIGHT_SINGLE_TENSOR_DIM = 5;   // single: [E, N/32, K/16, 16, 32]
constexpr int64_t NZ_WEIGHT_MULTI_TENSOR_DIM = 4;    // multi: each [N/32, K/16, 16, 32]
constexpr int64_t MIN_UB_FACTOR_DIM_X_N = 4600;
constexpr int64_t MID_UB_FACTOR_DIM_X_N = 8192;

using namespace matmul_tiling;

bool GroupedMatmulSwigluQuantV2FusionTiling::IsCapable()
{
    auto weightDesc = context_->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    ge::DataType weightDType = weightDesc->GetDataType();
    if (weightDType != ge::DataType::DT_INT8) {
        return false;
    }

    auto wTensor = context_->GetDynamicInputTensor(WEIGHT_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wTensor);
    if (!(wTensor->GetStorageShape().GetDimNum() == NZ_WEIGHT_DIM_LIMIT ||
          wTensor->GetStorageShape().GetDimNum() == NZ_WEIGHT_MULTI_TENSOR_DIM)) {
        return false;
    }
    return true;
}

ge::graphStatus GroupedMatmulSwigluQuantV2FusionTiling::ParseInputAndAttr()
{
    auto xTensor = context_->GetDynamicInputTensor(X_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xTensor);
    auto wTensor = context_->GetDynamicInputTensor(WEIGHT_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wTensor);
    auto groupListTensor = context_->GetDynamicInputTensor(GROUPLIST_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, groupListTensor);
    groupNum_ = groupListTensor->GetStorageShape().GetDim(0);
    auto wDimNum = wTensor->GetStorageShape().GetDimNum();
    if (wDimNum == NZ_WEIGHT_DIM_LIMIT) {
        isSingleTensor_ = 1;
    } else {
        isSingleTensor_ = 0;  // multi tensor: 4D per weight [N/32, K/16, 16, 32]
    }
    auto attr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attr); // check attr is not null
    const int64_t *groupListTypePtr = attr->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPLIST_TYPE);
    groupListType_ = groupListTypePtr != nullptr ? *groupListTypePtr : 0;
    OP_CHECK_IF(!(groupListType_ == 0 || groupListType_ == 1),
        OP_LOGE(context_->GetNodeName(), "GroupListType must be 0 or 1, but actual value is %ld.", groupListType_),
        return ge::GRAPH_FAILED);

    const auto swigluLimtPtr = attr->GetAttrPointer<double>(ATTR_INDEX_SWIGLU_LIMIT);
    double swigluLimt_ = swigluLimtPtr != nullptr ? *swigluLimtPtr : 1000000.0f;
    tilingData_.set_swigluLimit(swigluLimt_);
    m_ = xTensor->GetStorageShape().GetDim(0);
    k_ = xTensor->GetStorageShape().GetDim(1);
    if (wDimNum == NZ_WEIGHT_DIM_LIMIT) {
        n_ = wTensor->GetStorageShape().GetDim(DIM_1) * wTensor->GetStorageShape().GetDim(DIM_4);
    } else {
        // 4D multi tensor: [N/32, K/16, 16, 32] -> N = dim0 * dim3
        n_ = wTensor->GetStorageShape().GetDim(0) * wTensor->GetStorageShape().GetDim(3);
    }
    if (n_ < MIN_UB_FACTOR_DIM_X_N) {
        ubFactorDimx_ = 0x4;
    } else if (n_ >= MIN_UB_FACTOR_DIM_X_N && n_ < MID_UB_FACTOR_DIM_X_N) {
        ubFactorDimx_ = 0x2;
    } else {
        ubFactorDimx_ = 1;
    }

    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<GMMSwigluV2CompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "CompileInfo is nullptr"),
                    return ge::GRAPH_FAILED);
        aicCoreNum_ = compileInfoPtr->aicNum_;
        aivCoreNum_ = compileInfoPtr->aivNum_;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicCoreNum_ = ascendcPlatform.GetCoreNumAic();
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2FusionTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Begin Run GMM Swiglu Fusion Tiling.");

    if (ParseInputAndAttr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    MatmulApiTiling tiling(ascendcPlatform);
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT8);
    tiling.SetBType(TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_INT8);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    tiling.SetBias(false);
    tiling.SetShape(m_, BASE_N, k_);
    tiling.SetOrgShape(m_, n_, k_);
    tiling.SetBufferSpace(-1, -1, -1);
    OP_CHECK_IF(
        tiling.GetTiling(tilingData_.matmulTiling) == -1,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "grouped_matmul_swiglu_quant_tiling, get tiling failed"),
        return ge::GRAPH_FAILED);

    workspaceSize_ = static_cast<int64_t>(m_) * static_cast<int64_t>(n_) * sizeof(int32_t) + EXTEND_WORKSPACE_SIZE;
    tilingKey_ = A8W8_FUSION_KEY_MODE;
    FillTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulSwigluQuantV2FusionTiling::GetTilingKey() const
{
    return tilingKey_;
}

void GroupedMatmulSwigluQuantV2FusionTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "cubeBlockDim: %d", tilingData_.get_cubeBlockDim());
    OP_LOGD(context_->GetNodeName(), "vectorBlockDim: %d", tilingData_.get_vectorBlockDim());
    OP_LOGD(context_->GetNodeName(), "K: %d", tilingData_.get_K());
    OP_LOGD(context_->GetNodeName(), "M: %d", tilingData_.get_M());
    OP_LOGD(context_->GetNodeName(), "N: %d", tilingData_.get_N());
    OP_LOGD(context_->GetNodeName(), "ubFactorDimx: %d", tilingData_.get_ubFactorDimx());
    OP_LOGD(context_->GetNodeName(), "ubFactorDimy: %d", tilingData_.get_ubFactorDimy());
    OP_LOGD(context_->GetNodeName(), "groupListType: %ld", tilingData_.get_groupListType());
    OP_LOGD(context_->GetNodeName(), "isSingleTensor: %d", tilingData_.get_isSingleTensor());
}

void GroupedMatmulSwigluQuantV2FusionTiling::FillTilingData()
{
    tilingData_.set_cubeBlockDim(aicCoreNum_);
    tilingData_.set_vectorBlockDim(aivCoreNum_);
    tilingData_.set_groupNum(groupNum_);
    tilingData_.set_K(k_);
    tilingData_.set_N(n_);
    tilingData_.set_M(m_);
    tilingData_.set_ubFactorDimx(ubFactorDimx_);
    tilingData_.set_ubFactorDimy(n_ / UB_Y_FACTOR);
    tilingData_.set_groupListType(groupListType_);
    tilingData_.set_isSingleTensor(isSingleTensor_);

    blockDim_ = aicCoreNum_;
    tilingData_.matmulTiling.set_usedCoreNum(aicCoreNum_);
    tilingData_.matmulTiling.set_shareMode(0);
    tilingData_.matmulTiling.set_dbL0C(1);
    tilingData_.matmulTiling.set_baseM(BASE_M);
    tilingData_.matmulTiling.set_baseN(BASE_N);
    tilingData_.matmulTiling.set_baseK(BASE_K);
    tilingData_.matmulTiling.set_stepKa(0x4);  // 4: L1中左矩阵单次搬运基于baseK的4倍数据
    tilingData_.matmulTiling.set_stepKb(0x4);  // 4: L1中右矩阵单次搬运基于baseK的4倍数据
    tilingData_.matmulTiling.set_depthA1(0x8);  // 8: stepKa的两倍，开启double buffer
    tilingData_.matmulTiling.set_depthB1(0x8);  // 8: stepKb的两倍，开启double buffer
    tilingData_.matmulTiling.set_stepM(1);
    tilingData_.matmulTiling.set_stepN(1);
}

ge::graphStatus GroupedMatmulSwigluQuantV2FusionTiling::PostTiling()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->SetBlockDim(blockDim_);
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_CHECK_IF(workspaces == nullptr,
                OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "fusion tiling workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}
} // namespace GroupedMatmulSwigluQuantV2Tiling
} // namespace optiling
