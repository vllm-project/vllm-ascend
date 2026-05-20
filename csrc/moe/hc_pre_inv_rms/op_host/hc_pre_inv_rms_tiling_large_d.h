/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file hc_pre_inv_rms_tiling_large_d.cpp
 * \brief tiling for large d (R=28672, d=7168)
 */
#include <graph/utils/type_utils.h>
#include "hc_pre_inv_rms_tiling.h"

namespace optiling {
namespace HcPreInvRmsLargeD{

const static int64_t DEFAULT_WORKSPACE_SIZE = 16777216;
const static int64_t X_INPUT_INDEX = 0;
const static int64_t Y_OUTPUT_INDEX = 0;
const static int64_t EPS_ATTR_INDEX = 0;
const static size_t X_INPUT_BS_FUSED_DIMS = 3;
const static size_t X_INPUT_DIMS = 4;
const static int64_t UB_BLOCK_SIZE = 32;
const static uint64_t FULL_LOAD_LARGE_D_TILING_KEY = 1001;
const static int64_t R_LARGE_D = 28672;
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;

template <typename T>
static inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd)));
}

template <typename T>
static inline T CeilAlign(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd)) * (rnd));
}

class HcPreInvRmsTilingLargeD {
public:
    explicit HcPreInvRmsTilingLargeD(gert::TilingContext *context) : context_(context)
    {
        Reset();
    }
    ~HcPreInvRmsTilingLargeD()  = default;

    bool IsCapable()
    {
        return true;
    }
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus DoLibApiTiling();
    uint64_t GetTilingKey() const;
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    void Reset();

private:
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckAttr();
    ge::graphStatus CheckOutShape();
    void SplitA();
    void CalUbFactorA();

    const gert::Shape *xShape_ = nullptr;
    const gert::Shape *yShape_ = nullptr;

    float eps_ = 1e-6f;
    int64_t A_ = 0;
    int64_t R_ = 0;

    int64_t inputDtypeSize_;
    int64_t outputDtypeSize_;
    const char *opName_ = "";
    HcPreInvRmsFullLoadTilingData invRmsTilingData_;
    gert::TilingContext *context_ = nullptr;
    uint64_t workspaceSize_ = 0;

    uint64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t ubBlockSize_ = 0;
};

ge::graphStatus HcPreInvRmsTilingLargeD::CheckInputShape()
{
    size_t xDimNum = xShape_->GetDimNum();
    OPS_ERR_IF(xDimNum != X_INPUT_DIMS && xDimNum != X_INPUT_BS_FUSED_DIMS,
                OPS_LOG_E(context_, "The dim number of x is: %zu, but it should be %zu or %zu(bs fused)."
                    , xDimNum, X_INPUT_DIMS, X_INPUT_BS_FUSED_DIMS),
                return ge::GRAPH_FAILED);

    if (xDimNum == X_INPUT_DIMS) {
        A_ = xShape_->GetDim(DIM_0) * xShape_->GetDim(DIM_1);
        R_ = xShape_->GetDim(DIM_2) * xShape_->GetDim(DIM_3);
    } else if (xDimNum == X_INPUT_BS_FUSED_DIMS) {
        A_ = xShape_->GetDim(DIM_0);
        R_ = xShape_->GetDim(DIM_1) * xShape_->GetDim(DIM_2);
    }

    OPS_ERR_IF(R_ != R_LARGE_D,
                OPS_LOG_E(context_, "R is: %ld, but large_d tiling only supports R=%ld.", R_, R_LARGE_D),
                return ge::GRAPH_FAILED);

    invRmsTilingData_.set_A(A_);
    invRmsTilingData_.set_R(R_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreInvRmsTilingLargeD::CheckAttr()
{
    OPS_ERR_IF(eps_ <= 0, OPS_LOG_E(context_, "epsilon is: %ld, but it should not be less than 0.", eps_), return ge::GRAPH_FAILED);
    invRmsTilingData_.set_epsilon(eps_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreInvRmsTilingLargeD::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    auto xShapePtr = context_->GetInputShape(X_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, xShapePtr, return ge::GRAPH_FAILED);
    xShape_ = &xShapePtr->GetStorageShape();

    auto yShapePtr = context_->GetOutputShape(Y_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, yShapePtr, return ge::GRAPH_FAILED);
    yShape_ = &yShapePtr->GetStorageShape();

    auto xDesc = context_->GetInputDesc(X_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, xDesc, return ge::GRAPH_FAILED);
    auto xDtype = xDesc->GetDataType();
    OPS_ERR_IF(
        (xDtype != ge::DataType::DT_FLOAT && xDtype != ge::DataType::DT_FLOAT16 && xDtype != ge::DataType::DT_BF16),
        OPS_LOG_E(context_, "x dtype %s error, only supports float32, float16 and bfloat16. please check.",
             ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto yDesc = context_->GetOutputDesc(Y_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, yDesc, return ge::GRAPH_FAILED);
    auto yDtype = yDesc->GetDataType();
    OPS_ERR_IF((yDtype != ge::DataType::DT_FLOAT),
                OPS_LOG_E(context_, "y out dtype %s error, only support float32, please check",
                     ge::TypeUtils::DataTypeToSerialString(yDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);

    const float *epsPtr = attrs->GetAttrPointer<float>(EPS_ATTR_INDEX);
    if (epsPtr != nullptr) {
        eps_ = *epsPtr;
    }
    OPS_LOG_I(context_, "Attr eps is: %f ", eps_);

    inputDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(context_->GetInputDesc(X_INPUT_INDEX)->GetDataType()));
    outputDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(context_->GetOutputDesc(Y_OUTPUT_INDEX)->GetDataType()));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreInvRmsTilingLargeD::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOG_E(context_, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OPS_ERR_IF(
        coreNum_ <= 0, OPS_LOG_E(context_->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = static_cast<int64_t>(ubSizePlatForm);
    OPS_ERR_IF(
        ubSize_ <= 0, OPS_LOG_E(context_->GetNodeName(), "ubSize must be greater than 0."),
        return ge::GRAPH_FAILED);

    ubBlockSize_ = UB_BLOCK_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreInvRmsTilingLargeD::CheckOutShape()
{
    OPS_ERR_IF((yShape_->GetDim(0) != xShape_->GetDim(0)),
                OPS_LOG_E(context_, "y out dim[0] %ld not equal x dim[0] %ld, please check.", yShape_->GetDim(0),
                     xShape_->GetDim(0)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void HcPreInvRmsTilingLargeD::SplitA()
{
    int64_t blockFactorA = CeilDiv(A_, static_cast<int64_t>(coreNum_));
    int64_t blockNumA = CeilDiv(A_, blockFactorA);
    int64_t blockTailFactorA = A_ % blockFactorA == 0 ? blockFactorA : A_ % blockFactorA;
    invRmsTilingData_.set_blockNumA(blockNumA);
    invRmsTilingData_.set_blockFactorA(blockFactorA);
    invRmsTilingData_.set_blockTailFactorA(blockTailFactorA);
    int64_t ubFactorA = invRmsTilingData_.get_ubFactorA();
    if (ubFactorA > blockFactorA) {
        invRmsTilingData_.set_ubFactorA(blockFactorA);
    }
}

void HcPreInvRmsTilingLargeD::CalUbFactorA()
{
    invRmsTilingData_.set_ubFactorA(1);
}

ge::graphStatus HcPreInvRmsTilingLargeD::DoOpTiling()
{
    auto ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckInputShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckOutShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckAttr();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    CalUbFactorA();
    SplitA();

    ret = PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreInvRmsTilingLargeD::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreInvRmsTilingLargeD::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreInvRmsTilingLargeD::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(invRmsTilingData_.get_blockNumA());
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    invRmsTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                          context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(invRmsTilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t HcPreInvRmsTilingLargeD::GetTilingKey() const
{
    return FULL_LOAD_LARGE_D_TILING_KEY;
}

void HcPreInvRmsTilingLargeD::Reset()
{
    opName_ = nullptr;
    return;
}

} // namespace HcPreInvRmsLargeD
} // namespace optiling