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
 * \file hc_pre_inv_rms_tiling_arch35.cpp
 * \brief
 */
#include "hc_pre_inv_rms_tiling.h"
#include <graph/utils/type_utils.h>

namespace optiling {
namespace HcPreInvRmsRegbase{

const static int64_t DEFAULT_WORKSPACE_SIZE = 16777216; // 预留16M空间
const static int64_t X_INPUT_INDEX = 0;
const static int64_t Y_OUTPUT_INDEX = 0;
const static int64_t EPS_ATTR_INDEX = 0;
const static size_t X_INPUT_BS_FUSED_DIMS = 3;
const static size_t X_INPUT_DIMS = 4;
const static int64_t UB_BLOCK_SIZE = 32;
const static uint64_t FULL_LOAD_REGBASE_TILING_KEY = 2000;

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

class HcPreInvRmsTilingRegbase {
public:
    explicit HcPreInvRmsTilingRegbase(gert::TilingContext *context) : context_(context)
    {
        Reset();
    }
    ~HcPreInvRmsTilingRegbase()  = default;

    bool IsCapable()
    {
        if (socVersion_ != platform_ascendc::SocVersion::ASCEND910_95) {
            return false;
        }
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo();
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo();
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling();
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling();
    // 5、计算TilingKey
    uint64_t GetTilingKey() const;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize();
    // 7、保存Tiling数据
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
    platform_ascendc::SocVersion socVersion_;
};

ge::graphStatus HcPreInvRmsTilingRegbase::CheckInputShape()
{
    size_t xDimNum = xShape_->GetDimNum();
    OPS_ERR_IF(xDimNum != X_INPUT_DIMS && xDimNum != X_INPUT_BS_FUSED_DIMS,
                OPS_LOG_E(context_, "The dim number of x is: %zu, but it should be %zu or %zu(bs fused)."
                    , xDimNum, X_INPUT_DIMS, X_INPUT_BS_FUSED_DIMS),
                return ge::GRAPH_FAILED);

    if (xDimNum == X_INPUT_DIMS) {
        A_ = xShape_->GetDim(0) * xShape_->GetDim(1);
        R_ = xShape_->GetDim(2) * xShape_->GetDim(3);
    } else if (xDimNum == X_INPUT_BS_FUSED_DIMS) {
        A_ = xShape_->GetDim(0);
        R_ = xShape_->GetDim(1) * xShape_->GetDim(2);
    }
    
    invRmsTilingData_.set_A(A_);
    invRmsTilingData_.set_R(R_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreInvRmsTilingRegbase::CheckAttr()
{
    OPS_ERR_IF(eps_ <= 0, OPS_LOG_E(context_, "epsilon is: %ld, but it should not be less than 0.", eps_), return ge::GRAPH_FAILED);
    invRmsTilingData_.set_epsilon(eps_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreInvRmsTilingRegbase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    // 获取输入shape信息
    auto xShapePtr = context_->GetInputShape(X_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, xShapePtr, return ge::GRAPH_FAILED);
    xShape_ = &xShapePtr->GetStorageShape();

    // 获取输出shape
    auto yShapePtr = context_->GetOutputShape(Y_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, yShapePtr, return ge::GRAPH_FAILED);
    yShape_ = &yShapePtr->GetStorageShape();

    // 获取输入dtype
    auto xDesc = context_->GetInputDesc(X_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, xDesc, return ge::GRAPH_FAILED);
    auto xDtype = xDesc->GetDataType();
    OPS_ERR_IF(
        (xDtype != ge::DataType::DT_FLOAT && xDtype != ge::DataType::DT_FLOAT16 && xDtype != ge::DataType::DT_BF16),
        OPS_LOG_E(context_, "x dtype %s error, only supports float32, float16 and bfloat16. please check.",
             ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
        return ge::GRAPH_FAILED);

    // 获取输出dtype
    auto yDesc = context_->GetOutputDesc(Y_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, yDesc, return ge::GRAPH_FAILED);
    auto yDtype = yDesc->GetDataType();
    OPS_ERR_IF((yDtype != ge::DataType::DT_FLOAT),
                OPS_LOG_E(context_, "y out dtype %s error, only support float32, please check",
                     ge::TypeUtils::DataTypeToSerialString(yDtype).c_str()),
                return ge::GRAPH_FAILED);

    // 获取属性
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

ge::graphStatus HcPreInvRmsTilingRegbase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOG_E(context_, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    socVersion_ = ascendcPlatform.GetSocVersion();
    OPS_ERR_IF(
        coreNum_ <= 0, OPS_LOG_E(context_->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);
    // 获取UB大小
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = static_cast<int64_t>(ubSizePlatForm);
    OPS_ERR_IF(
        ubSize_ <= 0, OPS_LOG_E(context_->GetNodeName(), "ubSize must be greater than 0."),
        return ge::GRAPH_FAILED);

    ubBlockSize_ = UB_BLOCK_SIZE; // 32: ub block size

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreInvRmsTilingRegbase::CheckOutShape()
{
    OPS_ERR_IF((yShape_->GetDim(0) != xShape_->GetDim(0)),
                OPS_LOG_E(context_, "y out dim[0] %ld not euqal x dim[0] %ld, please check.", yShape_->GetDim(0),
                     xShape_->GetDim(0)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void HcPreInvRmsTilingRegbase::SplitA()
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

void HcPreInvRmsTilingRegbase::CalUbFactorA()
{
    // ubFactorA * R_AlignSize  * 2 + ubFactorA * outputDtypeSize_ + 32 = ubSize_
    int64_t rAlignSize = CeilAlign(R_ * inputDtypeSize_, UB_BLOCK_SIZE);
    int64_t ubFactorA = (ubSize_ - UB_BLOCK_SIZE) / (2 * rAlignSize + outputDtypeSize_);
    invRmsTilingData_.set_ubFactorA(ubFactorA);
}

ge::graphStatus HcPreInvRmsTilingRegbase::DoOpTiling()
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

ge::graphStatus HcPreInvRmsTilingRegbase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreInvRmsTilingRegbase::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreInvRmsTilingRegbase::PostTiling()
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

uint64_t HcPreInvRmsTilingRegbase::GetTilingKey() const
{
    // 全载场景
    return FULL_LOAD_REGBASE_TILING_KEY;
}

void HcPreInvRmsTilingRegbase::Reset()
{
    opName_ = nullptr;
    return;
}

} // namespace HcPreInvRmsRegbase
} // namespace optiling