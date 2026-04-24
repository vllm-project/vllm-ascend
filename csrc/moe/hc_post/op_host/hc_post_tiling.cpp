/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file gather_selection_kv_cache_tiling.cpp
 * \brief
 */

#include "hc_post_tiling.h"
#include "hc_post_tiling_arch35.h"

namespace optiling {
constexpr int64_t DEFAULT_DEAL_DPARAM = 2048;

ge::graphStatus HcPostTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOG_E(context_->GetNodeName(), "get platformInfo nullptr."),
        return ge::GRAPH_FAILED);
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

    ubBlockSize_ = 32;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTiling::GetShapeInfo()
{
    OPS_ERR_IF(
        context_ == nullptr, OPS_LOG_E("HcPostTiling", "context can not be nullptr."),
        return ge::GRAPH_FAILED);

    if (GetInputShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // dtype校验
    if (GetInputDtypeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTiling::GetInputShapeInfo()
{
    auto xInput = context_->GetInputShape(INPUT_IDX_X);
    OPS_ERR_IF(xInput == nullptr, OPS_LOG_E(context_->GetNodeName(), "get xInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape xShape = xInput->GetStorageShape();
    size_t dimsN = xShape.GetDimNum();
    OPS_ERR_IF((dimsN != CONST3),
        OPS_LOG_E(context_->GetNodeName(), "xInput dim:%lu should be 3.", dimsN),
        return ge::GRAPH_FAILED);

    auto residualInput = context_->GetInputShape(INPUT_IDX_RESIDUAL);
    OPS_ERR_IF(residualInput == nullptr, OPS_LOG_E(context_->GetNodeName(), "get residualInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape residualShape = residualInput->GetStorageShape();
    dimsN = residualShape.GetDimNum();
    OPS_ERR_IF((dimsN != CONST4),
        OPS_LOG_E(context_->GetNodeName(), "residualInput dim:%lu should be 4.", dimsN),
        return ge::GRAPH_FAILED);

    auto postInput = context_->GetInputShape(INPUT_IDX_POST);
    OPS_ERR_IF(postInput == nullptr, OPS_LOG_E(context_->GetNodeName(), "get residualInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape postShape = postInput->GetStorageShape();
    dimsN = postShape.GetDimNum();
    OPS_ERR_IF((dimsN != CONST3),
        OPS_LOG_E(context_->GetNodeName(), "postInput dim:%lu should be 3.", dimsN),
        return ge::GRAPH_FAILED);

    auto combInput = context_->GetInputShape(INPUT_IDX_COMB);
    OPS_ERR_IF(combInput == nullptr, OPS_LOG_E(context_->GetNodeName(), "get residualInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape combShape = combInput->GetStorageShape();
    dimsN = combShape.GetDimNum();
    OPS_ERR_IF((dimsN != CONST4),
        OPS_LOG_E(context_->GetNodeName(), "combInput dim:%lu should be 4.", dimsN),
        return ge::GRAPH_FAILED);
    bParam_ = xShape.GetDim(DIM_INDEX_0);
    sParam_ = xShape.GetDim(DIM_INDEX_1);
    dParam_ = xShape.GetDim(DIM_INDEX_2);
    hcParam_ = residualShape.GetDim(DIM_INDEX_2);
    tilingData_.set_bParam(bParam_);
    tilingData_.set_sParam(sParam_);
    tilingData_.set_dParam(dParam_);
    tilingData_.set_hcParam(hcParam_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTiling::GetInputDtypeInfo()
{
    auto xDesc = context_->GetInputDesc(INPUT_IDX_X);
    OPS_ERR_IF(xDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get xDesc nullptr."),
        return ge::GRAPH_FAILED);
    auto xDtype = xDesc->GetDataType();
    OPS_ERR_IF(
        (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT),
        OPS_LOG_E(context_->GetNodeName(), "xDtype is not supported."),
        return ge::GRAPH_FAILED);

    auto residualDesc = context_->GetInputDesc(INPUT_IDX_RESIDUAL);
    OPS_ERR_IF(residualDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get residualDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType residualDtype = residualDesc->GetDataType();
    OPS_ERR_IF(
        (residualDtype != xDtype),
        OPS_LOG_E(context_->GetNodeName(), "residualDtype is not equal to xDtype."),
        return ge::GRAPH_FAILED);

    auto postDesc = context_->GetInputDesc(INPUT_IDX_POST);
    OPS_ERR_IF(postDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get postDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType postDtype = postDesc->GetDataType();
    OPS_ERR_IF(
        (postDtype != ge::DT_FLOAT16 && postDtype != ge::DT_BF16 && postDtype != ge::DT_FLOAT),
        OPS_LOG_E(context_->GetNodeName(), "postDtype is not supported."),
        return ge::GRAPH_FAILED);

    auto combDesc = context_->GetInputDesc(INPUT_IDX_COMB);
    OPS_ERR_IF(combDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get combDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType combDtype = combDesc->GetDataType();
    OPS_ERR_IF(
        (combDtype != postDtype),
        OPS_LOG_E(context_->GetNodeName(), "combDtype is not equal to postDtype."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTiling::DoOpTiling()
{
    int64_t batchSize = bParam_ * sParam_;

    int64_t useCoreNum = batchSize < coreNum_ ? batchSize : coreNum_;
    int64_t batchOneCore = CeilDiv(batchSize, static_cast<int64_t>(useCoreNum));
    int64_t batchOneCoreTail = batchOneCore - 1;
    int64_t frontCore = batchSize - batchOneCoreTail * useCoreNum;
    tilingData_.set_usedCoreNum(useCoreNum);
    tilingData_.set_batchOneCore(batchOneCore);
    tilingData_.set_batchOneCoreTail(batchOneCoreTail);
    tilingData_.set_frontCore(frontCore);
    int64_t dSplitTime = CeilDiv(dParam_, DEFAULT_DEAL_DPARAM);
    tilingData_.set_dSplitTime(dSplitTime);
    tilingData_.set_dOnceDealing(DEFAULT_DEAL_DPARAM);
    tilingData_.set_dLastDealing(dParam_ - dSplitTime * DEFAULT_DEAL_DPARAM);
    context_->SetBlockDim(useCoreNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTiling::PostTiling()
{
    context_->SetTilingKey(0);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = static_cast<size_t>(16 * 1024 * 1024);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTiling::RunTiling()
{
    ge::graphStatus ret = GetShapeInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return PostTiling();
}

ge::graphStatus Tiling4HcPost(gert::TilingContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "TilingForHcPost running.");
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("TilingForHcPost", "Tiling context is null"),
               return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_REPORT_VECTOR_INNER_ERR("TilingForHcPost", "Tiling platformInfo is null"),
               return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion == platform_ascendc::SocVersion::ASCEND910_95) {
        OPS_LOG_I(context, "Using arch35 tiling for ASCEND910_95");
        HcPostTilingRegbase tiling(context);
        return tiling.RunTilingRegbase();
    }
    HcPostTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepare4HcPost(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(HcPost)
    .Tiling(Tiling4HcPost)
    .TilingParse<HcPostCompileInfo>(TilingPrepare4HcPost);

} // namespace optiling