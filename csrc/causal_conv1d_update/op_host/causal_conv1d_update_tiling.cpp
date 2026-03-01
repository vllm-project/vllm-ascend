/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_update_tiling.cpp
 * \brief
 */

#include "error_log.h"
//#include "tiling/tiling_templates_registry.h"
//#include "tiling/tiling_util.h"
#include "../tiling_base/tiling_templates_registry.h"
#include "../tiling_base/tiling_util.h"
#include "math_util.h"
#include "../op_kernel/causal_conv1d_update_tilingdata.h"
#include "../op_kernel/causal_conv1d_update_struct.h"
#include "causal_conv1d_update_tiling.h"

#include <algorithm>
#include <set>

namespace optiling {

namespace causalconv1dupdate {

using namespace Ops::Transformer::OpTiling;
//using namespace Ops::Base;

// constexpr uint32_t TILE_C = 256;
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t CONV_STATE_INDEX = 2;
constexpr uint32_t CONV_STATE_INDICES_INDEX = 3;
constexpr uint32_t BIAS_INDEX = 4;
constexpr uint32_t NUM_ACCEPT_INDEX = 5;
constexpr uint32_t QUERY_LOC_INDEX = 6;

constexpr int32_t ATTR_ACTIVATION_MODE_INDEX = 0;
constexpr int32_t ATTR_PAD_SLOT_ID_INDEX = 1;
constexpr size_t SYNC_WORKSPACE_SIZE = 0;
// constexpr size_t SYNC_WORKSPACE_SIZE = 16777216;
// constexpr int32_t ATTR_MAX_QUERY_LEN_INDEX = 2;

const std::set<ge::DataType> supportedXDtype = {ge::DT_BF16, ge::DT_FLOAT16};

ge::graphStatus CausalConv1dUpdate::DoCausalConv1dUpdateTiling()
{
    OP_CHECK_IF(
        (GetPlatform() != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "DoCausalConv1dUpdateTiling GetPlatform Failed."), 
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (GetOpParam() != ge::GRAPH_SUCCESS), 
        OP_LOGE(context_->GetNodeName(), "DoCausalConv1dUpdateTiling GetOpParam Failed."),
        return ge::GRAPH_FAILED);

    CalcTiling();
    CalcTilingKey();
    WriteTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dUpdate::GetPlatform() {
    OP_LOGD("CausalConv1dUpdateTiling", "Enter CausalConv1dUpdateTiling");
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (static_cast<int32_t>(coreNum) <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        (static_cast<int64_t>(ubSize) <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get ub size."),
        return ge::GRAPH_FAILED);

    coreNum_ = static_cast<int64_t>(coreNum);
    ubSize_ = ubSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dUpdate::CheckDtype()
{
    auto xInputDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputDesc);
    xDtype_ = xInputDesc->GetDataType();
    OP_CHECK_IF(
        supportedXDtype.count(xDtype_) == 0,
        OP_LOGE(
            context_->GetNodeName(), 
            "input x dtype [%s] not supported, only support [DT_BF16, DT_FLOAT16]",
            ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str()),
        return ge::GRAPH_FAILED);

    auto wInputDesc = context_->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wInputDesc);
    wDtype_ = wInputDesc->GetDataType();
    OP_CHECK_IF(
        wDtype_ != xDtype_,
        OP_LOGE(
            context_->GetNodeName(),
            "input weight dtype [%s] not equal to input x dtype",
            ge::TypeUtils::DataTypeToSerialString(wDtype_).c_str()),
        return ge::GRAPH_FAILED);

    auto convStateInputDesc = context_->GetInputDesc(CONV_STATE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, convStateInputDesc);
    convStateDtype_ = convStateInputDesc->GetDataType();
    OP_CHECK_IF(
        convStateDtype_ != xDtype_,
        OP_LOGE(
            context_->GetNodeName(),
            "input conv_state dtype [%s] not equal to input x dtype",
            ge::TypeUtils::DataTypeToSerialString(convStateDtype_).c_str()),
        return ge::GRAPH_FAILED);

    auto yOutputDesc = context_->GetOutputDesc(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yOutputDesc);
    yDtype_ = yOutputDesc->GetDataType();
    OP_CHECK_IF(
        yDtype_ != xDtype_,
        OP_LOGE(
            context_->GetNodeName(),
            "output y dtype [%s] not equal to input x dtype",
            ge::TypeUtils::DataTypeToSerialString(yDtype_).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dUpdate::CheckAttrs()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    // get activationMode
    const int64_t* activationModePtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVATION_MODE_INDEX);
    if (activationModePtr != nullptr) {
        activationMode_ = *activationModePtr;
        OP_CHECK_IF(
            activationMode_ != 0 && activationMode_ != 1, OP_LOGE(context_, "activation_mode only supports 0/1"),
            return ge::GRAPH_FAILED);
    }

    // get padSlotId
    const int64_t* padSlotIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_PAD_SLOT_ID_INDEX);
    if (padSlotIdPtr != nullptr) {
        padSlotId_ = *padSlotIdPtr;
    }    

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dUpdate::GetOpParam() {
    // x: (batch, dim) or (batch, seqLen, dim)
    auto xShapePtr = context_->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    xShape_ = EnsureNotScalar(xShapePtr->GetStorageShape());
    OP_CHECK_IF(
        xShape_.GetDimNum() != 2 && xShape_.GetDimNum() != 3,
        OP_LOGE(context_, "x must be 2D/3D: (batch, dim) or (batch, seqlen, dim) or (num_tokens, dim)"),
        return ge::GRAPH_FAILED);

    auto locShapePtr = context_->GetOptionalInputShape(QUERY_LOC_INDEX);
    if (locShapePtr != nullptr && locShapePtr->GetStorageShape().GetDimNum() != 0) {
        locShape_ = EnsureNotScalar(locShapePtr->GetStorageShape());
        OP_CHECK_IF(locShape_.GetDimNum() != 1, OP_LOGE(context_, "query_start_loc must be 1D: (batch + 1,)"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(locShape_.GetDim(0) < 2, OP_LOGE(context_, "query_start_loc must be >= 2"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(xShape_.GetDimNum() != 2, OP_LOGE(context_, "x must be 2D as input (num_tokens, dim) if using query_start_loc"), return ge::GRAPH_FAILED);
        hasQueryLoc_ = 1;
    }

    if (hasQueryLoc_) {
        batch_ = locShape_.GetDim(0) - 1;
        dim_ = xShape_.GetDim(1);
        OP_CHECK_IF(batch_ < 0 || dim_ <= 0, OP_LOGE(context_, "invalid x shape"), return ge::GRAPH_FAILED);
    } else {
        batch_ = xShape_.GetDim(0);
        if (xShape_.GetDimNum() == 2) {
            seqLen_ = 1;
            dim_ = xShape_.GetDim(1);
        } else if (xShape_.GetDimNum() == 3) {
            seqLen_ = xShape_.GetDim(1);
            dim_ = xShape_.GetDim(2);
        } else {
            return ge::GRAPH_FAILED;
        }

        OP_CHECK_IF(batch_ < 0 || dim_ <= 0 || seqLen_ < 0, OP_LOGE(context_, "invalid x shape"), return ge::GRAPH_FAILED);
    }

    auto wShapePtr = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wShapePtr);
    wShape_ = EnsureNotScalar(wShapePtr->GetStorageShape());
    OP_CHECK_IF(wShape_.GetDimNum() != 2, OP_LOGE(context_, "weight must be 2D: (dim, width)"), return ge::GRAPH_FAILED);

    width_ = wShape_.GetDim(0);
    OP_CHECK_IF(wShape_.GetDim(1) != dim_, OP_LOGE(context_, "[dim] in weight and [dim] in x must be equal"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(width_ != 4, OP_LOGE(context_, "currently weight.width only supports 4"), return ge::GRAPH_FAILED);

    // conv_state: (num_cache_lines, state_len, dim)
    auto convStateShapePtr = context_->GetInputShape(CONV_STATE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, convStateShapePtr);
    convStateShape_ = EnsureNotScalar(convStateShapePtr->GetStorageShape());
    stateLen_ = convStateShape_.GetDim(1);
    OP_CHECK_IF(
        convStateShape_.GetDimNum() != 3, OP_LOGE(context_, "conv_state must be 3D: (num_cache_lines, state_len, dim)"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(stateLen_ < (width_ - 1), OP_LOGE(context_, "[state_len] in conv_state must be >= width-1"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(convStateShape_.GetDim(2) != dim_, OP_LOGE(context_, "[dim] in conv_state and [dim] in x must be equal"), return ge::GRAPH_FAILED);

    // conv_state_indices: (batch)
    auto indicesShapePtr = context_->GetOptionalInputShape(CONV_STATE_INDICES_INDEX);
    // OP_CHECK_NULL_WITH_CONTEXT(context_, indicesShapePtr);
    if (indicesShapePtr != nullptr && indicesShapePtr->GetStorageShape().GetDimNum() != 0) {
        stateIndicesShape_ = EnsureNotScalar(indicesShapePtr->GetStorageShape());
        OP_CHECK_IF(stateIndicesShape_.GetDimNum() != 1, OP_LOGE(context_, "conv_state_indices must be 1D"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(stateIndicesShape_.GetDim(0) != batch_, OP_LOGE(context_, "conv_state_indices.size must equal batch"), return ge::GRAPH_FAILED);
        hasIndices_ = 1;
    }
    
    // bias: (dim) optional (allow empty optional tensor)
    auto biasShapePtr = context_->GetOptionalInputShape(BIAS_INDEX);
    if (biasShapePtr != nullptr && biasShapePtr->GetStorageShape().GetDimNum() != 0) {
        biasShape_ = EnsureNotScalar(biasShapePtr->GetStorageShape());
        OP_CHECK_IF(biasShape_.GetDimNum() != 1, OP_LOGE(context_, "bias must be 1D: (dim,)"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(biasShape_.GetDim(0) != dim_, OP_LOGE(context_, "bias.size must equal dim"), return ge::GRAPH_FAILED);
        hasBias_ = 1;
    }

    auto numShapePtr = context_->GetOptionalInputShape(NUM_ACCEPT_INDEX);
    if (numShapePtr != nullptr && numShapePtr->GetStorageShape().GetDimNum() != 0) {
        numShape_ = EnsureNotScalar(numShapePtr->GetStorageShape());
        OP_CHECK_IF(numShape_.GetDimNum() != 1, OP_LOGE(context_, "num_accepted_tokens must be 1D: (batch,)"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(numShape_.GetDim(0) != batch_, OP_LOGE(context_, "num_accepted_tokens.size must equal batch"), return ge::GRAPH_FAILED);
        hasNumAccept_ = 1;
    }

    OP_CHECK_IF(
        (CheckDtype() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "check input/output dtype failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckAttrs() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "op attrs is invalid."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

int64_t CausalConv1dUpdate::GetCoreNum(int64_t factor, int64_t coreNum) const
{
    // 计算用到的核数
    int64_t elePerCore = CeilDiv(factor, coreNum);
    int64_t actCore = CeilDiv(factor, elePerCore);
    return actCore;
}

void CausalConv1dUpdate::CalcBlockFactor(int64_t shape)
{
    blockFactor_ = CeilDiv(shape, actCoreNum_);
    blockTailFactor_ = shape - blockFactor_ * (actCoreNum_ - 1);
    blockTailFactor_ = blockTailFactor_ == 0 ? blockFactor_ : blockTailFactor_;
}

void CausalConv1dUpdate::CalcTiling()
{
    // batch * seqLen合轴，切合轴
    // int64_t shape = xShape_.GetDim(0) * xShape_.GetDim(1);

    // 分batch
    if(hasQueryLoc_) {
        int64_t shape = locShape_.GetDim(0) - 1;
        actCoreNum_ = GetCoreNum(shape, coreNum_);

        CalcBlockFactor(shape);
    } else {
        int64_t shape = xShape_.GetDim(0);
        actCoreNum_ = GetCoreNum(shape, coreNum_);
        
        CalcBlockFactor(shape);
    }

}

void CausalConv1dUpdate::CalcTilingKey()
{
    if (xDtype_ == ge::DT_BF16) {
        tilingKey_ = GET_TPL_TILING_KEY(TPL_BF16);
    } else if (xDtype_ == ge::DT_FLOAT16) {
        tilingKey_ = GET_TPL_TILING_KEY(TPL_FP16);
    }
}

void CausalConv1dUpdate::WriteTilingData() {
    //CausalConv1dUpdateTilingData* tilingData_ = context_->GetTilingData<CausalConv1dUpdateTilingData>();
    CausalConv1dUpdateTilingData tilingData_;

    // write tiling data
    OP_LOGD(context_->GetNodeName(), "coreNum: %ld, tilingKey: %lu", coreNum_, tilingKey_);
    context_->SetBlockDim(coreNum_);
    context_->SetTilingKey(tilingKey_);

    OP_LOGD(
        context_->GetNodeName(), "batch: %ld, seqLen: %ld, dim: %ld, width: %ld, stateLen: %ld, hasIndices: %ld, hasBias: %ld, hasNumAccept: %ld, hasQueryLoc: %ld, activationMode: %ld, padSlotId: %ld",  
                                  batch_, seqLen_, dim_, width_, stateLen_, hasIndices_, hasBias_, hasNumAccept_, hasQueryLoc_, activationMode_, padSlotId_);
    //tilingData_->batch = batch_;
    tilingData_.set_batch(batch_);
    //tilingData_->seqLen = seqLen_;
    tilingData_.set_seqLen(seqLen_);
    //tilingData_->dim = dim_;
    tilingData_.set_dim(dim_);
    //tilingData_->width = width_;
    tilingData_.set_width(width_);
    //tilingData_->stateLen = stateLen_;
    tilingData_.set_stateLen(stateLen_);
    //tilingData_->hasIndices = hasIndices_;
    tilingData_.set_hasIndices(hasIndices_);
    //tilingData_->hasBias = hasBias_;
    tilingData_.set_hasBias(hasBias_);
    //tilingData_->hasNumAccept = hasNumAccept_;
    tilingData_.set_hasNumAccept(hasNumAccept_);
    //tilingData_->hasQueryLoc = hasQueryLoc_;
    tilingData_.set_hasQueryLoc(hasQueryLoc_);
    //tilingData_->activationMode = activationMode_;
    tilingData_.set_activationMode(activationMode_);
    //tilingData_->padSlotId = padSlotId_;
    tilingData_.set_padSlotId(padSlotId_);

    OP_LOGD(
        context_->GetNodeName(), "actCoreNum: %ld, blockFactor: %ld, blockTailFactor: %ld", 
                                    actCoreNum_, blockFactor_, blockTailFactor_);
    //tilingData_->numCore = actCoreNum_;
    tilingData_.set_numCore(actCoreNum_);
    //tilingData_->blockFactor = blockFactor_;
    tilingData_.set_blockFactor(blockFactor_);
    //tilingData_->blockTailFactor = blockTailFactor_;
    tilingData_.set_blockTailFactor(blockTailFactor_);

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = SYNC_WORKSPACE_SIZE;

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

} // namespace causalconv1dupdate

static ge::graphStatus CausalConv1dUpdateTilingFunc(gert::TilingContext* context)
{
    OP_LOGD("CausalConv1dUpdateTiling", "Enter CausalConv1dUpdateTilingFunc");
    if (context == nullptr) {
        OP_LOGE("CausalConv1dUpdateTiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    causalconv1dupdate::CausalConv1dUpdate tiling(context);
    ge::graphStatus status = tiling.DoCausalConv1dUpdateTiling();
    return status;
}

static ge::graphStatus TilingParseForCausalConv1dUpdate([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CausalConv1dUpdate)
    .Tiling(CausalConv1dUpdateTilingFunc)
    .TilingParse<CausalConv1dUpdateCompileInfo>(TilingParseForCausalConv1dUpdate);

} // namespace optiling

