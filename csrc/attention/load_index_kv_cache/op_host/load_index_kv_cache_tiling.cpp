/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file load_index_kv_cache_tiling.cpp
 * \brief
 */

#include <sstream>
#include "load_index_kv_cache_tiling.h"

using namespace ge;
namespace optiling {
namespace {
constexpr uint64_t WORKSPACE_SIZE = 32;
int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return (x + y - 1) / y;
    }
    return x;
}
int64_t DownAlign(int64_t x, int64_t y) {
    if (y == 0) {
        return x;
    }
    return (x / y) * y;
}
int64_t RoundUp(int64_t x, int64_t y) {
    return CeilDiv(x, y) * y;
}

constexpr int64_t INPUT_KV_CACHE_IDX = 0;
constexpr int64_t INPUT_SLOT_MAPPING_IDX = 1;
constexpr int64_t ATTR_BLOCK_STRIDE_INDEX = 0;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t INPUTS_DIM_LIMIT = 4;
}

ge::graphStatus LoadIndexKvCacheTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<LoadIndexKvCacheCompileInfo>();
        OPS_ERR_IF(compileInfoPtr == nullptr, OPS_LOG_E(context_, "compile info is null"),
                      return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LoadIndexKvCacheTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);

    auto blockStrideAttr = attrs->GetAttrPointer<int64_t>(ATTR_BLOCK_STRIDE_INDEX);
    blockStride_ = blockStrideAttr != nullptr ? *blockStrideAttr : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LoadIndexKvCacheTiling::GetShapeAttrsInfoInner()
{
    auto shapeKvCache = context_->GetInputShape(INPUT_KV_CACHE_IDX);
    OPS_LOG_E_IF_NULL(context_, shapeKvCache, return ge::GRAPH_FAILED);
    auto kvCacheStorageShape = shapeKvCache->GetStorageShape();
    OPS_ERR_IF(
        (kvCacheStorageShape.GetDimNum() != INPUTS_DIM_LIMIT),
        OPS_LOG_E(context_, "the dim of kv_cache only support %d, please check.", INPUTS_DIM_LIMIT),
        return ge::GRAPH_FAILED);

    bn_ = kvCacheStorageShape.GetDim(DIM_0);
    bs_ = kvCacheStorageShape.GetDim(DIM_1);
    d_ = kvCacheStorageShape.GetDim(DIM_3);

    auto shapeSlotMapping = context_->GetInputShape(INPUT_SLOT_MAPPING_IDX);
    OPS_LOG_E_IF_NULL(context_, shapeSlotMapping, return ge::GRAPH_FAILED);
    auto slotMappingStorageShape = shapeSlotMapping->GetStorageShape();
    OPS_ERR_IF(
        (slotMappingStorageShape.GetDimNum() != 1),
        OPS_LOG_E(context_, "the dim of slot_mapping must be 1, please check."),
        return ge::GRAPH_FAILED);

    n_ = slotMappingStorageShape.GetDim(0);

    OPS_ERR_IF(GetAttr() != ge::GRAPH_SUCCESS,
                  OPS_LOG_E(context_->GetNodeName(), "get attr failed."),
                  return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LoadIndexKvCacheTiling::CheckShapesAndAttrs()
{
    blockStride_ = blockStride_ == 0 ? bs_ * d_ : blockStride_;
    OPS_ERR_IF(blockStride_ < bs_ * d_,
            OPS_LOG_E(context_, "stride_kvcache must be greater than last dim of kv_cache %ld.", d_),
            return ge::GRAPH_FAILED);
    // 如果blockStride_为0,则认为连续
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus LoadIndexKvCacheTiling::CalcOpTiling()
{
    // 默认尾轴在ub内全载
    rowOfFormerBlock_ = CeilDiv(n_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(CeilDiv(n_, rowOfFormerBlock_), static_cast<int64_t>(coreNum_));
    rowOfTailBlock_ = n_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

    tilingData_.set_bn(bn_);
    tilingData_.set_bs(bs_);
    tilingData_.set_d(d_);
    tilingData_.set_n(n_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_blockStride(blockStride_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LoadIndexKvCacheTiling::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (CheckShapesAndAttrs() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (CalcOpTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetWorkspaceSize() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (PostTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    context_->SetTilingKey(0);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LoadIndexKvCacheTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LoadIndexKvCacheTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForLoadIndexKvCache(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForLoadIndexKvCache(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("LoadIndexKvCache", "Tiling context is null"),
               return ge::GRAPH_FAILED);
    LoadIndexKvCacheTiling LoadIndexKvCacheTiling(context);
    return LoadIndexKvCacheTiling.DoOpTiling();
}

IMPL_OP_OPTILING(LoadIndexKvCache)
    .Tiling(TilingForLoadIndexKvCache)
    .TilingParse<LoadIndexKvCacheCompileInfo>(TilingPrepareForLoadIndexKvCache);

}  // namespace optiling