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
 * \file indexer_compress_epilog_v2_tiling.cpp
 * \brief
 */

#include <sstream>
#include "indexer_compress_epilog_v2_tiling.h"

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

constexpr int64_t INPUT_CACHE_IDX = 0;
constexpr int64_t INPUT_X_IDX = 1;
constexpr int64_t INPUT_SLOT_MAPPING_IDX = 2;
constexpr int64_t ATTR_LAYOUT_INDEX = 0;
constexpr int64_t ATTR_BLOCK_STRIDE_INDEX = 1;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t DOUBLE_BUFFER = 2;
// per_block量化,每128个f16需要量化出一个scale, 因此切分尾轴时，以128为factor进行切分
constexpr int64_t PER_BLOCK_FP16 = 128;
constexpr int64_t SINGLE_ROW = 1;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t SINGLE_ROW_TILING_KEY = 20020;
constexpr int64_t MULTI_ROW_TILING_KEY = 20021;
constexpr int64_t INPUTS_DIM_LIMIT = 4;
}

ge::graphStatus IndexerCompressEpilogV2Tiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<IndexerCompressEpilogV2CompileInfo>();
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

ge::graphStatus IndexerCompressEpilogV2Tiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);

    auto layout = attrs->GetAttrPointer<int64_t>(ATTR_LAYOUT_INDEX);
    layout_ = layout == nullptr ? 2 : *layout;

    auto blockStride = attrs->GetAttrPointer<int64_t>(ATTR_BLOCK_STRIDE_INDEX);
    blockStride_ = blockStride == nullptr ? 0 : *blockStride;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCompressEpilogV2Tiling::GetShapeAttrsInfoInner()
{
    auto shapeCache = context_->GetInputShape(INPUT_CACHE_IDX);
    OPS_LOG_E_IF_NULL(context_, shapeCache, return ge::GRAPH_FAILED);
    auto cacheStorageShape = shapeCache->GetStorageShape();
    OPS_ERR_IF(
        (cacheStorageShape.GetDimNum() != INPUTS_DIM_LIMIT),
        OPS_LOG_E(context_, "the dim of indexer_compress_cache only support %d, please check.", INPUTS_DIM_LIMIT),
        return ge::GRAPH_FAILED);
    cacheBs_ = cacheStorageShape.GetDim(DIM_1);
    OPS_ERR_IF(
        (cacheStorageShape.GetDim(DIM_2) != 1),
        OPS_LOG_E(context_, "the third dim of indexer_compress_cache must be 1, please check."),
        return ge::GRAPH_FAILED);
    cacheD_ = cacheStorageShape.GetDim(DIM_3);

    auto shapeX = context_->GetInputShape(INPUT_X_IDX);
    OPS_LOG_E_IF_NULL(context_, shapeX, return ge::GRAPH_FAILED);
    auto xStorageShape = shapeX->GetStorageShape();
    d_ = xStorageShape.GetDim(xStorageShape.GetDimNum() - 1);

    scaleCol_ = CeilDiv(d_, PER_BLOCK_FP16);

    auto shapeSlotMapping = context_->GetInputShape(INPUT_SLOT_MAPPING_IDX);
    OPS_LOG_E_IF_NULL(context_, shapeSlotMapping, return ge::GRAPH_FAILED);
    auto slotMappingStorageShape = shapeSlotMapping->GetStorageShape();
    tnd_ = 1;
    for (int i = 0; i < slotMappingStorageShape.GetDimNum(); i++) {
        tnd_ = tnd_ * slotMappingStorageShape.GetDim(i);
    }

    OPS_ERR_IF(GetAttr() != ge::GRAPH_SUCCESS,
                  OPS_LOG_E(context_->GetNodeName(), "get attr failed."),
                  return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCompressEpilogV2Tiling::CheckShapesAndAttrs()
{
    int64_t minBlockSize = cacheBs_ * cacheD_;
    OPS_ERR_IF((blockStride_ != 0 && blockStride_ < minBlockSize),
                OPS_LOG_E(context_, "block_stride must be greater than min block size %ld.", minBlockSize),
                return ge::GRAPH_FAILED);

    OPS_ERR_IF((layout_ != 2),
                OPS_LOG_E(context_, "layout only support 2, please check."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus IndexerCompressEpilogV2Tiling::CalcOpTiling()
{
    rowOfFormerBlock_ = CeilDiv(tnd_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(CeilDiv(tnd_, rowOfFormerBlock_), static_cast<int64_t>(coreNum_));
    rowOfTailBlock_ = tnd_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);

    rowFactor_ = rowOnceLoop;
    int64_t scaleByteSize = 4;
    int64_t perBlockScaleElemNum = BLOCK_SIZE / scaleByteSize;
    // d全载,尝试搬入更多的bs
    while (rowFactor_ <= rowOfFormerBlock_) {
        int64_t xSize = rowFactor_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
        int64_t ySize = rowFactor_ * RoundUp(d_, 32) * 1 * DOUBLE_BUFFER;
        int64_t scaleSize = rowFactor_ * RoundUp(scaleCol_, perBlockScaleElemNum) * scaleByteSize * DOUBLE_BUFFER;
        int64_t tmpBufferSize = RoundUp(rowFactor_, 8) * 4;
        int64_t totalSize = xSize + ySize + scaleSize + tmpBufferSize;
        if (totalSize > ubSize_) {
            rowFactor_ = rowFactor_ - 1;
            break;
        }
        rowFactor_ = rowFactor_ + 1;
    }
    if (rowFactor_ > rowOfFormerBlock_) {
        rowFactor_--;
    }

    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;

    // 如果未指定blockStride，设置默认值为单个block大小（连续排列）
    if (blockStride_ == 0) {
        blockStride_ = cacheBs_ * cacheD_;
    }

    tilingData_.set_d(d_);
    tilingData_.set_cacheBs(cacheBs_);
    tilingData_.set_scaleCol(scaleCol_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_rowFactor(rowFactor_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_blockStride(blockStride_);

    // SINGLE_ROW TILING_KEY : 20020
    // MULTI_ROW TILING_KEY : 20021
    if (rowFactor_ == SINGLE_ROW) {
        tilingKey_ = SINGLE_ROW_TILING_KEY;
    } else {
        tilingKey_ = MULTI_ROW_TILING_KEY;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCompressEpilogV2Tiling::DoOpTiling()
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
    context_->SetTilingKey(tilingKey_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCompressEpilogV2Tiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCompressEpilogV2Tiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForIndexerCompressEpilogV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForIndexerCompressEpilogV2(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("IndexerCompressEpilogV2", "Tiling context is null"),
               return ge::GRAPH_FAILED);
    IndexerCompressEpilogV2Tiling IndexerCompressEpilogV2Tiling(context);
    return IndexerCompressEpilogV2Tiling.DoOpTiling();
}

IMPL_OP_OPTILING(IndexerCompressEpilogV2)
    .Tiling(TilingForIndexerCompressEpilogV2)
    .TilingParse<IndexerCompressEpilogV2CompileInfo>(TilingPrepareForIndexerCompressEpilogV2);

}  // namespace optiling