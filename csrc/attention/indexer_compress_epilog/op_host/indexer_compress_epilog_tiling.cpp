/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file indexer_compress_epilog_tiling.cpp
 * \brief
 */

#include <sstream>
#include "indexer_compress_epilog_tiling.h"

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


constexpr int64_t INPUT_X_IDX = 0;
constexpr int64_t INPUT_SLOT_MAPPING_IDX = 1;
constexpr int64_t ATTR_QUANT_MODE_INDEX = 0;
constexpr int64_t ATTR_ROUND_SCALE_INDEX = 1;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t DOUBLE_BUFFER = 2;
// per_block量化,每128个f16需要量化出一个scale, 因此切分尾轴时，以128为factor进行切分
constexpr int64_t PER_BLOCK_FP16 = 128;
}

ge::graphStatus IndexerCompressEpilogTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<IndexerCompressEpilogCompileInfo>();
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

ge::graphStatus IndexerCompressEpilogTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);

    auto quantMode = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    quantMode_ = quantMode == nullptr ? 1 : *quantMode;

    auto roundScale = attrs->GetAttrPointer<int64_t>(ATTR_ROUND_SCALE_INDEX);
    roundScale_ = roundScale == nullptr ? true : *roundScale;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCompressEpilogTiling::GetShapeAttrsInfoInner()
{
    auto shapeX = context_->GetInputShape(INPUT_X_IDX);
    OPS_LOG_E_IF_NULL(context_, shapeX, return ge::GRAPH_FAILED);
    auto xStorageShape = shapeX->GetStorageShape();
    d_ = xStorageShape.GetDim(xStorageShape.GetDimNum() - 1);
    
    auto shapeSlotMapping = context_->GetInputShape(INPUT_SLOT_MAPPING_IDX);
    OPS_LOG_E_IF_NULL(context_, shapeSlotMapping, return ge::GRAPH_FAILED);
    auto slotMappingStorageShape = shapeSlotMapping->GetStorageShape();
    bs_ = slotMappingStorageShape.GetDim(0);

    scaleCol_ = CeilDiv(d_, PER_BLOCK_FP16);

    OPS_ERR_IF(GetAttr() != ge::GRAPH_SUCCESS,
                  OPS_LOG_E(context_->GetNodeName(), "get attr failed."),
                  return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus IndexerCompressEpilogTiling::CalcOpTiling() 
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(coreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);
    
    rowFactor_ = rowOnceLoop; 
    int64_t scaleByteSize = 4;
    if (quantMode_ == 0) {
        scaleByteSize = 1;
    }
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

    tilingData_.set_bs(bs_);
    tilingData_.set_d(d_);
    tilingData_.set_scaleCol(scaleCol_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_rowFactor(rowFactor_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_quantMode(quantMode_);
    int64_t roundScaleData = roundScale_ ? 1 : 0;
    tilingData_.set_roundScale(roundScaleData);

    tilingKey_ = 10000;
    if (rowFactor_ != 1) {
        tilingKey_ += 10;
    }
    if (quantMode_ == 1) {
        tilingKey_ += 1;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCompressEpilogTiling::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
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

ge::graphStatus IndexerCompressEpilogTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCompressEpilogTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForIndexerCompressEpilog(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForIndexerCompressEpilog(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("IndexerCompressEpilog", "Tiling context is null"),
               return ge::GRAPH_FAILED);
    IndexerCompressEpilogTiling IndexerCompressEpilogTiling(context);
    return IndexerCompressEpilogTiling.DoOpTiling();
}

IMPL_OP_OPTILING(IndexerCompressEpilog)
    .Tiling(TilingForIndexerCompressEpilog)
    .TilingParse<IndexerCompressEpilogCompileInfo>(TilingPrepareForIndexerCompressEpilog);

}  // namespace optiling
