/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_compress_epilog_tiling_arch35.cpp
 * \brief Architecture-specific tiling implementation for KvCompressEpilog (arch35)
 */

#include <sstream>
#include "kv_compress_epilog_tiling_arch35.h"

using namespace ge;

namespace optiling {

template <typename T>
static inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd)));
}

int64_t RoundUp(int64_t x, int64_t y) {
    return CeilDiv(x, y) * y;
}

ge::graphStatus KvCompressEpilogTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<KvCompressEpilogCompileInfo>();
        OPS_ERR_IF(compileInfoPtr == nullptr,
                  OPS_LOG_E(context_, "compileInfoPtr is null"),
                  return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTiling::GetInputShapes()
{
    // Get input x shape (2D tensor)
    auto shapeX = context_->GetInputShape(X_INPUT_INDEX);
    OPS_ERR_IF(shapeX == nullptr,
              OPS_LOG_E(context_, "input x shape is null"),
              return ge::GRAPH_FAILED);

    const gert::Shape& inputShapeX = shapeX->GetStorageShape();
    int64_t rankX = inputShapeX.GetDimNum();
    OPS_ERR_IF(rankX != 2,
              OPS_LOG_E(context_, "input x must be 2D tensor, got rank %ld", rankX),
              return ge::GRAPH_FAILED);

    bs_ = inputShapeX.GetDim(0);
    d_ = inputShapeX.GetDim(1);

    OPS_LOG_I(context_->GetNodeName(), "input x shape: [%ld, %ld]",
              bs_, d_);

    // Get slot_mapping shape
    auto shapeSlotMapping = context_->GetInputShape(SLOT_MAPPING_INDEX);
    OPS_ERR_IF(shapeSlotMapping == nullptr,
              OPS_LOG_E(context_, "slot_mapping shape is null"),
              return ge::GRAPH_FAILED);
    const gert::Shape& slotMappingShape = shapeSlotMapping->GetStorageShape();
    int64_t slotMappingRank = slotMappingShape.GetDimNum();
    OPS_ERR_IF(slotMappingRank != 1,
              OPS_LOG_E(context_, "slot_mapping must be 1D tensor, got rank %ld", slotMappingRank),
              return ge::GRAPH_FAILED);
    int64_t slotMappingSize = slotMappingShape.GetDim(0);
    OPS_ERR_IF(slotMappingSize != bs_,
              OPS_LOG_E(context_, "slot_mapping size must equal x first dimension, got slot_mapping_size=%ld, dimX0=%ld",
                       slotMappingSize, bs_),
              return ge::GRAPH_FAILED);

    OPS_LOG_I(context_->GetNodeName(), "slot_mapping size: %ld", slotMappingSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTiling::GetAttributes()
{
    auto* attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr,
              OPS_LOG_E(context_, "get attrs nullptr"),
              return ge::GRAPH_FAILED);

    const int64_t* attrQuantGroupSize = attrs->GetAttrPointer<int64_t>(QUANT_GROUP_SIZE_ATTR_INDEX);
    if (attrQuantGroupSize != nullptr) {
        quantGroupSize_ = *attrQuantGroupSize;
    } else {
        quantGroupSize_ = DEFAULT_QUANT_GROUP_SIZE;
    }

    const int64_t* attrQuantMode = attrs->GetAttrPointer<int64_t>(QUANT_MODE_ATTR_INDEX);
    if (attrQuantMode != nullptr) {
        quantMode_ = *attrQuantMode;
    }

    const int64_t* attrRoundScale = attrs->GetAttrPointer<int64_t>(ROUND_SCALE_ATTR_INDEX);
    if (attrRoundScale != nullptr) {
        roundScale_ = *attrRoundScale;
    }

    OPS_LOG_I(context_->GetNodeName(), "quant_group_size: %ld quantMode_: %ld roundScale_: %ld", quantGroupSize_, quantMode_, roundScale_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTiling::GetDtypeInfo()
{
    auto xDesc = context_->GetInputDesc(X_INPUT_INDEX);
    OPS_ERR_IF(xDesc == nullptr,
              OPS_LOG_E(context_, "get x desc nullptr"),
              return ge::GRAPH_FAILED);
    xDtype_ = xDesc->GetDataType();

    OPS_ERR_IF(xDtype_ != ge::DT_BF16,
              OPS_LOG_E(context_, "x dtype only support BF16, got %d",
                       static_cast<int>(xDtype_)),
              return ge::GRAPH_FAILED);

    auto slotMappingDesc = context_->GetInputDesc(SLOT_MAPPING_INDEX);
    OPS_ERR_IF(slotMappingDesc == nullptr,
              OPS_LOG_E(context_, "get slot_mapping desc nullptr"),
              return ge::GRAPH_FAILED);
    slotMappingDtype_ = slotMappingDesc->GetDataType();

    OPS_ERR_IF((slotMappingDtype_ != ge::DT_INT32 && slotMappingDtype_ != ge::DT_INT64),
              OPS_LOG_E(context_, "slot_mapping dtype only support INT32/INT64, got %d",
                       static_cast<int>(slotMappingDtype_)),
              return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTiling::ValidateShapes()
{
    OPS_ERR_IF(bs_ <= 0,
              OPS_LOG_E(context_, "input x first dimension must be positive, got %ld", bs_),
              return ge::GRAPH_FAILED);

    OPS_ERR_IF(d_ <= 0,
              OPS_LOG_E(context_, "input x second dimension must be positive, got %ld",
                       d_),
              return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTiling::ValidateDtypes()
{
    // Output should have same dtype as input kv_compress_cache
    auto kvCacheOutputDesc = context_->GetOutputDesc(KV_COMPRESS_CACHE_OUTPUT_INDEX);
    OPS_ERR_IF(kvCacheOutputDesc == nullptr,
              OPS_LOG_E(context_, "get output desc nullptr"),
              return ge::GRAPH_FAILED);

    ge::DataType outputDtype = kvCacheOutputDesc->GetDataType();
    OPS_ERR_IF((outputDtype != ge::DT_FLOAT8_E5M2 && outputDtype != ge::DT_FLOAT8_E4M3FN),
              OPS_LOG_E(context_, "kv_compress_cache dtype only support FLOAT8_E5M2/FLOAT8_E4M3FN, got %d",
                       static_cast<int>(kvCacheDtype_)),
              return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTiling::DoOpTiling() 
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(coreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

    if (quantMode_ == QUANT_MDOE_GROUP_MXFP8) {
        scaleCol_ = CeilDiv(d_ - 64, static_cast<int64_t>(64));
    } else {
        scaleCol_ = CeilDiv(d_ - 64, static_cast<int64_t>(128));
    }
    
    int64_t scaleBytes = 4;
    if (quantMode_ == QUANT_MDOE_GROUP_MXFP8) {
       scaleBytes = 1;
    }
    int64_t concatCol = d_ - SLICE_SIZE + SLICE_SIZE * 2 + scaleCol_ * scaleBytes;
    kvCacheCol_ = RoundUp(concatCol, DEFAULT_QUANT_GROUP_SIZE);

    int64_t padCol = kvCacheCol_ - concatCol;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);

    int64_t xSize = rowOnceLoop * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;       
    int64_t ySize = rowOnceLoop * RoundUp(kvCacheCol_, 32) * 1 * DOUBLE_BUFFER;    
    int64_t tmpBufferSize = RoundUp(rowOnceLoop, 8) * 4;
    int64_t totalSize = xSize + ySize + tmpBufferSize;
    rowFactor_ = rowOnceLoop; 
    // d全载,尝试搬入更多的bs
    while (rowFactor_ <= rowOfFormerBlock_) {
        xSize = rowFactor_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;     
        ySize = rowFactor_ * RoundUp(kvCacheCol_, 32) * 1 * DOUBLE_BUFFER;    
        tmpBufferSize = RoundUp(rowFactor_, 8) * 4; 
        totalSize = xSize + ySize + tmpBufferSize;
        if (totalSize > ubSize_) {
            rowFactor_ = rowFactor_ - 1;
            break;
        }
        rowFactor_ = rowFactor_ + 1;
    }
    rowFactor_ = rowFactor_ > rowOfFormerBlock_ ? rowFactor_ - 1 : rowFactor_;

    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;  


    tilingData_.set_bs(bs_);
    tilingData_.set_d(d_);
    tilingData_.set_kvCacheCol(kvCacheCol_);
    tilingData_.set_scaleCol(scaleCol_);
    tilingData_.set_concatCol(concatCol);
    tilingData_.set_quantMode(quantMode_);
    tilingData_.set_roundScale(roundScale_);
    tilingData_.set_padCol(padCol);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_rowFactor(rowFactor_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTiling::PostTiling()
{
    // Set block dimension (number of AI cores to use)
    context_->SetBlockDim(usedCoreNums_);

    // Set tiling key for kernel dispatch
    context_->SetTilingKey(GetTilingKey());

    // Set workspace size
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspaces == nullptr,
              OPS_LOG_E(context_, "get workspaces nullptr"),
              return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(DEFAULT_WORKSPACE_SIZE);

    // Save tiling data to buffer
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
              OPS_LOG_E(context_, "get tilingdata nullptr"),
              return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                            context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // Log tiling information
    DumpTilingInfo();

    return ge::GRAPH_SUCCESS;
}

void KvCompressEpilogTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "bs: " << tilingData_.get_bs();
    info << ", d: " << tilingData_.get_d();
    info << ", kvCacheCol: " << tilingData_.get_kvCacheCol();
    info << ", scaleCol: " << tilingData_.get_scaleCol();
    info << ", concatCol: " << tilingData_.get_concatCol();
    info << ", quantMode: " << tilingData_.get_quantMode();
    info << ", roundScale: " << tilingData_.get_roundScale();
    info << ", padCol: " << tilingData_.get_padCol();
    info << ", rowOfFormerBlock: " << tilingData_.get_rowOfFormerBlock();
    info << ", rowOfTailBlock: " << tilingData_.get_rowOfTailBlock();
    info << ", rowLoopOfFormerBlock: " << tilingData_.get_rowLoopOfFormerBlock();
    info << ", rowLoopOfTailBlock: " << tilingData_.get_rowLoopOfTailBlock();
    info << ", rowFactor: " << tilingData_.get_rowFactor();
    info << ", tailRowFactorOfFormerBlock: " << tilingData_.get_tailRowFactorOfFormerBlock();
    info << ", tailRowFactorOfTailBlock: " << tilingData_.get_tailRowFactorOfTailBlock();

    OPS_LOG_I(context_, "%s", info.str().c_str());
}

uint64_t KvCompressEpilogTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus KvCompressEpilogTiling::GetShapeAttrsInfo()
{
    OPS_ERR_IF(context_ == nullptr,
              OPS_LOG_E(context_, "context can not be nullptr"),
              return ge::GRAPH_FAILED);

    OPS_ERR_IF(GetInputShapes() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_, "GetInputShapes failed"),
              return ge::GRAPH_FAILED);

    OPS_ERR_IF(GetAttributes() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_, "GetAttributes failed"),
              return ge::GRAPH_FAILED);

    OPS_ERR_IF(GetDtypeInfo() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_, "GetDtypeInfo failed"),
              return ge::GRAPH_FAILED);

    OPS_ERR_IF(ValidateShapes() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_, "ValidateShapes failed"),
              return ge::GRAPH_FAILED);

    OPS_ERR_IF(ValidateDtypes() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_, "ValidateDtypes failed"),
              return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTiling::RunTiling()
{
    OPS_ERR_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_, "GetPlatformInfo failed"),
              return ge::GRAPH_FAILED);

    OPS_ERR_IF(GetShapeAttrsInfo() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_, "GetShapeAttrsInfo failed"),
              return ge::GRAPH_FAILED);

    OPS_ERR_IF(DoOpTiling() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_, "CalcOpTiling failed"),
              return ge::GRAPH_FAILED);

    OPS_ERR_IF(PostTiling() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_, "PostTiling failed"),
              return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ---------- Tiling Entry Points ----------
ge::graphStatus TilingForKvCompressEpilog(gert::TilingContext* context)
{
    OPS_ERR_IF(context == nullptr,
              OPS_REPORT_VECTOR_INNER_ERR("KvCompressEpilog", "Tiling context is null"),
              return ge::GRAPH_FAILED);

    KvCompressEpilogTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForKvCompressEpilog(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvCompressEpilog)
    .Tiling(TilingForKvCompressEpilog)
    .TilingParse<KvCompressEpilogCompileInfo>(TilingPrepareForKvCompressEpilog);

}  // namespace optiling
