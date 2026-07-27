/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_update_v2_tiling.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "platform/platform_infos_def.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_key.h"
#include "scatter_nd_update_v2_tiling.h"

namespace optiling {
// using namespace Ops::NN::Optiling;
constexpr uint64_t MAX_DIM_NUM = 8;
constexpr uint64_t MAX_LENGTH_INT32 = (1LL << 31) - 1;
constexpr uint64_t MAX_FLOAT_EXPRESS_INT32 = (1LL << 24) - 1;
constexpr uint64_t SORT_USE_GM_NUM = 2;
constexpr uint64_t SORT_BLOCK_LENGTH = 4096;
constexpr uint64_t GATHER_USE_NUM = 2;
constexpr uint64_t ALIGNED_NUM = 8;
constexpr uint64_t ALIGNED_SIZE = 32;
constexpr uint64_t ATTR_STRIDE = 0;
class ScatterNdUpdateV2Tiling {
public:
    explicit ScatterNdUpdateV2Tiling(gert::TilingContext* context) : tilingContext_(context){}
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint() const;

private:
    inline bool IsSort(uint64_t totalLength, uint64_t indexRow);
    inline bool IsLinearIndex(uint64_t totalLength);
    inline size_t CalcWorkSpaceSize(uint64_t indexRow);
    inline void SetTilingKeyMode();
    inline void GetDtypeSize();
    inline void Tiling4Scatter(uint64_t totalLength, uint64_t indexRow);
    inline void Tiling4LinearIndex(uint64_t indexRow, uint64_t indexDim);

    ScatterNdUpdateV2TilingData tilingData_;
    gert::TilingContext* tilingContext_ = nullptr;

    uint64_t coreNum_ = 0;
    uint64_t tilingKey_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t isLinearIndex_ = false;
    uint64_t isSort_ = false;
    uint64_t sortWorkspace_ = 0;
    uint64_t dataTypeSize_ = 0;
    uint64_t isInt64Indices_ = false;
    uint64_t needLargeIndexKernel_ = false;

private:
    // LinearIndex
    uint64_t indexDim_ = 0;
    uint64_t blockLength_ = 0;
    uint64_t blockNum_ = 0;
    uint64_t blockRemainLength_ = 0;
    uint64_t tailBlockNum_ = 0;
    uint64_t frontBlockNum_ = 0;
    uint64_t frontCoreNum_ = 0;
    uint64_t tailCoreNum_ =  0;
    uint64_t indicesMask_[MAX_DIM_NUM] = {0};

    // Scatter
    uint64_t scatterLength_ = 1;
    uint64_t tailRow_ = 0;
    uint64_t frontRow_ = 0;
    uint64_t frontNum_ = 0;
    uint64_t tailNum_ = 0;
    uint64_t ubLengthForUpdates_ = 0;
    uint64_t scatterAlignLength_ = 0;
    uint64_t formDim_ = 0;
    uint64_t copyRow_ = 0;
    uint64_t scatterTileNum_ = 1;
    uint64_t scatterTileLength_ = 0;
    uint64_t scatterTileTail_ = 0;
    uint64_t scatterTileAlignLength_ = 0;
};

inline void ScatterNdUpdateV2Tiling::SetTilingKeyMode()
{
    // tilingKey: indexType * 10 + sortFlag (indexType: 1=int32, 2=int64(cast), 3=int64(large))
    uint64_t indexType;
    if (!isInt64Indices_) {
        indexType = 1;
    } else if (needLargeIndexKernel_) {
        indexType = 3;
    } else {
        indexType = 2;
    }
    uint64_t sortFlag = (indexType == 3) ? 0 : (isSort_ ? 1 : 0);
    tilingKey_ = indexType * 10 + sortFlag;

    tilingContext_->SetTilingKey(tilingKey_);
    OP_LOGD(tilingContext_, "isLinearIndex=%lu, isSort=%lu, isInt64Indices=%lu, needLargeIndexKernel=%lu, tilingKey=%lu (indexType=%lu, sortFlag=%lu)",
            isLinearIndex_, isSort_, isInt64Indices_, needLargeIndexKernel_, tilingKey_, indexType, sortFlag);
}

inline bool ScatterNdUpdateV2Tiling::IsLinearIndex(uint64_t totalLength)
{
    return totalLength <= MAX_LENGTH_INT32;
}

inline bool ScatterNdUpdateV2Tiling::IsSort(uint64_t totalLength, uint64_t indexRow)
{
    return totalLength <= MAX_FLOAT_EXPRESS_INT32;
}

inline void ScatterNdUpdateV2Tiling::Tiling4LinearIndex(uint64_t indexRow, uint64_t indexDim)
{
    OP_LOGD(tilingContext_, "linearIndexTiling start");
    auto attrs = tilingContext_->GetAttrs();
    auto stridesPtr = attrs->GetListInt(ATTR_STRIDE);
    for (uint64_t i = 0; i < indexDim; ++i) {
        indicesMask_[i] = static_cast<uint64_t>(stridesPtr->GetData()[i]);
    }
    uint64_t coeff = isInt64Indices_ ? (2 * indexDim + 3) : (indexDim + 3);
    uint64_t maxBlockLength = ubSize_ / coeff / sizeof(int);
    blockLength_ = (maxBlockLength / ALIGNED_SIZE) * ALIGNED_SIZE;
    blockLength_ = std::min(blockLength_, (uint64_t)SORT_BLOCK_LENGTH);
    blockNum_ = indexRow / blockLength_;
    blockRemainLength_ = indexRow % blockLength_;

    if (blockNum_ == 0) {
        tailBlockNum_ = 0;
        frontBlockNum_ = 0;
        frontCoreNum_ = 1;
        tailCoreNum_ = 0;
    } else {
        tailBlockNum_ = blockNum_ / coreNum_;
        frontBlockNum_ = tailBlockNum_ + 1;
        frontCoreNum_ = blockNum_ % coreNum_;
        tailCoreNum_ =  tailBlockNum_ == 0 ? 0 : coreNum_ - frontCoreNum_;
    }
    OP_LOGD(tilingContext_, "linearIndexTiling finish");
}

inline void ScatterNdUpdateV2Tiling::Tiling4Scatter(uint64_t totalLength, uint64_t indexRow)
{
    OP_LOGD(tilingContext_, "scatterTiling start new");
    uint64_t scatterAlignNum = ALIGNED_SIZE / dataTypeSize_;
    tailRow_ = totalLength / coreNum_;
    frontRow_ = tailRow_ + 1;
    frontNum_ = totalLength % coreNum_;
    tailNum_ = tailRow_ == 0 ? 0 : coreNum_ - frontNum_;
    ubLengthForUpdates_ = ((ubSize_ - SORT_BLOCK_LENGTH * SORT_USE_GM_NUM * sizeof(int)) / ALIGNED_SIZE * ALIGNED_SIZE) / dataTypeSize_;
    scatterAlignLength_ = (scatterLength_ + scatterAlignNum - 1) & ~(scatterAlignNum - 1);
    formDim_ = scatterAlignLength_ / ubLengthForUpdates_;

    scatterTileLength_ = std::min(scatterLength_, ubLengthForUpdates_);
    if (scatterTileLength_ == 0) {
        scatterTileLength_ = 1;
    }
    scatterTileNum_ = (scatterLength_ + scatterTileLength_ - 1) / scatterTileLength_;
    scatterTileTail_ = scatterLength_ - (scatterTileNum_ - 1) * scatterTileLength_;
    scatterTileAlignLength_ = (scatterTileLength_ + scatterAlignNum - 1) & ~(scatterAlignNum - 1);

    if (scatterTileNum_ > 1) {
        copyRow_ = 1;
    } else {
        copyRow_ = formDim_ == 0 ? ubLengthForUpdates_ / scatterAlignLength_ : 1;
    }
    OP_LOGD(tilingContext_, "scatterTiling finish");
}

inline void ScatterNdUpdateV2Tiling::GetDtypeSize()
{
    uint64_t varDtype = tilingContext_->GetInputDesc(0)->GetDataType();
    switch (varDtype){
        case ge::DT_FLOAT:
            dataTypeSize_ = 4;
            break;
        case ge::DT_BF16:
            dataTypeSize_ = 2;
            break;
        case ge::DT_FLOAT16:
            dataTypeSize_ = 2;
            break;
        case ge::DT_BOOL:
            dataTypeSize_ = 1;
            break;
        case ge::DT_INT64:
            dataTypeSize_ = 8;
            break;
        case ge::DT_INT32:
            dataTypeSize_ = 4;
            break;
        case ge::DT_INT16:
            dataTypeSize_ = 2;
            break;
        case ge::DT_INT8:
            dataTypeSize_ = 1;
            break;
        default:
            break;
    }
}


ge::graphStatus ScatterNdUpdateV2Tiling::SetKernelTiling()
{
    tilingContext_->SetBlockDim(coreNum_);
    tilingData_.linearIndexTiling.set_indexDim(indexDim_);
    tilingData_.linearIndexTiling.set_ubSize(ubSize_);
    tilingData_.linearIndexTiling.set_indicesMask(indicesMask_);
    tilingData_.linearIndexTiling.set_coreNum(coreNum_);
    tilingData_.linearIndexTiling.set_blockLength(blockLength_);
    tilingData_.linearIndexTiling.set_blockNum(blockNum_);
    tilingData_.linearIndexTiling.set_blockRemainLength(blockRemainLength_);
    tilingData_.linearIndexTiling.set_tailBlockNum(tailBlockNum_);
    tilingData_.linearIndexTiling.set_frontBlockNum(frontBlockNum_);
    tilingData_.linearIndexTiling.set_frontCoreNum(frontCoreNum_);
    tilingData_.linearIndexTiling.set_tailCoreNum(tailCoreNum_);
    tilingData_.linearIndexTiling.set_sortWorkspace(sortWorkspace_);
    tilingData_.linearIndexTiling.set_isInt64Indices(isInt64Indices_);
    tilingData_.linearIndexTiling.set_needLargeIndexKernel(needLargeIndexKernel_);
    tilingData_.scatterTiling.set_scatterLength(scatterLength_);
    tilingData_.scatterTiling.set_tailRow(tailRow_);
    tilingData_.scatterTiling.set_frontRow(frontRow_);
    tilingData_.scatterTiling.set_frontNum(frontNum_);
    tilingData_.scatterTiling.set_tailNum(tailNum_);
    tilingData_.scatterTiling.set_ubLengthForUpdates(ubLengthForUpdates_);
    tilingData_.scatterTiling.set_scatterAlignLength(scatterAlignLength_);
    tilingData_.scatterTiling.set_formDim(formDim_);
    tilingData_.scatterTiling.set_copyRow(copyRow_);
    tilingData_.scatterTiling.set_scatterTileNum(scatterTileNum_);
    tilingData_.scatterTiling.set_scatterTileLength(scatterTileLength_);
    tilingData_.scatterTiling.set_scatterTileTail(scatterTileTail_);
    tilingData_.scatterTiling.set_scatterTileAlignLength(scatterTileAlignLength_);
    tilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

inline size_t ScatterNdUpdateV2Tiling::CalcWorkSpaceSize(uint64_t indexRow)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t indexRowAligned = (indexRow + ALIGNED_NUM - 1) & ~(ALIGNED_NUM - 1);
    sortWorkspace_ = indexRowAligned;
    size_t totalWorkspace = sysWorkspaceSize;
    if (isLinearIndex_) {
        totalWorkspace += sortWorkspace_ * SORT_USE_GM_NUM * sizeof(int);
    }
    if (isSort_) {
        totalWorkspace += sortWorkspace_ * SORT_USE_GM_NUM * sizeof(int);
    }
    return totalWorkspace;
}

void ScatterNdUpdateV2Tiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext_, "coreNum:                   %lu", coreNum_);
    OP_LOGD(tilingContext_, "tilingKey:                 %lu", tilingKey_);
    OP_LOGD(tilingContext_, "isInt64Indices:            %lu", isInt64Indices_);
    OP_LOGD(tilingContext_, "needLargeIndexKernel:      %lu", needLargeIndexKernel_);
    OP_LOGD(tilingContext_, "tiling for LinearIndex--------");
    OP_LOGD(tilingContext_, "indexDim:                  %lu", indexDim_);
    OP_LOGD(tilingContext_, "ubSize:                    %lu", ubSize_);
    OP_LOGD(tilingContext_, "blockLength:               %lu", blockLength_);
    OP_LOGD(tilingContext_, "blockNum:                  %lu", blockNum_);
    OP_LOGD(tilingContext_, "blockRemainLength:         %lu", blockRemainLength_);
    OP_LOGD(tilingContext_, "tailBlockNum:              %lu", tailBlockNum_);
    OP_LOGD(tilingContext_, "frontBlockNum:             %lu", frontBlockNum_);
    OP_LOGD(tilingContext_, "frontCoreNum:              %lu", frontCoreNum_);
    OP_LOGD(tilingContext_, "tailCoreNum:               %lu", tailCoreNum_);
    OP_LOGD(tilingContext_, "sortWorkspace:             %lu", sortWorkspace_);
    for (size_t i = 0; i < indexDim_; i++) {
        OP_LOGD(tilingContext_, "indicesMask[%lu]:            %lu", i, indicesMask_[i]);
    }
    OP_LOGD(tilingContext_, "tiling for Scatter------------");
    OP_LOGD(tilingContext_, "scatterLength:             %lu", scatterLength_);
    OP_LOGD(tilingContext_, "tailRow:                   %lu", tailRow_);
    OP_LOGD(tilingContext_, "frontRow:                  %lu", frontRow_);
    OP_LOGD(tilingContext_, "frontNum:                  %lu", frontNum_);
    OP_LOGD(tilingContext_, "tailNum:                   %lu", tailNum_);
    OP_LOGD(tilingContext_, "ubLengthForUpdates:        %lu", ubLengthForUpdates_);
    OP_LOGD(tilingContext_, "scatterAlignLength:        %lu", scatterAlignLength_);
    OP_LOGD(tilingContext_, "formDim:                   %lu", formDim_);
    OP_LOGD(tilingContext_, "copyRow:                   %lu", copyRow_);
    OP_LOGD(tilingContext_, "scatterTileNum:            %lu", scatterTileNum_);
    OP_LOGD(tilingContext_, "scatterTileLength:         %lu", scatterTileLength_);
    OP_LOGD(tilingContext_, "scatterTileTail:           %lu", scatterTileTail_);
    OP_LOGD(tilingContext_, "scatterTileAlignLength:    %lu", scatterTileAlignLength_);
}

ge::graphStatus ScatterNdUpdateV2Tiling::Init()
{
    OP_LOGD(tilingContext_, "Tiling initing");
    auto compileInfo = static_cast<const ScatterNdUpdateV2CompileInfo*>(tilingContext_->GetCompileInfo());
    auto varRefShape = tilingContext_->GetInputShape(0)->GetStorageShape();
    auto indicesShape = tilingContext_->GetInputShape(1)->GetStorageShape();
    auto updatesShape = tilingContext_->GetInputShape(2)->GetStorageShape();
    uint64_t varDimNum = varRefShape.GetDimNum();
    indexDim_ = indicesShape.GetDim(indicesShape.GetDimNum() - 1);

    auto indicesDtype = tilingContext_->GetInputDesc(1)->GetDataType();
    isInt64Indices_ = (indicesDtype == ge::DT_INT64);
    OP_LOGD(tilingContext_, "indicesDtype=%d, isInt64Indices=%lu", indicesDtype, isInt64Indices_);

    uint64_t totalLength = 1;
    for (uint64_t i = 0; i < indexDim_; ++i) {
        totalLength *= varRefShape.GetDim(i);
    }

    if (isInt64Indices_) {
        needLargeIndexKernel_ = !IsLinearIndex(totalLength);
    }

    if (varDimNum > indexDim_) {
        for (uint64_t i = indexDim_; i < varDimNum; i++) {
            scatterLength_ *= varRefShape.GetDim(i);
        }
    }
    uint64_t indexRow = 1;
    for (uint64_t i = 0; i < indicesShape.GetDimNum() - 1; i++) {
        indexRow *= indicesShape.GetDim(i);
    }

    if (needLargeIndexKernel_) {
        isSort_ = false;
        isLinearIndex_ = false;
    } else {
        isSort_ = false;
        isLinearIndex_ = IsLinearIndex(totalLength);
    }
    coreNum_ = std::min(compileInfo->totalCoreNum,
                    std::min(static_cast<uint64_t>(totalLength), static_cast<uint64_t>(indexRow)));
    coreNum_ = coreNum_ == 0 ? 1 : coreNum_;
    ubSize_ = compileInfo->ubSizePlatForm;
    GetDtypeSize();
    Tiling4LinearIndex(indexRow, indexDim_);
    uint64_t maxPhysicalOffset = 0;
    for (uint64_t i = 0; i < indexDim_; ++i) {
        maxPhysicalOffset += (varRefShape.GetDim(i) - 1) * indicesMask_[i];
    }
    uint64_t totalPhysicalRange = maxPhysicalOffset + scatterLength_;
    if (!needLargeIndexKernel_) {
        isSort_ = IsSort(totalPhysicalRange, indexRow);
    }
    SetTilingKeyMode();
    tilingContext_->SetScheduleMode(1);
    Tiling4Scatter(totalPhysicalRange, indexRow);
    size_t* currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = CalcWorkSpaceSize(indexRow);
    OP_LOGD(tilingContext_, "Tiling inited");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4ScatterNdUpdateV2(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("ScatterNdUpdateV2", "The context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "Tiling for ScatterNdUpdateV2 start.");
    ScatterNdUpdateV2Tiling tilingOp(context);
    if (tilingOp.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Tiling init fail");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "Tiling for ScatterNdUpdateV2 end.");
    return tilingOp.SetKernelTiling();
}

ge::graphStatus TilingPrepare4ScatterNdUpdateV2(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Tiling Prepare For ScatterNdUpdateV2 start.");
    auto compileInfo = context->GetCompiledInfo<ScatterNdUpdateV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->totalCoreNum == 0) {
        OP_LOGE(context, "coreNum %lu", compileInfo->totalCoreNum);
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_LOGD(context, "ubSizePlatForm is %lu.", compileInfo->ubSizePlatForm);
    OP_LOGD(context, "Tiling Prepare For ScatterNdUpdateV2 end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ScatterNdUpdateV2).Tiling(Tiling4ScatterNdUpdateV2).TilingParse<ScatterNdUpdateV2CompileInfo>(TilingPrepare4ScatterNdUpdateV2);
}
