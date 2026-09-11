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
 * \file scatter_nd_update_tiling.cpp
 * \brief scatter_nd_update arch22 tiling implementation
 */

#include "log/log.h"
#include "platform/platform_info.h"
#include "tiling/tiling_api.h"
#include "register/op_impl_registry.h"
#include "scatter_nd_update_sk_tiling.h"

namespace optiling {
constexpr uint64_t MAX_DIM_NUM = 8;
constexpr uint64_t MAX_LENGTH_INT32 = (1LL << 31) - 1;
constexpr uint64_t MAX_FLOAT_EXPRESS_INT32 = (1LL << 24) - 1;
constexpr uint64_t SORT_USE_GM_NUM = 2;
constexpr uint64_t SORT_BLOCK_LENGTH = 4096;
constexpr uint64_t GATHER_USE_NUM = 2;
constexpr uint64_t ALIGNED_NUM = 8;
constexpr uint64_t ALIGNED_SIZE = 32;
constexpr uint64_t HP_INDEX_TILE_MAX = 4096;
constexpr uint64_t HP_INDEX_TILE_ALIGN = 32;
constexpr uint64_t HP_UPDATE_UB_RATIO = 2;
constexpr uint64_t HP_DOUBLE_BUFFER = 2;
constexpr uint64_t HP_ROWS_PER_BATCH_MAX = 256;
constexpr uint64_t INDEX_TYPE_INT32 = 1;
constexpr uint64_t INDEX_TYPE_INT64 = 2;
constexpr uint64_t INDEX_TYPE_INT64_LARGE = 3;
constexpr uint64_t INT64_TO_INT_SIZE_RATIO = 2;
constexpr uint64_t TILING_KEY_BASE = 10;
constexpr uint64_t LINEAR_INDEX_COEFF_OFFSET = 3;
constexpr uint64_t DTYPE_SIZE_BF16 = 2;
constexpr uint64_t DTYPE_SIZE_FP16 = 2;
constexpr uint64_t DTYPE_SIZE_BOOL = 1;
constexpr uint64_t ATTR_STRIDE = 0;
constexpr uint64_t DIM_0 = 0;

inline void ScatterNdUpdateSkArch22Tiling::SetTilingKeyMode()
{
    // tilingKey: indexType * 10 + sortFlag (indexType: 1=int32, 2=int64(cast), 3=int64(large))
    uint64_t indexType;
    if (!isInt64Indices_) {
        indexType = INDEX_TYPE_INT32;
    } else if (needLargeIndexKernel_) {
        indexType = INDEX_TYPE_INT64_LARGE;
    } else {
        indexType = INDEX_TYPE_INT64;
    }
    uint64_t sortFlag;
    if (indexType == INDEX_TYPE_INT64_LARGE) {
        sortFlag = 0;
    } else {
        sortFlag = isSort_ ? 1 : 0;
    }
    tilingKey_ = indexType * TILING_KEY_BASE + sortFlag;

    tilingContext_->SetTilingKey(tilingKey_);
}

inline bool ScatterNdUpdateSkArch22Tiling::IsLinearIndex(uint64_t totalLength) const
{
    return totalLength <= MAX_LENGTH_INT32;
}

inline bool ScatterNdUpdateSkArch22Tiling::IsSort(uint64_t totalLength) const
{
    return totalLength <= MAX_FLOAT_EXPRESS_INT32;
}

inline void ScatterNdUpdateSkArch22Tiling::Tiling4LinearIndex(uint64_t indexRow, uint64_t indexDim)
{
    // 注意：torch_npu 环境下 aclTensor 的 originShape 来自 1-D storageDims（storage.nbytes/itemsize），
    // aclnn 层已通过 SetStorageShape(viewShape) 修正 storageShape，故此处必须读 GetStorageShape()
    // （与 scatter_nd_update_asc 一致），否则 var 会被解析成 1-D 扁平形状导致 scatterLength=1。
    auto varRefShape = tilingContext_->GetInputShape(0)->GetStorageShape();
    uint64_t strides = 1;
    for (int64_t i = indexDim - 1; i >= 0; --i) {
        indicesMask_[i] = strides;
        strides *= varRefShape.GetDim(i);
    }
    uint64_t coeff;
    if (isInt64Indices_) {
        coeff = INT64_TO_INT_SIZE_RATIO * indexDim + LINEAR_INDEX_COEFF_OFFSET;
    } else {
        coeff = indexDim + LINEAR_INDEX_COEFF_OFFSET;
    }
    uint64_t maxBlockLength = ubSize_ / coeff / sizeof(int);
    blockLength_ = (maxBlockLength / ALIGNED_SIZE) * ALIGNED_SIZE;
    blockLength_ = std::min(blockLength_, static_cast<uint64_t>(SORT_BLOCK_LENGTH));
    if (blockLength_ == 0) {
        blockLength_ = ALIGNED_SIZE;
    }
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
        tailCoreNum_ = tailBlockNum_ == 0 ? 0 : coreNum_ - frontCoreNum_;
    }
}

inline void ScatterNdUpdateSkArch22Tiling::Tiling4Scatter(uint64_t totalLength)
{
    uint64_t scatterAlignNum = ALIGNED_SIZE / dataTypeSize_;
    tailRow_ = totalLength / coreNum_;
    frontRow_ = tailRow_ + 1;
    frontNum_ = totalLength % coreNum_;
    tailNum_ = tailRow_ == 0 ? 0 : coreNum_ - frontNum_;
    ubLengthForUpdates_ = ((ubSize_ - SORT_BLOCK_LENGTH * SORT_USE_GM_NUM * sizeof(int)) / ALIGNED_SIZE *
                           ALIGNED_SIZE) /
                          dataTypeSize_;
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
}

inline uint64_t ScatterNdUpdateSkArch22Tiling::Tiling4HpScatterShape()
{
    uint64_t kMaxUpdateUbBytes = ubSize_ / HP_UPDATE_UB_RATIO;
    uint64_t fullRowBytes = scatterLength_ * dataTypeSize_;
    uint64_t updateUbBytes = 0;
    if (dataTypeSize_ == 0) {
        hpScatterTileLength_ = 1;
        hpRowBytesAligned_ = ALIGNED_SIZE;
        hpRowsPerBatch_ = 1;
        updateUbBytes = ALIGNED_SIZE;
    } else if (HP_DOUBLE_BUFFER * fullRowBytes <= kMaxUpdateUbBytes) {
        hpScatterTileLength_ = scatterLength_ == 0 ? 1 : scatterLength_;
        uint64_t rowBytes = hpScatterTileLength_ * dataTypeSize_;
        hpRowBytesAligned_ = (rowBytes + ALIGNED_SIZE - 1) & ~(ALIGNED_SIZE - 1);
        if (hpRowBytesAligned_ == 0)
            hpRowBytesAligned_ = ALIGNED_SIZE;
        uint64_t perBufBytes = kMaxUpdateUbBytes / HP_DOUBLE_BUFFER;
        hpRowsPerBatch_ = perBufBytes / hpRowBytesAligned_;
        if (hpRowsPerBatch_ > HP_ROWS_PER_BATCH_MAX)
            hpRowsPerBatch_ = HP_ROWS_PER_BATCH_MAX;
        if (hpRowsPerBatch_ == 0)
            hpRowsPerBatch_ = 1;
        updateUbBytes = HP_DOUBLE_BUFFER * hpRowsPerBatch_ * hpRowBytesAligned_;
    } else {
        updateUbBytes = (kMaxUpdateUbBytes / ALIGNED_SIZE) * ALIGNED_SIZE;
        uint64_t bufBytes = updateUbBytes / HP_DOUBLE_BUFFER;
        hpScatterTileLength_ = (bufBytes / dataTypeSize_ / ALIGNED_NUM) * ALIGNED_NUM;
        if (hpScatterTileLength_ == 0) {
            hpScatterTileLength_ = ALIGNED_NUM;
        }
        uint64_t sliceBytes = hpScatterTileLength_ * dataTypeSize_;
        hpRowBytesAligned_ = (sliceBytes + ALIGNED_SIZE - 1) & ~(ALIGNED_SIZE - 1);
        if (hpRowBytesAligned_ == 0)
            hpRowBytesAligned_ = ALIGNED_SIZE;
        hpRowsPerBatch_ = 1;
    }
    if (scatterLength_ == 0) {
        hpScatterTileNum_ = 1;
        hpScatterTileTail_ = 0;
    } else {
        hpScatterTileNum_ = (scatterLength_ + hpScatterTileLength_ - 1) / hpScatterTileLength_;
        hpScatterTileTail_ = scatterLength_ - (hpScatterTileNum_ - 1) * hpScatterTileLength_;
    }
    if (hpScatterTileNum_ > 1) {
        hpRowsPerBatch_ = 1;
    }
    return updateUbBytes;
}

inline void ScatterNdUpdateSkArch22Tiling::Tiling4HpIndexTile(uint64_t updateUbBytes)
{
    uint64_t ubForIndex = (ubSize_ > updateUbBytes) ? (ubSize_ - updateUbBytes) : 0;
    uint64_t coeff;
    if (isInt64Indices_) {
        coeff = INT64_TO_INT_SIZE_RATIO * indexDim_ + LINEAR_INDEX_COEFF_OFFSET;
    } else {
        coeff = indexDim_ + LINEAR_INDEX_COEFF_OFFSET;
    }
    if (coeff == 0) {
        coeff = 1;
    }
    uint64_t maxIndexTile = ubForIndex / coeff / sizeof(int);
    hpIndexTileLength_ = (maxIndexTile / HP_INDEX_TILE_ALIGN) * HP_INDEX_TILE_ALIGN;
    if (hpIndexTileLength_ > HP_INDEX_TILE_MAX) {
        hpIndexTileLength_ = HP_INDEX_TILE_MAX;
    }
    if (hpIndexTileLength_ == 0) {
        hpIndexTileLength_ = HP_INDEX_TILE_ALIGN;
    }
}

inline void ScatterNdUpdateSkArch22Tiling::Tiling4HpCorePartition(uint64_t indexRow)
{
    hpCoreNum_ = std::min(coreNum_, indexRow);
    if (hpCoreNum_ == 0) {
        hpCoreNum_ = 1;
    }
    hpTailIndexNum_ = indexRow / hpCoreNum_;
    hpFrontIndexNum_ = hpTailIndexNum_ + 1;
    hpFrontCoreNum_ = indexRow % hpCoreNum_;
    hpTailCoreNum_ = (hpTailIndexNum_ == 0) ? 0 : (hpCoreNum_ - hpFrontCoreNum_);
}

inline void ScatterNdUpdateSkArch22Tiling::Tiling4Hp(uint64_t indexRow)
{
    uint64_t updateUbBytes = Tiling4HpScatterShape();
    Tiling4HpIndexTile(updateUbBytes);
    Tiling4HpCorePartition(indexRow);
}

inline ge::graphStatus ScatterNdUpdateSkArch22Tiling::HandleViewStride()
{
    firstDimStrideRows_ = (indexDim_ > 1) ? indicesMask_[0] : 1;
    uint64_t stride0Expected = scatterLength_ * firstDimStrideRows_;

    // 默认按连续处理
    isViewStride0_ = 0;
    varStride0Elements_ = stride0Expected;

    // 不依赖 GetInputStride：torch_npu 环境下 GE 不会把非连续切片的 stride 数组下传，
    // GetInputStride 返回的 strideDimNum 为 0。故改为由 torch 侧显式传入 var.strides()
    // 属性，tiling 从属性中读取真实 stride。
    auto attrs = tilingContext_->GetAttrs();
    auto stridesPtr = attrs->GetListInt(ATTR_STRIDE);
    auto varShape = tilingContext_->GetInputShape(0)->GetStorageShape();
    uint64_t varDimNum = varShape.GetDimNum();
    if (stridesPtr != nullptr && stridesPtr->GetSize() == varDimNum) {
        int64_t stride0 = stridesPtr->GetData()[DIM_0];

        // 除被索引的前 indexDim 维外，其余轴必须连续（kernel 按整块展平拷贝，
        // scatterLength = 后 (varDimNum_-indexDim_) 维乘积）。只有首轴可非连续。
        // 对 var [d0, d1, ..., dk]，dim i (i>=indexDim_) 的连续期望 stride 为 var 第 i 维之后各维的乘积。
        for (int64_t i = indexDim_; i < static_cast<int64_t>(varDimNum); ++i) {
            int64_t expectedStride = 1;
            for (int64_t j = i + 1; j < static_cast<int64_t>(varDimNum); ++j) {
                expectedStride *= varShape.GetDim(j);
            }
            if (varShape.GetDim(i) > 1 && stridesPtr->GetData()[i] != expectedStride) {
                OP_LOGE(tilingContext_->GetNodeName(),
                        "var dim%ld must be contiguous, but got stride %ld, expected %ld. Only dim0 may be non-contiguous.",
                        i, stridesPtr->GetData()[i], expectedStride);
                return ge::GRAPH_FAILED;
            }
        }

        // 仅当实际首轴 stride 严格大于连续期望时才判定为 view
        if (static_cast<uint64_t>(stride0) > stride0Expected) {
            isViewStride0_ = 1;
            varStride0Elements_ = static_cast<uint64_t>(stride0);
        }
    }
    return ge::GRAPH_SUCCESS;
}

inline void ScatterNdUpdateSkArch22Tiling::GetDtypeSize()
{
    uint64_t varDtype = tilingContext_->GetInputDesc(0)->GetDataType();
    switch (varDtype) {
        case ge::DT_FLOAT:
            dataTypeSize_ = sizeof(float);
            break;
        case ge::DT_BF16:
            dataTypeSize_ = DTYPE_SIZE_BF16;
            break;
        case ge::DT_FLOAT16:
            dataTypeSize_ = DTYPE_SIZE_FP16;
            break;
        case ge::DT_BOOL:
            dataTypeSize_ = DTYPE_SIZE_BOOL;
            break;
        case ge::DT_INT64:
            dataTypeSize_ = sizeof(int64_t);
            break;
        case ge::DT_INT32:
            dataTypeSize_ = sizeof(int32_t);
            break;
        case ge::DT_INT16:
            dataTypeSize_ = sizeof(int16_t);
            break;
        case ge::DT_INT8:
            dataTypeSize_ = sizeof(int8_t);
            break;
        default:
            break;
    }
}

ge::graphStatus ScatterNdUpdateSkArch22Tiling::SetKernelTiling()
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
    tilingData_.viewTiling.set_isViewStride0(isViewStride0_);
    tilingData_.viewTiling.set_varStride0Elements(varStride0Elements_);
    tilingData_.viewTiling.set_firstDimStrideRows(firstDimStrideRows_);
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
    tilingData_.hpTiling.set_hpCoreNum(hpCoreNum_);
    tilingData_.hpTiling.set_hpFrontIndexNum(hpFrontIndexNum_);
    tilingData_.hpTiling.set_hpTailIndexNum(hpTailIndexNum_);
    tilingData_.hpTiling.set_hpFrontCoreNum(hpFrontCoreNum_);
    tilingData_.hpTiling.set_hpTailCoreNum(hpTailCoreNum_);
    tilingData_.hpTiling.set_hpIndexTileLength(hpIndexTileLength_);
    tilingData_.hpTiling.set_hpScatterTileLength(hpScatterTileLength_);
    tilingData_.hpTiling.set_hpScatterTileNum(hpScatterTileNum_);
    tilingData_.hpTiling.set_hpScatterTileTail(hpScatterTileTail_);
    tilingData_.hpTiling.set_hpRowBytesAligned(hpRowBytesAligned_);
    tilingData_.hpTiling.set_hpRowsPerBatch(hpRowsPerBatch_);
    tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                             tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

inline size_t ScatterNdUpdateSkArch22Tiling::CalcWorkSpaceSize(uint64_t indexRow)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t indexRowAligned = (indexRow + ALIGNED_NUM - 1) & ~(ALIGNED_NUM - 1);
    sortWorkspace_ = indexRowAligned;
    size_t totalWorkspace = sysWorkspaceSize;
    if (isLinearIndex_) {
        totalWorkspace += sortWorkspace_ * sizeof(int);
    }
    if (isSort_) {
        totalWorkspace += sortWorkspace_ * sizeof(int);
    }
    return totalWorkspace;
}

static ge::graphStatus TilingParseForScatterNdUpdateSkArch22(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ScatterNdUpdateSkArch22CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->vectorCoreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF((compileInfo->ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void ScatterNdUpdateSkArch22Tiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext_, "coreNum:                   %lu", coreNum_);
    OP_LOGD(tilingContext_, "ubSize:                    %lu", ubSize_);
    OP_LOGD(tilingContext_, "tilingKey:                 %lu", tilingKey_);
    OP_LOGD(tilingContext_, "isInt64Indices:            %d", isInt64Indices_);
    OP_LOGD(tilingContext_, "needLargeIndexKernel:      %d", needLargeIndexKernel_);
    OP_LOGD(tilingContext_, "isLinearIndex:             %d", isLinearIndex_);
    OP_LOGD(tilingContext_, "isSort:                    %d", isSort_);

    OP_LOGD(tilingContext_, "isViewStride0:             %lu", isViewStride0_);
    OP_LOGD(tilingContext_, "varStride0Elements:        %lu", varStride0Elements_);
    OP_LOGD(tilingContext_, "firstDimStrideRows:        %lu", firstDimStrideRows_);

    OP_LOGD(tilingContext_, "indexDim:                  %lu", indexDim_);
    OP_LOGD(tilingContext_, "blockLength:               %lu", blockLength_);
    OP_LOGD(tilingContext_, "blockNum:                  %lu", blockNum_);
    OP_LOGD(tilingContext_, "blockRemainLength:         %lu", blockRemainLength_);
    OP_LOGD(tilingContext_, "frontBlockNum:             %lu", frontBlockNum_);
    OP_LOGD(tilingContext_, "tailBlockNum:              %lu", tailBlockNum_);
    OP_LOGD(tilingContext_, "frontCoreNum:              %lu", frontCoreNum_);
    OP_LOGD(tilingContext_, "tailCoreNum:               %lu", tailCoreNum_);
    OP_LOGD(tilingContext_, "sortWorkspace:             %lu", sortWorkspace_);
    for (size_t i = 0; i < indexDim_; i++) {
        OP_LOGD(tilingContext_, "indicesMask[%zu]:            %lu", i, indicesMask_[i]);
    }

    OP_LOGD(tilingContext_, "scatterLength:             %lu", scatterLength_);
    OP_LOGD(tilingContext_, "frontRow:                  %lu", frontRow_);
    OP_LOGD(tilingContext_, "tailRow:                   %lu", tailRow_);
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

    OP_LOGD(tilingContext_, "hpCoreNum:                 %lu", hpCoreNum_);
    OP_LOGD(tilingContext_, "hpFrontIndexNum:           %lu", hpFrontIndexNum_);
    OP_LOGD(tilingContext_, "hpTailIndexNum:            %lu", hpTailIndexNum_);
    OP_LOGD(tilingContext_, "hpFrontCoreNum:            %lu", hpFrontCoreNum_);
    OP_LOGD(tilingContext_, "hpTailCoreNum:             %lu", hpTailCoreNum_);
    OP_LOGD(tilingContext_, "hpIndexTileLength:         %lu", hpIndexTileLength_);
    OP_LOGD(tilingContext_, "hpScatterTileLength:       %lu", hpScatterTileLength_);
    OP_LOGD(tilingContext_, "hpScatterTileNum:          %lu", hpScatterTileNum_);
    OP_LOGD(tilingContext_, "hpScatterTileTail:         %lu", hpScatterTileTail_);
    OP_LOGD(tilingContext_, "hpRowBytesAligned:         %lu", hpRowBytesAligned_);
    OP_LOGD(tilingContext_, "hpRowsPerBatch:            %lu", hpRowsPerBatch_);
}

namespace {
struct ScatterInitInfo {
    uint64_t totalLength;
    uint64_t indexRow;
    uint64_t indexDim;
    uint64_t isInt64Indices;
    uint64_t scatterLength;
};
} // namespace

static ScatterInitInfo ParseScatterShapes(const gert::TilingContext* ctx, uint64_t scatterLengthInit)
{
    ScatterInitInfo info{};
    info.scatterLength = scatterLengthInit;
    // torch_npu 环境下 originShape 为 1-D storageDims，必须读 GetStorageShape()（aclnn 已 SetStorageShape 修正）
    auto varRefShape = ctx->GetInputShape(0)->GetStorageShape();
    auto indicesShape = ctx->GetInputShape(1)->GetStorageShape();
    uint64_t varDimNum = varRefShape.GetDimNum();
    info.indexDim = indicesShape.GetDim(indicesShape.GetDimNum() - 1);
    auto indicesDtype = ctx->GetInputDesc(1)->GetDataType();
    info.isInt64Indices = (indicesDtype == ge::DT_INT64);
    info.totalLength = 1;
    for (uint64_t i = 0; i < info.indexDim; ++i) {
        info.totalLength *= varRefShape.GetDim(i);
    }
    if (varDimNum > info.indexDim) {
        for (uint64_t i = info.indexDim; i < varDimNum; i++) {
            info.scatterLength *= varRefShape.GetDim(i);
        }
    }
    info.indexRow = 1;
    for (uint64_t i = 0; i < indicesShape.GetDimNum() - 1; i++) {
        info.indexRow *= indicesShape.GetDim(i);
    }
    return info;
}

ge::graphStatus ScatterNdUpdateSkArch22Tiling::Init()
{
    auto info = ParseScatterShapes(tilingContext_, scatterLength_);
    indexDim_ = info.indexDim;
    isInt64Indices_ = info.isInt64Indices;
    scatterLength_ = info.scatterLength;
    if (isInt64Indices_) {
        needLargeIndexKernel_ = !IsLinearIndex(info.totalLength);
    }
    if (needLargeIndexKernel_) {
        isSort_ = false;
        isLinearIndex_ = false;
    } else {
        isSort_ = IsSort(info.totalLength);
        isLinearIndex_ = IsLinearIndex(info.totalLength);
    }
    auto compileInfo = tilingContext_->GetCompileInfo<ScatterNdUpdateSkArch22CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, compileInfo);
    coreNum_ = std::min(static_cast<uint64_t>(compileInfo->vectorCoreNum), std::min(info.totalLength, info.indexRow));
    coreNum_ = coreNum_ == 0 ? 1 : coreNum_;

    // Align coreNum_ to even: SyncAll-based kernels require an even number of
    // cores per block (see https://gitcode.com/cann/ops-nn/pull/9645).
    if (coreNum_ % 2 != 0) {
        coreNum_ += 1;
    }
    ubSize_ = compileInfo->ubSize;
    GetDtypeSize();
    SetTilingKeyMode();
    tilingContext_->SetScheduleMode(1);
    Tiling4LinearIndex(info.indexRow, indexDim_);
    Tiling4Scatter(info.totalLength);
    Tiling4Hp(info.indexRow);
    if (HandleViewStride() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    size_t* currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = CalcWorkSpaceSize(info.indexRow);
    return ge::GRAPH_SUCCESS;
}

// tiling dispatch entry
static ge::graphStatus ScatterNdUpdateSkArch22TilingFunc(gert::TilingContext* context)
{
    ScatterNdUpdateSkArch22Tiling tilingOp(context);
    if (tilingOp.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling init fail");
        return ge::GRAPH_FAILED;
    }
    return tilingOp.SetKernelTiling();
}

IMPL_OP_OPTILING(ScatterNdUpdateSk)
    .Tiling(ScatterNdUpdateSkArch22TilingFunc)
    .TilingParse<ScatterNdUpdateSkArch22CompileInfo>(TilingParseForScatterNdUpdateSkArch22);

} // namespace optiling
