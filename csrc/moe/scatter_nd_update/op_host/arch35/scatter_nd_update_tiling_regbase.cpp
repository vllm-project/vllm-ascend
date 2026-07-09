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
 * \file scatter_nd_update_tiling.cc
 * \brief ascendc scatter ND update tiling cpp
 */

#include "scatter_nd_update_tiling_regbase.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_common/op_host/util/math_util.h"
#include "error_util.h"

using namespace AscendC;
namespace optiling {

static constexpr uint16_t INPUT_IDX_VAR = 0;
static constexpr uint16_t INPUT_IDX_INDICES = 1;
static constexpr uint16_t INPUT_IDX_UPDATES = 2;
static constexpr uint16_t OUTPUT_IDX_SHAPE = 0;
static constexpr uint16_t RANK_MIN_VALUE = 1;
static constexpr uint16_t RANK_MAX_VALUE = 7;
static constexpr uint16_t STRIDE_MAX_VALUE = 8;
static constexpr uint64_t MIN_TILING_SIZE = 128;
static constexpr uint32_t DCACHE_SIZE = 32U * 1024U;
static constexpr uint32_t RESERVED_WORKSPACE_SIZE = 16U * 1024U * 1024U;
static constexpr uint32_t INPUT_ADDRESS_IN_INT32 = 100;
static constexpr uint32_t INPUT_ADDRESS_IN_INT64 = 200;
static constexpr uint32_t THREE = 3;
static constexpr uint32_t SIMT_SORT_USED_QUENUM = 5;

static constexpr uint64_t DB_BUFFER = 2;
static constexpr uint64_t RESERVE_SIZE = 256;
static constexpr int64_t ALIGN_SIZE = 32;
static constexpr int64_t MIN_HANDLE_SIZE = 128;
static constexpr int64_t MIN_SIZE_SIMD_NONDETERMINISTIC = 128;
static constexpr int64_t INDICES_MIN_BLOCK_SIZE = 1024;
static constexpr int64_t INT32_BYTES = 4;
static constexpr int64_t FP32_BYTES = 4;
static constexpr int64_t SIMT_SORT_LIMIT = 3;
static constexpr int64_t TWO = 2;
static constexpr int64_t MASK_CORE = 1000;
static constexpr int64_t MASK_VAR = 5;
static constexpr int64_t MASK_AFTER = 19;
static constexpr int64_t ONE = 1;
static constexpr int64_t ROW_THRESH_SIZE = 4096;
static constexpr float PARTIAL_UB = 0.1;
static constexpr int64_t MIN_THREAD_NUM = 128;
static constexpr int64_t MIN_SIZE_SIMD_DETERMINISTIC = 128;
static constexpr int64_t MIN_INDICES_PER_CORE_FOR_SIMD_SORT = 64;

static const std::set<ge::DataType> DETERMIN_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16};

static const gert::Shape g_vec_1_shape = {1};

static const gert::Shape& EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}

ge::graphStatus ScatterNdUpdateTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((aivNum <= 0), OP_LOGE(opName, "ScatterNdUpdateTiling fail to get totalCoreNum_."),
                return ge::GRAPH_FAILED);
    totalCoreNum_ = aivNum;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    OP_CHECK_IF((ubSizePlatForm <= DCACHE_SIZE), OP_LOGE(opName, "ub size less than Dcache Size. please check"),
                return ge::GRAPH_FAILED);
    // UB Size Need reserve space for Dcache / CCEC Compile Stack.
    ubSize_ = ubSizePlatForm - DCACHE_SIZE;
    auto res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(opName, "SetLocalMemorySize ubSize = %ld failed.", ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::ValidateVarInfo(const gert::Tensor* var, const gert::Shape& varOriginShape)
{
    // 获取并记录 var 的形状信息
    shapeRank_ = varOriginShape.GetDimNum();

    // 计算基于 origin shape 的 varShapeSize
    for (int64_t i = 0; i < shapeRank_; i++) {
        outputShapeSize *= varOriginShape.GetDim(i);
    }
    if (outputShapeSize <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, "var", std::to_string(outputShapeSize).c_str(),
                                                  "varShapeSize must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    std::ostringstream originShapeStr;
    for (int64_t i = 0; i < shapeRank_; i++) {
        originShapeStr << varOriginShape.GetDim(i);
        if (i < shapeRank_ - 1)
            originShapeStr << ", ";
    }
    OP_LOGI(opName, "Input var origin shape: (%s), dims: %lld", originShapeStr.str().c_str(), shapeRank_);

    auto varDesc = context_->GetInputDesc(INPUT_IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varDesc);
    varDtype_ = varDesc->GetDataType();
    varTypeSize_ = ge::GetSizeByDataType(varDtype_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::ValidateIndicesInfo(const gert::Tensor* indices, gert::Shape& indiceShape,
                                                           int64_t& indiceDims)
{
    indiceShapeSize = indices->GetShapeSize();
    if (indiceShapeSize < 0UL) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, "indices", std::to_string(indiceShapeSize).c_str(),
                                                  "indices shapeSize cannot be negative");
        return ge::GRAPH_FAILED;
    }
    auto indicesDesc = context_->GetInputDesc(INPUT_IDX_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);
    indiceDtype_ = indicesDesc->GetDataType();
    indicesTypeSize_ = ge::GetSizeByDataType(indiceDtype_);

    indiceShape = indices->GetStorageShape();
    indiceDims = indiceShape.GetDimNum();
    rankSize_ = indiceShape.GetDim(indiceDims - 1);
    if (indiceDims < TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, "indices", std::to_string(indiceDims).c_str(),
                                                 "The number of dimensions must be >= 2");
        return ge::GRAPH_FAILED;
    }

    if (RANK_MIN_VALUE > static_cast<uint16_t>(rankSize_) || static_cast<uint16_t>(rankSize_) > RANK_MAX_VALUE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "rankSize", std::to_string(rankSize_).c_str(),
                                              "rankSize must be in the range [1, 7]");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::HandleNonContiguousCase(const gert::Tensor* var,
                                                               const gert::Shape& varOriginShape)
{
    auto varViewStride = context_->GetInputStride(INPUT_IDX_VAR);
    constexpr uint16_t MAX_INDICES_RANK_FOR_VIEW = 4;
    if (static_cast<uint16_t>(rankSize_) > MAX_INDICES_RANK_FOR_VIEW) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "rankSize", std::to_string(rankSize_).c_str(),
                                              "In non-contiguous scenarios, rankSize must be <= 4.");
        return ge::GRAPH_FAILED;
    }

    // 检查非索引轴是否连续：var 的 [rankSize_, shapeRank-1] 维度范围应连续
    bool nonIndexAxesContiguous = true;
    if (rankSize_ < static_cast<int64_t>(shapeRank_)) {
        int64_t expectedStride = 1;
        for (int64_t dim = static_cast<int64_t>(shapeRank_) - 1; dim >= static_cast<int64_t>(rankSize_); --dim) {
            int64_t dimSize = varOriginShape.GetDim(dim);
            int64_t actualStride = varViewStride->GetStride(dim);
            if (dimSize > 1 && actualStride != expectedStride) {
                nonIndexAxesContiguous = false;
                break;
            }
            expectedStride *= dimSize;
        }
    }

    if (!nonIndexAxesContiguous) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "var", "non-contiguous strides",
                                              "non-indexed axis strides must be contiguous");
        return ge::GRAPH_FAILED;
    }

    IsContiguous_ = 0; // 非连续内存

    // 获取 var 的 storage shape，将第一维赋值给 outputStorageShapeSize_
    auto varStorageShape = var->GetStorageShape();
    int64_t storageShapeRank = varStorageShape.GetDimNum();
    if (storageShapeRank > 0) {
        outputStorageShapeSize_ = varStorageShape.GetDim(0);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::HandleContiguousCase()
{
    IsContiguous_ = 1; // 连续内存
    outputStorageShapeSize_ = outputShapeSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::ValidateUpdatesInfo(const gert::Tensor* updates, gert::Shape& updateShape)
{
    updateShapeSize = updates->GetShapeSize();
    if (updateShapeSize < 0UL) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, "updates", std::to_string(updateShapeSize).c_str(),
                                                  "updates shapeSize must not be negative.");
        return ge::GRAPH_FAILED;
    }

    auto updateDesc = context_->GetInputDesc(INPUT_IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updateDesc);
    updateShape = updates->GetStorageShape();
    updateDtype_ = updateDesc->GetDataType();
    if (updateDtype_ != varDtype_) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, "updates, var", Ops::Base::ToString(updateDtype_).c_str(),
                                               "updates and var must have the same dtype");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::CalculateDerivedParams(const gert::Shape& varOriginShape,
                                                              gert::Shape& indiceShape, gert::Shape& updateShape)
{
    if (shapeRank_ < rankSize_) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, "var", std::to_string(shapeRank_).c_str(),
                                                 "var dimension count must be >= rankSize.");
        return ge::GRAPH_FAILED;
    }

    for (int64_t idx = 0; idx < shapeRank_; idx++) {
        outPutShape[idx] = varOriginShape.GetDim(idx);
        OP_LOGI(opName, "outPutShape[%lld] = %lld", idx, outPutShape[idx]);
    }

    OP_LOGI(opName, "After outPutShape calculation, shapeRank_: %lld, outputShapeSize: %llu", shapeRank_,
            outputShapeSize);

    if (indiceShapeSize == 0UL || updateShapeSize == 0UL) {
        return ge::GRAPH_SUCCESS;
    }

    if (CheckScatterNdUpdateTensorShape(indiceShape, updateShape, varOriginShape)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, "updates, output", "updates_shape, output_shape",
                                               "The trailing dimension counts of updateRank and outputRank must match");
        return ge::GRAPH_FAILED;
    }

    // indicesAxis_ equal updatesInAxis
    indicesAxis_ = static_cast<int64_t>(indiceShapeSize / rankSize_);
    afterAxis_ = static_cast<int64_t>(updateShapeSize) / indicesAxis_;
    varInAxis_ = outputShapeSize / afterAxis_;
    varStorageInAxis_ = outputStorageShapeSize_ / afterAxis_;
    sliceSize = static_cast<uint64_t>(afterAxis_);

    if (context_->GetDeterministic() == 1 && indicesAxis_ > 1) {
        isDeterministic_ = 1;
        context_->SetScheduleMode(1);
    }
    if (isDeterministic_ != 1 && afterAxis_ * varTypeSize_ >= MIN_SIZE_SIMD_NONDETERMINISTIC) {
        isSimdNonDeterministic_ = 1;
    }

    // SIMD 排序条件
    // 1. indicesAxis_ > varInAxis_：索引数量大于原始索引数量，表示高重复度
    // 2. 单核 indices 数量 > MIN_INDICES_PER_CORE_FOR_SIMD_SORT(64)：批次足够大
    int64_t estimatedIndicesPerCore = Ops::Base::CeilDiv(indicesAxis_, totalCoreNum_);
    bool highDuplication = (indicesAxis_ > varInAxis_);
    bool enoughBatchPerCore = (estimatedIndicesPerCore > MIN_INDICES_PER_CORE_FOR_SIMD_SORT);
    if (isSimdNonDeterministic_ == 1 && highDuplication && enoughBatchPerCore) {
        isSimdWithSort_ = 1;
    }

    if (indicesAxis_ / varInAxis_ >= SIMT_SORT_LIMIT) {
        isSimtWithSort_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::GetShapeAttrsInfo()
{
    const gert::Tensor* var = context_->GetInputTensor(INPUT_IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, var);

    // 获取并记录 var 的形状信息
    auto varShape = context_->GetInputShape(INPUT_IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varShape);
    const auto& varOriginShape = EnsureNotScalar(varShape->GetOriginShape());

    // 验证和获取 var 的基本信息
    ge::graphStatus status = ValidateVarInfo(var, varOriginShape);
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, OP_LOGE(opName, "ValidateVarInfo failed"), return status);

    const gert::Tensor* indices = context_->GetInputTensor(INPUT_IDX_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indices);
    gert::Shape indiceShape;
    int64_t indiceDims = 0;

    // 验证和获取 indices 的基本信息
    status = ValidateIndicesInfo(indices, indiceShape, indiceDims);
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, OP_LOGE(opName, "ValidateIndicesInfo failed"), return status);

    // 处理内存连续性
    if (context_->InputIsView(INPUT_IDX_VAR)) {
        auto varViewStride = context_->GetInputStride(INPUT_IDX_VAR);
        if (varViewStride != nullptr && varViewStride->GetDimNum() != 0) {
            status = HandleNonContiguousCase(var, varOriginShape);
            OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, OP_LOGE(opName, "HandleNonContiguousCase failed"),
                            return status);
        } else {
            status = HandleContiguousCase();
            OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, OP_LOGE(opName, "HandleContiguousCase failed"), return status);
        }
    } else {
        status = HandleContiguousCase();
        OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, OP_LOGE(opName, "HandleContiguousCase failed"), return status);
    }

    const gert::Tensor* updates = context_->GetInputTensor(INPUT_IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updates);
    gert::Shape updateShape;

    // 验证和获取 updates 的基本信息
    status = ValidateUpdatesInfo(updates, updateShape);
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, OP_LOGE(opName, "ValidateUpdatesInfo failed"), return status);

    // 计算派生参数
    status = CalculateDerivedParams(varOriginShape, indiceShape, updateShape);
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, OP_LOGE(opName, "CalculateDerivedParams failed"), return status);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::CheckScatterNdUpdateTensorShape(const gert::Shape& indiceShape,
                                                                       const gert::Shape& updateShape,
                                                                       const gert::Shape& outputShape)
{
    int64_t indiceDims = indiceShape.GetDimNum();
    int64_t updateDims = updateShape.GetDimNum();
    int64_t outputDims = outputShape.GetDimNum();

    int64_t outputAxisDims = outputDims - static_cast<int64_t>(rankSize_);
    int64_t updateAxisDims = updateDims - (indiceDims - 1);
    if (outputAxisDims != updateAxisDims) {
        return ge::GRAPH_FAILED;
    }

    for (int64_t idx = 0; idx < outputAxisDims; idx++) {
        int64_t updateDim = updateShape.GetDim(idx + indiceDims - 1);
        int64_t outputDim = outputShape.GetDim(idx + rankSize_);
        if (updateDim != outputDim) {
            return ge::GRAPH_FAILED;
        }
    }

    for (int64_t idx = 0; idx < indiceDims - 1; idx++) {
        int64_t updateDim = updateShape.GetDim(idx);
        int64_t indiceDim = indiceShape.GetDim(idx);
        if (indiceDim != updateDim) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void ScatterNdUpdateTiling::BlockTiling()
{
    auto typeSize = ge::GetSizeByDataType(updateDtype_);
    OP_CHECK_IF(typeSize == 0, OP_LOGE(opName, "typeSize is 0"), return );
    alignFactor = Ops::Base::GetUbBlockSize(context_) / typeSize;
    auto blockFactor = Ops::Base::CeilDiv(updateShapeSize, static_cast<uint64_t>(totalCoreNum_));
    auto blockAlignFactor = Ops::Base::CeilDiv(blockFactor, alignFactor) * alignFactor;
    blockTilingSize = std::max(static_cast<uint64_t>(blockAlignFactor), MIN_TILING_SIZE);
    blockNum = Ops::Base::CeilDiv(updateShapeSize, blockTilingSize);
    tailBlockTilingSize = updateShapeSize - blockTilingSize * (blockNum - 1UL);
    OP_LOGD(opName,
            "updateShapeSize = %lld, blockFactor = %lld, blockAlignFactor = %lld,"
            "blockTilingSize = %d, tailBlockTilingSize = %d",
            updateShapeSize, blockFactor, blockAlignFactor, blockTilingSize, tailBlockTilingSize);
}

ge::graphStatus ScatterNdUpdateTiling::UbTiling()
{
    if (indiceShapeSize == 0UL || updateShapeSize == 0UL) {
        return ge::GRAPH_SUCCESS;
    }
    // halfUbSize for double buffer
    auto halfUbSize = ubSize_ / DB_BUFFER;
    auto indiceNum = indiceShapeSize / rankSize_;
    sliceSize = updateShapeSize / indiceNum;
    OP_CHECK_IF(sliceSize == static_cast<uint64_t>(0),
                OP_LOGE(opName, "sliceSize %lu is zero. please check.", sliceSize), return ge::GRAPH_FAILED);
    auto updateTypeSize = ge::GetSizeByDataType(updateDtype_);
    indiceDtype_ = context_->GetInputDesc(INPUT_IDX_INDICES)->GetDataType();
    auto indiceTypeSize = ge::GetSizeByDataType(indiceDtype_);
    // sliceUb : the required size of UB for one scatter operation;
    auto sliceUb = sliceSize * updateTypeSize + rankSize_ * indiceTypeSize;
    sliceUb = Ops::Base::CeilDiv(static_cast<uint64_t>(sliceUb), alignFactor) * alignFactor;
    OP_CHECK_IF(updateTypeSize == 0, OP_LOGE(opName, "updateTypeSize is 0"), return ge::GRAPH_FAILED);
    if (sliceUb > halfUbSize) {
        // for scatter operator. At least  rank size index need to be move in UB.
        ubTilingSize = (halfUbSize - rankSize_ * indiceTypeSize) / updateTypeSize;
    } else {
        // calculate the size of updates that need to be move in UB
        auto maxIndiceCnt = halfUbSize / sliceUb;
        ubTilingSize = maxIndiceCnt * sliceSize;
    }
    OP_LOGD(opName, "sliceUb = %lu, halfUbSize = %u, ubTilingSize = %u", sliceUb, halfUbSize, ubTilingSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::SortTiling()
{
    if (indiceShapeSize == static_cast<uint64_t>(0) || updateShapeSize == static_cast<uint64_t>(0)) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(sliceSize == static_cast<uint64_t>(0),
                OP_LOGE(opName, "sliceSize %lu is zero. please check.", sliceSize), return ge::GRAPH_FAILED);
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context_);

    // 分核策略：每个核平分行数
    uint64_t rows = indiceShapeSize / rankSize_;
    int64_t start = 1;
    int64_t end = static_cast<int64_t>(rows) + 1;
    int64_t mid = 0;
    int64_t sortTmpSize = 0;
    while (end - start > 1) {
        mid = (end + start) / TWO;
        int64_t totalIndexSize = Ops::Base::CeilAlign(mid * rankSize_ * indicesTypeSize_, ubBlockSize) + // indice
                                 Ops::Base::CeilAlign(mid * outOfSetTypeSize_, ubBlockSize) +            // outOfsetBuf
                                 Ops::Base::CeilAlign(mid * outOfSetTypeSize_, ubBlockSize) +
                                 TWO * ubBlockSize +                                               // sortIndiceBuf
                                 Ops::Base::CeilAlign(mid * indicesTypeSize_, ubBlockSize) +       // updateOrigin
                                 Ops::Base::CeilAlign((mid + 1) * indicesTypeSize_, ubBlockSize) + // uniqeIdCount
                                 Ops::Base::CeilAlign(STRIDE_MAX_VALUE * indicesTypeSize_, ubBlockSize) + // strideBuf
                                 MIN_HANDLE_SIZE * FP32_BYTES;                                            // maxScore
        sortTmpSize = GetSortTmpSize(outOfSetDtype_, mid, false);
        sortTmpSize = Ops::Base::CeilAlign(sortTmpSize, ubBlockSize);
        int64_t tmpToTalSize = totalIndexSize + sortTmpSize + static_cast<int64_t>(MIN_TILING_SIZE);
        if (tmpToTalSize <= static_cast<int64_t>(ubSize_)) {
            start = mid;
        } else {
            end = mid;
        }
    }

    ubTilingSize = static_cast<uint32_t>(start);
    uint64_t totalLoop = Ops::Base::CeilDiv(rows, static_cast<uint64_t>(ubTilingSize));
    uint64_t eachCoreLoop = Ops::Base::CeilDiv(totalLoop, static_cast<uint64_t>(totalCoreNum_));
    blockNum = Ops::Base::CeilDiv(totalLoop, eachCoreLoop);

    while (blockNum < static_cast<uint64_t>(totalCoreNum_ / TWO) && ubTilingSize > static_cast<uint32_t>(1)) {
        ubTilingSize = ubTilingSize / static_cast<uint32_t>(TWO);
        totalLoop = Ops::Base::CeilDiv(rows, static_cast<uint64_t>(ubTilingSize));
        eachCoreLoop = Ops::Base::CeilDiv(totalLoop, static_cast<uint64_t>(totalCoreNum_));
        blockNum = Ops::Base::CeilDiv(totalLoop, eachCoreLoop);
    }
    blockTilingSize = eachCoreLoop * ubTilingSize;
    tailBlockTilingSize = rows - blockTilingSize * (blockNum - 1UL);
    OP_LOGD(opName,
            "rows = %lld, blockTilingSize = %lld, tailBlockTilingSize = %lld,"
            "blockNum = %d ,eachCoreLoop = %d ,",
            rows, blockTilingSize, tailBlockTilingSize, blockNum, eachCoreLoop);
    return ge::GRAPH_SUCCESS;
}

uint32_t ScatterNdUpdateTiling::GetSortTmpSize(ge::DataType dataType, uint32_t lastAxisNum, bool isDescend)
{
    std::vector<int64_t> shapeVec = {lastAxisNum};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = isDescend;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, dataType, ge::DT_UINT32, false, config, maxValue, minValue);

    return maxValue;
}

int64_t ScatterNdUpdateTiling::GetRestAvailableSize(int64_t sampleNum, int64_t valueTypeBytes, int64_t originalSize,
                                                    int64_t postAxisSize, ge::DataType idType)
{
    int64_t ubBlock = Ops::Base::GetUbBlockSize(context_);
    int64_t occupy = Ops::Base::CeilAlign(sampleNum * rankSize_ * indicesTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign(sampleNum * outOfSetTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign(sampleNum * (outOfSetTypeSize_ + TWO * ALIGN_SIZE), ubBlock) +
                     Ops::Base::CeilAlign(sampleNum * INT32_BYTES, ubBlock) +
                     Ops::Base::CeilAlign(sampleNum * (INT32_BYTES * TWO), ubBlock) +
                     Ops::Base::CeilAlign(sampleNum * indicesTypeSize_, ubBlock) +
                     sampleNum * Ops::Base::CeilAlign((varTypeSize_)*postAxisSize, ubBlock) +
                     sampleNum * Ops::Base::CeilAlign((FP32_BYTES)*postAxisSize, ubBlock) +
                     sampleNum * Ops::Base::CeilAlign((FP32_BYTES)*postAxisSize, ubBlock) +
                     GetSortTmpSize(idType, sampleNum, false);
    return originalSize - occupy;
}

void ScatterNdUpdateTiling::ComputeCoreSplitAfterAxis()
{
    eachCoreAfterAxisCount_ = Ops::Base::CeilDiv(afterAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(afterAxis_, eachCoreAfterAxisCount_);
    tailCoreAfterAxisCount_ = afterAxis_ - eachCoreAfterAxisCount_ * (usedCoreNumBefore_ - 1);
}

void ScatterNdUpdateTiling::InitFactors(int64_t halfUbSize, int64_t indicesSize, int64_t alignNum)
{
    afterAxisFactor_ = Ops::Base::CeilAlign(eachCoreAfterAxisCount_, alignNum);
    indicesFactor_ = halfUbSize / (afterAxisFactor_ * (varTypeSize_ + FP32_BYTES) + indicesSize);
}

void ScatterNdUpdateTiling::HandleIndicesFactorGtOne(int64_t halfUbSize, int64_t indicesSize, int64_t alignNum,
                                                     int64_t ubBlock)
{
    int64_t oneBlockSize = indicesSize + varTypeSize_ * eachCoreAfterAxisCount_;
    indicesFactor_ = halfUbSize / oneBlockSize;
    int64_t occupy = Ops::Base::CeilAlign(rankSize_ * indicesTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign(outOfSetTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign(outOfSetTypeSize_ + TWO * ALIGN_SIZE, ubBlock) +
                     Ops::Base::CeilAlign(INT32_BYTES, ubBlock) + Ops::Base::CeilAlign(INT32_BYTES + 1, ubBlock) +
                     Ops::Base::CeilAlign(varTypeSize_ * eachCoreAfterAxisCount_, ubBlock) +
                     GetSortTmpSize(outOfSetDtype_, 1, false);
    if (occupy > halfUbSize) {
        int64_t indicesUbSize = std::min(INDICES_MIN_BLOCK_SIZE, indicesAxis_ * indicesSize);
        indicesFactor_ = Ops::Base::CeilAlign(indicesUbSize, ALIGN_SIZE) / indicesSize;
        afterAxisFactor_ = (halfUbSize - indicesFactor_ * indicesSize) / indicesFactor_ / varTypeSize_;
        afterAxisFactor_ = Ops::Base::FloorAlign(afterAxisFactor_, alignNum);
    } else {
        afterAxisFactor_ = Ops::Base::CeilAlign(eachCoreAfterAxisCount_, alignNum);
        indicesFactor_ = halfUbSize / (afterAxisFactor_ * (varTypeSize_ + FP32_BYTES) + indicesSize);
        int64_t restSize = static_cast<int64_t>(-1);
        while (restSize <= 0) {
            --indicesFactor_;
            restSize = halfUbSize -
                       (Ops::Base::CeilAlign(indicesFactor_ * rankSize_ * indicesTypeSize_, ubBlock) +
                        Ops::Base::CeilAlign(indicesFactor_ * outOfSetTypeSize_, ubBlock) +
                        Ops::Base::CeilAlign(indicesFactor_ * (outOfSetTypeSize_ + TWO * ALIGN_SIZE), ubBlock) +
                        Ops::Base::CeilAlign(indicesFactor_ * INT32_BYTES, ubBlock) +
                        Ops::Base::CeilAlign(indicesFactor_ * (INT32_BYTES + 1), ubBlock) +
                        indicesFactor_ * Ops::Base::CeilAlign((varTypeSize_)*eachCoreAfterAxisCount_, ubBlock) +
                        GetSortTmpSize(outOfSetDtype_, indicesFactor_, false));
            if (indicesFactor_ > indicesAxis_) {
                indicesFactor_ = indicesAxis_;
                break;
            }
        }
    }
}

void ScatterNdUpdateTiling::HandleIndicesFactorLeOne(int64_t halfUbSize, int64_t indicesSize, int64_t alignNum,
                                                     int64_t ubBlock)
{
    int64_t roughMaxElemByUb = (halfUbSize > indicesSize) ? (halfUbSize - indicesSize) / (varTypeSize_ + FP32_BYTES) :
                                                            0;
    int64_t initAfterAxis = std::min(eachCoreAfterAxisCount_, roughMaxElemByUb);
    afterAxisFactor_ = Ops::Base::FloorAlign(initAfterAxis, alignNum);
    indicesFactor_ = RoughMaxIdxByUb(afterAxisFactor_, halfUbSize, indicesSize);
    indicesFactor_ = indicesFactor_ < 1 ? 1 : indicesFactor_;
    indicesFactor_ = indicesFactor_ > indicesAxis_ ? indicesAxis_ : indicesFactor_;
    bool ok = false;
    while (true) {
        int64_t unitIdxOne = UnitIdxAligned(1, ubBlock);
        int64_t uintUpOne = UnitUpdAligned(afterAxisFactor_, ubBlock);
        int64_t maxIdxByAligned = 0;
        if (unitIdxOne + uintUpOne > 0) {
            maxIdxByAligned = (halfUbSize - GetSortTmpSize(outOfSetDtype_, 1, false)) / (unitIdxOne + uintUpOne);
        }
        int64_t tryIdx = std::max<int64_t>(1, std::min({indicesFactor_, maxIdxByAligned, indicesAxis_}));
        while (tryIdx >= 1) {
            int64_t occ = OccupyTotal(tryIdx, afterAxisFactor_, ubBlock);
            if (occ < halfUbSize) {
                indicesFactor_ = tryIdx;
                ok = true;
                break;
            }
            --tryIdx;
        }
        if (ok) {
            break;
        }
        afterAxisFactor_ -= alignNum;
        indicesFactor_ = RoughMaxIdxByUb(afterAxisFactor_, halfUbSize, indicesSize);
    }
}

int64_t ScatterNdUpdateTiling::UnitIdxAligned(int64_t idxFactor, int64_t ubBlock)
{
    return Ops::Base::CeilAlign(idxFactor * static_cast<int64_t>(rankSize_) * indicesTypeSize_, ubBlock) +
           Ops::Base::CeilAlign(idxFactor * outOfSetTypeSize_, ubBlock) +
           Ops::Base::CeilAlign(idxFactor * (outOfSetTypeSize_ + TWO * ALIGN_SIZE), ubBlock) +
           Ops::Base::CeilAlign(idxFactor * INT32_BYTES, ubBlock) +
           Ops::Base::CeilAlign(idxFactor * (INT32_BYTES + 1), ubBlock);
}

int64_t ScatterNdUpdateTiling::UnitUpdAligned(int64_t afterAxisFactor, int64_t ubBlock)
{
    return Ops::Base::CeilAlign(varTypeSize_ * afterAxisFactor, ubBlock) +
           Ops::Base::CeilAlign(FP32_BYTES * afterAxisFactor, ubBlock);
}

int64_t ScatterNdUpdateTiling::OccupyTotal(int64_t idxFactor, int64_t afterAxisFactor, int64_t ubBlock)
{
    int64_t indicesPart = UnitIdxAligned(idxFactor, ubBlock);
    int64_t updatesPart = idxFactor * UnitUpdAligned(afterAxisFactor, ubBlock);
    int64_t sortTmp = GetSortTmpSize(outOfSetDtype_, idxFactor, false);
    return indicesPart + updatesPart + sortTmp;
}

int64_t ScatterNdUpdateTiling::RoughMaxIdxByUb(int64_t afterAxisFactor, int64_t halfUbSize, int64_t indicesSize)
{
    int64_t denom = afterAxisFactor * (varTypeSize_ + FP32_BYTES) + indicesSize;
    if (denom <= 0) {
        return 1;
    }
    return halfUbSize / denom;
}

void ScatterNdUpdateTiling::DoOpTilingSplitAfter()
{
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE) / DB_BUFFER);
    int64_t alignNum = ALIGN_SIZE / varTypeSize_;
    int64_t oneIndexSize = static_cast<int64_t>(rankSize_) * indicesTypeSize_;
    needInt64_ = outOfSetTypeSize_ == sizeof(int64_t);

    int64_t indicesSize = oneIndexSize + outOfSetTypeSize_ + (outOfSetTypeSize_ + TWO * ALIGN_SIZE) + INT32_BYTES +
                          (INT32_BYTES + 1);
    int64_t ubBlock = Ops::Base::GetUbBlockSize(context_);
    ComputeCoreSplitAfterAxis();
    InitFactors(halfUbSize, indicesSize, alignNum);
    if (indicesFactor_ > 1) {
        HandleIndicesFactorGtOne(halfUbSize, indicesSize, alignNum, ubBlock);
    } else {
        HandleIndicesFactorLeOne(halfUbSize, indicesSize, alignNum, ubBlock);
    }
    /* 每个核分的indices相同 */
    indicesLoopSize_ = Ops::Base::CeilDiv(indicesAxis_, indicesFactor_);
    indiceTailNum_ = indicesAxis_ - (indicesLoopSize_ - 1) * indicesFactor_;
    /* 主核循环次数 */
    updateLoopSize_ = Ops::Base::CeilDiv(eachCoreAfterAxisCount_, afterAxisFactor_);
    /* 主核尾loop处理afterAxis大小 */
    updateTailNum_ = eachCoreAfterAxisCount_ - (updateLoopSize_ - 1) * afterAxisFactor_;

    /* 尾核循环次数 */
    tailUpdateLoopSize_ = Ops::Base::CeilDiv(tailCoreAfterAxisCount_, afterAxisFactor_);
    /* 尾核尾loop处理afterAxis大小 */
    tailUpdateTailNum_ = tailCoreAfterAxisCount_ - (tailUpdateLoopSize_ - 1) * afterAxisFactor_;
    isSplitAfterAxis_ = 1;
}

void ScatterNdUpdateTiling::DoOpTilingSimdSplitIndices()
{
    int64_t alignNum = ALIGN_SIZE / varTypeSize_;
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE) / DB_BUFFER);

    /* split indices分核 */
    eachCoreIndexCount_ = Ops::Base::CeilDiv(indicesAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(indicesAxis_, eachCoreIndexCount_);
    tailCoreIndexCount_ = indicesAxis_ - eachCoreIndexCount_ * (usedCoreNumBefore_ - 1);
    int64_t oneIndexSize = static_cast<int64_t>(rankSize_) * indicesTypeSize_;

    /* 同地址优化:搬入多少行indices,就搬入相同行数的updates, strideBuf放在RESERVE_SIZE中:
     * indicesFactor_: indiecesQue + outOfsetBuf + (sortIndicesQue + 2 * shiftOfset) + originIdxQue +
     *                 (uniqueIdCntQue_ + 1)
     * indicesFactor_ * eachCoreAfterAxisCount_: updatesQue_
     */
    int64_t ubBlock = Ops::Base::GetUbBlockSize(context_);
    int64_t indicesAlignSize = Ops::Base::CeilAlign(oneIndexSize, ubBlock) +
                               Ops::Base::CeilAlign(outOfSetTypeSize_, ubBlock) +
                               Ops::Base::CeilAlign(outOfSetTypeSize_ + TWO * ALIGN_SIZE, ubBlock) +
                               Ops::Base::CeilAlign(INT32_BYTES, ubBlock) +
                               Ops::Base::CeilAlign(INT32_BYTES + 1, ubBlock);

    int64_t updateAlignSize = Ops::Base::CeilAlign(varTypeSize_ * afterAxis_, ubBlock) +
                              GetSortTmpSize(outOfSetDtype_, 1, false);
    if (indicesAlignSize + updateAlignSize > halfUbSize) {
        int64_t indicesSize = std::min(INDICES_MIN_BLOCK_SIZE, indicesAxis_ * indicesAlignSize);
        /* indicesBuf_ + outOfstBuf_ */
        indicesFactor_ = Ops::Base::CeilAlign(indicesSize, ALIGN_SIZE) / indicesAlignSize;
        afterAxisFactor_ = (halfUbSize - indicesFactor_ * indicesAlignSize) / indicesFactor_;
        afterAxisFactor_ = Ops::Base::FloorAlign(afterAxisFactor_, alignNum);
    } else {
        afterAxisFactor_ = Ops::Base::CeilAlign(afterAxis_, alignNum);
        indicesFactor_ = halfUbSize / (updateAlignSize + indicesAlignSize);
        int64_t restSize = static_cast<int64_t>(-1);
        while (restSize <= 0) {
            --indicesFactor_;
            int64_t occupy = Ops::Base::CeilAlign(indicesFactor_ * rankSize_ * indicesTypeSize_, ubBlock) +
                             Ops::Base::CeilAlign(indicesFactor_ * outOfSetTypeSize_, ubBlock) +
                             Ops::Base::CeilAlign(indicesFactor_ * (outOfSetTypeSize_ + TWO * ALIGN_SIZE), ubBlock) +
                             Ops::Base::CeilAlign(indicesFactor_ * INT32_BYTES, ubBlock) +
                             Ops::Base::CeilAlign(indicesFactor_ * (INT32_BYTES + 1), ubBlock) +
                             indicesFactor_ * Ops::Base::CeilAlign((varTypeSize_)*afterAxisFactor_, ubBlock) +
                             GetSortTmpSize(outOfSetDtype_, indicesFactor_, false);
            restSize = halfUbSize - occupy;
            if (indicesFactor_ > indicesAxis_) {
                indicesFactor_ = indicesAxis_;
                break;
            }
        }
    }
    /* 每个核分的update相同 */
    updateLoopSize_ = Ops::Base::CeilDiv(afterAxis_, afterAxisFactor_);
    updateTailNum_ = afterAxis_ - (updateLoopSize_ - 1) * afterAxisFactor_;
}

void ScatterNdUpdateTiling::DoOpTilingForSimdNonDetermin()
{
    /* 优先分after */
    int64_t splitThresh = totalCoreNum_ * MIN_HANDLE_SIZE / varTypeSize_;
    if ((afterAxis_ > splitThresh) || (indicesAxis_ < (totalCoreNum_ / TWO))) {
        DoOpTilingSplitAfter();
        return;
    }
    DoOpTilingSimdSplitIndices();
    return;
}

void ScatterNdUpdateTiling::DoOpTilingForSimdMask()
{
    int64_t ubBlock = Ops::Base::GetUbBlockSize(context_);
    int64_t alignNum = ubBlock / varTypeSize_;
    uint64_t maskSize = static_cast<uint64_t>(
        Ops::Base::CeilAlign(static_cast<int64_t>(varInAxis_) * static_cast<int64_t>(sizeof(int8_t)), ubBlock));
    /* split indices分核 */
    eachCoreIndexCount_ = Ops::Base::CeilDiv(indicesAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(indicesAxis_, eachCoreIndexCount_);
    tailCoreIndexCount_ = indicesAxis_ - eachCoreIndexCount_ * (usedCoreNumBefore_ - 1);
    int64_t oneIndexSize = static_cast<int64_t>(rankSize_) * indicesTypeSize_;
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - maskSize - RESERVE_SIZE) / DB_BUFFER);

    int64_t indicesAlignSize = Ops::Base::CeilAlign(oneIndexSize, ubBlock) +
                               Ops::Base::CeilAlign(indicesTypeSize_, ubBlock);
    int64_t updateAlignSize = Ops::Base::CeilAlign(varTypeSize_ * afterAxis_, ubBlock);
    int64_t colTotalAlign = Ops::Base::CeilAlign(afterAxis_, alignNum);
    if (colTotalAlign * varTypeSize_ < ROW_THRESH_SIZE) {
        afterAxisFactor_ = colTotalAlign;
        indicesFactor_ = std::min(eachCoreIndexCount_, halfUbSize / (updateAlignSize + indicesAlignSize));
    } else {
        indicesFactor_ = ONE;
        afterAxisFactor_ = (halfUbSize - indicesAlignSize) / varTypeSize_;
        afterAxisFactor_ = Ops::Base::FloorAlign(afterAxisFactor_, alignNum);
        afterAxisFactor_ = std::min(colTotalAlign, afterAxisFactor_);
        isSplitOneLine_ = 1;
    }
    updateLoopSize_ = Ops::Base::CeilDiv(afterAxis_, afterAxisFactor_);
    updateTailNum_ = afterAxis_ - (updateLoopSize_ - 1) * afterAxisFactor_;
}

void ScatterNdUpdateTiling::CalcDeterministicCoreSplit()
{
    calcMaskUsedCoreNum_ = Ops::Base::CeilDiv(indicesAxis_, MIN_THREAD_NUM);
    calcMaskUsedCoreNum_ = std::min(totalCoreNum_, calcMaskUsedCoreNum_);
    normCoreHandleIdx_ = Ops::Base::CeilDiv(indicesAxis_, calcMaskUsedCoreNum_);
    tailCoreHandleIdx_ = indicesAxis_ - normCoreHandleIdx_ * (calcMaskUsedCoreNum_ - 1);
    maskNormBlockLen_ = Ops::Base::FloorDiv(varStorageInAxis_, calcMaskUsedCoreNum_);
    maskTailBlockLen_ = varStorageInAxis_ - maskNormBlockLen_ * (calcMaskUsedCoreNum_ - 1);

    eachCoreIndexCount_ = Ops::Base::CeilDiv(indicesAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(indicesAxis_, eachCoreIndexCount_);
    tailCoreIndexCount_ = indicesAxis_ - eachCoreIndexCount_ * (usedCoreNumBefore_ - 1);

    if (afterAxis_ * varTypeSize_ >= MIN_SIZE_SIMD_DETERMINISTIC) {
        isDeterminSimt_ = 0;
    } else {
        isDeterminSimt_ = 1;
    }
}

void ScatterNdUpdateTiling::CalcDeterministicUpdateSplit(int64_t ubBlock)
{
    int64_t alignNum = ubBlock / varTypeSize_;
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE) / DB_BUFFER);

    int64_t updateAlignSize = Ops::Base::CeilAlign(varTypeSize_ * afterAxis_, ubBlock);
    int64_t colTotalAlign = Ops::Base::CeilAlign(afterAxis_, alignNum);
    if (colTotalAlign * varTypeSize_ < halfUbSize) {
        if (isDeterminSimt_) {
            indicesFactor_ = std::min(eachCoreIndexCount_, halfUbSize / (updateAlignSize));
            afterAxisFactor_ = afterAxis_ * indicesFactor_;
        } else {
            indicesFactor_ = ONE;
            afterAxisFactor_ = afterAxis_;
        }
    } else {
        indicesFactor_ = ONE;
        afterAxisFactor_ = halfUbSize / varTypeSize_;
        afterAxisFactor_ = Ops::Base::FloorAlign(afterAxisFactor_, alignNum);
        afterAxisFactor_ = std::min(colTotalAlign, afterAxisFactor_);
    }
    updateLoopSize_ = Ops::Base::CeilDiv(afterAxis_, afterAxisFactor_);

    // 一次搬多行场景
    if (afterAxis_ < afterAxisFactor_) {
        updateTailNum_ = afterAxisFactor_;
    } else {
        updateTailNum_ = afterAxis_ - (updateLoopSize_ - 1) * afterAxisFactor_;
    }
}

void ScatterNdUpdateTiling::CalcDeterministicIndicesSplit(int64_t ubBlock)
{
    uint64_t rows = indicesAxis_;
    int64_t start = 1;
    int64_t end = static_cast<int64_t>(rows) + 1;
    int64_t mid = 0;
    int64_t sortTmpSize = 0;
    int64_t ubBlockSize = ubBlock;

    while (end - start > 1) {
        mid = (end + start) / TWO;
        int64_t totalIndexSize = Ops::Base::CeilAlign(mid * rankSize_ * indicesTypeSize_, ubBlockSize) + // indice
                                 Ops::Base::CeilAlign(mid * outOfSetTypeSize_, ubBlockSize) +            // outOfsetBuf
                                 Ops::Base::CeilAlign(mid * outOfSetTypeSize_, ubBlockSize) +
                                 TWO * ubBlockSize + // sortIndiceBuf
                                 Ops::Base::CeilAlign(mid * static_cast<int64_t>(sizeof(uint32_t)),
                                                      ubBlockSize) + // updateOrigin
                                 Ops::Base::CeilAlign((mid + 1) * static_cast<int64_t>(sizeof(uint32_t)),
                                                      ubBlockSize) + // uniqeIdCount
                                 Ops::Base::CeilAlign(STRIDE_MAX_VALUE * indicesTypeSize_, ubBlockSize) + // strideBuf
                                 MIN_HANDLE_SIZE * FP32_BYTES;                                            // maxScore
        sortTmpSize = GetSortTmpSize(outOfSetDtype_, mid, false);
        sortTmpSize = Ops::Base::CeilAlign(sortTmpSize, ubBlockSize);
        int64_t tmpToTalSize = totalIndexSize + sortTmpSize + static_cast<int64_t>(MIN_TILING_SIZE);
        if (tmpToTalSize <= static_cast<int64_t>(ubSize_)) {
            start = mid;
        } else {
            end = mid;
        }
    }

    indicesUbFactor_ = std::min(start, normCoreHandleIdx_);
    normBlockLoop_ = Ops::Base::CeilDiv(normCoreHandleIdx_, indicesUbFactor_);
    tailBlockLoop_ = Ops::Base::CeilDiv(tailCoreHandleIdx_, indicesUbFactor_);
    normBlockTail_ = normCoreHandleIdx_ - (normBlockLoop_ - 1) * indicesUbFactor_;
    tailBlockTail_ = tailCoreHandleIdx_ - (tailBlockLoop_ - 1) * indicesUbFactor_;
}

void ScatterNdUpdateTiling::DoOpTilingForDeterministic()
{
    CalcDeterministicCoreSplit();

    int64_t ubBlock = Ops::Base::GetUbBlockSize(context_);
    CalcDeterministicUpdateSplit(ubBlock);
    CalcDeterministicIndicesSplit(ubBlock);
}

void ScatterNdUpdateTiling::CalculateMask()
{
    int64_t eachCoreIndex = Ops::Base::CeilDiv(indicesAxis_, totalCoreNum_);
    int64_t usedCoreNumMask = Ops::Base::CeilDiv(indicesAxis_, eachCoreIndex);
    float ubBound = PARTIAL_UB * ubSize_;
    int64_t coreBound = MASK_CORE * usedCoreNumMask;
    int64_t varBound = MASK_VAR * varInAxis_;
    if ((varInAxis_ < ubBound) && (indicesAxis_ > varBound) && (indicesAxis_ > coreBound) &&
        (afterAxis_ > MASK_AFTER)) {
        isMask_ = 1;
    }
}
ge::graphStatus ScatterNdUpdateTiling::DoOpTiling()
{
    if (outputStorageShapeSize_ < INT32_MAX) {
        outOfSetTypeSize_ = indicesTypeSize_;
        outOfSetDtype_ = indiceDtype_;
    } else {
        outOfSetTypeSize_ = sizeof(int64_t);
        outOfSetDtype_ = ge::DataType::DT_INT64;
    }

    if (isSimdNonDeterministic_ == 1) {
        CalculateMask();
        if (isMask_ == 1) {
            DoOpTilingForSimdMask();
        } else {
            DoOpTilingForSimdNonDetermin();
        }
    } else if (isDeterministic_ == 1) {
        DoOpTilingForDeterministic();
    } else if (isSimtWithSort_ == 1) {
        ge::graphStatus res = SortTiling();
        if (res == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    } else {
        BlockTiling();
        ge::graphStatus res = UbTiling();
        if (res == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    }
    ge::graphStatus status = SetStride();
    OP_CHECK_IF(ge::GRAPH_SUCCESS != status, OP_LOGE(opName, "SetStride failed."), return ge::GRAPH_FAILED);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t ScatterNdUpdateTiling::GetTilingKey() const
{
    uint64_t tilingKey = 0;

    if (indiceShapeSize < UINT32_MAX && updateShapeSize < UINT32_MAX && outputStorageShapeSize_ < INT32_MAX) {
        tilingKey = INPUT_ADDRESS_IN_INT32;
    } else {
        tilingKey = INPUT_ADDRESS_IN_INT64;
    }
    OP_LOGD(opName, "tilingKey = %lld.", tilingKey);
    return tilingKey;
}

ge::graphStatus ScatterNdUpdateTiling::GetWorkspaceSize()
{
    workspaceSize = RESERVED_WORKSPACE_SIZE;
    if (isDeterministic_ == 1) {
        if (indiceShapeSize < UINT32_MAX && updateShapeSize < UINT32_MAX && outputStorageShapeSize_ < INT32_MAX) {
            workspaceSize = workspaceSize + (varStorageInAxis_ + indicesAxis_ + 1) * sizeof(int32_t);
        } else {
            workspaceSize = workspaceSize + (varStorageInAxis_ + indicesAxis_ + 1) * sizeof(int64_t);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize;
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(blockNum);
    if (indiceShapeSize == 0UL || updateShapeSize == 0UL) {
        // 输入为空tensor时，设置blockNum为1，在kernel中直接返回
        context_->SetBlockDim(1);
    }
    if (isDeterministic_ == 1 || isSimdNonDeterministic_ == 1 || isMask_ == 1) {
        context_->SetBlockDim(totalCoreNum_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateTiling::SetStride()
{
    auto varShape = context_->GetInputShape(INPUT_IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varShape);
    const auto& varOriginShape = EnsureNotScalar(varShape->GetOriginShape());

    // 检查输入是否为非连续视图
    if (context_->InputIsView(INPUT_IDX_VAR)) {
        OP_LOGI(opName, "Input is view, checking non-contiguous scenario");
        auto varViewStride = context_->GetInputStride(INPUT_IDX_VAR);
        if (varViewStride != nullptr && varViewStride->GetDimNum() != 0) {
            OP_LOGI(opName, "Processing non-contiguous scenario for stride, rankSize: %u, varViewStride dimNum: %zu",
                    rankSize_, varViewStride->GetDimNum());
            for (int16_t dim = 0; dim < static_cast<int16_t>(shapeRank_); ++dim) {
                strideList[dim] = varViewStride->GetStride(dim);
                OP_LOGI(opName, "Non-contiguous stride[%d]: %llu (from view stride)", dim, strideList[dim]);
            }
            return ge::GRAPH_SUCCESS;
        } else {
            OP_LOGI(opName, "varViewStride is null or dimNum is 0, falling back to contiguous scenario");
        }
    } else {
        OP_LOGI(opName, "Input is not view, processing contiguous scenario");
    }

    // 连续场景的默认 stride 计算
    OP_LOGI(opName, "Processing contiguous scenario for stride, rankSize: %u", rankSize_);
    strideList[shapeRank_ - ONE] = static_cast<uint64_t>(1);
    for (int16_t dim = static_cast<int16_t>(shapeRank_ - TWO); dim >= 0; --dim) {
        strideList[dim] = strideList[dim + 1] * varOriginShape.GetDim(dim + 1);
        OP_LOGI(opName, "Contiguous stride[%d]: %llu (shape[%d]: %lld, stride[%d]: %llu)", dim, strideList[dim],
                dim + 1, varOriginShape.GetDim(dim + 1), dim + 1, strideList[dim + 1]);
    }
    return ge::GRAPH_SUCCESS;
}

void ScatterNdUpdateTiling::SetTilingData()
{
    ScatterNdUpdateRegBaseTilingData* tilingData = context_->GetTilingData<ScatterNdUpdateRegBaseTilingData>();

    tilingData->blockNum = blockNum;
    tilingData->blockTilingSize = blockTilingSize;
    tilingData->tailBlockTilingSize = tailBlockTilingSize;
    tilingData->ubTilingSize = ubTilingSize;
    tilingData->sliceSize = sliceSize;
    tilingData->rankSize = rankSize_;
    for (int32_t i = 0; i < MAX_SHAPE_RANK; i++) {
        tilingData->strideList[i] = strideList[i];
    }
    for (int32_t i = 0; i < MAX_SHAPE_RANK; i++) {
        tilingData->outPutShape[i] = outPutShape[i];
    }
    tilingData->outputStorageShapeSize = outputStorageShapeSize_;
    tilingData->varInAxis = varInAxis_;
    tilingData->varStorageInAxis = varStorageInAxis_;
    tilingData->indexRankSize = rankSize_;
    tilingData->afterAxis = afterAxis_;
    tilingData->usedCoreNumBefore = usedCoreNumBefore_;
    tilingData->usedCoreNumAfter = usedCoreNumAfter_;
    tilingData->eachCoreAfterAxisCount = eachCoreAfterAxisCount_;
    tilingData->tailCoreAfterAxisCount = tailCoreAfterAxisCount_;

    tilingData->updateLoopSize = updateLoopSize_;
    tilingData->updateTailNum = updateTailNum_;
    tilingData->indicesLoopSize = indicesLoopSize_;
    tilingData->indiceTailNum = indiceTailNum_;
    tilingData->tailUpdateLoopSize = tailUpdateLoopSize_;
    tilingData->tailUpdateAxisNum = tailUpdateTailNum_;
    tilingData->isSplitAfterAxis = isSplitAfterAxis_;
    tilingData->eachCoreIndexCount = eachCoreIndexCount_;
    tilingData->tailCoreIndexCount = tailCoreIndexCount_;
    tilingData->eachCoreVarCount = eachCoreVarCount_;
    tilingData->tailCoreVarCount = tailCoreVarCount_;
    tilingData->indicesFactor = indicesFactor_;
    tilingData->afterAxisFactor = afterAxisFactor_;
    tilingData->ubQuantaIndxFactor = ubQuantaIndxFactor_;
    tilingData->ubRowFactor = ubRowFactor_;
    tilingData->isDeterministic = isDeterministic_;
    tilingData->isSimtWithSort = isSimtWithSort_;
    tilingData->isSimdWithSort = isSimdWithSort_;
    tilingData->isSimdNonDeterministic = isSimdNonDeterministic_;
    tilingData->isMask = isMask_;
    tilingData->IsContiguous = IsContiguous_;
    tilingData->isSplitOneLine = isSplitOneLine_;
    tilingData->calcMaskUsedCoreNum = calcMaskUsedCoreNum_;
    tilingData->normCoreHandleIdx = normCoreHandleIdx_;
    tilingData->tailCoreHandleIdx = tailCoreHandleIdx_;
    tilingData->maskNormBlockLen = maskNormBlockLen_;
    tilingData->maskTailBlockLen = maskTailBlockLen_;
    tilingData->isDeterminSimt = isDeterminSimt_;

    tilingData->indicesUbFactor = indicesUbFactor_;
    tilingData->normBlockLoop = normBlockLoop_;
    tilingData->tailBlockLoop = tailBlockLoop_;
    tilingData->normBlockTail = normBlockTail_;
    tilingData->tailBlockTail = tailBlockTail_;
}

void ScatterNdUpdateTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "outputStorageShapeSize: " << outputStorageShapeSize_ << std::endl;
    info << "normCoreHandleIdx: " << normCoreHandleIdx_ << std::endl;
    info << "tailCoreHandleIdx: " << tailCoreHandleIdx_ << std::endl;
    info << "maskNormBlockLen: " << maskNormBlockLen_ << std::endl;
    info << "maskTailBlockLen: " << maskTailBlockLen_ << std::endl;
    info << "indicesFactor: " << indicesFactor_ << std::endl;
    info << "isDeterminSimt: " << isDeterminSimt_ << std::endl;
    info << "isDeterministic: " << isDeterministic_ << std::endl;
    info << "calcMaskUsedCoreNum: " << calcMaskUsedCoreNum_ << std::endl;
    info << "usedCoreNumBefore: " << usedCoreNumBefore_ << std::endl;
    info << "afterAxisFactor: " << afterAxisFactor_ << std::endl;
    info << "varInAxis: " << varInAxis_ << std::endl;
    info << "varStorageInAxis: " << varStorageInAxis_ << std::endl;
    info << "afterAxis: " << afterAxis_ << std::endl;
    info << "updateLoopSize: " << updateLoopSize_ << std::endl;
    info << "updateTailNum: " << updateTailNum_ << std::endl;
    info << "eachCoreIndexCount: " << eachCoreIndexCount_ << std::endl;
    info << "tailCoreIndexCount: " << tailCoreIndexCount_ << std::endl;
    info << "sliceSize: " << sliceSize << std::endl;
    info << "rankSize: " << rankSize_ << std::endl;
    info << "isMask: " << isMask_ << std::endl;
    info << "isSimtWithSort: " << isSimtWithSort_ << std::endl;
    info << "isSimdWithSort: " << isSimdWithSort_ << std::endl;
    info << "isSimdNonDeterministic: " << isSimdNonDeterministic_ << std::endl;
    info << "isSplitAfterAxis: " << isSplitAfterAxis_ << std::endl;
    info << "isSplitOneLine: " << isSplitOneLine_ << std::endl;
    info << "ubRowFactor: " << ubRowFactor_ << std::endl;
    info << "ubQuantaIndxFactor: " << ubQuantaIndxFactor_ << std::endl;
    info << "eachCoreAfterAxisCount: " << eachCoreAfterAxisCount_ << std::endl;
    OP_LOGI(opName, "Tiling info is: %s", info.str().c_str());
}

} // namespace optiling
