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
 * \file scatter_nd_update_sk_tiling_regbase.cpp
 * \brief ascendc scatter_nd_update_sk regbase tiling cpp
 */

#include "scatter_nd_update_sk_tiling_regbase.h"
#include "tiling_base/error_log.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "tiling/tiling_api.h"

using namespace AscendC;
namespace optiling {

// host 侧数学工具（vllm-ascend 无共享 Ops::Base 数学库，参考 inplace_partial_rotary_mul 做法）
template <typename T, typename U>
static inline auto SkCeilDiv(T a, U b) -> decltype(a / b)
{
    return (a + b - 1) / b;
}

template <typename T, typename U>
static inline auto SkCeilAlign(T a, U b) -> decltype(a / b)
{
    return (a + b - 1) / b * b;
}

template <typename T, typename U>
static inline auto SkFloorDiv(T a, U b) -> decltype(a / b)
{
    return a / b;
}

template <typename T, typename U>
static inline auto SkFloorAlign(T a, U b) -> decltype(a / b)
{
    return a / b * b;
}

static constexpr uint64_t INPUT_IDX_VAR = 0;
static constexpr uint64_t INPUT_IDX_INDICES = 1;
static constexpr uint64_t INPUT_IDX_UPDATES = 2;
static constexpr uint64_t OUTPUT_IDX_SHAPE = 0;
static constexpr uint64_t ATTR_IDX_STRIDES = 0;
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
static constexpr int64_t MIN_SIZE_SIMD_NONDETERMINSTIC = 128;
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
static constexpr int64_t MIN_SIZE_SIMD_DETERMINSTIC = 128;
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

// UB block 大小固定 32B（与 platform GetUbBlockSize 语义一致）
static int64_t GetSkUbBlockSize()
{
    return 32;
}

bool ScatterNdUpdateSkTilingRegbase::IsCapable()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    return platform_ascendc::PlatformAscendC(platformInfo).GetSocVersion() ==
           platform_ascendc::SocVersion::ASCEND950;
}

ge::graphStatus ScatterNdUpdateSkTilingRegbase::GetPlatformInfo()
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

ge::graphStatus ScatterNdUpdateSkTilingRegbase::ValidateVarInfo(const gert::Tensor* var, const gert::Shape& varOriginShape)
{
    // 获取并记录 var 的形状信息
    shapeRank_ = varOriginShape.GetDimNum();

    // 计算基于 origin shape 的 varShapeSize
    for (int64_t i = 0; i < shapeRank_; i++) {
        outputShapeSize *= varOriginShape.GetDim(i);
    }
    if (outputShapeSize <= 0) {
        OP_LOGE(opName, "varShapeSize must be greater than 0, got %llu", outputShapeSize);
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

ge::graphStatus ScatterNdUpdateSkTilingRegbase::ValidateIndicesInfo(const gert::Tensor* indices, gert::Shape& indiceShape,
                                                           int64_t& indiceDims)
{
    indiceShapeSize = indices->GetShapeSize();
    if (indiceShapeSize < 0UL) {
        OP_LOGE(opName, "indices shapeSize cannot be negative, got %llu", indiceShapeSize);
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
        OP_LOGE(opName, "indices dimNum must be >= 2, got %lld", indiceDims);
        return ge::GRAPH_FAILED;
    }

    if (RANK_MIN_VALUE > static_cast<uint16_t>(rankSize_) || static_cast<uint16_t>(rankSize_) > RANK_MAX_VALUE) {
        OP_LOGE(opName, "rankSize must be in the range [1, 7], got %llu", rankSize_);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

bool ScatterNdUpdateSkTilingRegbase::ReadStridesAttr(const gert::Shape& varOriginShape)
{
    // 读取 strides 属性（torch view 语义的 var 步长），校验其与 varOriginShape 是否完全连续。
    // 连续时返回 true（走 HandleContiguousCase）；否则填充 attrStrides_ 并返回 false。
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto stridesPtr = attrs->GetListInt(ATTR_IDX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, stridesPtr);
    OP_TILING_CHECK(stridesPtr->GetSize() != static_cast<size_t>(shapeRank_),
                    OP_LOGE(opName, "strides attr size (%zu) must match var dimNum (%lld)", stridesPtr->GetSize(),
                            shapeRank_),
                    return false);

    bool isContiguous = true;
    int64_t expectedStride = 1;
    for (int64_t dim = shapeRank_ - 1; dim >= 0; --dim) {
        int64_t stride = stridesPtr->GetData()[dim];
        attrStrides_[dim] = stride;
        if (varOriginShape.GetDim(dim) > 1 && stride != expectedStride) {
            isContiguous = false;
        }
        expectedStride *= varOriginShape.GetDim(dim);
    }
    return isContiguous;
}

ge::graphStatus ScatterNdUpdateSkTilingRegbase::HandleNonContiguousCase(const gert::Shape& varOriginShape)
{
    constexpr uint16_t MAX_INDICES_RANK_FOR_VIEW = 4;
    if (static_cast<uint16_t>(rankSize_) > MAX_INDICES_RANK_FOR_VIEW) {
        OP_LOGE(opName, "In non-contiguous scenarios, rankSize must be <= 4, got %llu", rankSize_);
        return ge::GRAPH_FAILED;
    }

    // 检查非索引轴是否连续：var 的 [rankSize_, shapeRank-1] 维度范围应连续
    // attrStrides_ 已由 ReadStridesAttr 从 strides 属性读出并校验过整体连续性
    bool nonIndexAxesContiguous = true;
    if (rankSize_ < static_cast<int64_t>(shapeRank_)) {
        int64_t expectedStride = 1;
        for (int64_t dim = static_cast<int64_t>(shapeRank_) - 1; dim >= static_cast<int64_t>(rankSize_); --dim) {
            int64_t dimSize = varOriginShape.GetDim(dim);
            if (dimSize > 1 && attrStrides_[dim] != expectedStride) {
                nonIndexAxesContiguous = false;
                break;
            }
            expectedStride *= dimSize;
        }
    }

    if (!nonIndexAxesContiguous) {
        OP_LOGE(opName, "non-indexed axis strides must be contiguous");
        return ge::GRAPH_FAILED;
    }

    IsContiguous_ = 0; // 非连续内存

    // sk 场景 storage shape 已被 launcher 覆盖为 viewShape，无法从 GetStorageShape 取得
    // 真实 storage 覆盖大小；由首轴 stride 推导：storageSize = stride0 * afterAxis
    // （afterAxis 即 [rankSize_, shapeRank) 维度的乘积，此时与 updates 尾部维度一致）
    int64_t afterAxis = 1;
    for (int64_t dim = static_cast<int64_t>(rankSize_); dim < shapeRank_; ++dim) {
        afterAxis *= varOriginShape.GetDim(dim);
    }
    outputStorageShapeSize_ = static_cast<uint64_t>(attrStrides_[0]) * afterAxis;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateSkTilingRegbase::HandleContiguousCase()
{
    IsContiguous_ = 1; // 连续内存
    outputStorageShapeSize_ = outputShapeSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateSkTilingRegbase::ValidateUpdatesInfo(const gert::Tensor* updates, gert::Shape& updateShape)
{
    updateShapeSize = updates->GetShapeSize();
    if (updateShapeSize < 0UL) {
        OP_LOGE(opName, "updates shapeSize must not be negative, got %llu", updateShapeSize);
        return ge::GRAPH_FAILED;
    }

    auto updateDesc = context_->GetInputDesc(INPUT_IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updateDesc);
    updateShape = updates->GetStorageShape();
    updateDtype_ = updateDesc->GetDataType();
    if (updateDtype_ != varDtype_) {
        OP_LOGE(opName, "updates and var must have the same dtype, updates: %d, var: %d",
                static_cast<int32_t>(updateDtype_), static_cast<int32_t>(varDtype_));
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateSkTilingRegbase::CalculateDerivedParams(const gert::Shape& varOriginShape,
                                                              gert::Shape& indiceShape, gert::Shape& updateShape)
{
    if (shapeRank_ < rankSize_) {
        OP_LOGE(opName, "var dimNum must be >= rankSize, var: %lld, rankSize: %llu", shapeRank_, rankSize_);
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
        OP_LOGE(opName, "The trailing dimension counts of updateRank and outputRank must match");
        return ge::GRAPH_FAILED;
    }

    // indicesAxis_ equal updatesInAxis
    indicesAxis_ = static_cast<int64_t>(indiceShapeSize / rankSize_);
    afterAxis_ = static_cast<int64_t>(updateShapeSize) / indicesAxis_;
    varInAxis_ = outputShapeSize / afterAxis_;
    varStorageInAxis_ = outputStorageShapeSize_ / afterAxis_;
    sliceSize = static_cast<uint64_t>(afterAxis_);

    if (context_->GetDeterministic() == 1 && indicesAxis_ > 1) {
        isDeterminstic_ = 1;
        context_->SetScheduleMode(1);
    }
    if (isDeterminstic_ != 1 && afterAxis_ * varTypeSize_ >= MIN_SIZE_SIMD_NONDETERMINSTIC) {
        isSimdNonDeterminstic_ = 1;
    }

    // SIMD 排序条件
    // 1. indicesAxis_ > varInAxis_：索引数量大于原始索引数量，表示高重复度
    // 2. 单核 indices 数量 > MIN_INDICES_PER_CORE_FOR_SIMD_SORT(64)：批次足够大
    int64_t estimatedIndicesPerCore = SkCeilDiv(indicesAxis_, totalCoreNum_);
    bool highDuplication = (indicesAxis_ > varInAxis_);
    bool enoughBatchPerCore = (estimatedIndicesPerCore > MIN_INDICES_PER_CORE_FOR_SIMD_SORT);
    if (isSimdNonDeterminstic_ == 1 && highDuplication && enoughBatchPerCore) {
        isSimdWithSort_ = 1;
    }

    if (indicesAxis_ / varInAxis_ >= SIMT_SORT_LIMIT) {
        isSimtWithSort_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateSkTilingRegbase::GetShapeAttrsInfo()
{
    const gert::Tensor* var = context_->GetInputTensor(INPUT_IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, var);

    // 获取并记录 var 的形状信息
    auto varShape = context_->GetInputShape(INPUT_IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varShape);
    // vllm-ascend 运行时 aclnn launcher 会将 var 的存储压平（originShape 为 1-D），
    // 真实 view 语义保存在 storageShape + strides 属性中，这里以 storageShape 为准
    const gert::Shape& varOriginShape = EnsureNotScalar(varShape->GetStorageShape());

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
    // sk 场景 storage shape 已被 launcher 覆盖为 viewShape，连续性由 strides 属性判定：
    // strided attr 与 viewShape 完全匹配连续 stride => 连续；否则为非连续 view
    if (ReadStridesAttr(varOriginShape)) {
        status = HandleContiguousCase();
        OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, OP_LOGE(opName, "HandleContiguousCase failed"), return status);
    } else {
        status = HandleNonContiguousCase(varOriginShape);
        OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, OP_LOGE(opName, "HandleNonContiguousCase failed"),
                        return status);
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

ge::graphStatus ScatterNdUpdateSkTilingRegbase::CheckScatterNdUpdateTensorShape(const gert::Shape& indiceShape,
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

void ScatterNdUpdateSkTilingRegbase::BlockTiling()
{
    auto typeSize = ge::GetSizeByDataType(updateDtype_);
    OP_CHECK_IF(typeSize == 0, OP_LOGE(opName, "typeSize is 0"), return);
    alignFactor = GetSkUbBlockSize() / typeSize;
    auto blockFactor = SkCeilDiv(updateShapeSize, static_cast<uint64_t>(totalCoreNum_));
    auto blockAlignFactor = SkCeilDiv(blockFactor, alignFactor) * alignFactor;
    blockTilingSize = std::max(static_cast<uint64_t>(blockAlignFactor), MIN_TILING_SIZE);
    blockNum = SkCeilDiv(updateShapeSize, blockTilingSize);
    tailBlockTilingSize = updateShapeSize - blockTilingSize * (blockNum - 1UL);
    OP_LOGD(opName,
            "updateShapeSize = %lld, blockFactor = %lld, blockAlignFactor = %lld,"
            "blockTilingSize = %d, tailBlockTilingSize = %d",
            updateShapeSize, blockFactor, blockAlignFactor, blockTilingSize, tailBlockTilingSize);
}

ge::graphStatus ScatterNdUpdateSkTilingRegbase::UbTiling()
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
    sliceUb = SkCeilDiv(static_cast<uint64_t>(sliceUb), alignFactor) * alignFactor;
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

ge::graphStatus ScatterNdUpdateSkTilingRegbase::SortTiling()
{
    if (indiceShapeSize == static_cast<uint64_t>(0) || updateShapeSize == static_cast<uint64_t>(0)) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(sliceSize == static_cast<uint64_t>(0),
                OP_LOGE(opName, "sliceSize %lu is zero. please check.", sliceSize), return ge::GRAPH_FAILED);
    int64_t ubBlockSize = GetSkUbBlockSize();

    // 分核策略：每个核平分行数
    uint64_t rows = indiceShapeSize / rankSize_;
    int64_t start = 1;
    int64_t end = static_cast<int64_t>(rows) + 1;
    int64_t mid = 0;
    int64_t sortTmpSize = 0;
    while (end - start > 1) {
        mid = (end + start) / TWO;
        int64_t totalIndexSize = SkCeilAlign(mid * rankSize_ * indicesTypeSize_, ubBlockSize) + // indice
                                 SkCeilAlign(mid * outOfSetTypeSize_, ubBlockSize) +            // outOfsetBuf
                                 SkCeilAlign(mid * outOfSetTypeSize_, ubBlockSize) +
                                 TWO * ubBlockSize +                                               // sortIndiceBuf
                                 SkCeilAlign(mid * indicesTypeSize_, ubBlockSize) +       // updateOrigin
                                 SkCeilAlign((mid + 1) * indicesTypeSize_, ubBlockSize) + // uniqeIdCount
                                 SkCeilAlign(STRIDE_MAX_VALUE * indicesTypeSize_, ubBlockSize) + // strideBuf
                                 MIN_HANDLE_SIZE * FP32_BYTES;                                            // maxScore
        sortTmpSize = GetSortTmpSize(outOfSetDtype_, mid, false);
        sortTmpSize = SkCeilAlign(sortTmpSize, ubBlockSize);
        int64_t tmpToTalSize = totalIndexSize + sortTmpSize + static_cast<int64_t>(MIN_TILING_SIZE);
        if (tmpToTalSize <= static_cast<int64_t>(ubSize_)) {
            start = mid;
        } else {
            end = mid;
        }
    }

    ubTilingSize = static_cast<uint32_t>(start);
    uint64_t totalLoop = SkCeilDiv(rows, static_cast<uint64_t>(ubTilingSize));
    uint64_t eachCoreLoop = SkCeilDiv(totalLoop, static_cast<uint64_t>(totalCoreNum_));
    blockNum = SkCeilDiv(totalLoop, eachCoreLoop);

    while (blockNum < static_cast<uint64_t>(totalCoreNum_ / TWO) && ubTilingSize > static_cast<uint32_t>(1)) {
        ubTilingSize = ubTilingSize / static_cast<uint32_t>(TWO);
        totalLoop = SkCeilDiv(rows, static_cast<uint64_t>(ubTilingSize));
        eachCoreLoop = SkCeilDiv(totalLoop, static_cast<uint64_t>(totalCoreNum_));
        blockNum = SkCeilDiv(totalLoop, eachCoreLoop);
    }
    blockTilingSize = eachCoreLoop * ubTilingSize;
    tailBlockTilingSize = rows - blockTilingSize * (blockNum - 1UL);
    OP_LOGD(opName,
            "rows = %lld, blockTilingSize = %lld, tailBlockTilingSize = %lld,"
            "blockNum = %d ,eachCoreLoop = %d ,",
            rows, blockTilingSize, tailBlockTilingSize, blockNum, eachCoreLoop);
    return ge::GRAPH_SUCCESS;
}

uint32_t ScatterNdUpdateSkTilingRegbase::GetSortTmpSize(ge::DataType dataType, uint32_t lastAxisNum, bool isDescend)
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

int64_t ScatterNdUpdateSkTilingRegbase::GetRestAvailableSize(int64_t sampleNum, int64_t valueTypeBytes, int64_t originalSize,
                                                    int64_t postAxisSize, ge::DataType idType)
{
    int64_t ubBlock = GetSkUbBlockSize();
    int64_t occupy = SkCeilAlign(sampleNum * rankSize_ * indicesTypeSize_, ubBlock) +
                     SkCeilAlign(sampleNum * outOfSetTypeSize_, ubBlock) +
                     SkCeilAlign(sampleNum * (outOfSetTypeSize_ + TWO * ALIGN_SIZE), ubBlock) +
                     SkCeilAlign(sampleNum * INT32_BYTES, ubBlock) +
                     SkCeilAlign(sampleNum * (INT32_BYTES * TWO), ubBlock) +
                     SkCeilAlign(sampleNum * indicesTypeSize_, ubBlock) +
                     sampleNum * SkCeilAlign((varTypeSize_)*postAxisSize, ubBlock) +
                     sampleNum * SkCeilAlign((FP32_BYTES)*postAxisSize, ubBlock) +
                     sampleNum * SkCeilAlign((FP32_BYTES)*postAxisSize, ubBlock) +
                     GetSortTmpSize(idType, sampleNum, false);
    return originalSize - occupy;
}

void ScatterNdUpdateSkTilingRegbase::ComputeCoreSplitAfterAxis()
{
    eachCoreAfterAxisCount_ = SkCeilDiv(afterAxis_, totalCoreNum_);
    usedCoreNumBefore_ = SkCeilDiv(afterAxis_, eachCoreAfterAxisCount_);
    tailCoreAfterAxisCount_ = afterAxis_ - eachCoreAfterAxisCount_ * (usedCoreNumBefore_ - 1);
}

void ScatterNdUpdateSkTilingRegbase::InitFactors(int64_t halfUbSize, int64_t indicesSize, int64_t alignNum)
{
    afterAxisFactor_ = SkCeilAlign(eachCoreAfterAxisCount_, alignNum);
    indicesFactor_ = halfUbSize / (afterAxisFactor_ * (varTypeSize_ + FP32_BYTES) + indicesSize);
}

void ScatterNdUpdateSkTilingRegbase::HandleIndicesFactorGtOne(int64_t halfUbSize, int64_t indicesSize, int64_t alignNum,
                                                     int64_t ubBlock)
{
    int64_t oneBlockSize = indicesSize + varTypeSize_ * eachCoreAfterAxisCount_;
    indicesFactor_ = halfUbSize / oneBlockSize;
    int64_t occupy = SkCeilAlign(rankSize_ * indicesTypeSize_, ubBlock) +
                     SkCeilAlign(outOfSetTypeSize_, ubBlock) +
                     SkCeilAlign(outOfSetTypeSize_ + TWO * ALIGN_SIZE, ubBlock) +
                     SkCeilAlign(INT32_BYTES, ubBlock) + SkCeilAlign(INT32_BYTES + 1, ubBlock) +
                     SkCeilAlign(varTypeSize_ * eachCoreAfterAxisCount_, ubBlock) +
                     GetSortTmpSize(outOfSetDtype_, 1, false);
    if (occupy > halfUbSize) {
        int64_t indicesUbSize = std::min(INDICES_MIN_BLOCK_SIZE, indicesAxis_ * indicesSize);
        indicesFactor_ = SkCeilAlign(indicesUbSize, ALIGN_SIZE) / indicesSize;
        afterAxisFactor_ = (halfUbSize - indicesFactor_ * indicesSize) / indicesFactor_ / varTypeSize_;
        afterAxisFactor_ = SkFloorAlign(afterAxisFactor_, alignNum);
    } else {
        afterAxisFactor_ = SkCeilAlign(eachCoreAfterAxisCount_, alignNum);
        indicesFactor_ = halfUbSize / (afterAxisFactor_ * (varTypeSize_ + FP32_BYTES) + indicesSize);
        int64_t restSize = static_cast<int64_t>(-1);
        while (restSize <= 0) {
            restSize = halfUbSize -
                       (SkCeilAlign(indicesFactor_ * rankSize_ * indicesTypeSize_, ubBlock) +
                        SkCeilAlign(indicesFactor_ * outOfSetTypeSize_, ubBlock) +
                        SkCeilAlign(indicesFactor_ * (outOfSetTypeSize_ + TWO * ALIGN_SIZE), ubBlock) +
                        SkCeilAlign(indicesFactor_ * INT32_BYTES, ubBlock) +
                        SkCeilAlign(indicesFactor_ * (INT32_BYTES + 1), ubBlock) +
                        indicesFactor_ * SkCeilAlign((varTypeSize_)*eachCoreAfterAxisCount_, ubBlock) +
                        GetSortTmpSize(outOfSetDtype_, indicesFactor_, false));
            if (restSize >= 0) {
                if (indicesFactor_ > indicesAxis_) {
                    indicesFactor_ = indicesAxis_;
                }
                break;
            }
            --indicesFactor_;
        }
    }
}

void ScatterNdUpdateSkTilingRegbase::HandleIndicesFactorLeOne(int64_t halfUbSize, int64_t indicesSize, int64_t alignNum,
                                                     int64_t ubBlock)
{
    int64_t roughMaxElemByUb = (halfUbSize > indicesSize) ? (halfUbSize - indicesSize) / (varTypeSize_ + FP32_BYTES) :
                                                            0;
    int64_t initAfterAxis = std::min(eachCoreAfterAxisCount_, roughMaxElemByUb);
    afterAxisFactor_ = SkFloorAlign(initAfterAxis, alignNum);
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

int64_t ScatterNdUpdateSkTilingRegbase::UnitIdxAligned(int64_t idxFactor, int64_t ubBlock)
{
    return SkCeilAlign(idxFactor * static_cast<int64_t>(rankSize_) * indicesTypeSize_, ubBlock) +
           SkCeilAlign(idxFactor * outOfSetTypeSize_, ubBlock) +
           SkCeilAlign(idxFactor * (outOfSetTypeSize_ + TWO * ALIGN_SIZE), ubBlock) +
           SkCeilAlign(idxFactor * INT32_BYTES, ubBlock) +
           SkCeilAlign(idxFactor * (INT32_BYTES + 1), ubBlock);
}

int64_t ScatterNdUpdateSkTilingRegbase::UnitUpdAligned(int64_t afterAxisFactor, int64_t ubBlock)
{
    return SkCeilAlign(varTypeSize_ * afterAxisFactor, ubBlock) +
           SkCeilAlign(FP32_BYTES * afterAxisFactor, ubBlock);
}

int64_t ScatterNdUpdateSkTilingRegbase::OccupyTotal(int64_t idxFactor, int64_t afterAxisFactor, int64_t ubBlock)
{
    int64_t indicesPart = UnitIdxAligned(idxFactor, ubBlock);
    int64_t updatesPart = idxFactor * UnitUpdAligned(afterAxisFactor, ubBlock);
    int64_t sortTmp = GetSortTmpSize(outOfSetDtype_, idxFactor, false);
    return indicesPart + updatesPart + sortTmp;
}

int64_t ScatterNdUpdateSkTilingRegbase::RoughMaxIdxByUb(int64_t afterAxisFactor, int64_t halfUbSize, int64_t indicesSize)
{
    int64_t denom = afterAxisFactor * (varTypeSize_ + FP32_BYTES) + indicesSize;
    if (denom <= 0) {
        return 1;
    }
    return halfUbSize / denom;
}

void ScatterNdUpdateSkTilingRegbase::DoOpTilingSplitAfter()
{
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE) / DB_BUFFER);
    int64_t alignNum = ALIGN_SIZE / varTypeSize_;
    int64_t oneIndexSize = static_cast<int64_t>(rankSize_) * indicesTypeSize_;
    needInt64_ = outOfSetTypeSize_ == sizeof(int64_t);

    int64_t indicesSize = oneIndexSize + outOfSetTypeSize_ + (outOfSetTypeSize_ + TWO * ALIGN_SIZE) + INT32_BYTES +
                          (INT32_BYTES + 1);
    int64_t ubBlock = GetSkUbBlockSize();
    ComputeCoreSplitAfterAxis();
    InitFactors(halfUbSize, indicesSize, alignNum);
    if (indicesFactor_ > 1) {
        HandleIndicesFactorGtOne(halfUbSize, indicesSize, alignNum, ubBlock);
    } else {
        HandleIndicesFactorLeOne(halfUbSize, indicesSize, alignNum, ubBlock);
    }
    /* 每个核分的indices相同 */
    indicesLoopSize_ = SkCeilDiv(indicesAxis_, indicesFactor_);
    indiceTailNum_ = indicesAxis_ - (indicesLoopSize_ - 1) * indicesFactor_;
    /* 主核循环次数 */
    updateLoopSize_ = SkCeilDiv(eachCoreAfterAxisCount_, afterAxisFactor_);
    /* 主核尾loop处理afterAxis大小 */
    updateTailNum_ = eachCoreAfterAxisCount_ - (updateLoopSize_ - 1) * afterAxisFactor_;

    /* 尾核循环次数 */
    tailUpdateLoopSize_ = SkCeilDiv(tailCoreAfterAxisCount_, afterAxisFactor_);
    /* 尾核尾loop处理afterAxis大小 */
    tailUpdateTailNum_ = tailCoreAfterAxisCount_ - (tailUpdateLoopSize_ - 1) * afterAxisFactor_;
    isSplitAfterAxis_ = 1;
}

void ScatterNdUpdateSkTilingRegbase::DoOpTilingSimdSplitIndices()
{
    int64_t alignNum = ALIGN_SIZE / varTypeSize_;
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE) / DB_BUFFER);

    /* split indices分核 */
    eachCoreIndexCount_ = SkCeilDiv(indicesAxis_, totalCoreNum_);
    usedCoreNumBefore_ = SkCeilDiv(indicesAxis_, eachCoreIndexCount_);
    tailCoreIndexCount_ = indicesAxis_ - eachCoreIndexCount_ * (usedCoreNumBefore_ - 1);
    int64_t oneIndexSize = static_cast<int64_t>(rankSize_) * indicesTypeSize_;

    /* 同地址优化:搬入多少行indices,就搬入相同行数的updates, strideBuf放在RESERVE_SIZE中:
     * indicesFactor_: indiecesQue + outOfsetBuf + (sortIndicesQue + 2 * shiftOfset) + originIdxQue +
     *                 (uniqueIdCntQue_ + 1)
     * indicesFactor_ * eachCoreAfterAxisCount_: updatesQue_
     */
    int64_t ubBlock = GetSkUbBlockSize();
    int64_t indicesAlignSize = SkCeilAlign(oneIndexSize, ubBlock) +
                               SkCeilAlign(outOfSetTypeSize_, ubBlock) +
                               SkCeilAlign(outOfSetTypeSize_ + TWO * ALIGN_SIZE, ubBlock) +
                               SkCeilAlign(INT32_BYTES, ubBlock) +
                               SkCeilAlign(INT32_BYTES + 1, ubBlock);

    int64_t updateAlignSize = SkCeilAlign(varTypeSize_ * afterAxis_, ubBlock) +
                              GetSortTmpSize(outOfSetDtype_, 1, false);
    if (indicesAlignSize + updateAlignSize > halfUbSize) {
        int64_t indicesSize = std::min(INDICES_MIN_BLOCK_SIZE, indicesAxis_ * indicesAlignSize);
        /* indicesBuf_ + outOfstBuf_ */
        indicesFactor_ = SkCeilAlign(indicesSize, ALIGN_SIZE) / indicesAlignSize;
        afterAxisFactor_ = (halfUbSize - indicesFactor_ * indicesAlignSize) / indicesFactor_;
        afterAxisFactor_ = SkFloorAlign(afterAxisFactor_, alignNum);
    } else {
        afterAxisFactor_ = SkCeilAlign(afterAxis_, alignNum);
        indicesFactor_ = halfUbSize / (updateAlignSize + indicesAlignSize);
        int64_t restSize = static_cast<int64_t>(-1);
        while (restSize <= 0) {
            int64_t occupy = SkCeilAlign(indicesFactor_ * rankSize_ * indicesTypeSize_, ubBlock) +
                             SkCeilAlign(indicesFactor_ * outOfSetTypeSize_, ubBlock) +
                             SkCeilAlign(indicesFactor_ * (outOfSetTypeSize_ + TWO * ALIGN_SIZE), ubBlock) +
                             SkCeilAlign(indicesFactor_ * INT32_BYTES, ubBlock) +
                             SkCeilAlign(indicesFactor_ * (INT32_BYTES + 1), ubBlock) +
                             indicesFactor_ * SkCeilAlign((varTypeSize_)*afterAxisFactor_, ubBlock) +
                             GetSortTmpSize(outOfSetDtype_, indicesFactor_, false);
            restSize = halfUbSize - occupy;
            if (restSize >= 0) {
                if (indicesFactor_ > indicesAxis_) {
                    indicesFactor_ = indicesAxis_;
                }
                break;
            }
            --indicesFactor_;
        }
    }
    /* 每个核分的update相同 */
    updateLoopSize_ = SkCeilDiv(afterAxis_, afterAxisFactor_);
    updateTailNum_ = afterAxis_ - (updateLoopSize_ - 1) * afterAxisFactor_;
}

void ScatterNdUpdateSkTilingRegbase::DoOpTilingForSimdNonDetermin()
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

void ScatterNdUpdateSkTilingRegbase::DoOpTilingForSimdMask()
{
    int64_t ubBlock = GetSkUbBlockSize();
    int64_t alignNum = ubBlock / varTypeSize_;
    uint64_t maskSize = static_cast<uint64_t>(
        SkCeilAlign(static_cast<int64_t>(varInAxis_) * static_cast<int64_t>(sizeof(int8_t)), ubBlock));
    /* split indices分核 */
    eachCoreIndexCount_ = SkCeilDiv(indicesAxis_, totalCoreNum_);
    usedCoreNumBefore_ = SkCeilDiv(indicesAxis_, eachCoreIndexCount_);
    tailCoreIndexCount_ = indicesAxis_ - eachCoreIndexCount_ * (usedCoreNumBefore_ - 1);
    int64_t oneIndexSize = static_cast<int64_t>(rankSize_) * indicesTypeSize_;
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - maskSize - RESERVE_SIZE) / DB_BUFFER);

    int64_t indicesAlignSize = SkCeilAlign(oneIndexSize, ubBlock) +
                               SkCeilAlign(indicesTypeSize_, ubBlock);
    int64_t updateAlignSize = SkCeilAlign(varTypeSize_ * afterAxis_, ubBlock);
    int64_t colTotalAlign = SkCeilAlign(afterAxis_, alignNum);
    if (colTotalAlign * varTypeSize_ < ROW_THRESH_SIZE) {
        afterAxisFactor_ = colTotalAlign;
        indicesFactor_ = std::min(eachCoreIndexCount_, halfUbSize / (updateAlignSize + indicesAlignSize));
    } else {
        indicesFactor_ = ONE;
        afterAxisFactor_ = (halfUbSize - indicesAlignSize) / varTypeSize_;
        afterAxisFactor_ = SkFloorAlign(afterAxisFactor_, alignNum);
        afterAxisFactor_ = std::min(colTotalAlign, afterAxisFactor_);
        isSplitOneLine_ = 1;
    }
    updateLoopSize_ = SkCeilDiv(afterAxis_, afterAxisFactor_);
    updateTailNum_ = afterAxis_ - (updateLoopSize_ - 1) * afterAxisFactor_;
}

void ScatterNdUpdateSkTilingRegbase::CalcDeterministicCoreSplit()
{
    calcMaskUsedCoreNum_ = SkCeilDiv(indicesAxis_, MIN_THREAD_NUM);
    calcMaskUsedCoreNum_ = std::min(totalCoreNum_, calcMaskUsedCoreNum_);
    normCoreHandleIdx_ = SkCeilDiv(indicesAxis_, calcMaskUsedCoreNum_);
    tailCoreHandleIdx_ = indicesAxis_ - normCoreHandleIdx_ * (calcMaskUsedCoreNum_ - 1);
    maskNormBlockLen_ = SkFloorDiv(varStorageInAxis_, calcMaskUsedCoreNum_);
    maskTailBlockLen_ = varStorageInAxis_ - maskNormBlockLen_ * (calcMaskUsedCoreNum_ - 1);

    eachCoreIndexCount_ = SkCeilDiv(indicesAxis_, totalCoreNum_);
    usedCoreNumBefore_ = SkCeilDiv(indicesAxis_, eachCoreIndexCount_);
    tailCoreIndexCount_ = indicesAxis_ - eachCoreIndexCount_ * (usedCoreNumBefore_ - 1);

    if (afterAxis_ * varTypeSize_ >= MIN_SIZE_SIMD_DETERMINSTIC) {
        isDeterminSimt_ = 0;
    } else {
        isDeterminSimt_ = 1;
    }
}

void ScatterNdUpdateSkTilingRegbase::CalcDeterministicUpdateSplit(int64_t ubBlock)
{
    int64_t alignNum = ubBlock / varTypeSize_;
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE) / DB_BUFFER);

    int64_t updateAlignSize = SkCeilAlign(varTypeSize_ * afterAxis_, ubBlock);
    int64_t colTotalAlign = SkCeilAlign(afterAxis_, alignNum);
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
        afterAxisFactor_ = SkFloorAlign(afterAxisFactor_, alignNum);
        afterAxisFactor_ = std::min(colTotalAlign, afterAxisFactor_);
    }
    updateLoopSize_ = SkCeilDiv(afterAxis_, afterAxisFactor_);

    // 一次搬多行场景
    if (afterAxis_ < afterAxisFactor_) {
        updateTailNum_ = afterAxisFactor_;
    } else {
        updateTailNum_ = afterAxis_ - (updateLoopSize_ - 1) * afterAxisFactor_;
    }
}

void ScatterNdUpdateSkTilingRegbase::CalcDeterministicIndicesSplit(int64_t ubBlock)
{
    uint64_t rows = indicesAxis_;
    int64_t start = 1;
    int64_t end = static_cast<int64_t>(rows) + 1;
    int64_t mid = 0;
    int64_t sortTmpSize = 0;
    int64_t ubBlockSize = ubBlock;

    while (end - start > 1) {
        mid = (end + start) / TWO;
        int64_t totalIndexSize = SkCeilAlign(mid * rankSize_ * indicesTypeSize_, ubBlockSize) + // indice
                                 SkCeilAlign(mid * outOfSetTypeSize_, ubBlockSize) +            // outOfsetBuf
                                 SkCeilAlign(mid * outOfSetTypeSize_, ubBlockSize) +
                                 TWO * ubBlockSize + // sortIndiceBuf
                                 SkCeilAlign(mid * static_cast<int64_t>(sizeof(uint32_t)),
                                                      ubBlockSize) + // updateOrigin
                                 SkCeilAlign((mid + 1) * static_cast<int64_t>(sizeof(uint32_t)),
                                                      ubBlockSize) + // uniqeIdCount
                                 SkCeilAlign(STRIDE_MAX_VALUE * indicesTypeSize_, ubBlockSize) + // strideBuf
                                 MIN_HANDLE_SIZE * FP32_BYTES;                                            // maxScore
        sortTmpSize = GetSortTmpSize(outOfSetDtype_, mid, false);
        sortTmpSize = SkCeilAlign(sortTmpSize, ubBlockSize);
        int64_t tmpToTalSize = totalIndexSize + sortTmpSize + static_cast<int64_t>(MIN_TILING_SIZE);
        if (tmpToTalSize <= static_cast<int64_t>(ubSize_)) {
            start = mid;
        } else {
            end = mid;
        }
    }

    indicesUbFactor_ = std::min(start, normCoreHandleIdx_);
    normBlockLoop_ = SkCeilDiv(normCoreHandleIdx_, indicesUbFactor_);
    tailBlockLoop_ = SkCeilDiv(tailCoreHandleIdx_, indicesUbFactor_);
    normBlockTail_ = normCoreHandleIdx_ - (normBlockLoop_ - 1) * indicesUbFactor_;
    tailBlockTail_ = tailCoreHandleIdx_ - (tailBlockLoop_ - 1) * indicesUbFactor_;
}

void ScatterNdUpdateSkTilingRegbase::DoOpTilingForDeterministic()
{
    CalcDeterministicCoreSplit();

    int64_t ubBlock = GetSkUbBlockSize();
    CalcDeterministicUpdateSplit(ubBlock);
    CalcDeterministicIndicesSplit(ubBlock);
}

void ScatterNdUpdateSkTilingRegbase::CalculateMask()
{
    int64_t eachCoreIndex = SkCeilDiv(indicesAxis_, totalCoreNum_);
    int64_t usedCoreNumMask = SkCeilDiv(indicesAxis_, eachCoreIndex);
    float ubBound = PARTIAL_UB * ubSize_;
    int64_t coreBound = MASK_CORE * usedCoreNumMask;
    int64_t varBound = MASK_VAR * varInAxis_;
    if ((varInAxis_ < ubBound) && (indicesAxis_ > varBound) && (indicesAxis_ > coreBound) &&
        (afterAxis_ > MASK_AFTER)) {
        isMask_ = 1;
    }
}
ge::graphStatus ScatterNdUpdateSkTilingRegbase::DoOpTiling()
{
    if (outputStorageShapeSize_ < INT32_MAX) {
        outOfSetTypeSize_ = indicesTypeSize_;
        outOfSetDtype_ = indiceDtype_;
    } else {
        outOfSetTypeSize_ = sizeof(int64_t);
        outOfSetDtype_ = ge::DataType::DT_INT64;
    }

    if (isSimdNonDeterminstic_ == 1) {
        CalculateMask();
        if (isMask_ == 1) {
            DoOpTilingForSimdMask();
        } else {
            DoOpTilingForSimdNonDetermin();
        }
    } else if (isDeterminstic_ == 1) {
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

ge::graphStatus ScatterNdUpdateSkTilingRegbase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t ScatterNdUpdateSkTilingRegbase::GetTilingKey() const
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

ge::graphStatus ScatterNdUpdateSkTilingRegbase::GetWorkspaceSize()
{
    workspaceSize = RESERVED_WORKSPACE_SIZE;
    if (isDeterminstic_ == 1) {
        if (indiceShapeSize < UINT32_MAX && updateShapeSize < UINT32_MAX && outputStorageShapeSize_ < INT32_MAX) {
            workspaceSize = workspaceSize + (varStorageInAxis_ + indicesAxis_ + 1) * sizeof(int32_t);
        } else {
            workspaceSize = workspaceSize + (varStorageInAxis_ + indicesAxis_ + 1) * sizeof(int64_t);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateSkTilingRegbase::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize;
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(blockNum);
    if (indiceShapeSize == 0UL || updateShapeSize == 0UL) {
        // 输入为空tensor时，设置blockNum为1，在kernel中直接返回
        context_->SetBlockDim(1);
    }
    if (isDeterminstic_ == 1 || isSimdNonDeterminstic_ == 1 || isMask_ == 1) {
        context_->SetBlockDim(totalCoreNum_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateSkTilingRegbase::SetStride()
{
    auto varShape = context_->GetInputShape(INPUT_IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varShape);
    // vllm-ascend 运行时 originShape 可能被 launcher 压平为 1-D 存储，
    // 连续场景的默认 stride 需基于真实 view 形状（storageShape）计算
    const gert::Shape& varViewShape = EnsureNotScalar(varShape->GetStorageShape());

    // 检查输入是否为非连续视图（连续性已由 GetShapeAttrsInfo 通过 strides 属性判定）
    if (IsContiguous_ != 1) {
        OP_LOGI(opName, "Processing non-contiguous scenario for stride, rankSize: %u, shapeRank: %lld",
                rankSize_, shapeRank_);
        for (int16_t dim = 0; dim < static_cast<int16_t>(shapeRank_); ++dim) {
            strideList[dim] = static_cast<uint64_t>(attrStrides_[dim]);
            OP_LOGI(opName, "Non-contiguous stride[%d]: %llu (from strides attr)", dim, strideList[dim]);
        }
        return ge::GRAPH_SUCCESS;
    }

    // 连续场景的默认 stride 计算
    OP_LOGI(opName, "Processing contiguous scenario for stride, rankSize: %u", rankSize_);
    strideList[shapeRank_ - ONE] = static_cast<uint64_t>(1);
    for (int16_t dim = static_cast<int16_t>(shapeRank_ - TWO); dim >= 0; --dim) {
        strideList[dim] = strideList[dim + 1] * varViewShape.GetDim(dim + 1);
        OP_LOGI(opName, "Contiguous stride[%d]: %llu (shape[%d]: %lld, stride[%d]: %llu)", dim, strideList[dim],
                dim + 1, varViewShape.GetDim(dim + 1), dim + 1, strideList[dim + 1]);
    }
    return ge::GRAPH_SUCCESS;
}

void ScatterNdUpdateSkTilingRegbase::SetTilingData()
{
    ScatterNdUpdateSkRegBaseTilingData* tilingData = context_->GetTilingData<ScatterNdUpdateSkRegBaseTilingData>();

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
    tilingData->isDeterminstic = isDeterminstic_;
    tilingData->isSimtWithSort = isSimtWithSort_;
    tilingData->isSimdWithSort = isSimdWithSort_;
    tilingData->isSimdNonDeterminstic = isSimdNonDeterminstic_;
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

void ScatterNdUpdateSkTilingRegbase::DumpTilingInfo()
{
    std::ostringstream info;
    info << "outputStorageShapeSize: " << outputStorageShapeSize_ << std::endl;
    info << "normCoreHandleIdx: " << normCoreHandleIdx_ << std::endl;
    info << "tailCoreHandleIdx: " << tailCoreHandleIdx_ << std::endl;
    info << "maskNormBlockLen: " << maskNormBlockLen_ << std::endl;
    info << "maskTailBlockLen: " << maskTailBlockLen_ << std::endl;
    info << "indicesFactor: " << indicesFactor_ << std::endl;
    info << "isDeterminSimt: " << isDeterminSimt_ << std::endl;
    info << "isDeterminstic: " << isDeterminstic_ << std::endl;
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
    info << "isSimdNonDeterminstic: " << isSimdNonDeterminstic_ << std::endl;
    info << "isSplitAfterAxis: " << isSplitAfterAxis_ << std::endl;
    info << "isSplitOneLine: " << isSplitOneLine_ << std::endl;
    info << "ubRowFactor: " << ubRowFactor_ << std::endl;
    info << "ubQuantaIndxFactor: " << ubQuantaIndxFactor_ << std::endl;
    info << "eachCoreAfterAxisCount: " << eachCoreAfterAxisCount_ << std::endl;
    OP_LOGI(opName, "Tiling info is: %s", info.str().c_str());
}

} // namespace optiling
