/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Adapted from cann/ops-nn norm/add_rms_norm/op_host/add_rms_norm_tiling_arch35.cpp,
// v9.2.0-beta.2 @ 30ef7dd563c8a4b74c3161835c8e47d1d96f87b6.
// Local changes: isolated platform adapter, checked input/range contract and optional beta UB accounting.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "add_rms_norm_bias_tiling.h"
#include "log/ops_log.h"

namespace optiling {
namespace addRmsNormBiasRegbase {
namespace {
constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t X2_INDEX = 1;
constexpr uint32_t GAMMA_INDEX = 2;
constexpr uint32_t BETA_INDEX = 3;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t RSTD_INDEX = 1;
constexpr uint32_t X_INDEX = 2;
constexpr size_t MAX_DIM_NUM = 8;
constexpr uint64_t UB_USED = 1024;
constexpr uint64_t UB_RESERVE_FOR_RSTDALIGN = 1024;
constexpr uint64_t MODE_R_FULL_LOAD = 1000;
constexpr uint64_t MODE_SPLIT_D = 2000;
constexpr uint64_t DOUBLE_BUFFER_NUM = 2;
constexpr uint64_t FULL_LOAD_QUEUE_NUM = 4;
constexpr uint64_t SPLIT_QUEUE_NUM = 5;
constexpr uint64_t RETAINED_SIZE = 5120;
constexpr uint64_t SPLIT_ALIGN_BYTES = 512;
constexpr uint64_t NDDMA_BETTER_STAGE = 512;
constexpr uint64_t ONCE_VECTOR_SIZE = 256;
// Matches the copied arch35 kernel platform helpers; only used after ASCEND950 dispatch.
constexpr uint64_t UB_BLOCK_BYTES = 32;
constexpr uint64_t VECTOR_REGISTER_BYTES = 256;
constexpr uint64_t FP32_VECTOR_ELEMENTS = VECTOR_REGISTER_BYTES / sizeof(float);
constexpr uint64_t MAX_SPLIT_UB_FACTOR = ONCE_VECTOR_SIZE * 2 * FP32_VECTOR_ELEMENTS;
constexpr uint64_t UINT32_LIMIT = std::numeric_limits<uint32_t>::max();
constexpr uint64_t MAX_FULL_LOAD_ROWS = std::numeric_limits<uint16_t>::max();
constexpr uint64_t MAX_TOTAL_ELEMENTS = std::numeric_limits<int64_t>::max() / sizeof(float);
constexpr size_t SYSTEM_WORKSPACE_BYTES = 16UL * 1024UL * 1024UL;
constexpr size_t USER_WORKSPACE_BYTES = 256;

bool SameShape(const gert::Shape& left, const gert::Shape& right)
{
    if (left.GetDimNum() != right.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < left.GetDimNum(); ++i) {
        if (left.GetDim(i) != right.GetDim(i)) {
            return false;
        }
    }
    return true;
}

ge::graphStatus ValidateInputs(gert::TilingContext* context, uint64_t& numRow, uint64_t& numCol,
                              ge::DataType& dtype, bool& hasBeta, float& epsilon)
{
    const auto x1 = context->GetInputShape(X1_INDEX);
    const auto x2 = context->GetInputShape(X2_INDEX);
    const auto gamma = context->GetInputShape(GAMMA_INDEX);
    const auto y = context->GetOutputShape(Y_INDEX);
    const auto rstd = context->GetOutputShape(RSTD_INDEX);
    const auto x = context->GetOutputShape(X_INDEX);
    OP_CHECK_IF(x1 == nullptr || x2 == nullptr || gamma == nullptr || y == nullptr || rstd == nullptr || x == nullptr,
                OP_LOGE(context, "A5 AddRmsNormBias requires all mandatory input and output shapes."),
                return ge::GRAPH_FAILED);
    const auto& xShape = x1->GetStorageShape();
    const auto& gammaShape = gamma->GetStorageShape();
    const auto& rstdShape = rstd->GetStorageShape();
    const size_t xRank = xShape.GetDimNum();
    const size_t gammaRank = gammaShape.GetDimNum();
    OP_CHECK_IF(xRank == 0 || xRank > MAX_DIM_NUM || gammaRank == 0 || gammaRank > xRank,
                OP_LOGE(context, "A5 AddRmsNormBias requires 1 <= gamma rank <= x rank <= 8."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SameShape(xShape, x2->GetStorageShape()) || !SameShape(xShape, y->GetStorageShape()) ||
                    !SameShape(xShape, x->GetStorageShape()) || rstdShape.GetDimNum() != xRank,
                OP_LOGE(context, "A5 AddRmsNormBias x1/x2/y/x shapes must match and rstd must have x rank."),
                return ge::GRAPH_FAILED);

    numRow = 1;
    numCol = 1;
    uint64_t totalElements = 1;
    for (size_t i = 0; i < xRank; ++i) {
        const int64_t dim = xShape.GetDim(i);
        // GM offsets are widened in the A5 kernel; N and D remain uint32 SplitD fields.
        // Check products before multiplication, including the FP32 byte-address range.
        OP_CHECK_IF(dim <= 0 || static_cast<uint64_t>(dim) > MAX_TOTAL_ELEMENTS / totalElements,
                    OP_LOGE(context, "A5 AddRmsNormBias requires positive dimensions within the address range."),
                    return ge::GRAPH_FAILED);
        totalElements *= static_cast<uint64_t>(dim);
        const bool normalizedDim = i >= xRank - gammaRank;
        OP_CHECK_IF(rstdShape.GetDim(i) != (normalizedDim ? 1 : dim),
                    OP_LOGE(context, "A5 AddRmsNormBias rstd must keep the normalized dimensions as 1."),
                    return ge::GRAPH_FAILED);
        if (normalizedDim) {
            OP_CHECK_IF(gammaShape.GetDim(i - (xRank - gammaRank)) != dim,
                        OP_LOGE(context, "A5 AddRmsNormBias gamma must match the trailing input dimensions."),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(static_cast<uint64_t>(dim) > UINT32_LIMIT / numCol,
                        OP_LOGE(context, "A5 AddRmsNormBias normalized size D exceeds UINT32_MAX."),
                        return ge::GRAPH_FAILED);
            numCol *= static_cast<uint64_t>(dim);
        } else {
            OP_CHECK_IF(static_cast<uint64_t>(dim) > UINT32_LIMIT / numRow,
                        OP_LOGE(context, "A5 AddRmsNormBias row count N exceeds UINT32_MAX."),
                        return ge::GRAPH_FAILED);
            numRow *= static_cast<uint64_t>(dim);
        }
    }

    const auto x1Desc = context->GetInputDesc(X1_INDEX);
    const auto x2Desc = context->GetInputDesc(X2_INDEX);
    const auto gammaDesc = context->GetInputDesc(GAMMA_INDEX);
    const auto yDesc = context->GetOutputDesc(Y_INDEX);
    const auto rstdDesc = context->GetOutputDesc(RSTD_INDEX);
    const auto xDesc = context->GetOutputDesc(X_INDEX);
    OP_CHECK_IF(x1Desc == nullptr || x2Desc == nullptr || gammaDesc == nullptr || yDesc == nullptr ||
                    rstdDesc == nullptr || xDesc == nullptr,
                OP_LOGE(context, "A5 AddRmsNormBias requires all mandatory tensor descriptors."),
                return ge::GRAPH_FAILED);
    dtype = x1Desc->GetDataType();
    OP_CHECK_IF((dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT) ||
                    x2Desc->GetDataType() != dtype || gammaDesc->GetDataType() != dtype ||
                    yDesc->GetDataType() != dtype || xDesc->GetDataType() != dtype ||
                    rstdDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context, "A5 AddRmsNormBias requires matching FP16/BF16/FP32 tensors and FP32 rstd."),
                return ge::GRAPH_FAILED);

    const auto betaDesc = context->GetOptionalInputDesc(BETA_INDEX);
    hasBeta = betaDesc != nullptr;
    if (hasBeta) {
        const auto betaShape = context->GetOptionalInputShape(BETA_INDEX);
        OP_CHECK_IF(betaShape == nullptr || !SameShape(betaShape->GetStorageShape(), gammaShape) ||
                        betaDesc->GetDataType() != dtype,
                    OP_LOGE(context, "A5 AddRmsNormBias beta must have the same shape and dtype as gamma."),
                    return ge::GRAPH_FAILED);
    }
    const auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* epsilonAttr = attrs->GetFloat(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsilonAttr);
    epsilon = *epsilonAttr;
    OP_CHECK_IF(!std::isfinite(epsilon) || epsilon < 0,
                OP_LOGE(context, "A5 AddRmsNormBias epsilon must be finite and nonnegative."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

uint64_t ComputeSplitBufferBytes(ge::DataType dtype, uint64_t dtypeBytes, uint64_t length)
{
    // A single beta tile replaces the second gamma tile, preserving the no-beta budget.
    const uint64_t queueBytes = DOUBLE_BUFFER_NUM * length * dtypeBytes * SPLIT_QUEUE_NUM +
                                FP32_VECTOR_ELEMENTS * DOUBLE_BUFFER_NUM * sizeof(float);
    const uint64_t temporaryBytes = dtype == ge::DT_FLOAT ? 0 : length * sizeof(float) * 2;
    return queueBytes + temporaryBytes + RETAINED_SIZE;
}

template <typename TilingData>
ge::graphStatus SaveTiling(gert::TilingContext* context, TilingData& tiling, uint64_t key, uint32_t useCoreNum)
{
    auto rawTiling = context->GetRawTilingData();
    OP_CHECK_IF(rawTiling == nullptr || rawTiling->GetData() == nullptr ||
                    rawTiling->GetCapacity() < tiling.GetDataSize(),
                OP_LOGE(context, "A5 AddRmsNormBias tiling buffer is too small."), return ge::GRAPH_FAILED);
    auto workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = USER_WORKSPACE_BYTES + SYSTEM_WORKSPACE_BYTES;
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(key);
    context->SetBlockDim(useCoreNum);
    OPS_LOG_I(context, "A5 AddRmsNormBias key=%lu blockDim=%u nullptr_beta=%u", key, useCoreNum,
              tiling.get_nullptr_beta());
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus TilingAddRmsNormBiasRegbase(gert::TilingContext* context)
{
    uint64_t numRow = 0;
    uint64_t numCol = 0;
    ge::DataType dtype;
    bool hasBeta = false;
    float epsilon = 0;
    if (ValidateInputs(context, numRow, numCol, dtype, hasBeta, epsilon) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t numCore = 0;
    uint64_t rawUbSize = 0;
    const auto compileInfo = reinterpret_cast<const AddRmsNormBiasCompileInfo*>(context->GetCompileInfo());
    if (compileInfo != nullptr) {
        numCore = compileInfo->totalCoreNum;
        rawUbSize = compileInfo->totalUbSize;
    } else {
        OP_CHECK_NULL_WITH_CONTEXT(context, context->GetPlatformInfo());
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        numCore = platform.GetCoreNumAiv();
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, rawUbSize);
    }
    OP_CHECK_IF(numCore == 0 || rawUbSize <= UB_USED || rawUbSize > UINT32_LIMIT,
                OP_LOGE(context, "A5 AddRmsNormBias platform core count or UB capacity is invalid."),
                return ge::GRAPH_FAILED);
    const uint64_t ubSize = rawUbSize - UB_USED;
    const uint64_t dtypeBytes = dtype == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    const uint64_t blockFactor = CeilDiv<uint64_t>(numRow, numCore);
    const uint32_t useCoreNum = static_cast<uint32_t>(CeilDiv(numRow, blockFactor));
    const float avgFactor = 1.0f / static_cast<float>(numCol);
    uint64_t numColAlign = CeilAlign(numCol * dtypeBytes, UB_BLOCK_BYTES) / dtypeBytes;

    uint64_t binAddQuotient = 1;
    while (binAddQuotient <= numColAlign / 2) {
        binAddQuotient *= 2;
    }
    if (binAddQuotient == numColAlign) {
        binAddQuotient /= 2;
    }
    const uint64_t binAddBufferOneline = CeilAlign(CeilDiv(binAddQuotient, FP32_VECTOR_ELEMENTS),
                                                   UB_BLOCK_BYTES / sizeof(float));
    // Keep no-beta tiling: one residual-output row tile saves at least the single beta vector.
    const uint64_t parameterBytes = numColAlign * dtypeBytes;
    const uint64_t rowBytes = numColAlign * dtypeBytes * DOUBLE_BUFFER_NUM * FULL_LOAD_QUEUE_NUM +
                              numColAlign * sizeof(float) + sizeof(float) * (DOUBLE_BUFFER_NUM + 1) +
                              binAddBufferOneline * sizeof(float);
    const uint64_t binaryAddMaxLength = FP32_VECTOR_ELEMENTS * FP32_VECTOR_ELEMENTS * 4;
    uint64_t rowFactor = 0;
    if (ubSize > UB_RESERVE_FOR_RSTDALIGN + parameterBytes && numColAlign <= binaryAddMaxLength) {
        rowFactor = (ubSize - UB_RESERVE_FOR_RSTDALIGN - parameterBytes) / rowBytes;
    }
    if (rowFactor != 0) {
        // Full-load DMA blocks, VF row loops and CalculateXAdd's total vector
        // loop count are uint16. Bound local FP32 byte offsets as well.
        rowFactor = std::min({rowFactor, blockFactor, MAX_FULL_LOAD_ROWS,
                              MAX_FULL_LOAD_ROWS * FP32_VECTOR_ELEMENTS / numColAlign,
                              UINT32_LIMIT / (numColAlign * sizeof(float))});
        AddRMSNormBiasRegbaseRFullLoadTilingData tiling;
        tiling.set_numRow(numRow);
        tiling.set_numCol(numCol);
        tiling.set_numColAlign(numColAlign);
        tiling.set_blockFactor(blockFactor);
        tiling.set_rowFactor(rowFactor);
        tiling.set_binAddQuotient(binAddQuotient);
        tiling.set_epsilon(epsilon);
        tiling.set_avgFactor(avgFactor);
        tiling.set_nullptr_beta(hasBeta ? 0 : 1);
        return SaveTiling(context, tiling, MODE_R_FULL_LOAD, useCoreNum);
    }

    numColAlign = CeilAlign(numCol * dtypeBytes, SPLIT_ALIGN_BYTES) / dtypeBytes;
    uint64_t ubFactor = 1;
    while (ubFactor < MAX_SPLIT_UB_FACTOR &&
           ComputeSplitBufferBytes(dtype, dtypeBytes, ubFactor * 2) < ubSize) {
        ubFactor *= 2;
    }
    OP_CHECK_IF(numColAlign > UINT32_LIMIT || ubFactor > numCol || ubFactor * dtypeBytes < SPLIT_ALIGN_BYTES ||
                    ComputeSplitBufferBytes(dtype, dtypeBytes, ubFactor) >= ubSize,
                OP_LOGE(context, "A5 AddRmsNormBias SplitD shape or UB tile is outside the supported range."),
                return ge::GRAPH_FAILED);
    uint64_t ubLoop = 1;
    while (ubLoop * 2 * ubFactor <= numCol) {
        ubLoop *= 2;
    }
    AddRMSNormBiasRegbaseTilingData tiling;
    tiling.set_numRow(static_cast<uint32_t>(numRow));
    tiling.set_numCol(static_cast<uint32_t>(numCol));
    tiling.set_numColAlign(static_cast<uint32_t>(numColAlign));
    tiling.set_blockFactor(static_cast<uint32_t>(blockFactor));
    tiling.set_rowFactor(FP32_VECTOR_ELEMENTS);
    tiling.set_ubFactor(static_cast<uint32_t>(ubFactor));
    tiling.set_epsilon(epsilon);
    tiling.set_avgFactor(avgFactor);
    tiling.set_ubLoop(static_cast<uint32_t>(ubLoop));
    tiling.set_colBuferLength(static_cast<uint32_t>(ubFactor));
    tiling.set_multiNNum(0);
    tiling.set_isNddma(numCol >= NDDMA_BETTER_STAGE ? 0 : 1);
    tiling.set_nullptr_beta(hasBeta ? 0 : 1);
    return SaveTiling(context, tiling, MODE_SPLIT_D, useCoreNum);
}
} // namespace addRmsNormBiasRegbase
} // namespace optiling
