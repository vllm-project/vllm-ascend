/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_bias_tiling_arch35.cpp
 * \brief
 */
// Adapted from cann/ops-nn v9.2.0-beta.2 @ 30ef7dd563c8a4b74c3161835c8e47d1d96f87b6.
// Preserve upstream tiling; add optional beta storage and local platform/logging adapters.

#include <algorithm>
#include <map>
#include "register/op_impl_registry.h"
#include "add_rms_norm_bias_tiling.h"
#include "log/ops_log.h"

namespace optiling {
namespace addRmsNormBiasRegbase {
constexpr uint32_t ULONG_BIT_LEN = 64;
constexpr uint32_t DTYPE_KEY_FP16 = 1;
constexpr uint32_t DTYPE_KEY_FP32 = 2;
constexpr uint32_t DTYPE_KEY_BF16 = 3;
constexpr uint32_t FLOAT_BLOCK_ALIGN_NUM = 8;
constexpr uint32_t FLOAT_PER_REAPEAT = 64;
constexpr uint32_t BYTE_SIZE_2_BLOCK_ALIGN_NUM = 16;
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t GAMMA_INDEX = 2;
constexpr uint32_t BETA_INDEX = 3;
// A5-only dispatch; matches the copied Device platform helpers.
constexpr uint64_t UB_BLOCK_BYTES = 32;
constexpr uint64_t VECTOR_REGISTER_BYTES = 256;
constexpr uint32_t FLOAT_BYTE_SIZE = 4;
constexpr uint32_t UB_USED = 1024;
constexpr uint32_t UB_RESERVE_FOR_RSTDALIGN = 1024;
constexpr uint32_t MODE_NORMAL = 1000;
constexpr uint32_t MODE_SPLIT_D = 2000;
constexpr uint32_t QUE_NUM = 5;
constexpr uint32_t QUE_MODE_NORMAL_NUM = 4;
constexpr uint64_t ALING_FACTOR_256 = 256;
constexpr uint64_t ALING_FACTOR_512 = 512;
constexpr uint32_t RETAINED_SIZE = 5120; // 256 * 5 * 4;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t MULTI_FACTOR_2 = 2;
constexpr uint32_t NUM_2 = 2;
constexpr uint32_t NDDMA_BETTER_STAGE = 512;

const std::map<ge::DataType, uint32_t> dTypeByteMap = {
    {ge::DT_FLOAT16, 2},
    {ge::DT_FLOAT, 4},
    {ge::DT_BF16, 2},
};

template <typename T>
auto CeilDiv(T x, T y) -> T
{
    return y == 0 ? x : (x + y - 1) / y;
}

void SetByDtype(ge::DataType dataType, uint32_t& dtypeKey, uint32_t& dataPerBlock)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            dtypeKey = DTYPE_KEY_FP16;
            dataPerBlock = BYTE_SIZE_2_BLOCK_ALIGN_NUM;
            break;
        case ge::DT_BF16:
            dtypeKey = DTYPE_KEY_BF16;
            dataPerBlock = BYTE_SIZE_2_BLOCK_ALIGN_NUM;
            break;
        default:
            dtypeKey = DTYPE_KEY_FP32;
            dataPerBlock = FLOAT_BLOCK_ALIGN_NUM;
            break;
    }
}

uint32_t ComputeTotalBufSize(uint32_t bufferNum, ge::DataType dtype, uint32_t dtypeSize, uint32_t length, bool split, bool hasBeta)
{
    // queBuferSize: 计算搬运需要空间大小
    uint32_t queBufSize = bufferNum * length * dtypeSize * QUE_NUM + FLOAT_PER_REAPEAT * bufferNum * FLOAT_BYTE_SIZE;
    uint32_t tmpBufSzie = 0; // tmpBufSzie: UB内需要临时空间大小
    if (split) {
        // 切分场景下
        tmpBufSzie = (dtype == ge::DT_FLOAT) ? 0 : length * FLOAT_BYTE_SIZE * NUM_2;
    } else {
        // 普通场景下：如果是float16及bfloat16数据类型，需要一块：转FP32
        tmpBufSzie = length * FLOAT_BYTE_SIZE;
    }
    uint32_t betaBufSize = hasBeta ? bufferNum * length * dtypeSize : 0;
    return queBufSize + betaBufSize + tmpBufSzie + RETAINED_SIZE;
}

ge::graphStatus TilingAddRmsNormBiasRegbase(gert::TilingContext* context)
{
    OP_LOGD(context, " TilingAddRmsNormBiasRegbase");
    auto ptrCompileInfo = reinterpret_cast<const AddRmsNormBiasCompileInfo*>(context->GetCompileInfo());
    uint32_t numCore;
    uint64_t ubSize;
    if (nullptr == ptrCompileInfo) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        numCore = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        numCore = ptrCompileInfo->totalCoreNum;
        ubSize = ptrCompileInfo->totalUbSize;
    }
    const gert::Shape xShape = context->GetInputShape(X_INDEX)->GetStorageShape();

    const gert::Shape gammaShape = context->GetInputShape(GAMMA_INDEX)->GetStorageShape();
    std::string opType(context->GetNodeType());
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* epsilon = attrs->GetFloat(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsilon);
    OP_CHECK_IF(*epsilon < 0, OP_LOGE(context, "Epsilon less than zero, please check."), return ge::GRAPH_FAILED);
    const auto betaDesc = context->GetOptionalInputDesc(BETA_INDEX);
    const bool hasBeta = betaDesc != nullptr;
    if (hasBeta) {
        const auto betaShape = context->GetOptionalInputShape(BETA_INDEX);
        OP_CHECK_IF(betaShape == nullptr ||
                        betaDesc->GetDataType() != context->GetInputDesc(GAMMA_INDEX)->GetDataType(),
                    OP_LOGE(context, "Beta must have the same shape and dtype as gamma."), return ge::GRAPH_FAILED);
        const auto& shape = betaShape->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != gammaShape.GetDimNum(),
                    OP_LOGE(context, "Beta rank must match gamma."), return ge::GRAPH_FAILED);
        for (size_t i = 0; i < gammaShape.GetDimNum(); ++i) {
            OP_CHECK_IF(shape.GetDim(i) != gammaShape.GetDim(i),
                        OP_LOGE(context, "Beta shape must match gamma."), return ge::GRAPH_FAILED);
        }
    }
    uint64_t numCol = gammaShape.GetShapeSize();
    float avgFactor = (numCol == 0U) ? 0.0f : 1.0f / static_cast<float>(numCol);
    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();
    uint64_t numRow = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= xShape.GetDim(i);
    }
    for (size_t i = 0; i < xDimNum; i++) {
        OP_LOGD(context, " TilingAddRmsNormBiasRegbase x shape:%ld", xShape.GetDim(i));
    }
    for (size_t i = 0; i < gammaDimNum; i++) {
        OP_LOGD(context, " TilingAddRmsNormBiasRegbase gama shape:%ld", gammaShape.GetDim(i));
    }
    auto dataType = context->GetInputDesc(0)->GetDataType();
    uint32_t dtypeKey = DTYPE_KEY_FP16;
    size_t usrSize = 256;
    size_t sysWorkspaceSize = 16UL * 1024UL * 1024UL;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    uint64_t numColAlign = 0;
    uint64_t ubBlockSize = UB_BLOCK_BYTES;
    uint64_t ubfp32 = ubBlockSize / sizeof(float);
    uint64_t vlfp32 = VECTOR_REGISTER_BYTES / sizeof(float);
    uint64_t binaryAddElemtMaxLen = vlfp32 * vlfp32 * NUM_2 * NUM_2;
    uint64_t blockFactor;
    uint64_t ubFactor;
    uint64_t rowFactor = 0;
    uint32_t ubLoop{0};
    uint64_t colBuferLength{0};
    uint64_t multiNNum{0};

    ubSize = ubSize - UB_USED;
    uint32_t dataPerBlock;
    SetByDtype(dataType, dtypeKey, dataPerBlock);

    blockFactor = static_cast<uint32_t>(1);
    uint64_t tileNum = CeilDiv(numRow, static_cast<uint64_t>(numCore));
    blockFactor *= tileNum;
    uint32_t useCoreNum = CeilDiv(numRow, blockFactor);
    context->SetBlockDim(useCoreNum);

    auto dtypeByteIterator = dTypeByteMap.find(dataType);
    OP_CHECK_IF(dtypeByteIterator == dTypeByteMap.end(), OP_LOGE(context, "Fail to get dtype factor."),
                return ge::GRAPH_FAILED);
    uint32_t curElementByte = dtypeByteIterator->second;
    numColAlign = CeilDiv(numCol * curElementByte, ubBlockSize) * ubBlockSize / curElementByte;

    // 计算 二分累加 分界点
    uint64_t binAddQuotient = numColAlign == 0 ? 1 : (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(numColAlign)));
    binAddQuotient = (binAddQuotient == numColAlign) ? binAddQuotient / NUM_2 : binAddQuotient;
    uint64_t binAddBufferOneline = CeilDiv((binAddQuotient + vlfp32 - 1) / vlfp32, ubfp32) * ubfp32;

    // 可以全载的行数
    int64_t tmpSize = static_cast<int64_t>(ubSize) - UB_RESERVE_FOR_RSTDALIGN - (numColAlign * curElementByte * (hasBeta ? NUM_2 : 1));
    if (tmpSize > 0 && numColAlign <= binaryAddElemtMaxLen) {
        rowFactor = tmpSize / (numColAlign * curElementByte * DOUBLE_BUFFER_NUM * QUE_MODE_NORMAL_NUM +
                               numColAlign * sizeof(float) + sizeof(float) * (DOUBLE_BUFFER_NUM + 1) +
                               binAddBufferOneline * sizeof(float));
    }
    if (rowFactor >= 1) {
        // R能够全载
        rowFactor = std::min(rowFactor, blockFactor); // 实际需要全载的行数
        AddRMSNormBiasRegbaseRFullLoadTilingData tiling;
        tiling.set_numRow(numRow);
        tiling.set_numCol(numCol);
        tiling.set_numColAlign(numColAlign);
        tiling.set_blockFactor(blockFactor);
        tiling.set_rowFactor(rowFactor);
        tiling.set_binAddQuotient(binAddQuotient);
        tiling.set_epsilon(*epsilon);
        tiling.set_avgFactor(avgFactor);
        tiling.set_nullptr_beta(hasBeta ? 0 : 1);
        OPS_LOG_I(context,
                "TilingData numCore: %u, ubSize: %lu, numRow: %lu, numCol: %lu, numColAlign: %lu, "
                "blockFactor: %lu, rowFactor: %lu, binAddQuotient: %lu, "
                "epsilon: %f, avgFactor: %f",
                numCore, ubSize, tiling.get_numRow(), tiling.get_numCol(), tiling.get_numColAlign(),
                tiling.get_blockFactor(), tiling.get_rowFactor(), tiling.get_binAddQuotient(), tiling.get_epsilon(),
                tiling.get_avgFactor());

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->SetTilingKey(MODE_NORMAL);
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    } else {
        numColAlign = CeilDiv(numCol * curElementByte, ALING_FACTOR_512) * ALING_FACTOR_512 / curElementByte;
        rowFactor = FLOAT_PER_REAPEAT;
        ubFactor = 1U;
        while (ComputeTotalBufSize(DOUBLE_BUFFER_NUM, dataType, curElementByte, ubFactor * MULTI_FACTOR_2, true, hasBeta) <
               ubSize) {
            ubFactor *= MULTI_FACTOR_2;
        }
        ubLoop = 1U;
        while (ubLoop * MULTI_FACTOR_2 * ubFactor <= numCol) {
            ubLoop *= MULTI_FACTOR_2;
        }
        colBuferLength = ubFactor;
        uint32_t isNddma = numCol >= NDDMA_BETTER_STAGE ? 0U : 1U;

        AddRMSNormBiasRegbaseTilingData tiling;
        tiling.set_numRow(numRow);
        tiling.set_numCol(numCol);
        tiling.set_numColAlign(numColAlign);
        tiling.set_blockFactor(blockFactor);
        tiling.set_rowFactor(rowFactor);
        tiling.set_ubFactor(ubFactor);
        tiling.set_epsilon(*epsilon);
        tiling.set_avgFactor(avgFactor);
        tiling.set_nullptr_beta(hasBeta ? 0 : 1);
        tiling.set_ubLoop(ubLoop);
        tiling.set_colBuferLength(colBuferLength);
        tiling.set_multiNNum(multiNNum);
        tiling.set_isNddma(isNddma);
        OPS_LOG_I(context,
                "TilingData numCore: %u, ubSize: %lu, numRow: %u, numCol: %u, numColAlign: %u, colBuferLength: %u, "
                "blockFactor: %u, rowFactor: %u, ubFactor: %u, "
                "epsilon: %f, avgFactor: %f, ubLoop: %u, multiNNum: %u, isNddma: %u.",
                numCore, ubSize, tiling.get_numRow(), tiling.get_numCol(), tiling.get_numColAlign(),
                tiling.get_colBuferLength(), tiling.get_blockFactor(), tiling.get_rowFactor(), tiling.get_ubFactor(),
                tiling.get_epsilon(), tiling.get_avgFactor(), tiling.get_ubLoop(), tiling.get_multiNNum(),
                tiling.get_isNddma());

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->SetTilingKey(MODE_SPLIT_D);
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace addRmsNormBiasRegbase
} // namespace optiling
