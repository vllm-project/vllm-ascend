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
 * \file paged_attention_checker.cpp
 * \brief Checker for PagedAttention parameters (文档约束: Paged Attention参数组)
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fa_tiling_info.h"
#include "paged_attention_checker_flash_attn.h"

namespace optiling {
namespace flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus PagedAttentionChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (!faInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(
        faInfo.blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT || faInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_16,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "block_size", std::to_string(faInfo.blockSize).c_str(),
                                              "The value of block_size must be within the range [16, 1024]"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        faInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_16 != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "block_size", std::to_string(faInfo.blockSize).c_str(),
                                              "The value of block_size must be 16-aligned"),
        return ge::GRAPH_FAILED);

    auto &blockTableTensor = faInfo.opParamInfo.blockTable.tensor;
    // 这里和存在性校验冲突了，但是为了在单参数校验校验dtype需要先判空
    OP_CHECK_IF(blockTableTensor == nullptr, OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "block_table"),
                return ge::GRAPH_FAILED);

    const gert::CompileTimeTensorDesc *blockTableDesc = faInfo.opParamInfo.blockTable.desc;
    OP_CHECK_IF(blockTableDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "TensorDesc of block_table"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(blockTableDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "block_table",
                                          Ops::Base::ToString(blockTableDesc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(blockTableDesc, BLOCK_TABLE_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t dimNum = blockTableTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        dimNum != 2,
        OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "block_table", (std::to_string(dimNum) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    if (!faInfo.pageAttentionFlag) {
        // 非 PA 模式下，block_table 不应传入
        auto &blockTableTensor = faInfo.opParamInfo.blockTable.tensor;
        OP_CHECK_IF(blockTableTensor != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        faInfo.opName, "block_table", "provided",
                        "When layout_kv is not PA (PA_BBND/PA_BNBD/PA_NZ), block_table must not be provided"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    auto &blockTableTensor = faInfo.opParamInfo.blockTable.tensor;
    OP_CHECK_IF(blockTableTensor == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "block_table", "provided",
                                                      "When PagedAttention is enabled, block_table must not be empty"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckFeature(const FaTilingInfo &faInfo)
{
    if (!faInfo.pageAttentionFlag) {
        return ge::GRAPH_SUCCESS;
    }

    // 特性交叉校验: PA_NZ 的 D 轴分形粒度为 16 元素, head_dim 非 16 倍数时
    // kernel 内 d1 = headDim/16 截断会静默算错
    OP_CHECK_IF((faInfo.kvLayout == FaLayout::PA_NZ) && (faInfo.qkHeadDim % 16 != 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    faInfo.opName, "axis D of query and key", std::to_string(faInfo.qkHeadDim).c_str(),
                    "When layout_kv is PA_NZ, axis D must be 16-aligned (D axis fractal of NZ is 16 elements)"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    if (!faInfo.pageAttentionFlag) {
        OP_CHECK_IF(faInfo.keyNonContigDim != -1,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, "key",
                                                             "In non-PA scenarios, key tensors must be contiguous"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(faInfo.valueNonContigDim != -1,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, "value",
                                                             "In non-PA scenarios, value tensors must be contiguous"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    auto &blockTableTensor = faInfo.opParamInfo.blockTable.tensor;
    int64_t dim0 = blockTableTensor->GetStorageShape().GetDim(0);
    if (dim0 != faInfo.bSize) {
        std::string shapeStr = Ops::Base::ToString(blockTableTensor->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            faInfo.opName, "block_table", shapeStr.c_str(),
            ("The first dim of block_table must be equal to the batch size " + std::to_string(faInfo.bSize)).c_str());
        return ge::GRAPH_FAILED;
    }

    if (!faInfo.hasViewStride) {
        return ge::GRAPH_SUCCESS;
    }

    // PA场景下kvcache非连续stride校验 (dimIndex由parser预计算)
    if (faInfo.kvLayout == FaLayout::PA_BBND) {
        OP_CHECK_IF((faInfo.keyNonContigDim > 0),
                    OP_LOGE(faInfo.opName,
                            "In PA BBND scenarios, key only supports non-contiguous tensors in dimension 0, "
                            "but the first non-contiguous dimension is index %d.",
                            faInfo.keyNonContigDim),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((faInfo.valueNonContigDim > 0),
                    OP_LOGE(faInfo.opName,
                            "In PA BBND scenarios, value only supports non-contiguous tensors in dimension 0, "
                            "but the first non-contiguous dimension is index %d.",
                            faInfo.valueNonContigDim),
                    return ge::GRAPH_FAILED);
    } else if (faInfo.kvLayout == FaLayout::PA_BNBD || faInfo.kvLayout == FaLayout::PA_NZ) {
        OP_CHECK_IF((faInfo.keyNonContigDim > 1),
                    OP_LOGE(faInfo.opName,
                            "In PA BNBD/NZ scenarios, key only supports non-contiguous tensors in dimensions 0 or 1, "
                            "but the first non-contiguous dimension is index %d.",
                            faInfo.keyNonContigDim),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((faInfo.valueNonContigDim > 1),
                    OP_LOGE(faInfo.opName,
                            "In PA BNBD/NZ scenarios, value only supports non-contiguous tensors in dimensions 0 or 1, "
                            "but the first non-contiguous dimension is index %d.",
                            faInfo.valueNonContigDim),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace flash_attn
} // namespace optiling
