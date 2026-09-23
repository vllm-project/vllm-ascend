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
 *
 * kv layout is PA (PA_BBND/PA_BNBD/PA_NZ) -> block_table is REQUIRED (INT32, [b, max_blocks]).
 * Per-batch block counts must be consistent with cache_seqlens: each batch's
 * KV length must fit in its block_table row (len <= dim1 * block_size), which
 * bounds the needed blocks by max_block_num (block_table dim1). The value
 * cross-check runs only when host tensor data is present (same guard as the
 * seq_len checker / FIA tiling). v contiguity checks do not exist in MLA:
 * k_cache is a single kv buffer (k == v).
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_mla_with_kvcache_tiling_info.h"
#include "paged_attention_checker.h"

namespace optiling {
namespace flash_mla_with_kvcache {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35MLA;

ge::graphStatus PagedAttentionChecker::CheckSinglePara(const FlashMlaWithKvcacheTilingInfo &faInfo)
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

ge::graphStatus PagedAttentionChecker::CheckParaExistence(const FlashMlaWithKvcacheTilingInfo &faInfo)
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
    OP_CHECK_IF(
        blockTableTensor == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            faInfo.opName, "block_table", "provided",
            "When PagedAttention is enabled (layout_kv is PA_BBND/PA_BNBD/PA_NZ), block_table must not be empty"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckMultiPara(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    if (!faInfo.pageAttentionFlag) {
        OP_CHECK_IF(faInfo.keyNonContigDim != -1,
                    OP_LOGE(faInfo.opName, "In non-PA scenarios, k_cache must be contiguous."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    auto &blockTableTensor = faInfo.opParamInfo.blockTable.tensor;
    int64_t dim0 = blockTableTensor->GetStorageShape().GetDim(0);
    int64_t dim1 = blockTableTensor->GetStorageShape().GetDim(1);
    if (dim0 != faInfo.bSize) {
        std::string shapeStr = Ops::Base::ToString(blockTableTensor->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            faInfo.opName, "block_table", shapeStr.c_str(),
            ("The first dim of block_table must be equal to the batch size " + std::to_string(faInfo.bSize)).c_str());
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(dim1 <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    faInfo.opName, "block_table", Ops::Base::ToString(blockTableTensor->GetStorageShape()).c_str(),
                    "The second dim of block_table (max block num per batch) must be greater than 0"),
                return ge::GRAPH_FAILED);

    // PA 场景下 kvcache 非连续stride校验 (dimIndex由parser预计算); k_cache 是唯一 kv buffer
    // PA_BBND（batch-block 序）仅允许 dim0 非连续（block 表索引维）；PA_BNBD/PA_NZ（head-block 序）允许 dim0|dim1
    if (faInfo.hasViewStride && faInfo.kvLayout == FlashMlaWithKvcacheLayout::PA_BBND) {
        OP_CHECK_IF((faInfo.keyNonContigDim > 0),
                    OP_LOGE(faInfo.opName,
                            "In PA BBND scenarios, k_cache only supports non-contiguous tensors in dimension 0, "
                            "but the first non-contiguous dimension is index %d.",
                            faInfo.keyNonContigDim),
                    return ge::GRAPH_FAILED);
    } else if (faInfo.hasViewStride && (faInfo.kvLayout == FlashMlaWithKvcacheLayout::PA_BNBD ||
                                        faInfo.kvLayout == FlashMlaWithKvcacheLayout::PA_NZ)) {
        OP_CHECK_IF(
            (faInfo.keyNonContigDim > 1),
            OP_LOGE(faInfo.opName,
                    "In PA BNBD/PA_NZ scenarios, k_cache only supports non-contiguous tensors in dimensions 0 or 1, "
                    "but the first non-contiguous dimension is index %d.",
                    faInfo.keyNonContigDim),
            return ge::GRAPH_FAILED);
    }

    // Per-batch block counts consistent with cache_seqlens：值级校验在 ACLNN 直调路径下
    // 不可读（GetData 为设备指针，同 seq_len_checker），改由 kernel 分页解析器运行时兜底。
    (void)dim1;
    return ge::GRAPH_SUCCESS;
}

} // namespace flash_mla_with_kvcache
} // namespace optiling
